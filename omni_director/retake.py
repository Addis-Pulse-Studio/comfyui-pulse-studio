"""The Retake Scissor: replace a bad span of a rendered clip without re-rendering it.

Why this is the flagship
------------------------
H3 accepts exactly two positional anchors, frame 0 and frame_count-1, and nothing
between them. For a timeline that is a hard limitation. For patching it is
precisely the right shape: the frame before the cut and the frame after the cut
are exactly two anchors, and pinning both makes the patch land seamlessly on the
surrounding footage. This is the one job where H3's limitation is a feature.

The geometry
------------
Base clip frames are indexed 0..N-1. The user marks a bad span [a, b) -- frames a
through b-1 are to be replaced. The patch render is an fl2va call whose:

    first_frame = base[a-1]   (the last good frame BEFORE the cut)
    last_frame  = base[b]     (the first good frame AFTER the cut)

so the rendered patch *reproduces those two anchor frames at its own ends*. Its
interior is the new material. Stitching therefore drops the duplicated anchors:

    output = base[0:a] + patch[1:L-1] + base[b:N]

which means the patch's interior length must equal the gap: L - 2 == b - a. L
also has to sit on the 17k+5 grid, so L is fixed first and the cut is snapped to
match. Snapping the cut is deliberate -- the blueprint's rule is that the UI
moves the user's handles onto legal positions rather than rejecting the edit
after they have made it.

Two degenerate cases drop one anchor each: a cut at the very start of the clip
has no frame before it, and a cut running to the very end has no frame after it.
Both are handled, and both change the arithmetic, so neither is special-cased
away -- see _solve_geometry.

Everything here is headless: it computes index ranges and a render spec. The node
layer does the actual slicing and concatenation.
"""

from .constants import MAX_WINDOW_FRAMES, MIN_FRAMES
from .frames import align_frame_count, frames_to_seconds, is_on_grid, seconds_to_frames

__all__ = ["RetakePlan", "plan_retake", "RetakeError"]


class RetakeError(ValueError):
    """A cut that cannot be patched as asked."""


class RetakePlan:
    """A fully specified patch operation.

    Index ranges are half-open and refer to the base clip. `patch_take` is the
    slice of the rendered patch that survives stitching -- it excludes whichever
    anchor frames were pinned, because those already exist in the base.
    """

    __slots__ = ("base_frames", "cut_start", "cut_end", "requested_cut_start",
                 "requested_cut_end", "patch_frames", "patch_take",
                 "anchor_first_base_index", "anchor_last_base_index",
                 "keep_base_audio", "fps", "diagnostics")

    def __init__(self, base_frames, cut_start, cut_end, requested_cut_start,
                 requested_cut_end, patch_frames, patch_take,
                 anchor_first_base_index, anchor_last_base_index,
                 keep_base_audio, fps, diagnostics):
        self.base_frames = base_frames
        self.cut_start = cut_start
        self.cut_end = cut_end
        self.requested_cut_start = requested_cut_start
        self.requested_cut_end = requested_cut_end
        self.patch_frames = patch_frames
        self.patch_take = patch_take  # (start, stop) into the rendered patch
        self.anchor_first_base_index = anchor_first_base_index
        self.anchor_last_base_index = anchor_last_base_index
        self.keep_base_audio = keep_base_audio
        self.fps = fps
        self.diagnostics = diagnostics

    # ── the stitch ──────────────────────────────────────────────────────────

    @property
    def head_range(self):
        """Base frames kept before the patch."""
        return (0, self.cut_start)

    @property
    def tail_range(self):
        """Base frames kept after the patch."""
        return (self.cut_end, self.base_frames)

    @property
    def gap_frames(self):
        """How many base frames the patch replaces."""
        return self.cut_end - self.cut_start

    @property
    def output_frames(self):
        """Length of the stitched result. Always equals the base length -- a
        patch replaces material, it does not lengthen the clip."""
        take = self.patch_take[1] - self.patch_take[0]
        return self.cut_start + take + (self.base_frames - self.cut_end)

    @property
    def snapped(self):
        return (self.cut_start != self.requested_cut_start
                or self.cut_end != self.requested_cut_end)

    def anchors(self):
        """The fl2va anchor spec: which base frames to pin, and at which index of
        the patch render. Only 0 and patch_frames-1 are legal, per PackedLayout."""
        out = {}
        if self.anchor_first_base_index is not None:
            out["first_frame"] = {"base_index": self.anchor_first_base_index, "patch_index": 0}
        if self.anchor_last_base_index is not None:
            out["last_frame"] = {"base_index": self.anchor_last_base_index,
                                 "patch_index": self.patch_frames - 1}
        return out

    def describe(self):
        a, b = self.cut_start, self.cut_end
        return ("patch %d frame(s) [%d,%d) = %.3fs-%.3fs | render %d frames | "
                "keep %d..%d of the patch | anchors: %s | audio: %s"
                % (self.gap_frames, a, b, frames_to_seconds(a, self.fps),
                   frames_to_seconds(b, self.fps), self.patch_frames,
                   self.patch_take[0], self.patch_take[1] - 1,
                   ", ".join(sorted(self.anchors())) or "none",
                   "base kept" if self.keep_base_audio else "patched"))

    def to_dict(self):
        return {"base_frames": self.base_frames, "cut_start": self.cut_start,
                "cut_end": self.cut_end, "requested_cut_start": self.requested_cut_start,
                "requested_cut_end": self.requested_cut_end,
                "patch_frames": self.patch_frames, "patch_take": list(self.patch_take),
                "head_range": list(self.head_range), "tail_range": list(self.tail_range),
                "anchors": self.anchors(), "keep_base_audio": self.keep_base_audio,
                "output_frames": self.output_frames, "snapped": self.snapped,
                "diagnostics": list(self.diagnostics)}

    def __repr__(self):
        return "RetakePlan(%s)" % self.describe()


def _solve_geometry(base_frames, a, b):
    """Choose the patch length and the final cut for a requested span [a, b).

    Returns (a, b, patch_frames, patch_take, anchor_first, anchor_last).

    The relationship between patch length L and the gap depends on how many
    anchors exist, because each pinned anchor costs one patch frame that is
    already present in the base and must be dropped at stitch time:

        both anchors   (0 < a, b < N):  interior = L - 2  must equal b - a
        no head anchor (a == 0):        we keep patch[0:L-1], so L - 1 == b
        no tail anchor (b == N):        we keep patch[1:L],  so L - 1 == N - a
        neither        (full re-render): L == N
    """
    has_first = a > 0
    has_last = b < base_frames

    if has_first and has_last:
        overhead = 2
    elif has_first or has_last:
        overhead = 1
    else:
        overhead = 0

    gap = b - a
    patch_frames = align_frame_count(gap + overhead)
    # The grid moved the length up; give the extra frames back to the cut so the
    # arithmetic stays exact. Prefer growing the cut forward (into the tail),
    # since the head is usually the part the user has already approved.
    grown = (patch_frames - overhead) - gap
    if grown:
        room_after = base_frames - b
        take_after = min(grown, room_after)
        b += take_after
        remaining = grown - take_after
        if remaining:
            take_before = min(remaining, a)
            a -= take_before
            remaining -= take_before
        if remaining:
            # The clip is shorter than the smallest legal patch that covers it;
            # the only honest answer is to re-render the whole thing.
            a, b = 0, base_frames
            patch_frames = align_frame_count(base_frames)
            return a, b, patch_frames, (0, base_frames), None, None
        # Growing the cut can consume an anchor, changing the overhead. Re-solve
        # once against the new span rather than emitting an off-by-one stitch.
        if (a > 0) != has_first or (b < base_frames) != has_last:
            return _solve_geometry(base_frames, a, b)

    anchor_first = a - 1 if has_first else None
    anchor_last = b if has_last else None
    take_start = 1 if has_first else 0
    take_stop = patch_frames - 1 if has_last else patch_frames
    return a, b, patch_frames, (take_start, take_stop), anchor_first, anchor_last


def plan_retake(base_frames, cut_start=None, cut_end=None, cut_start_seconds=None,
                cut_end_seconds=None, keep_base_audio=True, fps=24,
                max_patch_frames=MAX_WINDOW_FRAMES):
    """Plan a patch of [cut_start, cut_end) within a clip of `base_frames` frames.

    Cut points may be given in frames or in seconds; seconds are converted at
    `fps` and then snapped. `keep_base_audio` leaves the base clip's audio
    untouched, which is almost always what you want: a re-rendered patch invents
    its own score and will not match the surrounding track.

    Returns a RetakePlan. Raises RetakeError only for a cut that cannot exist at
    all (empty span, out of bounds, patch longer than one H3 render).
    """
    diagnostics = []
    base_frames = int(base_frames)
    if base_frames < MIN_FRAMES:
        raise RetakeError("base clip is %d frames; nothing shorter than %d can be patched"
                          % (base_frames, MIN_FRAMES))

    if cut_start is None:
        if cut_start_seconds is None:
            raise RetakeError("a cut needs either cut_start or cut_start_seconds")
        cut_start = int(round(float(cut_start_seconds) * fps))
    if cut_end is None:
        if cut_end_seconds is None:
            raise RetakeError("a cut needs either cut_end or cut_end_seconds")
        cut_end = int(round(float(cut_end_seconds) * fps))

    requested_a, requested_b = int(cut_start), int(cut_end)
    a = max(0, min(requested_a, base_frames))
    b = max(0, min(requested_b, base_frames))
    if b < a:
        a, b = b, a
        diagnostics.append("cut points were reversed; treated as [%d, %d)" % (a, b))
    if a != requested_a or b != requested_b:
        diagnostics.append("cut clamped to the clip: [%d, %d) -> [%d, %d)"
                           % (requested_a, requested_b, a, b))
    if b == a:
        raise RetakeError("the cut is empty (frame %d to %d); mark a span to replace" % (a, b))

    clamped_a, clamped_b = a, b
    a, b, patch_frames, patch_take, anchor_first, anchor_last = _solve_geometry(base_frames, a, b)

    if patch_frames > max_patch_frames:
        raise RetakeError(
            "patching [%d, %d) needs a %d-frame render (%.2fs), beyond H3's %d-frame "
            "ceiling. Mark a shorter span, or re-render the clip."
            % (a, b, patch_frames, frames_to_seconds(patch_frames, fps), max_patch_frames))

    if not is_on_grid(patch_frames):  # pragma: no cover - guards the solver itself
        raise RetakeError("internal: patch length %d is off-grid" % (patch_frames,))

    if (a, b) != (clamped_a, clamped_b):
        diagnostics.append(
            "cut snapped onto the frame grid: [%d, %d) -> [%d, %d) so the patch is a legal "
            "%d-frame render" % (clamped_a, clamped_b, a, b, patch_frames))

    if anchor_first is None and anchor_last is None:
        diagnostics.append(
            "the cut spans the whole clip, so there is no surrounding frame to anchor to; "
            "this is a full re-render, not a patch")
    elif anchor_first is None:
        diagnostics.append(
            "the cut starts at frame 0, so there is no frame before it to anchor; only the "
            "tail seam is locked")
    elif anchor_last is None:
        diagnostics.append(
            "the cut runs to the end of the clip, so there is no frame after it to anchor; "
            "only the head seam is locked")

    if not keep_base_audio:
        diagnostics.append(
            "keep_base_audio is off: the patch renders its own audio, which will not match "
            "the surrounding score and is usually audible at both seams")

    plan = RetakePlan(base_frames, a, b, requested_a, requested_b, patch_frames,
                      patch_take, anchor_first, anchor_last, bool(keep_base_audio),
                      int(fps), diagnostics)

    # The stitch must be length-preserving; a mismatch here would silently shift
    # every frame after the patch, so it is checked rather than trusted.
    if plan.output_frames != base_frames:  # pragma: no cover - invariant guard
        raise RetakeError("internal: stitch would change clip length %d -> %d"
                          % (base_frames, plan.output_frames))
    return plan
