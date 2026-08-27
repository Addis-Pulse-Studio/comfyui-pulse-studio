"""The compiler: timeline JSON in, storyboard prompt + ordered file list out.

Headless by construction -- stdlib only. This module decides *what* each render
call will say and which files it will carry; it never touches a tensor, loads a
checkpoint, or talks to ComfyUI. That separation is what makes the correctness
properties here testable at all.

The output of `compile_timeline` is a CompiledPlan: one CompiledWindow per real
H3 call, each already quantised to the frame grid, each carrying its own resolved
prompt and its own socket->file mapping. The node layer's job is then purely
mechanical -- walk the windows, sample, decode, carry forward, stitch.

Three properties this module exists to guarantee:

  1. No ordinal is ever stored. <Picture i>/<Video k>/<Audio j> are computed from
     live bin order at compile time, per window, so add/remove/reorder cannot
     desynchronise prompt text from sockets.
  2. Every frame count handed downstream is on the 17k+5 grid.
  3. [Shot N] timestamps within a window are strictly increasing.

ATTRIBUTION
-----------
`CarryPolicy`'s audio carry-over -- feeding the previous window's decoded audio
tail back through the reference audio sockets so each window does not invent its
own score -- is an empirical finding of
muse-collective-26/MiniMaxH3-Director-Seed-Hunt (MIT), which this project forks.
It is behaviour that was observed on real renders rather than derived, and it is
reproduced here on that basis. See NOTICE.
"""

from .assets import (
    KIND_AUDIO,
    KIND_IMAGE,
    KIND_VIDEO,
    LITERAL_TAG_RE,
    REF_TOKEN_RE,
    Asset,
    AssetBin,
)
from .constants import (
    AUDIO_ROLE_LIP_SYNC,
    AUDIO_ROLE_RETENTION,
    BRANCH_FL2VA,
    BRANCH_REF2VA,
    DEFAULT_AUDIO_CARRY_SECONDS,
    MAX_WINDOW_FRAMES,
    MIN_TRAINED_FRAMES,
    RETENTION_DEFAULT,
    RETENTION_FULL,
    RETENTION_PARTIAL,
    SPEAKER_ID_FORMAT,
)
from .frames import (
    align_frame_count,
    frames_to_seconds,
    last_anchor_index,
    partition_windows,
    seconds_to_frames,
    window_bounds,
)
from .timeline import Timeline

__all__ = [
    "CompiledWindow",
    "CompiledPlan",
    "CarryPolicy",
    "compile_timeline",
    "format_timestamp",
    "resolve_references",
    "wrap_dialogue",
]

CARRY_IMAGE_ID = "__carry_frame__"
CARRY_VIDEO_ID = "__carry_clip__"
CARRY_AUDIO_ID = "__carry_audio__"


# ── small text helpers ──────────────────────────────────────────────────────

def format_timestamp(seconds):
    """MM:SS.mmm, H3's own shot-marker format.

    Minutes are not capped at 60 -- a 90-minute project would read '90:00.000'
    rather than wrapping to an hour field the model has never been shown.
    """
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return "%02d:%06.3f" % (minutes, seconds - minutes * 60)


def wrap_dialogue(text, language):
    """Wrap every "..." span as <d>[Language] ...</d>, H3's dialogue tag.

    Only quoted spans are touched; any tag the compiler already resolved outside
    the quotes is left exactly as it was.
    """
    out = []
    i = 0
    while True:
        a = text.find('"', i)
        if a == -1:
            out.append(text[i:])
            break
        b = text.find('"', a + 1)
        if b == -1:
            out.append(text[i:])
            break
        out.append(text[i:a])
        inner = _normalise_dialogue(text[a + 1:b])
        out.append("<d>[%s] %s</d>" % (language, inner))
        i = b + 1
    return "".join(out)


def _normalise_dialogue(text):
    """Collapse decorative and repeated punctuation, then terminate the line.

    A quote captured mid-sentence usually ends on a comma by ordinary grammar
    (he says, "this must be it," and turns away) -- that comma is the narration's,
    not the line's, so it is dropped before the terminator is added.
    """
    text = text.strip()
    for ch in "*_#~":
        text = text.replace(ch, "")
    for mark in ".!?,":
        while mark * 2 in text:
            text = text.replace(mark * 2, mark)
    text = text.rstrip(",").strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


# ── reference resolution ────────────────────────────────────────────────────

def resolve_references(text, tag_map, bin_, subject_tags=None, scope_ids=None):
    """Replace {{asset_id}} / @Name / @[Name With Spaces] with live tags.

    An image that has been promoted to a <Subject N> definition resolves to that
    Subject tag rather than to its raw <Picture i>: MiniMax's own reference-mode
    format wants prose to talk about subjects, with the picture cited once inside
    the definition. Assets with no subject (videos, audio, undescribed images)
    resolve to their raw tag.

    `scope_ids`, when given, is the set of asset ids this particular text may see
    (§10). It is how a scene-local reference on one `PulseShot` stays invisible to
    every other shot even though both shots' references occupy sockets on the same
    H3 call. Out-of-scope hits are reported and left as literal text rather than
    resolved to a number that would point at another scene's picture -- which is
    the one failure this package exists to make impossible.

    Returns (resolved_text, diagnostics).
    """
    subject_tags = subject_tags or {}
    diagnostics = []

    def _sub(match):
        by_id, bracketed, bare = match.group(1), match.group(2), match.group(3)
        key = by_id or bracketed or bare
        # The scope is applied inside the lookup, not after it: a name that is
        # ambiguous across the whole window may be perfectly unambiguous within
        # the shot that wrote it, which is the entire point of scene-local
        # references (§10).
        asset = (bin_.get(key) if by_id
                 else (bin_.get(key) or bin_.find_by_name(key, allowed=scope_ids)))
        if asset is None and not by_id and scope_ids is not None:
            # Not visible here -- but is it visible anywhere? "It belongs to
            # another shot" is a different problem from "it does not exist", and
            # sends the author somewhere different.
            elsewhere = bin_.find_by_name(key)
            if elsewhere is not None:
                diagnostics.append(
                    "reference %r belongs to another shot's own reference sockets and "
                    "is not visible here. Scene-local references are scoped to the "
                    "PulseShot that carries them; move it to the Asset Bin to share it "
                    "across shots." % (key,))
                return match.group(0)
        if asset is None:
            diagnostics.append(
                "unresolved reference %r -- no asset with that id or unique name" % (key,))
            return match.group(0)
        if scope_ids is not None and asset.asset_id not in scope_ids:
            diagnostics.append(
                "reference %r belongs to another shot's own reference sockets and is not "
                "visible here. Scene-local references are scoped to the PulseShot that "
                "carries them; move it to the Asset Bin to share it across shots." % (key,))
            return match.group(0)
        if asset.asset_id in subject_tags:
            return subject_tags[asset.asset_id]
        tag = tag_map.by_id.get(asset.asset_id)
        if tag is None:
            diagnostics.append(
                "asset %r (%s) is in the bin but carries no tag in this window -- it was "
                "dropped to make room for carry-over references" % (asset.name, asset.asset_id))
            return match.group(0)
        return tag

    resolved = REF_TOKEN_RE.sub(_sub, text)

    # A hand-typed ordinal is the exact hazard this design removes. It is not
    # rewritten -- silently "fixing" it would be its own guess -- but it is
    # always reported, because it will not track the bin.
    for m in LITERAL_TAG_RE.finditer(text):
        diagnostics.append(
            "prompt contains a hand-typed tag %r. Ordinals shift whenever the bin "
            "changes; reference the asset by name (@Name) or id ({{id}}) instead."
            % (m.group(0),))

    return resolved, diagnostics


# ── plan objects ────────────────────────────────────────────────────────────

class CarryPolicy:
    """How continuity is held across a window seam.

    `mode` selects what the previous window's tail contributes on a ref2va
    continuation window:
      'image' -- its last decoded frame enters as a reference image at the front
                 of the image sockets, so it becomes <Picture 1>. Cheapest, and
                 the blueprint's default.
      'video' -- its last decoded frames enter as reference video slot 0. Costs a
                 video socket but carries motion, not just a pose.
      'both'  -- both of the above.
      'none'  -- no visual carry-over; text alone holds continuity.

    `audio` feeds the previous window's decoded audio tail through the reference
    audio sockets. Without it each window invents its own score from scratch and
    the seam is audible -- muse-collective-26/MiniMaxH3-Director-Seed-Hunt (MIT)
    observed this directly on a real render, and this behaviour is derived from
    theirs. The tail length is a tunable, not a discovery: 4s is what upstream
    shipped, and whether it is optimal is untested. See NOTICE.
    """

    __slots__ = ("mode", "audio", "audio_seconds")

    def __init__(self, mode="image", audio=True, audio_seconds=DEFAULT_AUDIO_CARRY_SECONDS):
        if mode not in ("image", "video", "both", "none"):
            raise ValueError("carry mode must be image|video|both|none, got %r" % (mode,))
        self.mode = mode
        self.audio = bool(audio)
        self.audio_seconds = float(audio_seconds)

    @property
    def wants_image(self):
        return self.mode in ("image", "both")

    @property
    def wants_video(self):
        return self.mode in ("video", "both")


class FileRef:
    """One entry in a window's ordered file list.

    `socket` is the literal kwarg name the stock node expects, so the node layer
    can hand these straight through without re-deriving slot numbers.
    """

    __slots__ = ("socket", "kind", "asset_id", "name", "file", "tag",
                 "trim_start", "trim_end", "synthetic", "audio_role")

    def __init__(self, socket, kind, asset_id, name, file, tag,
                 trim_start=0.0, trim_end=None, synthetic=False, audio_role=None):
        self.socket = socket
        self.kind = kind
        self.asset_id = asset_id
        self.name = name
        self.file = file
        self.tag = tag
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.synthetic = synthetic
        # Carried from the asset so the executor can trim a lip-sync clip to the
        # window without re-reading the bin. See constants.AUDIO_ROLE_LIP_SYNC.
        self.audio_role = audio_role

    def to_dict(self):
        return {"socket": self.socket, "kind": self.kind, "asset_id": self.asset_id,
                "name": self.name, "file": self.file, "tag": self.tag,
                "trim_start": self.trim_start, "trim_end": self.trim_end,
                "synthetic": self.synthetic, "audio_role": self.audio_role}

    def __repr__(self):
        return "FileRef(%s -> %s %r)" % (self.socket, self.tag, self.name)


class CompiledWindow:
    """One real H3 call, fully specified."""

    __slots__ = ("index", "total", "branch", "frame_count", "start_seconds",
                 "end_seconds", "prompt", "files", "tag_map", "anchors",
                 "diagnostics", "shot_ids", "seed_offset",
                 "resolved_shots", "unresolved_shots", "speaker_bindings")

    def __init__(self, index, total, branch, frame_count, start_seconds, end_seconds,
                 prompt, files, tag_map, anchors, diagnostics, shot_ids, seed_offset=0,
                 resolved_shots=None, unresolved_shots=None, speaker_bindings=None):
        # shot_id -> the shot's own text after reference resolution, and the
        # aliases in it that did not resolve. Both are per shot rather than per
        # window because §7.1 hashes the resolved prompt of each shot into the
        # cache key, and §9.3 reports an unresolved alias against the shot it was
        # written in -- a window-level string can answer neither.
        self.resolved_shots = dict(resolved_shots or {})
        self.unresolved_shots = dict(unresolved_shots or {})
        # shot_id -> "<Subject 2> (S1)", the character this shot's dialogue and
        # reference audio are bound to. Carried out of the compiler because the
        # binding sentence lives in the window's subject definitions, which no
        # shot's `resolved_prompt` covers -- §7.1 has to hash it from somewhere.
        self.speaker_bindings = dict(speaker_bindings or {})
        self.index = index
        self.total = total
        self.branch = branch
        self.frame_count = frame_count
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.prompt = prompt
        self.files = files
        self.tag_map = tag_map
        self.anchors = anchors
        self.diagnostics = diagnostics
        self.shot_ids = shot_ids
        self.seed_offset = seed_offset

    @property
    def duration_seconds(self):
        return self.end_seconds - self.start_seconds

    @property
    def last_anchor_index(self):
        return last_anchor_index(self.frame_count)

    def socket_kwargs(self):
        """The reference sockets for this window, grouped as the stock node wants:
        {'ref_images': {...}, 'ref_videos': {...}, ...}. Values are FileRefs; the
        node layer swaps each for a loaded tensor."""
        groups = {"ref_images": {}, "ref_videos": {}, "ref_video_audios": {}, "ref_audios": {}}
        for f in self.files:
            if f.socket.startswith("ref_image_"):
                groups["ref_images"][f.socket] = f
            elif f.socket.startswith("ref_video_audio_"):
                groups["ref_video_audios"][f.socket] = f
            elif f.socket.startswith("ref_video_"):
                groups["ref_videos"][f.socket] = f
            elif f.socket.startswith("ref_audio_"):
                groups["ref_audios"][f.socket] = f
        return groups

    def to_dict(self):
        return {"index": self.index, "total": self.total, "branch": self.branch,
                "frame_count": self.frame_count, "start_seconds": self.start_seconds,
                "end_seconds": self.end_seconds, "prompt": self.prompt,
                "files": [f.to_dict() for f in self.files], "anchors": dict(self.anchors),
                "diagnostics": list(self.diagnostics), "shot_ids": list(self.shot_ids),
                "seed_offset": self.seed_offset}

    def __repr__(self):
        return "CompiledWindow(%d/%d, %s, %d frames)" % (
            self.index + 1, self.total, self.branch, self.frame_count)


class CompiledPlan:
    """Every window for one render, plus whole-plan diagnostics."""

    __slots__ = ("windows", "diagnostics", "problems")

    def __init__(self, windows, diagnostics, problems):
        self.windows = windows
        self.diagnostics = diagnostics
        self.problems = problems  # fatal: from Timeline.validate()

    @property
    def ok(self):
        return not self.problems

    @property
    def total_frames(self):
        return sum(w.frame_count for w in self.windows)

    @property
    def total_seconds(self):
        return frames_to_seconds(self.total_frames)

    def preview(self):
        """The live compiled-prompt preview -- read the exact string before
        spending a render."""
        parts = []
        for w in self.windows:
            parts.append("=== Window %d/%d | %s | %d frames (%.2fs) ===\n%s"
                         % (w.index + 1, w.total, w.branch, w.frame_count,
                            w.duration_seconds, w.prompt))
        return "\n\n".join(parts)

    def to_dict(self):
        return {"windows": [w.to_dict() for w in self.windows],
                "diagnostics": list(self.diagnostics),
                "problems": list(self.problems),
                "total_frames": self.total_frames,
                "ok": self.ok}

    def __repr__(self):
        return "CompiledPlan(%d window(s), %d frames, ok=%s)" % (
            len(self.windows), self.total_frames, self.ok)


# ── the compiler ────────────────────────────────────────────────────────────

def compile_timeline(timeline, window_frames=None, policy="balanced",
                     carry=None, strict=False):
    """Compile a Timeline into a CompiledPlan.

    `window_frames` overrides the per-window ceiling (defaults to the timeline's
    own window_seconds, else H3's trained ceiling). `policy` is passed to
    frames.partition_windows. `carry` is a CarryPolicy.

    Structural problems are collected rather than raised, so the UI can show all
    of them at once; pass strict=True to raise on the first fatal one instead.
    """
    if isinstance(timeline, dict):
        timeline = Timeline.from_dict(timeline)
    elif isinstance(timeline, str):
        timeline = Timeline.from_json(timeline)

    carry = carry or CarryPolicy()
    problems = timeline.validate()
    if strict and problems:
        raise ValueError("; ".join(problems))

    if window_frames is None:
        # "nearest", not "down". A window length is a ceiling the user is asking
        # for, and the grid steps by 0.708s: rounding 15.0s down produced a
        # 14.375s cap, which split a 15.0s timeline into two windows over a
        # shortfall the user could not even see. partition_windows clamps to
        # MAX_WINDOW_FRAMES, so rounding up can never leave the trained range.
        window_frames = (seconds_to_frames(timeline.window_seconds, timeline.fps, "nearest")
                         if timeline.window_seconds else MAX_WINDOW_FRAMES)

    diagnostics = []
    total_frames = seconds_to_frames(timeline.duration_seconds, timeline.fps, "up")
    windows = partition_windows(total_frames, window_frames, policy=policy,
                                diagnostics=diagnostics,
                                boundaries=_shot_boundaries(timeline))
    bounds = window_bounds(windows, timeline.fps)

    requested = timeline.duration_seconds
    actual = frames_to_seconds(sum(windows), timeline.fps)
    if abs(actual - requested) > 1e-6:
        diagnostics.append(
            "duration snapped to the frame grid: %.3fs requested -> %.3fs rendered "
            "(%d frames across %d window(s))" % (requested, actual, sum(windows), len(windows)))

    # partition_windows guarantees the floor and reports its own overrides, so the
    # only sub-floor window that can reach here is a whole render shorter than the
    # floor -- which it has already diagnosed. This is a safety net, not a policy.
    for i, f in enumerate(windows):
        if f < MIN_TRAINED_FRAMES and len(windows) > 1:  # pragma: no cover - invariant guard
            raise AssertionError(
                "partitioner emitted a %d-frame window (%d of %d), below the %d-frame floor"
                % (f, i + 1, len(windows), MIN_TRAINED_FRAMES))

    # Assigned across the whole film before any window is compiled -- see
    # assign_speaker_ids for why per-window numbering is the wrong answer.
    speaker_ids = assign_speaker_ids(timeline.ordered_shots(),
                                     timeline.assets.by_kind(KIND_AUDIO))

    compiled = []
    for i, (frame_count, (start, end)) in enumerate(zip(windows, bounds)):
        branch = timeline.branch if i == 0 else timeline.continuation_branch
        compiled.append(_compile_window(
            timeline, i, len(windows), branch, frame_count, start, end, carry,
            speaker_ids=speaker_ids))

    return CompiledPlan(compiled, diagnostics, problems)


def _window_bin(timeline, index, branch, carry, shots=()):
    """The effective asset bin for one window, carry-over included.

    Carry-over references claim the *front* of their socket group, so on a
    continuation window the carried frame is <Picture 1> and every user image
    shifts up by one. That shift is the whole reason ordinals are computed here
    rather than typed by the author.

    Scene-local references (§10) are appended after the global block, per shot,
    in the window's own shot order. Their *ordinals* are necessarily assigned in
    that flat socket order: H3 takes one `ref_items` list per call and the
    tokenizer numbers by position in it, so two shots sharing a window cannot both
    start their locals immediately after the global block. What §10's "within that
    shot only" buys is enforced by `scope_ids` in `resolve_references` instead --
    a shot's prose can never resolve an alias to another shot's local reference,
    which is the property that actually matters. A shot rendered alone in its own
    window gets the literal numbering §10 describes.

    Returns (bin, diagnostics). User assets that no longer fit after carry-over
    has taken its slots are dropped, loudly.
    """
    diagnostics = []
    if branch == BRANCH_FL2VA:
        # fl2va carries no references at all -- different checkpoint, disjoint inputs.
        return AssetBin(), diagnostics

    local = []
    for shot in shots:
        local.extend(getattr(timeline, "local_refs", {}).get(shot.shot_id) or [])

    user_images = timeline.assets.by_kind(KIND_IMAGE) + [a for a in local if a.kind == KIND_IMAGE]
    user_videos = timeline.assets.by_kind(KIND_VIDEO) + [a for a in local if a.kind == KIND_VIDEO]
    user_audios = timeline.assets.by_kind(KIND_AUDIO) + [a for a in local if a.kind == KIND_AUDIO]

    images, videos, audios = [], [], []
    if index > 0:
        if carry.wants_image:
            images.append(Asset(CARRY_IMAGE_ID, KIND_IMAGE, name="Previous window, last frame",
                                description="", retention=RETENTION_FULL, synthetic=True))
        if carry.wants_video:
            videos.append(Asset(CARRY_VIDEO_ID, KIND_VIDEO, name="Previous window, tail clip",
                                retention=RETENTION_FULL, synthetic=True))
        if carry.audio:
            audios.append(Asset(CARRY_AUDIO_ID, KIND_AUDIO, name="Previous window, audio tail",
                                retention=RETENTION_PARTIAL, synthetic=True))

    # Which recordings belong to a named character. A voice that is bound is the
    # one thing in the audio group that carries information nothing else can
    # replace: drop it and that character keeps their picture, keeps their lines,
    # and loses their voice from this window onward -- which reads as drift, not
    # as a missing reference. See _audio_keep_rank.
    bound_audio = {a.asset_id for a in user_audios if a.voice_of}
    for shot in shots:
        if not shot.speakers:
            continue
        for a in (getattr(timeline, "local_refs", {}).get(shot.shot_id) or []):
            if a.kind == KIND_AUDIO:
                bound_audio.add(a.asset_id)

    def _take(dst, src, limit, label, rank=None):
        room = max(0, limit - len(dst))
        dropped = set()
        if len(src) > room:
            # Choose what to drop by rank, then emit the survivors in their
            # original bin order -- ordinals are bin order, and re-sorting the
            # survivors would renumber a window that is not over budget at all.
            order = sorted(range(len(src)),
                           key=lambda i: ((rank(src[i]) if rank else 0), i))
            dropped = {src[i].asset_id for i in order[room:]}
        for a in src:
            if a.asset_id in dropped:
                diagnostics.append(
                    "%s %r dropped from window %d: only %d %s socket(s) exist and "
                    "carry-over holds the first%s"
                    % (label, a.name, index + 1, limit, label,
                       "" if rank is None else
                       ". Bound voices are kept ahead of unbound ones"))
                continue
            dst.append(a)

    def _audio_keep_rank(asset):
        """Lower survives. Bound before unbound, lip_sync before voice_timbre.

        A lip_sync reference is a temporal alignment -- losing it desynchronises a
        mouth, which is visible. A timbre reference only shifts how a voice
        sounds. An unbound clip belongs to nobody, so nothing about the film
        changes identity when it goes.
        """
        return ((0 if asset.asset_id in bound_audio else 1),
                (0 if asset.audio_role == AUDIO_ROLE_LIP_SYNC else 1))

    _take(images, user_images, timeline.limits.images, "image")
    _take(videos, user_videos, timeline.limits.videos, "video")
    _take(audios, user_audios, timeline.limits.audios, "audio", _audio_keep_rank)

    # Built under the project's ceiling: this bin has just been trimmed to it,
    # so validating it against the documented default would reject the very
    # window the ceiling was raised to allow.
    return AssetBin(images + videos + audios, limits=timeline.limits), diagnostics


def _build_subjects(bin_, tag_map):
    """Promote described reference images and videos to <Subject N> definitions.

    Per MiniMax's reference-mode format, an image whose job is to define how
    someone looks belongs inside a Subject definition citing that picture, not as
    a bare <Picture N> mention in prose. Undescribed assets get no subject and are
    referenced by their raw tag.

    Returns (subject_lines, subject_tag_by_asset, retention_meta).
    """
    subject_lines = []
    subject_tag = {}
    retention_meta = []
    n = 0

    for asset in list(bin_.by_kind(KIND_IMAGE)) + list(bin_.by_kind(KIND_VIDEO)):
        if asset.synthetic:
            continue  # a carry-over anchor is a keyframe citation, not a character
        if not asset.description:
            continue
        n += 1
        tag = tag_map.by_id[asset.asset_id]
        subject_lines.append("<Subject %d> is %s (from `%s`)." % (n, asset.description.rstrip("."), tag))
        subject_tag[asset.asset_id] = "<Subject %d>" % n
        retention_meta.append((n, tag, asset.retention or RETENTION_DEFAULT))

    return subject_lines, subject_tag, retention_meta


def assign_speaker_ids(shots, audio_assets=()):
    """Asset id -> "S1", "S2", ... in first-vocal-appearance order.

    Computed once over the *whole* timeline, deliberately not per window. A
    speaker id is what tells H3 that the person talking in shot 9 is the person
    who talked in shot 2, so it has to survive the seam between them. Numbering
    per window is the obvious implementation -- a window is the unit that gets
    compiled -- and it would quietly turn every character into a new person at
    each cut, which is the failure the ids exist to prevent.

    Order is first appearance rather than bin order for the same reason MiniMax
    specifies it that way: the ids read as a cast list in the order the audience
    meets them, and adding a character to the end of the film does not renumber
    the ones already on screen. "First" is measured on the clock, so pass
    `Timeline.ordered_shots()` -- a shot list is not required to be sorted, and
    numbering by list order would hand out ids in the order the nodes happened to
    be wired.
    """
    ids = {}
    for shot in shots:
        for asset_id in (shot.speakers or []):
            if asset_id not in ids:
                ids[asset_id] = SPEAKER_ID_FORMAT % (len(ids) + 1)
    # A bin voice names its own owner, and supplying somebody's voice makes them
    # a speaker whether or not a shot ever named them. They are numbered after
    # everyone who actually has a line, so adding a voice reference cannot
    # renumber a character who does.
    for asset in audio_assets:
        owner = getattr(asset, "voice_of", None)
        if owner and owner not in ids:
            ids[owner] = SPEAKER_ID_FORMAT % (len(ids) + 1)
    return ids


def _speaker_tags(tag_map, subject_tag, speaker_ids):
    """Asset id -> the citation naming both the character and their speaker id.

    "<Subject 2> (S1)" for a described character, "<Picture 3> (S1)" for one
    dropped in without a description. Both are legal in MiniMax's format and the
    second is much weaker, but a speaker id on a bare picture still binds the
    voice to a face, which is the whole job.

    An id whose asset carries no tag in this window is dropped rather than
    emitted against nothing: the reference was pushed out of the budget to make
    room for carry-over, and "(S1)" pointing at an absent picture is worse than
    no binding at all. The caller reports it.
    """
    tags = {}
    for asset_id, speaker_id in speaker_ids.items():
        base = subject_tag.get(asset_id) or tag_map.by_id.get(asset_id)
        if base is not None:
            tags[asset_id] = "%s (%s)" % (base, speaker_id)
    return tags


def _resolve_note_block(text, tag_map, bin_, subject_tag):
    """Resolve @Name / {{id}} references in a global-prompt note block.

    Returns (lines, diagnostics). Blank lines are dropped so an empty labelled
    section contributes nothing rather than an empty prompt line.
    """
    if not text:
        return [], []
    lines, diagnostics = [], []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        resolved, diags = resolve_references(raw, tag_map, bin_, subject_tag)
        diagnostics.extend(diags)
        lines.append(resolved)
    return lines, diagnostics


def _presence_clause(subject_tag, shot_texts):
    """"(present throughout)" only when the tag genuinely appears in every shot.

    A character who walks in at Shot 3 must not be asserted as present from the
    start -- the model takes that literally and puts them in Shot 1.
    """
    if not shot_texts:
        return "(referenced in this window)"
    hits = [i + 1 for i, t in enumerate(shot_texts) if subject_tag in t]
    if not hits:
        return "(referenced in this window)"
    if len(hits) == len(shot_texts):
        return "(present throughout)"
    return "(appears in " + ", ".join("[Shot %d]" % s for s in hits) + ")"


def _compile_window(timeline, index, total, branch, frame_count, start, end, carry,
                    speaker_ids=None):
    diagnostics = []
    shots = timeline.shots_in(start, end)
    shot_ids = [s.shot_id for s in shots]

    bin_, bin_diags = _window_bin(timeline, index, branch, carry, shots)
    diagnostics.extend(bin_diags)
    tag_map = bin_.tag_map()

    subject_lines, subject_tag, retention_meta = _build_subjects(bin_, tag_map)

    # §10 scoping. Built only when some shot actually carries local references --
    # with none, every asset is global and an unscoped resolve is both correct and
    # exactly what every pre-v3 project expects.
    local_refs = getattr(timeline, "local_refs", None) or {}
    scope_by_shot = {}
    if any(local_refs.get(s.shot_id) for s in shots):
        shared = {a.asset_id for a in bin_} - {
            a.asset_id for refs in local_refs.values() for a in refs}
        for shot in shots:
            scope_by_shot[shot.shot_id] = shared | {
                a.asset_id for a in (local_refs.get(shot.shot_id) or [])}

    # ── who is speaking, and with which voice ──────────────────────────────
    # The sockets say nothing about this. An <Audio j> reference reaches the
    # model as a bare ordinal and a waveform the text encoder never sees, so on
    # a two-hander it carries no clue whose voice it is -- which is the shape of
    # Comfy-Org/ComfyUI#15454, where the right mouth moves and the wrong accent
    # comes out of it. The binding is prose or it does not exist.
    speaker_tag = _speaker_tags(tag_map, subject_tag, speaker_ids or {})

    audio_bindings = {}      # audio asset id -> "<Subject 2> (S1)"
    speaker_bindings = {}    # shot id -> the same, for the document and the key

    # A bin voice says who it belongs to itself. Resolved before the shot pass so
    # that an explicit `voice_of` outranks the owner a shot would have inferred
    # for it -- the author naming the character wins over the wiring implying one.
    for asset in bin_.by_kind(KIND_AUDIO):
        owner = asset.voice_of
        if not owner:
            continue
        if owner in speaker_tag:
            audio_bindings[asset.asset_id] = speaker_tag[owner]
        else:
            other = bin_.get(owner)
            diagnostics.append(
                "audio %r is the voice of %r, which carries no reference in this "
                "window -- dropped to make room for carry-over. That voice compiles "
                "unbound rather than naming a character who is not here."
                % (asset.name, other.name if other is not None else owner))

    for shot in shots:
        named = list(shot.speakers or [])
        missing = [a for a in named if a not in speaker_tag]
        if missing:
            def _who(asset_id):
                asset = bin_.get(asset_id)
                return asset.name if asset is not None else asset_id

            diagnostics.append(
                "shot %r names speaker(s) %s, which carry no reference in this window "
                "-- dropped to make room for carry-over. Those lines compile without a "
                "speaker id rather than pointing (S1) at a picture that is not here."
                % (shot.shot_id, ", ".join(repr(_who(a)) for a in missing)))
        bound = [a for a in named if a in speaker_tag]
        if not bound:
            continue
        # Every named speaker is stamped in this shot's prose; the *first* is the
        # one its reference audio belongs to. A shot with two speakers and one
        # voice recording has to choose, and choosing the one the author wrote
        # first is the only ordering the data carries. `PulseShot.speaker` is a
        # single field, so this only arises in hand-written timeline_data.
        speaker_bindings[shot.shot_id] = speaker_tag[bound[0]]
        # A shot's own reference audio belongs to that shot's speaker. Bin audio
        # is deliberately left unbound: it is shared across the film, so there is
        # no shot to read a speaker off, and guessing one would bind a voice to
        # whoever happened to talk first.
        for asset in (local_refs.get(shot.shot_id) or []):
            if asset.kind == KIND_AUDIO:
                audio_bindings.setdefault(asset.asset_id, speaker_tag[bound[0]])

    # ── shot lines, with strictly increasing timestamps ─────────────────────
    shot_lines = []
    shot_texts = []
    resolved_shots = {}
    unresolved_shots = {}
    previous_ts = None
    ordinal = 0
    for shot in shots:
        text = (shot.prompt or "").strip()
        if not text:
            continue
        ordinal += 1
        # A speaker's id is stamped only in the shots they actually speak in.
        # Stamping every mention would put "(S1)" on a character standing
        # silently in the background, which reads to the model as a cue to give
        # them a line.
        speaking = {a: speaker_tag[a] for a in (shot.speakers or []) if a in speaker_tag}
        resolved, ref_diags = resolve_references(
            text, tag_map, bin_, dict(subject_tag, **speaking) if speaking else subject_tag,
            scope_ids=scope_by_shot.get(shot.shot_id))
        diagnostics.extend("shot %r: %s" % (shot.shot_id, d) for d in ref_diags)
        # Whatever still matches the reference syntax after substitution is, by
        # definition, exactly what failed to resolve -- every hit that resolved
        # was replaced by a tag. Cheaper and more precise than parsing the
        # diagnostic strings back apart.
        unresolved = [m.group(0) for m in REF_TOKEN_RE.finditer(resolved)]
        if unresolved:
            unresolved_shots[shot.shot_id] = unresolved
        resolved = wrap_dialogue(resolved, timeline.dialogue_language)
        resolved_shots[shot.shot_id] = resolved
        shot_texts.append(resolved)

        # A lip-sync reference and a quoted line are two different answers to
        # "what is this character saying": the <d> block instructs the model to
        # speak those words, and the recording says whatever it says. The model
        # will pick one, and nothing about the output reveals which. Reported
        # rather than resolved -- a quote that *is* the transcript of the
        # recording is legitimate and even helpful, and only the author knows.
        if "<d>" in resolved:
            lip_sync = [a for a in (local_refs.get(shot.shot_id) or [])
                        if a.kind == KIND_AUDIO and a.audio_role == AUDIO_ROLE_LIP_SYNC]
            if lip_sync:
                diagnostics.append(
                    "shot %r: has quoted dialogue and a lip_sync reference (%s). The "
                    "quote tells the model what to say and the recording says what it "
                    "says; unless the quote is that recording's transcript, drop it and "
                    "let the audio carry the words."
                    % (shot.shot_id, ", ".join("@" + a.name for a in lip_sync)))

        if ordinal == 1:
            # H3's format leaves the opening shot unstamped -- it is the window's
            # zero by definition, and stamping it invites the model to wait.
            shot_lines.append("[Shot 1] %s" % resolved)
            previous_ts = 0
            continue

        # Compare at the resolution the format actually emits. Two shots 100ns
        # apart are distinct floats but render the same MM:SS.mmm, so a raw
        # float comparison would pass the guard and still emit a duplicate stamp.
        offset_ms = max(0, round((shot.start - start) * 1000.0))
        if previous_ts is not None and offset_ms <= previous_ts:
            nudged = previous_ts + 1
            diagnostics.append(
                "shot %r at %s collides with the previous stamp in window %d; "
                "nudged to %s to keep timestamps strictly increasing"
                % (shot.shot_id, format_timestamp(offset_ms / 1000.0), index + 1,
                   format_timestamp(nudged / 1000.0)))
            offset_ms = nudged
        shot_lines.append("[Shot %d] At %s, %s"
                          % (ordinal, format_timestamp(offset_ms / 1000.0), resolved))
        previous_ts = offset_ms

    if not shot_lines:
        diagnostics.append(
            "window %d has no shot text; it will render on the style line and "
            "continuity instruction alone" % (index + 1,))

    # ── the ordered file list ──────────────────────────────────────────────
    files = []
    for socket, asset_id in sorted(tag_map.sockets.items(), key=_socket_sort_key):
        asset = bin_.get(asset_id)
        tag = (tag_map.by_id.get(asset_id + "#soundtrack")
               if socket.startswith("ref_video_audio_") else tag_map.by_id.get(asset_id))
        files.append(FileRef(socket, asset.kind, asset.asset_id, asset.name, asset.file,
                             tag, asset.trim_start, asset.trim_end, asset.synthetic,
                             asset.audio_role))

    # ── anchors (fl2va only) ───────────────────────────────────────────────
    anchors = {}
    if branch == BRANCH_FL2VA:
        # §11 keyframe_pairs. Per-shot anchors, when the node layer supplied them,
        # outrank the project-level pair. H3 accepts a keyframe at index 0 or at
        # frame_count-1 and nowhere else (constants.ANCHOR_FIRST), so "each shot's
        # start image pairs with the next shot's start image" is realisable exactly
        # when a window holds one shot -- which is what the mode is for. A window
        # holding several uses its first shot's opening frame and its last shot's
        # closing frame, which is the same statement at the window's own scale.
        shot_anchors = getattr(timeline, "shot_anchors", None) or {}
        explicit_first = (shot_anchors.get(shots[0].shot_id, {}).get("first")
                          if shots else None)
        explicit_last = (shot_anchors.get(shots[-1].shot_id, {}).get("last")
                         if shots else None)

        if explicit_first:
            anchors["first_frame"] = explicit_first
        elif index == 0:
            if timeline.first_frame:
                anchors["first_frame"] = timeline.first_frame
        else:
            # Continuation on fl2va: the previous window's last decoded frame is
            # the hard lock. This is the one thing fl2va does better than ref2va.
            anchors["first_frame"] = CARRY_IMAGE_ID

        if explicit_last:
            anchors["last_frame"] = explicit_last
        elif timeline.last_frame and index == total - 1:
            anchors["last_frame"] = timeline.last_frame
        # Both anchors are legal here and nowhere else: PackedLayout accepts
        # pixel_index 0 and frame_count-1 only.
        anchors["first_frame_index"] = 0
        if "last_frame" in anchors:
            anchors["last_frame_index"] = last_anchor_index(frame_count)

    # Identity and retention notes from the global prompt box are authored text
    # like any shot, so they get the same reference resolution. Without this an
    # @Name written in the global box would reach the model as a literal "@Name".
    extra_subject_lines, note_diags = _resolve_note_block(
        getattr(timeline, "identity_notes", ""), tag_map, bin_, subject_tag)
    diagnostics.extend("global prompt (identity): %s" % d for d in note_diags)
    extra_retention_lines, note_diags = _resolve_note_block(
        getattr(timeline, "retention_notes", ""), tag_map, bin_, subject_tag)
    diagnostics.extend("global prompt (retention): %s" % d for d in note_diags)

    prompt = (_assemble_reference_prompt if branch == BRANCH_REF2VA
              else _assemble_base_prompt)(
        timeline, index, total, bin_, tag_map, subject_lines, subject_tag,
        retention_meta, shot_lines, shot_texts, anchors, frame_count, carry,
        extra_subject_lines, extra_retention_lines, audio_bindings)

    diagnostics.extend(_uncited_references(bin_, tag_map, prompt))

    return CompiledWindow(index, total, branch, frame_count, start, end, prompt,
                          files, tag_map, anchors, diagnostics, shot_ids, seed_offset=index,
                          resolved_shots=resolved_shots, unresolved_shots=unresolved_shots,
                          speaker_bindings=speaker_bindings)


def _shot_boundaries(timeline):
    """Where the cuts are, in frames, for a boundary-aware partition policy.

    Deliberately NOT snapped to the 17k+5 grid. A cut is at whatever frame the
    user's durations put it at; snapping it here would move the target before the
    partitioner ever got to try to hit it, and it would then "align" to a
    boundary that is not where the shot actually ends.

    `frames` takes this as a plain sequence of integers and imports nothing from
    `timeline`, which is what keeps the partitioner testable on numbers alone.

    The last shot's end is not a cut. It is where the film stops, and there is no
    following window for a shot to spill into -- counting it would have the policy
    report the end of the timeline as a boundary it failed to reach. It only stays
    in when the timeline runs on past the shots, where it really is an interior
    boundary.
    """
    fps = float(timeline.fps)
    end_of_film = float(timeline.duration_seconds)
    cuts = []
    for shot in timeline.ordered_shots():
        if shot.end >= end_of_film - 1e-6:
            continue
        position = int(round(shot.end * fps))
        if position > 0 and position not in cuts:
            cuts.append(position)
    return cuts


def _uncited_references(bin_, tag_map, prompt):
    """Every reference that occupies a socket without being named in the prompt.

    THE MOST EXPENSIVE SILENT FAILURE IN THE PACK

    A reference the prompt never cites is not ignored -- it is packed into the
    conditioning and attended to on every sampling step of every frame, for the
    whole window. It costs its share of VRAM, it costs a slot out of the 9/3/3/12
    budget, and it can only push the render around at random, because nothing in
    the text tells the model what it is or what to do with it. The tokenizer sees
    the marker "<Picture 2>: " and nothing else; the image itself carries no
    instruction.

    Nothing reported this. The bin showed the asset, the budget meter counted it,
    and the render was quietly worse.

    WHICH REFERENCES CAN ACTUALLY REACH THIS STATE

    Fewer than you would expect, and the check is written against the finished
    prompt rather than against a rule so that it stays true as the assembler
    changes. As it stands:

    - a described image or video is cited by its own `<Subject N>` definition;
    - a video with `include_audio` is cited by the soundtrack line, which names
      the video's tag as well as the soundtrack's;
    - every audio asset is cited by the audio-role prose, which is the mechanism
      lip-sync depends on;
    - synthetic carry-over assets are cited by their own retention lines, and a
      caller cannot remove one anyway.

    What is left, and what this catches: an image or a video carrying no
    description, dropped into the bin and never named in any shot's prose. That
    is the whole failure -- somebody added a reference and forgot to use it.

    Substring matching is exact rather than approximate: a tag includes its
    closing bracket, so "<Picture 1>" cannot match inside "<Picture 10>".
    """
    notes = []
    for asset in bin_:
        if asset.synthetic:
            continue
        tag = tag_map.by_id.get(asset.asset_id)
        if tag and tag not in prompt:
            notes.append(
                "reference %r is loaded as `%s` but is never mentioned in this "
                "window's prompt. It still occupies a reference slot and is attended "
                "to on every sampling step -- cite it as @%s in a shot, describe it "
                "in the bin, or remove it." % (asset.name, tag, asset.name))
    return notes


def _socket_sort_key(item):
    """Order sockets the way the tokenizer walks ref_items: images, then each
    video preceded by its own soundtrack, then standalone audio."""
    socket = item[0]
    group, slot = socket.rsplit("_", 1)
    rank = {"ref_image": 0, "ref_video_audio": 1, "ref_video": 2, "ref_audio": 4}[group]
    # A soundtrack must sort immediately ahead of its own video, so both key on
    # the same slot with the soundtrack winning the tie.
    return (0 if rank == 0 else (1 if rank in (1, 2) else 2), int(slot), rank)


def _assemble_reference_prompt(timeline, index, total, bin_, tag_map, subject_lines,
                               subject_tag, retention_meta, shot_lines, shot_texts,
                               anchors, frame_count, carry,
                               extra_subject_lines=(), extra_retention_lines=(),
                               audio_bindings=None):
    """ref2va: MiniMax's six-section reference-mode format, in its required order."""
    subject_lines = list(subject_lines)
    retention_lines = [
        "<Subject %d> %s: %s - matches `%s`."
        % (n, _presence_clause("<Subject %d>" % n, shot_texts), retention, tag)
        for n, tag, retention in retention_meta
    ]

    # Audio definitions always follow every visual one, per MiniMax's own worked
    # example, regardless of the order the sockets happened to fill.
    audio_lines = []
    # Kept separate so every visual retention line still precedes every audio
    # one, the order MiniMax's worked example uses, whatever order the sockets
    # happened to fill.
    audio_retention_lines = []
    continuity_bits = []

    carry_image = bin_.get(CARRY_IMAGE_ID)
    if carry_image is not None:
        tag = tag_map.by_id[CARRY_IMAGE_ID]
        retention_lines.append(
            "`%s` ([Shot 1] continuity anchor): fully_preserved - the exact framing, "
            "pose, and camera angle at the end of the previous window." % (tag,))
        continuity_bits.append("continuing directly from `%s`" % (tag,))

    carry_video = bin_.get(CARRY_VIDEO_ID)
    if carry_video is not None:
        tag = tag_map.by_id[CARRY_VIDEO_ID]
        retention_lines.append(
            "`%s` (continuation source): fully_preserved - continues the immediately "
            "preceding window's action and camera framing without a cut." % (tag,))
        continuity_bits.append("continuing the motion of `%s`" % (tag,))

    for asset in bin_.by_kind(KIND_VIDEO):
        if asset.synthetic or not asset.include_audio:
            continue
        a_tag = tag_map.by_id.get(asset.asset_id + "#soundtrack")
        v_tag = tag_map.by_id.get(asset.asset_id)
        audio_lines.append("`%s` is the soundtrack of `%s`." % (a_tag, v_tag))

    carry_audio = bin_.get(CARRY_AUDIO_ID)
    if carry_audio is not None:
        tag = tag_map.by_id[CARRY_AUDIO_ID]
        audio_lines.append(
            "`%s` is the tail of the previous window's own score and ambience." % (tag,))
        retention_lines.append(
            "`%s` (previous window's tail): partially_copy - the same score and ambience "
            "continues into this window rather than restarting." % (tag,))

    audio_bindings = audio_bindings or {}
    for asset in bin_.by_kind(KIND_AUDIO):
        if asset.synthetic:
            continue
        tag = tag_map.by_id[asset.asset_id]
        detail = (": " + asset.description.rstrip(".")) if asset.description else ""
        # Who the voice belongs to, when a shot named a speaker for it. Without
        # one the sentences fall back to the anonymous phrasing they have always
        # had -- correct on a one-hander, and the only honest thing to say when
        # nothing in the graph identifies the speaker.
        who = audio_bindings.get(asset.asset_id)
        if asset.audio_role == AUDIO_ROLE_LIP_SYNC:
            # The directive *is* the mechanism. The tokenizer only ever emits the
            # marker "<Audio j>: " -- the waveform never reaches Qwen -- so the
            # sockets alone say nothing about what the audio is for. Asking for
            # the match in prose is what makes the DiT track those rows.
            audio_lines.append(
                "`%s` is the speech %s is saying%s. Their lip movements "
                "match `%s` precisely, in time with it."
                % (tag, who or "this character", detail, tag))
        elif who:
            audio_lines.append(
                "`%s` is the voice-timbre reference for %s%s." % (tag, who, detail))
        else:
            audio_lines.append("`%s` is a voice-timbre reference%s." % (tag, detail))

        # A retention line for an audio reference, in MiniMax's own vocabulary
        # rather than the picture words. Emitted only when the voice is bound to
        # someone: "fully_copy" against an unattributed recording tells the model
        # to reproduce a voice without saying whose mouth it comes out of, which
        # is the leak this whole binding exists to close.
        retention = AUDIO_ROLE_RETENTION.get(asset.audio_role)
        if retention and who:
            audio_retention_lines.append(
                "`%s` (the voice of %s): %s - %s"
                % (tag, who, retention,
                   "this exact recording is what is heard, and that mouth moves with it."
                   if asset.audio_role == AUDIO_ROLE_LIP_SYNC else
                   "borrow this voice's timbre and delivery; the words spoken are the "
                   "ones written in this window's shots."))

    summary = _summary_line(subject_tag, continuity_bits, index, total)

    soundscape = timeline.overall_soundscape.strip()
    if carry_audio is not None:
        note = ("The score and ambience carried in from the previous window continue "
                "unbroken through this one.")
        soundscape = (soundscape + " " + note) if soundscape else note

    # Identity locks typed into the global prompt box join the definitions the bin
    # generated. They go last so a generated <Subject N> line is never displaced
    # by prose that has no socket behind it.
    subject_lines = subject_lines + list(extra_subject_lines)
    retention_lines = retention_lines + audio_retention_lines + list(extra_retention_lines)

    sections = []
    if subject_lines or audio_lines:
        sections.append("subject_definitions:\n" + "\n".join(subject_lines + audio_lines))
    sections.append("summary:\n" + summary)
    if retention_lines:
        sections.append("retention_analysis:\n" + "\n".join(retention_lines))
    desc = ([timeline.style_line.strip()] if timeline.style_line.strip() else []) + shot_lines
    if desc:
        sections.append("detailed_description:\n" + "\n".join(desc))
    sections.append("overall_soundscape:\n" + (soundscape or "N/A"))
    sections.append("non_diegetic_music:\n" + (timeline.non_diegetic_music.strip() or "N/A"))
    return "\n\n".join(sections)


def _summary_line(subject_tag, continuity_bits, index, total):
    tags = list(subject_tag.values())
    if not tags:
        base = "The target video follows the shot description below."
    elif len(tags) == 1:
        base = "The target video shows %s." % tags[0]
    else:
        base = "The target video shows %s and %s." % (", ".join(tags[:-1]), tags[-1])
    if continuity_bits:
        base = base.rstrip(".") + ", " + ", ".join(continuity_bits) + "."
    if total > 1:
        base = "[window %d of %d] " % (index + 1, total) + base
    return base


def _assemble_base_prompt(timeline, index, total, bin_, tag_map, subject_lines,
                          subject_tag, retention_meta, shot_lines, shot_texts,
                          anchors, frame_count, carry,
                          extra_subject_lines=(), extra_retention_lines=(),
                          audio_bindings=None):
    """fl2va / t2va: MiniMax's three-section base-mode format.

    `audio_bindings` is accepted and ignored: fl2va takes no references at all
    (constants.BRANCH_ACCEPTS_REFERENCES), so there is no <Audio j> here to bind
    anything to. It is in the signature so both assemblers stay callable through
    the one call site.

    Base mode has no <Subject N> layer and no reference tags -- only the keyframe
    images, which MiniMaxH3ImageToVideo appends in the order (first_frame,
    last_frame). <Picture N> here follows that append order exactly, so the
    numbering below must be derived from which anchors are actually present, not
    from a fixed slot.
    """
    has_first = "first_frame" in anchors
    has_last = "last_frame" in anchors
    picture_n = 0
    lines = []

    if has_first or has_last:
        parts = []
        if has_first:
            picture_n += 1
            parts.append("`<Picture %d>` aligns with the 0.00-second mark" % picture_n)
        if has_last:
            picture_n += 1
            end_ts = format_timestamp(frames_to_seconds(frame_count - 1, timeline.fps))
            parts.append("`<Picture %d>` aligns with the %s mark" % (picture_n, end_ts))
        line = "How the reference pictures align with the target video — " + "; ".join(parts) + "."
        if index > 0:
            line += (" Continue the ongoing action naturally from this pose and framing — "
                     "no restart, no new take.")
        lines.append(line)

    if timeline.style_line.strip():
        lines.append(timeline.style_line.strip())
    # Base mode has no subject_definitions or retention_analysis section, so
    # identity and retention notes ride in the description rather than being
    # silently dropped -- they are still direction the user wrote.
    lines.extend(extra_subject_lines)
    lines.extend(extra_retention_lines)
    lines.extend(shot_lines)

    soundscape = timeline.overall_soundscape.strip()
    if index > 0:
        # This branch encodes no reference audio at all, so the window has no
        # sonic grounding. Left unconstrained, H3 tends to invent vocalisation.
        # Only say so when the shots do not themselves call for speech.
        has_dialogue = any("<d>[" in t for t in shot_texts)
        if not has_dialogue:
            note = ("No reference audio grounds this window — keep the soundscape ambient "
                    "and grounded only in what the shot description states, consistent with "
                    "the previous window's environment. No invented sound effects, and no "
                    "dialogue or vocalisation unless a shot explicitly includes spoken lines.")
            soundscape = (soundscape + " " + note) if soundscape else note

    sections = []
    if lines:
        sections.append("integrated_multimodal_description:\n" + "\n".join(lines))
    sections.append("overall_soundscape:\n" + (soundscape or "N/A"))
    sections.append("non_diegetic_music:\n" + (timeline.non_diegetic_music.strip() or "N/A"))
    return "\n\n".join(sections)
