"""The compiler report. Spec §9, §12.5.

WHY A REPORT IS WORTH A WHOLE MODULE
------------------------------------
A wrong ordinal binding renders successfully. `@Mimi` resolving to `<Picture 4>`
instead of `<Picture 3>` produces a complete, well-formed, plausible film of the
wrong person, after however many hours the render took, with nothing in any log
to say anything went wrong. There is no exception to catch and no frame to
inspect that looks broken.

So the report exists to make that failure visible *before* the GPU starts:
per-shot ordinal maps, every unresolved alias, the budget, the patch chain, and
what the cache is about to reuse versus re-render. `dry_run` on `PulseRender`
produces exactly this and nothing else -- no sampling, no decode, no file writes.

PACKED SEQUENCE LENGTHS (§9.7, §12.5)
-------------------------------------
Sol-Attn's Triton kernel autotunes keyed on sequence length, and the first run at
any new token count pays a JIT sweep *inside* the sampling loop. A ragged window
plan therefore pays that sweep once per distinct length. The row arithmetic below
is core's own, read out of `comfy/ldm/minimax/model.py`:

    PackedLayout.__init__     seq_len = text + refs + audio_t*2 + latent_t*frame_rows
    _frame_grid(h, w)         frame_rows = (h//2) * (w//2)
    _empty_av_latent          latent_h, latent_w = height//16, width//16
    patchify_video            patch_size (1, 2, 2)

so `frame_rows == (height // 32) * (width // 32)`.

Text rows are excluded on purpose. They vary with the prompt, so including them
would report every window as a distinct length and the warning would be noise;
what §12.5 actually asks the planner to keep uniform is the geometry, which is
what this measures.

Pure stdlib.
"""

from .constants import (
    AUDIO_LATENT_FPS,
    MAX_REF_AUDIOS,
    MAX_REF_FILES_TOTAL,
    MAX_REF_IMAGES,
    MAX_REF_VIDEOS,
)
from .fingerprint import NO_PATCHES_WARNING, patch_chain_summary
from .frames import frames_to_seconds, video_latent_t

__all__ = [
    "frame_rows",
    "packed_rows",
    "packed_signature",
    "distinct_packed_lengths",
    "build_report",
    "window_table",
]


# ── packed sequence geometry (§9.7) ─────────────────────────────────────────

def frame_rows(width, height):
    """Rows one latent frame occupies after 2x2 patching. Core's `_frame_grid`."""
    return max(1, (int(height) // 32)) * max(1, (int(width) // 32))


def packed_rows(window):
    """Structural packed-sequence rows for a window: target audio + target video.

    Reference and text rows are excluded -- see the module docstring. This is the
    part the *window plan* controls, and therefore the part §12.5 asks to be kept
    uniform across windows.
    """
    frames = int(window.get("frames") or 0)
    fps = int(window.get("fps") or 24) or 24
    rows = frame_rows(window.get("width") or 0, window.get("height") or 0)
    latent_t = video_latent_t(frames)
    audio_t = round(frames_to_seconds(frames, fps) * AUDIO_LATENT_FPS)
    return latent_t * rows + audio_t * 2


def packed_signature(window):
    """What Triton's autotune cache is keyed by, for our purposes.

    Two windows sharing this pay the sweep once between them; two that differ pay
    it twice.
    """
    return (int(window.get("frames") or 0), int(window.get("width") or 0),
            int(window.get("height") or 0))


def distinct_packed_lengths(windows):
    """`{packed_rows: [window_index, ...]}` across a plan, in first-seen order."""
    out = {}
    for window in windows or []:
        out.setdefault(packed_rows(window), []).append(window.get("window_index"))
    return out


# ── report sections ─────────────────────────────────────────────────────────

def _shot_labels(timeline, window):
    by_id = {s["shot_id"]: s for s in timeline.get("shots") or []}
    labels = []
    for sid in window.get("shot_ids") or []:
        shot = by_id.get(sid)
        if shot is None:
            continue
        labels.append(shot.get("label") or (shot.get("visual") or "")[:28].strip() or sid[:8])
    return labels


def window_table(timeline, decisions=None):
    """Section 1: index, shot labels, frames, duration, seed, cache status."""
    decisions = decisions or {}
    lines = ["%-4s %-34s %7s %9s %12s  %s"
             % ("win", "shots", "frames", "seconds", "seed", "cache")]
    lines.append("-" * 88)
    for window in timeline.get("windows") or []:
        index = window.get("window_index")
        labels = ", ".join(_shot_labels(timeline, window)) or "(no shot text)"
        if len(labels) > 34:
            labels = labels[:31] + "..."
        decision = decisions.get(index)
        if decision is None:
            status = "-"
        else:
            status = "will reuse" if decision.reuse else "will render"
        lines.append("%-4s %-34s %7d %9.2f %12d  %s"
                     % (index, labels, window.get("frames") or 0,
                        frames_to_seconds(window.get("frames") or 0,
                                          window.get("fps") or 24),
                        window.get("seed") or 0, status))
    return lines


def _ordinal_map(timeline):
    """Section 2: which alias resolved to which tag, per shot.

    Per shot rather than per project, because §10 gives scene-local references
    ordinals that continue after the global block *within that shot only*. The
    same alias can legitimately be a different number in two different shots, and
    a project-wide table would flatten exactly the distinction that matters.
    """
    lines = []
    globals_ = (timeline.get("refs") or {}).get("global") or []
    if globals_:
        lines.append("global references (identical in every shot):")
        for ref in globals_:
            lines.append("    %-22s -> %s" % (ref.get("alias") or "(unnamed)",
                                              _tag(ref)))
    for shot in timeline.get("shots") or []:
        local = shot.get("local_refs") or []
        if not local:
            continue
        lines.append("shot %s:" % (shot.get("label") or shot.get("shot_id")))
        for ref in local:
            lines.append("    %-22s -> %s  (scene-local)"
                         % (ref.get("alias") or "(unnamed)", _tag(ref)))
    return lines or ["    (no references)"]


def _tag(ref):
    kind = {"image": "Picture", "video": "Video", "audio": "Audio"}.get(ref.get("kind"), "Picture")
    return "<%s %s>" % (kind, ref.get("ordinal"))


def _unresolved(timeline):
    """Section 3: every unresolved @Alias, with the shot it appeared in."""
    lines = []
    for shot in timeline.get("shots") or []:
        for alias in shot.get("unresolved_aliases") or []:
            lines.append("    %-24s in shot %s"
                         % (alias, shot.get("label") or shot.get("shot_id")))
    return lines


def _budget(timeline):
    """Section 4: usage against the 9/3/3/12 limits."""
    budget = timeline.get("budget") or {}
    return ["    %d/%d images | %d/%d videos | %d/%d audio | %d/%d files"
            % (budget.get("images", 0), budget.get("max_images", MAX_REF_IMAGES),
               budget.get("videos", 0), budget.get("max_videos", MAX_REF_VIDEOS),
               budget.get("audio", 0), budget.get("max_audio", MAX_REF_AUDIOS),
               budget.get("total", 0), budget.get("max_total", MAX_REF_FILES_TOTAL))]


def _patches(descriptor, fingerprint):
    """Section 5: the detected chain and its fingerprint."""
    lines = []
    chain = patch_chain_summary(descriptor)
    if chain:
        lines.extend("    " + line for line in chain)
    else:
        lines.append("    (none detected)")
        lines.append("    WARNING: " + NO_PATCHES_WARNING)
    lines.append("    patch_fingerprint: %s" % (fingerprint or "(not computed)"))
    return lines


def _estimates(timeline, decisions, spf, vram_by_signature):
    """Section 6: estimated peak VRAM per window, and total render time.

    Both are measured rather than modelled wherever a previous run left numbers
    in the manifest. A formula for H3's peak VRAM would be a guess dressed as a
    figure; §12.7 is explicit that the correct handling of performance questions
    here is measurement, not a recommendation baked into code.
    """
    lines = []
    decisions = decisions or {}
    to_render = [w for w in timeline.get("windows") or []
                 if not (decisions.get(w.get("window_index")) or _RENDER).reuse]

    if spf:
        seconds = sum((w.get("frames") or 0) * spf for w in to_render)
        lines.append("    %d window(s) to render, %d cached"
                     % (len(to_render), len(timeline.get("windows") or []) - len(to_render)))
        lines.append("    estimated render time: %s at %.2fs/frame measured on this box"
                     % (_duration(seconds), spf))
    else:
        lines.append("    %d window(s) to render, %d cached"
                     % (len(to_render), len(timeline.get("windows") or []) - len(to_render)))
        lines.append("    estimated render time: unknown -- no completed segment has been "
                     "timed in this run folder yet. The estimate appears from the second "
                     "run onward.")

    for window in timeline.get("windows") or []:
        signature = packed_signature(window)
        measured = (vram_by_signature or {}).get(signature)
        if measured:
            lines.append("    window %s peak VRAM: %.1f GiB (measured)"
                         % (window.get("window_index"), measured / (1024 ** 3)))
    if not vram_by_signature:
        lines.append("    peak VRAM: not yet measured. Recorded per segment from this "
                     "run onward and reported here next time.")
    return lines


class _AlwaysRender:
    reuse = False


_RENDER = _AlwaysRender()


def _duration(seconds):
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "%dh %02dm" % (hours, minutes)
    if minutes:
        return "%dm %02ds" % (minutes, secs)
    return "%ds" % (secs,)


def _packed(timeline):
    """Section 7: distinct packed sequence lengths, with the §12.5 warning."""
    groups = distinct_packed_lengths(timeline.get("windows") or [])
    lines = []
    for rows, indices in groups.items():
        lines.append("    %9d rows  window(s) %s"
                     % (rows, ", ".join(str(i) for i in indices)))
    if len(groups) > 1:
        lines.append(
            "    WARNING: %d distinct packed sequence lengths. Sol-Attn's Triton kernel "
            "autotunes per length, and that sweep runs inside the sampling loop -- this "
            "plan pays it %d times instead of once. It happens when a timeline cannot be "
            "split into equal windows on the 17k+5 grid without dropping one below the "
            "124-frame floor." % (len(groups), len(groups)))
    return lines


# ── the whole report ────────────────────────────────────────────────────────

def build_report(timeline, decisions=None, patch_descriptor=None, patch_fingerprint="",
                 model_fingerprint="", seconds_per_frame=None, vram_by_signature=None,
                 run_directory="", warnings=(), dry_run=False, title="Pulse Render"):
    """The full §9 report as plain text, readable in PreviewAny.

    `decisions` is `{window_index: SegmentDecision}`. Passing none renders the
    table without a cache column, which is what `PulseSlate` does -- it cannot
    know cache status, because the cache key needs the model fingerprint and
    `PulseSlate` does not hold the model that will do the sampling.
    """
    windows = timeline.get("windows") or []
    total_frames = sum(w.get("frames") or 0 for w in windows)
    fps = (windows[0].get("fps") if windows else 24) or 24

    out = []
    out.append("=" * 88)
    out.append("%s  |  %d window(s), %d frames, %.2fs at %dfps"
               % (title, len(windows), total_frames,
                  frames_to_seconds(total_frames, fps), fps))
    if dry_run:
        out.append("DRY RUN -- nothing was sampled, decoded or written.")
    out.append("=" * 88)

    out.append("")
    out.append("1. WINDOWS")
    out.extend(window_table(timeline, decisions))

    out.append("")
    out.append("2. ORDINAL MAP")
    out.extend(_ordinal_map(timeline))

    out.append("")
    out.append("3. UNRESOLVED ALIASES")
    unresolved = _unresolved(timeline)
    if unresolved:
        out.extend(unresolved)
        out.append("    These reached the model as literal text. Check the spelling "
                   "against the Asset Bin, or the shot's own reference sockets.")
    else:
        out.append("    (none -- every @Alias resolved)")

    out.append("")
    out.append("4. REFERENCE BUDGET")
    out.extend(_budget(timeline))

    out.append("")
    out.append("5. UPSTREAM PATCH CHAIN")
    out.extend(_patches(patch_descriptor, patch_fingerprint))
    if model_fingerprint:
        out.append("    model_fingerprint: %s" % (model_fingerprint,))

    out.append("")
    out.append("6. ESTIMATES")
    out.extend(_estimates(timeline, decisions, seconds_per_frame, vram_by_signature))

    out.append("")
    out.append("7. PACKED SEQUENCE LENGTHS")
    out.extend(_packed(timeline))

    notes = list(timeline.get("warnings") or []) + list(warnings or [])
    if notes:
        out.append("")
        out.append("8. WARNINGS")
        for note in notes:
            out.append("    - %s" % (note,))

    if run_directory:
        out.append("")
        out.append("Segments: %s" % (run_directory,))

    return "\n".join(out)
