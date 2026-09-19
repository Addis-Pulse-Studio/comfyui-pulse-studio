"""Which frames of a finished film belong to one character's voice.

A per-character lip-sync pass (LatentSync and the like) repaints a mouth to a
recording. Handed the whole film and the whole mix, it repaints whichever face it
finds to whoever is talking -- the second speaker's lines land on the first
speaker's mouth, and the pass that was meant to fix sync breaks it. So the pass
has to be given one character at a time: only the seconds they speak, only their
own recording.

This module answers the two questions that takes, with no torch and no ComfyUI:

  * which recordings are this character's (`owner_voice_ids`), resolved the same
    way the compiler binds them -- a bin voice by its own `voice_of`, a shot's
    own voice by that shot's first speaker;
  * where in time the recording is actually speaking (`active_spans`), read off
    its level rather than off the window it sits in -- a film-clock narration
    track for one character is mostly silence while the other one talks.

The level is measured by the node layer; this module only reads a list of dB
values, which is what keeps it testable on a box with no audio stack.
"""

import math

from .assets import KIND_AUDIO

#: Hop the level is measured at. 10 ms is finer than a frame at 24 fps.
LEVEL_HOP_SECONDS = 0.01
DEFAULT_THRESHOLD_DB = -40.0
#: Speech has gaps between words that are not the end of a line.
DEFAULT_HANGOVER_SECONDS = 0.30
DEFAULT_MIN_SECONDS = 0.10


def resolve_owner(timeline, speaker):
    """The character asset `speaker` names ('@Ada' or 'Ada'), or ValueError."""
    name = (speaker or "").strip().lstrip("@").strip()
    if not name:
        raise ValueError("Name the character to correct, e.g. @Ada.")
    owner = timeline.assets.find_by_name(name)
    if owner is None:
        for assets in (timeline.local_refs or {}).values():
            for asset in assets:
                if asset.kind != KIND_AUDIO and asset.name.casefold() == name.casefold():
                    owner = asset
                    break
    if owner is None or owner.kind == KIND_AUDIO:
        raise ValueError(
            "No character named %r in this timeline. Use the @Name the Asset Bin "
            "shows for the face, not the voice's own name." % (name,))
    return owner


def owner_voice_ids(timeline, owner_id):
    """Every audio asset bound to `owner_id`.

    A bin (or slate) voice says whose it is in `voice_of`. A shot's own voice
    belongs to that shot's first speaker -- the compiler's rule, repeated here
    rather than re-derived differently.
    """
    ids = {a.asset_id for a in timeline.assets.by_kind(KIND_AUDIO)
           if getattr(a, "voice_of", None) == owner_id}
    for shot in timeline.ordered_shots():
        if not shot.speakers or shot.speakers[0] != owner_id:
            continue
        for asset in (timeline.local_refs or {}).get(shot.shot_id) or []:
            if asset.kind == KIND_AUDIO:
                ids.add(asset.asset_id)
    for assets in (timeline.local_refs or {}).values():
        for asset in assets:
            if asset.kind == KIND_AUDIO and getattr(asset, "voice_of", None) == owner_id:
                ids.add(asset.asset_id)
    return ids


def active_spans(levels_db, hop_seconds=LEVEL_HOP_SECONDS,
                 threshold_db=DEFAULT_THRESHOLD_DB,
                 hangover_seconds=DEFAULT_HANGOVER_SECONDS,
                 min_seconds=DEFAULT_MIN_SECONDS):
    """(start, end) seconds where the level is at or above `threshold_db`.

    A gap shorter than `hangover_seconds` is bridged -- it is a breath between
    words, not the end of a line -- and a burst shorter than `min_seconds` is
    dropped as a click.
    """
    spans = []
    start = None
    for i, level in enumerate(levels_db):
        if level >= threshold_db:
            if start is None:
                start = i
        elif start is not None:
            spans.append((start * hop_seconds, i * hop_seconds))
            start = None
    if start is not None:
        spans.append((start * hop_seconds, len(levels_db) * hop_seconds))

    bridged = []
    for a, b in spans:
        if bridged and a - bridged[-1][1] < hangover_seconds:
            bridged[-1] = (bridged[-1][0], b)
        else:
            bridged.append((a, b))
    return [(a, b) for a, b in bridged if b - a >= min_seconds]


def spans_to_frames(spans, fps, frame_count, handle_seconds=0.0):
    """Second spans -> merged [first, end) frame spans, widened by the handles."""
    frames = []
    for a, b in sorted(spans):
        # Rounded outward, and past float noise first: 7.5 s at 24 fps is frame
        # 180 exactly, not 180.00000000000003 and so 181.
        first = max(0, math.floor(round((a - handle_seconds) * fps, 6)))
        end = min(frame_count, math.ceil(round((b + handle_seconds) * fps, 6)))
        if end <= first:
            continue
        if frames and first <= frames[-1][1]:
            frames[-1] = (frames[-1][0], max(frames[-1][1], end))
        else:
            frames.append((first, end))
    return frames


def merge_spans(spans):
    """Sort [first, end) spans and join the ones that touch or overlap."""
    out = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract_spans(spans, remove):
    """`spans` with every frame in `remove` taken out."""
    out = merge_spans(spans)
    for ra, rb in merge_spans(remove):
        cut = []
        for a, b in out:
            if rb <= a or ra >= b:
                cut.append((a, b))
                continue
            if a < ra:
                cut.append((a, ra))
            if rb < b:
                cut.append((rb, b))
        out = cut
    return out


def speaker_frames(own_spans, other_spans, fps, frame_count, handle_seconds=0.0):
    """The frames one character's correction pass may repaint.

    Their own speech, widened by the handles -- but a handle only reaches into
    silence. Where another character is audible the handle stops, so a pass on
    one speaker never closes the other speaker's mouth at a cut between them.
    Their own speech is kept even where it overlaps someone else's.
    """
    core = spans_to_frames(own_spans, fps, frame_count)
    widened = spans_to_frames(own_spans, fps, frame_count, handle_seconds)
    others = spans_to_frames(other_spans, fps, frame_count)
    return merge_spans(subtract_spans(widened, others) + core)


def lip_sync_voice_ids(timeline):
    """Every audio asset in the film, bin and shot-local alike."""
    ids = {a.asset_id for a in timeline.assets.by_kind(KIND_AUDIO)}
    for assets in (timeline.local_refs or {}).values():
        ids.update(a.asset_id for a in assets if a.kind == KIND_AUDIO)
    return ids
