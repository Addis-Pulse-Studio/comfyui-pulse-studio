"""Autogrow socket-dict shape: contiguous, 0-based, gapless.

`io.Autogrow` inputs arrive at the stock H3 node as a **dict** keyed
`ref_image_0`, `ref_image_1`, ... -- not a list. The node then iterates
`ref_images.values()` and the tokenizer numbers by position, so the dict's
iteration order *is* the tag order.

Which makes a gap silent and dangerous. If `ref_image_1` is missing because its
file failed to load, the dict is {ref_image_0, ref_image_2}: two entries, still
iterating cleanly, and `ref_image_2` is now tagged `<Picture 2>` while every
prompt line written against the bin still means the third asset. The render
succeeds and describes the wrong picture.

Two defences, both here:

  * `drop_missing` runs *before* the tag map is computed, so an unloadable file
    is removed from the bin rather than punched out of the socket dict. Tags are
    then contiguous by construction because they were never assigned.
  * `assert_contiguous` is the net under that, asserted at render time. If a gap
    ever reaches the stock node, it stops loudly instead of misnumbering quietly.
"""

import re

__all__ = [
    "SocketGapError",
    "socket_prefix",
    "socket_index",
    "assert_contiguous",
    "check_socket_groups",
    "drop_missing",
]

_SOCKET_RE = re.compile(r"^(.*?)_(\d+)$")

# The four Autogrow groups the stock node exposes. `ref_video_audio_N` is index-
# paired to `ref_video_N`, so it is deliberately allowed to be sparse -- only
# some videos carry a soundtrack -- while the other three must be dense.
DENSE_GROUPS = ("ref_image", "ref_video", "ref_audio")
SPARSE_GROUPS = ("ref_video_audio",)


class SocketGapError(ValueError):
    """A socket dict with a hole in it, which would shift every tag after it."""


def socket_prefix(name):
    m = _SOCKET_RE.match(name)
    return m.group(1) if m else name


def socket_index(name):
    m = _SOCKET_RE.match(name)
    if not m:
        raise ValueError("%r is not an indexed socket name" % (name,))
    return int(m.group(2))


def assert_contiguous(mapping, prefix):
    """Raise unless `mapping`'s keys for `prefix` are 0..n-1 with no gaps.

    Also checks *iteration order*, because the tokenizer numbers by position in
    the dict, not by the integer in the key. A dict built out of order numbers
    out of order even when every index is present.
    """
    keys = [k for k in mapping if socket_prefix(k) == prefix]
    if not keys:
        return
    indices = [socket_index(k) for k in keys]
    expected = list(range(len(keys)))
    if sorted(indices) != expected:
        raise SocketGapError(
            "%s sockets are not contiguous: got %r, expected %r. A gap shifts every "
            "reference tag after it, so the render would describe the wrong assets."
            % (prefix, sorted(indices), expected))
    if indices != expected:
        raise SocketGapError(
            "%s sockets are present but out of order: %r. The tokenizer numbers by "
            "position in the dict, so iteration order is tag order." % (prefix, indices))


def check_socket_groups(groups):
    """Validate a full {group_name: {socket: value}} bundle.

    `groups` is the shape handed to the stock node:
    {'ref_images': {...}, 'ref_videos': {...}, 'ref_video_audios': {...},
     'ref_audios': {...}}. Empty or None groups are fine.
    """
    by_prefix = {"ref_images": "ref_image", "ref_videos": "ref_video",
                 "ref_audios": "ref_audio", "ref_video_audios": "ref_video_audio"}
    for group, mapping in (groups or {}).items():
        if not mapping:
            continue
        prefix = by_prefix.get(group)
        if prefix is None:
            continue
        if prefix in SPARSE_GROUPS:
            # A soundtrack socket must match a real video socket, but the set is
            # allowed to be sparse -- ref_video_audio_1 with no _0 is correct when
            # only the second video carries sound.
            video_indices = {socket_index(k) for k in (groups.get("ref_videos") or {})}
            for key in mapping:
                if socket_index(key) not in video_indices:
                    raise SocketGapError(
                        "%s has no matching ref_video_%d; a soundtrack is index-paired to "
                        "its own video." % (key, socket_index(key)))
            continue
        assert_contiguous(mapping, prefix)
    return True


def drop_missing(timeline, exists):
    """Remove assets whose files are unavailable, before any tag is assigned.

    `exists` is a callable taking an asset's `file` and returning whether it can
    be loaded. Injected rather than hardcoded so this stays headless and testable.

    Returns a list of human-readable notes, one per dropped asset. Dropping is
    the right move rather than raising: one missing reference should not stop a
    ten-asset project, and the alternative -- keeping it and failing to load it
    later -- is precisely what creates the socket gap.
    """
    notes = []
    for asset in list(timeline.assets):
        if asset.synthetic or not asset.file:
            continue
        if exists(asset.file):
            continue
        timeline.assets.remove(asset.asset_id)
        notes.append(
            "reference %r (%s) was dropped: its file %r could not be found. Every tag "
            "after it has been renumbered accordingly."
            % (asset.name, asset.kind, asset.file))
    return notes
