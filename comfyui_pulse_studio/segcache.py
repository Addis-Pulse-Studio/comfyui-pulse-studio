"""Segment cache, manifest and resume. Spec §7.

This is the module that turns a twelve-window render from an all-or-nothing
overnight job into something you can interrupt, edit and requeue. Everything it
does rests on one decision, §7.1: **the cache key is content, never position.**

Keying on window index cannot tell "shot 3 was edited" from "a shot was inserted
before shot 3". The first should re-render one window; the second should
re-render none, because every window still holds the same shots doing the same
things -- they have merely moved along the clock. An index-keyed cache gets both
wrong, and gets them wrong in the expensive direction.

WHAT GOES INTO A KEY
--------------------
Exactly the §7.1 list, in the §7.1 order: the global block, this window's shot
blocks, this window's resolved reference descriptors, the window's frame and
canvas geometry, its sampler settings, its continuity edges, and the two
fingerprints -- of the checkpoint and of every approximation patched onto it.
`node_version` closes the loop: a build that assembles prompts differently must
not reuse segments assembled by the old rules.

The order is preserved by hashing a *list of pairs* rather than a dict. Dicts go
through `canonical_json`, which sorts keys and would silently discard the
ordering the spec asks for.

CRASH SAFETY
------------
§7.4: the manifest is written after the media files are closed, never before, and
is fsynced before the next window starts. A crash between the two must leave a
manifest that does not claim a file that is absent -- so the write order is
media -> fsync media -> manifest temp -> fsync temp -> atomic replace -> fsync
dir. A manifest entry is a promise that the bytes are on disk, and this module
never makes that promise early.

Deliberately free of ComfyUI and torch imports (§15.3), so the whole resume
behaviour in §7.5 is testable on a machine with no GPU by stubbing the sampler.
"""

import json
import os
import tempfile

from .pulse_timeline import canonical_json, shots_of_window

__all__ = [
    "MANIFEST_SCHEMA",
    "MANIFEST_NAME",
    "CACHE_MODES",
    "CACHE_AUTO",
    "CACHE_FORCE",
    "CACHE_REUSE_ONLY",
    "ReuseOnlyMiss",
    "Manifest",
    "SegmentDecision",
    "cache_key",
    "cache_key_material",
    "derive_run_id",
    "segment_stem",
    "segment_paths",
    "plan_window",
    "seconds_per_frame",
]

MANIFEST_SCHEMA = 1
MANIFEST_NAME = "manifest.json"

CACHE_AUTO = "auto"
CACHE_FORCE = "force_rerender"
CACHE_REUSE_ONLY = "reuse_only"
CACHE_MODES = (CACHE_AUTO, CACHE_FORCE, CACHE_REUSE_ONLY)

STATUS_COMPLETE = "complete"

#: How many hex characters of the cache key go into a filename. The full key
#: stays in the manifest; the filename only has to be unique within one run
#: folder and short enough to read in a directory listing.
FILENAME_KEY_CHARS = 12


class ReuseOnlyMiss(RuntimeError):
    """`cache_mode=reuse_only` and a window is not on disk. Spec §7.4.4.

    Aborts the run rather than rendering, because `reuse_only` is what a user
    selects when they are assembling a final cut and want to be certain that not
    one frame is silently regenerated at a different seed or a different patch
    setting.
    """


# ── the cache key (§7.1) ────────────────────────────────────────────────────

def _ref_descriptor_key(ref):
    """The four fields of a reference that change the render. Spec §7.1.

    Deliberately not the filename: the same picture re-uploaded under a new name
    is the same picture, and its ordinal and digest say so. Deliberately *is* the
    ordinal, because moving a reference from <Picture 2> to <Picture 5> changes
    every prompt that cites it.
    """
    key = [ref.get("ordinal"), ref.get("kind"), ref.get("alias", ""), ref.get("sha256", "")]
    # §7.1 names four fields. `audio_role` is a fifth, and it has to be here: the
    # role decides what the prompt asks the model to do with this recording -- lip
    # sync or timbre -- and that sentence lives in the window's subject
    # definitions, not in any shot's `resolved_prompt`. Without it, flipping the
    # widget changes what the model is told and the cache hands back the segment
    # rendered under the other instruction, which is the exact failure §7.1 exists
    # to prevent.
    #
    # Appended only when set, so that every timeline with no audio reference keeps
    # the keys it already has on disk. Invalidating a user's whole cache to record
    # a distinction their film does not make would be a poor trade.
    if ref.get("audio_role"):
        key.append(ref["audio_role"])
    # And `voice_of` is a sixth, for the same reason again: it decides whether
    # this recording's definition line names a character or says "this
    # character", and it decides whether a retention line is emitted for it at
    # all. Both live in the window's subject definitions, which no shot's
    # `resolved_prompt` covers. Appended after `audio_role` and only when set, so
    # neither a role-less nor an unbound reference moves the key it already has.
    if ref.get("voice_of"):
        key.append(ref["voice_of"])
    return key


def _shot_key(shot):
    """The seven shot fields §7.1 names, in its order.

    `speaker_binding` is an eighth, appended for the same reason `audio_role` is
    appended to a reference descriptor: it changes what the model is told and it
    lives somewhere `resolved_prompt` does not reach. The binding sentence that
    names this shot's voice sits in the window's subject definitions, and the
    (Sx) number in it moves whenever an earlier shot gains a speaker -- so the
    resolved string is hashed, not the asset id behind it.

    Appended only when set, so every timeline that names no speaker keeps the
    keys already on disk.
    """
    key = [
        shot.get("shot_id"),
        shot.get("label", ""),
        shot.get("visual", ""),
        shot.get("audio_line", ""),
        shot.get("duration_seconds"),
        shot.get("continuity"),
        shot.get("resolved_prompt", ""),
    ]
    if shot.get("speaker_binding"):
        key.append(shot["speaker_binding"])
    return key


def cache_key_material(timeline, window, model_fingerprint, patch_fingerprint):
    """The exact list §7.1 hashes, in the order it specifies.

    Exposed separately from `cache_key` so a test can assert *what* is hashed --
    and so a debugging session can diff two windows' material and see which field
    moved, rather than staring at two different hex strings.
    """
    shots = shots_of_window(timeline, window)

    refs = list((timeline.get("refs") or {}).get("global") or [])
    for shot in shots:
        refs.extend(shot.get("local_refs") or [])

    return [
        ["global", timeline.get("global") or {}],
        ["shots", [_shot_key(s) for s in shots]],
        ["refs", [_ref_descriptor_key(r) for r in refs]],
        ["frames", window.get("frames")],
        ["fps", window.get("fps")],
        ["width", window.get("width")],
        ["height", window.get("height")],
        ["seed", window.get("seed")],
        ["steps", window.get("steps")],
        ["sampler", window.get("sampler")],
        ["scheduler", window.get("scheduler")],
        ["cfg", window.get("cfg")],
        ["continuity_in", window.get("continuity_in")],
        ["continuity_out", window.get("continuity_out")],
        ["model_fingerprint", model_fingerprint],
        ["patch_fingerprint", patch_fingerprint],
        ["node_version", timeline.get("node_version")],
    ]


def cache_key(timeline, window, model_fingerprint, patch_fingerprint):
    """The full sha256 hex cache key for one window. Spec §7.1.

    §14.8 forbids computing this without a patch fingerprint, so an empty one is
    refused here rather than quietly hashed -- a key that ignores the attention
    settings is exactly the key that hands back a film whose shots do not match.
    """
    import hashlib

    if not patch_fingerprint:
        raise ValueError(
            "refusing to compute a cache key with no patch_fingerprint (spec §14.8). "
            "Even 'nothing detected' has a fingerprint; an empty one means the "
            "detector was never run.")
    material = cache_key_material(timeline, window, model_fingerprint, patch_fingerprint)
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def derive_run_id(timeline):
    """The folder a project resumes into when `run_id` is left empty. Spec §7.2.

    Hashed over the timeline with every window's `seed` and `cache_key` stripped,
    so that changing the base seed reuses the same folder (and simply misses every
    key inside it) rather than stranding the previous run's segments in a
    directory nothing will look in again.
    """
    import hashlib

    reduced = dict(timeline)
    reduced["windows"] = [
        {k: v for k, v in window.items() if k not in ("seed", "cache_key")}
        for window in timeline.get("windows") or []
    ]
    digest = hashlib.sha256(canonical_json(reduced).encode("utf-8")).hexdigest()
    return digest[:FILENAME_KEY_CHARS]


# ── on-disk layout (§7.2) ───────────────────────────────────────────────────

def segment_stem(window_index, key):
    """`seg_0000_<cachekey12>` -- the shared stem of one segment's files."""
    return "seg_%04d_%s" % (int(window_index), str(key)[:FILENAME_KEY_CHARS])


def segment_paths(window_index, key):
    """Relative filenames for one segment. Relative, because the manifest stores
    them that way -- a run folder that is moved or copied must still resolve."""
    stem = segment_stem(window_index, key)
    return {
        "video_path": "%s.mp4" % stem,
        "audio_path": "%s.audio.flac" % stem,
        "last_frame_path": "%s.last.png" % stem,
    }


# ── the manifest (§7.3) ─────────────────────────────────────────────────────

class Manifest:
    """`manifest.json` for one run folder, with durable writes.

    Entries are addressed by `cache_key`, never by window index. Two windows with
    identical content share a key and therefore share one segment on disk -- a
    repeated establishing shot renders once.
    """

    def __init__(self, directory, run_id="", node_version="", model_fingerprint="",
                 created_utc="", data=None):
        self.directory = directory
        self.data = data if data is not None else {
            "schema": MANIFEST_SCHEMA,
            "run_id": run_id,
            "node_version": node_version,
            "created_utc": created_utc,
            "updated_utc": created_utc,
            "model_fingerprint": model_fingerprint,
            "segments": [],
        }

    # ── loading ─────────────────────────────────────────────────────────────

    @property
    def path(self):
        return os.path.join(self.directory, MANIFEST_NAME)

    @classmethod
    def load(cls, directory, **defaults):
        """Read a manifest, or return a fresh one.

        A manifest that exists but cannot be parsed is *not* silently replaced:
        it is moved aside to `manifest.corrupt.json` first. Overwriting it would
        destroy the only record of which of the files sitting in that folder are
        usable, and those files may represent hours of GPU time.
        """
        path = os.path.join(directory, MANIFEST_NAME)
        if not os.path.exists(path):
            return cls(directory, **defaults)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
                raise ValueError("manifest is not a segment document")
        except (ValueError, OSError):
            spoiled = os.path.join(directory, "manifest.corrupt.json")
            try:
                os.replace(path, spoiled)
            except OSError:  # pragma: no cover - unwritable directory
                pass
            return cls(directory, **defaults)
        return cls(directory, data=data)

    # ── reading ─────────────────────────────────────────────────────────────

    @property
    def segments(self):
        return self.data.setdefault("segments", [])

    def entry_for(self, key):
        """The completed entry with this cache key, or None."""
        for entry in self.segments:
            if entry.get("cache_key") == key:
                return entry
        return None

    def files_present(self, entry):
        """True when every file the entry names exists on disk. Spec §7.4.3.

        A manifest entry alone is not evidence. Folders get cleaned, drives get
        moved, and a claim that a segment exists is worth checking before the
        run skips rendering it.
        """
        if not entry:
            return False
        for field in ("video_path", "audio_path", "last_frame_path"):
            rel = entry.get(field)
            if not rel:
                continue
            if not os.path.exists(os.path.join(self.directory, rel)):
                return False
        return bool(entry.get("video_path"))

    def resolve(self, entry, field):
        """Absolute path for one of an entry's file fields, or None."""
        rel = (entry or {}).get(field)
        return os.path.join(self.directory, rel) if rel else None

    # ── writing ─────────────────────────────────────────────────────────────

    def upsert(self, entry):
        """Add or replace the entry with this cache key. Does not write to disk."""
        key = entry.get("cache_key")
        for i, existing in enumerate(self.segments):
            if existing.get("cache_key") == key:
                self.segments[i] = entry
                break
        else:
            self.segments.append(entry)
        return entry

    def save(self, updated_utc=""):
        """Write durably: temp file, fsync, atomic replace, fsync directory.

        Called only after the segment's media files are closed and fsynced. The
        ordering is the crash-safety contract in §7.4 -- a manifest must never
        claim a file that is not there.
        """
        if updated_utc:
            self.data["updated_utc"] = updated_utc
        os.makedirs(self.directory, exist_ok=True)
        payload = json.dumps(self.data, indent=2, sort_keys=True)

        handle, temp_path = tempfile.mkstemp(dir=self.directory, prefix=".manifest-",
                                             suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:  # pragma: no cover
                pass
            raise
        _fsync_dir(self.directory)
        return self.path

    # ── timing (§9.6, §12.5) ────────────────────────────────────────────────

    def seconds_per_frame(self):
        """Measured render cost, excluding the warm-up window. Spec §12.5.2.

        The first window at any new token count pays Sol-Attn's Triton autotune
        sweep inside the sampling loop. Averaging it in would inflate every
        estimate for the rest of the film, so entries flagged `warmup` are left
        out and recorded separately.
        """
        return seconds_per_frame(self.segments)


def _fsync_dir(directory):
    """fsync a directory so a rename is durable. No-op where unsupported."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except (OSError, AttributeError):  # pragma: no cover - Windows
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - filesystems that refuse
        pass
    finally:
        os.close(fd)


def seconds_per_frame(segments):
    """Average seconds-per-frame across completed, non-warmup segments, or None."""
    total_seconds = 0.0
    total_frames = 0
    for entry in segments or []:
        if entry.get("status") != STATUS_COMPLETE or entry.get("warmup"):
            continue
        frames = entry.get("frames") or 0
        seconds = entry.get("render_seconds") or 0.0
        if frames > 0 and seconds > 0:
            total_seconds += float(seconds)
            total_frames += int(frames)
    return (total_seconds / total_frames) if total_frames else None


# ── the execution decision (§7.4) ───────────────────────────────────────────

class SegmentDecision:
    """What the executor should do with one window, and why.

    `reason` is carried so the report can say "will reuse" or "will render" *and*
    say what decided it -- a user staring at an unexpected re-render needs to know
    whether it was the seed, the patch chain, or a missing file.
    """

    __slots__ = ("action", "key", "entry", "reason")

    def __init__(self, action, key, entry=None, reason=""):
        self.action = action          # "reuse" | "render"
        self.key = key
        self.entry = entry
        self.reason = reason

    @property
    def reuse(self):
        return self.action == "reuse"

    def to_dict(self):
        return {"action": self.action, "cache_key": self.key, "reason": self.reason}

    def __repr__(self):  # pragma: no cover - debugging aid
        return "SegmentDecision(%s, %s)" % (self.action, self.key[:12])


def plan_window(manifest, key, cache_mode=CACHE_AUTO, window_index=0):
    """Decide reuse vs render for one window. Spec §7.4 steps 1-5, in order.

    Raises `ReuseOnlyMiss` for `reuse_only` when the segment is not on disk,
    naming the window -- which §7.4.4 requires to be the *first* missing one, and
    is, because the executor calls this in window order and the exception stops
    the loop.
    """
    if cache_mode not in CACHE_MODES:
        raise ValueError("cache_mode must be one of %r, got %r" % (CACHE_MODES, cache_mode))

    if cache_mode == CACHE_FORCE:
        return SegmentDecision("render", key,
                               reason="cache_mode is force_rerender")

    entry = manifest.entry_for(key)
    if entry is not None and entry.get("status") == STATUS_COMPLETE:
        if manifest.files_present(entry):
            return SegmentDecision("reuse", key, entry,
                                   reason="cached segment on disk")
        missing = "a file named in the manifest is missing from disk"
    elif entry is not None:
        missing = "manifest entry is %r, not complete" % (entry.get("status"),)
    else:
        missing = "no cached segment for this content"

    if cache_mode == CACHE_REUSE_ONLY:
        raise ReuseOnlyMiss(
            "cache_mode is reuse_only, but window %d has no reusable segment: %s "
            "(cache key %s). Set cache_mode to 'auto' to render the windows that "
            "are missing, or check that run_dir and run_id still point at the run "
            "you meant." % (window_index, missing, key[:FILENAME_KEY_CHARS]))

    return SegmentDecision("render", key, reason=missing)


def segment_entry(window, key, paths, render_seconds=0.0, warmup=False,
                  patch_fingerprint="", status=STATUS_COMPLETE):
    """One `segments[]` record. Spec §7.3.

    `patch_fingerprint` is stored per segment as well as per run, because
    `PulseBench` (§12.7) groups timings by it -- and a run folder legitimately
    accumulates segments rendered under different chains as the user A/Bs them.
    """
    entry = {
        "window_index": int(window.get("window_index", 0)),
        "cache_key": key,
        "shot_ids": list(window.get("shot_ids") or []),
        "frames": int(window.get("frames") or 0),
        "fps": int(window.get("fps") or 0),
        "width": int(window.get("width") or 0),
        "height": int(window.get("height") or 0),
        "render_seconds": round(float(render_seconds), 3),
        "patch_fingerprint": patch_fingerprint,
        "status": status,
    }
    entry.update(paths or {})
    if warmup:
        # §12.5.2: kept out of the seconds-per-frame average, recorded so the
        # cost of the Triton sweep is visible rather than merely excluded.
        entry["warmup"] = True
        entry["warmup_seconds"] = round(float(render_seconds), 3)
    return entry


def stale_entries(manifest, live_keys):
    """Manifest entries whose key is not in the current timeline. Spec §7.4.

    Reported, never deleted (§14.5) -- the user may be flipping between two edits
    of the same project, and yesterday's segments are exactly what makes flipping
    back free.
    """
    live = set(live_keys or ())
    return [e for e in manifest.segments if e.get("cache_key") not in live]
