"""The `PULSE_TIMELINE` document: what the compiler hands the executor.

Spec §3, §6, §10, §11.

`PulseSlate` compiles. `PulseRender` executes. This module is the contract between
them, and it is deliberately a plain JSON-serialisable dict rather than an object
graph, because two things downstream depend on being able to hash it:

  * the per-window `cache_key` (§7.1), which decides whether a segment already on
    disk may be reused, and
  * the `run_id` (§7.2), which decides which folder a project resumes into.

A dict that cannot be canonically serialised cannot be hashed, and a cache whose
key is not reproducible across sessions is not a cache. So: no tensors in here,
ever. Image and audio tensors ride in a side channel keyed by the `*_ref` slot
names -- see `SideChannel` below -- and the dict holds only the slot name and the
content digest.

WHY SEEDS ARE DERIVED FROM SHOT IDS AND NOT FROM POSITION
---------------------------------------------------------
The v2 node set `seed_offset = window_index`. Insert one shot at the top of a
twelve-window film and every window after it changes seed, so every segment in
the cache is invalidated and the whole film re-renders differently. The user's
edit was "add an establishing shot"; what they got was a new film.

`window_seed` below keys on the *set of shots in the window* instead. Inserting,
deleting or reordering shots leaves the seed of any window whose shot set is
unchanged exactly where it was, so the cache survives an edit that did not touch
it. That is the property the whole resume design rests on.

Pure stdlib. No torch, no comfy, no file IO.
"""

import hashlib
import json
import uuid

from .constants import FPS

__all__ = [
    "SCHEMA",
    "NODE_VERSION",
    "CONTINUITY_MODES",
    "SHOT_CONTINUITY_MODES",
    "CONTINUITY_INHERIT",
    "CONTINUITY_NONE",
    "CONTINUITY_LAST_FRAME",
    "CONTINUITY_KEYFRAME_PAIRS",
    "ContinuityError",
    "SideChannel",
    "canonical_json",
    "new_shot_id",
    "text_shot_id",
    "window_seed",
    "resolve_continuity",
    "check_continuity",
    "ref_descriptor",
    "shot_block",
    "build_timeline",
    "timeline_shots_by_id",
    "shots_of_window",
]

# Bumped together. `schema` is the document's shape; `node_version` is the build
# that wrote it, and it is folded into every cache key so that a change to how
# prompts are assembled cannot reuse segments assembled by the old rules.
SCHEMA = 3
NODE_VERSION = "3.0.0"

# ── continuity (§11) ────────────────────────────────────────────────────────

CONTINUITY_NONE = "none"
CONTINUITY_LAST_FRAME = "last_frame_carry"
CONTINUITY_KEYFRAME_PAIRS = "keyframe_pairs"
CONTINUITY_INHERIT = "inherit"

#: What a project may be set to.
CONTINUITY_MODES = (CONTINUITY_NONE, CONTINUITY_LAST_FRAME, CONTINUITY_KEYFRAME_PAIRS)
#: What a single shot may be set to -- the same three, plus deferring to the project.
SHOT_CONTINUITY_MODES = (CONTINUITY_INHERIT,) + CONTINUITY_MODES

#: Modes that cannot run on the reference checkpoint alone.
NEEDS_FL2VA = (CONTINUITY_LAST_FRAME, CONTINUITY_KEYFRAME_PAIRS)


class ContinuityError(ValueError):
    """A continuity mode that cannot be honoured with the inputs provided.

    Raised at compile time and never softened into a fallback. §11 is explicit:
    a mode that needs `model_fl2va` and does not have it must fail loudly. The
    silent alternative -- render it on the reference checkpoint anyway -- produces
    a film whose cuts are visibly wrong, hours later, with nothing in the log.
    """


def resolve_continuity(project_mode, shot_mode):
    """The effective continuity for one shot.

    `inherit` (the shot default) means "whatever the project says", so a user who
    never opens the per-shot dropdown gets one coherent setting for the film.
    """
    if shot_mode in (None, "", CONTINUITY_INHERIT):
        return project_mode
    if shot_mode not in CONTINUITY_MODES:
        raise ContinuityError(
            "shot continuity must be one of %r, got %r" % (SHOT_CONTINUITY_MODES, shot_mode))
    return shot_mode


def check_continuity(shots, project_mode, fl2va_connected):
    """Fatal problems with the continuity plan, as a list of strings.

    Returned rather than raised so the node can show every problem at once, the
    same convention `Timeline.validate` follows. The caller turns a non-empty
    list into an ExecutionBlocker.

    `shots` is a list of shot blocks (see `shot_block`).
    """
    problems = []
    if project_mode not in CONTINUITY_MODES:
        problems.append("continuity must be one of %r, got %r" % (CONTINUITY_MODES, project_mode))
        return problems

    effective = []
    for shot in shots:
        try:
            effective.append(resolve_continuity(project_mode, shot.get("continuity")))
        except ContinuityError as exc:
            problems.append("shot %r: %s" % (shot.get("label") or shot.get("shot_id"), exc))
            effective.append(project_mode)

    used = {m for m in effective if m in NEEDS_FL2VA}
    if used and not fl2va_connected:
        problems.append(
            "continuity mode %s needs the First/Last-Frame checkpoint, but model_fl2va "
            "is not connected. Wire it, or set continuity to 'none'. This is not "
            "falling back silently: the reference checkpoint cannot pin a frame, so "
            "the seam would simply not be held and you would not find out until you "
            "watched the result." % ", ".join(sorted(used)))

    # keyframe_pairs pins shot i's start_image and shot i+1's start_image as the
    # first/last pair of one render. A shot in that chain with no start image has
    # nothing to pin, and H3 accepts a keyframe only at index 0 or frame_count-1
    # (constants.ANCHOR_FIRST) -- so there is no half-measure available.
    chain = [i for i, m in enumerate(effective) if m == CONTINUITY_KEYFRAME_PAIRS]
    for i in chain:
        shot = shots[i]
        if not shot.get("start_image_ref"):
            problems.append(
                "shot %r is set to keyframe_pairs but has no start_image connected. "
                "Every shot in a keyframe_pairs chain needs one -- the mode pairs this "
                "shot's start frame with the next shot's start frame."
                % (shot.get("label") or shot.get("shot_id"),))
    return problems


# ── identity (§6) ───────────────────────────────────────────────────────────

def new_shot_id():
    """A fresh shot id, for a `PulseShot` node being created for the first time.

    UUID4 rather than anything derived from content: the id must survive the user
    rewriting every word of the shot, or editing a line of prose would reroll the
    seed and re-render a window whose *content* the user was refining.
    """
    return str(uuid.uuid4())


def text_shot_id(label, visual):
    """The stable id of a shot that came from the shot text box, not a node.

    Text-box shots have no widget to hold a uuid, so the id is derived from what
    the user typed. Truncated to 16 hex chars -- long enough that a collision
    inside one film is not a practical concern, short enough to read in a report.

    Consequence, and it is the correct one: editing a text-box shot's words gives
    it a new id, which re-seeds and re-renders exactly the window containing it,
    and leaves every other window's cache intact.

    DEVIATION FROM THE SPEC, DELIBERATE
    -----------------------------------
    §6.2 gives the formula as `sha1(label + "|" + visual)`. That concatenation is
    ambiguous: a label of `"a|b"` with a visual of `"c"` hashes identically to a
    label of `"a"` with a visual of `"b|c"`. Two different shots would then share
    an id, and therefore share a seed *and* a cached segment -- one shot would
    silently render as the other, which is the exact class of failure this package
    exists to prevent. The fields are joined through `canonical_json` instead,
    which escapes the separator. No ids were persisted by any earlier build, so
    this costs no migration.
    """
    payload = canonical_json([label or "", visual or ""])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def window_seed(base_seed, shot_ids):
    """The seed for a window holding exactly `shot_ids`. Spec §6.3.

    Deterministic in the shot set and in nothing else -- not the window index, not
    the number of windows, not the position of this window in the film. Two
    renders of the same shots at the same base seed therefore produce the same
    window, which is what makes a cached segment reusable at all.

    Masked to 31 bits because ComfyUI's RandomNoise takes a non-negative int and
    torch's generator seeding is happiest well inside int64.
    """
    digest = hashlib.sha256("|".join(shot_ids).encode("utf-8")).digest()
    return (int(base_seed) ^ int.from_bytes(digest[:4], "big")) & 0x7FFFFFFF


# ── serialisation ───────────────────────────────────────────────────────────

def canonical_json(obj):
    """The one serialisation used wherever this document is hashed. Spec §3.

    `sort_keys` makes the byte string independent of dict insertion order, and the
    tight separators keep it independent of json's default whitespace. Both matter:
    a cache key that changes when Python changes its dict ordering is a cache that
    silently empties itself on an upgrade.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class SideChannel:
    """Tensors belonging to a timeline, kept out of the JSON that gets hashed.

    The timeline dict records slot names (`tensor_slot_0`) and content digests;
    the actual IMAGE/AUDIO tensors live here, addressed by those slot names. The
    executor also needs the CLIP that compiled the prompts -- conditioning for
    window *i+1* cannot be built until window *i* has been decoded (that is what
    `last_frame_carry` means), so it has to be built at render time, and the only
    node holding a CLIP is the compiler.

    Passing it here rather than adding a `clip` socket to `PulseRender` keeps that
    node's face exactly as §2.3 specifies, and keeps the encoder that wrote the
    prompts and the encoder that encodes them provably the same object.
    """

    __slots__ = ("tensors", "digests", "clip", "vae", "audio_vae", "plan", "timeline",
                 "shift_video", "shift_audio", "ref_image_size", "resize_method",
                 "carry_audio_seconds")

    def __init__(self, tensors=None, clip=None, vae=None, audio_vae=None,
                 plan=None, timeline=None, shift_video=12.0, shift_audio=3.0,
                 ref_image_size="match", resize_method="crop", carry_audio_seconds=4.0):
        self.tensors = dict(tensors or {})
        # slot -> content digest, computed once where the tensor is. Asset uses
        # __slots__, so a digest cannot be hung on the asset itself, and it must
        # not be recomputed per window -- hashing nine reference images twelve
        # times over is real time for an answer that cannot have changed.
        self.digests = {}
        self.clip = clip
        self.vae = vae
        self.audio_vae = audio_vae
        # The CompiledPlan and the Timeline it came from. The executor needs both:
        # the plan holds each window's resolved prompt and socket->file mapping,
        # and its windows are index-aligned with the JSON document's `windows`
        # by construction. Rebuilding them in PulseRender would mean compiling
        # twice and hoping the two compiles agree.
        self.plan = plan
        self.timeline = timeline
        # Compile-time sampler settings that are model patches rather than
        # document fields. They belong to the render, so they travel with it.
        self.shift_video = float(shift_video)
        self.shift_audio = float(shift_audio)
        self.ref_image_size = ref_image_size
        self.resize_method = resize_method
        self.carry_audio_seconds = float(carry_audio_seconds)

    def put(self, slot, tensor, digest=""):
        self.tensors[slot] = tensor
        if digest:
            self.digests[slot] = digest
        return slot

    def digest(self, slot):
        return self.digests.get(slot, "")

    def get(self, slot):
        return self.tensors.get(slot) if slot else None

    def __repr__(self):  # pragma: no cover - debugging aid
        return "SideChannel(%d tensor(s), clip=%s)" % (
            len(self.tensors), self.clip is not None)


# ── document builders ───────────────────────────────────────────────────────

# References arriving through an IMAGE/AUDIO socket have no file on disk, so they
# are marked in the one field every layer already carries: the asset id. The node
# layer sees the prefix and pulls the tensor out of the side channel instead of
# calling media.load_image. They are *not* `synthetic` -- a carry-over frame is
# synthetic and must not move the 12-file budget meter, whereas a picture the user
# wired in is a real reference and must.
SOCKET_ID_PREFIX = "__socket__:"


def socket_asset_id(slot):
    """The asset id for a reference arriving on socket `slot`."""
    return SOCKET_ID_PREFIX + str(slot)


def socket_slot_of(asset_id):
    """The socket name behind a socket asset id, or None for a bin asset."""
    text = str(asset_id or "")
    return text[len(SOCKET_ID_PREFIX):] if text.startswith(SOCKET_ID_PREFIX) else None


def ref_descriptor(ordinal, kind, alias, source, file=None, sha256=None,
                   audio_role=None):
    """One entry in `refs.global` or in a shot's `local_refs`. Spec §3.

    `source` is `bin` for a file dragged onto the Asset Bin and `socket` for a
    tensor arriving through an IMAGE/AUDIO input. `sha256` is the content digest:
    of the file's bytes for a bin asset, of the tensor's bytes for a socket one
    (§7.1). It is what lets the cache tell "the same picture, moved" from "a
    different picture in the same slot".
    """
    entry = {"ordinal": int(ordinal), "kind": kind, "alias": alias or "",
             "source": source, "sha256": sha256 or ""}
    if file:
        entry["file"] = file
    if audio_role:
        entry["audio_role"] = audio_role
    return entry


def shot_block(shot_id, index, label="", visual="", audio_line="",
               duration_seconds=5.0, continuity=CONTINUITY_INHERIT,
               start_image_ref=None, end_image_ref=None, local_refs=None,
               resolved_prompt="", unresolved_aliases=None):
    """One entry in `shots`. Spec §3.

    Field order in the returned dict is irrelevant -- everything that hashes this
    goes through `canonical_json`, which sorts.
    """
    return {
        "shot_id": str(shot_id),
        "index": int(index),
        "label": label or "",
        "visual": visual or "",
        "audio_line": audio_line or "",
        "duration_seconds": round(float(duration_seconds), 4),
        "continuity": continuity or CONTINUITY_INHERIT,
        "start_image_ref": start_image_ref,
        "end_image_ref": end_image_ref,
        "local_refs": list(local_refs or []),
        "resolved_prompt": resolved_prompt or "",
        "unresolved_aliases": list(unresolved_aliases or []),
    }


def window_block(window_index, shot_ids, frames, fps=FPS, width=1344, height=736,
                 seed=0, steps=20, sampler="res_multistep", scheduler="simple",
                 cfg=1.0, continuity_in=CONTINUITY_NONE, continuity_out=CONTINUITY_NONE,
                 branch=None, prompt="", cache_key=None):
    """One entry in `windows`. Spec §3.

    `cache_key` is left `None` here on purpose. It cannot be computed by the
    compiler: §7.1 folds `model_fingerprint` and `patch_fingerprint` into it, and
    neither is knowable until the executor has the patched model in hand. The
    executor fills it in. `run_id` derivation (§7.2) strips the field anyway, so a
    null here and a hex string there hash to the same folder.
    """
    return {
        "window_index": int(window_index),
        "shot_ids": list(shot_ids),
        "frames": int(frames),
        "fps": int(fps),
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "steps": int(steps),
        "sampler": sampler,
        "scheduler": scheduler,
        "cfg": float(cfg),
        "continuity_in": continuity_in,
        "continuity_out": continuity_out,
        "branch": branch,
        "prompt": prompt,
        "cache_key": cache_key,
    }


def build_timeline(global_block, global_refs, shots, windows, budget, warnings=()):
    """Assemble the whole `PULSE_TIMELINE` document. Spec §3."""
    return {
        "schema": SCHEMA,
        "node_version": NODE_VERSION,
        "global": dict(global_block),
        "refs": {"global": list(global_refs)},
        "shots": list(shots),
        "windows": list(windows),
        "budget": dict(budget),
        "warnings": list(warnings),
    }


def global_block(style="", identity="", retention="", soundscape="", music="", raw=""):
    """The five labelled global-prompt fields plus the raw text. Spec §3.

    `raw` is carried alongside the parsed fields because the cache key hashes the
    global block (§7.1) and the parse is lossy: two different raw prompts can
    parse to the same five fields, and they are not the same render.
    """
    return {"style": style or "", "identity": identity or "", "retention": retention or "",
            "soundscape": soundscape or "", "music": music or "", "raw": raw or ""}


# ── reading a document back ─────────────────────────────────────────────────

def timeline_shots_by_id(timeline):
    """`shot_id -> shot block`, for the executor and the report builder."""
    return {s["shot_id"]: s for s in timeline.get("shots", [])}


def shots_of_window(timeline, window):
    """The shot blocks belonging to one window, in the window's own order.

    Order matters -- it is hashed into the cache key -- so this walks
    `window["shot_ids"]` rather than filtering the shot list, which would return
    them in document order instead.
    """
    by_id = timeline_shots_by_id(timeline)
    return [by_id[sid] for sid in window.get("shot_ids", []) if sid in by_id]


def is_pulse_timeline(obj):
    """Duck-check for something claiming to be a PULSE_TIMELINE document."""
    return (isinstance(obj, dict) and obj.get("schema") == SCHEMA
            and isinstance(obj.get("windows"), list))
