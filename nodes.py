"""ComfyUI node layer for Pulse Studio.

The nodes here are thin. Every decision that can be made without a tensor was
already made in comfyui_pulse_studio/ and is covered by the headless test suite;
this file marshals sockets and widgets into a compiled plan, and render.py walks
that plan. Nothing here re-derives a frame count, an ordinal, or a cut point.

THE SPLIT (spec §2)
-------------------
Up to v2 one node compiled *and* rendered, and the starter graph carried the
short path and the long path as two parallel wired groups with one of them muted.
That was fragile in a way that bit: a 15-second timeline that split into two
windows sampled internally, and the still-wired single-window group re-sampled
the last window alone and saved 7 seconds of video as if it were the whole film.
No error anywhere -- 175 frames is a perfectly valid latent.

v3 splits the node instead of muting a branch:

    PulseSlate  compiles a PULSE_TIMELINE. Never samples.
    PulseShot   one shot, with real IMAGE sockets so frames can come from
                upstream generators instead of only from files on disk.
    PulseRender executes a PULSE_TIMELINE, reusing every window already cached.
    PulseBench  reads manifests back and says which patch chain is actually
                faster on this box.

A short timeline wires PulseSlate -> your own sampler. A long one wires
PulseSlate -> PulseRender. Nothing is muted and nothing is duplicated.

A standing rule, tested in tests/test_widget_state.py: **execute() never writes to
a widget.** Widget values are inputs. Queuing a graph must not be able to change
what the user typed.
"""

import logging
import math
import os

import torch

import folder_paths
from comfy_execution.graph import ExecutionBlocker

from . import media, render
from .comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO, Asset
from .comfyui_pulse_studio.bench import format_table, group_by_fingerprint, load_manifests
from .comfyui_pulse_studio.compiler import CarryPolicy, compile_timeline
from .comfyui_pulse_studio.constants import (
    AUDIO_ROLE_LIP_SYNC,
    AUDIO_ROLES,
    BRANCH_FL2VA,
    CANVAS_MULTIPLE,
    DEFAULT_AUDIO_CARRY_SECONDS,
    MAX_PIXELS,
    MAX_REF_AUDIOS,
    MAX_REF_AUDIOS_CEILING,
    MAX_REF_FILES_TOTAL,
    MAX_REF_IMAGES,
    MAX_REF_VIDEOS,
    REF_IMAGE_SIZE_OPTIONS,
    SCHEMA_VERSION,
)
from .comfyui_pulse_studio.fingerprint import (
    describe_model_patches,
    patch_fingerprint,
    patch_warnings,
)
from .comfyui_pulse_studio.patches import check_model_patches, check_single_checkpoint
from .comfyui_pulse_studio.pulse_timeline import (
    CONTINUITY_INHERIT,
    CONTINUITY_KEYFRAME_PAIRS,
    CONTINUITY_LAST_FRAME,
    CONTINUITY_MODES,
    CONTINUITY_NONE,
    SHOT_CONTINUITY_MODES,
    SideChannel,
    build_timeline as build_document,
    check_continuity,
    global_block,
    ref_descriptor,
    resolve_continuity,
    shot_block,
    socket_asset_id,
    socket_slot_of,
    text_shot_id,
    window_block,
    window_seed,
)
from .comfyui_pulse_studio.report import build_report
from .comfyui_pulse_studio.retake import RetakeError, plan_retake
from .comfyui_pulse_studio.segcache import CACHE_MODES, ReuseOnlyMiss
from .comfyui_pulse_studio.sockets import SocketGapError, check_socket_groups, drop_missing
from .comfyui_pulse_studio.still import StillError, plan_still
from .comfyui_pulse_studio.timeline import Shot
from .comfyui_pulse_studio.widget_state import EMPTY_TIMELINE_DATA, build_timeline

log = logging.getLogger(__name__)


# ── The slot contract, in one place ─────────────────────────────────────────
# Every node in this pack opens its widget list with `schema_version`, so that
# a saved workflow always states which layout it was written in before anything
# tries to read it back. See spec §3 and js/ps_widget_order.js.


def schema_widget():
    """The `schema_version` widget. Always widget index 0. Never displayed."""
    return ("STRING", {
        "default": SCHEMA_VERSION, "multiline": False,
        "tooltip": "Which widget layout this node was saved with. Written by the "
                   "node, read at load time to restore values by name. Do not edit."})


# `hidden` inputs are not widgets and consume no widgets_values slot, so adding
# one is outside the §3 append-only rule. UNIQUE_ID is what lets a warning be
# addressed to the node that raised it; PROMPT is what lets PulseRender ask
# whether its `frames` output is wired to anything (spec §8).
HIDDEN_INPUTS = {"unique_id": "UNIQUE_ID"}
HIDDEN_INPUTS_WITH_PROMPT = {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}


def _sage_attention_global():
    """ComfyUI's own --use-sage-attention flag, which leaves no model_options
    trace because it patches attention process-wide. Absent on older cores."""
    try:
        import comfy.model_management as mm
        return bool(mm.sage_attention_enabled())
    except Exception:
        return False


def _warn_on_node(unique_id, warnings):
    """Put warnings on the node's face as well as in the console.

    A console message is easy to miss, and the whole point of the patch warning
    is that the failure it predicts (an out-of-memory error on a full-length
    window) arrives much later and looks unrelated. Best-effort: if the frontend
    channel is unavailable -- headless API runs, an older core -- the console
    warning still happened, and nothing here may break the render.
    """
    if not warnings or unique_id is None:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            "pulse_studio.warnings",
            {"node_id": str(unique_id), "warnings": list(warnings)})
    except Exception as exc:  # pragma: no cover - no server in the test env
        log.debug("[PulseStudio] could not push warnings to the node face: %s", exc)


def _report_patches(model, unique_id, branches_used=(), fl2va_connected=False):
    """§10 + §18.1. Warn, never block -- the user may be deliberately unpatched."""
    report = check_model_patches(model, sage_attention_global=_sage_attention_global())
    warnings = list(report.warnings)
    warnings.extend(check_single_checkpoint(branches_used, fl2va_connected))
    for note in warnings:
        log.warning("[PulseStudio] %s", note)
    _warn_on_node(unique_id, warnings)
    return warnings


SAMPLERS = ["res_multistep", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "ddim"]
SCHEDULERS = ["simple", "normal", "beta", "sgm_uniform", "karras", "exponential"]
RESIZE_METHODS = ["crop", "pad", "stretch"]

# H3's six documented aspect ratios, plus 'custom' to use the width/height
# widgets directly. Each is fitted into the pixel budget at load time rather than
# hardcoded, so the table cannot drift from MAX_PIXELS.
ASPECT_RATIOS = {
    "16:9 landscape": (16, 9),
    "9:16 portrait": (9, 16),
    "1:1 square": (1, 1),
    "4:3 landscape": (4, 3),
    "3:4 portrait": (3, 4),
    "21:9 ultrawide": (21, 9),
}
ASPECT_OPTIONS = ["custom"] + list(ASPECT_RATIOS)


def resolution_for(aspect, width, height):
    """Canvas for an aspect-ratio choice, rounded down to /32 inside the budget."""
    if aspect == "custom" or aspect not in ASPECT_RATIOS:
        w = max(CANVAS_MULTIPLE, int(width) // CANVAS_MULTIPLE * CANVAS_MULTIPLE)
        h = max(CANVAS_MULTIPLE, int(height) // CANVAS_MULTIPLE * CANVAS_MULTIPLE)
        return w, h
    rw, rh = ASPECT_RATIOS[aspect]
    ratio = rw / rh
    w = math.sqrt(MAX_PIXELS * ratio)
    h = math.sqrt(MAX_PIXELS / ratio)
    return (max(CANVAS_MULTIPLE, int(w // CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, int(h // CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


# ── dynamic sockets (spec §4) ───────────────────────────────────────────────
#
# Declared as dot-namespaced optional inputs, which is the shape ComfyUI's own
# io.Autogrow expands to (`refs.ref_image_0`, ...) and therefore the shape the
# frontend already knows how to draw with the optional socket outline. The
# growing behaviour -- only ever one free socket showing -- is js/ps_sockets.js;
# the backend simply declares the whole range and ignores the empty ones.
#
# Numbering is 1-based here because these are the user's own sockets and appear
# on the node face. Ordering is by the numeric suffix, never by connection order.

MAX_SHOT_SOCKETS = 24
MAX_SLATE_REF_IMAGES = 8   # §4: refs.ref_image_1..8 on PulseSlate
MAX_SHOT_REF_IMAGES = 4    # per-shot scene-local references


def _numbered(kwargs, prefix, count):
    """Connected values from a dot-namespaced socket group, in suffix order.

    Returns `[(suffix, value), ...]`. Gaps are skipped rather than closed up here
    -- the caller decides what a gap means. For shots it means nothing at all
    (the user unplugged the middle one); for references it must never produce a
    hole in the socket dict, which is why they are re-indexed densely downstream.
    """
    found = []
    for i in range(1, count + 1):
        value = kwargs.get("%s%d" % (prefix, i))
        if value is not None:
            found.append((i, value))
    return found


# ── Pulse Shot ──────────────────────────────────────────────────────────────

class PulseShot:
    """One shot: its text, its length, and its own frames and references.

    Accepting IMAGE inputs is the point of this node (§2.2). It lets a shot's
    first frame come from an upstream generator in the same graph -- a PulseStill,
    a photo, another sampler -- instead of only from a file dragged onto the bin.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "start_image": ("IMAGE", {"tooltip":
                "First frame of this shot. With continuity 'keyframe_pairs' this is "
                "the frame the render is pinned to at time zero."}),
            "end_image": ("IMAGE", {"tooltip":
                "Last frame of this shot. H3 accepts a keyframe at frame 0 or at the "
                "final frame and nowhere else, so this pins the end of the window "
                "this shot lands in."}),
            "ref_audio": ("AUDIO", {"tooltip":
                "A voice or effect sample for this scene only. Not visible to any "
                "other shot -- put shared references in the Asset Bin instead."}),
        }
        for i in range(1, MAX_SHOT_REF_IMAGES + 1):
            optional["refs.ref_image_%d" % i] = ("IMAGE", {"tooltip":
                "Scene-local reference image %d. Numbered after the global "
                "references, and visible only to this shot." % i})
        return {
            "required": {
                # ── the frozen prefix — widget indices 0 and 1 ──────────────
                # Both hidden on the node face. They lead the widget list for the
                # same reason PulseSlate's do: LiteGraph serialises widgets
                # positionally, so a saved file must be able to say which layout
                # wrote it before anything reads it back.
                "schema_version": schema_widget(),
                "shot_id": ("STRING", {"default": "", "multiline": False, "tooltip":
                    "This shot's stable identity. Written once when the node is "
                    "created and never changed -- it is what keeps this shot's seed "
                    "and its cached segment attached to it when you insert a shot "
                    "above it. Do not edit."}),

                "label": ("STRING", {"default": "", "multiline": False, "tooltip":
                    "A name for this shot, shown in the render report. Not sent to "
                    "the model."}),
                "visual": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                      "default": "", "tooltip":
                    "What happens on screen. @Name references the Asset Bin or this "
                    "node's own reference sockets -- never type an ordinal."}),
                "audio_line": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                          "default": "", "tooltip":
                    "What is heard. Quoted \"text\" becomes dialogue."}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 15.08,
                                               "step": 0.01, "tooltip":
                    "How long this shot runs. The ceiling is H3's trained window "
                    "length; longer stories are built from more shots, not longer ones."}),
                "continuity": (list(SHOT_CONTINUITY_MODES), {"default": CONTINUITY_INHERIT,
                    "tooltip":
                    "How this shot joins the one before it. 'inherit' takes the "
                    "PulseSlate setting, which is what you want unless one particular "
                    "cut needs different handling."}),
                "ref_audio_mode": (list(AUDIO_ROLES), {"default": AUDIO_ROLE_LIP_SYNC,
                    "tooltip":
                    "What the ref_audio socket is for. 'lip_sync': the character's "
                    "mouth matches that recording, and the clip is trimmed to this "
                    "window's exact span so the two describe the same seconds. "
                    "'voice_timbre': the model speaks this shot's own dialogue and "
                    "only borrows the voice's character. Ignored when nothing is "
                    "connected to ref_audio."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": optional,
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("PULSE_SHOT",)
    RETURN_NAMES = ("shot",)
    FUNCTION = "execute"
    CATEGORY = "AddisPulse/H3"
    DESCRIPTION = ("One shot of a Pulse Slate timeline, with its own first/last frames "
                   "and its own scene-local references. Chain several into PulseSlate's "
                   "shot sockets; they replace the shot text box entirely.")

    def execute(self, schema_version, shot_id, label, visual, audio_line,
                duration_seconds, continuity, ref_audio_mode=AUDIO_ROLE_LIP_SYNC,
                start_image=None, end_image=None,
                ref_audio=None, unique_id=None, **kwargs):
        # A shot_id is normally written into the widget by the frontend when the
        # node is created. Deriving one here from the text covers the headless
        # case -- an API run, a workflow built by script -- where no frontend ever
        # touched the node. It is stable for identical text, which is the property
        # the seed and the cache actually need.
        resolved_id = (shot_id or "").strip() or text_shot_id(label, visual)

        refs = [tensor for _, tensor in
                _numbered(kwargs, "refs.ref_image_", MAX_SHOT_REF_IMAGES)]

        return ({
            "shot_id": resolved_id,
            "label": label or "",
            "visual": visual or "",
            "audio_line": audio_line or "",
            "duration_seconds": float(duration_seconds),
            "continuity": continuity or CONTINUITY_INHERIT,
            "start_image": start_image,
            "end_image": end_image,
            "ref_images": refs,
            "ref_audio": ref_audio,
            "ref_audio_mode": ref_audio_mode or AUDIO_ROLE_LIP_SYNC,
        },)


# ── Pulse Slate: the compiler ───────────────────────────────────────────────

class PulseSlate:
    """Two prompt boxes, an Asset Bin and a chain of shots -> a PULSE_TIMELINE."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "model_fl2va": ("MODEL", {"tooltip":
                "The First/Last-Frame checkpoint. Required by the 'last_frame_carry' "
                "and 'keyframe_pairs' continuity modes. Loading both checkpoints is "
                "~42GB."}),
            "ref_video": ("IMAGE", {"tooltip":
                "A reference video as frames, from an upstream loader."}),
            "ref_video_audio": ("AUDIO", {"tooltip":
                "Soundtrack of ref_video. Claims an <Audio N> ordinal ahead of every "
                "standalone audio reference."}),
            "ref_music": ("AUDIO", {"tooltip":
                "Non-diegetic score. Always the last audio ordinal."}),
        }
        # Declaration order must match GROUPS in js/ps_sockets.js: the frontend
        # rebuilds this tail in its own order, and a mismatch would put a saved
        # graph's wires on the wrong slots. References first, shots last.
        for i in range(1, MAX_SLATE_REF_IMAGES + 1):
            optional["refs.ref_image_%d" % i] = ("IMAGE", {"tooltip":
                "Global reference image %d, visible to every shot." % i})
        for i in range(1, MAX_SHOT_SOCKETS + 1):
            optional["shots.shot_%d" % i] = ("PULSE_SHOT", {"tooltip":
                "Shot %d. Connecting any shot socket makes the shot text box "
                "inactive." % i})

        return {
            "required": {
                "model": ("MODEL", {"tooltip":
                    "The ref2va (Reference) checkpoint. Read here only to detect the "
                    "upstream patch chain and warn about it; this node no longer "
                    "samples. Anchors and references are different checkpoints; wire "
                    "the fl2va one into model_fl2va."}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE", {"tooltip":
                    "Required on both branches -- H3 always builds a joint audio+video "
                    "latent, even when no reference audio is encoded."}),

                # ── the frozen prefix — widget indices 0 and 1 ──────────────
                "schema_version": schema_widget(),
                "timeline_data": ("STRING", {
                    "default": EMPTY_TIMELINE_DATA, "multiline": False,
                    "tooltip":
                    "The Asset Bin's storage: assets and cast, as JSON. Managed by "
                    "the panel and never written to by the execute path."}),

                # ── the two prompt boxes ────────────────────────────────────
                "global_prompt": ("STRING", {
                    "multiline": True, "dynamicPrompts": False,
                    "default": "style: \nidentity: \nsoundscape: \nmusic: ",
                    "tooltip":
                    "Art style, lighting, camera rules, character identity locks and score. "
                    "Compiles into subject_definitions and retention_analysis.\n\n"
                    "Optional labels at the start of a line: style: / identity: / retention: "
                    "/ soundscape: / music:. Unlabelled text is treated as style.\n\n"
                    "Reference assets by name (@Mimi) or id ({{mimi}}) -- never by number. "
                    "Ordinals are assigned from bin order at compile time."}),
                "shot_prompt": ("STRING", {
                    "multiline": True, "dynamicPrompts": False,
                    "default": "[Shot 1] \n[Shot 2] At 00:05.000, ",
                    "tooltip":
                    "Timecoded shots, one per line. Begin each with [Shot N] or a "
                    "[MM:SS.mmm] timecode.\n\n"
                    "IGNORED whenever any PulseShot node is connected -- the two are "
                    "never merged. See spec §5.\n\n"
                    "Quoted \"text\" becomes dialogue. @Name references the Asset Bin."}),

                # ── exposed control panel ───────────────────────────────────
                "duration_seconds": ("FLOAT", {
                    "default": 10.0, "min": 0.2, "max": 600.0, "step": 0.5,
                    "tooltip": "Total length of the finished video. Snapped up to the "
                               "17k+5 frame grid, and split into windows if longer than "
                               "window_seconds. Ignored when PulseShot nodes are "
                               "connected -- their durations define the length."}),
                "aspect_ratio": (ASPECT_OPTIONS, {"default": "16:9 landscape", "tooltip":
                    "Preset canvases fitted into H3's 1,032,192px budget. Choose 'custom' "
                    "to use the width and height widgets instead."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32,
                                  "tooltip": "Used only when aspect_ratio is 'custom'."}),
                "height": ("INT", {"default": 736, "min": 32, "max": 4096, "step": 32,
                                   "tooltip": "Used only when aspect_ratio is 'custom'."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "sampler_name": (SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (SCHEDULERS, {"default": "simple"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip":
                    "1.0 uses BasicGuider, which is H3's own native path -- its reference "
                    "pipeline has no negative conditioning anywhere. Above 1.0 switches to "
                    "CFGGuider with an empty negative prompt. Leave at 1.0 unless you are "
                    "deliberately experimenting."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True, "tooltip":
                    "The base seed. Each window's actual seed is derived from this and "
                    "from the set of shots in that window, so inserting a shot does not "
                    "reroll the windows that did not change."}),
                "partition_strategy": (["balanced", "fill"], {"default": "balanced",
                    "tooltip":
                    "How a long timeline is split. 'balanced' spreads it into near-equal "
                    "windows, avoiding a short trailing window below H3's 124-frame trained "
                    "floor. 'fill' packs full windows and merges any short tail backwards."}),
                "window_seconds": ("FLOAT", {"default": 15.0, "min": 5.2, "max": 15.1,
                                             "step": 0.1, "tooltip":
                    "Length of each individual H3 call. The trained ceiling is ~15.08s "
                    "(362 frames) and the floor is ~5.17s (124 frames)."}),
                "resize_method": (RESIZE_METHODS, {"default": "crop", "tooltip":
                    "How a reference image whose aspect does not match the canvas is fitted. "
                    "'pad' letterboxes, which makes the bars themselves reference content."}),
                "carry_mode": (["image", "video", "both", "none"], {"default": "image",
                    "tooltip":
                    "What the previous window contributes to the next as a REFERENCE on "
                    "the ref2va branch. Distinct from `continuity`, which chooses whether "
                    "a frame is pinned as a keyframe."}),
                "carry_audio": ("BOOLEAN", {"default": True, "tooltip":
                    "Feed the previous window's audio tail forward. Without it each window "
                    "invents its own score and the seam is audible."}),
                "carry_audio_seconds": ("FLOAT", {"default": DEFAULT_AUDIO_CARRY_SECONDS,
                                                  "min": 0.5, "max": 15.0, "step": 0.5}),
                "ref_image_size": (list(REF_IMAGE_SIZE_OPTIONS), {"default": "match",
                    "tooltip":
                    "'match' scales references to the render's pixel area (faster). 'max' "
                    "uses a 2048px short edge for stronger identity, but reference tokens "
                    "ride every sampling step, so it is several times slower."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0,
                                          "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0,
                                          "step": 0.01}),
                "audio_ref_ceiling": ("INT", {
                    "default": MAX_REF_AUDIOS, "min": MAX_REF_AUDIOS,
                    "max": MAX_REF_AUDIOS_CEILING, "step": 1, "tooltip":
                    "How many standalone audio references the bin may hold. 3 is what "
                    "MiniMax documents and what ComfyUI's ref_audios socket declares. "
                    "Above 3 goes past both. Raise it to test, not to set and forget."}),

                # ── appended in 3.0.0 ───────────────────────────────────────
                "continuity": (list(CONTINUITY_MODES), {"default": CONTINUITY_NONE,
                    "tooltip":
                    "How windows join. 'none' samples each independently. "
                    "'last_frame_carry' pins the previous window's final frame as the "
                    "next one's first frame. 'keyframe_pairs' pins each shot's start "
                    "and end frames. The last two need model_fl2va and fail at compile "
                    "time without it -- they do not fall back silently."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": optional,
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("PULSE_TIMELINE", "CONDITIONING", "LATENT", "AUDIO", "IMAGE", "STRING")
    RETURN_NAMES = ("timeline", "positive", "latent", "combined_audio", "images",
                    "compiled_prompt")
    FUNCTION = "execute"
    CATEGORY = "AddisPulse/H3"
    DESCRIPTION = ("Compiles prompts, an Asset Bin and a chain of PulseShot nodes into a "
                   "MiniMax H3 timeline. Does not render: a single-window timeline hands "
                   "back positive/latent for your own sampler, and a longer one goes to "
                   "PulseRender.")

    def execute(self, model, clip, vae, audio_vae, schema_version, timeline_data,
                global_prompt, shot_prompt,
                duration_seconds, aspect_ratio, width, height, steps, sampler_name,
                scheduler, cfg, seed, partition_strategy, window_seconds, resize_method,
                carry_mode, carry_audio, carry_audio_seconds, ref_image_size,
                shift_video, shift_audio, audio_ref_ceiling=MAX_REF_AUDIOS,
                continuity=CONTINUITY_NONE, model_fl2va=None, ref_video=None,
                ref_video_audio=None, ref_music=None, unique_id=None, **kwargs):

        width, height = resolution_for(aspect_ratio, width, height)
        side = SideChannel(clip=clip, vae=vae, audio_vae=audio_vae,
                           shift_video=shift_video, shift_audio=shift_audio,
                           ref_image_size=ref_image_size, resize_method=resize_method,
                           carry_audio_seconds=carry_audio_seconds)

        # Read widgets, build in memory. Nothing below assigns to a widget --
        # see tests/test_widget_state.py::TestQueueDoesNotEraseTypedText.
        timeline, notes = build_timeline(
            timeline_data, global_prompt=global_prompt, shot_prompt=shot_prompt,
            duration_seconds=duration_seconds, window_seconds=window_seconds,
            audio_ref_ceiling=audio_ref_ceiling)

        if timeline.limits.beyond_spec:
            notes.append(
                "audio_ref_ceiling is %d. MiniMax documents 3 standalone audio "
                "references and ComfyUI's ref_audios socket declares 3; this render "
                "goes past both. It will run, but nothing about the result is covered "
                "by anything published, and every extra reference rides all %d "
                "sampling steps." % (timeline.limits.audios, steps))

        # ── global socket references (§4, §10.1) ────────────────────────────
        notes.extend(_attach_global_refs(timeline, side, kwargs,
                                         ref_video, ref_video_audio, ref_music))

        # Drop unloadable bin references BEFORE tags are assigned, so ordinals
        # stay dense by construction rather than developing a hole at render time.
        notes.extend(drop_missing(timeline, lambda f: bool(media.resolve_path(f))))

        # ── shots: sockets win outright over the text box (§5) ──────────────
        shot_payloads = [payload for _, payload in
                         _numbered(kwargs, "shots.shot_", MAX_SHOT_SOCKETS)]
        if shot_payloads:
            notes.insert(0, SHOTS_WIN % len(shot_payloads))
            notes.extend(_apply_shot_nodes(timeline, side, shot_payloads, continuity))

        shot_blocks = _shot_blocks(timeline, shot_payloads, continuity)

        # ── continuity (§11) ────────────────────────────────────────────────
        problems = check_continuity(shot_blocks, continuity, model_fl2va is not None)
        if problems:
            message = ("This timeline cannot be compiled:\n  - "
                       + "\n  - ".join(problems))
            log.error("[PulseStudio] %s", message)
            _warn_on_node(unique_id, problems)
            blocked = ExecutionBlocker(message)
            return (blocked, blocked, blocked, blocked, blocked, message)

        _apply_continuity_branch(timeline, continuity)

        # ── compile ─────────────────────────────────────────────────────────
        carry = CarryPolicy(mode=carry_mode, audio=carry_audio,
                            audio_seconds=carry_audio_seconds)
        plan = compile_timeline(timeline, policy=partition_strategy, carry=carry)

        for note in notes + plan.diagnostics:
            log.info("[PulseStudio] %s", note)
        for window in plan.windows:
            for note in window.diagnostics:
                log.warning("[PulseStudio] window %d: %s", window.index + 1, note)

        if not plan.ok:
            message = "Timeline cannot be compiled:\n  - " + "\n  - ".join(plan.problems)
            log.error("[PulseStudio] %s", message)
            blocked = ExecutionBlocker(message)
            return (blocked, blocked, blocked, blocked, blocked, message)

        # ── the document (§3) ───────────────────────────────────────────────
        document = _build_document(
            timeline, plan, shot_blocks, side, global_prompt=global_prompt, width=width,
            height=height, steps=steps, sampler_name=sampler_name, scheduler=scheduler,
            cfg=cfg, seed=seed, continuity=continuity, notes=notes)

        side.plan = plan
        side.timeline = timeline

        # §12.6: the one patch setting whose wrong value damages precisely what
        # this pack is for. Checked here, at compile time, because it is a
        # property of the timeline being compiled and not of the render.
        descriptor = describe_model_patches(
            model, sage_attention_global=_sage_attention_global())
        audio_warnings = patch_warnings(
            descriptor, paired_audio_count=_paired_audio_count(timeline))
        for note in audio_warnings:
            log.warning("[PulseStudio] %s", note)
        document["warnings"].extend(audio_warnings)

        _report_patches(model, unique_id,
                        branches_used={w.branch for w in plan.windows},
                        fl2va_connected=model_fl2va is not None)
        _warn_on_node(unique_id, audio_warnings)

        preview = _compiled_prompt(document, plan, descriptor, notes)
        log.info("[PulseStudio] compiled %d window(s): %s (%.2fs total, %dx%d)",
                 len(plan.windows), ", ".join(str(w.frame_count) for w in plan.windows),
                 plan.total_seconds, width, height)

        # ── the short path: hand back conditioning for the graph's own sampler ─
        if len(plan.windows) == 1:
            try:
                positive, latent = render.condition_window(
                    plan.windows[0], side, width, height, None, None, None)
            except SocketGapError as exc:
                message = ("Reference sockets would have been misnumbered, so the "
                           "compile was stopped:\n  %s" % (exc,))
                log.error("[PulseStudio] %s", message)
                blocked = ExecutionBlocker(message)
                return (blocked, blocked, blocked, blocked, blocked, message)
            return ((document, side), positive, latent, media.empty_audio(),
                    media.empty_images(width, height), preview)

        # ── the long path: PulseRender does the work ────────────────────────
        #
        # positive/latent are NOT handed back here, and the blocker says why.
        # They would be the last window's -- and a graph still wired to a sampler
        # would re-sample that one window and save it as if it were the whole
        # film. Which is exactly what happened in v2: a 15s timeline that split
        # into 192 + 175 frames produced a 7-second file, with no error anywhere,
        # because 175 frames is a perfectly valid latent.
        blocked = ExecutionBlocker(
            "This timeline compiles to %d windows, so there is no single latent to "
            "hand back -- `positive` and `latent` would be the last window alone "
            "(%d of %d frames). Connect the `timeline` output to a Pulse Render "
            "node, which renders every window and stitches them."
            % (len(plan.windows), plan.windows[-1].frame_count,
               sum(w.frame_count for w in plan.windows)))
        return ((document, side), blocked, blocked, media.empty_audio(),
                media.empty_images(width, height), preview)


# ── PulseSlate helpers ──────────────────────────────────────────────────────

def _attach_global_refs(timeline, side, kwargs, ref_video, ref_video_audio, ref_music):
    """Add socket-borne global references to the bin. Spec §4, §10.1.

    They join the bin *after* whatever the Asset Bin panel holds, so a project
    that has both keeps its dragged-in assets at the low ordinals where its prompt
    text already expects them.
    """
    notes = []

    def _add(kind, slot, tensor, name, description=""):
        # Never reuse a name the bin already holds. The bin panel numbers its own
        # drops Image1, Image2..., so a socket reference auto-named `Image1`
        # alongside a dropped `Image1` makes `@Image1` ambiguous -- and an
        # ambiguous name resolves to nothing, silently.
        name = timeline.assets.unique_name(name, kind)
        asset = Asset(socket_asset_id(slot), kind, name=name, file="",
                      description=description)
        ok, reason = timeline.assets.can_add(asset, limits=timeline.limits)
        if not ok:
            notes.append("socket reference %r did not fit the budget and was "
                         "dropped: %s" % (name, reason))
            return
        timeline.assets.add(asset, limits=timeline.limits)
        side.put(slot, tensor, digest=_digest_of(kind, tensor))

    for i, tensor in _numbered(kwargs, "refs.ref_image_", MAX_SLATE_REF_IMAGES):
        _add(KIND_IMAGE, "slate.ref_image_%d" % i, tensor, "Image%d" % i)
    if ref_video is not None:
        _add(KIND_VIDEO, "slate.ref_video", ref_video, "RefVideo")
        if ref_video_audio is not None:
            # A soundtrack is a property of its video, not a separate file, and it
            # claims an <Audio N> ordinal ahead of every standalone audio.
            asset = timeline.assets.get(socket_asset_id("slate.ref_video"))
            if asset is not None:
                asset.include_audio = True
                side.put("slate.ref_video_audio", ref_video_audio,
                         digest=media.audio_digest(ref_video_audio))
    elif ref_video_audio is not None:
        notes.append("ref_video_audio is connected but ref_video is not; a soundtrack "
                     "is index-paired to its own video, so it was ignored.")
    if ref_music is not None:
        # §10.1: always the last audio ordinal. It is added last, and the tag map
        # numbers standalone audio in bin order, so "last" is structural here
        # rather than a rule someone has to remember.
        _add(KIND_AUDIO, "slate.ref_music", ref_music, "Music",
             description="the non-diegetic score for the whole film")
    return notes


def _apply_shot_nodes(timeline, side, payloads, project_continuity):
    """Replace the timeline's shots with the connected PulseShot chain. Spec §5.

    Never merged with the text box, never silently preferred: the caller has
    already put a visible line at the top of `compiled_prompt` saying the box was
    ignored and how many nodes replaced it.
    """
    notes = []
    shots = []
    cursor = 0.0

    for index, payload in enumerate(payloads):
        shot_id = payload["shot_id"]
        text = payload["visual"].strip()
        if payload["audio_line"].strip():
            # One prose line per shot is what H3's format takes; the audio line is
            # direction like any other, and its quoted spans become <d> dialogue
            # in the compiler exactly as they would from the text box.
            text = (text + " " + payload["audio_line"].strip()).strip()
        duration = max(0.01, float(payload["duration_seconds"]))
        shots.append(Shot(shot_id, start=cursor, duration=duration, prompt=text))
        cursor += duration

        # Scene-local references get short, fixed handles -- @Ref1 and @Voice --
        # rather than names derived from the shot's label. They are scoped to this
        # shot, so every shot may use the same handles without colliding, and a
        # handle you can type is the difference between the feature being used and
        # being ignored. See AssetBin.find_by_name on why the scope has to be
        # applied inside the lookup for that to hold.
        local = []
        for j, tensor in enumerate(payload.get("ref_images") or [], start=1):
            slot = "shot.%s.ref_image_%d" % (shot_id, j)
            local.append(Asset(socket_asset_id(slot), KIND_IMAGE,
                               name="Ref%d" % j, file=""))
            side.put(slot, tensor, digest=media.tensor_digest(tensor))
        if payload.get("ref_audio") is not None:
            slot = "shot.%s.ref_audio" % shot_id
            local.append(Asset(socket_asset_id(slot), KIND_AUDIO,
                               name="Voice", file="",
                               audio_role=payload.get("ref_audio_mode")
                               or AUDIO_ROLE_LIP_SYNC))
            side.put(slot, payload["ref_audio"],
                     digest=media.audio_digest(payload["ref_audio"]))
        if local:
            timeline.local_refs[shot_id] = local

        anchors = {}
        if payload.get("start_image") is not None:
            slot = "shot.%s.start_image" % shot_id
            side.put(slot, payload["start_image"],
                     digest=media.tensor_digest(payload["start_image"]))
            anchors["first"] = socket_asset_id(slot)
        if payload.get("end_image") is not None:
            slot = "shot.%s.end_image" % shot_id
            side.put(slot, payload["end_image"],
                     digest=media.tensor_digest(payload["end_image"]))
            anchors["last"] = socket_asset_id(slot)
        if anchors:
            timeline.shot_anchors[shot_id] = anchors

    # keyframe_pairs pairs each shot's start frame with the *next* shot's start
    # frame (§11). A shot that set its own end_image keeps it; one that did not
    # borrows the next shot's opening frame, which is what makes the cut land on
    # a frame the following window is also pinned to.
    for index, payload in enumerate(payloads[:-1]):
        effective = resolve_continuity(project_continuity, payload["continuity"])
        if effective != CONTINUITY_KEYFRAME_PAIRS:
            continue
        shot_id = payload["shot_id"]
        anchors = timeline.shot_anchors.setdefault(shot_id, {})
        if "last" not in anchors:
            following = timeline.shot_anchors.get(payloads[index + 1]["shot_id"], {})
            if following.get("first"):
                anchors["last"] = following["first"]

    timeline.shots = shots
    timeline.duration_seconds = cursor
    if not shots:
        notes.append("every connected PulseShot is empty; there is nothing to render.")
    return notes


def _apply_continuity_branch(timeline, continuity):
    """Map §11's continuity modes onto the branch machinery that implements them.

    `none` leaves continuation windows on ref2va, where identity references stay
    alive across the whole film. The other two need a pinned frame, and pinning a
    frame *is* the fl2va branch -- ComfyUI exposes it as a different checkpoint
    with disjoint inputs, not as a setting. `check_continuity` has already
    refused the case where model_fl2va is absent, so this cannot silently
    downgrade.
    """
    if continuity == CONTINUITY_LAST_FRAME:
        timeline.continuation_branch = BRANCH_FL2VA
    elif continuity == CONTINUITY_KEYFRAME_PAIRS:
        timeline.branch = BRANCH_FL2VA
        timeline.continuation_branch = BRANCH_FL2VA
    else:
        timeline.continuation_branch = timeline.branch


def _shot_blocks(timeline, payloads, continuity):
    """The document's `shots` list, before the compiler fills in resolved text."""
    by_id = {p["shot_id"]: p for p in payloads}
    blocks = []
    for index, shot in enumerate(timeline.ordered_shots()):
        payload = by_id.get(shot.shot_id, {})
        blocks.append(shot_block(
            shot.shot_id, index,
            label=payload.get("label") or "",
            visual=payload.get("visual") or shot.prompt,
            audio_line=payload.get("audio_line") or "",
            duration_seconds=shot.duration,
            continuity=payload.get("continuity") or CONTINUITY_INHERIT,
            start_image_ref=(timeline.shot_anchors.get(shot.shot_id, {}) or {}).get("first"),
            end_image_ref=(timeline.shot_anchors.get(shot.shot_id, {}) or {}).get("last"),
        ))
    return blocks


def _paired_audio_count(timeline):
    """How many standalone audio references this timeline carries (§12.6)."""
    return len(timeline.assets.by_kind(KIND_AUDIO))


def _build_document(timeline, plan, shot_blocks, side, global_prompt, width, height, steps,
                    sampler_name, scheduler, cfg, seed, continuity, notes):
    """Assemble the PULSE_TIMELINE. Spec §3."""
    tag_map = timeline.assets.tag_map()

    global_refs = []
    for kind, asset_id, ordinal in tag_map.order:
        # A video's paired soundtrack is keyed `<id>#soundtrack` in the tag map;
        # it is a real ordinal belonging to a real asset, so it is described here
        # under its own number and resolved back to the file it rides inside.
        base_id = asset_id.split("#")[0]
        asset = timeline.assets.get(base_id)
        if asset is None:
            continue
        global_refs.append(ref_descriptor(
            ordinal, kind, asset.name,
            source="socket" if socket_slot_of(base_id) else "bin",
            file=asset.file or None,
            sha256=_digest_for(asset, side)))

    # Per-shot resolved text and unresolved aliases, from whichever window
    # compiled each shot. A shot spanning a window boundary is compiled into
    # both; the first is taken, since both resolve against the same scope.
    resolved, unresolved = {}, {}
    for window in plan.windows:
        for shot_id, text in window.resolved_shots.items():
            resolved.setdefault(shot_id, text)
        for shot_id, aliases in window.unresolved_shots.items():
            unresolved.setdefault(shot_id, aliases)

    local_by_shot = {}
    for shot_id, assets in (timeline.local_refs or {}).items():
        counters = {}
        entries = []
        for asset in assets:
            base = len([r for r in global_refs if r["kind"] == asset.kind])
            counters[asset.kind] = counters.get(asset.kind, 0) + 1
            entries.append(ref_descriptor(
                base + counters[asset.kind], asset.kind, asset.name,
                source="socket", sha256=_digest_for(asset, side),
                audio_role=asset.audio_role))
        local_by_shot[shot_id] = entries

    for block in shot_blocks:
        block["resolved_prompt"] = resolved.get(block["shot_id"], "")
        block["unresolved_aliases"] = unresolved.get(block["shot_id"], [])
        block["local_refs"] = local_by_shot.get(block["shot_id"], [])

    windows = []
    for window in plan.windows:
        windows.append(window_block(
            window.index, window.shot_ids, window.frame_count, fps=timeline.fps,
            width=width, height=height,
            seed=window_seed(seed, window.shot_ids),
            steps=steps, sampler=sampler_name, scheduler=scheduler, cfg=cfg,
            continuity_in=(CONTINUITY_NONE if window.index == 0 else continuity),
            continuity_out=(CONTINUITY_NONE if window.index == window.total - 1
                            else continuity),
            branch=window.branch, prompt=window.prompt))

    # Reported as the PEAK across windows, not as the bin's own occupancy. The
    # limits bind per H3 call, and a window carries more than the bin does: the
    # scene-local references of every shot in it, plus the carry-over frame and
    # audio tail a continuation window prepends. Reporting the bin alone said
    # "3/9 images" for a render whose second window actually carried six.
    peak = {"images": 0, "videos": 0, "audio": 0, "total": 0}
    for window in plan.windows:
        groups = window.socket_kwargs()
        peak["images"] = max(peak["images"], len(groups["ref_images"]))
        peak["videos"] = max(peak["videos"], len(groups["ref_videos"]))
        peak["audio"] = max(peak["audio"], len(groups["ref_audios"]))
        peak["total"] = max(peak["total"],
                            sum(1 for f in window.files if not f.synthetic))
    budget = {
        "images": peak["images"], "max_images": MAX_REF_IMAGES,
        "videos": peak["videos"], "max_videos": MAX_REF_VIDEOS,
        "audio": peak["audio"], "max_audio": timeline.limits.audios,
        "total": peak["total"], "max_total": max(MAX_REF_FILES_TOTAL, timeline.limits.files),
    }

    fields = global_block(
        style=timeline.style_line, identity=timeline.identity_notes,
        retention=timeline.retention_notes, soundscape=timeline.overall_soundscape,
        music=timeline.non_diegetic_music, raw=global_prompt or "")

    warnings = list(notes) + list(plan.diagnostics)
    for window in plan.windows:
        warnings.extend("window %d: %s" % (window.index, d) for d in window.diagnostics)

    return build_document(fields, global_refs, shot_blocks, windows, budget, warnings)


def _digest_of(kind, tensor):
    """Content digest for a tensor arriving on a socket."""
    return media.audio_digest(tensor) if kind == KIND_AUDIO else media.tensor_digest(tensor)


def _digest_for(asset, side):
    """Content digest for a reference. Spec §7.1.

    Socket references were hashed where the tensor was, at the moment they were
    attached, and the digest is on the side channel; bin references are hashed
    from their file here. An unreadable file yields an empty digest rather than an
    exception -- it was already dropped by `drop_missing`, so reaching this with
    one is a race, not a state worth failing the whole compile over.
    """
    slot = socket_slot_of(asset.asset_id)
    if slot is not None:
        return side.digest(slot)
    return media.file_digest(asset.file) if asset.file else ""


#: §5. Prepended verbatim to `compiled_prompt` whenever shot sockets win, because
#: a user who has just typed into the shot box and seen it have no effect needs
#: the reason on the first line, not in a warnings section further down.
SHOTS_WIN = "Shot text box ignored: %d PulseShot node(s) are connected."


def _compiled_prompt(document, plan, descriptor, notes):
    """The `compiled_prompt` output: the §9 report, then the exact prompt text.

    The report first, because it is what catches a wrong binding before any GPU
    time is spent, and the prompt after, because reading the literal string is
    still the last check before a long render.
    """
    banner = [n for n in notes if n.startswith("Shot text box ignored")]

    head = build_report(document, patch_descriptor=descriptor,
                        patch_fingerprint=patch_fingerprint(descriptor),
                        title="Pulse Slate (compile only)")
    if banner:
        head = "!! " + banner[0] + "\n   The two are never merged. Clear the box, or " \
               "disconnect the shot nodes.\n\n" + head
    body = "\n\n".join(
        "=== Window %d/%d | %s | %d frames (%.2fs) ===\n%s"
        % (w.index + 1, w.total, w.branch, w.frame_count, w.duration_seconds, w.prompt)
        for w in plan.windows)
    if len(plan.windows) > 1:
        head += ("\n\nThis timeline needs a Pulse Render node. Connect the `timeline` "
                 "output to one; `positive` and `latent` are blocked on this path "
                 "because they would be the last window alone.")
    return head + "\n\n" + body


# ── Pulse Render: the executor ──────────────────────────────────────────────

class PulseRender:
    """Walk a PULSE_TIMELINE: reuse every window already on disk, render the rest."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("PULSE_TIMELINE", {"tooltip":
                    "From a Pulse Slate node. Carries the compiled plan, the window "
                    "seeds and the reference digests the segment cache keys on."}),
                "model": ("MODEL", {"tooltip":
                    "The ref2va checkpoint, with every attention and memory patch "
                    "already applied upstream. This node samples with the model it is "
                    "handed, so a patch applied downstream would do nothing."}),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),

                "schema_version": schema_widget(),

                "cache_mode": (list(CACHE_MODES), {"default": "auto", "tooltip":
                    "'auto' reuses any window whose content, seed and patch chain are "
                    "unchanged. 'force_rerender' ignores the cache. 'reuse_only' "
                    "refuses to render anything and aborts if a window is missing -- "
                    "for assembling a final cut without regenerating a frame."}),
                "run_dir": ("STRING", {"default": "pulseslate", "multiline": False,
                                       "tooltip":
                    "Folder under ComfyUI/output that holds run folders."}),
                "run_id": ("STRING", {"default": "", "multiline": False, "tooltip":
                    "Which run folder to resume into. Empty derives it from the "
                    "timeline, so the same project reopens into the same folder across "
                    "sessions -- and a seed change still reuses that folder."}),
                "save_segments": ("BOOLEAN", {"default": True, "tooltip":
                    "Write each window to disk as it finishes. This is what makes a "
                    "killed render resumable. Turning it off gives up the cache."}),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip":
                    "Accumulate assembled frames as 8-bit and release VRAM between "
                    "windows. The finished video is assembled by joining the segment "
                    "files, so with this on a twelve-window film never exists in RAM."}),
                "dry_run": ("BOOLEAN", {"default": False, "tooltip":
                    "Produce the report and nothing else: no sampling, no decode, no "
                    "file writes. A wrong reference binding renders successfully and "
                    "gives you the wrong film -- this is how you catch it first."}),
                "prune_unused": ("BOOLEAN", {"default": False, "tooltip":
                    "Delete segments in this run folder that the current timeline no "
                    "longer references. Off by default: yesterday's segments are what "
                    "make flipping back to yesterday's edit free."}),
                "use_reference_audio": ("BOOLEAN", {"default": False, "tooltip":
                    "Put the lip_sync reference recordings into the finished film "
                    "instead of the audio H3 generated. H3 always synthesises its own "
                    "track, and on a lip-sync shot that track is a re-synthesis of your "
                    "recording -- close, but not your take. Off by default because it "
                    "silences everything the model scored around the voice; the "
                    "generated audio is still written to each segment's .flac either "
                    "way, so this is reversible without re-rendering."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": {
                "model_fl2va": ("MODEL", {"tooltip":
                    "The First/Last-Frame checkpoint. Needed by the continuity modes "
                    "that pin a frame."}),
            },
            "hidden": HIDDEN_INPUTS_WITH_PROMPT,
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "frames", "audio", "segment_paths", "report")
    FUNCTION = "execute"
    CATEGORY = "AddisPulse/H3"
    DESCRIPTION = ("Renders a Pulse Slate timeline window by window, writing each segment "
                   "to disk as it completes. Requeue after a crash and only the windows "
                   "that are missing render; edit one shot and only its window re-renders.")

    def execute(self, timeline, model, vae, audio_vae, schema_version, cache_mode,
                run_dir, run_id, save_segments, low_memory, dry_run, prune_unused,
                use_reference_audio=False,
                model_fl2va=None, unique_id=None, prompt=None):
        document, side = _unwrap_timeline(timeline)
        if document is None:
            message = ("The `timeline` input did not carry a compiled plan. Connect it "
                       "to a Pulse Slate node's `timeline` output.")
            blocked = ExecutionBlocker(message)
            return (blocked, blocked, blocked, message, message)

        options = render.RenderOptions(
            cache_mode=cache_mode, run_dir=run_dir, run_id=run_id,
            save_segments=save_segments, low_memory=low_memory, dry_run=dry_run,
            prune_unused=prune_unused, use_reference_audio=use_reference_audio,
            # §8: never assemble a frame stack nobody asked for.
            want_frames=render.output_is_connected(prompt, unique_id, 1))

        try:
            result = render.run(document, side, model, vae, audio_vae,
                                model_fl2va=model_fl2va, options=options)
        except ReuseOnlyMiss as exc:
            message = "Render refused: %s" % (exc,)
            log.error("[PulseStudio] %s", message)
            _warn_on_node(unique_id, [str(exc)])
            blocked = ExecutionBlocker(message)
            return (blocked, blocked, blocked, message, message)
        except SocketGapError as exc:
            message = ("Reference sockets would have been misnumbered, so the render "
                       "was stopped:\n  %s" % (exc,))
            log.error("[PulseStudio] %s", message)
            blocked = ExecutionBlocker(message)
            return (blocked, blocked, blocked, message, message)

        _warn_on_node(unique_id, result.warnings)

        if dry_run:
            # §9: video/frames/audio return nothing on a dry run. Blocked rather
            # than None so a wired downstream node stops with a message instead of
            # failing on a null it did not expect.
            blocked = ExecutionBlocker(
                "dry_run is on: nothing was sampled, decoded or written. Turn it off "
                "to render.")
            return (blocked, blocked, blocked, "", result.report)

        frames = result.frames
        if frames is None:
            frames = media.empty_images()
        audio = result.audio if result.audio is not None else media.empty_audio()
        video = result.video
        if video is None:
            video = ExecutionBlocker(
                "No video was assembled.\n\n"
                "If save_segments is off there are no segment files to join -- take "
                "the result from `frames` and `audio` instead, or turn it on.\n\n"
                "Otherwise the windows rendered but joining them failed. They are "
                "safe: `segment_paths` lists every one, and requeueing reuses them "
                "from the cache rather than re-rendering. See the report for the "
                "reason.")
        return (video, frames, audio, result.segment_paths, result.report)


def _unwrap_timeline(value):
    """Accept either the (document, side_channel) tuple or a bare document."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], dict):
        return value[0], value[1]
    if isinstance(value, dict):
        return value, SideChannel()
    return None, None


# ── Pulse Bench ─────────────────────────────────────────────────────────────

class PulseBench:
    """Which patch chain is actually faster on this box. Spec §12.7."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schema_version": schema_widget(),
                "run_dirs": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "One run folder per line -- the folders holding manifest.json, "
                    "printed by PulseRender's `segment_paths`. Absolute paths, or "
                    "paths relative to ComfyUI/output."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("table",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "AddisPulse/H3"
    DESCRIPTION = ("Reads PulseRender manifests and prints seconds-per-frame and peak "
                   "VRAM grouped by patch chain. Sol-Attn and Spectrum address different "
                   "memory and the community disagrees about their speed; this turns the "
                   "question into a lookup instead of an argument.")

    def execute(self, schema_version, run_dirs, unique_id=None):
        paths = []
        for line in (run_dirs or "").splitlines():
            line = line.strip()
            if not line:
                continue
            paths.append(line if os.path.isabs(line)
                         else os.path.join(folder_paths.get_output_directory(), line))
        manifests, problems = load_manifests(paths)
        return (format_table(group_by_fingerprint(manifests), problems),)


# ── Retake Scissor ──────────────────────────────────────────────────────────

class PulseRetake:
    """Replace a bad span of a rendered clip, anchored to the frames either side."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_fl2va": ("MODEL", {"tooltip":
                    "The First/Last-Frame checkpoint. Patching pins both surrounding "
                    "frames, which is the fl2va branch specifically."}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "images": ("IMAGE", {"tooltip": "The rendered clip to patch."}),

                # Widget index 0, under the same contract as PulseSlate (§3).
                "schema_version": schema_widget(),

                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                      "default": "",
                                      "tooltip": "What should happen in the patched span. "
                                                 "Defaults to describing the shots that "
                                                 "overlapped the cut."}),
                "cut_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0,
                                                "step": 0.01}),
                "cut_end_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3600.0,
                                              "step": 0.01}),
                "keep_base_audio": ("BOOLEAN", {"default": True, "tooltip":
                    "Keep the original clip's audio. A re-rendered patch invents its own "
                    "score and will not match the surrounding track."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "sampler_name": (SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (SCHEDULERS, {"default": "simple"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0,
                                          "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0,
                                          "step": 0.01}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": {
                "base_audio": ("AUDIO", {"tooltip": "The clip's original audio, returned "
                                                    "as-is when keep_base_audio is on."}),
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "plan")
    FUNCTION = "execute"
    CATEGORY = "AddisPulse/H3"
    DESCRIPTION = ("Cuts a bad span out of a rendered clip and re-renders only the gap, "
                   "pinned to the exact frames either side. Patch length snaps to the "
                   "17k+5 grid and the cut moves with it, so the stitched result is the "
                   "same length as the base.")

    def execute(self, model_fl2va, clip, vae, audio_vae, images, schema_version, prompt,
                cut_start_seconds, cut_end_seconds, keep_base_audio, fps, seed, steps,
                sampler_name, scheduler, cfg, shift_video, shift_audio, base_audio=None,
                unique_id=None):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3SigmaShift

        try:
            plan = plan_retake(images.shape[0], cut_start_seconds=cut_start_seconds,
                               cut_end_seconds=cut_end_seconds,
                               keep_base_audio=keep_base_audio, fps=fps)
        except RetakeError as exc:
            message = "Retake refused: %s" % (exc,)
            log.error("[PulseStudio] %s", message)
            blocked = ExecutionBlocker(message)
            return (blocked, blocked, message)

        for note in plan.diagnostics:
            log.warning("[PulseStudio] retake: %s", note)
        log.info("[PulseStudio] %s", plan.describe())

        # This node always samples internally, so an unpatched model is felt here
        # too. Only one checkpoint is ever wired in, so no single-checkpoint check.
        _report_patches(model_fl2va, unique_id)

        height, width = images.shape[1], images.shape[2]
        anchors = plan.anchors()
        first = last = None
        if "first_frame" in anchors:
            i = anchors["first_frame"]["base_index"]
            first = images[i:i + 1]
        if "last_frame" in anchors:
            i = anchors["last_frame"]["base_index"]
            last = images[i:i + 1]

        shifted = render.unpack(render.execute_node(
            MiniMaxH3SigmaShift, model=model_fl2va,
            shift_video=shift_video, shift_audio=shift_audio))[0]
        positive, latent = render.unpack(render.execute_node(
            MiniMaxH3ImageToVideo, clip=clip, vae=vae, prompt=prompt, width=width,
            height=height, length=plan.patch_frames, first_frame=first, last_frame=last))[:2]

        _, patch_images, patch_audio = render.sample(
            shifted, positive, latent, seed, steps, sampler_name, scheduler, vae,
            audio_vae, cfg=cfg, clip=clip)

        head = images[plan.head_range[0]:plan.head_range[1]]
        interior = patch_images[plan.patch_take[0]:plan.patch_take[1]]
        tail = images[plan.tail_range[0]:plan.tail_range[1]]
        stitched = torch.cat([p for p in (head, interior, tail) if p.shape[0]], dim=0)

        if keep_base_audio and base_audio is not None:
            audio = base_audio
        elif keep_base_audio:
            log.warning("[PulseStudio] keep_base_audio is on but base_audio is not "
                        "connected; returning the patch's own audio, which covers only "
                        "the patched span.")
            audio = patch_audio
        else:
            audio = patch_audio

        return (stitched, audio, plan.describe())


# ── Still Mode ──────────────────────────────────────────────────────────────

class PulseStill:
    """Generate or edit a single image through the same H3 pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE", {"tooltip":
                    "H3 always builds a joint audio+video latent, so this is required even "
                    "though a still discards the audio."}),

                # Widget index 0, under the same contract as PulseSlate (§3).
                "schema_version": schema_widget(),

                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                      "default": ""}),
                "aspect_ratio": (ASPECT_OPTIONS, {"default": "16:9 landscape"}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 736, "min": 32, "max": 4096, "step": 32}),
                "frame_pick": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1,
                                       "display": "slider", "tooltip":
                    "Which of the 5 rendered frames to keep. The source is pinned at frame "
                    "0, so 0 hugs the original and 4 drifts furthest. This is the "
                    "edit-strength dial."}),
                "canvas_from_reference": ("BOOLEAN", {"default": True, "tooltip":
                    "Let the source image set the canvas instead of cropping it to the "
                    "chosen aspect. Its aspect is fitted into H3's pixel budget, rounded "
                    "down to multiples of 32."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "sampler_name": (SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (SCHEDULERS, {"default": "simple"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0,
                                          "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0,
                                          "step": 0.01}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": {
                "source_image": ("IMAGE", {"tooltip":
                    "Editing a still pins this at frame 0, which is the fl2va branch -- it "
                    "takes no references. Leave unconnected to generate instead."}),
                "ref_images": ("IMAGE", {"tooltip":
                    "Reference images for generation (ref2va). Ignored when source_image is "
                    "connected, since anchors and references cannot coexist in one render."}),
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "plan")
    FUNCTION = "execute"
    CATEGORY = "AddisPulse/H3"
    DESCRIPTION = ("A still is a 5-frame H3 render where one frame is kept. Generate a "
                   "reference image through the same reference set that will drive the "
                   "video, or edit an existing one in place.")

    def execute(self, model, clip, vae, audio_vae, schema_version,
                prompt, aspect_ratio, width, height,
                frame_pick, canvas_from_reference, seed, steps, sampler_name, scheduler,
                cfg, shift_video, shift_audio, source_image=None, ref_images=None,
                unique_id=None):
        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
            MiniMaxH3SigmaShift,
        )

        width, height = resolution_for(aspect_ratio, width, height)
        source_size = None
        if source_image is not None:
            source_size = (int(source_image.shape[2]), int(source_image.shape[1]))
        try:
            plan = plan_still(prompt=prompt, width=width, height=height,
                              frame_pick=frame_pick,
                              source_asset="source_image" if source_image is not None else None,
                              source_size=source_size,
                              canvas_from_reference_enabled=canvas_from_reference,
                              reference_ids=["ref_images"] if ref_images is not None else None)
        except StillError as exc:
            message = "Still refused: %s" % (exc,)
            log.error("[PulseStudio] %s", message)
            return (ExecutionBlocker(message), message)

        # A still is a 5-frame render, so the memory case is mild -- but the
        # attention patch still applies, and a silently unpatched graph here is
        # the same graph the user will scale up to a timeline.
        _report_patches(model, unique_id)

        for note in plan.diagnostics:
            log.warning("[PulseStudio] still: %s", note)

        shifted = render.unpack(render.execute_node(
            MiniMaxH3SigmaShift, model=model,
            shift_video=shift_video, shift_audio=shift_audio))[0]

        if plan.branch == BRANCH_FL2VA:
            first = media.fit_image(source_image, plan.width, plan.height, "crop")
            out = render.execute_node(
                MiniMaxH3ImageToVideo, clip=clip, vae=vae, prompt=prompt,
                width=plan.width, height=plan.height, length=plan.length,
                first_frame=first, last_frame=None)
        else:
            refs = None
            if ref_images is not None:
                # Contiguous 0-based keys, built by enumerate so a gap is impossible.
                refs = {"ref_image_%d" % i: media.fit_image(ref_images[i:i + 1], plan.width,
                                                            plan.height, "crop")
                        for i in range(min(ref_images.shape[0], 9))}
                check_socket_groups({"ref_images": refs})
            out = render.execute_node(
                MiniMaxH3ReferenceToVideo, clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=prompt, width=plan.width, height=plan.height,
                length=plan.length, ref_image_size="match", ref_images=refs)

        positive, latent = render.unpack(out)[:2]
        _, images, _ = render.sample(shifted, positive, latent, seed, steps, sampler_name,
                                     scheduler, vae, audio_vae, cfg=cfg, clip=clip)

        index = min(plan.frame_pick, images.shape[0] - 1)
        return (images[index:index + 1], repr(plan))


NODE_CLASS_MAPPINGS = {
    "PulseSlate": PulseSlate,
    "PulseShot": PulseShot,
    "PulseRender": PulseRender,
    "PulseBench": PulseBench,
    "PulseRetake": PulseRetake,
    "PulseStill": PulseStill,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulseSlate": "Pulse Slate · MiniMax H3",
    "PulseShot": "Pulse Shot",
    "PulseRender": "Pulse Render",
    "PulseBench": "Pulse Bench",
    "PulseRetake": "Pulse Retake · MiniMax H3",
    "PulseStill": "Pulse Still · MiniMax H3",
}
