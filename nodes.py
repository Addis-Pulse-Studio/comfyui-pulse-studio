"""ComfyUI node layer for Pulse Studio.

The nodes here are thin. Every decision that can be made without a tensor was
already made in comfyui_pulse_studio/ and is covered by the headless test suite; this
file marshals files into tensors, calls the stock H3 nodes, and runs the sampling
loop. Nothing here re-derives a frame count, an ordinal, or a cut point.

Nothing is ever pushed to the ComfyUI API. Multi-window renders loop in-process,
one real H3 call per window.

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
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    MiniMaxH3SigmaShift,
)

from . import media
from .comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO
from .comfyui_pulse_studio.compiler import (
    CARRY_AUDIO_ID,
    CARRY_IMAGE_ID,
    CARRY_VIDEO_ID,
    CarryPolicy,
    compile_timeline,
)
from .comfyui_pulse_studio.constants import (
    BRANCH_FL2VA,
    BRANCH_REF2VA,
    CANVAS_MULTIPLE,
    DEFAULT_AUDIO_CARRY_SECONDS,
    MAX_PIXELS,
    MAX_REF_AUDIOS,
    MAX_REF_AUDIOS_CEILING,
    REF_IMAGE_SIZE_OPTIONS,
    SCHEMA_VERSION,
)
from .comfyui_pulse_studio.patches import check_model_patches, check_single_checkpoint
from .comfyui_pulse_studio.retake import RetakeError, plan_retake
from .comfyui_pulse_studio.sockets import SocketGapError, check_socket_groups, drop_missing
from .comfyui_pulse_studio.still import StillError, plan_still
from .comfyui_pulse_studio.widget_state import EMPTY_TIMELINE_DATA, build_timeline

log = logging.getLogger(__name__)


# ── The slot contract, in one place ─────────────────────────────────────────
# Every node in this pack opens its widget list with `schema_version`, so that
# a saved workflow always states which layout it was written in before anything
# tries to read it back. Spelled as a helper rather than copied three times:
# three copies is three chances for one of them to drift, and drift here is the
# failure mode the whole contract exists to prevent. See spec §3.


def schema_widget():
    """The `schema_version` widget. Always widget index 0. Never displayed."""
    return ("STRING", {
        "default": SCHEMA_VERSION, "multiline": False,
        "tooltip": "Which widget layout this node was saved with. Written by the "
                   "node, read at load time to restore values by name. Do not edit."})


# `hidden` inputs are not widgets and consume no widgets_values slot, so adding
# one is outside the §3 append-only rule. UNIQUE_ID is what lets a warning be
# addressed to the node that raised it.
HIDDEN_INPUTS = {"unique_id": "UNIQUE_ID"}


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
    warning above still happened, and nothing here may break the render.
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


# ── stock-node plumbing ─────────────────────────────────────────────────────

def _execute(node_class, **kwargs):
    """Call a ComfyUI node's entrypoint, io.ComfyNode or legacy alike."""
    if hasattr(node_class, "execute"):
        return node_class.execute(**kwargs)
    fn = getattr(node_class, "FUNCTION", None)
    instance = node_class()
    if fn and hasattr(instance, fn):
        return getattr(instance, fn)(**kwargs)
    raise RuntimeError("cannot determine how to execute %r" % (node_class,))


def _unpack(out):
    """Normalise a node return (io.NodeOutput, tuple, list, dict) to a tuple."""
    if out is None:
        return ()
    for attr in ("result", "args", "values", "outputs"):
        if hasattr(out, attr):
            value = getattr(out, attr)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if isinstance(value, (tuple, list)):
                return tuple(value)
    if isinstance(out, (tuple, list)):
        return tuple(out)
    if isinstance(out, dict) and isinstance(out.get("result"), (tuple, list)):
        return tuple(out["result"])
    return (out,)


def _stock(name):
    from nodes import NODE_CLASS_MAPPINGS
    return NODE_CLASS_MAPPINGS[name]


def _negative_conditioning(clip):
    """An empty-prompt conditioning, for CFG only.

    H3's own reference pipeline has no negative conditioning anywhere -- it uses
    BasicGuider, which takes a single conditioning input. CFG is offered because
    it was asked for, but cfg=1.0 (BasicGuider, no negative) is the model's
    native path and the correct default.
    """
    tokens = clip.tokenize("")
    return clip.encode_from_tokens_scheduled(tokens)


def _sample(model, positive, latent, seed, steps, sampler_name, scheduler,
            vae, audio_vae, cfg=1.0, clip=None):
    """One real H3 call: noise -> guider -> sigmas -> sample -> decode both streams."""
    noise = _unpack(_execute(_stock("RandomNoise"), noise_seed=seed))[0]

    if cfg is not None and abs(float(cfg) - 1.0) > 1e-6 and clip is not None:
        negative = _negative_conditioning(clip)
        guider = _unpack(_execute(_stock("CFGGuider"), model=model, positive=positive,
                                  negative=negative, cfg=float(cfg)))[0]
    else:
        guider = _unpack(_execute(_stock("BasicGuider"), model=model,
                                  conditioning=positive))[0]

    sampler = _unpack(_execute(_stock("KSamplerSelect"), sampler_name=sampler_name))[0]
    sigmas = _unpack(_execute(_stock("BasicScheduler"), model=model, scheduler=scheduler,
                              steps=steps, denoise=1.0))[0]
    sampled = _unpack(_execute(_stock("SamplerCustomAdvanced"), noise=noise, guider=guider,
                               sampler=sampler, sigmas=sigmas, latent_image=latent))[0]
    images = _unpack(_execute(_stock("VAEDecode"), samples=sampled, vae=vae))[0]
    audio = _unpack(_execute(_stock("VAEDecodeAudio"), samples=sampled, vae=audio_vae))[0]
    return sampled, images, audio


# ── reference marshalling ───────────────────────────────────────────────────

def _load_window_references(window, width, height, resize_method, carry_frame, carry_clip,
                            carry_audio_clip, carry_seconds):
    """Turn a CompiledWindow's ordered file list into the stock node's kwargs.

    The compiler already decided every socket name and every ordinal; this only
    resolves each FileRef to a tensor. Synthetic entries are carry-over signals,
    which come from the previous window rather than from disk.

    Files that fail to load here would leave a HOLE in the socket dict, and a
    hole shifts every tag after it. Missing files were already dropped from the
    bin before compiling (see drop_missing), so a failure at this point is a
    decode error, not a missing path -- and it is raised rather than skipped.
    """
    groups = {"ref_images": {}, "ref_videos": {}, "ref_video_audios": {}, "ref_audios": {}}
    failures = []

    for ref in window.files:
        if ref.synthetic:
            if ref.asset_id == CARRY_IMAGE_ID and carry_frame is not None:
                groups["ref_images"][ref.socket] = media.fit_image(
                    carry_frame, width, height, resize_method)
            elif ref.asset_id == CARRY_VIDEO_ID and carry_clip is not None:
                groups["ref_videos"][ref.socket] = carry_clip
            elif ref.asset_id == CARRY_AUDIO_ID and carry_audio_clip is not None:
                groups["ref_audios"][ref.socket] = media.audio_tail(
                    carry_audio_clip, carry_seconds)
            continue

        if ref.kind == KIND_IMAGE:
            tensor = media.load_image(ref.file)
            if tensor is None:
                failures.append(ref)
                continue
            groups["ref_images"][ref.socket] = media.fit_image(
                tensor, width, height, resize_method)
        elif ref.kind == KIND_VIDEO:
            if ref.socket.startswith("ref_video_audio_"):
                clip = media.load_audio(ref.file, ref.trim_start, ref.trim_end)
                if clip is None:
                    failures.append(ref)
                    continue
                groups["ref_video_audios"][ref.socket] = clip
            else:
                frames = media.load_video(ref.file, ref.trim_start, ref.trim_end)
                if frames is None:
                    failures.append(ref)
                    continue
                groups["ref_videos"][ref.socket] = frames
        elif ref.kind == KIND_AUDIO:
            clip = media.load_audio(ref.file, ref.trim_start, ref.trim_end)
            if clip is None:
                failures.append(ref)
                continue
            groups["ref_audios"][ref.socket] = clip

    if failures:
        raise SocketGapError(
            "could not decode %s. Skipping them would leave a gap in the reference "
            "sockets and silently renumber every tag after it, so the render is stopped "
            "instead." % ", ".join("%s (%s)" % (f.name, f.file) for f in failures))

    # The net under the whole design: contiguous, 0-based, gapless.
    check_socket_groups(groups)
    return {k: (v or None) for k, v in groups.items()}


def _condition_window(window, timeline, clip, vae, audio_vae, width, height,
                      ref_image_size, resize_method, carry_frame, carry_clip,
                      carry_audio_clip, carry_seconds):
    """Build (positive, latent) for one window through the correct stock node."""
    if window.branch == BRANCH_REF2VA:
        refs = _load_window_references(window, width, height, resize_method, carry_frame,
                                       carry_clip, carry_audio_clip, carry_seconds)
        out = _execute(MiniMaxH3ReferenceToVideo, clip=clip, vae=vae, audio_vae=audio_vae,
                       prompt=window.prompt, width=width, height=height,
                       length=window.frame_count, ref_image_size=ref_image_size, **refs)
    else:
        first = last = None
        anchors = window.anchors
        if anchors.get("first_frame") == CARRY_IMAGE_ID:
            first = media.fit_image(carry_frame, width, height, resize_method)
        elif anchors.get("first_frame"):
            asset = timeline.assets.get(anchors["first_frame"])
            if asset is not None:
                first = media.fit_image(media.load_image(asset.file), width, height, resize_method)
        if anchors.get("last_frame"):
            asset = timeline.assets.get(anchors["last_frame"])
            if asset is not None:
                last = media.fit_image(media.load_image(asset.file), width, height, resize_method)
        out = _execute(MiniMaxH3ImageToVideo, clip=clip, vae=vae, prompt=window.prompt,
                       width=width, height=height, length=window.frame_count,
                       first_frame=first, last_frame=last)
    return _unpack(out)[:2]


# ── the Director ────────────────────────────────────────────────────────────

class PulseSlate:
    """Timeline -> compiled storyboard -> one or more real H3 calls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip":
                    "The ref2va (Reference) checkpoint. Anchors and references are different "
                    "checkpoints; wire the fl2va one into model_fl2va."}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE", {"tooltip":
                    "Required on both branches -- H3 always builds a joint audio+video "
                    "latent, even when no reference audio is encoded."}),

                # ── the frozen prefix — widget indices 0 and 1 ──────────────
                # Both are hidden on the node face (the JS sets type="hidden"
                # and a zero computeSize). They lead the widget list because
                # LiteGraph serialises widgets positionally, so a saved file has
                # to be able to say which order it was written in BEFORE anything
                # tries to read it. See js/ps_widget_order.js and spec §3.
                #
                # Nothing may ever be inserted ahead of these two, and nothing
                # between them. New widgets append at the end of this dict.
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
                    "Shots without a timecode are spread evenly between the ones that have "
                    "them, so you can stamp only the moments you care about.\n\n"
                    "Quoted \"text\" becomes dialogue. @Name references the Asset Bin."}),

                # ── exposed control panel ───────────────────────────────────
                "duration_seconds": ("FLOAT", {
                    "default": 10.0, "min": 0.2, "max": 600.0, "step": 0.5,
                    "tooltip": "Total length of the finished video. Snapped up to the "
                               "17k+5 frame grid, and split into windows if longer than "
                               "window_seconds."}),
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
                                 "control_after_generate": True}),
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
                    "What the previous window contributes to the next. 'image' carries its "
                    "last frame as a reference picture; 'video' carries motion at the cost "
                    "of a video socket."}),
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
                    "Above 3 goes past both: that cap is enforced by graph validation "
                    "and this node calls the reference encoder in-process, so the extra "
                    "audio really is marshalled and rendered -- but no published render "
                    "uses more than three, and reference audio rides every sampling "
                    "step. The file total rises with it. Raise it to test, not to set "
                    "and forget."}),
                # ── append new widgets HERE, at the end, and nowhere else ────
            },
            "optional": {
                "model_fl2va": ("MODEL", {"tooltip":
                    "The First/Last-Frame checkpoint, needed only if a window uses the "
                    "fl2va branch. Loading both checkpoints is ~42GB."}),
            },
            "hidden": HIDDEN_INPUTS,
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "AUDIO", "IMAGE", "STRING")
    RETURN_NAMES = ("model", "positive", "latent", "combined_audio", "images", "compiled_prompt")
    FUNCTION = "execute"
    CATEGORY = "AddisPulse/H3"
    DESCRIPTION = ("Compiles two prompt boxes and an Asset Bin into a MiniMax H3 storyboard. "
                   "A single window hands back positive/latent for your own sampler; a longer "
                   "timeline renders internally, one H3 call per window, and is stitched.")

    def execute(self, model, clip, vae, audio_vae, schema_version, timeline_data,
                global_prompt, shot_prompt,
                duration_seconds, aspect_ratio, width, height, steps, sampler_name,
                scheduler, cfg, seed, partition_strategy, window_seconds, resize_method,
                carry_mode, carry_audio, carry_audio_seconds, ref_image_size,
                shift_video, shift_audio, audio_ref_ceiling=MAX_REF_AUDIOS,
                model_fl2va=None, unique_id=None):

        width, height = resolution_for(aspect_ratio, width, height)

        # Read widgets, build in memory. Nothing below assigns to a widget --
        # see tests/test_widget_state.py::TestQueueDoesNotEraseTypedText.
        timeline, notes = build_timeline(
            timeline_data, global_prompt=global_prompt, shot_prompt=shot_prompt,
            duration_seconds=duration_seconds, window_seconds=window_seconds,
            audio_ref_ceiling=audio_ref_ceiling)

        # Said once, on the render that does it, rather than buried in a tooltip
        # nobody reads twice. It is a warning and not a block: going past the
        # documented budget is a legitimate experiment, and the user chose it.
        if timeline.limits.beyond_spec:
            notes.append(
                "audio_ref_ceiling is %d. MiniMax documents 3 standalone audio "
                "references and ComfyUI's ref_audios socket declares 3; this render "
                "goes past both. It will run -- the cap is graph-validation only and "
                "this node calls the encoder in-process -- but nothing about the "
                "result is covered by anything published, and every extra reference "
                "rides all %d sampling steps."
                % (timeline.limits.audios, steps))

        # Drop unloadable references BEFORE tags are assigned, so ordinals stay
        # dense by construction rather than developing a hole at render time.
        notes.extend(drop_missing(timeline, lambda f: bool(media.resolve_path(f))))

        carry = CarryPolicy(mode=carry_mode, audio=carry_audio,
                            audio_seconds=carry_audio_seconds)
        plan = compile_timeline(timeline, policy=partition_strategy, carry=carry)

        preview = plan.preview()
        for note in notes + plan.diagnostics:
            log.info("[PulseStudio] %s", note)
        for window in plan.windows:
            for note in window.diagnostics:
                log.warning("[PulseStudio] window %d: %s", window.index + 1, note)

        if not plan.ok:
            message = "Timeline cannot be compiled:\n  - " + "\n  - ".join(plan.problems)
            log.error("[PulseStudio] %s", message)
            blocked = ExecutionBlocker(message)
            return (model, blocked, blocked, blocked, blocked, message)

        shifted = _unpack(_execute(MiniMaxH3SigmaShift, model=model,
                                   shift_video=shift_video, shift_audio=shift_audio))[0]
        shifted_fl2va = None
        if model_fl2va is not None:
            shifted_fl2va = _unpack(_execute(MiniMaxH3SigmaShift, model=model_fl2va,
                                             shift_video=shift_video,
                                             shift_audio=shift_audio))[0]

        if any(w.branch == BRANCH_FL2VA for w in plan.windows) and shifted_fl2va is None:
            log.warning("[PulseStudio] a window uses the fl2va branch but model_fl2va is not "
                        "connected; falling back to the main model, which normally holds the "
                        "ref2va checkpoint. Expect degraded output.")

        # §10: check the model we were HANDED, not the one we return. A patch
        # applied downstream of this node would do nothing for the internal
        # sampling loop below, which is the case where it matters most.
        _report_patches(model, unique_id,
                        branches_used={w.branch for w in plan.windows},
                        fl2va_connected=model_fl2va is not None)

        log.info("[PulseStudio] %d window(s): %s (%.2fs total, %dx%d)",
                 len(plan.windows), ", ".join(str(w.frame_count) for w in plan.windows),
                 plan.total_seconds, width, height)

        try:
            # ── single window: let the graph sample ──────────────────────────
            if len(plan.windows) == 1:
                window = plan.windows[0]
                positive, latent = _condition_window(
                    window, timeline, clip, vae, audio_vae, width, height, ref_image_size,
                    resize_method, None, None, None, carry_audio_seconds)
                out_model = (shifted_fl2va if (window.branch == BRANCH_FL2VA and shifted_fl2va)
                             else shifted)
                return (out_model, positive, latent, media.empty_audio(),
                        media.empty_images(width, height), preview)

            # ── multi-window: sample internally, carrying continuity forward ─
            all_images, all_audio = [], []
            carry_frame = carry_clip = carry_audio_clip = None
            positive = latent = None

            for window in plan.windows:
                positive, latent = _condition_window(
                    window, timeline, clip, vae, audio_vae, width, height, ref_image_size,
                    resize_method, carry_frame, carry_clip, carry_audio_clip,
                    carry_audio_seconds)
                window_model = (shifted_fl2va
                                if (window.branch == BRANCH_FL2VA and shifted_fl2va)
                                else shifted)
                log.info("[PulseStudio] window %d/%d: %s, %d frames, seed %d",
                         window.index + 1, window.total, window.branch,
                         window.frame_count, seed + window.seed_offset)

                latent, images, audio = _sample(
                    window_model, positive, latent, seed + window.seed_offset, steps,
                    sampler_name, scheduler, vae, audio_vae, cfg=cfg, clip=clip)
                all_images.append(images)
                all_audio.append(audio)

                carry_frame = images[-1:]
                carry_clip = images[-min(images.shape[0], 48):]
                carry_audio_clip = audio

            return (shifted, positive, latent, media.concat_audio(all_audio),
                    torch.cat(all_images, dim=0), preview)

        except SocketGapError as exc:
            message = ("Reference sockets would have been misnumbered, so the render was "
                       "stopped:\n  %s" % (exc,))
            log.error("[PulseStudio] %s", message)
            blocked = ExecutionBlocker(message)
            return (model, blocked, blocked, blocked, blocked, message)


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

        shifted = _unpack(_execute(MiniMaxH3SigmaShift, model=model_fl2va,
                                   shift_video=shift_video, shift_audio=shift_audio))[0]
        positive, latent = _unpack(_execute(
            MiniMaxH3ImageToVideo, clip=clip, vae=vae, prompt=prompt, width=width,
            height=height, length=plan.patch_frames, first_frame=first, last_frame=last))[:2]

        _, patch_images, patch_audio = _sample(shifted, positive, latent, seed, steps,
                                               sampler_name, scheduler, vae, audio_vae,
                                               cfg=cfg, clip=clip)

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

        shifted = _unpack(_execute(MiniMaxH3SigmaShift, model=model,
                                   shift_video=shift_video, shift_audio=shift_audio))[0]

        if plan.branch == BRANCH_FL2VA:
            first = media.fit_image(source_image, plan.width, plan.height, "crop")
            out = _execute(MiniMaxH3ImageToVideo, clip=clip, vae=vae, prompt=prompt,
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
            out = _execute(MiniMaxH3ReferenceToVideo, clip=clip, vae=vae, audio_vae=audio_vae,
                           prompt=prompt, width=plan.width, height=plan.height,
                           length=plan.length, ref_image_size="match", ref_images=refs)

        positive, latent = _unpack(out)[:2]
        _, images, _ = _sample(shifted, positive, latent, seed, steps, sampler_name,
                               scheduler, vae, audio_vae, cfg=cfg, clip=clip)

        index = min(plan.frame_pick, images.shape[0] - 1)
        return (images[index:index + 1], repr(plan))


NODE_CLASS_MAPPINGS = {
    "PulseSlate": PulseSlate,
    "PulseRetake": PulseRetake,
    "PulseStill": PulseStill,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulseSlate": "Pulse Slate · MiniMax H3",
    "PulseRetake": "Pulse Retake · MiniMax H3",
    "PulseStill": "Pulse Still · MiniMax H3",
}
