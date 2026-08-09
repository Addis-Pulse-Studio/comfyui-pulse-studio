"""The executor: walk a PULSE_TIMELINE's windows, reuse what is cached, render the rest.

Spec §7 (cache, manifest, resume), §8 (memory), §11 (continuity), §12 (patches).

This is the tensor half of `PulseRender`. It is separate from nodes.py so that the
node faces stay declarative and this -- the part with the loop, the disk writes
and the crash-safety ordering -- can be read on its own.

THE ORDERING THAT MATTERS
-------------------------
Per window, in this order and no other:

    render -> write mp4 -> write flac -> write last-frame png -> fsync each
           -> upsert the manifest entry -> fsync the manifest -> next window

The manifest is a promise that the bytes are on disk. Writing it first would make
that promise early, and a crash in the gap would leave a manifest claiming a
segment that is not there -- which the next run would happily "reuse", producing
a film with a hole in it and no error anywhere. §7.4 is explicit about this and it
is the one ordering in the file that must not be rearranged for convenience.

WHY THE CLIP COMES THROUGH THE SIDE CHANNEL
-------------------------------------------
§2.3 gives `PulseRender` four model inputs and no CLIP. But conditioning for
window i+1 cannot be built until window i has been decoded -- that is precisely
what `last_frame_carry` means -- so conditioning has to be built here, at render
time, and building it needs the text encoder. It arrives on the `SideChannel`
attached to the timeline instead of on a fifth socket: the node face stays exactly
as specified, and the encoder that wrote the prompts is provably the same object
that encodes them.
"""

import logging
import os
import time
from datetime import datetime, timezone

import torch

import folder_paths
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    MiniMaxH3SigmaShift,
)

from . import media
from .comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO
from .comfyui_pulse_studio.compiler import CARRY_AUDIO_ID, CARRY_IMAGE_ID, CARRY_VIDEO_ID
from .comfyui_pulse_studio.constants import (
    AUDIO_ROLE_LIP_SYNC,
    BRANCH_FL2VA,
    BRANCH_REF2VA,
)
from .comfyui_pulse_studio.fingerprint import (
    describe_model_patches,
    model_fingerprint,
    patch_fingerprint,
    patch_warnings,
)
from .comfyui_pulse_studio.pulse_timeline import socket_slot_of
from .comfyui_pulse_studio.report import build_report
from .comfyui_pulse_studio.segcache import (
    CACHE_AUTO,
    Manifest,
    cache_key,
    derive_run_id,
    plan_window,
    segment_entry,
    segment_paths,
    stale_entries,
)
from .comfyui_pulse_studio.sockets import SocketGapError, check_socket_groups

log = logging.getLogger(__name__)


# ── stock-node plumbing ─────────────────────────────────────────────────────

def execute_node(node_class, **kwargs):
    """Call a ComfyUI node's entrypoint, io.ComfyNode or legacy alike."""
    if hasattr(node_class, "execute"):
        return node_class.execute(**kwargs)
    fn = getattr(node_class, "FUNCTION", None)
    instance = node_class()
    if fn and hasattr(instance, fn):
        return getattr(instance, fn)(**kwargs)
    raise RuntimeError("cannot determine how to execute %r" % (node_class,))


def unpack(out):
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


def stock(name):
    from nodes import NODE_CLASS_MAPPINGS
    return NODE_CLASS_MAPPINGS[name]


def _negative_conditioning(clip):
    """An empty-prompt conditioning, for CFG only.

    H3's own reference pipeline has no negative conditioning anywhere -- it uses
    BasicGuider, which takes a single conditioning input. CFG is offered because
    it was asked for, but cfg=1.0 is the model's native path and the default.
    """
    tokens = clip.tokenize("")
    return clip.encode_from_tokens_scheduled(tokens)


def sample(model, positive, latent, seed, steps, sampler_name, scheduler,
           vae, audio_vae, cfg=1.0, clip=None):
    """One real H3 call: noise -> guider -> sigmas -> sample -> decode both streams."""
    noise = unpack(execute_node(stock("RandomNoise"), noise_seed=seed))[0]

    if cfg is not None and abs(float(cfg) - 1.0) > 1e-6 and clip is not None:
        negative = _negative_conditioning(clip)
        guider = unpack(execute_node(stock("CFGGuider"), model=model, positive=positive,
                                     negative=negative, cfg=float(cfg)))[0]
    else:
        guider = unpack(execute_node(stock("BasicGuider"), model=model,
                                     conditioning=positive))[0]

    sampler = unpack(execute_node(stock("KSamplerSelect"), sampler_name=sampler_name))[0]
    sigmas = unpack(execute_node(stock("BasicScheduler"), model=model, scheduler=scheduler,
                                 steps=steps, denoise=1.0))[0]
    sampled = unpack(execute_node(stock("SamplerCustomAdvanced"), noise=noise, guider=guider,
                                  sampler=sampler, sigmas=sigmas, latent_image=latent))[0]
    images = unpack(execute_node(stock("VAEDecode"), samples=sampled, vae=vae))[0]
    audio = unpack(execute_node(stock("VAEDecodeAudio"), samples=sampled, vae=audio_vae))[0]
    return sampled, images, audio


# ── reference marshalling ───────────────────────────────────────────────────

def load_window_references(window, side, width, height, resize_method, carry_frame,
                           carry_clip, carry_audio_clip, carry_seconds,
                           window_seconds=None):
    """Turn a CompiledWindow's ordered file list into the stock node's kwargs.

    The compiler already decided every socket name and every ordinal; this only
    resolves each FileRef to a tensor. Three sources feed it: a synthetic entry is
    a carry-over signal from the previous window, a socket entry is a tensor
    already in the side channel, and everything else is a file on disk.

    A failure here would leave a HOLE in the socket dict, and a hole shifts every
    tag after it. Missing files were dropped from the bin before compiling (see
    drop_missing), so a failure at this point is a decode error, not a missing
    path -- and it is raised rather than skipped.
    """
    groups = {"ref_images": {}, "ref_videos": {}, "ref_video_audios": {}, "ref_audios": {}}
    failures = []

    def _audio_for(ref, clip):
        """A lip-sync clip is cut to this window's own span; a timbre clip is not.

        Alignment is the whole difference between the two roles. The model packs
        reference audio rows against the target audio grid, so a lip-sync clip has
        to cover exactly the seconds the window covers -- see
        constants.AUDIO_ROLE_LIP_SYNC. A timbre reference is asked no temporal
        question at all, and trimming it would only throw away voice to learn from.
        """
        if clip is None or ref.audio_role != AUDIO_ROLE_LIP_SYNC or not window_seconds:
            return clip
        return media.audio_span(clip, window.start_seconds, window_seconds)

    for ref in window.files:
        if ref.synthetic:
            # A carry-over signal the compiler allocated a socket for. It is NOT
            # optional at this point: the socket exists, the ordinals after it
            # were assigned around it, and skipping it punches a hole that shifts
            # every tag behind it. Before the segment cache existed this could not
            # happen -- window i always rendered before window i+1 -- but a
            # *reused* window produces no fresh tensors, so the executor has to
            # reconstruct them from disk. If it could not, stop and say so.
            if ref.asset_id == CARRY_IMAGE_ID:
                value = (media.fit_image(carry_frame, width, height, resize_method)
                         if carry_frame is not None else None)
                groups["ref_images"][ref.socket] = value
            elif ref.asset_id == CARRY_VIDEO_ID:
                value = carry_clip
                groups["ref_videos"][ref.socket] = value
            elif ref.asset_id == CARRY_AUDIO_ID:
                value = (media.audio_tail(carry_audio_clip, carry_seconds)
                         if carry_audio_clip is not None else None)
                groups["ref_audios"][ref.socket] = value
            else:  # pragma: no cover - unknown synthetic kind
                value = None
            if value is None:
                failures.append(ref)
            continue

        slot = socket_slot_of(ref.asset_id)
        if slot is not None:
            # A reference wired straight in from an upstream generator. Already a
            # tensor; it never had a file, and never needs one.
            tensor = side.get(slot)
            if tensor is None:
                failures.append(ref)
                continue
            if ref.kind == KIND_AUDIO:
                groups["ref_audios"][ref.socket] = _audio_for(ref, tensor)
            elif ref.kind == KIND_VIDEO:
                if ref.socket.startswith("ref_video_audio_"):
                    groups["ref_video_audios"][ref.socket] = tensor
                else:
                    groups["ref_videos"][ref.socket] = tensor
            else:
                groups["ref_images"][ref.socket] = media.fit_image(
                    tensor, width, height, resize_method)
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
            groups["ref_audios"][ref.socket] = _audio_for(ref, clip)

    if failures:
        raise SocketGapError(
            "could not resolve %s. Skipping them would leave a gap in the reference "
            "sockets and silently renumber every tag after it, so the render is stopped "
            "instead." % ", ".join("%s (%s)" % (f.name, f.file or "socket") for f in failures))

    # The net under the whole design: contiguous, 0-based, gapless.
    check_socket_groups(groups)
    return {k: (v or None) for k, v in groups.items()}


def carry_from_segment(manifest, entry, wants_video=False, wants_audio=False):
    """Rebuild the carry-over signals of a window that was loaded, not rendered.

    A reused window decodes nothing, so the tensors the next window's carry-over
    sockets need do not exist in memory. They do exist on disk: the last frame as
    its own PNG, the audio as the segment's lossless FLAC, and the tail motion in
    the segment video itself.

    Only what the next window actually asks for is read. The frame and the audio
    are cheap; the video tail means decoding a segment, so it happens solely when
    `carry_mode` is video or both -- which is why the caller inspects the next
    window's socket list rather than reconstructing everything unconditionally.
    """
    frame = media.read_png(manifest.resolve(entry, "last_frame_path"))
    audio = None
    clip = None
    if wants_audio:
        audio_path = manifest.resolve(entry, "audio_path")
        if audio_path and os.path.exists(audio_path):
            audio = media.load_audio(audio_path)
    if wants_video:
        frames = media.read_segment_frames(manifest.resolve(entry, "video_path"))
        if frames is not None:
            clip = frames[-min(frames.shape[0], 48):]
    return frame, clip, audio


def window_wants(compiled, asset_id):
    """Does this compiled window carry a socket for that carry-over signal?"""
    return any(f.synthetic and f.asset_id == asset_id for f in (compiled.files if compiled else []))


def condition_window(window, side, width, height, carry_frame, carry_clip,
                     carry_audio_clip, fps=24):
    """Build (positive, latent) for one window through the correct stock node."""
    clip, vae, audio_vae = side.clip, side.vae, side.audio_vae
    resize_method = side.resize_method

    if window.branch == BRANCH_REF2VA:
        # frames/fps, not end_seconds - start_seconds: the window is exactly
        # frame_count frames long after grid snapping, and it is the frames the
        # reference audio has to line up with.
        refs = load_window_references(
            window, side, width, height, resize_method, carry_frame, carry_clip,
            carry_audio_clip, side.carry_audio_seconds,
            window_seconds=(window.frame_count / float(fps or 24)))
        out = execute_node(MiniMaxH3ReferenceToVideo, clip=clip, vae=vae, audio_vae=audio_vae,
                           prompt=window.prompt, width=width, height=height,
                           length=window.frame_count, ref_image_size=side.ref_image_size,
                           **refs)
    else:
        first = last = None
        anchors = window.anchors
        timeline = side.timeline
        if anchors.get("first_frame") == CARRY_IMAGE_ID:
            first = media.fit_image(carry_frame, width, height, resize_method)
        elif anchors.get("first_frame"):
            first = _anchor_image(anchors["first_frame"], side, width, height, resize_method,
                                  timeline)
        if anchors.get("last_frame"):
            last = _anchor_image(anchors["last_frame"], side, width, height, resize_method,
                                 timeline)
        out = execute_node(MiniMaxH3ImageToVideo, clip=clip, vae=vae, prompt=window.prompt,
                           width=width, height=height, length=window.frame_count,
                           first_frame=first, last_frame=last)
    return unpack(out)[:2]


def _anchor_image(asset_id, side, width, height, resize_method, timeline):
    """Resolve a keyframe anchor, whether it came from a socket or from the bin."""
    slot = socket_slot_of(asset_id)
    if slot is not None:
        return media.fit_image(side.get(slot), width, height, resize_method)
    asset = timeline.assets.get(asset_id) if timeline is not None else None
    if asset is None:
        return None
    return media.fit_image(media.load_image(asset.file), width, height, resize_method)


# ── output-connectivity (spec §8) ───────────────────────────────────────────

def output_is_connected(prompt, unique_id, output_index):
    """Is this node's output `output_index` wired to anything?

    §8 asks for the frame stack never to be assembled unless the `frames` socket
    is actually connected. ComfyUI does not tell a node which of its outputs are
    used, but the whole prompt graph is available as a hidden input and a link is
    just `[node_id, output_index]` sitting in some other node's inputs. So the
    question is answerable, and answering it is what keeps a twelve-window render
    from holding the entire film in RAM to satisfy a socket nobody wired.
    """
    if not prompt or unique_id is None:
        return False
    target = str(unique_id)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if (isinstance(value, list) and len(value) == 2
                    and str(value[0]) == target and int(value[1]) == output_index):
                return True
    return False


# ── the run ─────────────────────────────────────────────────────────────────

def reference_audio_track(plan_windows, windows_doc, side, fps=24):
    """The lip-sync recordings themselves, laid end to end over the whole film.

    Built from the timeline rather than from the render loop on purpose: a reused
    window decodes nothing and never calls `load_window_references`, so collecting
    these during rendering would silently produce a track with holes in it exactly
    where the cache did its job.

    A window with no lip-sync reference contributes silence of its own length, so
    the track stays in step with the video whatever mix of shots the film has.
    Returns None when no window carries one at all.
    """
    pieces = []
    for position, window in enumerate(windows_doc):
        seconds = (window.get("frames") or 0) / float(window.get("fps") or fps or 24)
        compiled = plan_windows[position] if position < len(plan_windows) else None
        clip = None
        for ref in (compiled.files if compiled is not None else []):
            if ref.synthetic or ref.kind != KIND_AUDIO:
                continue
            if ref.audio_role != AUDIO_ROLE_LIP_SYNC:
                continue
            slot = socket_slot_of(ref.asset_id)
            raw = side.get(slot) if slot is not None else media.load_audio(
                ref.file, ref.trim_start, ref.trim_end)
            if raw is None:
                continue
            clip = media.audio_span(raw, compiled.start_seconds, seconds)
            break
        pieces.append((clip, seconds))

    real = [c for c, _ in pieces if c is not None]
    if not real:
        return None
    # Silence has to match the clips it is spliced between. concat_audio joins on
    # the sample axis and keeps the first chunk's rate, so filling a gap at some
    # default rate would leave the rest of the film playing at the wrong speed.
    sample_rate = int(real[0]["sample_rate"])
    channels = int(real[0]["waveform"].shape[1])
    joined = []
    for clip, seconds in pieces:
        if clip is None:
            joined.append(media.silent_audio(seconds, sample_rate, channels))
        elif int(clip["sample_rate"]) != sample_rate:
            # Different shots may be fed from different files. Resample rather
            # than concatenate two rates into one buffer.
            joined.append(media.resample_audio(clip, sample_rate))
        else:
            joined.append(clip)
    return media.concat_audio(joined)


class RenderOptions:
    """The `PulseRender` widgets, in one object."""

    __slots__ = ("cache_mode", "run_dir", "run_id", "save_segments", "low_memory",
                 "dry_run", "prune_unused", "want_frames", "use_reference_audio")

    def __init__(self, cache_mode=CACHE_AUTO, run_dir="pulseslate", run_id="",
                 save_segments=True, low_memory=True, dry_run=False,
                 prune_unused=False, want_frames=False, use_reference_audio=False):
        self.cache_mode = cache_mode
        self.run_dir = (run_dir or "pulseslate").strip() or "pulseslate"
        self.run_id = (run_id or "").strip()
        self.save_segments = bool(save_segments)
        self.low_memory = bool(low_memory)
        self.dry_run = bool(dry_run)
        self.prune_unused = bool(prune_unused)
        self.want_frames = bool(want_frames)
        self.use_reference_audio = bool(use_reference_audio)


class RenderResult:
    """What `PulseRender` hands back."""

    __slots__ = ("video", "frames", "audio", "segment_paths", "report", "warnings")

    def __init__(self, video=None, frames=None, audio=None, segment_paths="",
                 report="", warnings=()):
        self.video = video
        self.frames = frames
        self.audio = audio
        self.segment_paths = segment_paths
        self.report = report
        self.warnings = list(warnings)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sage_attention_global():
    """ComfyUI's own --use-sage-attention flag, which leaves no model_options
    trace because it patches attention process-wide. Absent on older cores."""
    try:
        import comfy.model_management as mm
        return bool(mm.sage_attention_enabled())
    except Exception:
        return False


def run(timeline_dict, side, model, vae, audio_vae, model_fl2va=None,
        options=None, sampler=None):
    """Render a whole timeline, reusing every window already on disk. Spec §7.4.

    `sampler` is injected so the entire loop -- cache decisions, write ordering,
    manifest durability, resume -- is testable on a machine with no GPU by
    handing in a stub. That is what tests/test_segment_cache.py does for each
    bullet of §7.5.
    """
    options = options or RenderOptions()
    sampler = sampler or sample
    windows_doc = timeline_dict.get("windows") or []
    plan_windows = side.plan.windows if side.plan is not None else []

    # ── identity (§7.1, §12.4) ──────────────────────────────────────────────
    model_fp = model_fingerprint(model)
    descriptor = describe_model_patches(
        model, sage_attention_global=_sage_attention_global())
    patch_fp = patch_fingerprint(descriptor)
    warnings = patch_warnings(descriptor)

    # ── run folder (§7.2) ───────────────────────────────────────────────────
    run_id = options.run_id or derive_run_id(timeline_dict)
    directory = os.path.join(folder_paths.get_output_directory(), options.run_dir, run_id)
    os.makedirs(directory, exist_ok=True)
    manifest = Manifest.load(directory, run_id=run_id,
                             node_version=timeline_dict.get("node_version", ""),
                             model_fingerprint=model_fp, created_utc=_now())

    # ── decide, for every window, before rendering any of them ──────────────
    decisions = {}
    keys = []
    for window in windows_doc:
        key = cache_key(timeline_dict, window, model_fp, patch_fp)
        window["cache_key"] = key
        keys.append(key)
        decisions[window["window_index"]] = plan_window(
            manifest, key, options.cache_mode, window["window_index"])

    stale = stale_entries(manifest, keys)
    if stale:
        # §14.5: never deleted automatically. Said out loud so the folder's size
        # is explained rather than mysterious.
        warnings.append(
            "%d segment(s) in this run folder belong to earlier edits and are kept "
            "(spec §14.5). Turn prune_unused on to delete them." % len(stale))

    vram_by_signature = {}
    for entry in manifest.segments:
        peak = entry.get("peak_vram_bytes") or 0
        if peak:
            signature = (entry.get("frames"), entry.get("width"), entry.get("height"))
            vram_by_signature[signature] = max(vram_by_signature.get(signature, 0), peak)

    def _report(dry):
        return build_report(
            timeline_dict, decisions=decisions, patch_descriptor=descriptor,
            patch_fingerprint=patch_fp, model_fingerprint=model_fp,
            seconds_per_frame=manifest.seconds_per_frame(),
            vram_by_signature=vram_by_signature, run_directory=directory,
            warnings=warnings, dry_run=dry)

    # ── dry run (§9) ────────────────────────────────────────────────────────
    if options.dry_run:
        # No sampling, no decode, no file writes -- not even the manifest. The
        # point of a dry run is to cost nothing and change nothing.
        return RenderResult(report=_report(True), segment_paths="", warnings=warnings)

    shifted = unpack(execute_node(MiniMaxH3SigmaShift, model=model,
                                  shift_video=side.shift_video,
                                  shift_audio=side.shift_audio))[0]
    shifted_fl2va = None
    if model_fl2va is not None:
        shifted_fl2va = unpack(execute_node(MiniMaxH3SigmaShift, model=model_fl2va,
                                            shift_video=side.shift_video,
                                            shift_audio=side.shift_audio))[0]

    # ── the loop ────────────────────────────────────────────────────────────
    carry_frame = carry_clip = carry_audio_clip = None
    video_files, audio_files, video_frames = [], [], []
    frame_chunks = []
    audio_chunks = []
    rendered = 0

    for position, window in enumerate(windows_doc):
        index = window["window_index"]
        decision = decisions[index]
        compiled = plan_windows[position] if position < len(plan_windows) else None
        paths = segment_paths(index, decision.key)
        absolute = {k: os.path.join(directory, v) for k, v in paths.items()}

        if decision.reuse:
            entry = decision.entry
            log.info("[PulseStudio] window %d/%d: reused from %s",
                     index + 1, len(windows_doc), entry.get("video_path"))
            video_files.append(manifest.resolve(entry, "video_path"))
            video_frames.append(entry.get("frames") or window["frames"])
            audio_path = manifest.resolve(entry, "audio_path")
            if audio_path and os.path.exists(audio_path):
                audio_files.append(audio_path)
            # A reused window decodes nothing, so the next window's carry-over
            # sockets have no tensors to fill. They are reconstructed from the
            # files this segment left behind -- exactly the ones the next window
            # asks for, and no more.
            following = (plan_windows[position + 1]
                         if position + 1 < len(plan_windows) else None)
            carry_frame, carry_clip, carry_audio_clip = carry_from_segment(
                manifest, entry,
                wants_video=window_wants(following, CARRY_VIDEO_ID),
                wants_audio=window_wants(following, CARRY_AUDIO_ID))
            if options.want_frames:
                chunk = media.read_segment_frames(video_files[-1])
                if chunk is not None:
                    frame_chunks.append(media.to_uint8(chunk) if options.low_memory else chunk)
            continue

        if compiled is None:  # pragma: no cover - plan/document mismatch
            raise RuntimeError(
                "window %d is in the timeline document but not in the compiled plan. "
                "Requeue the graph; if it persists, the PulseSlate node and this "
                "PulseRender are from different builds." % index)

        width, height = window["width"], window["height"]
        positive, latent = condition_window(
            compiled, side, width, height, carry_frame, carry_clip, carry_audio_clip,
            fps=window.get("fps") or 24)
        window_model = (shifted_fl2va
                        if (compiled.branch == BRANCH_FL2VA and shifted_fl2va is not None)
                        else shifted)
        if compiled.branch == BRANCH_FL2VA and shifted_fl2va is None:
            warnings.append(
                "window %d uses the fl2va branch but model_fl2va is not connected; it "
                "sampled on the reference checkpoint instead. Expect a soft seam."
                % index)

        log.info("[PulseStudio] window %d/%d: %s, %d frames, seed %d",
                 index + 1, len(windows_doc), compiled.branch, window["frames"],
                 window["seed"])

        media.reset_peak_vram()
        started = time.time()
        _, images, audio = sampler(
            window_model, positive, latent, window["seed"], window["steps"],
            window["sampler"], window["scheduler"], vae, audio_vae,
            cfg=window["cfg"], clip=side.clip)
        elapsed = time.time() - started
        peak = media.peak_vram_bytes()
        rendered += 1

        carry_frame = images[-1:]
        carry_clip = images[-min(images.shape[0], 48):]
        carry_audio_clip = audio

        if options.save_segments:
            # Media first, manifest second. See the module docstring.
            media.write_segment_video(absolute["video_path"], images, audio,
                                      fps=window["fps"])
            written_audio = media.write_segment_audio(absolute["audio_path"], audio)
            # §7.4 asks for this when continuity_out is last_frame_carry. It is
            # written unconditionally instead: it costs one PNG, and it is what
            # lets a later run resume mid-film after the user turns continuity on
            # -- which would otherwise force a full re-render to obtain a frame
            # that was sitting in VRAM at the time and was thrown away.
            media.write_png(absolute["last_frame_path"], carry_frame)

            entry = segment_entry(
                window, decision.key,
                {k: v for k, v in paths.items()
                 if k != "audio_path" or written_audio},
                render_seconds=elapsed,
                warmup=(rendered == 1),
                patch_fingerprint=patch_fp)
            entry["peak_vram_bytes"] = peak
            manifest.upsert(entry)
            manifest.save(_now())
            video_files.append(absolute["video_path"])
            video_frames.append(window["frames"])
            if written_audio:
                audio_files.append(absolute["audio_path"])
        else:
            audio_chunks.append(audio)

        if options.want_frames:
            frame_chunks.append(media.to_uint8(images) if options.low_memory else images)

        del images
        if options.low_memory:
            _release()

    # ── assembly (§8) ───────────────────────────────────────────────────────
    #
    # Everything below this line is convenience. The render itself is already
    # complete and durable -- every segment is on disk and recorded in the
    # manifest -- so no failure here may be allowed to raise. Losing hours of GPU
    # time to a container-format problem in the last thirty seconds is the single
    # worst thing this node could do.
    # Audio first: the assembled video needs it. Each segment's mp4 carries its
    # own AAC stream, but AAC opens on a priming delay that cannot be
    # concatenated away -- so the finished track is joined from the lossless
    # per-segment FLACs instead and muxed in once.
    try:
        if audio_files:
            audio_out = media.concat_audio([media.load_audio(p) for p in audio_files])
        else:
            audio_out = media.concat_audio(audio_chunks)
    except Exception as exc:  # pragma: no cover - decode failure
        log.error("[PulseStudio] could not join the segment audio (%s)", exc)
        audio_out = None

    if options.use_reference_audio:
        # The generated track is still on disk in every segment's .flac, so this
        # is a mux choice rather than a loss -- flip the widget and requeue and
        # the cache hands back the same segments with the other track.
        try:
            supplied = reference_audio_track(plan_windows, windows_doc, side)
        except Exception as exc:  # pragma: no cover - decode failure
            log.error("[PulseStudio] could not build the reference audio track (%s)", exc)
            supplied = None
        if supplied is not None:
            audio_out = supplied
            log.info("[PulseStudio] using the supplied lip-sync audio; H3's generated "
                     "track is still in each segment's .flac")
        else:
            warnings.append(
                "use_reference_audio is on, but no shot carries a lip_sync reference "
                "audio, so the film kept the audio H3 generated.")

    video = None
    if video_files:
        final_path = os.path.join(directory, "assembled.mp4")
        try:
            joined = media.concat_videos(
                video_files, final_path, frame_counts=video_frames,
                fps=(windows_doc[0].get("fps") if windows_doc else 24) or 24,
                audio=audio_out)
            if joined:
                video = media.video_from_file(joined)
        except Exception as exc:  # pragma: no cover - container-level failure
            log.error("[PulseStudio] assembly failed (%s). Every segment is intact "
                      "in %s; requeue to reuse them and retry.", exc, directory)
            warnings.append(
                "The segments rendered and are safe in %s, but joining them into "
                "one file failed: %s. Requeue -- every window will be reused from "
                "the cache rather than re-rendered." % (directory, exc))

    frames_out = None
    if frame_chunks:
        stacked = torch.cat(frame_chunks, dim=0)
        frames_out = media.from_uint8(stacked) if options.low_memory else stacked

    if options.prune_unused and stale:
        _prune(manifest, stale, directory)
        manifest.save(_now())

    report = _report(False)
    listing = "\n".join(video_files)
    log.info("[PulseStudio] %d window(s) rendered, %d reused -> %s",
             rendered, len(windows_doc) - rendered, directory)
    return RenderResult(video=video, frames=frames_out, audio=audio_out,
                        segment_paths=listing, report=report, warnings=warnings)


def _release():
    """Hand freed VRAM back between windows.

    A twelve-window run allocates and frees the same shapes twelve times; without
    this the allocator's cache grows to the high-water mark of the largest window
    and stays there, which on a 32 GB card is the difference between finishing and
    an out-of-memory error on window nine.
    """
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:  # pragma: no cover - no comfy in the test env
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _prune(manifest, stale, directory):
    """Delete segments the current timeline no longer references. Opt-in only (§14.5)."""
    keys = {e.get("cache_key") for e in stale}
    for entry in stale:
        for field in ("video_path", "audio_path", "last_frame_path"):
            rel = entry.get(field)
            if not rel:
                continue
            try:
                os.unlink(os.path.join(directory, rel))
            except OSError:
                pass
    manifest.data["segments"] = [e for e in manifest.segments
                                 if e.get("cache_key") not in keys]
    log.info("[PulseStudio] pruned %d stale segment(s)", len(stale))
