"""File loading and tensor fitting for the Omni-Director nodes.

Deliberately separate from omni_director/: everything in that package is stdlib
only and runs without torch, which is what lets the compiler, the frame grid, the
asset bin and the scissor be tested headless. This module is where torch, PIL and
PyAV enter, and nothing in the tested core imports it.
"""

import logging
import os

import numpy as np
import torch

log = logging.getLogger(__name__)

try:  # ComfyUI runtime
    import folder_paths
except ImportError:  # pragma: no cover - lets the module be imported standalone
    folder_paths = None


# ── path resolution ─────────────────────────────────────────────────────────

def resolve_path(rel):
    """Find an uploaded asset in ComfyUI's input directory.

    Assets are uploaded through ComfyUI's own /upload/image endpoint, which
    accepts any file type despite the name, so videos and audio land in the same
    tree as images. Subdirectories are searched because the upload endpoint
    honours a subfolder parameter.
    """
    if not rel:
        return ""
    if os.path.isabs(rel) and os.path.exists(rel):
        return rel
    if folder_paths is None:
        return rel if os.path.exists(rel) else ""
    input_dir = folder_paths.get_input_directory()
    candidates = [
        os.path.join(input_dir, rel),
        os.path.join(input_dir, os.path.basename(rel)),
        os.path.join(input_dir, "omnidirector", os.path.basename(rel)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


# ── images ──────────────────────────────────────────────────────────────────

def load_image(path):
    """Load an image as an IMAGE tensor [1, H, W, 3] in 0..1."""
    from PIL import Image

    resolved = resolve_path(path)
    if not resolved:
        log.warning("[OmniDirector] image not found: %s", path)
        return None
    try:
        with Image.open(resolved) as im:
            arr = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as exc:
        log.warning("[OmniDirector] could not load image %s: %s", path, exc)
        return None


def fit_image(tensor, width, height, method="crop"):
    """Resize an [N,H,W,C] IMAGE tensor to exactly (height, width).

    Done explicitly rather than left to H3's internal preprocessing, which
    stretches on an aspect mismatch. 'crop' scales to cover then centre-crops,
    'pad' scales to fit then letterboxes, 'stretch' ignores aspect entirely.
    'pad' is rarely right for references -- the bars become reference content.
    """
    if tensor is None:
        return None
    n, h, w, c = tensor.shape
    if h == height and w == width:
        return tensor
    chw = tensor.permute(0, 3, 1, 2)
    if method == "stretch":
        out = torch.nn.functional.interpolate(chw, size=(height, width),
                                              mode="bilinear", align_corners=False)
        return out.permute(0, 2, 3, 1).clamp(0, 1)

    scale = (max(width / w, height / h) if method == "crop" else min(width / w, height / h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    out = torch.nn.functional.interpolate(chw, size=(nh, nw), mode="bilinear", align_corners=False)
    out = out.permute(0, 2, 3, 1).clamp(0, 1)

    if method == "crop":
        top, left = max(0, (nh - height) // 2), max(0, (nw - width) // 2)
        return out[:, top:top + height, left:left + width, :]
    canvas = torch.zeros((n, height, width, c), dtype=out.dtype)
    top, left = max(0, (height - nh) // 2), max(0, (width - nw) // 2)
    canvas[:, top:top + nh, left:left + nw, :] = out
    return canvas


# ── video ───────────────────────────────────────────────────────────────────

def load_video(path, trim_start=0.0, trim_end=None, max_frames=400):
    """Decode [trim_start, trim_end) of a video into an IMAGE tensor [T,H,W,3]."""
    import av

    resolved = resolve_path(path)
    if not resolved:
        log.warning("[OmniDirector] video not found: %s", path)
        return None
    start = float(trim_start or 0.0)
    frames = []
    try:
        with av.open(resolved) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            # Seek slightly early: seeking lands on the nearest keyframe, and
            # decoding forward from there is what makes the trim frame-accurate.
            if stream.time_base:
                container.seek(int(max(0.0, start - 0.5) / float(stream.time_base)),
                               stream=stream, backward=True)
            for frame in container.decode(stream):
                t = frame.time
                if t is None and frame.pts is not None and stream.time_base:
                    t = float(frame.pts * stream.time_base)
                t = 0.0 if t is None else t
                if t < start - 0.01:
                    continue
                if trim_end is not None and t > float(trim_end):
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= max_frames:
                    break
    except Exception as exc:
        log.warning("[OmniDirector] video decode failed for %s: %s", path, exc)
        return None
    if not frames:
        return None
    return torch.from_numpy(np.array(frames, dtype=np.float32) / 255.0)


# ── audio ───────────────────────────────────────────────────────────────────

def load_audio(path, trim_start=0.0, trim_end=None, sample_rate=44100):
    """Decode [trim_start, trim_end) of an audio (or video) file into an AUDIO dict.

    PyAV does not care whether the container is labelled video or audio, so this
    also pulls a video file's embedded soundtrack -- which is exactly what a
    reference video's paired <Audio j> needs.
    """
    import av

    resolved = resolve_path(path)
    if not resolved:
        log.warning("[OmniDirector] audio not found: %s", path)
        return None
    try:
        chunks = []
        with av.open(resolved) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
            for frame in container.decode(stream):
                for rf in resampler.resample(frame):
                    chunks.append(torch.from_numpy(rf.to_ndarray()))
            for rf in resampler.resample(None):
                chunks.append(torch.from_numpy(rf.to_ndarray()))
        if not chunks:
            return None
        waveform = torch.cat(chunks, dim=1)  # [channels, samples]
        a = max(0, min(int(float(trim_start or 0.0) * sample_rate), waveform.shape[1]))
        b = (waveform.shape[1] if trim_end is None
             else max(a, min(int(float(trim_end) * sample_rate), waveform.shape[1])))
        trimmed = waveform[:, a:b]
        if trimmed.shape[1] == 0:
            return None
        return {"waveform": trimmed.unsqueeze(0), "sample_rate": sample_rate}
    except Exception as exc:
        log.warning("[OmniDirector] audio decode failed for %s: %s", path, exc)
        return None


def audio_tail(audio, seconds):
    """The last `seconds` of an AUDIO dict -- the window carry-over signal.

    H3 treats every ref_audio as a short reference clip, so the whole previous
    window is both wasteful and out of range; the tail is what carries the score
    and ambience forward.
    """
    if audio is None:
        return None
    sr = audio["sample_rate"]
    n = min(audio["waveform"].shape[-1], max(1, int(float(seconds) * sr)))
    return {"waveform": audio["waveform"][..., -n:], "sample_rate": sr}


def concat_audio(chunks):
    """Concatenate AUDIO dicts along time. Returns None for an empty list."""
    chunks = [c for c in chunks if c is not None]
    if not chunks:
        return None
    sr = chunks[0]["sample_rate"]
    return {"waveform": torch.cat([c["waveform"] for c in chunks], dim=-1), "sample_rate": sr}


def empty_audio(sample_rate=44100):
    return {"waveform": torch.zeros((1, 2, 0)), "sample_rate": sample_rate}


def empty_images(width=64, height=64):
    return torch.zeros((0, height, width, 3))
