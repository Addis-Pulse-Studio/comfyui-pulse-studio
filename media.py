"""File loading and tensor fitting for the Pulse Studio nodes.

Deliberately separate from comfyui_pulse_studio/: everything in that package is stdlib
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
        log.warning("[PulseStudio] image not found: %s", path)
        return None
    try:
        with Image.open(resolved) as im:
            arr = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as exc:
        log.warning("[PulseStudio] could not load image %s: %s", path, exc)
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
        log.warning("[PulseStudio] video not found: %s", path)
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
        log.warning("[PulseStudio] video decode failed for %s: %s", path, exc)
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
        log.warning("[PulseStudio] audio not found: %s", path)
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
        log.warning("[PulseStudio] audio decode failed for %s: %s", path, exc)
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


def audio_span(audio, start_seconds, seconds):
    """`seconds` of an AUDIO dict beginning at `start_seconds`, zero-padded if short.

    A lip-sync reference has to describe the same span of time as the window it
    rides in. The DiT packs the reference audio rows against the target audio
    grid, so a clip of a different length is asking the model to align two
    different stretches of time -- which is the failure that reads as "lip sync
    does not work" rather than as any kind of error.

    Padding rather than truncating the window is deliberate: a recording that
    runs out mid-window should leave the mouth still, not shorten the shot.
    """
    from .comfyui_pulse_studio.concat import audio_span_bounds

    if audio is None:
        return None
    sr = int(audio["sample_rate"])
    waveform = audio["waveform"]
    start, stop, pad = audio_span_bounds(waveform.shape[-1], sr, start_seconds, seconds)
    piece = waveform[..., start:stop]
    if pad:
        piece = torch.cat(
            [piece, piece.new_zeros(piece.shape[:-1] + (pad,))], dim=-1)
    return {"waveform": piece, "sample_rate": sr}


def concat_audio(chunks):
    """Concatenate AUDIO dicts along time. Returns None for an empty list."""
    chunks = [c for c in chunks if c is not None]
    if not chunks:
        return None
    sr = chunks[0]["sample_rate"]
    return {"waveform": torch.cat([c["waveform"] for c in chunks], dim=-1), "sample_rate": sr}


def resample_audio(audio, sample_rate):
    """An AUDIO dict at `sample_rate`, using torchaudio when it is available.

    Linear interpolation is the fallback rather than the default: it is audibly
    worse than a windowed-sinc resampler, and this runs on a voice track the user
    supplied and expects to hear back. torchaudio ships with every working
    ComfyUI, so the fallback is for the headless test path.
    """
    if audio is None:
        return None
    src = int(audio["sample_rate"])
    dst = int(sample_rate)
    if src == dst:
        return audio
    waveform = audio["waveform"]
    try:
        import torchaudio
        out = torchaudio.functional.resample(waveform, src, dst)
    except Exception:  # pragma: no cover - torchaudio absent
        n = max(1, int(round(waveform.shape[-1] * dst / float(src))))
        out = torch.nn.functional.interpolate(
            waveform.reshape(-1, 1, waveform.shape[-1]).float(),
            size=n, mode="linear", align_corners=False).reshape(
                waveform.shape[:-1] + (n,))
    return {"waveform": out, "sample_rate": dst}


def silent_audio(seconds, sample_rate=44100, channels=2):
    """`seconds` of silence, so a window with no supplied audio still takes up its
    own length in a concatenated track rather than pulling everything after it
    forward."""
    n = max(0, int(round(float(seconds) * sample_rate)))
    return {"waveform": torch.zeros((1, channels, n)), "sample_rate": sample_rate}


def empty_audio(sample_rate=44100):
    return {"waveform": torch.zeros((1, 2, 0)), "sample_rate": sample_rate}


def empty_images(width=64, height=64):
    return torch.zeros((0, height, width, 3))


# ── content digests (spec §7.1) ─────────────────────────────────────────────
#
# A reference arriving through a socket has no file to hash, so its identity is
# the bytes of the tensor. This is what lets the segment cache tell "the same
# picture, re-wired" from "a different picture in the same slot" -- the first must
# reuse the cached segment and the second must not.

def tensor_digest(tensor):
    """sha256 of an IMAGE tensor's bytes, 16 hex chars. Stable across processes."""
    import hashlib

    if tensor is None:
        return ""
    array = tensor.detach().to("cpu").contiguous().numpy()
    digest = hashlib.sha256()
    # Shape rides into the hash: two tensors with identical bytes at different
    # shapes are different pictures, and .tobytes() alone would not say so.
    digest.update(("%s|%s|" % (array.dtype.str, array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def audio_digest(audio):
    """sha256 of an AUDIO dict, 16 hex chars."""
    import hashlib

    if audio is None:
        return ""
    digest = hashlib.sha256()
    digest.update(("%d|" % int(audio.get("sample_rate") or 0)).encode("utf-8"))
    waveform = audio.get("waveform")
    if waveform is not None:
        array = waveform.detach().to("cpu").contiguous().numpy()
        digest.update(("%s|%s|" % (array.dtype.str, array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def file_digest(path):
    """sha256 of a file's contents, 16 hex chars. Empty string when unreadable."""
    import hashlib

    resolved = resolve_path(path)
    if not resolved:
        return ""
    digest = hashlib.sha256()
    try:
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()[:16]


# ── 8-bit frame accumulation (spec §8) ──────────────────────────────────────

def to_uint8(images):
    """Float IMAGE tensor -> uint8, for accumulating an assembled frame stack.

    §8: a 12-window timeline held as float32 is four times the RAM of the same
    frames held as bytes, and the frames are 0-255 data the moment they leave the
    VAE. The conversion back happens once, at the encode boundary.
    """
    if images is None:
        return None
    return (images.detach().to("cpu").clamp(0, 1) * 255).round().to(torch.uint8)


def from_uint8(images):
    """uint8 frame stack -> float IMAGE tensor, at the encode boundary."""
    if images is None:
        return None
    return images.to(torch.float32) / 255.0


# ── segment IO (spec §7.2, §7.4) ────────────────────────────────────────────

def _fsync_path(path):
    """Force one file's bytes to disk.

    §7.4 requires the manifest to be written only after the media files are
    closed, so that a crash between the two cannot leave a manifest claiming a
    file that is absent. Closing a file is not the same as its bytes reaching the
    platter, so the promise needs this.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - filesystems that refuse
        pass
    finally:
        os.close(fd)


def _video_api():
    """ComfyUI's own video encoder. Raises with a readable message if too old."""
    try:
        from comfy_api.input_impl import VideoFromComponents, VideoFromFile
        from comfy_api.util import VideoComponents
    except ImportError as exc:  # pragma: no cover - old ComfyUI
        raise RuntimeError(
            "PulseRender writes segments through ComfyUI's own video API "
            "(comfy_api.input_impl), which this build does not have (%s). Update "
            "ComfyUI, or turn save_segments off to render without the cache." % (exc,))
    return VideoFromComponents, VideoFromFile, VideoComponents


def video_from_file(path):
    """Wrap a file on disk as a VIDEO output."""
    _, VideoFromFile, _ = _video_api()
    return VideoFromFile(path)


def write_segment_video(path, images, audio, fps=24):
    """Encode one window to `path` as mp4, video and audio muxed together.

    The muxed audio is what makes a segment independently playable -- a user
    checking whether window 7 is worth re-rendering opens the file and watches it.
    The lossless copy written alongside by `write_segment_audio` is what the final
    assembly uses.
    """
    from fractions import Fraction

    VideoFromComponents, _, VideoComponents = _video_api()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    components = VideoComponents(images=images, audio=audio, frame_rate=Fraction(int(fps)))
    VideoFromComponents(components).save_to(path)
    _fsync_path(path)
    return path


def write_segment_audio(path, audio):
    """Write a window's audio to `path` as FLAC. Lossless, so repeated assembly
    of the same cached segments produces the same track every time."""
    import av

    if audio is None or audio["waveform"].shape[-1] == 0:
        return ""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    waveform = audio["waveform"][0].detach().to("cpu").contiguous()  # [C, N]
    sample_rate = int(audio["sample_rate"])
    layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[0], "stereo")

    with av.open(path, mode="w") as container:
        stream = container.add_stream("flac", rate=sample_rate)
        stream.format = "s32"  # FLAC is integer-only; s32 encodes as 24-bit
        frame = av.AudioFrame.from_ndarray(
            waveform.numpy().astype(np.float32), format="fltp", layout=layout)
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    _fsync_path(path)
    return path


def write_png(path, image):
    """Write a single IMAGE frame as PNG -- the `last_frame_carry` handoff (§7.4)."""
    from PIL import Image

    if image is None:
        return ""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame = image[0] if image.dim() == 4 else image
    array = (frame.detach().to("cpu").clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(array, mode="RGB").save(path)
    _fsync_path(path)
    return path


def read_png(path):
    """Read a PNG back as an IMAGE tensor [1, H, W, 3]."""
    from PIL import Image

    if not path or not os.path.exists(path):
        return None
    with Image.open(path) as im:
        array = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def read_segment_frames(path):
    """Decode a whole cached segment back to an IMAGE tensor.

    Only called when the `frames` output is actually connected (§8). The default
    path never does this -- the final video is assembled by remuxing the segment
    files, so a reused window's pixels need never enter RAM at all.
    """
    if not path or not os.path.exists(path):
        return None
    return load_video(path, 0.0, None, max_frames=MAX_SEGMENT_FRAMES)


#: Above H3's 362-frame ceiling with room to spare; a segment cannot be longer.
MAX_SEGMENT_FRAMES = 400


def concat_videos(paths, out_path, frame_counts=None, fps=24, audio=None):
    """Join segment files into one video. Spec §8.

    The video is stream-copied: no decode, no re-encode, no frame stack, so a
    twelve-window assembly costs a constant, small amount of RAM regardless of how
    long the film is. `frame_counts` places each segment exactly, and `audio` is
    the finished track -- joined from the segments' lossless FLACs by the caller
    and encoded once here. See `_remux_concat` for why the audio is not copied.

    Every segment was encoded by `write_segment_video` with identical parameters,
    which is what makes the video copy legal. Returns the output path, or "" if
    there was nothing to join or the join failed.
    """
    paths = [p for p in (paths or []) if p and os.path.exists(p)]
    if not paths:
        return ""
    if frame_counts and len(frame_counts) == len(paths):
        # The caller knows exactly, from the manifest. Preferred: it needs no
        # second pass over the files and cannot disagree with what was rendered.
        frame_counts = list(frame_counts)
    else:
        frame_counts = [_count_frames(p) for p in paths]
        if any(not n for n in frame_counts):
            log.error("[PulseStudio] could not determine the frame count of every "
                      "segment, so they cannot be placed exactly")
            return ""

    if len(paths) == 1:
        # One window: no join to do, and nothing to gain from re-encoding its
        # audio. Copied rather than referenced so the output is a file in its own
        # right and deleting the cache cannot empty it.
        import shutil

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        shutil.copyfile(paths[0], out_path)
        _fsync_path(out_path)
        return out_path

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        _remux_concat(paths, out_path, frame_counts, fps=fps, audio=audio)
    except Exception as exc:
        # A muxing failure at the assembly step must never destroy a render that
        # has already cost hours. Every segment is on disk and in the manifest, so
        # the work survives; only the convenience file does not.
        log.error("[PulseStudio] could not join the segments into one file (%s). "
                  "The individual segments are intact in the run folder and can be "
                  "joined with any tool; requeueing will reuse them and try again.",
                  exc)
        try:
            if os.path.exists(out_path):
                os.unlink(out_path)
        except OSError:  # pragma: no cover
            pass
        return ""
    _fsync_path(out_path)
    return out_path


def _rescale(value, source_tb, target_tb):
    """A timestamp moved from one time base to another, or None."""
    from fractions import Fraction

    if value is None:
        return None
    if not source_tb or not target_tb or source_tb == target_tb:
        return int(value)
    return int(round(int(value) * (Fraction(source_tb) / Fraction(target_tb))))


def _encode_audio_packets(output, audio):
    """Encode a whole AUDIO dict into AAC packets for the assembled file.

    The audio is rebuilt from the segments' lossless FLACs rather than copied out
    of their mp4s, which is why the seam is clean: a per-segment AAC stream opens
    on a priming delay that cannot be concatenated away, whereas the waveform can
    simply be joined.

    Audio is small -- a three-minute stereo track is tens of megabytes -- so
    holding the encoded packets is affordable in a way that holding the frames
    never is.
    """
    import av

    if audio is None or audio["waveform"].shape[-1] == 0:
        return None, []

    waveform = audio["waveform"][0].detach().to("cpu").contiguous()
    sample_rate = int(audio["sample_rate"])
    layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[0], "stereo")

    stream = output.add_stream("aac", rate=sample_rate, layout=layout)
    frame = av.AudioFrame.from_ndarray(
        waveform.numpy().astype(np.float32), format="fltp", layout=layout)
    frame.sample_rate = sample_rate
    frame.pts = 0

    packets = list(stream.encode(frame))
    packets.extend(stream.encode(None))
    return stream, packets


def _packet_seconds(packet):
    """A packet's decode time in seconds, for interleaving. Never None."""
    stamp = packet.dts if packet.dts is not None else packet.pts
    if stamp is None:
        return 0.0
    return float(stamp) * float(packet.time_base or 0 or 1)


def _count_frames(path):
    """Video frames in a segment, for a caller that did not say. 0 if unknown.

    The container's own count first; failing that, one demux pass counting
    packets, which is still cheap because nothing is decoded.
    """
    import av

    try:
        with av.open(path) as source:
            if not source.streams.video:
                return 0
            stream = source.streams.video[0]
            if stream.frames:
                return int(stream.frames)
            return sum(1 for packet in source.demux(stream) if packet.dts is not None)
    except Exception as exc:  # pragma: no cover - unreadable segment
        log.warning("[PulseStudio] could not count frames in %s: %s", path, exc)
        return 0


def _remux_concat(paths, out_path, frame_counts, fps=24, audio=None):
    """Join segments: video copied packet for packet, audio rebuilt from the FLACs.

    The video is never decoded -- packets are read and written straight out with
    their timestamps shifted -- which is what keeps assembling a twelve-window
    film inside a constant, small amount of RAM (§8). Each segment is placed at
    `frames-so-far / fps`, exactly, so the seams are frame-accurate.

    The audio is *not* copied. Each segment's mp4 carries its own AAC stream, and
    AAC opens on a priming delay that cannot be concatenated away: packing those
    streams nose to tail leaves a hole at every seam, or overlaps their decode
    timestamps and is rejected outright. The lossless per-segment FLAC exists for
    exactly this -- joined as a waveform it has no seam at all -- so the caller
    hands the finished track in and it is encoded once here.

    See `comfyui_pulse_studio.concat` for the placement arithmetic and its tests.
    """
    import av

    from .comfyui_pulse_studio.concat import video_is_gapless, video_shifts

    if not frame_counts or len(frame_counts) != len(paths):
        raise RuntimeError("a frame count is required per segment to place it exactly")

    with av.open(out_path, mode="w") as output:
        video_out = None
        audio_packets = []
        audio_index = 0
        shifts = None
        frame_duration = None

        for index, path in enumerate(paths):
            with av.open(path) as source:
                if not source.streams.video:
                    raise RuntimeError("segment %s carries no video stream" % path)
                video_in = source.streams.video[0]

                if video_out is None:
                    video_out = output.add_stream_from_template(video_in)
                    _, audio_packets = _encode_audio_packets(output, audio)
                    # Force the header out before any time base is read:
                    # libavformat may settle on a different one than the template's.
                    try:
                        output.start_encoding()
                    except Exception:  # pragma: no cover - older PyAV
                        pass

                    time_base = float(video_out.time_base or video_in.time_base)
                    shifts = video_shifts(frame_counts, fps, time_base)
                    frame_duration = int(round((1.0 / float(fps)) / time_base))
                    if not video_is_gapless(frame_counts, fps, time_base, shifts,
                                            frame_duration):  # pragma: no cover
                        raise RuntimeError(
                            "the computed segment placement would leave a gap at a "
                            "seam; refusing to write a stuttering assembly")
                    audio_packets.sort(key=_packet_seconds)

                source_tb = video_in.time_base
                target_tb = video_out.time_base or source_tb
                shift = shifts[index]

                for packet in source.demux(video_in):
                    if packet.dts is None:
                        continue
                    packet.dts = _rescale(packet.dts, source_tb, target_tb) + shift
                    packet.pts = ((_rescale(packet.pts, source_tb, target_tb) + shift)
                                  if packet.pts is not None else packet.dts)
                    packet.duration = _rescale(packet.duration, source_tb,
                                               target_tb) or frame_duration
                    packet.stream = video_out
                    try:
                        packet.time_base = target_tb
                    except (AttributeError, TypeError):  # pragma: no cover
                        pass

                    # Keep the two streams interleaved as they are written, rather
                    # than handing the muxer every video packet and then every
                    # audio packet and asking it to buffer the difference.
                    now = _packet_seconds(packet)
                    while (audio_index < len(audio_packets)
                           and _packet_seconds(audio_packets[audio_index]) <= now):
                        output.mux(audio_packets[audio_index])
                        audio_index += 1
                    output.mux(packet)

        for packet in audio_packets[audio_index:]:
            output.mux(packet)
    return out_path


# ── VRAM accounting (spec §9.6, §12.7) ──────────────────────────────────────

def reset_peak_vram():
    """Zero the peak-allocation counter before a window renders."""
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover - no CUDA in the test env
        pass


def peak_vram_bytes():
    """Peak CUDA allocation since the last reset, or 0 where there is no CUDA.

    Recorded per segment so §12.7's question -- which patch chain actually fits
    and which is actually faster on *this* box -- is answered by a lookup instead
    of by an argument.
    """
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:  # pragma: no cover - no CUDA in the test env
        pass
    return 0
