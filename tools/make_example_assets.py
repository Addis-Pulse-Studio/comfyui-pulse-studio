#!/usr/bin/env python3
"""Generate the placeholder references PulseSlate_Cast.json opens on.

WHY THESE ARE GENERATED AND NOT COMMITTED AS ART

`tests/test_shipped_assets.py` exists because the starter graph once shipped four
`Generated Image ....jpg` references buried inside `timeline_data` -- client work,
one `git push` from being published. The rule that came out of it is that a
shipped workflow may only name placeholder assets the user supplies themselves.

A graph whose whole subject is the asset bin cannot honour that rule and still
demonstrate anything, so the pack supplies the placeholders. Generating them from
this script rather than committing four opaque binaries means anyone can see
exactly what is in them -- flat tones and a sine tone, no photograph of anyone, no
model output, nothing with a provenance question attached.

Every name starts with `example_`, which is the convention
`test_every_media_filename_is_a_placeholder` enforces, and carries `pulse_` after
it so that installing them into a shared ComfyUI `input/` cannot quietly collide
with a file the user already had.

Stdlib only: `zlib` and `struct` write the PNGs, `wave` writes the WAV. No Pillow,
no numpy, so this runs in the same bare environment as the test suite.

    python3 tools/make_example_assets.py

Rerunning is idempotent -- the output is a pure function of the table below.
"""

import math
import os
import struct
import wave
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "example_workflows", "assets")

WIDTH, HEIGHT = 336, 192          # 16:9-ish, small; these are placeholders
SAMPLE_RATE = 44100

#: name -> (top colour, bottom colour, stripe colour). Deliberately flat and
#: obviously synthetic: nobody should mistake one of these for a reference they
#: are meant to keep.
IMAGES = {
    "example_pulse_subject_a.png": ((196, 88, 74), (58, 32, 34), (232, 196, 120)),
    "example_pulse_subject_b.png": ((74, 118, 168), (26, 34, 54), (150, 210, 232)),
    "example_pulse_place.png": ((92, 104, 88), (24, 30, 26), (206, 198, 150)),
}

#: A one-second 220 Hz tone under a slow tremolo. Long enough for the lip_sync
#: path to have something to trim to a window, short enough to be a few KB.
AUDIO_NAME = "example_pulse_voice.wav"
AUDIO_SECONDS = 1.0
AUDIO_HZ = 220.0


def _chunk(tag, payload):
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def write_png(path, rows):
    """Minimal 8-bit RGB PNG. `rows` is height lists of (r, g, b) triples."""
    raw = b"".join(b"\x00" + bytes(v for pixel in row for v in pixel) for row in rows)
    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(_chunk(b"IHDR", header))
        handle.write(_chunk(b"IDAT", zlib.compress(raw, 9)))
        handle.write(_chunk(b"IEND", b""))


def gradient(top, bottom, stripe):
    """A vertical gradient with one horizontal band, so orientation is visible."""
    rows = []
    for y in range(HEIGHT):
        t = y / float(HEIGHT - 1)
        base = tuple(int(round(top[c] + (bottom[c] - top[c]) * t)) for c in range(3))
        band = abs(y - HEIGHT // 3) < 4
        rows.append([stripe if band else base] * WIDTH)
    return rows


def write_wav(path):
    total = int(SAMPLE_RATE * AUDIO_SECONDS)
    samples = []
    for n in range(total):
        t = n / float(SAMPLE_RATE)
        envelope = 0.5 * (1.0 - math.cos(2 * math.pi * min(t / 0.05, 1.0) / 2))
        tremolo = 0.75 + 0.25 * math.sin(2 * math.pi * 5.0 * t)
        value = int(18000 * envelope * tremolo * math.sin(2 * math.pi * AUDIO_HZ * t))
        samples.append(max(-32768, min(32767, value)))
    frames = struct.pack("<%dh" % total, *samples)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(frames)


def main(out_dir=None, quiet=False):
    """Write every placeholder. `out_dir` lets the test regenerate elsewhere."""
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, (top, bottom, stripe) in IMAGES.items():
        path = os.path.join(out_dir, name)
        write_png(path, gradient(top, bottom, stripe))
        written.append(path)
    path = os.path.join(out_dir, AUDIO_NAME)
    write_wav(path)
    written.append(path)
    if not quiet:
        for path in written:
            print("%s  %d bytes" % (os.path.basename(path), os.path.getsize(path)))
    return written


if __name__ == "__main__":
    main()
