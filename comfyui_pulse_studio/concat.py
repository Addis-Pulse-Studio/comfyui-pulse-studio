"""Where each rendered segment sits in the assembled film. Spec §8.

This is the arithmetic half of `media.concat_videos`, split out here for one
reason: it is the part that was wrong twice, and it is the part that can be
tested without PyAV, an encoder or a GPU.

WHY THE PLACEMENT IS COMPUTED AND NOT MEASURED
----------------------------------------------
The obvious implementation reads each segment's timestamps and packs the next one
in behind them. It does not work, and it fails in two different ways in
succession -- both of them observed on a real two-window render.

*Attempt one: carry on from the last DTS seen on each stream.* This ignores where
a stream **began**. Segments do not begin at zero: video opens on a negative
decode timestamp because of B-frame reordering, and AAC opens earlier still
because of its priming delay. On a real segment both opened at `dts = -1024`, but
in different time bases -- 1/12288 and 1/32000 -- so that is -83ms of video
against -32ms of audio. Measuring the audio's length from zero understated it by
its priming delay, the next segment's audio was placed at a timestamp already
used, and libavformat rejected the non-monotonic DTS with EINVAL after both
windows had rendered.

*Attempt two: shift each segment by its full DTS extent.* Monotonic, and still
wrong: a segment's decode extent is **longer than its content**, precisely because
of those negative starts. Packing nose to tail on DTS spaced the segments 84ms
further apart than their duration -- two dropped frames of dead air at every seam,
which no timestamp assertion catches and which is obvious the moment you watch it.

So the placement is not derived from timestamps at all. Every segment is exactly
`frames` frames at a known fps, so segment *i* belongs at `sum(frames before it) /
fps` seconds and nowhere else. That is exact, gapless, and independent of whatever
the encoder did with its priming delays.

The audio is not *placed* by this module. It is rebuilt from the segments'
lossless per-segment FLACs as one continuous waveform, which has no priming delay
to repeat and no timestamps to align -- see `media._remux_concat`.

That is a statement about the container, and it used to be the whole story. It is
not: the two sides of an audio seam are two independently generated scores, and
nothing about a correct timestamp makes them meet. The seam arithmetic further
down is about that, and only that.

Pure stdlib. `tests/test_concat.py` drives it with numbers read out of real
segment files.
"""

import math

__all__ = ["video_shifts", "video_is_gapless", "frame_duration",
           "audio_span_bounds", "seam_dip_samples", "seam_dip_gains",
           "seam_gain_match", "SEAM_FADE_MS_DEFAULT",
           "SEAM_FADE_MS_MIN", "SEAM_FADE_MS_MAX"]


def frame_duration(fps, time_base):
    """How many stream units one frame occupies. 512, for 24fps at 1/12288."""
    return int(round((1.0 / float(fps)) / float(time_base)))


def video_shifts(frame_counts, fps, time_base):
    """Where each segment's video starts, in the video stream's own units.

    The returned shift is added to a packet's pts *and* its dts alike, so the gap
    between the two -- which is what encodes B-frame ordering -- is untouched.
    """
    shifts = []
    elapsed = 0
    for frames in frame_counts:
        shifts.append(int(round((elapsed / float(fps)) / float(time_base))))
        elapsed += int(frames)
    return shifts


def video_is_gapless(frame_counts, fps, time_base, shifts, frame_units=None):
    """Does each segment begin exactly where the previous one ended?

    A gap shows as a freeze at the seam and an overlap as a stutter. Both survive
    every timestamp assertion and are obvious on playback, so this is checked
    before a byte is written rather than discovered afterwards.
    """
    if frame_units is None:
        frame_units = frame_duration(fps, time_base)
    for index in range(1, len(shifts)):
        previous_end = shifts[index - 1] + int(frame_counts[index - 1]) * frame_units
        if shifts[index] != previous_end:
            return False
    return True


def audio_span_bounds(total_samples, sample_rate, start_seconds, seconds):
    """Where a lip-sync reference is cut for one window: (start, stop, pad).

    Pure index arithmetic, here rather than in `media` for the same reason the
    placement maths is: this is what has to be right for a mouth to track a
    recording, and nothing that needs torch can be reached by the suite on a box
    with no GPU. `media.audio_span` is the three lines of tensor slicing around it.

    `stop` may run past the buffer; `pad` is how many samples of silence make up
    the difference. Padding rather than returning a short clip is deliberate -- a
    recording that ends mid-window should leave the mouth still for the rest of
    it, not shorten the window and desynchronise every window after.
    """
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError("sample_rate must be positive, got %r" % (sample_rate,))
    total = max(0, int(total_samples))
    want = max(1, int(round(float(seconds) * rate)))
    start = max(0, min(int(round(float(start_seconds or 0.0) * rate)), total))
    stop = min(total, start + want)
    return start, stop, want - (stop - start)


# ── the audio seam, at content level ────────────────────────────────────────
#
# WHY THIS IS A DIP AND NOT A CROSSFADE
#
# A crossfade needs overlap material. There is none: windows are sequential, butt
# joined, and every sample belongs to exactly one of them. Overlapping by F
# samples to crossfade properly would shorten the track by F at every seam -- and
# that is disqualifying, because the video side is a packet-level remux at fixed
# frame counts whose placement is *computed* from those counts (see the module
# docstring above). Audio that loses samples drifts against a video timeline that
# did not move, and the drift accumulates across every seam after it.
#
# So what is on offer is a length-preserving equal-power dip: the last half of the
# region attenuates window N from 1.0 to 1/sqrt(2), and the first half brings
# window N+1 back from 1/sqrt(2) to 1.0. Sample count is exactly preserved, there
# is no drift, and the discontinuity is replaced by a -3 dB notch one region wide.
# At 20-50 ms that is below where it reads as a duck, and it is a great deal less
# audible than the click it replaces.
#
# The gain match runs FIRST and is not cosmetic: bringing window N+1's opening
# level to the carry tail's leaves the dip less discontinuity to cover.

SEAM_FADE_MS_MIN = 20.0
SEAM_FADE_MS_MAX = 50.0
SEAM_FADE_MS_DEFAULT = 30.0


def seam_dip_samples(sample_rate, milliseconds=SEAM_FADE_MS_DEFAULT,
                     before=None, after=None):
    """Half-width of the dip in samples: how many to taper on each side of a seam.

    `before` and `after` are the sample counts actually available either side.
    The dip cannot eat a whole window, so it is clamped to both -- a 20 ms taper
    against a buffer of 12 samples is 12 samples, not an IndexError.

    Returns 0 when there is nothing to work with, which callers treat as "leave
    this seam alone".
    """
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError("sample_rate must be positive, got %r" % (sample_rate,))
    span = min(max(float(milliseconds), SEAM_FADE_MS_MIN), SEAM_FADE_MS_MAX)
    half = int(round(span / 1000.0 * rate / 2.0))
    for available in (before, after):
        if available is not None:
            half = min(half, max(0, int(available)))
    return max(0, half)


def seam_dip_gains(n, rising=False):
    """The taper, as a plain list of `n` gains.

    A cosine quarter-wave, so the pair of gains meeting at the seam are both
    1/sqrt(2) and their squares sum to 1 -- the same curve a real equal-power
    crossfade uses, applied to a join that cannot have one.

    `rising=False` is the outgoing side: 1.0 at the far end, 1/sqrt(2) at the
    seam. `rising=True` is its mirror, for the incoming side. The far end is
    exactly 1.0 in both cases, because that is where the taper meets audio it
    must not touch.
    """
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        # One sample is not a taper, it is a click. Leave it alone.
        return [1.0]
    # Both endpoints are exact: i=0 gives cos(0) = 1 where the taper meets
    # untouched audio, and i=n-1 gives cos(pi/4) = 1/sqrt(2) at the seam itself,
    # so the two sides really do satisfy a^2 + b^2 == 1 there.
    gains = [math.cos(i / float(n - 1) * (math.pi / 4.0)) for i in range(n)]
    return gains[::-1] if rising else gains


def seam_gain_match(tail_rms, head_rms, head_peak=0.0,
                    max_change_db=6.0, ceiling=1.0):
    """Scalar to bring a window's opening up (or down) to the previous tail's level.

    A level jump at a seam is heard as a jump even when there is no click, and it
    is the half of the problem a taper cannot fix -- the taper is 30 ms wide and
    the level difference lasts the whole window.

    Three guards, each earning its place:

    * `max_change_db` bounds the correction to +/-6 dB. A window that genuinely
      opens on near-silence -- a cut to a quiet room -- must not be dragged up to
      match the tail of a loud one. That would not be matching the seam, it would
      be flattening the film.
    * `ceiling` keeps the corrected peak inside the format. Applying a gain that
      pushes a peak past 1.0 trades a level step for clipping.
    * a silent tail or head returns 1.0. There is no ratio to take, and inventing
      one would amplify the noise floor.
    """
    tail_rms, head_rms = float(tail_rms), float(head_rms)
    if tail_rms <= 1e-9 or head_rms <= 1e-9:
        return 1.0
    limit = 10.0 ** (abs(float(max_change_db)) / 20.0)
    gain = min(max(tail_rms / head_rms, 1.0 / limit), limit)
    peak = float(head_peak)
    if peak > 0 and gain * peak > ceiling:
        gain = max(1.0 / limit, ceiling / peak)
    return gain
