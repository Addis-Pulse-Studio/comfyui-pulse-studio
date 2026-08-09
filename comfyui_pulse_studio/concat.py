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

The audio is not placed by this module. It is rebuilt from the segments' lossless
per-segment FLACs as one continuous waveform, which has no priming delay to repeat
and no seam to align -- see `media._remux_concat`.

Pure stdlib. `tests/test_concat.py` drives it with numbers read out of real
segment files.
"""

__all__ = ["video_shifts", "video_is_gapless", "frame_duration"]


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
