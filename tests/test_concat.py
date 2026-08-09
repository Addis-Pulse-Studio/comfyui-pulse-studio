"""Joining rendered segments end to end. Spec §8.

THE RUN THIS FILE COMES FROM
----------------------------
A two-window render finished -- 870s and 875s of GPU time, both segments written
and manifested -- and then the join raised

    av.error.ArgumentError: Invalid argument: '...assembled.mp4' returned 22

EINVAL out of `av_interleaved_write_frame`, which is what libavformat returns for
a non-monotonic DTS. The numbers used throughout this file are the real ones, read
out of those two files with ffprobe:

    video   time_base 1/12288   first dts -1024   last dts+dur 114688
    audio   time_base 1/32000   first dts -1024   last dts+dur 301344

Both streams open on a *negative* decode timestamp, and both do so legitimately:
the video carries B-frames, and AAC has a priming delay. Crucially -1024 is a
different amount of wall-clock time in each (-83ms against -32ms), which is
exactly what the first implementation got wrong.
"""

import unittest

from comfyui_pulse_studio.concat import frame_duration, video_is_gapless, video_shifts

VIDEO_TB = 1.0 / 12288
AUDIO_TB = 1.0 / 32000




class TestVideoPlacement(unittest.TestCase):
    """How segments are actually placed now. Spec §8.

    The DTS-derived spacing above is monotonic but not *gapless*: a segment's
    decode extent is longer than its content, so packing nose to tail on DTS left
    an 84ms hole at every seam -- two dropped frames, visible as a freeze. The
    placement is therefore computed from what is exactly known instead: every
    segment is `frames` frames at a known fps.
    """

    FRAME = 512          # 1/24s at 1/12288
    COUNTS = [226, 226]  # the real run

    def test_segments_are_placed_one_frame_duration_apart(self):
        shifts = video_shifts(self.COUNTS, 24, VIDEO_TB)
        self.assertEqual(shifts, [0, 226 * self.FRAME])

    def test_the_placement_is_gapless(self):
        shifts = video_shifts(self.COUNTS, 24, VIDEO_TB)
        self.assertTrue(video_is_gapless(self.COUNTS, 24, VIDEO_TB, shifts, self.FRAME))

    def test_a_ragged_plan_is_still_gapless(self):
        counts = [362, 362, 192]
        shifts = video_shifts(counts, 24, VIDEO_TB)
        self.assertTrue(video_is_gapless(counts, 24, VIDEO_TB, shifts, self.FRAME))

    def test_twelve_segments_land_on_exact_frame_boundaries(self):
        counts = [362] * 12
        shifts = video_shifts(counts, 24, VIDEO_TB)
        for index, shift in enumerate(shifts):
            self.assertEqual(shift, index * 362 * self.FRAME)
            self.assertEqual(shift % self.FRAME, 0, "a shift landed mid-frame")

    def test_the_total_length_is_the_sum_of_the_windows(self):
        counts = [226, 226]
        shifts = video_shifts(counts, 24, VIDEO_TB)
        end = (shifts[-1] + counts[-1] * self.FRAME) * VIDEO_TB
        self.assertAlmostEqual(end, sum(counts) / 24.0, places=6)

    def test_the_gapless_check_can_fail(self):
        """A guard that cannot fail guards nothing."""
        shifts = video_shifts(self.COUNTS, 24, VIDEO_TB)
        shifts[1] += self.FRAME       # one frame of dead air at the seam
        self.assertFalse(video_is_gapless(self.COUNTS, 24, VIDEO_TB, shifts, self.FRAME))

    def test_a_single_segment_sits_at_zero(self):
        self.assertEqual(video_shifts([226], 24, VIDEO_TB), [0])


class TestFrameDuration(unittest.TestCase):
    def test_one_frame_at_24fps_in_the_real_time_base(self):
        self.assertEqual(frame_duration(24, VIDEO_TB), 512)

    def test_it_is_the_default_used_by_the_gapless_check(self):
        counts = [226, 226]
        shifts = video_shifts(counts, 24, VIDEO_TB)
        self.assertTrue(video_is_gapless(counts, 24, VIDEO_TB, shifts))


if __name__ == "__main__":
    unittest.main()
