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

from comfyui_pulse_studio.concat import (
    SEAM_FADE_MS_MAX,
    SEAM_FADE_MS_MIN,
    frame_duration,
    seam_dip_gains,
    seam_dip_samples,
    seam_gain_match,
    video_is_gapless,
    video_shifts,
)

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


class TestTheAudioSeamIsADipNotACrossfade(unittest.TestCase):
    """The join between two independently generated scores.

    The container seam was solved long ago -- placement is computed from frame
    counts, so the picture is gapless by construction. Nothing addressed the fact
    that the two sides are two separate takes: a score that restarts, a level that
    steps.

    A real crossfade is not available here. It needs overlap material, and there
    is none -- windows are butt-joined and every sample belongs to exactly one of
    them. Overlapping to crossfade would shorten the track at every seam, and the
    video is a packet remux at FIXED frame counts, so lost samples drift against a
    picture that did not move and the drift accumulates. Hence a length-preserving
    dip, and hence the sample-count assertion below, which is the whole constraint
    written down.
    """

    def test_the_two_gains_meeting_at_the_seam_are_equal_power(self):
        for n in (2, 3, 240, 721):
            with self.subTest(n=n):
                falling = seam_dip_gains(n)
                rising = seam_dip_gains(n, rising=True)
                self.assertAlmostEqual(falling[-1] ** 2 + rising[0] ** 2, 1.0, places=9)

    def test_each_taper_starts_at_unity_where_it_meets_untouched_audio(self):
        for n in (2, 240):
            with self.subTest(n=n):
                self.assertEqual(seam_dip_gains(n)[0], 1.0)
                self.assertEqual(seam_dip_gains(n, rising=True)[-1], 1.0)

    def test_the_dip_bottoms_out_at_minus_three_decibels(self):
        falling = seam_dip_gains(240)
        self.assertAlmostEqual(falling[-1], 2 ** -0.5, places=9)

    def test_the_taper_is_monotonic(self):
        falling = seam_dip_gains(240)
        self.assertEqual(falling, sorted(falling, reverse=True))
        self.assertEqual(seam_dip_gains(240, rising=True), sorted(falling))

    def test_the_two_sides_are_mirrors(self):
        self.assertEqual(seam_dip_gains(64, rising=True), seam_dip_gains(64)[::-1])

    def test_one_sample_is_left_alone_because_it_would_be_a_click(self):
        self.assertEqual(seam_dip_gains(1), [1.0])
        self.assertEqual(seam_dip_gains(0), [])

    # ── the width ───────────────────────────────────────────────────────────

    def test_the_width_is_clamped_to_the_documented_range(self):
        rate = 48000
        narrow = seam_dip_samples(rate, 1.0)
        wide = seam_dip_samples(rate, 5000.0)
        self.assertEqual(narrow, int(round(SEAM_FADE_MS_MIN / 1000.0 * rate / 2.0)))
        self.assertEqual(wide, int(round(SEAM_FADE_MS_MAX / 1000.0 * rate / 2.0)))

    def test_the_width_never_exceeds_what_is_available(self):
        """A 20ms taper against a 12-sample buffer is 12 samples, not a crash."""
        self.assertEqual(seam_dip_samples(48000, 30.0, before=12, after=99999), 12)
        self.assertEqual(seam_dip_samples(48000, 30.0, before=99999, after=7), 7)
        self.assertEqual(seam_dip_samples(48000, 30.0, before=0, after=0), 0)

    def test_a_nonsense_sample_rate_is_refused(self):
        for rate in (0, -1):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    seam_dip_samples(rate)

    def test_the_treatment_cannot_change_the_sample_count(self):
        """The A/V sync constraint, as arithmetic.

        Both tapers are applied in place over samples that already exist. The
        number of gains equals the number of samples they scale, on both sides,
        so no implementation built on these can shorten or lengthen the track.
        """
        for rate, ms in ((48000, 30.0), (44100, 20.0), (22050, 50.0)):
            with self.subTest(rate=rate, ms=ms):
                half = seam_dip_samples(rate, ms)
                self.assertEqual(len(seam_dip_gains(half)), half)
                self.assertEqual(len(seam_dip_gains(half, rising=True)), half)


class TestTheGainMatch(unittest.TestCase):
    """Levelling the next window's opening to the previous window's tail.

    Runs before the taper and is not cosmetic: a level step lasts the whole
    window, and a 30ms taper cannot hide something 30 seconds long.
    """

    def test_a_quiet_opening_is_brought_up_towards_the_tail(self):
        self.assertAlmostEqual(seam_gain_match(0.15, 0.10), 1.5, places=6)

    def test_a_loud_opening_is_brought_down(self):
        self.assertAlmostEqual(seam_gain_match(0.10, 0.15), 1.0 / 1.5, places=6)

    def test_doubling_sits_just_outside_the_bound_and_is_clamped(self):
        """2x is +6.02 dB, which the +/-6 dB bound just catches.

        Asserted deliberately: it is the least intuitive consequence of the
        bound, and a future widening of it should have to change this line.
        """
        self.assertAlmostEqual(seam_gain_match(0.2, 0.1), 10 ** (6.0 / 20.0), places=6)
        self.assertLess(seam_gain_match(0.2, 0.1), 2.0)

    def test_a_matched_seam_is_left_alone(self):
        self.assertAlmostEqual(seam_gain_match(0.15, 0.15), 1.0, places=9)

    def test_the_correction_is_bounded(self):
        """A cut to a genuinely quiet room must not be dragged up to match a loud
        tail. That is not matching a seam, it is flattening the film."""
        self.assertAlmostEqual(seam_gain_match(1.0, 0.0001), 10 ** (6.0 / 20.0), places=6)
        self.assertAlmostEqual(seam_gain_match(0.0001, 1.0), 10 ** (-6.0 / 20.0), places=6)

    def test_silence_on_either_side_returns_unity(self):
        self.assertEqual(seam_gain_match(0.0, 0.1), 1.0)
        self.assertEqual(seam_gain_match(0.1, 0.0), 1.0)

    def test_the_gain_never_pushes_a_peak_into_clipping(self):
        """Trading a level step for clipping would not be an improvement."""
        gain = seam_gain_match(0.9, 0.5, head_peak=0.95)
        self.assertLessEqual(gain * 0.95, 1.0 + 1e-9)

    def test_a_peak_already_at_the_ceiling_is_not_amplified(self):
        gain = seam_gain_match(0.9, 0.4, head_peak=1.0)
        self.assertLessEqual(gain, 1.0 + 1e-9)


class TestTheTensorRind(unittest.TestCase):
    """media.treat_audio_seams, which is slicing over the arithmetic above.

    Mirrors the pure/impure split tests/test_audio_role.py already uses for
    audio_span_bounds against media.audio_span: everything that can be decided is
    decided in concat.py, and this covers the three lines around it.
    """

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in this environment")

    @staticmethod
    def _chunk(level, samples=4096, rate=48000):
        import torch

        return {"waveform": torch.full((1, 2, samples), float(level)),
                "sample_rate": rate}

    def test_the_total_sample_count_is_unchanged(self):
        """The constraint that rules out a real crossfade."""
        from media import treat_audio_seams

        chunks = [self._chunk(0.5), self._chunk(0.5), self._chunk(0.5)]
        before = sum(c["waveform"].shape[-1] for c in chunks)
        after = sum(c["waveform"].shape[-1] for c in treat_audio_seams(chunks))
        self.assertEqual(before, after)

    def test_the_input_chunks_are_not_mutated(self):
        from media import treat_audio_seams

        chunks = [self._chunk(0.5), self._chunk(0.2)]
        originals = [c["waveform"].clone() for c in chunks]
        treat_audio_seams(chunks)
        for chunk, original in zip(chunks, originals):
            self.assertTrue(bool((chunk["waveform"] == original).all()))

    def test_the_samples_at_the_seam_are_attenuated(self):
        from media import treat_audio_seams

        out = treat_audio_seams([self._chunk(0.5), self._chunk(0.5)])
        self.assertAlmostEqual(float(out[0]["waveform"][0, 0, -1]),
                               0.5 * 2 ** -0.5, places=5)
        self.assertAlmostEqual(float(out[1]["waveform"][0, 0, 0]),
                               0.5 * 2 ** -0.5, places=5)

    def test_audio_far_from_a_seam_is_untouched(self):
        from media import treat_audio_seams

        out = treat_audio_seams([self._chunk(0.5), self._chunk(0.5)])
        self.assertAlmostEqual(float(out[0]["waveform"][0, 0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(out[1]["waveform"][0, 0, -1]), 0.5, places=6)

    def test_a_level_step_is_reduced(self):
        from media import treat_audio_seams

        out = treat_audio_seams([self._chunk(0.4), self._chunk(0.1)])
        # The second window is scaled towards the first, capped at +6 dB -- so it
        # closes some of a 4x step, not all of it. Closing all of it is exactly
        # what the bound exists to prevent.
        tail = float(out[1]["waveform"][0, 0, -1])
        self.assertGreater(tail, 0.1)
        self.assertLessEqual(tail, 0.1 * 10 ** (6.0 / 20.0) + 1e-6)
        self.assertLess(tail, 0.4)

    def test_a_single_chunk_has_no_seam_to_treat(self):
        from media import treat_audio_seams

        chunk = self._chunk(0.5)
        out = treat_audio_seams([chunk])
        self.assertAlmostEqual(float(out[0]["waveform"][0, 0, -1]), 0.5, places=6)

    def test_concat_audio_leaves_seams_alone_unless_asked(self):
        from media import concat_audio

        chunks = [self._chunk(0.5), self._chunk(0.5)]
        plain = concat_audio(chunks)
        treated = concat_audio(chunks, seam_treatment=True)
        self.assertEqual(plain["waveform"].shape, treated["waveform"].shape)
        self.assertAlmostEqual(float(plain["waveform"][0, 0, 4095]), 0.5, places=6)
        self.assertLess(float(treated["waveform"][0, 0, 4095]), 0.5)




if __name__ == "__main__":
    unittest.main()
