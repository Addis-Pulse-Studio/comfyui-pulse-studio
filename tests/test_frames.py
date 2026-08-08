"""Frame-grid quantisation and window partitioning."""

import unittest

from omni_director.constants import MAX_WINDOW_FRAMES, MIN_FRAMES, MIN_TRAINED_FRAMES
from omni_director.frames import (
    align_frame_count,
    frames_from_step,
    frames_to_seconds,
    grid_step_index,
    is_on_grid,
    last_anchor_index,
    partition_windows,
    seconds_to_frames,
    snap_frames,
    snap_frames_down,
    snap_frames_nearest,
    video_latent_t,
    window_bounds,
)

# The grid, computed independently of the implementation under test.
LEGAL = [17 * k + 5 for k in range(0, 40)]


class TestGrid(unittest.TestCase):
    def test_legal_values_are_on_grid(self):
        for n in LEGAL:
            self.assertTrue(is_on_grid(n), "%d should be on-grid" % n)

    def test_illegal_values_are_off_grid(self):
        for n in (0, 1, 4, 6, 21, 23, 100, 361, 363):
            self.assertFalse(is_on_grid(n), "%d should be off-grid" % n)

    def test_bounds_of_grid(self):
        self.assertEqual(LEGAL[0], MIN_FRAMES)
        # 362 is the trained ceiling and must itself be a legal grid point --
        # a ceiling off the grid would be unreachable.
        self.assertTrue(is_on_grid(MAX_WINDOW_FRAMES))
        self.assertEqual(grid_step_index(MAX_WINDOW_FRAMES), 21)
        self.assertTrue(is_on_grid(MIN_TRAINED_FRAMES))


class TestAlignUp(unittest.TestCase):
    def test_matches_core_algorithm(self):
        """Behavioural parity with nodes_minimax_h3.align_frame_count."""
        for n in range(-20, 400):
            expected = max(MIN_FRAMES, n)
            while expected % 17 != 5:
                expected += 1
            self.assertEqual(align_frame_count(n), expected, "n=%d" % n)

    def test_on_grid_values_are_fixed_points(self):
        for n in LEGAL:
            self.assertEqual(align_frame_count(n), n)

    def test_rounds_up_never_down(self):
        for n in range(MIN_FRAMES, 400):
            self.assertGreaterEqual(align_frame_count(n), n)

    def test_floors_at_five(self):
        for n in (-100, -1, 0, 1, 4, 5):
            self.assertEqual(align_frame_count(n), MIN_FRAMES)


class TestSnapDown(unittest.TestCase):
    def test_rounds_down_never_up(self):
        for n in range(MIN_FRAMES, 400):
            self.assertLessEqual(snap_frames_down(n), n)
            self.assertTrue(is_on_grid(snap_frames_down(n)))

    def test_floors_at_five(self):
        """Core's reference-video trim decrements without a floor; ours must not
        walk below the minimum legal count."""
        for n in (-10, 0, 3, 4, 5, 6, 21):
            self.assertGreaterEqual(snap_frames_down(n), MIN_FRAMES)
        self.assertEqual(snap_frames_down(21), 5)
        self.assertEqual(snap_frames_down(22), 22)


class TestSnapNearest(unittest.TestCase):
    def test_picks_the_closer_point(self):
        self.assertEqual(snap_frames_nearest(6), 5)     # 1 away vs 16
        self.assertEqual(snap_frames_nearest(21), 22)   # 16 away vs 1
        self.assertEqual(snap_frames_nearest(13), 5)    # 8 away vs 9
        self.assertEqual(snap_frames_nearest(14), 22)   # 9 away vs 8

    def test_tie_rounds_up(self):
        # Midpoint of 5 and 22 is 13.5; 13 is nearer 5, 14 nearer 22. Construct
        # an exact tie on an even-width span instead.
        n = 5 + 17  # 22
        self.assertEqual(snap_frames_nearest(n), 22)

    def test_always_lands_on_grid(self):
        for n in range(MIN_FRAMES, 400):
            self.assertTrue(is_on_grid(snap_frames_nearest(n)))


class TestSnapDispatch(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(snap_frames(100, "up"), align_frame_count(100))
        self.assertEqual(snap_frames(100, "down"), snap_frames_down(100))
        self.assertEqual(snap_frames(100, "nearest"), snap_frames_nearest(100))

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            snap_frames(100, "sideways")


class TestStepConversion(unittest.TestCase):
    def test_roundtrip(self):
        for k in range(0, 30):
            n = frames_from_step(k)
            self.assertTrue(is_on_grid(n))
            self.assertEqual(grid_step_index(n), k)

    def test_rejects_off_grid(self):
        with self.assertRaises(ValueError):
            grid_step_index(100)


class TestSeconds(unittest.TestCase):
    def test_known_durations(self):
        # 362 frames at 24fps is the ~15.08s ceiling the blueprint quotes.
        self.assertAlmostEqual(frames_to_seconds(362), 15.0833, places=3)
        self.assertAlmostEqual(frames_to_seconds(124), 5.1667, places=3)

    def test_seconds_to_frames_quantises(self):
        self.assertTrue(is_on_grid(seconds_to_frames(5.0)))
        self.assertTrue(is_on_grid(seconds_to_frames(10.0)))
        self.assertTrue(is_on_grid(seconds_to_frames(0.01)))
        # Rounding up means the render is never shorter than asked.
        self.assertGreaterEqual(frames_to_seconds(seconds_to_frames(7.3)), 7.3)


class TestLatentT(unittest.TestCase):
    def test_matches_core_formula(self):
        for n in LEGAL[:25]:
            expected = 2 if n <= 5 else ((n - 5) // 17) * 5 + 2
            self.assertEqual(video_latent_t(n), expected)


class TestAnchorIndex(unittest.TestCase):
    def test_last_anchor_is_frame_count_minus_one(self):
        """PackedLayout accepts pixel_index 0 or frame_count-1, nothing else."""
        for n in LEGAL[:10]:
            self.assertEqual(last_anchor_index(n), n - 1)

    def test_rejects_off_grid_frame_count(self):
        with self.assertRaises(ValueError):
            last_anchor_index(100)


class TestPartitionWindows(unittest.TestCase):
    def _check(self, windows, total, cap):
        for w in windows:
            self.assertTrue(is_on_grid(w), "window %d off-grid" % w)
            self.assertLessEqual(w, cap, "window %d exceeds cap %d" % (w, cap))
            self.assertGreaterEqual(w, MIN_FRAMES)
        # Never render less than asked; a shortfall would truncate a shot. The
        # total itself is a request and is not snapped -- only the windows are.
        self.assertGreaterEqual(sum(windows), total)

    def test_single_window_when_it_fits(self):
        self.assertEqual(partition_windows(124), [124])
        self.assertEqual(partition_windows(362), [362])
        self.assertEqual(partition_windows(100), [107])  # snapped up first

    def test_exhaustive_balanced(self):
        for total in range(5, 2000, 7):
            w = partition_windows(total, MAX_WINDOW_FRAMES, policy="balanced")
            self._check(w, total, MAX_WINDOW_FRAMES)

    def test_balanced_windows_differ_by_at_most_one_step(self):
        for total in range(5, 2000, 13):
            w = partition_windows(total, MAX_WINDOW_FRAMES, policy="balanced")
            self.assertLessEqual(max(w) - min(w), 17,
                                 "windows %r differ by more than one grid step" % (w,))

    def test_balanced_uses_the_fewest_windows_that_fit(self):
        import math
        for total in range(5, 2000, 11):
            w = partition_windows(total, MAX_WINDOW_FRAMES, policy="balanced")
            self.assertEqual(len(w), max(1, math.ceil(total / MAX_WINDOW_FRAMES)))

    def test_balanced_avoids_the_runt(self):
        """16s at a 15s cap must not become 15s + 1s.

        A trailing window far below the trained floor renders as visible garbage;
        two near-equal windows are strictly better and cost the same.
        """
        total = seconds_to_frames(16.0)
        w = partition_windows(total, MAX_WINDOW_FRAMES, policy="balanced")
        self.assertEqual(len(w), 2)
        self.assertLessEqual(max(w) - min(w), 17)
        self.assertGreater(min(w), MIN_TRAINED_FRAMES)

    def test_fill_policy_packs_then_tails(self):
        w = partition_windows(362 * 2 + 124, MAX_WINDOW_FRAMES, policy="fill")
        self.assertEqual(w[:2], [362, 362])
        self.assertEqual(w[2], 124)

    def test_fill_policy_rebalances_a_runt_tail(self):
        """A tail below the trained floor is worse than no tail; fill falls back."""
        total = 362 + 22
        w = partition_windows(total, MAX_WINDOW_FRAMES, policy="fill",
                              min_window_frames=MIN_TRAINED_FRAMES)
        self.assertNotIn(22, w)
        self.assertGreaterEqual(min(w), MIN_TRAINED_FRAMES)
        self._check(w, total, MAX_WINDOW_FRAMES)

    def test_fill_exact_multiple_has_no_tail(self):
        w = partition_windows(362 * 3, MAX_WINDOW_FRAMES, policy="fill")
        self.assertEqual(w, [362, 362, 362])

    def test_custom_window_cap_is_snapped_down(self):
        """An off-grid cap must tighten, never loosen -- rounding a cap up would
        silently render windows longer than the caller allowed."""
        w = partition_windows(1000, 200, policy="balanced")
        self._check(w, 1000, 192)  # 192 = snap_down(200)

    def test_total_below_the_trained_floor_passes_through_as_one_window(self):
        """The single documented exception to the 124-frame floor: there is
        nothing to merge a short total into, and refusing a deliberately short
        clip would be worse than rendering it."""
        notes = []
        w = partition_windows(50, 1, diagnostics=notes)
        self.assertEqual(len(w), 1)
        self.assertTrue(is_on_grid(w[0]))
        self.assertGreaterEqual(w[0], 50)
        self.assertTrue(any("trained floor" in n for n in notes))

    def test_rejects_unknown_policy(self):
        with self.assertRaises(ValueError):
            partition_windows(1000, policy="whatever")

    def test_duration_agnostic(self):
        """Raising the ceiling must need one number, not a code change."""
        w = partition_windows(2000, 700, policy="balanced")
        self._check(w, 2000, 696)  # 696 = snap_down(700)


class TestWindowBounds(unittest.TestCase):
    def test_bounds_are_contiguous_and_ordered(self):
        windows = partition_windows(1200)
        bounds = window_bounds(windows)
        self.assertEqual(len(bounds), len(windows))
        self.assertEqual(bounds[0][0], 0.0)
        for (s0, e0), (s1, e1) in zip(bounds, bounds[1:]):
            self.assertAlmostEqual(e0, s1, places=9)
            self.assertLess(s0, e0)

    def test_total_matches_frame_sum(self):
        windows = partition_windows(1200)
        bounds = window_bounds(windows)
        self.assertAlmostEqual(bounds[-1][1], frames_to_seconds(sum(windows)), places=9)


if __name__ == "__main__":
    unittest.main()
