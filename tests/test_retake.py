"""Retake Scissor: cut geometry, anchor legality, stitch integrity."""

import unittest

from comfyui_pulse_studio.constants import MAX_WINDOW_FRAMES
from comfyui_pulse_studio.frames import is_on_grid
from comfyui_pulse_studio.retake import RetakeError, plan_retake


class TestGeometry(unittest.TestCase):
    def test_patch_length_is_always_on_grid(self):
        for a in range(1, 200, 7):
            for gap in range(1, 120, 11):
                plan = plan_retake(362, cut_start=a, cut_end=min(a + gap, 361))
                self.assertTrue(is_on_grid(plan.patch_frames),
                                "patch %d off-grid" % plan.patch_frames)

    def test_stitch_preserves_clip_length(self):
        """A patch replaces material; it must never lengthen or shorten the clip,
        or every frame after it shifts and the audio desyncs."""
        for a in range(0, 300, 13):
            for gap in range(1, 60, 7):
                b = min(a + gap, 362)
                if b <= a:
                    continue
                plan = plan_retake(362, cut_start=a, cut_end=b)
                self.assertEqual(plan.output_frames, 362)

    def test_interior_exactly_fills_the_gap(self):
        plan = plan_retake(362, cut_start=100, cut_end=140)
        take = plan.patch_take[1] - plan.patch_take[0]
        self.assertEqual(take, plan.gap_frames)

    def test_both_anchors_cost_two_patch_frames(self):
        plan = plan_retake(362, cut_start=100, cut_end=100 + 20)
        self.assertIsNotNone(plan.anchor_first_base_index)
        self.assertIsNotNone(plan.anchor_last_base_index)
        self.assertEqual(plan.patch_frames - 2, plan.gap_frames)
        self.assertEqual(plan.patch_take, (1, plan.patch_frames - 1))

    def test_anchors_are_the_frames_either_side_of_the_cut(self):
        plan = plan_retake(362, cut_start=100, cut_end=120)
        self.assertEqual(plan.anchor_first_base_index, plan.cut_start - 1)
        self.assertEqual(plan.anchor_last_base_index, plan.cut_end)

    def test_head_and_tail_ranges_are_contiguous_with_the_cut(self):
        plan = plan_retake(362, cut_start=100, cut_end=120)
        self.assertEqual(plan.head_range, (0, plan.cut_start))
        self.assertEqual(plan.tail_range, (plan.cut_end, 362))


class TestAnchorLegality(unittest.TestCase):
    def test_anchor_indices_are_only_zero_and_last(self):
        """PackedLayout raises ValueError for any other index -- there is no
        workaround, so the planner may never emit one."""
        for a in range(1, 300, 11):
            plan = plan_retake(362, cut_start=a, cut_end=a + 30)
            for spec in plan.anchors().values():
                self.assertIn(spec["patch_index"], (0, plan.patch_frames - 1))

    def test_first_anchor_is_patch_index_zero(self):
        plan = plan_retake(362, cut_start=50, cut_end=90)
        self.assertEqual(plan.anchors()["first_frame"]["patch_index"], 0)

    def test_last_anchor_is_patch_frame_count_minus_one(self):
        plan = plan_retake(362, cut_start=50, cut_end=90)
        self.assertEqual(plan.anchors()["last_frame"]["patch_index"], plan.patch_frames - 1)


class TestDegenerateCuts(unittest.TestCase):
    def test_cut_at_the_very_start_has_no_head_anchor(self):
        plan = plan_retake(362, cut_start=0, cut_end=40)
        self.assertIsNone(plan.anchor_first_base_index)
        self.assertIn("last_frame", plan.anchors())
        self.assertNotIn("first_frame", plan.anchors())
        self.assertEqual(plan.patch_take[0], 0)
        self.assertEqual(plan.output_frames, 362)
        self.assertTrue(any("no frame before it" in d for d in plan.diagnostics))

    def test_cut_to_the_very_end_has_no_tail_anchor(self):
        plan = plan_retake(362, cut_start=300, cut_end=362)
        self.assertIsNone(plan.anchor_last_base_index)
        self.assertIn("first_frame", plan.anchors())
        self.assertEqual(plan.patch_take[1], plan.patch_frames)
        self.assertEqual(plan.output_frames, 362)
        self.assertTrue(any("no frame after it" in d for d in plan.diagnostics))

    def test_whole_clip_is_a_full_rerender(self):
        plan = plan_retake(124, cut_start=0, cut_end=124)
        self.assertEqual(plan.anchors(), {})
        self.assertEqual(plan.patch_frames, 124)
        self.assertEqual(plan.output_frames, 124)
        self.assertTrue(any("full re-render" in d for d in plan.diagnostics))

    def test_single_frame_cut_still_produces_a_legal_patch(self):
        plan = plan_retake(362, cut_start=100, cut_end=101)
        self.assertTrue(is_on_grid(plan.patch_frames))
        self.assertEqual(plan.output_frames, 362)


class TestSnapping(unittest.TestCase):
    def test_cut_is_snapped_not_rejected(self):
        """The blueprint's rule: move the handles onto legal positions rather
        than rejecting the edit after the user has made it."""
        plan = plan_retake(362, cut_start=100, cut_end=104)
        self.assertTrue(plan.snapped)
        self.assertTrue(is_on_grid(plan.patch_frames))
        self.assertEqual(plan.requested_cut_start, 100)
        self.assertEqual(plan.requested_cut_end, 104)

    def test_snapping_grows_the_cut_forward_first(self):
        """The head is usually the part the user already approved."""
        plan = plan_retake(362, cut_start=100, cut_end=104)
        self.assertEqual(plan.cut_start, 100)
        self.assertGreater(plan.cut_end, 104)

    def test_snapping_falls_back_to_growing_backwards_at_the_clip_end(self):
        plan = plan_retake(362, cut_start=355, cut_end=360)
        self.assertEqual(plan.output_frames, 362)
        self.assertTrue(is_on_grid(plan.patch_frames))
        self.assertLessEqual(plan.cut_end, 362)

    def test_exact_grid_cut_is_not_snapped(self):
        # gap 20 + 2 anchors = 22, already on-grid.
        plan = plan_retake(362, cut_start=100, cut_end=120)
        self.assertFalse(plan.snapped)
        self.assertEqual(plan.patch_frames, 22)

    def test_seconds_input_is_converted_and_snapped(self):
        plan = plan_retake(362, cut_start_seconds=4.0, cut_end_seconds=5.0, fps=24)
        self.assertEqual(plan.requested_cut_start, 96)
        self.assertTrue(is_on_grid(plan.patch_frames))
        self.assertEqual(plan.output_frames, 362)

    def test_clip_too_short_for_the_smallest_patch_becomes_a_rerender(self):
        plan = plan_retake(22, cut_start=10, cut_end=12)
        self.assertEqual(plan.output_frames, 22)
        self.assertTrue(is_on_grid(plan.patch_frames))


class TestAudio(unittest.TestCase):
    def test_keep_base_audio_defaults_on(self):
        """A re-rendered patch invents its own score and will not match the
        surrounding track, so keeping the base audio is the sane default."""
        self.assertTrue(plan_retake(362, cut_start=50, cut_end=90).keep_base_audio)

    def test_disabling_it_is_warned_about(self):
        plan = plan_retake(362, cut_start=50, cut_end=90, keep_base_audio=False)
        self.assertFalse(plan.keep_base_audio)
        self.assertTrue(any("will not match" in d for d in plan.diagnostics))


class TestRejections(unittest.TestCase):
    def test_empty_cut_rejected(self):
        with self.assertRaises(RetakeError) as ctx:
            plan_retake(362, cut_start=100, cut_end=100)
        self.assertIn("empty", str(ctx.exception))

    def test_patch_longer_than_one_render_rejected(self):
        """Patching more than a single H3 call is not a patch."""
        with self.assertRaises(RetakeError) as ctx:
            plan_retake(2000, cut_start=10, cut_end=1500)
        self.assertIn("ceiling", str(ctx.exception))

    def test_reversed_cut_is_corrected(self):
        plan = plan_retake(362, cut_start=120, cut_end=80)
        self.assertLess(plan.cut_start, plan.cut_end)
        self.assertTrue(any("reversed" in d for d in plan.diagnostics))

    def test_out_of_bounds_cut_is_clamped(self):
        plan = plan_retake(362, cut_start=300, cut_end=9999)
        self.assertLessEqual(plan.cut_end, 362)
        self.assertEqual(plan.output_frames, 362)

    def test_missing_cut_points_rejected(self):
        with self.assertRaises(RetakeError):
            plan_retake(362)

    def test_base_clip_too_short_rejected(self):
        with self.assertRaises(RetakeError):
            plan_retake(3, cut_start=0, cut_end=2)

    def test_plan_is_serialisable(self):
        import json
        json.dumps(plan_retake(362, cut_start=50, cut_end=90).to_dict())


class TestExhaustiveInvariants(unittest.TestCase):
    def test_every_reachable_cut_yields_a_sound_plan(self):
        """The properties that must hold for any cut the UI can produce."""
        base = 362
        for a in range(0, base, 5):
            for b in range(a + 1, min(a + 200, base + 1), 9):
                try:
                    plan = plan_retake(base, cut_start=a, cut_end=b)
                except RetakeError:
                    continue
                self.assertTrue(is_on_grid(plan.patch_frames))
                self.assertLessEqual(plan.patch_frames, MAX_WINDOW_FRAMES)
                self.assertEqual(plan.output_frames, base)
                self.assertGreaterEqual(plan.cut_start, 0)
                self.assertLessEqual(plan.cut_end, base)
                self.assertLess(plan.cut_start, plan.cut_end)
                take = plan.patch_take[1] - plan.patch_take[0]
                self.assertEqual(take, plan.gap_frames)
                for spec in plan.anchors().values():
                    self.assertIn(spec["patch_index"], (0, plan.patch_frames - 1))
                    self.assertGreaterEqual(spec["base_index"], 0)
                    self.assertLess(spec["base_index"], base)


if __name__ == "__main__":
    unittest.main()
