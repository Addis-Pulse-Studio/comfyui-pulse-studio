"""Still Mode: canvas fitting, frame_pick, branch selection."""

import unittest

from comfyui_pulse_studio.assets import KIND_IMAGE, Asset, AssetBin
from comfyui_pulse_studio.constants import (
    BRANCH_FL2VA,
    BRANCH_REF2VA,
    CANVAS_MULTIPLE,
    MAX_PIXELS,
    STILL_FRAMES,
)
from comfyui_pulse_studio.still import (
    StillError,
    adapt_canvas_core,
    canvas_from_reference,
    plan_still,
)


class TestCanvasFromReference(unittest.TestCase):
    def test_never_exceeds_the_pixel_budget(self):
        """1,032,192 px is the cap, and it is enforced by construction.

        The long edge is floored to the grid and the short edge is then capped at
        `budget // long_edge` before rounding, so no pair can be built that
        exceeds the budget -- which is what allows the short edge to round to
        nearest instead of down.
        """
        for w, h in [(1920, 1080), (1080, 1920), (1000, 1000), (3000, 500),
                     (500, 3000), (4096, 2160), (37, 41), (5000, 5000)]:
            cw, ch = canvas_from_reference(w, h)
            self.assertLessEqual(cw * ch, MAX_PIXELS, "%dx%d -> %dx%d" % (w, h, cw, ch))

    def test_both_axes_are_multiples_of_32(self):
        for w, h in [(1920, 1080), (1234, 567), (100, 100), (7, 3000)]:
            cw, ch = canvas_from_reference(w, h)
            self.assertEqual(cw % CANVAS_MULTIPLE, 0)
            self.assertEqual(ch % CANVAS_MULTIPLE, 0)

    def test_the_long_edge_rounds_down(self):
        """The long edge still floors; only the short edge rounds to nearest.

        This was `test_rounds_down_never_up` and asserted flooring on *both* axes.
        That was the bug, not the contract: flooring twice compounds two losses
        and moved 16:9 to a measured 1.826. The long edge floors, the short edge
        is then chosen against the budget, and the budget is asserted separately
        above.
        """
        import math
        for w, h in [(1920, 1080), (1000, 1000), (1600, 900), (1080, 1920)]:
            ratio = w / h
            cw, ch = canvas_from_reference(w, h)
            long_edge, ideal = ((cw, math.sqrt(MAX_PIXELS * ratio)) if cw >= ch
                                else (ch, math.sqrt(MAX_PIXELS / ratio)))
            self.assertLessEqual(long_edge, ideal, "%dx%d -> %dx%d" % (w, h, cw, ch))
            self.assertGreater(long_edge, ideal - CANVAS_MULTIPLE,
                               "%dx%d floored more than one grid step" % (w, h))

    def test_aspect_is_approximately_preserved(self):
        for w, h in [(1920, 1080), (1080, 1920), (1600, 900), (2000, 1000)]:
            cw, ch = canvas_from_reference(w, h)
            self.assertAlmostEqual(cw / ch, w / h, delta=0.12,
                                   msg="%dx%d -> %dx%d" % (w, h, cw, ch))

    def test_uses_most_of_the_budget(self):
        """Rounding down must not throw away the budget -- an edit canvas far
        below the cap needlessly loses source detail."""
        for w, h in [(1920, 1080), (1000, 1000), (1600, 900)]:
            cw, ch = canvas_from_reference(w, h)
            self.assertGreater(cw * ch, MAX_PIXELS * 0.85)

    def test_sixteen_by_nine_is_the_familiar_canvas(self):
        """1344x768, and it must equal what the 16:9 *preset* resolves to.

        A reference-derived canvas and a preset-selected one reach the same node.
        They disagreed until 2026-08-17 -- 1344x736 here, 1344x768 there -- so the
        two are asserted against each other rather than against two literals.
        """
        from comfyui_pulse_studio.canvas import resolution_for

        self.assertEqual(canvas_from_reference(1920, 1080), (1344, 768))
        self.assertEqual(canvas_from_reference(1920, 1080),
                         resolution_for("16:9 landscape", 0, 0))

    def test_square_fills_the_budget(self):
        cw, ch = canvas_from_reference(1000, 1000)
        self.assertEqual(cw, ch)
        self.assertLessEqual(cw * ch, MAX_PIXELS)

    def test_floors_at_one_multiple(self):
        cw, ch = canvas_from_reference(10000, 1)
        self.assertGreaterEqual(ch, CANVAS_MULTIPLE)
        self.assertGreaterEqual(cw, CANVAS_MULTIPLE)

    def test_rejects_nonpositive(self):
        with self.assertRaises(StillError):
            canvas_from_reference(0, 100)


class TestCoreCanvasMirror(unittest.TestCase):
    def test_matches_core_adapt_canvas(self):
        """Behavioural parity with nodes_minimax_h3.adapt_canvas, recomputed here
        independently."""
        import math
        for w, h in [(1920, 1080), (1080, 1920), (1000, 1000), (3000, 500)]:
            ratio = w / h
            if ratio >= 1.0:
                nw, nh = 768 * ratio, 768.0
            else:
                nw, nh = 768.0, 768 / ratio
            if nw * nh > MAX_PIXELS:
                s = math.sqrt(MAX_PIXELS / (nw * nh))
                nw, nh = nw * s, nh * s
            expected = (max(32, round(nw / 32) * 32), max(32, round(nh / 32) * 32))
            self.assertEqual(adapt_canvas_core(w, h), expected)

    def test_core_normalises_the_short_edge_where_ours_fills_the_budget(self):
        """A square is the clearest case: core stays at 768x768, we fill more."""
        self.assertEqual(adapt_canvas_core(1000, 1000), (768, 768))
        cw, ch = canvas_from_reference(1000, 1000)
        self.assertGreater(cw * ch, 768 * 768)


class TestFramePick(unittest.TestCase):
    def test_length_is_always_five(self):
        self.assertEqual(plan_still(prompt="a cat").length, STILL_FRAMES)

    def test_valid_picks_accepted(self):
        for pick in range(0, 5):
            self.assertEqual(plan_still(prompt="x", frame_pick=pick).frame_pick, pick)

    def test_out_of_range_rejected(self):
        for pick in (-1, 5, 99):
            with self.assertRaises(StillError):
                plan_still(prompt="x", frame_pick=pick)

    def test_pick_zero_on_an_edit_is_flagged_as_a_near_copy(self):
        """The source is pinned at frame 0, so pick 0 reproduces it."""
        plan = plan_still(prompt="x", source_asset="src", frame_pick=0,
                          source_size=(1920, 1080))
        self.assertTrue(any("closely reproduce the source" in d for d in plan.diagnostics))

    def test_higher_pick_is_not_flagged(self):
        plan = plan_still(prompt="x", source_asset="src", frame_pick=4,
                          source_size=(1920, 1080))
        self.assertFalse(any("closely reproduce" in d for d in plan.diagnostics))


class TestBranchSelection(unittest.TestCase):
    def test_editing_uses_fl2va_and_pins_frame_zero(self):
        plan = plan_still(prompt="x", source_asset="src", source_size=(1920, 1080))
        self.assertEqual(plan.branch, BRANCH_FL2VA)
        self.assertEqual(plan.anchors["first_frame"], "src")
        self.assertEqual(plan.anchors["first_frame_index"], 0)
        self.assertTrue(plan.is_edit)

    def test_editing_never_emits_a_last_frame_anchor(self):
        """A still has no meaningful end anchor; pinning one would fight the edit."""
        plan = plan_still(prompt="x", source_asset="src", source_size=(1920, 1080))
        self.assertNotIn("last_frame", plan.anchors)

    def test_generating_uses_ref2va(self):
        plan = plan_still(prompt="x", width=1344, height=736)
        self.assertEqual(plan.branch, BRANCH_REF2VA)
        self.assertFalse(plan.is_edit)
        self.assertEqual(plan.anchors, {})

    def test_references_on_an_edit_are_refused_loudly(self):
        """Anchors and references are mutually exclusive -- different checkpoints."""
        plan = plan_still(prompt="x", source_asset="src", source_size=(1920, 1080),
                          reference_ids=["a", "b"])
        self.assertEqual(plan.files, [])
        self.assertTrue(any("takes no references" in d for d in plan.diagnostics))

    def test_generating_with_references_carries_tagged_files(self):
        bin_ = AssetBin([Asset("mimi", KIND_IMAGE, name="Mimi", file="m.png"),
                         Asset("bg", KIND_IMAGE, name="BG", file="b.png")])
        plan = plan_still(prompt="@Mimi in @BG", width=1344, height=736,
                          bin_=bin_, reference_ids=["mimi", "bg"])
        self.assertEqual([f.tag for f in plan.files], ["<Picture 1>", "<Picture 2>"])

    def test_reference_subset_keeps_bin_numbering(self):
        """Tags come from the whole bin, not from the selected subset -- picking
        only the second asset must still yield <Picture 2>."""
        bin_ = AssetBin([Asset("a", KIND_IMAGE, name="A", file="a.png"),
                         Asset("b", KIND_IMAGE, name="B", file="b.png")])
        plan = plan_still(prompt="x", width=1344, height=736,
                          bin_=bin_, reference_ids=["b"])
        self.assertEqual([f.tag for f in plan.files], ["<Picture 2>"])


class TestCanvasSelection(unittest.TestCase):
    def test_canvas_from_reference_overrides_explicit_size(self):
        plan = plan_still(prompt="x", width=512, height=512, source_asset="src",
                          source_size=(1920, 1080), canvas_from_reference_enabled=True)
        self.assertEqual((plan.width, plan.height), canvas_from_reference(1920, 1080))
        self.assertEqual(plan.canvas_source, "reference")

    def test_disabling_it_keeps_the_explicit_canvas(self):
        plan = plan_still(prompt="x", width=512, height=512, source_asset="src",
                          source_size=(1920, 1080), canvas_from_reference_enabled=False)
        self.assertEqual((plan.width, plan.height), (512, 512))
        self.assertEqual(plan.canvas_source, "explicit")

    def test_explicit_canvas_is_rounded_down_to_32(self):
        plan = plan_still(prompt="x", width=1000, height=700,
                          canvas_from_reference_enabled=False)
        self.assertEqual(plan.width % CANVAS_MULTIPLE, 0)
        self.assertEqual(plan.height % CANVAS_MULTIPLE, 0)
        self.assertLessEqual(plan.width, 1000)
        self.assertTrue(any("rounded down" in d for d in plan.diagnostics))

    def test_over_budget_explicit_canvas_is_warned_about(self):
        plan = plan_still(prompt="x", width=2048, height=2048,
                          canvas_from_reference_enabled=False)
        self.assertTrue(any("budget" in d for d in plan.diagnostics))

    def test_no_canvas_defaults_within_budget(self):
        plan = plan_still(prompt="x")
        self.assertLessEqual(plan.width * plan.height, MAX_PIXELS)
        self.assertEqual(plan.canvas_source, "default")

    def test_plan_is_serialisable(self):
        import json
        json.dumps(plan_still(prompt="x", source_asset="s", source_size=(1920, 1080)).to_dict())


if __name__ == "__main__":
    unittest.main()
