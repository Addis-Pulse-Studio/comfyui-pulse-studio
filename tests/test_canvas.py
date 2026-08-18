"""The preset canvases, asserted as properties rather than as a copied table.

A literal expected table would pass whatever the rule happened to produce -- it
would have passed just as happily for 1344x736, which is what shipped. So the
budget, the grid and the shape are each asserted directly, and the six literals
appear only once, in the regression test that pins the values the release changed.
"""

import unittest

from comfyui_pulse_studio.canvas import (
    ASPECT_OPTIONS,
    ASPECT_RATIOS,
    fit_canvas,
    resolution_for,
)
from comfyui_pulse_studio.constants import CANVAS_MULTIPLE, MAX_PIXELS


class TestEveryPresetIsLegal(unittest.TestCase):
    def test_both_axes_are_on_the_grid(self):
        for name in ASPECT_RATIOS:
            with self.subTest(preset=name):
                w, h = resolution_for(name, 0, 0)
                self.assertEqual(w % CANVAS_MULTIPLE, 0, "%s: %dx%d" % (name, w, h))
                self.assertEqual(h % CANVAS_MULTIPLE, 0, "%s: %dx%d" % (name, w, h))

    def test_no_preset_exceeds_the_pixel_budget(self):
        for name in ASPECT_RATIOS:
            with self.subTest(preset=name):
                w, h = resolution_for(name, 0, 0)
                self.assertLessEqual(w * h, MAX_PIXELS, "%s: %dx%d" % (name, w, h))

    def test_no_preset_wastes_more_than_a_grid_step_on_the_long_edge(self):
        """The failure the release fixed: 4.2% of the budget thrown away.

        A canvas that undershoots is legal but is the whole bug -- 1344x736 was
        legal too. Every preset now sits within one grid step of the largest long
        edge its own shape allows.
        """
        for name, (rw, rh) in ASPECT_RATIOS.items():
            with self.subTest(preset=name):
                w, h = resolution_for(name, 0, 0)
                largest = fit_canvas(rw / rh)
                self.assertEqual((w, h), largest)
                self.assertGreater(w * h, MAX_PIXELS * 0.95, "%s: %dx%d" % (name, w, h))

    def test_the_shape_the_preset_names_is_the_shape_it_returns(self):
        """Within one grid step -- the grid cannot express every ratio exactly."""
        for name, (rw, rh) in ASPECT_RATIOS.items():
            with self.subTest(preset=name):
                w, h = resolution_for(name, 0, 0)
                self.assertAlmostEqual(w / h, rw / rh, delta=0.06,
                                       msg="%s: %dx%d is %.4f, not %.4f"
                                           % (name, w, h, w / h, rw / rh))

    def test_the_exactly_expressible_presets_are_exact(self):
        """1:1, 4:3 and 3:4 land on the grid exactly, so they must not be moved.

        This is the assertion that rules out "largest short edge that fits", which
        hits the budget everywhere at the cost of turning 4:3 into 1.2857 and the
        square preset into 992x1024.
        """
        for name in ("1:1 square", "4:3 landscape", "3:4 portrait"):
            with self.subTest(preset=name):
                rw, rh = ASPECT_RATIOS[name]
                w, h = resolution_for(name, 0, 0)
                self.assertEqual(w * rh, h * rw, "%s: %dx%d is not exactly %d:%d"
                                                 % (name, w, h, rw, rh))

    def test_portrait_is_its_landscape_transposed(self):
        for landscape, portrait in (("16:9 landscape", "9:16 portrait"),
                                    ("4:3 landscape", "3:4 portrait")):
            with self.subTest(preset=portrait):
                w, h = resolution_for(landscape, 0, 0)
                self.assertEqual(resolution_for(portrait, 0, 0), (h, w))


class TestTheValuesThisReleaseChanged(unittest.TestCase):
    """The one place the literals live. Changing these is changing the canvas."""

    EXPECTED = {
        "16:9 landscape": (1344, 768),
        "9:16 portrait": (768, 1344),
        "1:1 square": (992, 992),
        "4:3 landscape": (1152, 864),
        "3:4 portrait": (864, 1152),
        "21:9 ultrawide": (1536, 672),
    }

    def test_the_table(self):
        for name, expected in self.EXPECTED.items():
            with self.subTest(preset=name):
                self.assertEqual(resolution_for(name, 0, 0), expected)

    def test_the_table_covers_every_preset(self):
        self.assertEqual(set(self.EXPECTED), set(ASPECT_RATIOS))

    def test_the_three_that_moved(self):
        """Named individually so the diff says which canvases changed."""
        self.assertEqual(resolution_for("16:9 landscape", 0, 0), (1344, 768))
        self.assertEqual(resolution_for("9:16 portrait", 0, 0), (768, 1344))
        self.assertEqual(resolution_for("21:9 ultrawide", 0, 0), (1536, 672))

    def test_sixteen_by_nine_lands_exactly_on_the_budget(self):
        w, h = resolution_for("16:9 landscape", 0, 0)
        self.assertEqual(w * h, MAX_PIXELS)


class TestCustom(unittest.TestCase):
    def test_custom_snaps_down_and_is_not_capped(self):
        """'custom' means the widgets, so a typed number is not overridden.

        Only the grid is enforced. A custom canvas over the budget is refused
        where it can be explained -- still.plan_still -- not silently resized here.
        """
        self.assertEqual(resolution_for("custom", 1000, 700), (992, 672))
        w, h = resolution_for("custom", 4096, 4096)
        self.assertEqual((w, h), (4096, 4096))
        self.assertGreater(w * h, MAX_PIXELS)

    def test_an_unknown_preset_is_treated_as_custom(self):
        self.assertEqual(resolution_for("nonsense", 1000, 700),
                         resolution_for("custom", 1000, 700))

    def test_custom_floors_at_one_grid_step(self):
        self.assertEqual(resolution_for("custom", 1, 1),
                         (CANVAS_MULTIPLE, CANVAS_MULTIPLE))

    def test_custom_leads_the_options(self):
        self.assertEqual(ASPECT_OPTIONS[0], "custom")
        self.assertEqual(ASPECT_OPTIONS[1:], list(ASPECT_RATIOS))


class TestFitCanvas(unittest.TestCase):
    def test_arbitrary_ratios_stay_inside_the_budget_and_on_the_grid(self):
        for rw, rh in [(1, 1), (2, 1), (1, 2), (37, 41), (100, 3), (3, 100),
                       (16, 9), (21, 9), (5, 4), (1000, 999)]:
            with self.subTest(ratio="%d:%d" % (rw, rh)):
                w, h = fit_canvas(rw / rh)
                self.assertLessEqual(w * h, MAX_PIXELS)
                self.assertEqual(w % CANVAS_MULTIPLE, 0)
                self.assertEqual(h % CANVAS_MULTIPLE, 0)
                self.assertGreaterEqual(w, CANVAS_MULTIPLE)
                self.assertGreaterEqual(h, CANVAS_MULTIPLE)

    def test_the_short_edge_is_never_longer_than_the_long_edge(self):
        """The square case is the one that catches a naive "largest that fits"."""
        for rw, rh in [(1, 1), (16, 9), (9, 16), (4, 3), (3, 4), (100, 99), (99, 100)]:
            with self.subTest(ratio="%d:%d" % (rw, rh)):
                w, h = fit_canvas(rw / rh)
                if rw >= rh:
                    self.assertGreaterEqual(w, h)
                else:
                    self.assertGreaterEqual(h, w)

    def test_rejects_a_nonpositive_ratio(self):
        for bad in (0, -1, -0.5):
            with self.subTest(ratio=bad):
                with self.assertRaises(ValueError):
                    fit_canvas(bad)


if __name__ == "__main__":
    unittest.main()
