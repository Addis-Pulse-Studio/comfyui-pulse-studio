"""Asset Bin operations: live state, renumber preview, name safety."""

import unittest

from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO, Asset, AssetBin, BudgetError
from comfyui_pulse_studio.binops import (
    apply_operation,
    bin_state,
    name_problems,
    preview_change,
    suggest_name,
)
from comfyui_pulse_studio.constants import MAX_REF_IMAGES


def img(i, name=None):
    return Asset("i%d" % i, KIND_IMAGE, name=name or "Image %d" % i, file="i%d.png" % i)


class TestBinState(unittest.TestCase):
    def test_state_carries_live_tags(self):
        state = bin_state(AssetBin([img(1), img(2)]))
        self.assertEqual([r["tag"] for r in state["assets"]], ["<Picture 1>", "<Picture 2>"])

    def test_meter_is_present(self):
        state = bin_state(AssetBin([img(1)]))
        self.assertEqual(state["budget"]["meter"], "1/9 images | 0/3 videos | 0/3 audio | 1/12 files")
        self.assertTrue(state["budget"]["ok"])

    def test_video_rows_expose_soundtrack_tag(self):
        bin_ = AssetBin([Asset("v", KIND_VIDEO, name="V", trim_end=5.0, include_audio=True)])
        row = bin_state(bin_)["assets"][0]
        self.assertEqual(row["soundtrack_tag"], "<Audio 1>")
        self.assertTrue(row["include_audio"])

    def test_state_is_json_serialisable(self):
        import json
        json.dumps(bin_state(AssetBin([img(1), Asset("a", KIND_AUDIO, name="A")])))


class TestRenumberPreview(unittest.TestCase):
    def test_move_reports_every_shifted_tag(self):
        bin_ = AssetBin([img(1), img(2), img(3)])
        deltas, err = preview_change(bin_, "move", asset_id="i3", new_index=0)
        self.assertIsNone(err)
        moved = {d.asset_id: (d.before, d.after) for d in deltas}
        self.assertEqual(moved["i3"], ("<Picture 3>", "<Picture 1>"))
        self.assertEqual(moved["i1"], ("<Picture 1>", "<Picture 2>"))
        self.assertEqual(moved["i2"], ("<Picture 2>", "<Picture 3>"))

    def test_preview_does_not_mutate_the_live_bin(self):
        bin_ = AssetBin([img(1), img(2)])
        preview_change(bin_, "move", asset_id="i2", new_index=0)
        self.assertEqual(bin_.tag_map().tag("i1"), "<Picture 1>")
        self.assertEqual(len(bin_), 2)

    def test_removal_shows_the_tag_disappearing(self):
        bin_ = AssetBin([img(1), img(2)])
        deltas, err = preview_change(bin_, "remove", asset_id="i1")
        self.assertIsNone(err)
        by_id = {d.asset_id: d for d in deltas}
        self.assertTrue(by_id["i1"].removed)
        self.assertEqual(by_id["i2"].after, "<Picture 1>")

    def test_add_at_front_shows_the_cascade(self):
        bin_ = AssetBin([img(1), img(2)])
        deltas, err = preview_change(
            bin_, "add", asset={"id": "new", "kind": KIND_IMAGE, "name": "New"}, index=0)
        self.assertIsNone(err)
        by_id = {d.asset_id: d for d in deltas}
        self.assertTrue(by_id["new"].added)
        self.assertEqual(by_id["new"].after, "<Picture 1>")
        self.assertEqual(by_id["i1"].after, "<Picture 2>")

    def test_appending_at_the_end_renumbers_nothing(self):
        """The safe edit. Worth asserting so the UI can stay quiet for it."""
        bin_ = AssetBin([img(1), img(2)])
        deltas, err = preview_change(
            bin_, "add", asset={"id": "new", "kind": KIND_IMAGE, "name": "New"})
        self.assertIsNone(err)
        self.assertEqual([d.asset_id for d in deltas], ["new"])
        self.assertTrue(deltas[0].added)

    def test_soundtrack_toggle_preview_shows_audio_cascade(self):
        bin_ = AssetBin([Asset("v", KIND_VIDEO, name="V", trim_end=5.0),
                         Asset("a", KIND_AUDIO, name="A")])
        deltas, err = preview_change(bin_, "set_include_audio", asset_id="v", include=True)
        self.assertIsNone(err)
        by_id = {d.asset_id: d for d in deltas}
        self.assertEqual(by_id["v#soundtrack"].after, "<Audio 1>")
        self.assertEqual(by_id["a"].before, "<Audio 1>")
        self.assertEqual(by_id["a"].after, "<Audio 2>")

    def test_refused_operation_returns_an_error_not_deltas(self):
        bin_ = AssetBin([img(i) for i in range(MAX_REF_IMAGES)])
        deltas, err = preview_change(
            bin_, "add", asset={"id": "extra", "kind": KIND_IMAGE, "name": "Extra"})
        self.assertEqual(deltas, [])
        self.assertIn("images", err)

    def test_unknown_operation_rejected(self):
        with self.assertRaises(ValueError):
            apply_operation(AssetBin(), "teleport")


class TestNames(unittest.TestCase):
    def test_rename_succeeds(self):
        bin_ = AssetBin([img(1)])
        apply_operation(bin_, "rename", asset_id="i1", name="Mimi")
        self.assertEqual(bin_.get("i1").name, "Mimi")
        self.assertIsNotNone(bin_.find_by_name("Mimi"))

    def test_rename_to_a_taken_name_is_refused(self):
        """Two assets sharing a name makes @Name ambiguous, and an ambiguous
        reference resolves to nothing rather than guessing."""
        bin_ = AssetBin([img(1, "Mimi"), img(2, "Kaleb")])
        with self.assertRaises(ValueError):
            apply_operation(bin_, "rename", asset_id="i2", name="Mimi")
        self.assertEqual(bin_.get("i2").name, "Kaleb")

    def test_rename_is_case_insensitive_about_collisions(self):
        bin_ = AssetBin([img(1, "Mimi"), img(2, "Kaleb")])
        with self.assertRaises(ValueError):
            apply_operation(bin_, "rename", asset_id="i2", name="mimi")

    def test_renaming_to_its_own_name_is_allowed(self):
        bin_ = AssetBin([img(1, "Mimi")])
        apply_operation(bin_, "rename", asset_id="i1", name="Mimi")
        self.assertEqual(bin_.get("i1").name, "Mimi")

    def test_empty_name_refused(self):
        bin_ = AssetBin([img(1, "Mimi")])
        with self.assertRaises(ValueError):
            apply_operation(bin_, "rename", asset_id="i1", name="   ")

    def test_duplicate_names_are_reported_not_corrected(self):
        bin_ = AssetBin([Asset("x", KIND_IMAGE, name="Mimi"),
                         Asset("y", KIND_IMAGE, name="Mimi")])
        problems = name_problems(bin_)
        self.assertEqual(len(problems), 1)
        self.assertIn("will not resolve", problems[0]["problem"])

    def test_clean_bin_has_no_name_problems(self):
        self.assertEqual(name_problems(AssetBin([img(1, "Mimi"), img(2, "Kaleb")])), [])

    def test_suggest_name_avoids_collisions(self):
        bin_ = AssetBin([img(1, "Mimi")])
        self.assertEqual(suggest_name(bin_, "Kaleb"), "Kaleb")
        self.assertEqual(suggest_name(bin_, "Mimi"), "Mimi 2")

    def test_suggest_name_keeps_counting(self):
        bin_ = AssetBin([img(1, "Mimi"), img(2, "Mimi 2")])
        self.assertEqual(suggest_name(bin_, "Mimi"), "Mimi 3")


class TestDropAliases(unittest.TestCase):
    """A short alias is assigned on drop so the user never types a filename."""

    def test_aliases_count_per_kind_from_one(self):
        from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_VIDEO
        from comfyui_pulse_studio.binops import suggest_alias
        bin_ = AssetBin()
        self.assertEqual(suggest_alias(bin_, KIND_IMAGE), "Image1")
        bin_.add(Asset("x", KIND_IMAGE, name="Image1"))
        self.assertEqual(suggest_alias(bin_, KIND_IMAGE), "Image2")
        # A different kind starts its own run.
        self.assertEqual(suggest_alias(bin_, KIND_VIDEO), "Video1")
        self.assertEqual(suggest_alias(bin_, KIND_AUDIO), "Audio1")

    def test_alias_skips_a_name_the_user_already_took(self):
        from comfyui_pulse_studio.binops import suggest_alias
        bin_ = AssetBin([Asset("x", KIND_IMAGE, name="Image1"),
                         Asset("y", KIND_IMAGE, name="Image2")])
        self.assertEqual(suggest_alias(bin_, KIND_IMAGE), "Image3")

    def test_alias_is_immediately_usable_as_a_reference(self):
        """The point of the alias: @Image1 must resolve with no renaming first."""
        from comfyui_pulse_studio.assets import AssetBin as Bin
        from comfyui_pulse_studio.binops import suggest_alias
        from comfyui_pulse_studio.compiler import resolve_references
        bin_ = Bin()
        name = suggest_alias(bin_, KIND_IMAGE)
        bin_.add(Asset("a1", KIND_IMAGE, name=name, file="a.png"))
        text, diagnostics = resolve_references("@%s walks in" % name, bin_.tag_map(), bin_)
        self.assertEqual(diagnostics, [])
        self.assertIn("<Picture 1>", text)


if __name__ == "__main__":
    unittest.main()
