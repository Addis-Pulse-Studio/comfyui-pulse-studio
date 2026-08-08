"""Asset bin: budget enforcement and tag renumbering under add/remove/reorder.

Tag numbering is the single highest-value correctness property in the project.
Silent misnumbering produces plausible-looking wrong output -- the render
succeeds, describes the wrong picture, and nothing anywhere reports an error.
These tests pin the numbering to the rule read out of ComfyUI's own source.
"""

import unittest

from omni_director.assets import (
    KIND_AUDIO,
    KIND_IMAGE,
    KIND_VIDEO,
    Asset,
    AssetBin,
    BudgetError,
)
from omni_director.constants import (
    MAX_REF_AUDIOS,
    MAX_REF_FILES_TOTAL,
    MAX_REF_IMAGES,
    MAX_REF_VIDEOS,
)


def img(i, **kw):
    return Asset("i%d" % i, KIND_IMAGE, name="Image %d" % i, file="i%d.png" % i, **kw)


def vid(i, **kw):
    kw.setdefault("trim_end", 5.0)
    return Asset("v%d" % i, KIND_VIDEO, name="Video %d" % i, file="v%d.mp4" % i, **kw)


def aud(i, **kw):
    return Asset("a%d" % i, KIND_AUDIO, name="Audio %d" % i, file="a%d.wav" % i, **kw)


class TestTagOrdering(unittest.TestCase):
    """The rule, from comfy/text_encoders/minimax.py + nodes_minimax_h3.py:
    images first, then each video preceded by its own soundtrack, then
    standalone audio. Counters are 1-based and per type."""

    def test_images_number_in_bin_order(self):
        bin_ = AssetBin([img(1), img(2), img(3)])
        tm = bin_.tag_map()
        self.assertEqual(tm.tag("i1"), "<Picture 1>")
        self.assertEqual(tm.tag("i2"), "<Picture 2>")
        self.assertEqual(tm.tag("i3"), "<Picture 3>")

    def test_each_type_counts_independently(self):
        bin_ = AssetBin([img(1), vid(1), aud(1)])
        tm = bin_.tag_map()
        self.assertEqual(tm.tag("i1"), "<Picture 1>")
        self.assertEqual(tm.tag("v1"), "<Video 1>")
        self.assertEqual(tm.tag("a1"), "<Audio 1>")

    def test_kind_grouping_beats_insertion_order(self):
        """Sockets are grouped by type, so an audio dropped between two images
        does not interleave -- it still numbers after every image."""
        bin_ = AssetBin([img(1), aud(1), img(2)])
        tm = bin_.tag_map()
        self.assertEqual(tm.tag("i1"), "<Picture 1>")
        self.assertEqual(tm.tag("i2"), "<Picture 2>")
        self.assertEqual(tm.tag("a1"), "<Audio 1>")

    def test_video_soundtrack_claims_an_audio_ordinal_before_standalone_audio(self):
        """The subtle one. A video's paired soundtrack is appended to ref_items
        BEFORE its own <Video k> entry and before every standalone audio, so it
        takes <Audio 1> and pushes the user's audio to <Audio 2>."""
        bin_ = AssetBin([vid(1, include_audio=True), aud(1)])
        tm = bin_.tag_map()
        self.assertEqual(tm.audio_tag_for_video("v1"), "<Audio 1>")
        self.assertEqual(tm.tag("v1"), "<Video 1>")
        self.assertEqual(tm.tag("a1"), "<Audio 2>")

    def test_multiple_soundtracks_number_in_video_order(self):
        bin_ = AssetBin([vid(1, include_audio=True), vid(2), vid(3, include_audio=True), aud(1)])
        tm = bin_.tag_map()
        self.assertEqual(tm.audio_tag_for_video("v1"), "<Audio 1>")
        self.assertIsNone(tm.audio_tag_for_video("v2"))
        self.assertEqual(tm.audio_tag_for_video("v3"), "<Audio 2>")
        self.assertEqual(tm.tag("a1"), "<Audio 3>")
        self.assertEqual(tm.tag("v1"), "<Video 1>")
        self.assertEqual(tm.tag("v2"), "<Video 2>")
        self.assertEqual(tm.tag("v3"), "<Video 3>")

    def test_socket_names_match_the_stock_node_kwargs(self):
        bin_ = AssetBin([img(1), img(2), vid(1, include_audio=True), aud(1)])
        tm = bin_.tag_map()
        self.assertEqual(tm.sockets["ref_image_0"], "i1")
        self.assertEqual(tm.sockets["ref_image_1"], "i2")
        self.assertEqual(tm.sockets["ref_video_0"], "v1")
        self.assertEqual(tm.sockets["ref_video_audio_0"], "v1")
        self.assertEqual(tm.sockets["ref_audio_0"], "a1")

    def test_soundtrack_socket_index_pairs_with_its_video(self):
        """Core pairs ref_video_audio_N to ref_video_N by the trailing index."""
        bin_ = AssetBin([vid(1), vid(2, include_audio=True)])
        tm = bin_.tag_map()
        self.assertEqual(tm.sockets["ref_video_1"], "v2")
        self.assertEqual(tm.sockets["ref_video_audio_1"], "v2")
        self.assertNotIn("ref_video_audio_0", tm.sockets)


class TestRenumbering(unittest.TestCase):
    """Ordinals must track every edit, because nothing stores them."""

    def test_insert_at_front_shifts_everything_after(self):
        bin_ = AssetBin([img(1), img(2)])
        self.assertEqual(bin_.tag_map().tag("i1"), "<Picture 1>")
        bin_.add(img(9), index=0)
        tm = bin_.tag_map()
        self.assertEqual(tm.tag("i9"), "<Picture 1>")
        self.assertEqual(tm.tag("i1"), "<Picture 2>")
        self.assertEqual(tm.tag("i2"), "<Picture 3>")

    def test_remove_from_middle_closes_the_gap(self):
        bin_ = AssetBin([img(1), img(2), img(3)])
        bin_.remove("i2")
        tm = bin_.tag_map()
        self.assertEqual(tm.tag("i1"), "<Picture 1>")
        self.assertEqual(tm.tag("i3"), "<Picture 2>")
        self.assertNotIn("i2", tm.by_id)

    def test_reorder_swaps_ordinals(self):
        bin_ = AssetBin([img(1), img(2), img(3)])
        bin_.move("i3", 0)
        tm = bin_.tag_map()
        self.assertEqual(tm.tag("i3"), "<Picture 1>")
        self.assertEqual(tm.tag("i1"), "<Picture 2>")
        self.assertEqual(tm.tag("i2"), "<Picture 3>")

    def test_reorder_to_end(self):
        bin_ = AssetBin([img(1), img(2), img(3)])
        bin_.move("i1", 99)
        tm = bin_.tag_map()
        self.assertEqual(tm.tag("i2"), "<Picture 1>")
        self.assertEqual(tm.tag("i3"), "<Picture 2>")
        self.assertEqual(tm.tag("i1"), "<Picture 3>")

    def test_removing_an_image_does_not_touch_video_or_audio_numbering(self):
        bin_ = AssetBin([img(1), img(2), vid(1), aud(1)])
        before = bin_.tag_map()
        bin_.remove("i1")
        after = bin_.tag_map()
        self.assertEqual(before.tag("v1"), after.tag("v1"))
        self.assertEqual(before.tag("a1"), after.tag("a1"))
        self.assertEqual(after.tag("i2"), "<Picture 1>")

    def test_toggling_a_soundtrack_renumbers_every_standalone_audio(self):
        """The least obvious renumbering trigger in the system."""
        bin_ = AssetBin([vid(1), aud(1), aud(2)])
        self.assertEqual(bin_.tag_map().tag("a1"), "<Audio 1>")
        self.assertEqual(bin_.tag_map().tag("a2"), "<Audio 2>")
        bin_.set_include_audio("v1", True)
        tm = bin_.tag_map()
        self.assertEqual(tm.audio_tag_for_video("v1"), "<Audio 1>")
        self.assertEqual(tm.tag("a1"), "<Audio 2>")
        self.assertEqual(tm.tag("a2"), "<Audio 3>")
        bin_.set_include_audio("v1", False)
        tm = bin_.tag_map()
        self.assertEqual(tm.tag("a1"), "<Audio 1>")

    def test_tag_map_is_recomputed_not_cached(self):
        bin_ = AssetBin([img(1)])
        first = bin_.tag_map()
        bin_.add(img(2), index=0)
        second = bin_.tag_map()
        self.assertEqual(first.tag("i1"), "<Picture 1>")
        self.assertEqual(second.tag("i1"), "<Picture 2>")

    def test_ordinals_are_dense_and_gapless(self):
        bin_ = AssetBin([img(1), img(2), img(3), img(4)])
        bin_.remove("i2")
        bin_.remove("i3")
        tags = [bin_.tag_map().tag(a.asset_id) for a in bin_.by_kind(KIND_IMAGE)]
        self.assertEqual(tags, ["<Picture 1>", "<Picture 2>"])


class TestBudget(unittest.TestCase):
    def test_nine_images_fit_and_ten_do_not(self):
        bin_ = AssetBin([img(i) for i in range(MAX_REF_IMAGES)])
        self.assertTrue(bin_.budget().ok)
        with self.assertRaises(BudgetError):
            bin_.add(img(99))
        self.assertEqual(len(bin_), MAX_REF_IMAGES)  # rejected drop left the bin intact

    def test_three_videos_fit_and_four_do_not(self):
        bin_ = AssetBin([vid(i) for i in range(MAX_REF_VIDEOS)])
        with self.assertRaises(BudgetError):
            bin_.add(vid(99))

    def test_three_audios_fit_and_four_do_not(self):
        bin_ = AssetBin([aud(i) for i in range(MAX_REF_AUDIOS)])
        with self.assertRaises(BudgetError):
            bin_.add(aud(99))

    def test_twelve_file_cap_binds_before_per_type_caps(self):
        """9 + 3 + 3 = 15 sockets exist but only 12 files may be presented."""
        bin_ = AssetBin([img(i) for i in range(MAX_REF_IMAGES)] + [vid(i) for i in range(MAX_REF_VIDEOS)])
        self.assertEqual(bin_.budget().files, MAX_REF_FILES_TOTAL)
        self.assertTrue(bin_.budget().ok)
        # An audio slot is free, but no file slot is.
        with self.assertRaises(BudgetError) as ctx:
            bin_.add(aud(1))
        self.assertIn("reference files", str(ctx.exception))

    def test_can_add_reports_without_mutating(self):
        bin_ = AssetBin([img(i) for i in range(MAX_REF_IMAGES)])
        ok, reason = bin_.can_add(img(99))
        self.assertFalse(ok)
        self.assertIn("images", reason)
        self.assertEqual(len(bin_), MAX_REF_IMAGES)

    def test_can_add_accepts_a_legal_drop(self):
        bin_ = AssetBin([img(1)])
        ok, reason = bin_.can_add(img(2))
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_duplicate_id_rejected(self):
        bin_ = AssetBin([img(1)])
        ok, reason = bin_.can_add(img(1))
        self.assertFalse(ok)
        with self.assertRaises(BudgetError):
            bin_.add(img(1))

    def test_constructing_an_over_budget_bin_raises(self):
        with self.assertRaises(BudgetError):
            AssetBin([img(i) for i in range(MAX_REF_IMAGES + 1)])

    def test_synthetic_assets_take_sockets_but_not_file_slots(self):
        """Carry-over references are generated, never uploaded; counting them as
        files would make the meter lie to the user."""
        assets = [img(i) for i in range(MAX_REF_IMAGES)] + [vid(i) for i in range(MAX_REF_VIDEOS)]
        bin_ = AssetBin(assets)
        self.assertEqual(bin_.budget().files, MAX_REF_FILES_TOTAL)
        carry = Asset("__carry_audio__", KIND_AUDIO, name="carry", synthetic=True)
        bin_.add(carry)
        report = bin_.budget()
        self.assertEqual(report.files, MAX_REF_FILES_TOTAL)  # unchanged
        self.assertEqual(report.audios, 1)                   # socket still consumed
        self.assertTrue(report.ok)

    def test_reference_video_too_short_is_refused_at_drop_time(self):
        """H3 accepts 2-15s reference clips. Refusing the drop is the whole point
        of the meter -- failing at queue time instead wastes the user's setup."""
        bin_ = AssetBin()
        with self.assertRaises(BudgetError) as ctx:
            bin_.add(Asset("v_short", KIND_VIDEO, name="Blink", trim_start=0.0, trim_end=1.0))
        self.assertIn("2-15s", str(ctx.exception))
        self.assertEqual(len(bin_), 0)

    def test_reference_video_too_long_is_refused(self):
        bin_ = AssetBin()
        with self.assertRaises(BudgetError):
            bin_.add(Asset("v_long", KIND_VIDEO, name="Epic", trim_start=0.0, trim_end=20.0))

    def test_trimming_an_existing_video_out_of_range_is_reported(self):
        """A trim edited after the drop still has to surface, since the drop-time
        guard never saw it."""
        bin_ = AssetBin([vid(1)])
        self.assertTrue(bin_.budget().ok)
        bin_.get("v1").trim_end = 20.0
        self.assertFalse(bin_.budget().ok)

    def test_open_ended_trim_is_not_flagged(self):
        """Duration is unknown until the file is decoded; do not guess."""
        bin_ = AssetBin()
        bin_.add(Asset("v_open", KIND_VIDEO, name="Unknown", trim_start=0.0, trim_end=None))
        self.assertTrue(bin_.budget().ok)

    def test_meter_string(self):
        bin_ = AssetBin([img(1), img(2), vid(1), aud(1)])
        self.assertEqual(bin_.budget().meter(),
                         "2/9 images | 1/3 videos | 1/3 audio | 4/12 files")

    def test_set_include_audio_rejects_a_non_video(self):
        bin_ = AssetBin([img(1)])
        with self.assertRaises(ValueError):
            bin_.set_include_audio("i1", True)


class TestLookup(unittest.TestCase):
    def test_find_by_name_is_case_insensitive(self):
        bin_ = AssetBin([Asset("x", KIND_IMAGE, name="Mimi")])
        self.assertIsNotNone(bin_.find_by_name("mimi"))
        self.assertIsNotNone(bin_.find_by_name("  MIMI "))

    def test_ambiguous_name_resolves_to_nothing(self):
        """Two assets sharing a name must not silently pick one."""
        bin_ = AssetBin([Asset("x", KIND_IMAGE, name="Mimi"),
                         Asset("y", KIND_IMAGE, name="Mimi")])
        self.assertIsNone(bin_.find_by_name("Mimi"))

    def test_roundtrip_serialisation(self):
        bin_ = AssetBin([img(1), vid(1, include_audio=True), aud(1)])
        restored = AssetBin.from_list(bin_.to_list())
        self.assertEqual(restored.tag_map().by_id, bin_.tag_map().by_id)
        self.assertEqual(restored.tag_map().sockets, bin_.tag_map().sockets)


if __name__ == "__main__":
    unittest.main()
