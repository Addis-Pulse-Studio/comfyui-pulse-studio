"""The PULSE_TIMELINE document, stable ids, window seeds and continuity.

Spec §3, §6, §11.
"""

import json
import unittest
import uuid

from comfyui_pulse_studio.pulse_timeline import (
    CONTINUITY_INHERIT,
    CONTINUITY_KEYFRAME_PAIRS,
    CONTINUITY_LAST_FRAME,
    CONTINUITY_NONE,
    NODE_VERSION,
    SCHEMA,
    ContinuityError,
    SideChannel,
    build_timeline,
    canonical_json,
    check_continuity,
    global_block,
    is_pulse_timeline,
    new_shot_id,
    ref_descriptor,
    resolve_continuity,
    shot_block,
    shots_of_window,
    socket_asset_id,
    socket_slot_of,
    text_shot_id,
    window_block,
    window_seed,
)


class TestShotIdentity(unittest.TestCase):
    """§6.1-2 -- an id that survives the user rewriting the shot."""

    def test_a_new_shot_id_is_a_uuid(self):
        value = new_shot_id()
        uuid.UUID(value)
        self.assertNotEqual(value, new_shot_id())

    def test_text_shot_ids_are_content_derived_and_sixteen_chars(self):
        value = text_shot_id("The Delivery", "she walks in")
        self.assertEqual(len(value), 16)
        self.assertEqual(value, text_shot_id("The Delivery", "she walks in"))

    def test_editing_a_text_shot_gives_it_a_new_id(self):
        """Deliberate: it re-seeds and re-renders exactly the window holding it,
        and leaves every other window's cache intact."""
        self.assertNotEqual(text_shot_id("A", "she walks in"),
                            text_shot_id("A", "she runs in"))

    def test_the_separator_cannot_be_smuggled_across_fields(self):
        self.assertNotEqual(text_shot_id("a|b", "c"), text_shot_id("a", "b|c"))


class TestWindowSeed(unittest.TestCase):
    """§6.3 -- seeds key on the shot set, never on position."""

    def test_deterministic_in_the_shot_set(self):
        self.assertEqual(window_seed(1234, ["a", "b"]), window_seed(1234, ["a", "b"]))

    def test_independent_of_where_the_window_sits(self):
        """The whole point. The same shots at the same base seed produce the same
        window whether they are window 0 or window 9."""
        self.assertEqual(window_seed(7, ["x", "y"]), window_seed(7, ["x", "y"]))

    def test_a_different_shot_set_gives_a_different_seed(self):
        self.assertNotEqual(window_seed(7, ["x", "y"]), window_seed(7, ["x", "z"]))

    def test_reordering_shots_within_a_window_changes_the_seed(self):
        """Order is part of the content: the same two shots in the other order is
        a different window, and must not reuse the cached one."""
        self.assertNotEqual(window_seed(7, ["x", "y"]), window_seed(7, ["y", "x"]))

    def test_the_base_seed_changes_every_window(self):
        self.assertNotEqual(window_seed(1, ["x"]), window_seed(2, ["x"]))

    def test_stays_inside_31_bits(self):
        for base in (0, 1, 2 ** 63 - 1, 0xFFFFFFFFFFFFFFFF):
            value = window_seed(base, ["a", "b", "c"])
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, 2 ** 31)


class TestCanonicalJson(unittest.TestCase):
    def test_key_order_does_not_change_the_bytes(self):
        self.assertEqual(canonical_json({"b": 1, "a": 2}),
                         canonical_json({"a": 2, "b": 1}))

    def test_no_incidental_whitespace(self):
        self.assertEqual(canonical_json({"a": 1, "b": [1, 2]}), '{"a":1,"b":[1,2]}')


class TestDocument(unittest.TestCase):
    """§3 -- the document's shape."""

    def build(self):
        shots = [shot_block("s1", 0, label="A", visual="x", duration_seconds=5.0),
                 shot_block("s2", 1, label="B", visual="y", duration_seconds=5.0)]
        windows = [window_block(0, ["s1", "s2"], 245, seed=99)]
        return build_timeline(
            global_block(style="noir", raw="style: noir"),
            [ref_descriptor(1, "image", "Mimi", "bin", file="mimi.png", sha256="abc")],
            shots, windows, {"images": 1, "videos": 0, "audio": 0, "total": 1},
            warnings=["a note"])

    def test_carries_the_schema_and_version(self):
        document = self.build()
        self.assertEqual(document["schema"], SCHEMA)
        self.assertEqual(document["node_version"], NODE_VERSION)
        self.assertTrue(is_pulse_timeline(document))

    def test_is_json_serialisable(self):
        """Not decorative: the document is hashed for the cache key and written
        into the manifest. A tensor in here would break both."""
        json.dumps(self.build())

    def test_no_tensors_ride_in_the_document(self):
        document = self.build()
        for shot in document["shots"]:
            self.assertIsNone(shot["start_image_ref"])
        self.assertEqual(document["refs"]["global"][0]["sha256"], "abc")

    def test_cache_key_starts_empty(self):
        """§7.1 folds the model and patch fingerprints in, and the compiler holds
        neither -- so it cannot fill this and does not pretend to."""
        self.assertIsNone(self.build()["windows"][0]["cache_key"])

    def test_shots_of_window_follows_the_window_order(self):
        document = self.build()
        document["windows"][0]["shot_ids"] = ["s2", "s1"]
        self.assertEqual([s["shot_id"] for s in shots_of_window(document,
                                                                document["windows"][0])],
                         ["s2", "s1"])

    def test_an_unknown_shot_id_is_skipped_rather_than_raising(self):
        document = self.build()
        document["windows"][0]["shot_ids"] = ["s1", "ghost"]
        self.assertEqual(len(shots_of_window(document, document["windows"][0])), 1)


class TestSocketAssetIds(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(socket_slot_of(socket_asset_id("shot.abc.ref_image_1")),
                         "shot.abc.ref_image_1")

    def test_a_bin_asset_is_not_a_socket_asset(self):
        self.assertIsNone(socket_slot_of("mimi"))


class TestSideChannel(unittest.TestCase):
    def test_tensors_and_digests_are_kept_apart_from_the_document(self):
        side = SideChannel()
        side.put("slot", "a-tensor", digest="deadbeef")
        self.assertEqual(side.get("slot"), "a-tensor")
        self.assertEqual(side.digest("slot"), "deadbeef")

    def test_an_unknown_slot_is_none_rather_than_a_key_error(self):
        self.assertIsNone(SideChannel().get("nope"))
        self.assertEqual(SideChannel().digest("nope"), "")


class TestContinuityResolution(unittest.TestCase):
    """§11 -- inherit, and the modes that need a second checkpoint."""

    def test_inherit_takes_the_project_setting(self):
        self.assertEqual(resolve_continuity(CONTINUITY_LAST_FRAME, CONTINUITY_INHERIT),
                         CONTINUITY_LAST_FRAME)
        self.assertEqual(resolve_continuity(CONTINUITY_NONE, None), CONTINUITY_NONE)

    def test_a_shot_override_wins(self):
        self.assertEqual(resolve_continuity(CONTINUITY_NONE, CONTINUITY_LAST_FRAME),
                         CONTINUITY_LAST_FRAME)

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ContinuityError):
            resolve_continuity(CONTINUITY_NONE, "crossfade")


class TestContinuityValidation(unittest.TestCase):
    """§11 -- fail at compile time, never fall back silently."""

    def shots(self, count=2, continuity=CONTINUITY_INHERIT, start_image=None):
        return [shot_block("s%d" % i, i, label="S%d" % i, visual="x",
                           continuity=continuity, start_image_ref=start_image)
                for i in range(count)]

    def test_none_needs_nothing(self):
        self.assertEqual(check_continuity(self.shots(), CONTINUITY_NONE, False), [])

    def test_last_frame_carry_without_fl2va_is_fatal(self):
        problems = check_continuity(self.shots(), CONTINUITY_LAST_FRAME, False)
        self.assertEqual(len(problems), 1)
        self.assertIn("model_fl2va", problems[0])

    def test_last_frame_carry_with_fl2va_passes(self):
        self.assertEqual(check_continuity(self.shots(), CONTINUITY_LAST_FRAME, True), [])

    def test_a_single_shot_override_is_enough_to_require_fl2va(self):
        shots = self.shots()
        shots[1]["continuity"] = CONTINUITY_LAST_FRAME
        problems = check_continuity(shots, CONTINUITY_NONE, False)
        self.assertTrue(any("model_fl2va" in p for p in problems))

    def test_keyframe_pairs_needs_a_start_image_on_every_shot_in_the_chain(self):
        problems = check_continuity(self.shots(2, CONTINUITY_KEYFRAME_PAIRS),
                                    CONTINUITY_NONE, True)
        self.assertEqual(len(problems), 2)
        for problem in problems:
            self.assertIn("start_image", problem)

    def test_keyframe_pairs_with_start_images_passes(self):
        shots = self.shots(2, CONTINUITY_KEYFRAME_PAIRS, start_image="tensor_slot_0")
        self.assertEqual(check_continuity(shots, CONTINUITY_NONE, True), [])

    def test_an_unknown_project_mode_is_refused(self):
        self.assertTrue(check_continuity(self.shots(), "dissolve", True))


if __name__ == "__main__":
    unittest.main()
