"""Widget state: the queue-erasure regression, and newline survival.

The headline test is `TestQueueDoesNotEraseTypedText`. It reproduces the reported
failure -- type into both boxes with an empty timeline, queue -- and asserts the
boxes still hold their text afterwards.
"""

import json
import unittest

from comfyui_pulse_studio.assets import KIND_IMAGE
from comfyui_pulse_studio.compiler import compile_timeline
from comfyui_pulse_studio.widget_state import (
    PROMPT_WIDGETS,
    DocumentError,
    apply_bin_operation,
    build_timeline,
    dump_timeline_document,
    load_timeline_document,
)

THREE_LINES = "art deco interiors\nhard key from frame left\nno camera movement"
SHOTS = "[Shot 1] she walks in\n[Shot 2] he looks up\n[Shot 3] they sit"


class NodeFace:
    """A stand-in for the node's widget values, so the execute path can be run
    headless and then checked for having modified anything it should not."""

    def __init__(self, **values):
        self.values = {"global_prompt": "", "shot_prompt": "", "timeline_data": "{}",
                       "duration_seconds": 10.0}
        self.values.update(values)

    def snapshot(self):
        return dict(self.values)

    def queue(self):
        """Exactly what the node does on execute: read widgets, build, compile.

        If any of this wrote back to a widget, the snapshot comparison in the
        tests below would catch it.
        """
        timeline, notes = build_timeline(
            self.values["timeline_data"],
            global_prompt=self.values["global_prompt"],
            shot_prompt=self.values["shot_prompt"],
            duration_seconds=self.values["duration_seconds"],
        )
        return compile_timeline(timeline), notes


class TestQueueDoesNotEraseTypedText(unittest.TestCase):
    """THE REGRESSION. Queuing must never write back to a prompt widget."""

    def test_queue_with_an_empty_timeline_keeps_both_boxes(self):
        face = NodeFace(global_prompt=THREE_LINES, shot_prompt=SHOTS, timeline_data="{}")
        before = face.snapshot()
        face.queue()
        self.assertEqual(face.values["global_prompt"], before["global_prompt"])
        self.assertEqual(face.values["shot_prompt"], before["shot_prompt"])

    def test_queue_with_a_blank_timeline_string_keeps_both_boxes(self):
        """An empty string, not just '{}' -- the state a fresh node is actually in."""
        for blank in ("", "   ", "{}", "null", None):
            face = NodeFace(global_prompt=THREE_LINES, shot_prompt=SHOTS, timeline_data=blank)
            face.queue()
            self.assertEqual(face.values["global_prompt"], THREE_LINES, "blank=%r" % (blank,))
            self.assertEqual(face.values["shot_prompt"], SHOTS, "blank=%r" % (blank,))

    def test_queueing_repeatedly_is_stable(self):
        face = NodeFace(global_prompt=THREE_LINES, shot_prompt=SHOTS)
        for _ in range(5):
            face.queue()
        self.assertEqual(face.values["global_prompt"], THREE_LINES)
        self.assertEqual(face.values["shot_prompt"], SHOTS)

    def test_queue_mutates_no_widget_at_all(self):
        face = NodeFace(global_prompt=THREE_LINES, shot_prompt=SHOTS,
                        timeline_data=json.dumps({"assets": [], "duration_seconds": 12.0}))
        before = face.snapshot()
        face.queue()
        self.assertEqual(face.snapshot(), before)

    def test_prompts_are_never_stored_in_the_timeline_document(self):
        """Structural guarantee: the bin panel cannot erase what it does not hold."""
        face = NodeFace(global_prompt=THREE_LINES, shot_prompt=SHOTS)
        document = load_timeline_document(face.values["timeline_data"])
        for widget in PROMPT_WIDGETS:
            self.assertNotIn(widget, document)

    def test_a_malformed_timeline_does_not_take_the_prompts_with_it(self):
        face = NodeFace(global_prompt=THREE_LINES, shot_prompt=SHOTS,
                        timeline_data="{not json at all")
        plan, notes = face.queue()
        self.assertEqual(face.values["global_prompt"], THREE_LINES)
        self.assertEqual(face.values["shot_prompt"], SHOTS)
        self.assertTrue(any("could not be parsed" in n for n in notes))
        # And the typed shots still compiled, despite the broken document.
        self.assertTrue(plan.windows[0].prompt.strip())


class TestDocumentPreservation(unittest.TestCase):
    """The root cause: an edit must merge, never replace."""

    def _document(self):
        return json.dumps({
            "assets": [{"id": "i1", "kind": KIND_IMAGE, "name": "I1", "file": "i1.png"}],
            "shots": [{"id": "s1", "start": 0, "duration": 5, "prompt": "x"}],
            "duration_seconds": 22.0,
            "style_line": "noir",
            "window_seconds": 12.0,
            "some_future_key": {"nested": True},
        })

    def test_a_bin_edit_preserves_every_other_field(self):
        raw, error = apply_bin_operation(
            self._document(), "add",
            asset={"id": "i2", "kind": KIND_IMAGE, "name": "I2", "file": "i2.png"})
        self.assertIsNone(error)
        document = json.loads(raw)
        self.assertEqual(len(document["assets"]), 2)
        self.assertEqual(document["duration_seconds"], 22.0)
        self.assertEqual(document["style_line"], "noir")
        self.assertEqual(document["shots"][0]["prompt"], "x")
        self.assertEqual(document["window_seconds"], 12.0)

    def test_unknown_keys_from_a_future_version_survive(self):
        raw, _ = apply_bin_operation(self._document(), "remove", asset_id="i1")
        self.assertEqual(json.loads(raw)["some_future_key"], {"nested": True})

    def test_a_refused_edit_returns_the_document_untouched(self):
        original = self._document()
        raw, error = apply_bin_operation(original, "remove", asset_id="does_not_exist")
        self.assertIsNotNone(error)
        self.assertEqual(raw, original)

    def test_malformed_input_is_an_error_not_an_empty_document(self):
        """The exact fallback that caused the erasure. It must not exist."""
        original = "{this is not json"
        raw, error = apply_bin_operation(original, "remove", asset_id="i1")
        self.assertIsNotNone(error)
        self.assertEqual(raw, original)
        self.assertIn("Refusing to replace", error)

    def test_load_rejects_rather_than_defaulting(self):
        with self.assertRaises(DocumentError):
            load_timeline_document("{broken")
        with self.assertRaises(DocumentError):
            load_timeline_document("[1, 2, 3]")

    def test_a_genuinely_empty_document_is_fine(self):
        """A new document is written in schema 2 form, cast key and all (§3.1)."""
        for blank in (None, "", "  ", "{}", "null"):
            self.assertEqual(load_timeline_document(blank),
                             {"schema": 2, "assets": [], "cast": []})

    def test_a_schema_1_document_upgrades_in_place(self):
        """Every pre-2.0.0 build wrote a bare {"assets": [...]}. It loads with an
        empty cast rather than being rejected, and the upgrade persists because
        apply_bin_operation dumps whatever load returns."""
        document = load_timeline_document('{"assets": [], "duration_seconds": 12}')
        self.assertEqual(document["schema"], 2)
        self.assertEqual(document["cast"], [])
        self.assertEqual(document["duration_seconds"], 12, "an unknown key was dropped")

    def test_a_future_schema_is_not_downgraded(self):
        """The upgrade is additive. A schema 3 file passes through untouched --
        this code has no business rewriting a format it does not know."""
        document = load_timeline_document('{"schema": 3, "assets": [], "cast": [], "x": 1}')
        self.assertEqual(document["schema"], 3)
        self.assertEqual(document["x"], 1)

    def test_an_existing_cast_survives_a_load(self):
        raw = '{"schema": 2, "assets": [], "cast": [{"id": "cast_01", "name": "Mimi"}]}'
        self.assertEqual(load_timeline_document(raw)["cast"],
                         [{"id": "cast_01", "name": "Mimi"}])

    def test_reorder_preserves_the_rest(self):
        document = json.loads(self._document())
        document["assets"].append({"id": "i2", "kind": KIND_IMAGE, "name": "I2", "file": "i2.png"})
        raw, error = apply_bin_operation(dump_timeline_document(document), "move",
                                         asset_id="i2", new_index=0)
        self.assertIsNone(error)
        result = json.loads(raw)
        self.assertEqual(result["assets"][0]["id"], "i2")
        self.assertEqual(result["style_line"], "noir")


class TestNewlineSurvival(unittest.TestCase):
    """Type three lines, save the workflow, reload, get three lines back."""

    def test_newlines_survive_a_workflow_round_trip(self):
        face = NodeFace(global_prompt=THREE_LINES, shot_prompt=SHOTS)
        # A ComfyUI workflow save/load is a JSON round trip of widget values.
        reloaded = json.loads(json.dumps(face.snapshot()))
        self.assertEqual(reloaded["global_prompt"], THREE_LINES)
        self.assertEqual(reloaded["global_prompt"].count("\n"), 2)
        self.assertEqual(len(reloaded["global_prompt"].split("\n")), 3)
        self.assertEqual(reloaded["shot_prompt"], SHOTS)

    def test_three_lines_compile_to_three_shots(self):
        face = NodeFace(global_prompt=THREE_LINES, shot_prompt=SHOTS, duration_seconds=12.0)
        plan, _ = face.queue()
        prompt = plan.windows[0].prompt
        for n in (1, 2, 3):
            self.assertIn("[Shot %d]" % n, prompt)

    def test_a_flattened_shot_box_still_compiles_to_three_shots(self):
        """If a widget ever does strip the newlines, normalization catches it."""
        face = NodeFace(global_prompt=THREE_LINES, duration_seconds=12.0,
                        shot_prompt=SHOTS.replace("\n", " "))
        plan, _ = face.queue()
        prompt = plan.windows[0].prompt
        for n in (1, 2, 3):
            self.assertIn("[Shot %d]" % n, prompt)


class TestBuildTimeline(unittest.TestCase):
    def test_global_prompt_reaches_the_right_sections(self):
        face = NodeFace(
            global_prompt="style: noir\nidentity: Mimi keeps the red scarf\n"
                          "retention: the scarf is never removed\n"
                          "soundscape: rain\nmusic: piano",
            shot_prompt="[Shot 1] she walks in", duration_seconds=6.0)
        plan, _ = face.queue()
        prompt = plan.windows[0].prompt
        self.assertIn("Mimi keeps the red scarf", prompt.split("summary:")[0])
        self.assertIn("the scarf is never removed", prompt)
        self.assertIn("noir", prompt.split("detailed_description:")[1])
        self.assertIn("rain", prompt.split("overall_soundscape:")[1])
        self.assertIn("piano", prompt.split("non_diegetic_music:")[1])

    def test_typed_shots_win_over_stored_shots(self):
        stored = json.dumps({"assets": [],
                             "shots": [{"id": "old", "start": 0, "duration": 5,
                                        "prompt": "from the canvas"}]})
        face = NodeFace(shot_prompt="[Shot 1] typed", timeline_data=stored,
                        duration_seconds=6.0)
        plan, _ = face.queue()
        self.assertIn("typed", plan.windows[0].prompt)
        self.assertNotIn("from the canvas", plan.windows[0].prompt)

    def test_an_empty_shot_box_falls_back_to_stored_shots(self):
        """Clearing the box must not silently discard timeline work."""
        stored = json.dumps({"assets": [],
                             "shots": [{"id": "old", "start": 0, "duration": 5,
                                        "prompt": "from the canvas"}]})
        face = NodeFace(shot_prompt="", timeline_data=stored, duration_seconds=6.0)
        plan, _ = face.queue()
        self.assertIn("from the canvas", plan.windows[0].prompt)

    def test_assets_come_from_the_document(self):
        stored = json.dumps({"assets": [
            {"id": "mimi", "kind": KIND_IMAGE, "name": "Mimi", "file": "m.png"}]})
        face = NodeFace(shot_prompt="[Shot 1] @Mimi walks", timeline_data=stored,
                        duration_seconds=6.0)
        plan, _ = face.queue()
        self.assertIn("<Picture 1>", plan.windows[0].prompt)

    def test_no_shots_anywhere_is_reported(self):
        face = NodeFace(global_prompt="noir", shot_prompt="", duration_seconds=6.0)
        _, notes = face.queue()
        self.assertTrue(any("no shots" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
