"""End-to-end: a realistic project through compile, retake and still.

These exercise the modules composing rather than each in isolation, and assert
the invariants that must hold across the whole pipeline -- the ones a user would
actually notice being wrong.
"""

import json
import unittest

from omni_director import compile_timeline
from omni_director.assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO
from omni_director.binops import preview_change
from omni_director.compiler import CarryPolicy
from omni_director.constants import MAX_REF_FILES_TOTAL, MAX_WINDOW_FRAMES
from omni_director.frames import is_on_grid
from omni_director.retake import plan_retake
from omni_director.still import plan_still
from omni_director.timeline import Timeline

PROJECT = {
    "assets": [
        {"id": "mimi", "kind": KIND_IMAGE, "name": "Mimi", "file": "mimi.png",
         "description": "the young woman with long dark hair", "retention": "fully_preserved"},
        {"id": "kaleb", "kind": KIND_IMAGE, "name": "Kaleb", "file": "kaleb.png",
         "description": "the man with the grey beard"},
        {"id": "cafe", "kind": KIND_IMAGE, "name": "Cafe", "file": "cafe.png"},
        {"id": "handheld", "kind": KIND_VIDEO, "name": "Handheld", "file": "hh.mp4",
         "trim_start": 0.0, "trim_end": 6.0, "include_audio": True},
        {"id": "vo", "kind": KIND_AUDIO, "name": "Kaleb VO", "file": "vo.wav",
         "description": "a low gravelly voice"},
    ],
    "shots": [
        {"id": "s1", "start": 0.0, "duration": 8.0,
         "prompt": "@Mimi steps into @Cafe, shaking off the rain."},
        {"id": "s2", "start": 8.0, "duration": 9.0,
         "prompt": "@Kaleb looks up and says \"you came\"", "speakers": ["kaleb"]},
        {"id": "s3", "start": 17.0, "duration": 13.0,
         "prompt": "Camera drifts like @Handheld as they sit together."},
    ],
    "duration_seconds": 30.0,
    "style_line": "Shot on 35mm, warm practical light.",
    "overall_soundscape": "Rain outside, espresso machine hiss.",
    "non_diegetic_music": "Sparse piano.",
}


class TestRealProject(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline.from_dict(json.loads(json.dumps(PROJECT)))
        self.plan = compile_timeline(self.timeline, carry=CarryPolicy(mode="image", audio=True))

    def test_project_compiles_cleanly(self):
        self.assertTrue(self.plan.ok, self.plan.problems)
        self.assertGreater(len(self.plan.windows), 1)

    def test_no_window_carries_an_unresolved_reference(self):
        for window in self.plan.windows:
            for note in window.diagnostics:
                self.assertNotIn("unresolved reference", note)

    def test_no_hand_typed_tag_survives_into_any_prompt(self):
        """Every tag in the output must have been generated, never authored."""
        for window in self.plan.windows:
            self.assertNotIn("@Mimi", window.prompt)
            self.assertNotIn("@Kaleb", window.prompt)
            self.assertNotIn("@Handheld", window.prompt)
            self.assertNotIn("{{", window.prompt)

    def test_every_window_is_renderable(self):
        for window in self.plan.windows:
            self.assertTrue(is_on_grid(window.frame_count))
            self.assertLessEqual(window.frame_count, MAX_WINDOW_FRAMES)
            self.assertTrue(window.prompt.strip())

    def test_file_budget_holds_in_every_window(self):
        for window in self.plan.windows:
            real = [f for f in window.files if not f.synthetic]
            self.assertLessEqual(len(real), MAX_REF_FILES_TOTAL)

    def test_tags_are_internally_consistent_per_window(self):
        """Every tag mentioned in a prompt must correspond to a socket that
        window actually carries -- otherwise the model is told about a picture
        it was never shown."""
        import re
        for window in self.plan.windows:
            carried = {f.tag for f in window.files}
            for match in re.finditer(r"<(?:Picture|Video|Audio) \d+>", window.prompt):
                self.assertIn(match.group(0), carried,
                              "window %d cites %s but does not carry it"
                              % (window.index + 1, match.group(0)))

    def test_socket_names_are_unique_and_dense(self):
        for window in self.plan.windows:
            sockets = [f.socket for f in window.files]
            self.assertEqual(len(sockets), len(set(sockets)))
            for group in ("ref_image_", "ref_video_", "ref_audio_"):
                idx = sorted(int(s.rsplit("_", 1)[1]) for s in sockets
                             if s.startswith(group) and not s.startswith("ref_video_audio_"))
                self.assertEqual(idx, list(range(len(idx))), "%s not dense: %r" % (group, idx))

    def test_reordering_the_bin_changes_tags_but_not_prompt_text(self):
        """The property the whole design exists for."""
        before = self.plan.windows[0].prompt
        self.timeline.assets.move("cafe", 0)
        after = compile_timeline(self.timeline).windows[0].prompt
        self.assertNotEqual(before, after)
        # The authored words are untouched; only the generated ordinals moved.
        for phrase in ("steps into", "shaking off the rain", "looks up and says"):
            self.assertIn(phrase, before)
            self.assertIn(phrase, after)

    def test_removing_an_asset_is_previewed_before_it_bites(self):
        deltas, err = preview_change(self.timeline.assets, "remove", asset_id="mimi")
        self.assertIsNone(err)
        self.assertTrue(any(d.removed for d in deltas))

    def test_plan_survives_a_json_roundtrip(self):
        restored = Timeline.from_json(self.timeline.to_json())
        self.assertEqual(compile_timeline(restored).preview(), self.plan.preview())


class TestPipelineComposition(unittest.TestCase):
    def test_a_rendered_window_can_be_patched(self):
        """The scissor operates on what the director produced."""
        plan = compile_timeline(Timeline.from_dict(PROJECT))
        rendered = plan.windows[0].frame_count
        retake = plan_retake(rendered, cut_start_seconds=2.0, cut_end_seconds=4.0)
        self.assertEqual(retake.output_frames, rendered)
        self.assertTrue(is_on_grid(retake.patch_frames))

    def test_a_still_can_be_made_at_the_render_canvas(self):
        still = plan_still(prompt="@Mimi, three-quarter view", width=1344, height=736)
        self.assertEqual(still.length, 5)
        self.assertEqual(still.width % 32, 0)

    def test_a_still_edited_from_a_frame_keeps_that_frames_aspect(self):
        """Anchor repair: pull a frame, fix it, use it -- without a round trip
        through another tool changing its shape."""
        still = plan_still(prompt="open her eyes", source_asset="frame",
                           source_size=(1344, 736), frame_pick=2)
        self.assertAlmostEqual(still.width / still.height, 1344 / 736, delta=0.1)
        self.assertTrue(still.is_edit)


if __name__ == "__main__":
    unittest.main()
