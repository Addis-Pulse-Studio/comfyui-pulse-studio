"""Asking for 15 seconds must render 15 seconds.

THE BUG THIS EXISTS TO PREVENT

A 15.0s timeline produced a 7-second file. Three faults in a row, each of which
looked fine on its own:

1. The window cap snapped DOWN to the grid. 15.0s is 360 frames; the grid steps
   by 17, so the cap became 345 frames (14.375s). The trained ceiling is 362
   (15.083s), so the most natural number a user can type could not express one
   window, and a 15.0s render silently became two windows over a 0.7s shortfall.

2. Two windows means the internal sampling path, which handed back the LAST
   window's latent on the `latent` output.

3. The starter graph's single-window group samples that latent. 175 frames is a
   perfectly valid latent, so it rendered, saved, and produced 7.29 seconds with
   no error anywhere.

Every one of those steps succeeded. That is the shape of failure this package
exists to refuse, so each step is pinned here separately.

A fourth fault made it worse: the shot box used `[00:00.000 - 00:05.000]` range
markers, which did not parse, so three shots collapsed into one whose prompt was
the entire text -- a render with no per-shot direction in it at all.
"""

import unittest

from comfyui_pulse_studio.compiler import compile_timeline
from comfyui_pulse_studio.constants import MAX_WINDOW_FRAMES
from comfyui_pulse_studio.frames import partition_windows, seconds_to_frames
from comfyui_pulse_studio.parsing import parse_shots
from comfyui_pulse_studio.widget_state import build_timeline

# The shot box from the render that produced 7 seconds, trimmed to its markers.
SHOT_BOX = """[00:00.000 - 00:05.000]
Shot 1: Exterior / Door Entrance
Camera: Medium push-in tracking shot following motion.
Visual: @Image1 pushes open the heavy wooden glass door of a cozy cafe.
Audio: Heavy rain falling, door chime ringing.

[00:05.000 - 00:10.000]
Shot 2: Cafe Counter & Dialogue
Camera: Smooth side-panning dolly shot.
Visual: At 00:05.000, @Image1 walks towards the counter.
Audio: Dialogue: "You're still open, thank god."

[00:10.000 - 00:15.000]
Shot 3: Window Seat / Outro
Camera: Static medium-wide shot.
Visual: At 00:10.000, she sits down in a leather booth by the window.
Audio: Soft rain tapping against windowpane."""

BIN = '{"schema": 2, "assets": [{"id": "im1", "kind": "image", "name": "Image1", ' \
      '"file": "example_character_a.png"}], "cast": []}'


class TestFifteenSecondsIsOneWindow(unittest.TestCase):
    def test_the_window_cap_rounds_to_the_nearest_grid_point(self):
        """15.0s must reach 362, not fall back to 345."""
        self.assertEqual(seconds_to_frames(15.0, 24, "nearest"), MAX_WINDOW_FRAMES)

    def test_a_fifteen_second_request_is_a_single_window(self):
        self.assertEqual(partition_windows(360, window_frames=15.0 * 24), [MAX_WINDOW_FRAMES])

    def test_the_widget_default_can_express_the_ceiling(self):
        """The default window_seconds must be a value that yields one full
        window. A default that cannot is a trap set for every new user."""
        timeline, _ = build_timeline(BIN, shot_prompt=SHOT_BOX, duration_seconds=15.0,
                                     window_seconds=15.0)
        plan = compile_timeline(timeline)
        self.assertEqual([w.frame_count for w in plan.windows], [MAX_WINDOW_FRAMES])

    def test_nothing_ever_exceeds_the_trained_ceiling(self):
        """Rounding up is only safe because the clamp is still there."""
        for seconds in (14.9, 15.0, 15.08, 15.1, 16.0, 30.0):
            with self.subTest(window_seconds=seconds):
                windows = partition_windows(3600, window_frames=seconds * 24)
                self.assertLessEqual(max(windows), MAX_WINDOW_FRAMES)

    def test_shorter_windows_are_still_honoured(self):
        """Rounding to nearest must not quietly inflate every window to the
        ceiling -- a user asking for 8s windows means 8s windows."""
        windows = partition_windows(24 * 30, window_frames=8.0 * 24, policy="fill")
        self.assertTrue(all(w <= seconds_to_frames(8.0, 24, "nearest") for w in windows),
                        windows)


class TestTimecodeRangesParse(unittest.TestCase):
    def test_three_ranges_make_three_shots(self):
        shots, _ = parse_shots(SHOT_BOX, total_duration=15.0)
        self.assertEqual(len(shots), 3)

    def test_each_shot_starts_where_it_was_written(self):
        shots, _ = parse_shots(SHOT_BOX, total_duration=15.0)
        self.assertEqual([s["start"] for s in shots], [0.0, 5.0, 10.0])
        self.assertEqual([s["duration"] for s in shots], [5.0, 5.0, 5.0])

    def test_the_body_lines_stay_with_their_shot(self):
        shots, _ = parse_shots(SHOT_BOX, total_duration=15.0)
        self.assertIn("Door Entrance", shots[0]["prompt"])
        self.assertIn("thank god", shots[1]["prompt"])
        self.assertIn("leather booth", shots[2]["prompt"])
        self.assertNotIn("leather booth", shots[0]["prompt"])

    def test_every_separator_a_person_might_type(self):
        for dash in ("-", "--", "–", "—", "to"):
            with self.subTest(separator=dash):
                shots, _ = parse_shots(
                    "[00:00.000 %s 00:05.000] a\n[00:05.000 %s 00:10.000] b" % (dash, dash),
                    total_duration=10.0)
                self.assertEqual([s["start"] for s in shots], [0.0, 5.0])

    def test_a_plain_timecode_still_parses(self):
        shots, _ = parse_shots("[00:00.000] a\n[00:05.000] b", total_duration=10.0)
        self.assertEqual([s["start"] for s in shots], [0.0, 5.0])

    def test_a_written_range_that_does_not_match_is_reported(self):
        """Shot spans run to the next shot's start. When the author's stated end
        disagrees, the timeline they get is not the one they wrote, and saying so
        is the whole job."""
        _, notes = parse_shots("[00:00.000 - 00:04.000] a\n[00:05.000 - 00:10.000] b",
                               total_duration=10.0)
        self.assertTrue(any("written as ending at" in n for n in notes), notes)

    def test_contiguous_ranges_report_nothing(self):
        _, notes = parse_shots(SHOT_BOX, total_duration=15.0)
        self.assertEqual([n for n in notes if "written as ending" in n], [])

    def test_the_whole_box_compiles_to_three_shot_markers(self):
        timeline, _ = build_timeline(BIN, shot_prompt=SHOT_BOX, duration_seconds=15.0,
                                     window_seconds=15.0)
        plan = compile_timeline(timeline)
        self.assertTrue(plan.ok, plan.problems)
        prompt = plan.windows[0].prompt
        for n in (1, 2, 3):
            self.assertIn("[Shot %d]" % n, prompt)
        # And the alias resolved rather than surviving as raw text.
        self.assertIn("<Picture 1>", prompt)
        self.assertNotIn("@Image1", prompt)


if __name__ == "__main__":
    unittest.main()
