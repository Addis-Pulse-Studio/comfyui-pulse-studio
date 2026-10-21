"""Seams on cuts, else in pauses -- never through a word.

A seam is where two windows sampled apart meet. Inside a word one window's mouth
closes the syllable and the next one's opens it again, which reads as a stutter
however good either side is. `speech_aware` moves seams into the pauses of the
film's own recordings; every other policy now reports a seam that cuts a word.
"""

import unittest

from comfyui_pulse_studio.compiler import CarryPolicy, compile_timeline
from comfyui_pulse_studio.frames import (
    MAX_WINDOW_FRAMES,
    POLICIES,
    in_speech,
    is_on_grid,
    partition_windows,
)
from comfyui_pulse_studio.timeline import Timeline


def seams(windows):
    out, position = [], 0
    for frames in windows[:-1]:
        position += frames
        out.append(position)
    return out


class ThePolicy(unittest.TestCase):

    def test_it_is_offered_and_appended(self):
        self.assertEqual(POLICIES[-1], "speech_aware")
        self.assertEqual(POLICIES[:3], ("balanced", "fill", "shot_aligned"))

    def test_with_nothing_to_hear_it_is_full_windows(self):
        notes = []
        self.assertEqual(partition_windows(720, 362, policy="speech_aware",
                                           diagnostics=notes), [362, 362])
        self.assertIn("no lip-sync recording", notes[0])

    def test_seams_land_in_the_pauses(self):
        speech = [(0, 300), (330, 620), (650, 720)]
        windows = partition_windows(720, 362, policy="speech_aware", speech=speech)
        self.assertTrue(all(is_on_grid(w) and w <= MAX_WINDOW_FRAMES for w in windows))
        self.assertGreaterEqual(sum(windows), 720)
        for seam in seams(windows):
            self.assertFalse(in_speech(seam, speech), seam)

    def test_a_cut_in_a_pause_is_where_the_seam_goes(self):
        # 260 = 17*15 + 5, so a seam on the cut is reachable, and the rest (340
        # frames) fits one window. Any other seam would cut a word.
        speech = [(0, 250), (270, 600)]
        windows = partition_windows(600, 362, policy="speech_aware", speech=speech,
                                    boundaries=[260])
        self.assertEqual(seams(windows), [260])

    def test_a_cut_in_silence_is_taken_when_the_grid_reaches_it(self):
        # 243 = 17*14 + 5 is a legal window, so a seam at the cut is reachable.
        windows = partition_windows(486, 362, policy="speech_aware", boundaries=[243])
        self.assertEqual(seams(windows), [243])

    def test_it_says_so_when_no_pause_is_reachable(self):
        notes = []
        partition_windows(720, 362, policy="speech_aware", speech=[(0, 720)],
                          diagnostics=notes)
        self.assertIn("inside speech", notes[0])

    def test_a_long_film_plans_quickly(self):
        import time
        started = time.time()
        windows = partition_windows(14400, 362, policy="speech_aware",
                                    speech=[(i, i + 200) for i in range(0, 14400, 240)])
        self.assertLess(time.time() - started, 5.0)
        self.assertGreaterEqual(sum(windows), 14400)


class EveryOtherPolicyReportsAWordCut(unittest.TestCase):

    def plan(self, policy):
        shots = [{"id": "s1", "start": 0.0, "duration": 30.0, "prompt": "One long talk."}]
        timeline = Timeline(assets=[], shots=shots, duration_seconds=30.0, fps=24,
                            window_seconds=15.1)
        timeline.speech_spans = [(0, 720)]
        return compile_timeline(timeline, policy=policy,
                                carry=CarryPolicy(mode="image", audio=False))

    def test_balanced_says_which_seam_cuts_a_word(self):
        text = " ".join(self.plan("balanced").diagnostics)
        self.assertIn("land inside speech", text)
        self.assertIn("speech_aware", text)

    def test_no_speech_known_says_nothing(self):
        shots = [{"id": "s1", "start": 0.0, "duration": 30.0, "prompt": "One long talk."}]
        timeline = Timeline(assets=[], shots=shots, duration_seconds=30.0, fps=24,
                            window_seconds=15.1)
        plan = compile_timeline(timeline, policy="balanced")
        self.assertFalse([d for d in plan.diagnostics if "inside speech" in d])


if __name__ == "__main__":
    unittest.main()
