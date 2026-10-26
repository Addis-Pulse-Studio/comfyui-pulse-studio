"""One voice track per speaker per window, each take cut at its own shot's edges.

Before tracks a shot's take was cut to the *window*. A 9-second line on a
6-second shot kept playing into the next shot, where the next speaker's mouth
moved to it; and three takes in one window were three <Audio j> references, each
claiming the whole window, each mostly silence. Now every lip-sync recording
bound to one character in a window rides one track, at its own seconds, silent
outside its own shot -- and a window's three audio slots are three *voices*.

Headless: the tracks are decided by the compiler, and the gate arithmetic lives
in concat.py. The tensor half (media.gate_audio) is three lines over it.
"""

import unittest

from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, Asset
from comfyui_pulse_studio.compiler import CarryPolicy, compile_timeline, voice_tracks, wrap_dialogue
from comfyui_pulse_studio.concat import gate_bounds
from comfyui_pulse_studio.constants import (
    AUDIO_ROLE_LIP_SYNC,
    AUDIO_ROLE_TIMBRE,
    BRANCH_FL2VA,
    BRANCH_REF2VA,
    VOICE_GATE_TAIL_SECONDS,
)
from comfyui_pulse_studio.parsing import parse_global_prompt
from comfyui_pulse_studio.timeline import Timeline
from comfyui_pulse_studio.widget_state import build_timeline

CAST = [
    {"id": "ada", "kind": KIND_IMAGE, "name": "Ada", "file": "a.png",
     "description": "the woman in the red coat"},
    {"id": "ben", "kind": KIND_IMAGE, "name": "Ben", "file": "b.png",
     "description": "the man with the grey beard"},
]


def dialogue(lengths=(4.0, 4.0, 4.0), speakers=("ada", "ben", "ada"), take_seconds=None,
             take=True, free_run=False, window_seconds=15.1, continuation=BRANCH_REF2VA):
    """Shots back to back, each with its own take placed at the shot's start."""
    shots, cursor = [], 0.0
    for i, (duration, who) in enumerate(zip(lengths, speakers), 1):
        shots.append({"id": "s%d" % i, "start": cursor, "duration": duration,
                      "prompt": "@%s speaks." % who.title(), "speakers": [who]})
        cursor += duration
    timeline = Timeline(assets=[dict(a) for a in CAST], shots=shots,
                        duration_seconds=cursor, fps=24, window_seconds=window_seconds,
                        continuation_branch=continuation)
    for shot in timeline.shots:
        timeline.local_refs[shot.shot_id] = [Asset(
            "take_" + shot.shot_id, KIND_AUDIO, name="Take" + shot.shot_id[1:], file="",
            audio_role=AUDIO_ROLE_LIP_SYNC, audio_offset=shot.start,
            source_seconds=take_seconds if take_seconds is not None else shot.duration,
            take=take, free_run=free_run)]
    return timeline


def compile_(timeline, policy="fill"):
    return compile_timeline(timeline, policy=policy,
                            carry=CarryPolicy(mode="image", audio=False))


def lip_refs(window):
    return [f for f in window.files if f.kind == KIND_AUDIO and f.audio_role == AUDIO_ROLE_LIP_SYNC]


class OneTrackPerVoice(unittest.TestCase):

    def test_a_two_hander_with_three_lines_is_two_audio_references(self):
        window = compile_(dialogue()).windows[0]
        refs = lip_refs(window)
        self.assertEqual(len(refs), 2, "three lines, two voices")
        ada = next(r for r in refs if r.asset_id == "take_s1")
        self.assertEqual([p.asset_id for p in ada.pieces], ["take_s1", "take_s3"])
        ben = next(r for r in refs if r.asset_id == "take_s2")
        self.assertEqual([p.asset_id for p in ben.pieces], ["take_s2"])

    def test_each_take_is_gated_to_its_own_shot(self):
        ada = lip_refs(compile_(dialogue()).windows[0])[0]
        first, third = ada.pieces
        self.assertEqual(first.gate_start, 0.0)
        self.assertAlmostEqual(first.gate_end, 4.0 + VOICE_GATE_TAIL_SECONDS)
        self.assertEqual(third.gate_start, 8.0)

    def test_the_prompt_says_which_shots_each_voice_speaks_in(self):
        prompt = compile_(dialogue()).windows[0].prompt
        self.assertIn("is saying in [Shot 1] and [Shot 3]", prompt)
        self.assertIn("is saying in [Shot 2]", prompt)

    def test_four_voices_still_fit_three_slots_when_two_are_one_character(self):
        timeline = dialogue(lengths=(3.0,) * 4, speakers=("ada", "ben", "ada", "ben"))
        refs = lip_refs(compile_(timeline).windows[0])
        self.assertEqual(len(refs), 2)
        self.assertEqual(sum(len(r.pieces) for r in refs), 4)

    def test_a_take_whose_shot_is_the_whole_window_keeps_the_plain_path(self):
        """No gate to apply, so nothing changes: same audio, prompt and key."""
        # 5.875s is 141 frames, on the grid, so each window is exactly its shot.
        timeline = dialogue(lengths=(5.875, 5.875), speakers=("ada", "ben"),
                            window_seconds=5.9)
        plan = compile_(timeline, policy="shot_aligned")
        for window in plan.windows:
            for ref in lip_refs(window):
                self.assertIsNone(ref.pieces)
            self.assertIsNone(window.audio_plan())

    def test_free_run_is_never_gated(self):
        timeline = dialogue(free_run=True)
        for ref in lip_refs(compile_(timeline).windows[0]):
            self.assertTrue(all(not p.gated for p in (ref.pieces or [])))

    def test_timbre_and_music_pass_through_untouched(self):
        timeline = dialogue()
        timeline.assets.add(Asset("music", KIND_AUDIO, name="Score", file="m.wav"))
        audios, pieces = voice_tracks(timeline, timeline.shots,
                                      timeline.assets.by_kind(KIND_AUDIO))
        self.assertEqual([a.asset_id for a in audios], ["music"])
        self.assertEqual(pieces, {})


class TheGateArithmetic(unittest.TestCase):

    def test_a_gate_inside_the_clip_fades_both_edges(self):
        a, b, up, down = gate_bounds(48000, 48000, 0.25, 0.75, 0.04)
        self.assertEqual((a, b), (12000, 36000))
        self.assertEqual((up, down), (1920, 1920))

    def test_an_edge_the_window_cuts_is_not_faded(self):
        # The shot began before the window: the voice is mid-line at the seam,
        # and fading there would dip it exactly where a dip is audible.
        a, b, up, down = gate_bounds(48000, 48000, -2.0, 0.5, 0.04)
        self.assertEqual((a, up), (0, 0))
        self.assertEqual(down, 1920)
        a, b, up, down = gate_bounds(48000, 48000, 0.5, 3.0, 0.04)
        self.assertEqual((b, down), (48000, 0))

    def test_a_shared_edge_switches_over_without_a_fade(self):
        _, _, up, down = gate_bounds(48000, 48000, 0.25, 0.75, 0.04,
                                     fade_in=False, fade_out=False)
        self.assertEqual((up, down), (0, 0))

    def test_a_fade_never_exceeds_half_the_gate(self):
        _, _, up, down = gate_bounds(48000, 48000, 0.5, 0.52, 0.04)
        self.assertLessEqual(up + down, 960)


class ATakeLongerThanItsShot(unittest.TestCase):

    def test_is_named_once_with_the_seconds_cut(self):
        notes = " ".join(compile_(dialogue(take_seconds=5.5)).windows[0].diagnostics)
        # A 5.5s take on a 4.0s shot: 1.5s of it is past the cut.
        self.assertIn("`@Take1` runs 1.50s past the end of its shot", notes)
        self.assertIn("duration_seconds to 0", notes)

    def test_a_film_clock_recording_wired_to_a_shot_is_not_a_take(self):
        # One narration wired into every shot runs past each of them by design.
        notes = " ".join(compile_(dialogue(take_seconds=12.0, take=False)).windows[0].diagnostics)
        self.assertNotIn("past the end of its shot", notes)


class TheContinuationWindowSpeaksTheTake(unittest.TestCase):
    """fl2va takes no references, so @Ada was reaching the model as literal text."""

    def test_names_become_descriptions_on_fl2va(self):
        timeline = dialogue(lengths=(12.0, 12.0), speakers=("ada", "ben"),
                            window_seconds=12.1, continuation=BRANCH_FL2VA)
        timeline.first_frame = None
        plan = compile_(timeline, policy="shot_aligned")
        second = plan.windows[1]
        self.assertEqual(second.branch, BRANCH_FL2VA)
        self.assertIn("the man with the grey beard speaks", second.prompt)
        self.assertNotIn("@Ben", second.prompt)
        self.assertFalse([d for d in second.diagnostics if "unresolved reference" in d])

    def test_a_driven_window_is_not_told_to_stay_silent(self):
        timeline = dialogue(lengths=(12.0, 12.0), speakers=("ada", "ben"),
                            window_seconds=12.1, continuation=BRANCH_FL2VA)
        second = compile_(timeline, policy="shot_aligned").windows[1]
        self.assertTrue(second.drive_files)
        self.assertNotIn("no dialogue or vocalisation", second.prompt)
        self.assertIn("is a recorded voice", second.prompt)


class TheDialogueLanguage(unittest.TestCase):

    def test_a_language_line_in_the_global_prompt_sets_it(self):
        fields, _ = parse_global_prompt("style: warm\nlanguage: Amharic\nmusic: none")
        self.assertEqual(fields["dialogue_language"], "Amharic")
        self.assertEqual(fields["style_line"], "warm")
        timeline, _ = build_timeline("", global_prompt="language: Amharic",
                                     shot_prompt='[Shot 1] She says "selam"',
                                     duration_seconds=5.0)
        self.assertEqual(timeline.dialogue_language, "Amharic")

    def test_without_one_it_is_english_as_before(self):
        timeline, _ = build_timeline("", global_prompt="style: warm",
                                     shot_prompt='[Shot 1] She says "hello"',
                                     duration_seconds=5.0)
        self.assertEqual(timeline.dialogue_language, "English")

    def test_a_shot_spoken_by_a_voice_with_its_own_language_uses_it(self):
        timeline = dialogue(lengths=(5.0, 5.0), speakers=("ada", "ben"))
        timeline.shots[0].prompt = '@Ada says "selam, endet neh"'
        timeline.shots[0].language = "Amharic"
        timeline.shots[1].prompt = '@Ben says "fine, thanks"'
        prompt = compile_(timeline).windows[0].prompt
        self.assertIn("<d>[Amharic] selam, endet neh.</d>", prompt)
        self.assertIn("<d>[English] fine, thanks.</d>", prompt)

    def test_typographic_quotes_are_dialogue_too(self):
        self.assertEqual(wrap_dialogue("She says “hello there”.", "English"),
                         "She says <d>[English] hello there.</d>.")
        self.assertEqual(wrap_dialogue("«selam»", "Amharic"),
                         "<d>[Amharic] selam.</d>")


if __name__ == "__main__":
    unittest.main()
