"""Where a lip-sync recording sits on the clock, and what happens when it does not fit.

A lip-sync reference is cut to the window it rides in, which makes *which seconds
of the file* the question the whole feature turns on. Until `audio_offset` existed
there was no way to answer it: the trim assumed the recording began with the film
and said so nowhere, so a per-shot take -- the reading `PulseShot.ref_audio`'s own
tooltip invited -- was sliced at an offset past its own end and the window was
handed silence. Not an error and not a warning; a mouth that does not move, which
is indistinguishable from the model being bad at this.

Everything here is arithmetic, so it runs headless with no torch and no GPU. The
three tensor-level cases that cannot are marked and skipped.
"""

import unittest

from comfyui_pulse_studio.assets import KIND_AUDIO, Asset
from comfyui_pulse_studio.compiler import (
    DIALOGUE_BLOCK_RE,
    CarryPolicy,
    compile_timeline,
    wrap_dialogue,
)
from comfyui_pulse_studio.concat import audio_span_bounds, audio_span_report
from comfyui_pulse_studio.constants import (
    AUDIO_ROLE_LIP_SYNC,
    AUDIO_ROLE_TIMBRE,
    LIP_SYNC_END_OF_FILM_TOLERANCE_SECONDS,
    MAX_DIALOGUE_WPM,
    MAX_WINDOW_FRAMES,
)
from comfyui_pulse_studio.pulse_timeline import ref_descriptor
from comfyui_pulse_studio.segcache import _ref_descriptor_key
from comfyui_pulse_studio.timeline import Timeline

SR = 44100

#: The user's 45-second explainer, which is the shape every case here is a
#: variation of: four shots whose cuts land exactly on the window seams.
EXPLAINER = [12.25, 12.25, 12.25, 8.0]


def film(durations=EXPLAINER, window_seconds=12.3, policy="fill", prompts=None,
         role=AUDIO_ROLE_LIP_SYNC, source_seconds=None, offsets=None,
         carry_audio=False):
    """A film of `durations`, each shot carrying its own voice reference.

    `offsets` is per shot and defaults to zero -- the film clock, which is what
    every project written before the field existed means.
    """
    shots, cursor = [], 0.0
    for i, duration in enumerate(durations, 1):
        shots.append({"id": "s%d" % i, "start": cursor, "duration": duration,
                      "prompt": (prompts[i - 1] if prompts
                                 else "Medium close-up of the speaker as they talk.")})
        cursor += duration
    timeline = Timeline(assets=[], shots=shots, duration_seconds=cursor, fps=24,
                        window_seconds=window_seconds)
    if role is not None:
        for i in range(1, len(durations) + 1):
            timeline.local_refs["s%d" % i] = [Asset(
                "aud_s%d" % i, KIND_AUDIO, name="Voice", file="", audio_role=role,
                audio_offset=(offsets or {}).get(i, 0.0),
                source_seconds=source_seconds)]
    return compile_timeline(timeline, policy=policy,
                            carry=CarryPolicy(mode="image", audio=carry_audio))


def notes(compiled):
    """Everything a user would see: plan-level notes plus every window's.

    `nodes.py` surfaces both, so a test reading only one of the two can pass while
    the message never reaches anybody.
    """
    lines = list(compiled.diagnostics)
    for window in compiled.windows:
        lines.extend(window.diagnostics)
    return " ".join(lines)


class TheSpanArithmetic(unittest.TestCase):
    """`concat.audio_span_bounds`, which decides which samples a window is handed."""

    WINDOW = 12.25

    def test_a_negative_start_leads_in_with_silence_rather_than_clamping(self):
        # The clamp was the old behaviour and it was silent: a recording that
        # starts three seconds into the film was slid forward to the window's
        # opening, and every lip in it then ran three seconds early.
        start, stop, head, tail = audio_span_bounds(
            30 * SR, SR, -3.0, self.WINDOW)
        self.assertEqual(head, 3 * SR)
        self.assertEqual(start, 0)
        self.assertEqual(stop - start, round(self.WINDOW * SR) - 3 * SR)
        self.assertEqual(tail, 0)

    def test_the_length_is_the_window_whatever_the_recording_does(self):
        """Real or padded, at either end. A span that came back short would drift
        the track against a video timeline that did not move, a little further at
        every seam."""
        want = round(self.WINDOW * SR)
        for total in (0, 2, 12, 30, 90):
            for offset in (-30.0, -3.0, 0.0, 6.125, 12.25, 60.0):
                start, stop, head, tail = audio_span_bounds(
                    total * SR, SR, offset, self.WINDOW)
                self.assertEqual(head + (stop - start) + tail, want,
                                 "%ds file at %+.3fs" % (total, offset))
                self.assertGreaterEqual(head, 0)
                self.assertGreaterEqual(tail, 0)

    def test_a_lead_in_longer_than_the_window_is_all_silence(self):
        _, stop, head, tail = audio_span_bounds(30 * SR, SR, -60.0, self.WINDOW)
        self.assertEqual(head, round(self.WINDOW * SR))
        self.assertEqual(tail, 0)

    def test_the_zero_offset_path_is_exactly_what_it_always_was(self):
        # The regression guard for every render already on disk: at offset 0 with
        # a recording that covers the window, nothing about the cut has moved.
        start, stop, head, tail = audio_span_bounds(45 * SR, SR, 24.5, self.WINDOW)
        self.assertEqual(start, round(24.5 * SR))
        self.assertEqual(stop - start, round(self.WINDOW * SR))
        self.assertEqual((head, tail), (0, 0))


class TheSpanReport(unittest.TestCase):
    """`audio_span_report`, the same arithmetic in seconds for the compiler.

    The compiler runs before anything is decoded and imports no torch, so it
    cannot count samples -- but it is the only place that can put the number in
    front of the author before a render is spent finding out.
    """

    def test_it_agrees_with_the_sample_arithmetic(self):
        for total in (0.0, 2.0, 12.4, 45.0):
            for offset in (-30.0, -3.0, 0.0, 24.5, 60.0):
                head, tail = audio_span_report(total, offset, 12.25)
                _, _, s_head, s_tail = audio_span_bounds(
                    int(total * SR), SR, offset, 12.25)
                self.assertAlmostEqual(head, s_head / float(SR), places=3)
                self.assertAlmostEqual(tail, s_tail / float(SR), places=3)

    def test_an_unmeasurable_recording_reports_nothing_rather_than_zero(self):
        # None means unknown, and unknown is never a problem. A file nothing could
        # measure is not a file that is empty.
        self.assertEqual(audio_span_report(None, 0.0, 12.25), (0.0, 12.25))


class TheCoverageDiagnostic(unittest.TestCase):
    """The failure this whole change exists for, said out loud."""

    def test_a_whole_film_narration_on_the_film_clock_says_nothing(self):
        # The user's live graph. This must stay silent, or the channel is noise.
        self.assertNotIn("does not move", notes(film(source_seconds=45.5)))
        self.assertNotIn("still through it", notes(film(source_seconds=45.5)))

    def test_a_per_shot_take_left_on_the_film_clock_is_named(self):
        # 12.4s takes, one per shot, all at offset 0. Window 3 asks for the
        # seconds from 24.50 onward and every one of them is past the file's end.
        text = notes(film(source_seconds=12.4))
        self.assertIn("nothing of this recording reaches this window", text)
        self.assertIn("24.50", text)
        self.assertIn("@Voice", text)

    def test_a_silent_shot_over_a_gap_has_a_still_mouth(self):
        # No <d> block: nothing told the model to speak and no recording is
        # driving it, so the mouth really is still.
        text = notes(film(source_seconds=12.4))
        self.assertIn("the mouth is still through it", text)
        self.assertNotIn("dead air", text)

    def test_a_speaking_shot_over_a_gap_moves_its_mouth_anyway(self):
        """The worse half, and the one the first draft of this check got wrong.

        A <d> block instructs the model to say those words, so it generates speech
        and the mouth moves -- while `use_reference_audio` mixes the reference's
        silence over the top. Observed on a real render: a 7.6s narration in a
        13.88s film, mouth moving through every second of the silence. Saying "the
        mouth is still" there sends the author looking for a longer recording when
        the fix may be to cut the words.
        """
        lines = ['They talk to camera. They say "the line that is written here."'] * 4
        text = notes(film(prompts=lines, source_seconds=12.4))
        self.assertIn("the mouth moves over dead air", text)
        self.assertIn("take the quoted lines out", text)
        self.assertNotIn("the mouth is still through it", text)

    def test_the_two_wordings_never_appear_together(self):
        for prompts in (None, ['He says "a line."'] * 4):
            text = notes(film(prompts=prompts, source_seconds=12.4))
            self.assertNotEqual("dead air" in text,
                                "mouth is still through it" in text,
                                "a window is either speaking or it is not")

    def test_setting_the_offsets_silences_it(self):
        cursor, offsets = 0.0, {}
        for i, duration in enumerate(EXPLAINER, 1):
            offsets[i] = cursor
            cursor += duration
        self.assertNotIn("silence", notes(film(source_seconds=12.4, offsets=offsets)))

    def test_a_narration_that_runs_out_says_how_much_is_missing(self):
        text = notes(film(source_seconds=41.0))
        self.assertIn("runs out", text)
        self.assertIn("still through it", text)

    def test_the_film_merely_ending_is_not_a_shortfall(self):
        """A narration stopping a fraction of a second before the last frame is
        ordinary: the recording ends when the speaking ends, and the grid rounds
        the final window up to a legal frame count regardless.

        Two real graphs tripped the quarter-second rule on exactly this -- 0.46s
        on a 45s explainer, 0.37s on an 8s one, both correct films -- and a
        warning channel that fires on correct films is worth less than none.
        """
        # The film renders to 45.46s; a 45.0s narration is 0.46s short of that.
        self.assertNotIn("runs out", notes(film(source_seconds=45.0)))

    def test_a_real_shortfall_at_the_end_is_still_named(self):
        # 6.24s on the last window was the 7.6s-narration test. Relaxing the
        # threshold must not swallow the case the check exists for.
        text = notes(film([6.5833, 6.5833], window_seconds=6.6,
                          policy="shot_aligned", source_seconds=7.632))
        self.assertIn("runs out 6.24s", text)

    def test_the_relaxation_is_the_last_window_only(self):
        # Half a second of silence in the middle of a film is a mistake; the same
        # half second at the end is the film ending.
        text = notes(film(source_seconds=12.0))
        self.assertIn("window 2", text)

    def test_a_late_start_is_never_relaxed(self):
        # Lead-in silence is a recording that starts late, and that is wrong
        # wherever it happens -- including on the last window.
        compiled = film(source_seconds=45.0, offsets={4: 37.25})
        self.assertIn("starts 0.50s after the window opens", notes(compiled))

    def test_the_threshold_is_where_the_constant_says_it_is(self):
        self.assertEqual(LIP_SYNC_END_OF_FILM_TOLERANCE_SECONDS, 1.0)
        # 45.46 - 44.60 = 0.86s, under the floor; 45.46 - 44.20 = 1.26s, over it.
        self.assertNotIn("runs out", notes(film(source_seconds=44.60)))
        self.assertIn("runs out", notes(film(source_seconds=44.20)))

    def test_an_unmeasured_recording_is_never_diagnosed(self):
        # source_seconds is None whenever nothing measured the file. Reporting
        # silence there would put a warning on a render that is perfectly fine.
        self.assertNotIn("silence", notes(film(source_seconds=None)))

    def test_rounding_the_window_to_the_frame_grid_is_not_a_shortfall(self):
        # A 362-frame window is 15.0833s and no recording is cut to four decimal
        # places. Milliseconds of padding are the grid, not a problem.
        self.assertNotIn("runs out", notes(film(source_seconds=45.46)))

    def test_a_timbre_reference_is_never_measured(self):
        # A timbre clip is not trimmed to the window and carries no temporal
        # claim, so it cannot fail to cover one.
        self.assertNotIn("silence", notes(film(role=AUDIO_ROLE_TIMBRE,
                                               source_seconds=2.0)))


class TheOneFrameFloor(unittest.TestCase):
    """`Timeline.shots_in`, and the duplicate references it used to produce."""

    def test_a_sub_frame_straddle_no_longer_lands_a_shot_in_two_windows(self):
        # 15.08 was the widget's old ceiling; a full window is 362/24 = 15.0833s.
        # Every cut landed 3ms short, so every shot was compiled into two windows
        # and each window was handed two voices describing the same seconds.
        compiled = film([15.08, 15.08, 14.5], window_seconds=15.1, policy="balanced")
        for window in compiled.windows:
            lip = [f for f in window.files if f.audio_role == AUDIO_ROLE_LIP_SYNC]
            self.assertEqual(len(lip), 1, "window %d" % (window.index + 1))
        self.assertNotIn("lip-sync references", notes(compiled))

    def test_the_widget_ceiling_now_reaches_a_whole_window(self):
        # The cap that made the straddle unavoidable. A shot may now be exactly as
        # long as the window it renders in.
        import ast
        import pathlib

        source = pathlib.Path("nodes.py").read_text(encoding="utf-8")
        self.assertIn('"max": MAX_WINDOW_FRAMES / FPS', source)
        ast.parse(source)
        self.assertAlmostEqual(MAX_WINDOW_FRAMES / 24.0, 15.0833, places=4)

    def test_a_genuine_two_shot_window_keeps_both_shots(self):
        compiled = film([6.0, 6.0], window_seconds=15.1, policy="fill")
        self.assertEqual(len(compiled.windows), 1)
        self.assertEqual(compiled.windows[0].shot_ids, ["s1", "s2"])

    def test_a_window_is_never_left_with_no_direction(self):
        """The fallback that earns its place: rendering on the style line alone is
        worse than the duplicate the floor exists to remove."""
        timeline = Timeline(
            assets=[], shots=[{"id": "s1", "start": 0.0, "duration": 20.0,
                               "prompt": "One long take."}],
            duration_seconds=20.0, fps=24, window_seconds=10.0)
        for window in compile_timeline(timeline, policy="balanced").windows:
            self.assertEqual(window.shot_ids, ["s1"])


class DuplicateLipSyncReferences(unittest.TestCase):

    def test_two_voices_in_one_window_are_reported(self):
        # What remains after the floor: two genuine speakers, two recordings cut
        # to the same seconds, and two "lip movements match" directives that
        # cannot both be obeyed.
        compiled = film([6.0, 6.0], window_seconds=15.1, policy="fill")
        text = " ".join(compiled.windows[0].diagnostics)
        self.assertIn("2 lip-sync references", text)
        self.assertIn("more than one answer", text)

    def test_nothing_is_dropped_to_fix_it(self):
        # Reported, never repaired. Choosing which copy to discard means guessing
        # which shot the author meant, and guessing wrong desynchronises the
        # mouth that was right.
        compiled = film([6.0, 6.0], window_seconds=15.1, policy="fill")
        lip = [f for f in compiled.windows[0].files
               if f.audio_role == AUDIO_ROLE_LIP_SYNC]
        self.assertEqual(len(lip), 2)
        self.assertTrue(compiled.ok)
        self.assertEqual(compiled.problems, [])


class DialogueThatDoesNotFitTheWindow(unittest.TestCase):
    """The estimate, consulted only when no recording answered the question."""

    def wordy(self, words, duration=12.25, **kwargs):
        line = " ".join("word%d" % i for i in range(words))
        return film([duration], window_seconds=12.3,
                    prompts=['He says "%s"' % line], **kwargs)

    def test_a_script_that_cannot_be_spoken_in_the_time_is_flagged(self):
        text = notes(self.wordy(60))
        self.assertIn("words of dialogue", text)
        self.assertIn("words per minute", text)
        self.assertRegex(text, r"this is \d+ over")

    def test_an_ordinary_delivery_is_silent(self):
        # 30 words in 12.25s is ~147 wpm, which is how narration is actually read.
        self.assertNotIn("words per minute", notes(self.wordy(30)))

    def test_the_threshold_is_where_the_constant_says_it_is(self):
        seconds = 12.25
        just_under = int(MAX_DIALOGUE_WPM * seconds / 60.0) - 1
        self.assertNotIn("words per minute", notes(self.wordy(just_under)))
        self.assertIn("words per minute", notes(self.wordy(just_under + 20)))

    def test_a_measured_recording_beats_the_estimate(self):
        # The take is 12.4s long and the window is 12.25s, so those words demonstrably
        # fit however fast the guess says they read. An exact answer is available;
        # second-guessing it would report a problem that is not there.
        self.assertNotIn("words per minute",
                         notes(self.wordy(60, source_seconds=12.4)))

    def test_direction_is_not_counted_as_speech(self):
        # Only <d> blocks count. Prose about the scene is instruction to the
        # camera, not words anybody has to say inside the window.
        long_direction = " ".join("adjective%d" % i for i in range(200))
        compiled = film([12.25], window_seconds=12.3,
                        prompts=[long_direction + ' He says "Access."'])
        self.assertNotIn("words per minute", notes(compiled))

    def test_a_shot_with_no_quoted_line_is_not_measured(self):
        self.assertNotIn("words per minute",
                         notes(film([12.25], window_seconds=12.3,
                                    prompts=["The speaker talks to camera."])))


class TheDialogueRegex(unittest.TestCase):

    def test_it_reads_back_what_wrap_dialogue_writes(self):
        written = wrap_dialogue('He says "Access." Then "Run it yourself."', "English")
        self.assertEqual(DIALOGUE_BLOCK_RE.findall(written),
                         ["Access.", "Run it yourself."])

    def test_the_language_tag_is_not_counted_as_speech(self):
        found = DIALOGUE_BLOCK_RE.findall("<d>[English] Two words.</d>")
        self.assertEqual(found, ["Two words."])
        self.assertEqual(len(found[0].split()), 2)


class TheOffsetReachesTheCacheKey(unittest.TestCase):
    """Two renders identical in every other field are different films.

    The digest cannot tell them apart -- it hashes the whole file either way --
    so the offset has to be in the key itself or the cache hands back a segment
    cut from the wrong seconds.
    """

    def base(self, **kwargs):
        return ref_descriptor(1, KIND_AUDIO, "Voice", "socket", sha256="abc",
                              audio_role=AUDIO_ROLE_LIP_SYNC, **kwargs)

    def test_a_zero_offset_leaves_every_key_on_disk_alone(self):
        self.assertNotIn("audio_offset", self.base())
        self.assertNotIn("audio_offset", self.base(audio_offset=0.0))
        self.assertEqual(_ref_descriptor_key(self.base()),
                         _ref_descriptor_key(self.base(audio_offset=0.0)))

    def test_a_real_offset_moves_the_key(self):
        self.assertNotEqual(_ref_descriptor_key(self.base()),
                            _ref_descriptor_key(self.base(audio_offset=24.5)))
        self.assertNotEqual(_ref_descriptor_key(self.base(audio_offset=12.25)),
                            _ref_descriptor_key(self.base(audio_offset=24.5)))

    def test_it_is_appended_last_so_the_fields_before_it_do_not_shift(self):
        plain = _ref_descriptor_key(self.base())
        self.assertEqual(_ref_descriptor_key(self.base(audio_offset=24.5)),
                         plain + [24.5])


class TheAssetCarriesIt(unittest.TestCase):

    def test_an_offsetless_bin_serialises_exactly_as_it_always_did(self):
        asset = Asset("a1", KIND_AUDIO, name="Voice", audio_role=AUDIO_ROLE_LIP_SYNC)
        self.assertNotIn("audio_offset", asset.to_dict())
        self.assertNotIn("source_seconds", asset.to_dict())

    def test_it_round_trips(self):
        asset = Asset("a1", KIND_AUDIO, name="Voice", audio_offset=24.5,
                      source_seconds=12.4)
        back = Asset.from_dict(asset.to_dict())
        self.assertEqual(back.audio_offset, 24.5)
        self.assertEqual(back.source_seconds, 12.4)

    def test_only_audio_carries_an_offset(self):
        # A picture has no clock to sit on, and an offset on one would be a field
        # nothing reads pretending to mean something.
        from comfyui_pulse_studio.assets import KIND_IMAGE

        self.assertEqual(Asset("i1", KIND_IMAGE, audio_offset=5.0).audio_offset, 0.0)

    def test_the_compiler_hands_it_to_the_executor(self):
        compiled = film(source_seconds=45.0, offsets={2: 12.25})
        window = compiled.windows[1]
        lip = [f for f in window.files if f.audio_role == AUDIO_ROLE_LIP_SYNC]
        self.assertEqual(len(lip), 1)
        self.assertEqual(lip[0].audio_offset, 12.25)
        self.assertEqual(lip[0].source_seconds, 45.0)


class TheTensorSlicing(unittest.TestCase):
    """`media.audio_span`, which is three lines around the arithmetic above."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in this environment")

    @staticmethod
    def _clip(seconds, level=0.5, rate=SR):
        import torch

        return {"waveform": torch.full((1, 2, int(seconds * rate)), float(level)),
                "sample_rate": rate}

    def test_a_lead_in_is_silence_then_the_recording(self):
        from media import audio_span

        out = audio_span(self._clip(10.0), -2.0, 5.0)
        waveform = out["waveform"]
        self.assertEqual(waveform.shape[-1], round(5.0 * SR))
        self.assertEqual(float(waveform[..., :2 * SR].abs().max()), 0.0)
        self.assertGreater(float(waveform[..., 2 * SR:].abs().max()), 0.0)

    def test_a_window_past_the_end_is_all_silence(self):
        from media import audio_span

        out = audio_span(self._clip(12.4), 24.5, 12.25)
        self.assertEqual(out["waveform"].shape[-1], round(12.25 * SR))
        self.assertEqual(float(out["waveform"].abs().max()), 0.0)

    def test_two_voices_are_mixed_rather_than_one_being_dropped(self):
        from media import mix_audio

        out = mix_audio([self._clip(2.0, 0.25), self._clip(2.0, 0.25)])
        self.assertAlmostEqual(float(out["waveform"].max()), 0.5, places=5)

    def test_a_mix_is_clamped_rather_than_normalised(self):
        # Normalising would make one loud window quieter than its neighbours,
        # which is a level step at a seam -- the artefact seam_gain_match removes.
        from media import mix_audio

        out = mix_audio([self._clip(2.0, 0.8), self._clip(2.0, 0.8)])
        self.assertAlmostEqual(float(out["waveform"].max()), 1.0, places=5)

    def test_mixing_nothing_is_nothing_and_mixing_one_is_that_one(self):
        from media import mix_audio

        self.assertIsNone(mix_audio([]))
        self.assertIsNone(mix_audio([None, None]))
        only = self._clip(1.0)
        self.assertIs(mix_audio([only, None]), only)

    def test_a_length_is_measured_or_honestly_unknown(self):
        from media import audio_length_seconds

        self.assertAlmostEqual(audio_length_seconds(self._clip(3.5)), 3.5, places=4)
        self.assertIsNone(audio_length_seconds(None))
        self.assertIsNone(audio_length_seconds({"waveform": None, "sample_rate": 0}))


if __name__ == "__main__":
    unittest.main()
