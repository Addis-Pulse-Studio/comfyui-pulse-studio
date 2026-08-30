"""What a reference audio is for, and what that changes.

Lip-sync to a supplied recording is a real H3 capability and this project shipped
without it working. Two things were missing and each was individually sufficient
to break it:

  * the prompt never asked for the match. The tokenizer emits only the marker
    "<Audio j>: " -- `comfy/text_encoders/minimax.py` says "audio never enters
    Qwen" -- so nothing about wiring the socket tells the model what the audio is
    for. The directive in the prose *is* the mechanism;
  * the clip was never trimmed to the window. Reference audio rows are packed
    against the target audio grid, so a 30s file against a 9.42s window asks the
    model to align two different spans of time.

Both are asserted here against the compiled output rather than against the node,
because both are properties of what reaches the model.
"""

import unittest

from comfyui_pulse_studio import compile_timeline
from comfyui_pulse_studio.assets import Asset, KIND_AUDIO, KIND_IMAGE
from comfyui_pulse_studio.compiler import CarryPolicy
from comfyui_pulse_studio.concat import audio_span_bounds
from comfyui_pulse_studio.constants import AUDIO_ROLE_LIP_SYNC, AUDIO_ROLE_TIMBRE
from comfyui_pulse_studio.pulse_timeline import ref_descriptor
from comfyui_pulse_studio.segcache import _ref_descriptor_key
from comfyui_pulse_studio.timeline import Timeline


def _timeline(role, shot_text='She says "I love the rain" @Voice.'):
    tl = Timeline(
        assets=[],
        shots=[{"id": "s1", "start": 0.0, "duration": 6.0, "prompt": shot_text}],
        duration_seconds=6.0, fps=24,
    )
    tl.local_refs["s1"] = [
        Asset("img_s1", KIND_IMAGE, name="Ref1", file=""),
        Asset("aud_s1", KIND_AUDIO, name="Voice", file="", audio_role=role),
    ]
    return tl


def _compile(tl):
    return compile_timeline(tl, carry=CarryPolicy(mode="image", audio=True))


class TestTheDirectiveReachesTheModel(unittest.TestCase):
    def test_lip_sync_asks_for_the_match_by_tag(self):
        window = _compile(_timeline(AUDIO_ROLE_LIP_SYNC)).windows[0]
        tag = window.tag_map.by_id["aud_s1"]
        self.assertIn("lip movements match", window.prompt)
        # Naming the tag is the part that matters: "their lips match the audio"
        # with no ordinal leaves the model to guess which reference is meant on a
        # shot carrying more than one.
        directive = [line for line in window.prompt.splitlines()
                     if "lip movements match" in line][0]
        self.assertIn(tag, directive)

    def test_timbre_does_not_promise_a_match(self):
        window = _compile(_timeline(AUDIO_ROLE_TIMBRE)).windows[0]
        self.assertIn("voice-timbre reference", window.prompt)
        self.assertNotIn("lip movements", window.prompt)

    def test_the_two_roles_produce_different_prompts(self):
        """Otherwise the widget is decoration and the cache cannot tell them apart."""
        lip = _compile(_timeline(AUDIO_ROLE_LIP_SYNC)).windows[0].prompt
        timbre = _compile(_timeline(AUDIO_ROLE_TIMBRE)).windows[0].prompt
        self.assertNotEqual(lip, timbre)

    def test_the_role_rides_on_the_file_ref_for_the_executor(self):
        """render.py trims on this field; losing it silently disables the trim."""
        window = _compile(_timeline(AUDIO_ROLE_LIP_SYNC)).windows[0]
        audio = [f for f in window.files if f.kind == KIND_AUDIO and not f.synthetic]
        self.assertEqual([f.audio_role for f in audio], [AUDIO_ROLE_LIP_SYNC])

    def test_carry_over_audio_is_never_given_a_role(self):
        """It is the previous window's score, not anyone's voice. Trimming it to the
        window would defeat the tail, and calling it lip-sync would be a lie."""
        tl = Timeline(
            assets=[], fps=24, duration_seconds=30.0,
            shots=[{"id": "s%d" % i, "start": i * 10.0, "duration": 10.0,
                    "prompt": "A shot."} for i in range(3)])
        plan = _compile(tl)
        self.assertGreater(len(plan.windows), 1)
        for window in plan.windows[1:]:
            for ref in window.files:
                if ref.synthetic and ref.kind == KIND_AUDIO:
                    self.assertIsNone(ref.audio_role)


class TestQuotedDialogueAgainstALipSyncReference(unittest.TestCase):
    """Two answers to "what is she saying", and no way to see which the model took.

    The <d> block instructs the model to speak those words; the recording says
    whatever it says. Reported rather than resolved, because a quote that IS the
    recording's transcript is legitimate and only the author knows.
    """

    def _diags(self, line, role=AUDIO_ROLE_LIP_SYNC):
        plan = _compile(_timeline(role, shot_text=line))
        return [d for d in plan.windows[0].diagnostics if "lip_sync reference" in d]

    def test_a_quote_beside_a_lip_sync_reference_is_reported(self):
        self.assertTrue(self._diags('She says "I love the rain" @Voice.'))

    def test_the_diagnostic_names_the_reference(self):
        """"Some audio somewhere conflicts" is not actionable on a shot with two."""
        self.assertIn("@Voice", self._diags('She says "hello" @Voice.')[0])

    def test_no_quote_no_complaint(self):
        self.assertEqual(self._diags("She speaks to camera @Voice."), [])

    def test_a_timbre_reference_may_have_all_the_dialogue_it_likes(self):
        """That is the mode where the model *should* speak the written words."""
        self.assertEqual(
            self._diags('She says "I love the rain" @Voice.', AUDIO_ROLE_TIMBRE), [])

    def test_the_compile_still_succeeds(self):
        """A warning, not a refusal -- the author may know the quote matches."""
        plan = _compile(_timeline(AUDIO_ROLE_LIP_SYNC,
                                  shot_text='She says "hello" @Voice.'))
        self.assertTrue(plan.ok, plan.problems)


class TestAssetCarriesTheRole(unittest.TestCase):
    def test_round_trips_through_a_dict(self):
        asset = Asset("a", KIND_AUDIO, name="Voice", audio_role=AUDIO_ROLE_LIP_SYNC)
        self.assertEqual(Asset.from_dict(asset.to_dict()).audio_role,
                         AUDIO_ROLE_LIP_SYNC)

    def test_absent_from_a_dict_written_before_the_field_existed(self):
        self.assertIsNone(
            Asset.from_dict({"id": "a", "kind": KIND_AUDIO, "name": "V"}).audio_role)

    def test_only_audio_carries_it(self):
        """A role on an image would ride into the cache key and invalidate segments
        for a distinction that means nothing."""
        self.assertIsNone(
            Asset("i", KIND_IMAGE, name="Ref1", audio_role=AUDIO_ROLE_LIP_SYNC).audio_role)


class TestTheRoleReachesTheCacheKey(unittest.TestCase):
    """The role changes the prompt, so it must change the key.

    The directive lives in the window's subject definitions, which no shot's
    `resolved_prompt` covers -- so without this the widget would change what the
    model is told while the cache quietly returned the segment rendered under the
    other instruction. Silent reuse of work done under different instructions is
    the specific thing the content-addressed key exists to make impossible.
    """

    def _key(self, role):
        return _ref_descriptor_key(ref_descriptor(
            1, KIND_AUDIO, "Voice", source="socket", sha256="abc",
            audio_role=role))

    def test_the_two_roles_key_differently(self):
        self.assertNotEqual(self._key(AUDIO_ROLE_LIP_SYNC),
                            self._key(AUDIO_ROLE_TIMBRE))

    def test_a_reference_with_no_role_keys_exactly_as_before(self):
        """Adding a field must not invalidate the cache of every project that has
        no audio reference at all."""
        self.assertEqual(self._key(None),
                         [1, KIND_AUDIO, "Voice", "abc"])

    def test_the_role_is_absent_from_the_descriptor_when_unset(self):
        self.assertNotIn("audio_role",
                         ref_descriptor(1, KIND_IMAGE, "Ref1", source="socket"))


class TestTheTrimIsToTheWindow(unittest.TestCase):
    """The alignment arithmetic, on the pure function, with no torch anywhere.

    `media.audio_span` is three lines of tensor slicing around
    `concat.audio_span_bounds`. The split exists because the placement maths in
    this project shipped wrong twice while nothing without PyAV could reach it,
    and this is the same class of arithmetic.
    """

    SR = 44100
    WINDOW = 9.4167          # 226 frames at 24fps, the real long-form window

    def test_a_long_clip_is_cut_to_the_window(self):
        start, stop, head, tail = audio_span_bounds(
            30 * self.SR, self.SR, 0.0, self.WINDOW)
        self.assertEqual(stop - start, round(self.WINDOW * self.SR))
        self.assertEqual((head, tail), (0, 0))

    def test_window_two_begins_exactly_where_window_one_ended(self):
        """The property the whole feature rests on. A gap or an overlap here puts
        every mouth after the seam out of step with its own voice."""
        _, first_stop, _, _ = audio_span_bounds(
            30 * self.SR, self.SR, 0.0, self.WINDOW)
        second_start, _, _, _ = audio_span_bounds(
            30 * self.SR, self.SR, self.WINDOW, self.WINDOW)
        self.assertEqual(second_start, first_stop)

    def test_a_short_clip_is_padded_rather_than_shortening_the_window(self):
        start, stop, head, tail = audio_span_bounds(
            2 * self.SR, self.SR, 0.0, self.WINDOW)
        self.assertEqual(stop - start, 2 * self.SR)
        self.assertEqual(head, 0)
        self.assertEqual((stop - start) + tail, round(self.WINDOW * self.SR))

    def test_a_window_past_the_end_of_the_recording_is_all_silence(self):
        start, stop, head, tail = audio_span_bounds(
            2 * self.SR, self.SR, 60.0, self.WINDOW)
        self.assertEqual(stop - start, 0)
        self.assertEqual(head, 0)
        self.assertEqual(tail, round(self.WINDOW * self.SR))

    def test_every_window_gets_the_same_number_of_samples(self):
        """Real or padded, the length is the window's length -- otherwise the track
        drifts against the video a little further at each seam."""
        want = round(self.WINDOW * self.SR)
        for total_seconds in (30, 12, 2, 0):
            for w in range(3):
                start, stop, head, tail = audio_span_bounds(
                    total_seconds * self.SR, self.SR, w * self.WINDOW, self.WINDOW)
                self.assertEqual(head + (stop - start) + tail, want,
                                 "%ds file, window %d" % (total_seconds, w))

    def test_a_non_44k_rate_is_honoured(self):
        _, stop, _, _ = audio_span_bounds(30 * 32000, 32000, 0.0, 5.0)
        self.assertEqual(stop, 5 * 32000)

    def test_a_zero_sample_rate_is_refused_rather_than_dividing_by_it(self):
        with self.assertRaises(ValueError):
            audio_span_bounds(1000, 0, 0.0, 1.0)


class TestTheTensorSlicing(unittest.TestCase):
    """The three lines around the arithmetic. Skipped without torch by design --
    everything that can be checked without it is in the class above."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("tensor slicing needs torch")

    def _clip(self, seconds, sample_rate=44100, channels=2):
        import torch
        n = int(seconds * sample_rate)
        ramp = torch.arange(n, dtype=torch.float32).repeat(channels, 1)
        return {"waveform": ramp.unsqueeze(0), "sample_rate": sample_rate}

    def test_it_starts_where_the_window_starts(self):
        import media
        out = media.audio_span(self._clip(30.0), 9.4167, 9.4167)
        self.assertAlmostEqual(float(out["waveform"][0, 0, 0]),
                               round(9.4167 * 44100), delta=2)

    def test_a_short_clip_is_padded_with_silence(self):
        import media
        out = media.audio_span(self._clip(2.0), 0.0, 9.4167)
        self.assertEqual(out["waveform"].shape[-1], round(9.4167 * 44100))
        self.assertEqual(float(out["waveform"][0, 0, -1]), 0.0)

    def test_silence_matches_the_rate_and_channels_it_is_spliced_between(self):
        import media
        out = media.silent_audio(2.0, sample_rate=32000, channels=1)
        self.assertEqual(out["sample_rate"], 32000)
        self.assertEqual(out["waveform"].shape[1], 1)
        self.assertEqual(out["waveform"].shape[-1], 2 * 32000)


if __name__ == "__main__":
    unittest.main()


class TestAnInertAudioModeIsReported(unittest.TestCase):
    """ref_audio_mode with nothing wired to ref_audio does nothing, and said so
    nowhere.

    The rule is deliberately not "warn whenever the mode is set and the socket is
    empty". `ref_audio_mode` *defaults* to lip_sync, so that condition is the
    resting state of every shot in every film that does not use reference audio,
    and firing on it would put a warning on each of the three shots in the
    shipped long-form graph -- which is doing nothing wrong. See
    `nodes._inert_audio_modes`.

    Tested against the helper rather than through PulseSlate.execute, which needs
    torch and a running ComfyUI; the helper is the whole decision.
    """

    @staticmethod
    def _helper():
        # nodes.py imports torch, so the two pure functions are lifted out of it
        # by ast rather than imported. Same technique as tests/test_workflow.py.
        import ast
        from pathlib import Path

        source = Path(__file__).parent.parent / "nodes.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        wanted = {"_inert_audio_modes", "_shot_list"}
        module = ast.Module(
            body=[n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name in wanted],
            type_ignores=[])
        namespace = {"AUDIO_ROLE_LIP_SYNC": AUDIO_ROLE_LIP_SYNC,
                     "AUDIO_ROLE_TIMBRE": AUDIO_ROLE_TIMBRE}
        exec(compile(module, "nodes.py", "exec"), namespace)
        return namespace["_inert_audio_modes"]

    def test_a_film_with_no_reference_audio_stays_quiet(self):
        """The shipped long-form graph's exact shape. Three shots, no audio."""
        inert = [("Shot %d" % i, AUDIO_ROLE_LIP_SYNC) for i in (1, 2, 3)]
        self.assertEqual(self._helper()(inert, False), [])

    def test_a_shot_missed_in_a_film_that_does_use_audio_is_reported(self):
        notes = self._helper()([("Shot 3", AUDIO_ROLE_LIP_SYNC)], True)
        self.assertEqual(len(notes), 1)
        self.assertIn("Shot 3", notes[0])
        self.assertIn(AUDIO_ROLE_LIP_SYNC, notes[0])

    def test_voice_timbre_is_reported_even_with_no_audio_anywhere(self):
        """Not the default, so somebody chose it, and it cannot do anything."""
        notes = self._helper()([("Shot 2", AUDIO_ROLE_TIMBRE)], False)
        self.assertEqual(len(notes), 1)
        self.assertIn(AUDIO_ROLE_TIMBRE, notes[0])

    def test_a_fully_wired_film_reports_nothing(self):
        self.assertEqual(self._helper()([], True), [])

    def test_several_missed_shots_are_one_note_not_several(self):
        inert = [("Shot %d" % i, AUDIO_ROLE_LIP_SYNC) for i in (2, 3, 4)]
        notes = self._helper()(inert, True)
        self.assertEqual(len(notes), 1)
        for label in ("Shot 2", "Shot 3", "Shot 4"):
            self.assertIn(label, notes[0])

    def test_a_long_list_is_truncated_rather_than_dumped(self):
        inert = [("Shot %d" % i, AUDIO_ROLE_LIP_SYNC) for i in range(9)]
        notes = self._helper()(inert, True)
        self.assertIn("5 others", notes[0])
        self.assertNotIn("Shot 8", notes[0])

    def test_both_kinds_at_once_are_two_notes(self):
        notes = self._helper()([("Shot 1", AUDIO_ROLE_TIMBRE),
                                ("Shot 2", AUDIO_ROLE_LIP_SYNC)], True)
        self.assertEqual(len(notes), 2)
