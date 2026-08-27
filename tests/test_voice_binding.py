"""Which recording is whose voice, on a bin that holds more than one.

`PulseShot.speaker` answers this for a shot's own `ref_audio`: the node carrying
the recording also carries the character. A voice dropped in the Asset Bin has no
shot to read a speaker off, so it has to say who it belongs to itself -- and
until `voice_of` existed it could not, which meant a film with three characters
and three voice files in the bin could bind none of them.

Bound by asset id, never by ordinal or slot. A binding to "reference 3" follows
whatever lands in slot 3 after the next bin edit and reports nothing; that is the
exact failure the whole asset module is built to prevent.
"""

import ast
import unittest
from pathlib import Path

from comfyui_pulse_studio import compile_timeline
from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO, Asset, AssetBin
from comfyui_pulse_studio.binops import apply_operation, bin_state
from comfyui_pulse_studio.compiler import CarryPolicy, assign_speaker_ids
from comfyui_pulse_studio.constants import AUDIO_ROLE_LIP_SYNC, AUDIO_ROLE_TIMBRE
from comfyui_pulse_studio.pulse_timeline import ref_descriptor
from comfyui_pulse_studio.segcache import _ref_descriptor_key
from comfyui_pulse_studio.timeline import Timeline

CAST = [
    {"id": "mimi", "kind": KIND_IMAGE, "name": "Mimi", "file": "a.png",
     "description": "a woman in her thirties with dark hair"},
    {"id": "kade", "kind": KIND_IMAGE, "name": "Kade", "file": "b.png",
     "description": "a man in his fifties in a blue work coat"},
]


def _film(voices, shots=None, **kwargs):
    tl = Timeline(
        assets=[dict(a) for a in CAST] + list(voices),
        shots=shots or [
            {"id": "s1", "start": 0.0, "duration": 5.0,
             "prompt": '@Mimi says "you\'re still open"', "speakers": ["mimi"]},
            {"id": "s2", "start": 5.0, "duration": 5.0,
             "prompt": '@Kade says "for you, always"', "speakers": ["kade"]},
        ],
        duration_seconds=10.0, fps=24, **kwargs)
    return tl


def _voice(asset_id, name, owner, role=AUDIO_ROLE_TIMBRE):
    return {"id": asset_id, "kind": KIND_AUDIO, "name": name, "file": name + ".wav",
            "audio_role": role, "voice_of": owner}


def _compile(tl):
    return compile_timeline(tl, carry=CarryPolicy(mode="image", audio=True))


class TestTheFieldItself(unittest.TestCase):
    def test_it_round_trips_through_a_dict(self):
        asset = Asset("a", KIND_AUDIO, name="VO", voice_of="mimi")
        self.assertEqual(Asset.from_dict(asset.to_dict()).voice_of, "mimi")

    def test_absent_from_a_dict_written_before_the_field_existed(self):
        self.assertIsNone(Asset.from_dict({"id": "a", "kind": KIND_AUDIO, "name": "V"}).voice_of)

    def test_only_audio_carries_it(self):
        """A picture is not the voice of anybody, and a stray value here would
        ride into the cache key for a distinction that means nothing."""
        self.assertIsNone(Asset("i", KIND_IMAGE, name="Mimi", voice_of="kade").voice_of)

    def test_it_is_absent_from_the_dict_when_unset(self):
        self.assertNotIn("voice_of", Asset("a", KIND_AUDIO, name="VO").to_dict())


class TestTwoVoicesInTheBin(unittest.TestCase):
    """The case the feature exists for: more than one recording, more than one
    character, and nothing in the sockets to tell them apart."""

    def setUp(self):
        self.prompt = _compile(_film([
            _voice("v_mimi", "MimiVO", "mimi"),
            _voice("v_kade", "KadeVO", "kade"),
        ])).windows[0].prompt

    def test_each_recording_names_its_own_character(self):
        self.assertIn("`<Audio 1>` is the voice-timbre reference for <Subject 1> (S1).",
                      self.prompt)
        self.assertIn("`<Audio 2>` is the voice-timbre reference for <Subject 2> (S2).",
                      self.prompt)

    def test_each_gets_its_own_retention_line(self):
        self.assertIn("`<Audio 1>` (the voice of <Subject 1> (S1)): reference", self.prompt)
        self.assertIn("`<Audio 2>` (the voice of <Subject 2> (S2)): reference", self.prompt)

    def test_a_lip_sync_bin_voice_asks_for_the_match_by_character(self):
        prompt = _compile(_film([
            _voice("v_mimi", "MimiVO", "mimi", AUDIO_ROLE_LIP_SYNC)])).windows[0].prompt
        self.assertIn("`<Audio 1>` is the speech <Subject 1> (S1) is saying", prompt)
        self.assertIn("`<Audio 1>` (the voice of <Subject 1> (S1)): fully_copy", prompt)

    def test_an_unbound_recording_beside_a_bound_one_stays_anonymous(self):
        """Ambience does not become somebody's voice because a voice is in the
        bin next to it."""
        prompt = _compile(_film([
            _voice("v_mimi", "MimiVO", "mimi"),
            {"id": "amb", "kind": KIND_AUDIO, "name": "Rain", "file": "r.wav"},
        ])).windows[0].prompt
        self.assertIn("`<Audio 1>` is the voice-timbre reference for <Subject 1> (S1).",
                      prompt)
        self.assertIn("`<Audio 2>` is a voice-timbre reference.", prompt)


class TestAVoiceMakesItsOwnerASpeaker(unittest.TestCase):
    def test_a_character_no_shot_names_still_gets_an_id(self):
        """Supplying somebody's voice makes them a speaker whether or not a shot
        named them -- otherwise the binding cannot be expressed at all."""
        tl = _film([_voice("v_kade", "KadeVO", "kade")],
                   shots=[{"id": "s1", "start": 0.0, "duration": 5.0,
                           "prompt": "@Mimi waits.", "speakers": ["mimi"]}])
        self.assertEqual(
            assign_speaker_ids(tl.ordered_shots(), tl.assets.by_kind(KIND_AUDIO)),
            {"mimi": "S1", "kade": "S2"})

    def test_a_voice_never_renumbers_a_character_who_has_lines(self):
        """Bin voices are numbered after every speaking part, so adding one
        cannot shift an id that is already on screen."""
        tl = _film([_voice("v_kade", "KadeVO", "kade")])
        self.assertEqual(
            assign_speaker_ids(tl.ordered_shots(), tl.assets.by_kind(KIND_AUDIO))["mimi"],
            "S1")


class TestAnExplicitBindingOutranksAnInferredOne(unittest.TestCase):
    def test_voice_of_wins_over_the_shot_that_carries_the_recording(self):
        """The author naming a character beats the wiring implying one."""
        tl = _film([])
        tl.local_refs["s1"] = [Asset("aud_s1", KIND_AUDIO, name="Voice", file="",
                                     audio_role=AUDIO_ROLE_TIMBRE, voice_of="kade")]
        prompt = _compile(tl).windows[0].prompt
        self.assertIn("`<Audio 1>` is the voice-timbre reference for <Subject 2> (S2).",
                      prompt)


class TestABindingThatPointsAtNothing(unittest.TestCase):
    def test_an_absent_owner_is_refused(self):
        plan = _compile(_film([_voice("v", "VO", "nobody")]))
        self.assertFalse(plan.ok)
        self.assertTrue([p for p in plan.problems if "not in the bin" in p])

    def test_another_recording_is_refused(self):
        tl = _film([_voice("v1", "One", None), _voice("v2", "Two", "v1")])
        plan = _compile(tl)
        self.assertFalse(plan.ok)
        self.assertTrue([p for p in plan.problems if "a voice belongs to a character" in p])


class TestTheBinPanelCanSetIt(unittest.TestCase):
    def _bin(self):
        return AssetBin([Asset(a["id"], a["kind"], name=a["name"], file=a["file"],
                               description=a.get("description", "")) for a in CAST]
                        + [Asset("v", KIND_AUDIO, name="VO", file="v.wav")])

    def test_binding_stores_an_id(self):
        bin_ = self._bin()
        apply_operation(bin_, "set_voice_of", asset_id="v", voice_of="mimi")
        self.assertEqual(bin_.get("v").voice_of, "mimi")

    def test_an_empty_value_unbinds(self):
        bin_ = self._bin()
        apply_operation(bin_, "set_voice_of", asset_id="v", voice_of="mimi")
        apply_operation(bin_, "set_voice_of", asset_id="v", voice_of="")
        self.assertIsNone(bin_.get("v").voice_of)

    def test_binding_to_an_unknown_id_is_refused(self):
        with self.assertRaises(ValueError):
            apply_operation(self._bin(), "set_voice_of", asset_id="v", voice_of="ghost")

    def test_binding_a_picture_to_someone_is_refused(self):
        with self.assertRaises(ValueError):
            apply_operation(self._bin(), "set_voice_of", asset_id="mimi", voice_of="kade")

    def test_the_role_is_settable_on_a_bin_recording(self):
        """Until this existed every bin audio compiled as a timbre reference and
        only a PulseShot's own socket could ever ask for lip sync."""
        bin_ = self._bin()
        apply_operation(bin_, "set_audio_role", asset_id="v",
                        audio_role=AUDIO_ROLE_LIP_SYNC)
        self.assertEqual(bin_.get("v").audio_role, AUDIO_ROLE_LIP_SYNC)

    def test_an_invented_role_is_refused_rather_than_written_into_the_prompt(self):
        with self.assertRaises(ValueError):
            apply_operation(self._bin(), "set_audio_role", asset_id="v",
                            audio_role="lipsync")

    def test_the_panel_is_told_who_a_voice_can_belong_to(self):
        state = bin_state(self._bin())
        self.assertEqual([c["id"] for c in state["cast"]], ["mimi", "kade"])
        self.assertEqual(state["audio_roles"], [AUDIO_ROLE_LIP_SYNC, AUDIO_ROLE_TIMBRE])

    def test_a_recording_reports_its_own_role_and_owner(self):
        bin_ = self._bin()
        apply_operation(bin_, "set_voice_of", asset_id="v", voice_of="kade")
        row = [r for r in bin_state(bin_)["assets"] if r["id"] == "v"][0]
        self.assertEqual(row["voice_of"], "kade")
        self.assertIn("audio_role", row)


class TestTheBindingReachesTheCacheKey(unittest.TestCase):
    def _key(self, **kwargs):
        return _ref_descriptor_key(ref_descriptor(
            1, KIND_AUDIO, "VO", source="bin", sha256="abc", **kwargs))

    def test_two_owners_key_differently(self):
        self.assertNotEqual(self._key(voice_of="mimi"), self._key(voice_of="kade"))

    def test_binding_a_previously_unbound_voice_moves_the_key(self):
        self.assertNotEqual(self._key(), self._key(voice_of="mimi"))

    def test_an_unbound_reference_keys_exactly_as_before(self):
        """Adding a field must not invalidate the cache of every project that
        binds nothing."""
        self.assertEqual(self._key(), [1, KIND_AUDIO, "VO", "abc"])

    def test_the_role_still_comes_first(self):
        """Appended after audio_role, so a bound-and-roled reference keys
        stably rather than depending on dict order."""
        self.assertEqual(self._key(audio_role=AUDIO_ROLE_LIP_SYNC, voice_of="mimi"),
                         [1, KIND_AUDIO, "VO", "abc", AUDIO_ROLE_LIP_SYNC, "mimi"])


class TestCarryOverDoesNotEvictABoundVoice(unittest.TestCase):
    """Carry-over claims the front of the audio group, so on a continuation
    window the last user recording is dropped. Chosen by bin position, that is a
    coin flip; a character keeps their picture and their lines and loses their
    voice from window 2 onward, which reads as drift rather than as a missing
    reference.
    """

    def _windows(self, voices):
        tl = _film(voices, shots=[
            {"id": "s%d" % i, "start": i * 10.0, "duration": 10.0,
             "prompt": "@Mimi talks." if i % 2 == 0 else "@Kade talks.",
             "speakers": ["mimi"] if i % 2 == 0 else ["kade"]} for i in range(4)])
        tl.duration_seconds = 40.0
        plan = _compile(tl)
        self.assertGreater(len(plan.windows), 1, "needs a continuation window")
        return plan.windows

    def test_an_unbound_clip_is_dropped_before_a_bound_one(self):
        voices = [
            {"id": "amb", "kind": KIND_AUDIO, "name": "Rain", "file": "r.wav"},
            _voice("v_mimi", "MimiVO", "mimi"),
            _voice("v_kade", "KadeVO", "kade"),
        ]
        second = self._windows(voices)[1]
        kept = {f.asset_id for f in second.files if f.kind == KIND_AUDIO}
        self.assertIn("v_mimi", kept)
        self.assertIn("v_kade", kept)
        self.assertNotIn("amb", kept)

    def test_the_drop_is_still_reported(self):
        voices = [
            {"id": "amb", "kind": KIND_AUDIO, "name": "Rain", "file": "r.wav"},
            _voice("v_mimi", "MimiVO", "mimi"),
            _voice("v_kade", "KadeVO", "kade"),
        ]
        second = self._windows(voices)[1]
        self.assertTrue([d for d in second.diagnostics if "Rain" in d and "dropped" in d])

    def test_timbre_is_dropped_before_lip_sync_when_both_are_bound(self):
        """Losing an alignment desynchronises a mouth, which is visible. Losing a
        timbre reference only changes how a voice sounds."""
        voices = [
            _voice("v_kade", "KadeVO", "kade", AUDIO_ROLE_TIMBRE),
            _voice("v_mimi", "MimiVO", "mimi", AUDIO_ROLE_LIP_SYNC),
            _voice("v_extra", "Extra", "mimi", AUDIO_ROLE_LIP_SYNC),
        ]
        second = self._windows(voices)[1]
        kept = {f.asset_id for f in second.files if f.kind == KIND_AUDIO}
        self.assertNotIn("v_kade", kept)

    def test_survivors_keep_their_bin_order(self):
        """Ordinals are bin order. Re-sorting the survivors would renumber a
        window that is not over budget at all."""
        voices = [
            {"id": "amb", "kind": KIND_AUDIO, "name": "Rain", "file": "r.wav"},
            _voice("v_mimi", "MimiVO", "mimi"),
            _voice("v_kade", "KadeVO", "kade"),
        ]
        second = self._windows(voices)[1]
        audio = [f.asset_id for f in second.files
                 if f.kind == KIND_AUDIO and not f.synthetic]
        self.assertEqual(audio, ["v_mimi", "v_kade"])


class TestTheSinkWarningCountsCharacters(unittest.TestCase):
    """§12.6 asserts that this timeline pairs voices with separate characters, and
    that assertion has to be true before it is worth printing. Counting audio
    assets fired it at films whose three references were ambience.

    Lifted out of nodes.py by ast -- nodes.py imports torch.
    """

    @staticmethod
    def _count():
        source = Path(__file__).parent.parent / "nodes.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        module = ast.Module(
            body=[n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_paired_audio_count"],
            type_ignores=[])
        namespace = {"KIND_AUDIO": KIND_AUDIO}
        exec(compile(module, "nodes.py", "exec"), namespace)
        return namespace["_paired_audio_count"]

    def test_unbound_clips_count_for_nothing(self):
        tl = _film([{"id": "a%d" % i, "kind": KIND_AUDIO, "name": "Amb%d" % i,
                     "file": "a.wav"} for i in range(3)])
        self.assertEqual(self._count()(tl), 0)

    def test_two_voices_on_one_character_count_once(self):
        tl = _film([_voice("v1", "One", "mimi"), _voice("v2", "Two", "mimi")])
        self.assertEqual(self._count()(tl), 1)

    def test_two_characters_with_voices_count_two(self):
        tl = _film([_voice("v1", "One", "mimi"), _voice("v2", "Two", "kade")])
        self.assertEqual(self._count()(tl), 2)

    def test_a_shot_local_recording_counts_through_its_speaker(self):
        tl = _film([])
        tl.local_refs["s1"] = [Asset("aud_s1", KIND_AUDIO, name="Voice", file="")]
        tl.local_refs["s2"] = [Asset("aud_s2", KIND_AUDIO, name="Voice", file="")]
        self.assertEqual(self._count()(tl), 2)

    def test_a_shot_recording_with_no_speaker_counts_for_nothing(self):
        tl = _film([])
        tl.shots[0].speakers = []
        tl.shots[1].speakers = []
        tl.local_refs["s1"] = [Asset("aud_s1", KIND_AUDIO, name="Voice", file="")]
        self.assertEqual(self._count()(tl), 0)


class TestASoundtrackRenumbersBoundLines(unittest.TestCase):
    """Enabling a video's own soundtrack claims an <Audio j> ordinal ahead of
    every standalone recording, so every bound sentence renumbers. The tags are
    computed from bin order on every compile, so this is safe by construction --
    which is exactly why it is worth an assertion rather than an assumption.
    """

    def _prompt(self, include_audio):
        tl = _film([_voice("v_mimi", "MimiVO", "mimi")])
        tl.assets.add(Asset("clip", KIND_VIDEO, name="Clip", file="c.mp4",
                            description="the street outside",
                            include_audio=include_audio))
        return _compile(tl).windows[0].prompt

    def test_the_bound_line_moves_to_the_next_ordinal(self):
        self.assertIn("`<Audio 1>` is the voice-timbre reference for <Subject 1> (S1).",
                      self._prompt(False))
        self.assertIn("`<Audio 2>` is the voice-timbre reference for <Subject 1> (S1).",
                      self._prompt(True))

    def test_it_still_names_the_same_character(self):
        for include in (False, True):
            prompt = self._prompt(include)
            line = next(l for l in prompt.splitlines() if "voice-timbre reference for" in l)
            self.assertIn("<Subject 1> (S1)", line)

    def test_the_retention_line_moves_with_it(self):
        self.assertIn("`<Audio 2>` (the voice of <Subject 1> (S1)): reference",
                      self._prompt(True))


if __name__ == "__main__":
    unittest.main()
