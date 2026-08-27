"""Binding a voice to a face: <Subject N>, a speaker id, and the audio that is theirs.

H3 has no structural channel for this. `comfy/text_encoders/minimax.py` emits the
literal string "<Audio j>: " and nothing else -- the waveform never reaches Qwen
-- so an audio socket carries an ordinal and no owner. On a one-hander that is
enough, because there is only one mouth. On a two-hander it is the shape of
Comfy-Org/ComfyUI#15454: the right character's lips move, and the other
character's accent comes out of them.

The binding is prose or it does not exist, and it is three pieces of prose:

  * a <Subject N> definition citing the picture that defines the character;
  * a *global* speaker id, (S1), (S2), assigned at first vocal appearance and
    unchanged for the rest of the film;
  * a sentence naming the <Audio j> as that subject's voice.

Everything here is asserted against the compiled prompt rather than against the
node, because the prompt is the entire mechanism.
"""

import ast
import unittest
from pathlib import Path

from comfyui_pulse_studio import compile_timeline
from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, Asset
from comfyui_pulse_studio.compiler import CarryPolicy, assign_speaker_ids
from comfyui_pulse_studio.constants import AUDIO_ROLE_LIP_SYNC, AUDIO_ROLE_TIMBRE
from comfyui_pulse_studio.pulse_timeline import shot_block
from comfyui_pulse_studio.segcache import _shot_key
from comfyui_pulse_studio.timeline import Timeline

CAST = [
    {"id": "mimi", "kind": KIND_IMAGE, "name": "Mimi", "file": "a.png",
     "description": "a woman in her thirties with dark hair and a red scarf"},
    {"id": "kade", "kind": KIND_IMAGE, "name": "Kade", "file": "b.png",
     "description": "a man in his fifties in a blue work coat"},
]


def _two_hander(roles=(AUDIO_ROLE_LIP_SYNC, AUDIO_ROLE_TIMBRE), speakers=True):
    tl = Timeline(
        assets=[dict(a) for a in CAST],
        shots=[
            {"id": "s1", "start": 0.0, "duration": 5.0,
             "prompt": '@Mimi crosses to the counter. She says "you\'re still open"',
             "speakers": ["mimi"] if speakers else []},
            {"id": "s2", "start": 5.0, "duration": 5.0,
             "prompt": "@Kade sets the glass down.",
             "speakers": ["kade"] if speakers else []},
        ],
        duration_seconds=10.0, fps=24)
    for shot_id, role in zip(("s1", "s2"), roles):
        tl.local_refs[shot_id] = [
            Asset("aud_" + shot_id, KIND_AUDIO, name="Voice", file="", audio_role=role)]
    return tl


def _compile(tl):
    return compile_timeline(tl, carry=CarryPolicy(mode="image", audio=True))


def _line(prompt, needle):
    return next(line for line in prompt.splitlines() if needle in line)


class TestTheIdsThemselves(unittest.TestCase):
    def test_assigned_in_order_of_first_appearance(self):
        tl = _two_hander()
        self.assertEqual(assign_speaker_ids(tl.ordered_shots()),
                         {"mimi": "S1", "kade": "S2"})

    def test_a_character_who_speaks_twice_keeps_one_id(self):
        tl = _two_hander()
        tl.shots.append(type(tl.shots[0])("s3", start=10.0, duration=5.0,
                                          prompt="@Mimi again.", speakers=["mimi"]))
        self.assertEqual(assign_speaker_ids(tl.ordered_shots()),
                         {"mimi": "S1", "kade": "S2"})

    def test_order_is_the_clock_not_the_list(self):
        """A shot list is not required to be sorted, and numbering by list order
        would hand out ids in whatever order the PulseShot nodes were wired."""
        tl = _two_hander()
        tl.shots.reverse()
        self.assertEqual(assign_speaker_ids(tl.ordered_shots()),
                         {"mimi": "S1", "kade": "S2"})

    def test_a_film_with_no_speakers_assigns_nothing(self):
        self.assertEqual(assign_speaker_ids(_two_hander(speakers=False).ordered_shots()), {})


class TestTheIdsSurviveTheSeam(unittest.TestCase):
    """The regression that per-window numbering would cause, and the reason the
    assignment happens once for the whole film rather than inside _compile_window.

    A speaker id is what tells the model that the person talking after the cut is
    the person who talked before it. Renumbering at each window turns every
    character into a new person at every seam -- and it would look correct in any
    single-window test.
    """

    def _plan(self):
        tl = Timeline(
            assets=[dict(a) for a in CAST],
            shots=[{"id": "s%d" % i, "start": i * 10.0, "duration": 10.0,
                    "prompt": "@Mimi talks." if i % 2 == 0 else "@Kade talks.",
                    "speakers": ["mimi"] if i % 2 == 0 else ["kade"]}
                   for i in range(4)],
            duration_seconds=40.0, fps=24)
        plan = _compile(tl)
        self.assertGreater(len(plan.windows), 1, "needs a seam to be a test")
        return plan

    def test_the_same_character_carries_the_same_id_in_every_window(self):
        for window in self._plan().windows:
            for line in window.prompt.splitlines():
                if "(S1)" in line:
                    self.assertIn("<Subject 1>", line)
                if "(S2)" in line:
                    self.assertIn("<Subject 2>", line)

    def test_a_window_that_opens_on_the_second_speaker_still_calls_them_s2(self):
        """The window's own first speaker is not S1. Numbering per window would
        make it S1, which is the whole failure."""
        second = self._plan().windows[1]
        self.assertIn("<Subject 2> (S2)", second.prompt)
        self.assertNotIn("<Subject 2> (S1)", second.prompt)


class TestWhereTheIdIsStamped(unittest.TestCase):
    def test_a_speaking_character_is_stamped(self):
        window = _compile(_two_hander()).windows[0]
        self.assertIn("<Subject 1> (S1) crosses to the counter", window.prompt)

    def test_a_character_standing_silently_in_another_shot_is_not(self):
        """Stamping every mention would put "(S1)" on someone standing in the
        background, which reads to the model as a cue to give them a line."""
        tl = _two_hander()
        tl.shots[1].prompt = "@Kade watches @Mimi from the counter."
        tl.shots[1].speakers = ["kade"]
        window = _compile(tl).windows[0]
        shot_two = _line(window.prompt, "watches")
        self.assertIn("<Subject 2> (S2)", shot_two)
        self.assertIn("<Subject 1>", shot_two)
        self.assertNotIn("<Subject 1> (S1)", shot_two)


class TestTheAudioIsBoundToSomeone(unittest.TestCase):
    def test_lip_sync_names_the_subject_it_belongs_to(self):
        prompt = _compile(_two_hander()).windows[0].prompt
        self.assertIn(
            "`<Audio 1>` is the speech <Subject 1> (S1) is saying. Their lip movements "
            "match `<Audio 1>` precisely, in time with it.", prompt)

    def test_timbre_names_the_subject_it_belongs_to(self):
        prompt = _compile(_two_hander()).windows[0].prompt
        self.assertIn("`<Audio 2>` is the voice-timbre reference for <Subject 2> (S2).",
                      prompt)

    def test_two_voices_are_bound_to_two_different_people(self):
        """The one thing that distinguishes this from the unbound prompt: on a
        two-hander each <Audio j> says whose it is, and they do not say the same."""
        prompt = _compile(_two_hander()).windows[0].prompt
        first = _line(prompt, "`<Audio 1>` is")
        second = _line(prompt, "`<Audio 2>` is")
        self.assertIn("(S1)", first)
        self.assertIn("(S2)", second)

    def test_bin_audio_is_left_unbound(self):
        """Shared across the whole film, so there is no shot to read a speaker
        off. Guessing one would bind the voice to whoever happened to talk first."""
        tl = _two_hander()
        tl.local_refs.clear()
        tl.assets.add(Asset("score", KIND_AUDIO, name="Take", file="t.wav",
                            audio_role=AUDIO_ROLE_LIP_SYNC))
        prompt = _compile(tl).windows[0].prompt
        self.assertIn("is the speech this character is saying", prompt)


class TestTheUnboundPromptIsUnchanged(unittest.TestCase):
    """Naming nobody must produce exactly the prompt this pack produced before
    speakers existed -- otherwise adding the field invalidates the cache of every
    project that does not use it."""

    def setUp(self):
        self.prompt = _compile(_two_hander(speakers=False)).windows[0].prompt

    def test_no_speaker_id_appears_anywhere(self):
        self.assertNotIn("(S1)", self.prompt)
        self.assertNotIn("(S2)", self.prompt)

    def test_lip_sync_keeps_the_anonymous_sentence(self):
        self.assertIn(
            "`<Audio 1>` is the speech this character is saying. Their lip movements "
            "match `<Audio 1>` precisely, in time with it.", self.prompt)

    def test_timbre_keeps_the_anonymous_sentence(self):
        self.assertIn("`<Audio 2>` is a voice-timbre reference.", self.prompt)

    def test_no_audio_retention_line_is_added(self):
        self.assertNotIn("the voice of", self.prompt)


class TestTheAudioRetentionLine(unittest.TestCase):
    """MiniMax's retention vocabulary for audio is not the picture word list, and
    the widget's two values map onto it rather than being stored in it."""

    def test_lip_sync_is_fully_copy(self):
        prompt = _compile(_two_hander()).windows[0].prompt
        self.assertIn("`<Audio 1>` (the voice of <Subject 1> (S1)): fully_copy", prompt)

    def test_timbre_is_reference(self):
        prompt = _compile(_two_hander()).windows[0].prompt
        self.assertIn("`<Audio 2>` (the voice of <Subject 2> (S2)): reference", prompt)

    def test_every_visual_retention_line_still_precedes_every_audio_one(self):
        """The order MiniMax's worked example uses, whatever order the sockets
        happened to fill."""
        prompt = _compile(_two_hander()).windows[0].prompt
        section = prompt.split("retention_analysis:\n", 1)[1].split("\n\n", 1)[0]
        lines = section.splitlines()
        audio = [i for i, line in enumerate(lines) if line.startswith("`<Audio")]
        visual = [i for i, line in enumerate(lines) if line.startswith("<Subject")]
        self.assertTrue(audio and visual)
        self.assertGreater(min(audio), max(visual))


class TestASpeakerWithNoReference(unittest.TestCase):
    """Two different failures, refused at two different layers.

    A speaker naming an asset that is not in the bin at all is a structural
    problem: `Timeline.validate` has refused it since the field existed, and the
    node layer never produces one -- `_bind_speakers` reports an unresolvable
    name rather than writing it through.

    A speaker whose asset *is* in the bin but carries no tag in this particular
    window is different and survivable: the reference was pushed out of the
    budget to make room for carry-over, and "(S1)" pointing at an absent picture
    is worse than no binding at all. That one is diagnosed and dropped.
    """

    def test_an_asset_that_is_not_in_the_bin_is_refused_outright(self):
        tl = _two_hander()
        tl.shots[0].speakers = ["nobody"]
        plan = _compile(tl)
        self.assertFalse(plan.ok)
        self.assertTrue([p for p in plan.problems if "names speaker" in p])

    def test_a_speaker_dropped_from_this_window_loses_the_id_rather_than_faking_one(self):
        from comfyui_pulse_studio.assets import AssetBin
        from comfyui_pulse_studio.compiler import _speaker_tags

        bin_ = AssetBin([Asset("mimi", KIND_IMAGE, name="Mimi", file="a.png",
                               description="a woman")])
        tag_map = bin_.tag_map()
        self.assertEqual(_speaker_tags(tag_map, {}, {"mimi": "S1"}),
                         {"mimi": "<Picture 1> (S1)"})
        self.assertEqual(_speaker_tags(tag_map, {}, {"gone": "S1"}), {})

    def test_an_undescribed_character_still_gets_bound_to_their_picture(self):
        """Much weaker than a <Subject N> definition, and still a face for the
        voice to belong to, which is the whole job."""
        tl = _two_hander()
        tl.assets.get("mimi").description = ""
        prompt = _compile(tl).windows[0].prompt
        self.assertIn("<Picture 1> (S1)", prompt)


class TestTheBindingReachesTheCacheKey(unittest.TestCase):
    """The binding sentence lives in the window's subject definitions, which no
    shot's `resolved_prompt` covers -- the same hole `audio_role` had to be
    plugged for. Without this the cache would hand back a segment rendered with
    the voice bound to somebody else."""

    def test_two_bindings_key_differently(self):
        self.assertNotEqual(
            _shot_key(shot_block("s1", 0, speaker_binding="<Subject 1> (S1)")),
            _shot_key(shot_block("s1", 0, speaker_binding="<Subject 2> (S2)")))

    def test_the_id_moving_under_a_stable_subject_still_keys_differently(self):
        """An earlier shot gaining a character renumbers everyone downstream. The
        resolved string is hashed for exactly this reason -- the asset id behind
        it would not have moved."""
        self.assertNotEqual(
            _shot_key(shot_block("s1", 0, speaker_binding="<Subject 2> (S1)")),
            _shot_key(shot_block("s1", 0, speaker_binding="<Subject 2> (S2)")))

    def test_a_shot_naming_nobody_keys_exactly_as_before(self):
        self.assertEqual(
            _shot_key(shot_block("s1", 0, label="L", visual="V", audio_line="A",
                                 duration_seconds=5.0, continuity="inherit",
                                 resolved_prompt="R")),
            ["s1", "L", "V", "A", 5.0, "inherit", "R"])

    def test_the_field_is_absent_from_the_block_when_unset(self):
        self.assertNotIn("speaker_binding", shot_block("s1", 0))


class TestTheReportShowsTheBinding(unittest.TestCase):
    """A bound voice is invisible in the prompt to anyone not reading the prompt
    -- an unbound one just says "this character" -- so the ordinal map, which is
    where an author checks that what they typed reached the model as what they
    meant, shows it too."""

    def _map(self, **shot_kwargs):
        from comfyui_pulse_studio.report import _ordinal_map
        return "\n".join(_ordinal_map(
            {"shots": [shot_block("s1", 0, label="One", **shot_kwargs)]}))

    def test_a_bound_shot_names_its_speaker(self):
        out = self._map(speaker_binding="<Subject 1> (S1)")
        self.assertIn("shot One:", out)
        self.assertIn("<Subject 1> (S1)", out)

    def test_a_shot_with_neither_refs_nor_a_speaker_gets_no_block(self):
        self.assertEqual(self._map(), "    (no references)")


class TestTheNameIsResolvedAgainstTheRightScope(unittest.TestCase):
    """`_bind_speakers` searches the shot's own sockets before the bin, so two
    shots may each name their own `@Ref1` and mean different people (§10).

    Lifted out of nodes.py by ast rather than imported -- nodes.py imports torch.
    Same technique as tests/test_audio_role.py.
    """

    @staticmethod
    def _helper():
        source = Path(__file__).parent.parent / "nodes.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        wanted = {"_bind_speakers", "_shot_list"}
        module = ast.Module(
            body=[n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name in wanted],
            type_ignores=[])
        namespace = {}
        exec(compile(module, "nodes.py", "exec"), namespace)
        return namespace["_bind_speakers"]

    def _bind(self, name, local=()):
        tl = Timeline(assets=[dict(a) for a in CAST],
                      shots=[{"id": "s1", "start": 0.0, "duration": 5.0, "prompt": "x"}],
                      duration_seconds=5.0, fps=24)
        tl.local_refs["s1"] = list(local)
        shots = list(tl.ordered_shots())
        notes = self._helper()(tl, shots, [{"shot_id": "s1", "label": "One",
                                            "speaker": name}])
        return shots[0].speakers, notes

    def test_a_bin_name_resolves(self):
        self.assertEqual(self._bind("@Mimi")[0], ["mimi"])

    def test_the_at_sign_is_optional(self):
        self.assertEqual(self._bind("Mimi")[0], ["mimi"])

    def test_matching_is_case_insensitive(self):
        self.assertEqual(self._bind("@mimi")[0], ["mimi"])

    def test_a_scene_local_reference_wins_over_a_bin_asset_of_the_same_name(self):
        local = [Asset("local_mimi", KIND_IMAGE, name="Mimi", file="")]
        self.assertEqual(self._bind("@Mimi", local)[0], ["local_mimi"])

    def test_a_blank_speaker_binds_nothing_and_says_nothing(self):
        self.assertEqual(self._bind(""), ([], []))

    def test_an_unknown_name_is_reported_not_guessed(self):
        speakers, notes = self._bind("@Nobody")
        self.assertEqual(speakers, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("One", notes[0])


if __name__ == "__main__":
    unittest.main()
