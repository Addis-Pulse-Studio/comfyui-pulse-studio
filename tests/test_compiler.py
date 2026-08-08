"""The compiler: prompt assembly, reference resolution, timestamps, windows."""

import unittest

from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO, Asset
from comfyui_pulse_studio.compiler import (
    CARRY_AUDIO_ID,
    CARRY_IMAGE_ID,
    CarryPolicy,
    compile_timeline,
    format_timestamp,
    resolve_references,
    wrap_dialogue,
)
from comfyui_pulse_studio.constants import BRANCH_FL2VA, BRANCH_REF2VA, MAX_WINDOW_FRAMES
from comfyui_pulse_studio.frames import is_on_grid
from comfyui_pulse_studio.timeline import Shot, Timeline


def make_timeline(**kw):
    base = dict(
        assets=[
            {"id": "mimi", "kind": KIND_IMAGE, "name": "Mimi", "file": "mimi.png",
             "description": "the young woman with long dark hair"},
            {"id": "kaleb", "kind": KIND_IMAGE, "name": "Kaleb", "file": "kaleb.png",
             "description": "the man with the grey beard"},
        ],
        shots=[
            {"id": "s1", "start": 0.0, "duration": 4.0, "prompt": "@Mimi walks into the cafe."},
            {"id": "s2", "start": 4.0, "duration": 4.0, "prompt": "@Kaleb looks up and says \"you came\""},
        ],
        duration_seconds=8.0,
    )
    base.update(kw)
    return Timeline.from_dict(base)


class TestTimestampFormat(unittest.TestCase):
    def test_format(self):
        self.assertEqual(format_timestamp(0), "00:00.000")
        self.assertEqual(format_timestamp(4.5), "00:04.500")
        self.assertEqual(format_timestamp(65.25), "01:05.250")
        self.assertEqual(format_timestamp(600), "10:00.000")

    def test_negative_clamps_to_zero(self):
        self.assertEqual(format_timestamp(-3), "00:00.000")

    def test_minutes_are_not_capped_at_sixty(self):
        self.assertEqual(format_timestamp(3600), "60:00.000")

    def test_zero_padding_is_stable_for_sorting(self):
        self.assertEqual(len(format_timestamp(4.5)), len(format_timestamp(59.999)))


class TestTimestampMonotonicity(unittest.TestCase):
    def _stamps(self, prompt_text):
        import re
        return [m.group(1) for m in re.finditer(r"\[Shot \d+\] At (\d\d:\d\d\.\d\d\d)", prompt_text)]

    def test_first_shot_is_unstamped(self):
        plan = compile_timeline(make_timeline())
        self.assertIn("[Shot 1] ", plan.windows[0].prompt)
        self.assertNotIn("[Shot 1] At ", plan.windows[0].prompt)

    def test_stamps_strictly_increase(self):
        tl = make_timeline(shots=[
            {"id": "a", "start": 0.0, "duration": 2.0, "prompt": "one"},
            {"id": "b", "start": 2.0, "duration": 2.0, "prompt": "two"},
            {"id": "c", "start": 4.0, "duration": 2.0, "prompt": "three"},
        ], duration_seconds=6.0)
        stamps = self._stamps(compile_timeline(tl).windows[0].prompt)
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(len(stamps), len(set(stamps)))

    def test_out_of_order_input_is_sorted_by_clock(self):
        tl = make_timeline(shots=[
            {"id": "c", "start": 4.0, "duration": 2.0, "prompt": "third"},
            {"id": "a", "start": 0.0, "duration": 2.0, "prompt": "first"},
            {"id": "b", "start": 2.0, "duration": 2.0, "prompt": "second"},
        ], duration_seconds=6.0)
        prompt = compile_timeline(tl).windows[0].prompt
        self.assertLess(prompt.index("first"), prompt.index("second"))
        self.assertLess(prompt.index("second"), prompt.index("third"))

    def test_colliding_stamps_are_nudged_not_duplicated(self):
        """Two shots can round onto the same millisecond; the format requires
        strictly increasing stamps, so the collision is nudged and reported."""
        tl = make_timeline(shots=[
            {"id": "a", "start": 0.0, "duration": 1.0, "prompt": "one"},
            {"id": "b", "start": 2.0, "duration": 1.0, "prompt": "two"},
            {"id": "c", "start": 2.0000001, "duration": 1.0, "prompt": "three"},
        ], duration_seconds=4.0)
        window = compile_timeline(tl).windows[0]
        stamps = self._stamps(window.prompt)
        self.assertEqual(len(stamps), len(set(stamps)), "duplicate timestamp emitted")
        self.assertTrue(any("strictly increasing" in d for d in window.diagnostics))

    def test_duplicate_starts_are_reported_as_a_problem(self):
        tl = make_timeline(shots=[
            {"id": "a", "start": 1.0, "duration": 1.0, "prompt": "one"},
            {"id": "b", "start": 1.0, "duration": 1.0, "prompt": "two"},
        ], duration_seconds=4.0)
        self.assertTrue(any("strictly increasing" in p for p in tl.validate()))

    def test_stamps_are_relative_to_their_own_window(self):
        """A shot at t=20s in window 2 stamps as its offset into that window,
        not as its absolute position on the project clock."""
        tl = make_timeline(shots=[
            {"id": "a", "start": 0.0, "duration": 10.0, "prompt": "early"},
            {"id": "b", "start": 12.0, "duration": 4.0, "prompt": "late"},
        ], duration_seconds=20.0, window_seconds=10.0)
        plan = compile_timeline(tl)
        self.assertGreater(len(plan.windows), 1)
        w2 = plan.windows[1]
        stamps = self._stamps(w2.prompt)
        for s in stamps:
            self.assertLess(s, format_timestamp(w2.duration_seconds + 0.5))


class TestReferenceResolution(unittest.TestCase):
    def test_at_name_resolves(self):
        plan = compile_timeline(make_timeline())
        prompt = plan.windows[0].prompt
        self.assertNotIn("@Mimi", prompt)
        self.assertIn("<Subject 1>", prompt)

    def test_braced_id_resolves(self):
        tl = make_timeline(shots=[
            {"id": "s1", "start": 0, "duration": 4, "prompt": "{{mimi}} enters."}])
        prompt = compile_timeline(tl).windows[0].prompt
        self.assertNotIn("{{mimi}}", prompt)
        self.assertIn("<Subject 1>", prompt)

    def test_bracketed_name_with_spaces(self):
        tl = make_timeline(
            assets=[{"id": "cafe", "kind": KIND_IMAGE, "name": "Cafe wide", "file": "c.png"}],
            shots=[{"id": "s1", "start": 0, "duration": 4, "prompt": "wide on @[Cafe wide]."}])
        prompt = compile_timeline(tl).windows[0].prompt
        self.assertIn("<Picture 1>", prompt)

    def test_described_image_resolves_to_its_subject_not_its_picture(self):
        """MiniMax's format wants prose to talk about subjects, citing the
        picture once inside the definition."""
        tl = make_timeline(shots=[
            {"id": "s1", "start": 0, "duration": 4, "prompt": "@Mimi smiles."}])
        w = compile_timeline(tl).windows[0]
        desc = w.prompt.split("detailed_description:")[1]
        self.assertIn("<Subject 1>", desc)
        self.assertNotIn("<Picture 1>", desc)
        self.assertIn("<Subject 1> is the young woman", w.prompt)

    def test_undescribed_image_resolves_to_its_raw_picture_tag(self):
        tl = make_timeline(
            assets=[{"id": "bg", "kind": KIND_IMAGE, "name": "Backdrop", "file": "b.png"}],
            shots=[{"id": "s1", "start": 0, "duration": 4, "prompt": "set in @Backdrop."}])
        self.assertIn("<Picture 1>", compile_timeline(tl).windows[0].prompt)

    def test_unresolved_reference_is_reported_and_left_alone(self):
        tl = make_timeline(shots=[
            {"id": "s1", "start": 0, "duration": 4, "prompt": "@Nobody waves."}])
        w = compile_timeline(tl).windows[0]
        self.assertTrue(any("unresolved reference" in d for d in w.diagnostics))
        self.assertIn("@Nobody", w.prompt)

    def test_hand_typed_ordinal_is_flagged(self):
        """The hazard the whole design removes: a typed tag will not track the bin."""
        tl = make_timeline(shots=[
            {"id": "s1", "start": 0, "duration": 4, "prompt": "<Picture 2> walks in."}])
        w = compile_timeline(tl).windows[0]
        self.assertTrue(any("hand-typed tag" in d for d in w.diagnostics))

    def test_resolution_tracks_reorder(self):
        """The core property: reordering the bin changes the emitted ordinal
        without the prompt text changing at all."""
        tl = make_timeline(
            assets=[
                {"id": "a", "kind": KIND_IMAGE, "name": "Alpha", "file": "a.png"},
                {"id": "b", "kind": KIND_IMAGE, "name": "Beta", "file": "b.png"},
            ],
            shots=[{"id": "s1", "start": 0, "duration": 4, "prompt": "@Beta arrives."}])
        self.assertIn("<Picture 2>", compile_timeline(tl).windows[0].prompt)
        tl.assets.move("b", 0)
        self.assertIn("<Picture 1>", compile_timeline(tl).windows[0].prompt)

    def test_resolution_tracks_removal(self):
        tl = make_timeline(
            assets=[
                {"id": "a", "kind": KIND_IMAGE, "name": "Alpha", "file": "a.png"},
                {"id": "b", "kind": KIND_IMAGE, "name": "Beta", "file": "b.png"},
            ],
            shots=[{"id": "s1", "start": 0, "duration": 4, "prompt": "@Beta arrives."}])
        tl.assets.remove("a")
        self.assertIn("<Picture 1>", compile_timeline(tl).windows[0].prompt)

    def test_resolve_references_is_pure(self):
        from comfyui_pulse_studio.assets import AssetBin
        bin_ = AssetBin([Asset("x", KIND_IMAGE, name="Ex")])
        text, diags = resolve_references("@Ex here", bin_.tag_map(), bin_)
        self.assertEqual(text, "<Picture 1> here")
        self.assertEqual(diags, [])


class TestDialogue(unittest.TestCase):
    def test_quotes_are_wrapped(self):
        self.assertEqual(wrap_dialogue('he says "hello"', "English"),
                         'he says <d>[English] hello.</d>')

    def test_language_is_honoured(self):
        self.assertIn("[Amharic]", wrap_dialogue('"selam"', "Amharic"))

    def test_trailing_comma_is_the_narrations_not_the_lines(self):
        self.assertEqual(wrap_dialogue('"this must be it," he turns', "English"),
                         '<d>[English] this must be it.</d> he turns')

    def test_existing_terminator_is_kept(self):
        self.assertIn("really?", wrap_dialogue('"really?"', "English"))

    def test_unpaired_quote_is_left_alone(self):
        self.assertEqual(wrap_dialogue('he said "hello', "English"), 'he said "hello')

    def test_tags_outside_quotes_survive(self):
        out = wrap_dialogue('<Subject 1> says "hi"', "English")
        self.assertIn("<Subject 1>", out)

    def test_decorative_punctuation_stripped(self):
        self.assertIn("wow.", wrap_dialogue('"wow!!!"', "English").replace("!", "."))


class TestWindowing(unittest.TestCase):
    def test_short_project_is_one_window(self):
        plan = compile_timeline(make_timeline())
        self.assertEqual(len(plan.windows), 1)
        self.assertTrue(is_on_grid(plan.windows[0].frame_count))

    def test_long_project_splits(self):
        tl = make_timeline(duration_seconds=40.0)
        plan = compile_timeline(tl)
        self.assertGreater(len(plan.windows), 1)
        for w in plan.windows:
            self.assertTrue(is_on_grid(w.frame_count))
            self.assertLessEqual(w.frame_count, MAX_WINDOW_FRAMES)

    def test_every_window_frame_count_is_legal(self):
        for seconds in (1, 5, 15, 16, 30, 45, 90, 300):
            plan = compile_timeline(make_timeline(duration_seconds=float(seconds)))
            for w in plan.windows:
                self.assertTrue(is_on_grid(w.frame_count),
                                "%ss -> %d frames off-grid" % (seconds, w.frame_count))

    def test_quantisation_is_reported(self):
        plan = compile_timeline(make_timeline(duration_seconds=7.3))
        self.assertTrue(any("snapped to the frame grid" in d for d in plan.diagnostics))

    def test_shot_spanning_a_boundary_appears_in_both_windows(self):
        """Dropping it from the second would leave that window with no direction,
        which renders as a stall rather than as continued action."""
        tl = make_timeline(
            shots=[{"id": "long", "start": 0.0, "duration": 20.0, "prompt": "@Mimi keeps walking."}],
            duration_seconds=20.0, window_seconds=10.0)
        plan = compile_timeline(tl)
        self.assertGreater(len(plan.windows), 1)
        for w in plan.windows:
            self.assertIn("long", w.shot_ids)

    def test_window_bounds_are_contiguous(self):
        plan = compile_timeline(make_timeline(duration_seconds=45.0))
        for a, b in zip(plan.windows, plan.windows[1:]):
            self.assertAlmostEqual(a.end_seconds, b.start_seconds, places=9)

    def test_seed_offset_differs_per_window(self):
        plan = compile_timeline(make_timeline(duration_seconds=45.0))
        offsets = [w.seed_offset for w in plan.windows]
        self.assertEqual(len(offsets), len(set(offsets)))

    def test_total_frames_never_short_of_request(self):
        from comfyui_pulse_studio.frames import seconds_to_frames
        for seconds in (3.0, 12.0, 16.0, 33.3):
            plan = compile_timeline(make_timeline(duration_seconds=seconds))
            self.assertGreaterEqual(plan.total_frames, seconds_to_frames(seconds))


class TestCarryOver(unittest.TestCase):
    def test_continuation_window_carries_a_frame_as_picture_one(self):
        tl = make_timeline(duration_seconds=30.0, window_seconds=10.0)
        plan = compile_timeline(tl, carry=CarryPolicy(mode="image", audio=True))
        w2 = plan.windows[1]
        self.assertEqual(w2.tag_map.tag(CARRY_IMAGE_ID), "<Picture 1>")

    def test_carry_over_shifts_user_pictures(self):
        """The renumbering that must not desynchronise: on window 2 the carried
        frame takes <Picture 1>, so Mimi becomes <Picture 2> -- and the prompt
        text, which never named a number, follows automatically."""
        tl = make_timeline(
            assets=[{"id": "bg", "kind": KIND_IMAGE, "name": "Backdrop", "file": "b.png"}],
            shots=[{"id": "s1", "start": 0, "duration": 30, "prompt": "wide on @Backdrop."}],
            duration_seconds=30.0, window_seconds=10.0)
        plan = compile_timeline(tl, carry=CarryPolicy(mode="image"))
        self.assertIn("<Picture 1>", plan.windows[0].prompt)
        self.assertIn("<Picture 2>", plan.windows[1].prompt)
        self.assertNotIn("<Picture 1>", plan.windows[1].prompt.split("detailed_description:")[1])

    def test_first_window_has_no_carry(self):
        tl = make_timeline(duration_seconds=30.0, window_seconds=10.0)
        plan = compile_timeline(tl)
        self.assertNotIn(CARRY_IMAGE_ID, plan.windows[0].tag_map.by_id)
        self.assertNotIn(CARRY_AUDIO_ID, plan.windows[0].tag_map.by_id)

    def test_audio_carry_claims_an_audio_ordinal(self):
        tl = make_timeline(duration_seconds=30.0, window_seconds=10.0)
        plan = compile_timeline(tl, carry=CarryPolicy(mode="image", audio=True))
        self.assertEqual(plan.windows[1].tag_map.tag(CARRY_AUDIO_ID), "<Audio 1>")

    def test_audio_carry_shifts_user_audio_tags(self):
        tl = make_timeline(
            assets=[{"id": "vo", "kind": KIND_AUDIO, "name": "VO", "file": "vo.wav"}],
            shots=[{"id": "s1", "start": 0, "duration": 30, "prompt": "voice like @VO."}],
            duration_seconds=30.0, window_seconds=10.0)
        plan = compile_timeline(tl, carry=CarryPolicy(audio=True))
        self.assertIn("<Audio 1>", plan.windows[0].prompt)
        w2_desc = plan.windows[1].prompt.split("detailed_description:")[1]
        self.assertIn("<Audio 2>", w2_desc)

    def test_carry_none_adds_nothing(self):
        tl = make_timeline(duration_seconds=30.0, window_seconds=10.0)
        plan = compile_timeline(tl, carry=CarryPolicy(mode="none", audio=False))
        self.assertNotIn(CARRY_IMAGE_ID, plan.windows[1].tag_map.by_id)

    def test_carry_video_uses_a_video_socket(self):
        tl = make_timeline(duration_seconds=30.0, window_seconds=10.0)
        plan = compile_timeline(tl, carry=CarryPolicy(mode="video"))
        sockets = plan.windows[1].tag_map.sockets
        self.assertEqual(sockets.get("ref_video_0"), "__carry_clip__")

    def test_carry_over_never_inflates_the_file_meter(self):
        tl = make_timeline(duration_seconds=30.0, window_seconds=10.0)
        plan = compile_timeline(tl, carry=CarryPolicy(mode="both", audio=True))
        real = [f for f in plan.windows[1].files if not f.synthetic]
        self.assertEqual(len(real), 2)  # mimi + kaleb

    def test_carry_over_evicts_the_lowest_priority_user_asset_loudly(self):
        """Only three video sockets exist. If carry-over takes one, a user's
        third video cannot ride -- and the user must be told, not surprised."""
        tl = make_timeline(
            assets=[{"id": "v%d" % i, "kind": KIND_VIDEO, "name": "V%d" % i,
                     "file": "v%d.mp4" % i, "trim_start": 0.0, "trim_end": 5.0}
                    for i in range(3)],
            shots=[{"id": "s1", "start": 0, "duration": 30, "prompt": "action"}],
            duration_seconds=30.0, window_seconds=10.0)
        plan = compile_timeline(tl, carry=CarryPolicy(mode="video"))
        self.assertTrue(any("dropped from window 2" in d for d in plan.windows[1].diagnostics))
        video_sockets = [s for s in plan.windows[1].tag_map.sockets if s.startswith("ref_video_")]
        self.assertEqual(len(video_sockets), 3)
        # The carried clip holds slot 0; only two of the three user videos ride.
        self.assertEqual(plan.windows[1].tag_map.sockets["ref_video_0"], "__carry_clip__")

    def test_reference_audio_is_never_described_as_sync(self):
        """ref_audios is reference conditioning. Nothing in the prompt may
        promise lip-sync or beat-matching, because the model cannot deliver it."""
        tl = make_timeline(
            assets=[{"id": "vo", "kind": KIND_AUDIO, "name": "VO", "file": "vo.wav"}],
            shots=[{"id": "s1", "start": 0, "duration": 8, "prompt": "she speaks"}],
            duration_seconds=8.0)
        prompt = compile_timeline(tl).windows[0].prompt.lower()
        for forbidden in ("lip-sync", "lip sync", "beat-match", "beat match", "in sync with"):
            self.assertNotIn(forbidden, prompt)


class TestBranches(unittest.TestCase):
    def test_ref2va_emits_six_sections(self):
        prompt = compile_timeline(make_timeline()).windows[0].prompt
        for section in ("subject_definitions:", "summary:", "retention_analysis:",
                        "detailed_description:", "overall_soundscape:", "non_diegetic_music:"):
            self.assertIn(section, prompt)

    def test_fl2va_emits_three_sections_and_no_subjects(self):
        tl = make_timeline(assets=[], branch=BRANCH_FL2VA)
        prompt = compile_timeline(tl).windows[0].prompt
        self.assertIn("integrated_multimodal_description:", prompt)
        self.assertIn("overall_soundscape:", prompt)
        self.assertNotIn("subject_definitions:", prompt)
        self.assertNotIn("retention_analysis:", prompt)

    def test_fl2va_carries_no_references(self):
        tl = make_timeline(assets=[], branch=BRANCH_FL2VA)
        self.assertEqual(compile_timeline(tl).windows[0].files, [])

    def test_references_with_fl2va_is_a_validation_problem(self):
        """Different checkpoints, disjoint inputs -- mutually exclusive per render."""
        tl = make_timeline(branch=BRANCH_FL2VA)
        self.assertTrue(any("takes no references" in p for p in tl.validate()))

    def test_anchors_with_ref2va_is_a_validation_problem(self):
        tl = make_timeline(branch=BRANCH_REF2VA, first_frame="mimi")
        self.assertTrue(any("no anchor inputs" in p for p in tl.validate()))

    def test_anchor_indices_are_only_ever_zero_or_last(self):
        tl = make_timeline(assets=[
            {"id": "f", "kind": KIND_IMAGE, "name": "F", "file": "f.png"},
            {"id": "l", "kind": KIND_IMAGE, "name": "L", "file": "l.png"}],
            branch=BRANCH_FL2VA, first_frame="f", last_frame="l")
        w = compile_timeline(tl).windows[0]
        self.assertEqual(w.anchors["first_frame_index"], 0)
        self.assertEqual(w.anchors["last_frame_index"], w.frame_count - 1)

    def test_last_anchor_only_on_the_final_window(self):
        """A last_frame pinned on every window would lock each seam to the same
        image; it belongs at the end of the sequence only."""
        tl = make_timeline(assets=[
            {"id": "l", "kind": KIND_IMAGE, "name": "L", "file": "l.png"}],
            branch=BRANCH_FL2VA, last_frame="l", duration_seconds=40.0)
        plan = compile_timeline(tl)
        self.assertGreater(len(plan.windows), 1)
        for w in plan.windows[:-1]:
            self.assertNotIn("last_frame", w.anchors)
        self.assertIn("last_frame", plan.windows[-1].anchors)

    def test_fl2va_continuation_anchors_the_carried_frame(self):
        tl = make_timeline(assets=[], branch=BRANCH_FL2VA, duration_seconds=40.0)
        plan = compile_timeline(tl)
        self.assertEqual(plan.windows[1].anchors["first_frame"], CARRY_IMAGE_ID)
        self.assertEqual(plan.windows[1].anchors["first_frame_index"], 0)

    def test_mixed_branch_continuation(self):
        """ref2va first window for identity, fl2va afterwards for a hard seam lock."""
        tl = make_timeline(branch=BRANCH_REF2VA, continuation_branch=BRANCH_FL2VA,
                           duration_seconds=40.0)
        plan = compile_timeline(tl)
        self.assertEqual(plan.windows[0].branch, BRANCH_REF2VA)
        self.assertEqual(plan.windows[1].branch, BRANCH_FL2VA)
        self.assertTrue(plan.windows[0].files)
        self.assertEqual(plan.windows[1].files, [])


class TestFileList(unittest.TestCase):
    def test_files_are_ordered_as_the_tokenizer_walks_them(self):
        tl = make_timeline(assets=[
            {"id": "i1", "kind": KIND_IMAGE, "name": "I1", "file": "i1.png"},
            {"id": "v1", "kind": KIND_VIDEO, "name": "V1", "file": "v1.mp4",
             "trim_start": 0.0, "trim_end": 5.0, "include_audio": True},
            {"id": "a1", "kind": KIND_AUDIO, "name": "A1", "file": "a1.wav"},
        ])
        sockets = [f.socket for f in compile_timeline(tl).windows[0].files]
        self.assertEqual(sockets, ["ref_image_0", "ref_video_audio_0", "ref_video_0", "ref_audio_0"])

    def test_socket_kwargs_group_correctly(self):
        tl = make_timeline(assets=[
            {"id": "i1", "kind": KIND_IMAGE, "name": "I1", "file": "i1.png"},
            {"id": "v1", "kind": KIND_VIDEO, "name": "V1", "file": "v1.mp4",
             "trim_start": 0.0, "trim_end": 5.0, "include_audio": True},
        ])
        groups = compile_timeline(tl).windows[0].socket_kwargs()
        self.assertEqual(list(groups["ref_images"]), ["ref_image_0"])
        self.assertEqual(list(groups["ref_videos"]), ["ref_video_0"])
        self.assertEqual(list(groups["ref_video_audios"]), ["ref_video_audio_0"])
        self.assertEqual(groups["ref_audios"], {})

    def test_every_file_carries_its_tag(self):
        plan = compile_timeline(make_timeline())
        for f in plan.windows[0].files:
            self.assertTrue(f.tag and f.tag.startswith("<"))

    def test_file_paths_survive_to_the_plan(self):
        plan = compile_timeline(make_timeline())
        self.assertEqual({f.file for f in plan.windows[0].files}, {"mimi.png", "kaleb.png"})


class TestPresence(unittest.TestCase):
    def test_subject_in_every_shot_is_present_throughout(self):
        tl = make_timeline(shots=[
            {"id": "a", "start": 0, "duration": 2, "prompt": "@Mimi walks"},
            {"id": "b", "start": 2, "duration": 2, "prompt": "@Mimi sits"},
        ], assets=[{"id": "mimi", "kind": KIND_IMAGE, "name": "Mimi", "file": "m.png",
                    "description": "the young woman"}], duration_seconds=4.0)
        self.assertIn("(present throughout)", compile_timeline(tl).windows[0].prompt)

    def test_late_entrance_is_not_claimed_from_the_start(self):
        """A character entering at Shot 2 must not be asserted as present in
        Shot 1 -- the model takes that literally and puts them there."""
        tl = make_timeline(shots=[
            {"id": "a", "start": 0, "duration": 2, "prompt": "empty room"},
            {"id": "b", "start": 2, "duration": 2, "prompt": "@Mimi enters"},
        ], assets=[{"id": "mimi", "kind": KIND_IMAGE, "name": "Mimi", "file": "m.png",
                    "description": "the young woman"}], duration_seconds=4.0)
        prompt = compile_timeline(tl).windows[0].prompt
        self.assertIn("(appears in [Shot 2])", prompt)
        self.assertNotIn("(present throughout)", prompt)


class TestPlanShape(unittest.TestCase):
    def test_accepts_dict_and_json(self):
        tl = make_timeline()
        self.assertTrue(compile_timeline(tl.to_dict()).windows)
        self.assertTrue(compile_timeline(tl.to_json()).windows)

    def test_roundtrip_json(self):
        tl = make_timeline()
        restored = Timeline.from_json(tl.to_json())
        self.assertEqual(compile_timeline(restored).windows[0].prompt,
                         compile_timeline(tl).windows[0].prompt)

    def test_plan_is_serialisable(self):
        import json
        plan = compile_timeline(make_timeline())
        json.dumps(plan.to_dict())  # must not raise

    def test_preview_covers_every_window(self):
        plan = compile_timeline(make_timeline(duration_seconds=40.0))
        preview = plan.preview()
        for i in range(len(plan.windows)):
            self.assertIn("Window %d/" % (i + 1), preview)

    def test_empty_timeline_does_not_crash(self):
        plan = compile_timeline(Timeline())
        self.assertFalse(plan.ok)
        self.assertTrue(any("duration is zero" in p for p in plan.problems))

    def test_strict_mode_raises_on_a_bad_timeline(self):
        with self.assertRaises(ValueError):
            compile_timeline(make_timeline(branch=BRANCH_FL2VA), strict=True)

    def test_problems_are_collected_not_raised_by_default(self):
        plan = compile_timeline(make_timeline(branch=BRANCH_FL2VA))
        self.assertFalse(plan.ok)
        self.assertTrue(plan.problems)

    def test_shot_with_empty_prompt_is_skipped_without_gap(self):
        tl = make_timeline(shots=[
            {"id": "a", "start": 0, "duration": 2, "prompt": "one"},
            {"id": "b", "start": 2, "duration": 2, "prompt": "   "},
            {"id": "c", "start": 4, "duration": 2, "prompt": "three"},
        ], duration_seconds=6.0)
        prompt = compile_timeline(tl).windows[0].prompt
        self.assertIn("[Shot 1]", prompt)
        self.assertIn("[Shot 2]", prompt)
        self.assertNotIn("[Shot 3]", prompt)


if __name__ == "__main__":
    unittest.main()
