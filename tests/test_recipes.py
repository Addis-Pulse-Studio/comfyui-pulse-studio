"""Sampling recipes: one per checkpoint, chosen per window branch.

A film that opens on FastH3-Omni (8 Euler steps, `simple`) and carries on in
TaoMate (3 steps on a hand-written schedule) cannot be expressed with one set of
sampler widgets. These pin the presets to the numbers each model was released
with, the fallbacks between branches, and the checks that used to surface only
inside the first window's sampler.
"""

import unittest

from comfyui_pulse_studio.constants import BRANCH_FL2VA, BRANCH_REF2VA
from comfyui_pulse_studio.pulse_timeline import build_timeline, global_block, window_block
from comfyui_pulse_studio.recipes import (
    RECIPE_NAMES,
    TAOMATE_3STEP_SIGMAS,
    RecipeError,
    make_recipe,
    parse_sigmas,
    recipe_problems,
    recipe_warnings,
    sampling_for_branch,
)
from comfyui_pulse_studio.segcache import cache_key, cache_key_material, derive_run_id

SLATE = {"steps": 20, "sampler": "res_multistep", "scheduler": "simple", "cfg": 1.0}


class ThePresets(unittest.TestCase):

    def test_fasth3_omni_is_eight_euler_steps_on_simple(self):
        r = make_recipe("fasth3_omni_8step")
        self.assertEqual((r["steps"], r["sampler"], r["scheduler"], r["cfg"], r["sigmas"]),
                         (8, "euler", "simple", 1.0, None))

    def test_taomate_is_its_own_three_interval_schedule(self):
        r = make_recipe("taomate_fl2va_3step")
        self.assertEqual(r["sigmas"], [1.0, 0.961165, 0.853333, 0.0])
        self.assertEqual(r["steps"], 3)
        self.assertEqual(tuple(r["sigmas"]), TAOMATE_3STEP_SIGMAS)

    def test_hyperflow_is_its_scheduler_at_eight(self):
        r = make_recipe("hyperflow_8step")
        self.assertEqual((r["steps"], r["scheduler"], r["sampler"]), (8, "hyperflow", "euler"))

    def test_a_preset_ignores_the_custom_widgets(self):
        self.assertEqual(make_recipe("fasth3_omni_8step", steps=30, sampler="ddim")["steps"], 8)

    def test_custom_reads_the_widgets_and_a_schedule_fixes_the_steps(self):
        r = make_recipe("custom", steps=12, sampler="euler", scheduler="beta", cfg=1.0,
                        sigmas="1.0, 0.97, 0.64, 0.0")
        self.assertEqual(r["sigmas"], [1.0, 0.97, 0.64, 0.0])
        self.assertEqual(r["steps"], 3)
        self.assertEqual(make_recipe("custom", steps=12)["steps"], 12)

    def test_every_preset_is_sound(self):
        for name in RECIPE_NAMES:
            with self.subTest(recipe=name):
                self.assertEqual(recipe_problems(make_recipe(name)), [])


class WhatIsRefused(unittest.TestCase):

    def test_hyperflow_at_anything_but_eight_steps(self):
        bad = dict(SLATE, scheduler="hyperflow", steps=6)
        self.assertTrue(any("must be 8" in p for p in recipe_problems(bad)))

    def test_a_schedule_that_does_not_end_at_zero(self):
        self.assertTrue(recipe_problems(make_recipe("custom", sigmas="1.0 0.8 0.2")))

    def test_a_schedule_that_runs_upward(self):
        self.assertTrue(recipe_problems(make_recipe("custom", sigmas="1.0 0.5 0.7 0.0")))

    def test_a_schedule_with_text_in_it(self):
        with self.assertRaises(RecipeError):
            parse_sigmas("1.0, half, 0.0")


class WhatIsWarned(unittest.TestCase):

    def test_a_pdd_student_on_the_wrong_schedule_and_without_vsa(self):
        notes = recipe_warnings(dict(SLATE, name=None), {"pdd_heads": 32}, BRANCH_REF2VA)
        text = " ".join(notes)
        self.assertIn("32 stacked PDD output heads", text)
        self.assertIn("VSA", text)

    def test_fasth3_on_its_recipe_with_vsa_is_quiet(self):
        traits = {"pdd_heads": 32, "vsa": True, "sparse": True, "curve_form": True}
        self.assertEqual(recipe_warnings(make_recipe("fasth3_omni_8step"), traits,
                                         BRANCH_REF2VA), [])

    def test_the_hyperflow_grid_without_hyperflow(self):
        notes = recipe_warnings(make_recipe("hyperflow_8step"), {}, BRANCH_REF2VA)
        self.assertTrue(any("no HyperFlow adapter" in n for n in notes))

    def test_taomate_without_a_lora_or_on_ref2va(self):
        notes = " ".join(recipe_warnings(make_recipe("taomate_fl2va_3step"), {}, BRANCH_REF2VA))
        self.assertIn("no LoRA", notes)
        self.assertIn("fl2va LoRA", notes)
        self.assertEqual(recipe_warnings(make_recipe("taomate_fl2va_3step"), {"lora": True},
                                         BRANCH_FL2VA), [])


class WhichRecipeAWindowGets(unittest.TestCase):

    def test_no_recipe_is_the_slate_widgets(self):
        chosen = sampling_for_branch(BRANCH_FL2VA, SLATE)
        self.assertEqual((chosen["steps"], chosen["sigmas"]), (20, None))

    def test_fl2va_takes_its_own_then_the_reference_recipe(self):
        fast, tao = make_recipe("fasth3_omni_8step"), make_recipe("taomate_fl2va_3step")
        self.assertEqual(sampling_for_branch(BRANCH_REF2VA, SLATE, fast, tao)["name"],
                         "fasth3_omni_8step")
        self.assertEqual(sampling_for_branch(BRANCH_FL2VA, SLATE, fast, tao)["name"],
                         "taomate_fl2va_3step")
        self.assertEqual(sampling_for_branch(BRANCH_FL2VA, SLATE, fast)["name"],
                         "fasth3_omni_8step")


class TheScheduleReachesTheKey(unittest.TestCase):

    def doc(self, sigmas=None):
        windows = [window_block(0, ["s1"], 124, steps=3, sampler="euler", sigmas=sigmas)]
        return build_timeline(global_block(), [], [], windows,
                              {"images": 0, "videos": 0, "audio": 0, "total": 0})

    def test_a_window_without_one_is_the_dict_it_always_was(self):
        self.assertNotIn("sigmas", self.doc()["windows"][0])

    def test_a_hand_written_schedule_moves_the_key_and_keeps_the_folder(self):
        plain, tao = self.doc(), self.doc(list(TAOMATE_3STEP_SIGMAS))
        self.assertIn(["sigmas", [1.0, 0.961165, 0.853333, 0.0]],
                      cache_key_material(tao, tao["windows"][0], "m", "p"))
        self.assertNotEqual(cache_key(plain, plain["windows"][0], "m", "p"),
                            cache_key(tao, tao["windows"][0], "m", "p"))
        other = self.doc([1.0, 0.97, 0.64, 0.0])
        self.assertNotEqual(cache_key(tao, tao["windows"][0], "m", "p"),
                            cache_key(other, other["windows"][0], "m", "p"))

    def test_extra_material_is_appended_only_when_given(self):
        doc = self.doc()
        window = doc["windows"][0]
        self.assertEqual(cache_key_material(doc, window, "m", "p"),
                         cache_key_material(doc, window, "m", "p", extra=[]))
        self.assertEqual(cache_key_material(doc, window, "m", "p",
                                            extra=[["upstream", "abc"]])[-1],
                         ["upstream", "abc"])

    def test_the_audio_plan_does_not_move_the_run_folder(self):
        doc = self.doc()
        before = derive_run_id(doc)
        doc["windows"][0]["audio_plan"] = [["ref_audio_0", [["Voice", -0.5]]]]
        self.assertEqual(derive_run_id(doc), before)


if __name__ == "__main__":
    unittest.main()
