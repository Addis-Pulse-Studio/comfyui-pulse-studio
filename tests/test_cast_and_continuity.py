"""Characters with two pictures, a seam held by a guide, and what the key can see.

Three additions that change what reaches the model without changing a socket
count, so each is pinned against the compiled plan:

  * a full-body view is the same person seen again -- cited inside one
    <Subject N>, never as a second Subject, which would tell H3 there are two;
  * `ref_guided_carry` keeps every window on the reference checkpoint and pins
    the seam frame as a guide, so the carried frame stops costing a picture slot;
  * the patch fingerprint now sees core's sparse attention, HyperFlow and LoRA
    weight patches -- and a model with none of them hashes exactly as before.
"""

import types
import unittest

from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, Asset
from comfyui_pulse_studio.compiler import (
    CARRY_IMAGE_ID,
    GUIDE_FIRST,
    CarryPolicy,
    compile_timeline,
)
from comfyui_pulse_studio.concat import upscale_size
from comfyui_pulse_studio.constants import AUDIO_ROLE_LIP_SYNC, BRANCH_REF2VA
from comfyui_pulse_studio.fingerprint import (
    PACK_FRAGMENTS,
    describe_patches,
    describe_weight_patches,
    patch_fingerprint,
    sink_conditioning,
)
from comfyui_pulse_studio.pulse_timeline import CONTINUITY_MODES, CONTINUITY_REF_GUIDED
from comfyui_pulse_studio.timeline import Timeline


def cast_film(guided=False, seconds=24.0):
    assets = [
        {"id": "ada", "kind": KIND_IMAGE, "name": "Ada", "file": "a.png",
         "description": "the woman in the red coat", "retention": "fully_preserved"},
        {"id": "ada_body", "kind": KIND_IMAGE, "name": "AdaBody", "file": "a_full.png",
         "view_of": "ada"},
        {"id": "ada_voice", "kind": KIND_AUDIO, "name": "AdaVoice", "file": "a.wav",
         "audio_role": AUDIO_ROLE_LIP_SYNC, "voice_of": "ada"},
    ]
    half = seconds / 2.0
    shots = [{"id": "s1", "start": 0.0, "duration": half, "prompt": "@Ada walks in.",
              "speakers": ["ada"]},
             {"id": "s2", "start": half, "duration": half, "prompt": "@Ada sits down.",
              "speakers": ["ada"]}]
    timeline = Timeline(assets=assets, shots=shots, duration_seconds=seconds, fps=24,
                        window_seconds=12.1)
    timeline.guided_carry = guided
    return compile_timeline(timeline, policy="shot_aligned",
                            carry=CarryPolicy(mode="image", audio=False))


class ACharacterWithTwoPictures(unittest.TestCase):

    def test_both_pictures_are_one_subject(self):
        prompt = cast_film().windows[0].prompt
        self.assertIn("<Subject 1> is the woman in the red coat "
                      "(from `<Picture 1>` and `<Picture 2>`).", prompt)
        self.assertNotIn("<Subject 2>", prompt)

    def test_the_retention_line_covers_both(self):
        prompt = cast_film().windows[0].prompt
        self.assertIn("matches `<Picture 1>` and `<Picture 2>`.", prompt)

    def test_the_voice_is_hers(self):
        prompt = cast_film().windows[0].prompt
        self.assertIn("is the speech <Subject 1> (S1) is saying", prompt)

    def test_the_view_round_trips(self):
        asset = Asset("b", KIND_IMAGE, name="AdaBody", view_of="ada")
        self.assertEqual(Asset.from_dict(asset.to_dict()).view_of, "ada")
        self.assertNotIn("view_of", Asset("a", KIND_IMAGE, name="Ada").to_dict())


class TheSeamHeldByAGuide(unittest.TestCase):

    def test_the_mode_is_appended(self):
        self.assertEqual(CONTINUITY_MODES[-1], CONTINUITY_REF_GUIDED)

    def test_continuation_windows_stay_on_ref2va_with_a_guide(self):
        plan = cast_film(guided=True)
        self.assertEqual(len(plan.windows), 2)
        first, second = plan.windows
        self.assertNotIn(GUIDE_FIRST, first.anchors)
        self.assertEqual(second.branch, BRANCH_REF2VA)
        self.assertEqual(second.anchors.get(GUIDE_FIRST), CARRY_IMAGE_ID)
        self.assertIn("opening on the exact frame the previous window ended on",
                      second.prompt)

    def test_the_carried_frame_no_longer_costs_a_picture_slot(self):
        guided = cast_film(guided=True).windows[1]
        carried = cast_film(guided=False).windows[1]
        self.assertTrue(any(f.asset_id == CARRY_IMAGE_ID for f in carried.files))
        self.assertFalse(any(f.asset_id == CARRY_IMAGE_ID for f in guided.files))
        # so the character keeps the low numbers on every window
        self.assertIn("(from `<Picture 1>` and `<Picture 2>`)", guided.prompt)


class WhatTheFingerprintSees(unittest.TestCase):

    def sparse(self, vsa, keep):
        source = (
            "class SparseAttnPatch:\n"
            "    def __init__(self, vsa, keep):\n"
            "        self.vsa = vsa; self.topk_ratio = keep; self.tau = 1.3\n"
            "        self.sink_conditioning = 'exact_kv_and_rows'; self.installed = set()\n"
            "def make_attention_override(patch, current):\n"
            "    def override(*a):\n"
            "        return patch, current\n"
            "    return override\n")
        module = types.ModuleType("comfy_extras.nodes_sparse_attention")
        exec(source, module.__dict__)
        override = module.make_attention_override(module.SparseAttnPatch(vsa, keep), None)
        return describe_patches(
            {"transformer_options": {"optimized_attention_override": override}})

    def test_a_plain_model_hashes_as_it_always_did(self):
        descriptor = describe_patches({})
        self.assertEqual(sorted(k for k in descriptor if k != "detected"),
                         sorted(name for name, _ in PACK_FRAGMENTS))
        self.assertEqual(patch_fingerprint(descriptor), "375a3ca8575dada6")

    def test_vsa_and_its_keep_ratio_move_the_fingerprint(self):
        vsa10, vsa20, sol = self.sparse(True, 0.1), self.sparse(True, 0.2), self.sparse(False, 0.0)
        self.assertTrue(vsa10["block_sparse"]["vsa"])
        self.assertEqual(len({patch_fingerprint(d) for d in (vsa10, vsa20, sol)}), 3)
        self.assertEqual(sink_conditioning(vsa10), "exact_kv_and_rows")

    def test_hyperflow_is_seen(self):
        descriptor = describe_patches({"hyperflow": {"version": "1.0", "gate": 0.25}})
        self.assertTrue(descriptor["hyperflow"]["present"])

    def test_a_lora_and_its_strength_are_seen(self):
        shape = types.SimpleNamespace(shape=(128, 5376))

        def patches(strength):
            return {"diffusion_model.blocks.0.attn.to_q.weight":
                    [(strength, ("lora", (shape, shape)), 1.0, None, None)]}

        full, soft = describe_weight_patches(patches(1.0)), describe_weight_patches(patches(0.6))
        self.assertEqual(full["count"], 1)
        self.assertNotEqual(full["digest"], soft["digest"])
        self.assertIsNone(describe_weight_patches({}))
        self.assertNotIn("weight_patches", describe_patches({}, weight_patches={}))


class TheDeliverySize(unittest.TestCase):

    def test_fit_keeps_the_aspect(self):
        self.assertEqual(upscale_size(1344, 768, "1080p_fit"), (1890, 1080, 1890, 1080))
        self.assertEqual(upscale_size(768, 1344, "1080p_fit"), (1080, 1890, 1080, 1890))

    def test_exact_covers_then_crops(self):
        self.assertEqual(upscale_size(1344, 768, "1080p_exact")[2:], (1920, 1080))
        scaled_w, scaled_h, _, _ = upscale_size(1344, 768, "1080p_exact")
        self.assertGreaterEqual((scaled_w, scaled_h), (1920, 1080))
        self.assertEqual(upscale_size(768, 1344, "1080p_exact")[2:], (1080, 1920))
        self.assertEqual(upscale_size(1024, 1024, "1080p_exact")[2:], (1080, 1080))

    def test_every_size_is_even(self):
        for mode in ("1080p_fit", "1080p_exact"):
            for w, h in ((1344, 768), (768, 1344), (1376, 768), (992, 416)):
                self.assertTrue(all(v % 2 == 0 for v in upscale_size(w, h, mode)))

    def test_off_is_nothing(self):
        self.assertIsNone(upscale_size(1344, 768, "off"))
        with self.assertRaises(ValueError):
            upscale_size(1344, 768, "4k")


if __name__ == "__main__":
    unittest.main()
