"""Patch detection and `patch_fingerprint`. Spec §12.4, deliverable §15.7.

Each of the four packs is stubbed present and absent. The stubs mirror the real
shapes wherever the pack was available to read:

  * KJNodes `PathchSageAttentionKJ.patch` sets
    `model_options["transformer_options"]["optimized_attention_override"]` to a
    closure named `attention_override_sage`, and records the mode nowhere.
  * Spectrum `comfyui_spectrum_h3/sampling.py` sets
    `model_options["spectrum_h3_binding"]` to a binding object, and reads
    `transformer_options["easycache"]` when checking for the §12.7 conflict.

Sol-Attn was not installed on the machine this was written on, so its stub is
built from the descriptor shape §12.4 specifies rather than from its source. That
is exactly why detection is a scan over key names and callable identities instead
of a table of literal keys -- the pack lands in the fingerprint either way.
"""

import unittest

from comfyui_pulse_studio.fingerprint import (
    NO_PATCHES_WARNING,
    describe_model_patches,
    describe_patches,
    fingerprint_from_shapes,
    model_fingerprint,
    patch_chain_summary,
    patch_fingerprint,
    patch_warnings,
    sink_conditioning,
)


# ── stubs ───────────────────────────────────────────────────────────────────

def sage_override(func, *args, **kwargs):  # pragma: no cover - never called
    return None


sage_override.__qualname__ = "PathchSageAttentionKJ.patch.<locals>.attention_override_sage"
sage_override.__module__ = "nodes.model_optimization_nodes"


def sol_override(func, *args, **kwargs):  # pragma: no cover - never called
    return None


sol_override.__qualname__ = "SolAttnScheduledPatch.patch.<locals>.sol_attn_forward"
sol_override.__module__ = "comfyui_sol_attn.patcher"


class SpectrumBinding:
    """Mirrors the real binding: a small object carrying the widget values."""

    __slots__ = ("history_storage", "step_skip", "start_percent")

    def __init__(self, history_storage="system_ram", step_skip=2, start_percent=0.1):
        self.history_storage = history_storage
        self.step_skip = step_skip
        self.start_percent = start_percent


SOL_SETTINGS = {
    "node": "scheduled", "tau_start": 2.0, "tau_end": 0.8, "curve": "linear",
    "dense_percent": 0.2, "min_tokens": 8192,
    "sink_conditioning": "exact_kv_and_rows", "int8_qk": False, "thresh_type": "diag",
}


class StubModel:
    def __init__(self, model_options=None, object_patches=None):
        self.model_options = model_options if model_options is not None else {}
        self.object_patches = object_patches or {}


def bare():
    return StubModel({"transformer_options": {}})


def with_sage():
    return StubModel({"transformer_options": {"optimized_attention_override": sage_override}})


def with_spectrum(**kwargs):
    return StubModel({"transformer_options": {}, "spectrum_h3_binding": SpectrumBinding(**kwargs)})


def with_sol(**overrides):
    settings = dict(SOL_SETTINGS)
    settings.update(overrides)
    return StubModel({"transformer_options": {
        "optimized_attention_override": sol_override,
        "sol_attn_settings": settings,
    }})


def with_ff_chunk():
    return StubModel(
        {"transformer_options": {}},
        object_patches={"diffusion_model.blocks.0.ffn.forward": _chunked_forward})


def _chunked_forward(*args, **kwargs):  # pragma: no cover - never called
    return None


_chunked_forward.__qualname__ = "MiniMaxH3ChunkFeedForward.patch.<locals>.chunk_ff_forward"
_chunked_forward.__module__ = "comfyui_sol_attn.chunk_feedforward"


def with_easycache():
    return StubModel({"transformer_options": {"easycache": {"threshold": 0.2}}})


def full_chain():
    """The §12.3 chain: Spectrum -> Sage -> Sol -> FF chunking."""
    return StubModel(
        {
            "transformer_options": {
                # Sol was applied last, so its override is the live one -- which is
                # precisely why applying Sage *after* Sol shadows it (§12.3).
                "optimized_attention_override": sol_override,
                "sol_attn_settings": dict(SOL_SETTINGS),
            },
            "spectrum_h3_binding": SpectrumBinding(),
        },
        object_patches={"diffusion_model.blocks.0.ffn.forward": _chunked_forward})


# ── presence and absence ────────────────────────────────────────────────────

class TestDetectionPerPack(unittest.TestCase):
    def test_nothing_on_a_bare_model(self):
        descriptor = describe_model_patches(bare())
        self.assertFalse(descriptor["detected"])
        for pack in ("sol_attn", "sage", "spectrum", "ff_chunk", "easycache"):
            self.assertFalse(descriptor[pack]["present"], pack)

    def test_sage_alone(self):
        descriptor = describe_model_patches(with_sage())
        self.assertTrue(descriptor["detected"])
        self.assertTrue(descriptor["sage"]["present"])
        self.assertFalse(descriptor["sol_attn"]["present"])
        self.assertFalse(descriptor["spectrum"]["present"])

    def test_sage_via_the_launch_flag_leaves_no_model_options_trace(self):
        """--use-sage-attention patches attention process-wide, so it has to be
        passed in; a detector reading only model_options would miss it entirely."""
        descriptor = describe_patches({}, sage_attention_global=True)
        self.assertTrue(descriptor["sage"]["present"])
        self.assertTrue(descriptor["detected"])

    def test_spectrum_alone_and_its_settings(self):
        descriptor = describe_model_patches(with_spectrum())
        self.assertTrue(descriptor["spectrum"]["present"])
        self.assertEqual(descriptor["spectrum"]["history_storage"], "system_ram")
        self.assertEqual(descriptor["spectrum"]["step_skip"], 2)

    def test_sol_alone_and_its_settings(self):
        descriptor = describe_model_patches(with_sol())
        self.assertTrue(descriptor["sol_attn"]["present"])
        self.assertEqual(descriptor["sol_attn"]["tau_start"], 2.0)
        self.assertEqual(descriptor["sol_attn"]["dense_percent"], 0.2)
        # Sol's override must not be misattributed to Sage.
        self.assertFalse(descriptor["sage"]["present"])

    def test_ff_chunking_is_found_through_the_object_patch_registry(self):
        """It patches an MLP forward, not attention, so it appears nowhere in
        transformer_options and only the object-patch keys reveal it."""
        descriptor = describe_model_patches(with_ff_chunk())
        self.assertTrue(descriptor["ff_chunk"]["present"])

    def test_easycache_alone(self):
        self.assertTrue(describe_model_patches(with_easycache())["easycache"]["present"])

    def test_the_whole_chain(self):
        descriptor = describe_model_patches(full_chain())
        for pack in ("sol_attn", "spectrum", "ff_chunk"):
            self.assertTrue(descriptor[pack]["present"], pack)
        self.assertTrue(descriptor["detected"])

    def test_an_unrecognised_attention_pack_still_moves_the_hash(self):
        """A pack this module has never heard of is still an approximation."""
        def mystery(func, *a, **k):  # pragma: no cover
            return None
        mystery.__module__ = "some_other_pack.attn"
        descriptor = describe_patches(
            {"transformer_options": {"optimized_attention_override": mystery}})
        self.assertTrue(descriptor["attention_override"]["present"])
        self.assertTrue(descriptor["detected"])
        self.assertNotEqual(patch_fingerprint(descriptor),
                            patch_fingerprint(describe_model_patches(bare())))


class TestNeverRaises(unittest.TestCase):
    """A detector that can throw inside a render is worse than one that is wrong."""

    def test_none_and_junk_are_treated_as_unpatched(self):
        for junk in (None, "", 5, [], {"transformer_options": "not a dict"}):
            descriptor = describe_patches(junk)
            self.assertIn("detected", descriptor)

    def test_a_model_missing_every_attribute(self):
        descriptor = describe_model_patches(object())
        self.assertFalse(descriptor["detected"])

    def test_a_settings_object_whose_properties_raise(self):
        class Hostile:
            @property
            def boom(self):
                raise RuntimeError("no")

        model = StubModel({"spectrum_binding_hostile": Hostile()})
        describe_model_patches(model)  # must not raise


# ── the fingerprint itself ──────────────────────────────────────────────────

class TestFingerprint(unittest.TestCase):
    def test_is_sixteen_hex_chars(self):
        value = patch_fingerprint(describe_model_patches(full_chain()))
        self.assertEqual(len(value), 16)
        int(value, 16)

    def test_stable_across_repeated_calls(self):
        first = patch_fingerprint(describe_model_patches(full_chain()))
        second = patch_fingerprint(describe_model_patches(full_chain()))
        self.assertEqual(first, second)

    def test_no_memory_address_reaches_the_descriptor(self):
        """The failure that would silently empty the cache on every restart.

        A repr() of a settings object carries `0x7f...`, which differs per
        process. Two runs of the identical graph would then key differently and
        re-render everything, for ever, with no error to notice.
        """
        import json

        blob = json.dumps(describe_model_patches(full_chain()))
        self.assertNotIn("0x", blob)
        self.assertNotIn(" at ", blob)

    def test_a_settings_change_moves_the_fingerprint(self):
        dense = patch_fingerprint(describe_model_patches(with_sol(dense_percent=0.2)))
        sparse = patch_fingerprint(describe_model_patches(with_sol(dense_percent=0.5)))
        self.assertNotEqual(dense, sparse)

    def test_ff_chunking_is_included_even_though_it_is_bit_identical(self):
        """§12.4: an unnecessary re-render costs minutes; a wrong reuse costs the
        deliverable."""
        with_chunk = patch_fingerprint(describe_model_patches(full_chain()))
        without = patch_fingerprint(describe_model_patches(with_sol()))
        self.assertNotEqual(with_chunk, without)

    def test_each_pack_present_differs_from_each_pack_absent(self):
        seen = set()
        for factory in (bare, with_sage, with_spectrum, with_sol, with_ff_chunk,
                        with_easycache, full_chain):
            seen.add(patch_fingerprint(describe_model_patches(factory())))
        self.assertEqual(len(seen), 7)


class TestModelFingerprint(unittest.TestCase):
    """§7.1 -- a short stable identity for the checkpoint."""

    def test_shapes_hash_independently_of_dict_order(self):
        a = fingerprint_from_shapes([("b.weight", (4, 4)), ("a.weight", (8,))])
        b = fingerprint_from_shapes([("a.weight", (8,)), ("b.weight", (4, 4))])
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)

    def test_a_different_checkpoint_hashes_differently(self):
        self.assertNotEqual(fingerprint_from_shapes([("a", (8,))]),
                            fingerprint_from_shapes([("a", (16,))]))

    def test_an_unreadable_model_yields_unknown_rather_than_raising(self):
        self.assertEqual(model_fingerprint(object()), "unknown")

    def test_a_declared_identity_is_preferred(self):
        model = StubModel()
        model.model_fingerprint = "deadbeefdeadbeef"
        self.assertEqual(model_fingerprint(model), "deadbeefdeadbeef")


# ── what the report is told ─────────────────────────────────────────────────

class TestWarnings(unittest.TestCase):
    def test_an_empty_chain_warns_that_the_cache_cannot_protect_itself(self):
        warnings = patch_warnings(describe_model_patches(bare()))
        self.assertIn(NO_PATCHES_WARNING, warnings)

    def test_a_detected_chain_does_not(self):
        self.assertEqual(patch_warnings(describe_model_patches(full_chain())), [])

    def test_spectrum_and_easycache_together_are_reported_not_blocked(self):
        """§12.7 -- 'mutually exclusive in practice' is a community finding this
        pack cannot verify, so it is a warning and the run continues."""
        model = StubModel({"transformer_options": {"easycache": {"threshold": 0.2}},
                           "spectrum_h3_binding": SpectrumBinding()})
        warnings = patch_warnings(describe_model_patches(model))
        self.assertTrue(any("mutually exclusive" in w for w in warnings))

    def test_paired_audio_warns_on_cheap_sink_conditioning(self):
        """§12.6 -- the one setting whose wrong value damages exactly what this
        pack is for: per-character audio sync lives in the rows it protects."""
        descriptor = describe_model_patches(with_sol(sink_conditioning="exact_kv"))
        warnings = patch_warnings(descriptor, paired_audio_count=3)
        self.assertTrue(any("sink_conditioning" in w for w in warnings))

    def test_no_such_warning_with_the_expensive_setting(self):
        descriptor = describe_model_patches(with_sol())
        self.assertEqual(sink_conditioning(descriptor), "exact_kv_and_rows")
        self.assertEqual(patch_warnings(descriptor, paired_audio_count=3), [])

    def test_no_such_warning_with_a_single_audio_reference(self):
        descriptor = describe_model_patches(with_sol(sink_conditioning="off"))
        self.assertEqual(patch_warnings(descriptor, paired_audio_count=1), [])


class TestChainSummary(unittest.TestCase):
    def test_lists_every_detected_pack(self):
        lines = patch_chain_summary(describe_model_patches(full_chain()))
        joined = "\n".join(lines)
        for pack in ("sol_attn", "spectrum", "ff_chunk"):
            self.assertIn(pack, joined)

    def test_is_empty_for_a_bare_model(self):
        self.assertEqual(patch_chain_summary(describe_model_patches(bare())), [])


if __name__ == "__main__":
    unittest.main()
