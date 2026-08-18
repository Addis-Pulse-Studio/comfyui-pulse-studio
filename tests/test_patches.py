"""The model-patch warning (spec §10) and the single-checkpoint warning (§18.1).

WHAT THIS IS PROTECTING

The node consumes a patched MODEL and never applies a patch itself. That means
an unpatched graph is indistinguishable from a patched one right up until the
render, where it shows up as either "this is inexplicably slow" or, on a
full-length window, an out-of-memory error that arrives twenty minutes in and
looks like it came from somewhere else.

So the node says so up front. Every finding is a WARNING: the user may be
deliberately unpatched -- on a bigger card, or while measuring what the patches
actually buy -- and a hard failure would be this pack overruling a decision that
is not its to make.

Detection is duck-typed over model_options by key presence, so these tests use
plain stubs. That is not a shortcut around a real dependency; it is the design.
This pack must not import Spectrum or any attention pack, because a hard
dependency would make it uninstallable for anyone who wants it without them.

The key names were read out of source, not guessed:
  comfy/ldm/modules/attention.py    -> transformer_options["optimized_attention_override"]
  comfy/ (multigpu_clones, offload_device)
  ComfyUI-Spectrum-MiniMax-H3       -> BINDING_KEY = "spectrum_h3_binding"
"""

import ast
import unittest
from pathlib import Path

from comfyui_pulse_studio.patches import (
    ATTENTION_KEYS,
    OFFLOAD_KEYS,
    RECOMMENDED_CHAIN,
    check_model_patches,
    check_single_checkpoint,
    inspect_model_options,
)


class StubModel:
    """A MODEL as far as this detector is concerned: something with a dict."""

    def __init__(self, model_options=None):
        self.model_options = model_options


PATCHED = {
    "spectrum_h3_binding": object(),
    "transformer_options": {"optimized_attention_override": lambda *a, **k: None},
}


class TestTheUnpatchedCase(unittest.TestCase):
    """§10's named requirement: a stub model with an empty model_options."""

    def test_an_empty_model_options_triggers_both_warnings(self):
        report = inspect_model_options({})
        self.assertFalse(report.has_attention)
        self.assertFalse(report.has_offload)
        self.assertFalse(report.ok)
        joined = " ".join(report.warnings)
        self.assertIn("attention", joined.lower())
        self.assertIn("offload", joined.lower())

    def test_the_stub_model_object_works_too(self):
        report = check_model_patches(StubModel({}))
        self.assertEqual(len(report.warnings), 3)

    def test_a_model_with_no_model_options_attribute_does_not_raise(self):
        """A detector that can throw inside a render is worse than a wrong one."""
        report = check_model_patches(object())
        self.assertFalse(report.has_attention)
        self.assertFalse(report.ok)

    def test_none_and_junk_are_treated_as_unpatched(self):
        for junk in (None, [], "model_options", 7, object()):
            with self.subTest(value=junk):
                report = inspect_model_options(junk)
                self.assertFalse(report.has_attention)
                self.assertFalse(report.has_offload)


class TestTheOffloadWordingIsSpecific(unittest.TestCase):
    """§18.1: offload is mandatory on this hardware, not a speed knob.

    The wording is asserted because it is the whole point. "Offload makes it
    faster" invites the user to turn it off and then meet an out-of-memory error
    they have no reason to connect back to it.
    """

    def _offload_warning(self):
        report = inspect_model_options({})
        found = [w for w in report.warnings if "offload" in w.lower()]
        self.assertEqual(len(found), 1, report.warnings)
        return found[0]

    def test_it_names_spectrum_system_ram(self):
        text = self._offload_warning()
        self.assertIn("system_ram", text)
        self.assertIn("Spectrum", text)

    def test_it_says_fit_rather_than_faster(self):
        text = self._offload_warning().lower()
        self.assertIn("fit at all", text)
        self.assertIn("not a speed", text)
        self.assertIn("out-of-memory", text)

    def test_it_names_the_card_and_the_window_size(self):
        text = self._offload_warning()
        self.assertIn("32 GB", text)
        self.assertIn("362-frame", text)

    def test_the_attention_warning_names_sage(self):
        report = inspect_model_options({})
        found = [w for w in report.warnings if "attention" in w.lower()]
        self.assertTrue(found)
        self.assertIn("Sage Attention", found[0])

    def test_every_warning_names_the_recommended_chain_or_explains_order(self):
        report = inspect_model_options({})
        joined = " ".join(report.warnings)
        self.assertIn(RECOMMENDED_CHAIN, joined)
        self.assertIn("upstream", joined.lower())


class TestPatchDetection(unittest.TestCase):
    def test_a_fully_patched_model_warns_about_nothing(self):
        report = inspect_model_options(PATCHED)
        self.assertTrue(report.has_attention)
        self.assertTrue(report.has_offload)
        self.assertTrue(report.ok)
        self.assertEqual(report.warnings, [])

    def test_every_documented_attention_key_is_detected(self):
        for path in ATTENTION_KEYS:
            with self.subTest(key=".".join(path)):
                options = {}
                node = options
                for key in path[:-1]:
                    node = node.setdefault(key, {})
                node[path[-1]] = lambda *a, **k: None
                self.assertTrue(inspect_model_options(options).has_attention)

    def test_every_documented_offload_key_is_detected(self):
        for path in OFFLOAD_KEYS:
            with self.subTest(key=".".join(path)):
                options = {}
                node = options
                for key in path[:-1]:
                    node = node.setdefault(key, {})
                node[path[-1]] = object()
                self.assertTrue(inspect_model_options(options).has_offload)

    def test_the_launch_flag_counts_as_an_attention_patch(self):
        """--use-sage-attention patches attention process-wide and leaves no
        trace in model_options, so it has to be passed in separately."""
        report = inspect_model_options({}, sage_attention_global=True)
        self.assertTrue(report.has_attention)
        self.assertEqual(report.attention_evidence, "--use-sage-attention")
        # Matched on the opening clause, not on the word: every warning quotes
        # the recommended chain, and the chain itself contains "Sage Attention".
        self.assertFalse(any(w.startswith("No attention patch") for w in report.warnings))

    def test_an_empty_transformer_options_is_not_evidence(self):
        """ComfyUI creates transformer_options and patches_replace as empty
        dicts on its own. Treating that as a patch would silence the warning on
        every unpatched graph -- the exact case it exists for."""
        report = inspect_model_options({"transformer_options": {},
                                        "patches_replace": {}})
        self.assertFalse(report.has_attention)

    def test_an_empty_patches_replace_is_not_evidence(self):
        report = inspect_model_options(
            {"transformer_options": {"patches_replace": {}}})
        self.assertFalse(report.has_attention)

    def test_evidence_names_the_key_that_was_found(self):
        report = inspect_model_options(PATCHED)
        self.assertEqual(report.offload_evidence, "spectrum_h3_binding")
        self.assertEqual(report.attention_evidence,
                         "transformer_options.optimized_attention_override")

    def test_one_patch_present_warns_only_about_the_other(self):
        offload_only = inspect_model_options({"spectrum_h3_binding": object()})
        self.assertTrue(offload_only.has_offload)
        self.assertFalse(offload_only.has_attention)
        self.assertTrue(offload_only.warnings[0].startswith("No attention patch"))
        self.assertFalse(any(w.startswith("No offload patch") for w in offload_only.warnings))

    def test_the_report_serialises_for_the_node_face(self):
        payload = inspect_model_options({}).to_dict()
        self.assertEqual(sorted(payload), ["attention_evidence", "has_attention",
                                           "has_offload", "offload_evidence",
                                           "warnings"])
        self.assertIsInstance(payload["warnings"], list)


class TestSingleCheckpointWarning(unittest.TestCase):
    """§18.1: ref2va and fl2va are ~21 GB each; using both in one execution on a
    32 GB card forces an evict-and-reload mid-render."""

    def test_both_branches_live_with_fl2va_connected_warns(self):
        warnings, notes = check_single_checkpoint({"ref2va", "fl2va"}, fl2va_connected=True)
        self.assertEqual(len(warnings), 1)
        self.assertIn("21 GB", warnings[0])
        self.assertIn("evict", warnings[0])
        self.assertEqual(notes, [], "the loud case must not also emit the quiet one")

    def test_connecting_fl2va_without_using_it_is_a_note_not_a_warning(self):
        """Silent until 2026-08-17, on the reasoning that lazy loading made it free.

        It is not free. With Spectrum offloading history to system RAM on a 32 GB
        box, a ~20 GB checkpoint nobody samples with is still 20 GB of system RAM
        that the render wanted. Not a mistake either -- leaving the socket wired
        is a reasonable way to work -- so it is a note, and it stays off the node
        face.
        """
        warnings, notes = check_single_checkpoint({"ref2va"}, fl2va_connected=True)
        self.assertEqual(warnings, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("model_fl2va", notes[0])
        self.assertIn("no window uses it", notes[0])

    def test_both_branches_without_the_second_model_is_neither(self):
        """That case has its own message about falling back to the main model."""
        self.assertEqual(check_single_checkpoint({"ref2va", "fl2va"},
                                                 fl2va_connected=False), ([], []))

    def test_an_unconnected_fl2va_is_silent_whatever_the_branches(self):
        for branches in (None, set(), {"ref2va"}, {"ref2va", "fl2va"}):
            with self.subTest(branches=branches):
                self.assertEqual(check_single_checkpoint(branches, False), ([], []))

    def test_empty_and_none_still_produce_a_readable_note(self):
        """A connected checkpoint and no compiled windows: the note must not
        render an empty branch list into the middle of a sentence."""
        for branches in (None, set()):
            with self.subTest(branches=branches):
                warnings, notes = check_single_checkpoint(branches, True)
                self.assertEqual(warnings, [])
                self.assertEqual(len(notes), 1)
                self.assertIn("the reference branch", notes[0])

    def test_a_none_branch_is_ignored_rather_than_counted(self):
        warnings, notes = check_single_checkpoint({None, "ref2va"}, True)
        self.assertEqual(warnings, [])
        self.assertEqual(len(notes), 1)


class TestItNeverBlocks(unittest.TestCase):
    def test_the_report_carries_no_blocking_signal(self):
        """There is deliberately no `fatal` or `block` on PatchReport. The user
        may be running unpatched on purpose, and that is their call."""
        report = inspect_model_options({})
        for attribute in ("fatal", "block", "error", "raise_"):
            self.assertFalse(hasattr(report, attribute))

    def test_warnings_are_plain_strings(self):
        for warning in inspect_model_options({}).warnings:
            self.assertIsInstance(warning, str)
            self.assertTrue(warning.strip())


PROJECT_ROOT = Path(__file__).parent.parent
NODES_PY = PROJECT_ROOT / "nodes.py"
JS_DIR = PROJECT_ROOT / "js"
NODE_CLASSES = ("PulseSlate", "PulseRetake", "PulseStill")


def _js_code_only(path):
    """JavaScript source with /* */ and // comments removed.

    Crude but sufficient: it is used to assert what the code *calls*, and a
    banned name inside an explanatory comment is not a call.
    """
    import re
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def _class_def(name):
    tree = ast.parse(NODES_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError("no class %s in nodes.py" % name)


class TestTheWarningIsActuallyWired(unittest.TestCase):
    """A correct detector that nothing calls is worth nothing.

    nodes.py imports torch and comfy, so this is asserted with ast rather than
    by importing -- the same approach the rest of the suite takes, and the reason
    it all runs in a bare environment.
    """

    def test_every_node_calls_the_patch_check_in_execute(self):
        for name in NODE_CLASSES:
            with self.subTest(node=name):
                execute = next(
                    (f for f in _class_def(name).body
                     if isinstance(f, ast.FunctionDef) and f.name == "execute"), None)
                self.assertIsNotNone(execute, "%s has no execute" % name)
                calls = [n.func.id for n in ast.walk(execute)
                         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
                self.assertIn("_report_patches", calls,
                              "%s never checks whether its model is patched" % name)

    def test_every_node_accepts_unique_id(self):
        """Without it the warning has no node to attach itself to."""
        for name in NODE_CLASSES:
            with self.subTest(node=name):
                execute = next(f for f in _class_def(name).body
                               if isinstance(f, ast.FunctionDef) and f.name == "execute")
                args = [a.arg for a in execute.args.args] + \
                       [a.arg for a in execute.args.kwonlyargs]
                self.assertIn("unique_id", args)

    def test_every_node_declares_the_hidden_input(self):
        """Every node in the pack, not just the ones that check patches.

        UNIQUE_ID is what lets any warning be addressed to the node that raised
        it. PulseRender declares HIDDEN_INPUTS_WITH_PROMPT, which is the same
        dict plus PROMPT -- it needs the graph to answer whether its `frames`
        output is wired to anything (§8).
        """
        source = NODES_PY.read_text(encoding="utf-8")
        tree = ast.parse(source)
        declared = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not node.name.startswith("Pulse"):
                continue
            body = ast.get_source_segment(source, node) or ""
            self.assertIn('"hidden": HIDDEN_INPUTS', body,
                          "%s declares no hidden inputs, so a warning raised by it "
                          "has no node to attach to" % node.name)
            declared += 1
        self.assertEqual(declared, 6)
        self.assertIn('HIDDEN_INPUTS = {"unique_id": "UNIQUE_ID"}', source)
        self.assertIn('HIDDEN_INPUTS_WITH_PROMPT = {"unique_id": "UNIQUE_ID", '
                      '"prompt": "PROMPT"}', source)

    def test_the_slate_checks_the_model_it_was_given_not_the_one_it_returns(self):
        """§10's actual point. Path B samples with the INPUT model, so a patch
        applied to this node's output would do nothing for long timelines."""
        source = NODES_PY.read_text(encoding="utf-8")
        self.assertIn("_report_patches(model, unique_id,", source)
        self.assertNotIn("_report_patches(shifted", source)

    def test_the_node_face_channel_is_best_effort(self):
        """A missing frontend must not break a headless API render."""
        source = NODES_PY.read_text(encoding="utf-8")
        start = source.index("def _warn_on_node")
        body = source[start:start + 1200]
        self.assertIn("try:", body)
        self.assertIn("except Exception", body)


class TestTheNodeFaceRenderer(unittest.TestCase):
    def test_the_warning_layer_ships(self):
        self.assertTrue((JS_DIR / "ps_warnings.js").is_file())

    def test_it_listens_for_the_event_the_backend_sends(self):
        js = (JS_DIR / "ps_warnings.js").read_text(encoding="utf-8")
        py = NODES_PY.read_text(encoding="utf-8")
        self.assertIn('"pulse_studio.warnings"', py)
        self.assertIn('EVENT = "pulse_studio.warnings"', js)

    def test_it_costs_no_widget_slot(self):
        """§3.2: one addDOMWidget in the whole JS layer, and the bin owns it.

        Scanned with comments stripped. This file's own header explains why it
        does not call addDOMWidget, and a scan that cannot tell prose from code
        would fail on the explanation -- the same reason the Python no-network
        scan walks the AST instead of grepping.
        """
        js = _js_code_only(JS_DIR / "ps_warnings.js")
        self.assertNotIn("addDOMWidget", js)
        self.assertNotIn("addWidget", js)
        self.assertIn("onDrawForeground", js)

    def test_it_does_not_compose_its_own_warning_text(self):
        """The backend owns the wording, so console and node face cannot drift.
        The banner heading is the one string this file contributes."""
        js = (JS_DIR / "ps_warnings.js").read_text(encoding="utf-8")
        for phrase in ("system_ram", "Sage Attention", "362-frame", "32 GB"):
            self.assertNotIn(phrase, js,
                             "%s is the backend's wording; the renderer must not "
                             "carry a second copy of it" % phrase)

    def test_stale_warnings_are_cleared_when_a_run_starts(self):
        js = (JS_DIR / "ps_warnings.js").read_text(encoding="utf-8")
        self.assertIn("execution_start", js)


if __name__ == "__main__":
    unittest.main()
