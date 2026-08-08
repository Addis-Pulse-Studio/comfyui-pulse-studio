"""Run the JavaScript tests, and keep the JS widget spec in sync with the node.

Two production bugs came out of JavaScript that no Python test could see:

  * `TypeError: Cannot redefine property: value` aborted every workflow load.
  * DOM header widgets spliced into `node.widgets` shifted the positional
    `widgets_values` array, so ComfyUI received `duration_seconds='res_multistep'`
    and rejected the prompt with thirteen type errors.

Both now have Node tests, run from here so a regression surfaces in the same
command as everything else. Node is optional: `run_tests.py` must keep working in
a bare ComfyUI environment, so those cases skip rather than fail.

The cross-language test is the important one. `js/ps_widget_order.js` hardcodes
the node's widget order, and a widget added to `INPUT_TYPES` without updating it
would silently stop the detector from working. That is asserted here, in Python,
where INPUT_TYPES actually lives.
"""

import ast
import re
import shutil
import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
JS_DIR = PROJECT_ROOT / "js"
JS_TESTS = PROJECT_ROOT / "tests" / "js"
PANEL = JS_DIR / "pulse_slate.js"
NODES_PY = PROJECT_ROOT / "nodes.py"

NODE = shutil.which("node")
CONNECTION_TYPES = {"MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "AUDIO"}


def required_widgets(class_name="PulseSlate"):
    """Widget names, in order, from INPUT_TYPES in nodes.py (which imports torch,
    so it is read with ast rather than imported)."""
    tree = ast.parse(NODES_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for fn in node.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name != "INPUT_TYPES":
                continue
            for stmt in ast.walk(fn):
                if not isinstance(stmt, ast.Return):
                    continue
                for key, value in zip(stmt.value.keys, stmt.value.values):
                    if getattr(key, "value", None) != "required":
                        continue
                    names = []
                    for wkey, wvalue in zip(value.keys, value.values):
                        first = wvalue.elts[0] if isinstance(wvalue, ast.Tuple) else None
                        if isinstance(first, ast.Constant) and first.value in CONNECTION_TYPES:
                            continue
                        names.append(wkey.value)
                    return names
    raise AssertionError("could not read INPUT_TYPES for %s" % class_name)


def js_native_widgets():
    """The widget names listed in js/ps_widget_order.js."""
    source = (JS_DIR / "ps_widget_order.js").read_text(encoding="utf-8")
    block = source.split("NATIVE_WIDGETS = [", 1)[1].split("\n];", 1)[0]
    return re.findall(r'name:\s*"([^"]+)"', block)


class TestJavaScriptSuites(unittest.TestCase):
    def test_the_js_test_files_exist(self):
        """Asserted without Node, so a deleted test is caught even when skipped."""
        found = sorted(p.name for p in JS_TESTS.glob("test_*.mjs"))
        self.assertIn("test_widget_guard.mjs", found)
        self.assertIn("test_widget_order.mjs", found)

    @unittest.skipIf(NODE is None, "node is not installed")
    def test_every_js_suite_passes(self):
        for path in sorted(JS_TESTS.glob("test_*.mjs")):
            with self.subTest(suite=path.name):
                result = subprocess.run([NODE, str(path)], cwd=str(PROJECT_ROOT),
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0,
                                 "%s failed:\n%s\n%s"
                                 % (path.name, result.stdout, result.stderr))

    @unittest.skipIf(NODE is None, "node is not installed")
    def test_all_shipped_javascript_parses(self):
        for path in sorted(JS_DIR.glob("*.js")):
            with self.subTest(file=path.name):
                result = subprocess.run([NODE, "--check", str(path)],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0,
                                 "%s failed to parse:\n%s" % (path.name, result.stderr))


class TestWidgetOrderInvariant(unittest.TestCase):
    """The panel must never insert a widget ahead of a native one.

    LiteGraph serialises `widgets_values[i] = node.widgets[i].value`. A DOM widget
    inserted before a native one takes its slot and shifts every value after it --
    which is exactly how `duration_seconds` ended up holding 'res_multistep'.
    """

    def test_the_panel_does_not_splice_the_widget_list(self):
        source = PANEL.read_text(encoding="utf-8")
        for banned in ("node.widgets.splice", "placeBefore", "makeHeader"):
            self.assertNotIn(
                banned, source,
                "%s reorders node.widgets; custom widgets must be APPENDED only, "
                "or saved widget values shift and corrupt the node" % banned)

    def test_the_panel_checks_the_invariant_at_runtime(self):
        source = PANEL.read_text(encoding="utf-8")
        self.assertIn("checkWidgetOrder", source)
        self.assertIn("validateWidgetValues", source)

    def test_only_one_dom_widget_is_added(self):
        """Each extra DOM widget is another trailing slot. One is enough."""
        source = PANEL.read_text(encoding="utf-8")
        self.assertEqual(source.count("addDOMWidget("), 1,
                         "every addDOMWidget call appends a widgets_values slot")


class TestJsSpecMatchesTheNode(unittest.TestCase):
    def test_js_widget_list_matches_input_types(self):
        expected = required_widgets()
        # The frontend inserts control_after_generate immediately after seed.
        seed_at = expected.index("seed")
        expected = expected[:seed_at + 1] + ["control_after_generate"] + expected[seed_at + 1:]
        self.assertEqual(
            js_native_widgets(), expected,
            "js/ps_widget_order.js has drifted from INPUT_TYPES. Update NATIVE_WIDGETS, "
            "or the misalignment detector will validate against the wrong shape.")

    def test_js_list_has_the_prompt_boxes_first(self):
        self.assertEqual(js_native_widgets()[:2], ["global_prompt", "shot_prompt"])

    def test_js_list_ends_with_the_storage_widget(self):
        self.assertEqual(js_native_widgets()[-1], "timeline_data")


class TestHooksCannotAbortAWorkflowLoad(unittest.TestCase):
    def test_every_litegraph_hook_is_wrapped(self):
        source = PANEL.read_text(encoding="utf-8")
        for hook in ("onConfigure", "onNodeCreated", "onDragDrop"):
            start = source.find("nodeType.prototype.%s = function" % hook)
            self.assertGreater(start, -1, "%s hook is missing" % hook)
            self.assertIn("try {", source[start:start + 900],
                          "%s is not wrapped; a throw there aborts workflow loading" % hook)

    def test_the_panel_uses_the_tested_guard(self):
        source = PANEL.read_text(encoding="utf-8")
        self.assertIn("ps_widget_guard.js", source)
        self.assertNotIn('Object.defineProperty(widget, "value"', source,
                         "the panel re-implements the value trap; use the tested guard")


if __name__ == "__main__":
    unittest.main()
