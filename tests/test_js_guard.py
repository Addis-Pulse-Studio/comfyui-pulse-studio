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
ORDER_JS = JS_DIR / "ps_widget_order.js"
NODES_PY = PROJECT_ROOT / "nodes.py"
CONSTANTS_PY = PROJECT_ROOT / "comfyui_pulse_studio" / "constants.py"

NODE = shutil.which("node")
CONNECTION_TYPES = {"MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "AUDIO",
                    # 3.0.0's own wired types. PULSE_TIMELINE and PULSE_SHOT carry
                    # compiled data between the pack's nodes; VIDEO is core's.
                    "PULSE_TIMELINE", "PULSE_SHOT", "VIDEO"}

# Every node in the pack is under the slot contract, so every node is compared.
NODE_CLASSES = ("PulseSlate", "PulseShot", "PulseRender", "PulseBench",
                "PulseRetake", "PulseStill")

# Dot-namespaced growing socket groups (§4). They are declared in Python by a
# loop rather than as dict literals, so they are deliberately invisible to the
# AST helpers below -- their sizes are asserted separately, against the same
# constants both languages read.
DYNAMIC_PREFIXES = ("shots.shot_", "refs.ref_image_")


def _input_types(class_name):
    """The parsed INPUT_TYPES dict node for a class in nodes.py.

    nodes.py imports torch, so it is read with ast rather than imported -- which
    keeps this test runnable in the same bare environment as the rest.
    """
    tree = ast.parse(NODES_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for fn in node.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name != "INPUT_TYPES":
                continue
            for stmt in ast.walk(fn):
                if isinstance(stmt, ast.Return):
                    return stmt.value
    raise AssertionError("could not read INPUT_TYPES for %s" % class_name)


def _local_dict(class_name, variable):
    """A dict literal assigned to a local in INPUT_TYPES, e.g. `optional = {...}`.

    The growing socket groups (§4) are added to that local by a `for` loop, which
    an AST reader cannot evaluate. Resolving only the literal is exactly right:
    it yields the fixed sockets, whose ORDER is the thing under contract, and
    leaves the generated ones to be checked by count.
    """
    tree = ast.parse(NODES_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for fn in node.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name != "INPUT_TYPES":
                continue
            for stmt in fn.body:
                if (isinstance(stmt, ast.Assign)
                        and any(getattr(t, "id", None) == variable for t in stmt.targets)
                        and isinstance(stmt.value, ast.Dict)):
                    return stmt.value
    return None


def _section(class_name, section):
    """The `required` or `optional` sub-dict of a class's INPUT_TYPES."""
    schema = _input_types(class_name)
    for key, value in zip(schema.keys, schema.values):
        if getattr(key, "value", None) != section:
            continue
        if isinstance(value, ast.Dict):
            return value
        # `"optional": optional` -- a local built up by a loop. Resolve the
        # literal part of it; see _local_dict.
        if isinstance(value, ast.Name):
            return _local_dict(class_name, value.id)
        return None
    return None


def _is_connection(value):
    """("MODEL", {...}) is a wired input; ("STRING", {...}) is a widget.

    A helper call such as `schema_widget()` is not a Tuple at all, so it falls
    through to False -- which is correct, it declares a widget.
    """
    first = value.elts[0] if isinstance(value, ast.Tuple) else None
    return isinstance(first, ast.Constant) and first.value in CONNECTION_TYPES


def required_widgets(class_name="PulseSlate"):
    """Widget names, in order, from INPUT_TYPES."""
    section = _section(class_name, "required")
    return [k.value for k, v in zip(section.keys, section.values) if not _is_connection(v)]


def declared_inputs(class_name="PulseSlate"):
    """Connection input names, in order: required first, then optional (§4.1)."""
    names = []
    for part in ("required", "optional"):
        section = _section(class_name, part)
        if section is None:
            continue
        names.extend(k.value for k, v in zip(section.keys, section.values)
                     if _is_connection(v))
    return names


def _class_tuple(class_name, attribute):
    """A class-level tuple constant such as RETURN_NAMES, as a list of strings."""
    tree = ast.parse(NODES_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if getattr(target, "id", None) == attribute:
                    return [e.value for e in stmt.value.elts]
    raise AssertionError("%s has no %s" % (class_name, attribute))


def js_widget_names(class_name="PulseSlate", version=None):
    """Widget names for one node class from the SPECS table in ps_widget_order.js.

    Parsed out of the source rather than executed, for the same reason the
    Python side is: this test must run with nothing installed and no Node
    required. The JS suite exercises the same table by importing it.
    """
    source = ORDER_JS.read_text(encoding="utf-8")
    version = version or js_current_version()
    body = source.split("const SPECS = {", 1)[1]
    block = body.split("\n  %s: {" % class_name, 1)[1]
    block = block.split('"%s": [' % version, 1)[1].split("\n    ],", 1)[0]
    return re.findall(r'name:\s*"([^"]+)"', block)


def js_current_version():
    source = ORDER_JS.read_text(encoding="utf-8")
    return re.search(r'export const CURRENT = "([^"]+)"', source).group(1)


def js_native_widgets():
    """The widget names in the NATIVE_WIDGETS export (PulseSlate, current)."""
    return js_widget_names("PulseSlate")


def python_schema_version():
    source = CONSTANTS_PY.read_text(encoding="utf-8")
    return re.search(r'^SCHEMA_VERSION = "([^"]+)"', source, re.M).group(1)


class TestJavaScriptSuites(unittest.TestCase):
    def test_the_js_test_files_exist(self):
        """Asserted without Node, so a deleted test is caught even when skipped."""
        found = sorted(p.name for p in JS_TESTS.glob("test_*.mjs"))
        self.assertIn("test_widget_guard.mjs", found)
        self.assertIn("test_widget_order.mjs", found)
        self.assertIn("test_sockets.mjs", found)

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
        """Banned by source scan across the whole JS layer, not just the panel."""
        for path in sorted(JS_DIR.glob("*.js")):
            source = path.read_text(encoding="utf-8")
            for banned in ("node.widgets.splice", ".widgets.splice",
                           "placeBefore", "makeHeader"):
                self.assertNotIn(
                    banned, source,
                    "%s in %s reorders node.widgets; custom widgets must be APPENDED "
                    "only, or saved widget values shift and corrupt the node"
                    % (banned, path.name))

    def test_the_panel_checks_the_invariant_at_runtime(self):
        source = PANEL.read_text(encoding="utf-8")
        self.assertIn("checkWidgetOrder", source)
        self.assertIn("validateWidgetValues", source)

    def test_only_one_dom_widget_is_added(self):
        """Each extra DOM widget is another trailing slot. One is enough, in the
        entire JS layer -- §3.2 counts across files, not per file."""
        total = sum(path.read_text(encoding="utf-8").count("addDOMWidget(")
                    for path in sorted(JS_DIR.glob("*.js")))
        self.assertEqual(total, 1,
                         "every addDOMWidget call appends a widgets_values slot")

    def test_loading_is_never_positional(self):
        """§3.3: positional assignment is never used to load a workflow again."""
        source = PANEL.read_text(encoding="utf-8")
        self.assertIn("applySavedValues", source,
                      "the panel must restore saved values by name")
        self.assertNotIn("widgets_values[i]", source)
        self.assertNotIn("widgets[i].value =", source)


class TestJsSpecMatchesTheNode(unittest.TestCase):
    """The cross-language parity check.

    js/ps_widget_order.js hardcodes each node's slot layout, and Python declares
    it in INPUT_TYPES. A widget added to one without the other silently loads
    every saved workflow into the wrong widgets. Asserted here, in Python, where
    INPUT_TYPES actually lives -- and extended past widgets to inputs and
    outputs, because §3.2 freezes all three.
    """

    def _expected(self, class_name):
        """INPUT_TYPES widget order, with the frontend's inserted control.

        `control_after_generate` is inserted by the frontend immediately after any
        widget declared with it, which in this pack is always `seed`. Nodes with
        no seed -- PulseShot, PulseRender, PulseBench -- get no insertion.
        """
        names = required_widgets(class_name)
        if "seed" not in names:
            return names
        seed_at = names.index("seed")
        return names[:seed_at + 1] + ["control_after_generate"] + names[seed_at + 1:]

    def test_js_widget_list_matches_input_types(self):
        for class_name in NODE_CLASSES:
            with self.subTest(node=class_name):
                self.assertEqual(
                    js_widget_names(class_name), self._expected(class_name),
                    "js/ps_widget_order.js has drifted from %s.INPUT_TYPES. Add the "
                    "widget to SPECS in the same commit, or every saved workflow "
                    "loads its values into the wrong widgets." % class_name)

    def test_the_js_and_python_schema_versions_agree(self):
        self.assertEqual(
            js_current_version(), python_schema_version(),
            "CURRENT in ps_widget_order.js and SCHEMA_VERSION in constants.py must "
            "match; the node writes one and the loader reads the other")

    def test_every_node_declares_schema_version_at_slot_zero(self):
        for class_name in NODE_CLASSES:
            with self.subTest(node=class_name):
                self.assertEqual(required_widgets(class_name)[0], "schema_version")
                self.assertEqual(js_widget_names(class_name)[0], "schema_version")

    def test_the_slate_frozen_prefix_is_schema_then_timeline(self):
        self.assertEqual(required_widgets("PulseSlate")[:2],
                         ["schema_version", "timeline_data"])
        self.assertEqual(js_widget_names("PulseSlate")[:2],
                         ["schema_version", "timeline_data"])

    def test_the_prompt_boxes_follow_the_frozen_prefix(self):
        self.assertEqual(js_widget_names("PulseSlate")[2:4],
                         ["global_prompt", "shot_prompt"])

    def test_no_node_declares_a_duplicate_widget_name(self):
        for class_name in NODE_CLASSES:
            with self.subTest(node=class_name):
                names = required_widgets(class_name)
                self.assertEqual(sorted(set(names)), sorted(names))


class TestInputAndOutputOrderIsFrozen(unittest.TestCase):
    """§3.4: extend the parity test to RETURN_TYPES / RETURN_NAMES and inputs.

    Link endpoints serialise as [link_id, origin_node, origin_slot, target_node,
    target_slot] -- positional, exactly like widgets. Reordering an input or an
    output silently reconnects every saved graph to the wrong slot, which is
    harder to notice than a shifted widget because the graph still looks right.

    These are literal expected lists rather than a derived comparison. That is
    the point: changing the node requires changing this file, in the same
    commit, deliberately.
    """

    EXPECTED_INPUTS = {
        # The growing groups are appended after these and are checked by
        # test_dynamic_group_sizes_match_the_frontend instead.
        "PulseSlate": ["model", "clip", "vae", "audio_vae",
                       "model_fl2va", "ref_video", "ref_video_audio", "ref_music"],
        "PulseShot": ["start_image", "end_image", "ref_audio"],
        "PulseRender": ["timeline", "model", "vae", "audio_vae", "model_fl2va"],
        "PulseBench": [],
        "PulseRetake": ["model_fl2va", "clip", "vae", "audio_vae", "images", "base_audio"],
        "PulseStill": ["model", "clip", "vae", "audio_vae", "source_image", "ref_images"],
    }

    EXPECTED_OUTPUTS = {
        # 3.0.0 replaced slot 0 with the timeline. `positive` and `latent` stay at
        # slots 1 and 2 so a v2 graph wired into an external sampler keeps working
        # (§13); the old `model` link on slot 0 is dropped by the frontend as a
        # type mismatch, which is the loud half of that trade and is intended.
        "PulseSlate": (
            ["PULSE_TIMELINE", "CONDITIONING", "LATENT", "AUDIO", "IMAGE", "STRING"],
            ["timeline", "positive", "latent", "combined_audio", "images",
             "compiled_prompt"],
        ),
        "PulseShot": (["PULSE_SHOT"], ["shot"]),
        "PulseRender": (["VIDEO", "IMAGE", "AUDIO", "STRING", "STRING"],
                        ["video", "frames", "audio", "segment_paths", "report"]),
        "PulseBench": (["STRING"], ["table"]),
        "PulseRetake": (["IMAGE", "AUDIO", "STRING"], ["images", "audio", "plan"]),
        "PulseStill": (["IMAGE", "STRING"], ["image", "plan"]),
    }

    #: §4. Both languages must agree on how many sockets each group has, or the
    #: frontend will draw sockets the backend does not declare (or stop short of
    #: ones it does, making them unreachable).
    EXPECTED_GROUPS = {
        "PulseSlate": {"refs.ref_image_": 8, "shots.shot_": 24},
        "PulseShot": {"refs.ref_image_": 4},
    }

    def test_dynamic_group_sizes_match_the_frontend(self):
        source = NODES_PY.read_text(encoding="utf-8")
        js = (JS_DIR / "ps_sockets.js").read_text(encoding="utf-8")
        for class_name, groups in self.EXPECTED_GROUPS.items():
            block = js.split("%s: [" % class_name, 1)[1].split("],", 1)[0]
            for prefix, size in groups.items():
                with self.subTest(node=class_name, group=prefix):
                    self.assertIn('prefix: "%s", max: %d' % (prefix, size), block,
                                  "js/ps_sockets.js and nodes.py disagree about the "
                                  "size of %s on %s" % (prefix, class_name))
        # And that the Python constants the loops use are the same numbers.
        for name, size in (("MAX_SLATE_REF_IMAGES", 8), ("MAX_SHOT_SOCKETS", 24),
                           ("MAX_SHOT_REF_IMAGES", 4)):
            self.assertIn("%s = %d" % (name, size), source)

    def test_the_growing_groups_are_declared_last(self):
        """js/ps_sockets.js rebuilds the dynamic tail in its own order, so the
        generated sockets must sit after every fixed one and in the same group
        order, or a saved graph's wires land on the wrong slots."""
        source = NODES_PY.read_text(encoding="utf-8")
        slate = source.split("class PulseSlate", 1)[1].split("RETURN_TYPES", 1)[0]
        self.assertLess(slate.index('optional["refs.ref_image_%d" % i]'),
                        slate.index('optional["shots.shot_%d" % i]'),
                        "references must be declared before shots, matching GROUPS")

    def test_input_order_is_unchanged(self):
        for class_name, expected in self.EXPECTED_INPUTS.items():
            with self.subTest(node=class_name):
                self.assertEqual(
                    declared_inputs(class_name), expected,
                    "%s's connection inputs changed order. Link endpoints are "
                    "positional; every saved graph would rewire itself." % class_name)

    def test_output_order_and_names_are_unchanged(self):
        for class_name, (types, names) in self.EXPECTED_OUTPUTS.items():
            with self.subTest(node=class_name):
                self.assertEqual(_class_tuple(class_name, "RETURN_TYPES"), types)
                self.assertEqual(_class_tuple(class_name, "RETURN_NAMES"), names)

    def test_return_types_and_names_are_the_same_length(self):
        for class_name in NODE_CLASSES:
            with self.subTest(node=class_name):
                self.assertEqual(len(_class_tuple(class_name, "RETURN_TYPES")),
                                 len(_class_tuple(class_name, "RETURN_NAMES")))


class TestShotIdentity(unittest.TestCase):
    """§6.1 -- the id is written once, never changes, and a copy gets a new one.

    The whole seed and cache design rests on this. If a duplicated PulseShot kept
    the original's id, two different shots would share a seed and a cached
    segment, and one would silently render as the other.
    """

    def source(self):
        return PANEL.read_text(encoding="utf-8")

    def test_a_shot_id_is_minted_when_it_is_empty(self):
        source = self.source()
        self.assertIn("function ensureShotId", source)
        self.assertIn("if (!widget.value || taken) widget.value = newShotId();", source)

    def test_a_duplicated_shot_is_re_minted(self):
        """LiteGraph implements duplicate and paste by replaying a serialised
        node, so a copy arrives carrying the original's id."""
        source = self.source()
        block = source.split("function ensureShotId", 1)[1].split("\n}", 1)[0]
        self.assertIn("other !== node", block)
        self.assertIn('other.comfyClass === "PulseShot"', block)

    def test_the_id_is_settled_after_the_restore_not_before(self):
        """Otherwise a genuine load would re-mint every shot in the file and
        invalidate the whole segment cache on open."""
        source = self.source()
        configure = source.split("nodeType.prototype.onConfigure = function (info)", 1)[1]
        configure = configure.split("};", 1)[0]
        self.assertLess(configure.index("restoreByName"), configure.index("ensureShotId"))

    def test_the_hidden_widgets_are_hidden_on_every_node(self):
        source = self.source()
        self.assertIn("hideWidget(this, SCHEMA_WIDGET)", source)
        self.assertIn("hideWidget(this, SHOT_ID_WIDGET)", source)

    def test_python_mints_a_stable_fallback_when_no_frontend_ever_ran(self):
        """A headless API run has no frontend to write the widget, so the node
        derives an id from its own text -- stable for identical text, which is
        the property the seed and the cache actually need."""
        source = NODES_PY.read_text(encoding="utf-8")
        block = source.split("class PulseShot", 1)[1].split("class PulseSlate", 1)[0]
        self.assertIn('text_shot_id(label, visual)', block)


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
