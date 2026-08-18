"""The shipped workflow must stay in sync with the nodes' INPUT_TYPES.

A workflow stores widget values as a positional list. Insert a widget in
INPUT_TYPES and every stored value after it shifts by one -- the sampler name
lands in the scheduler slot, the seed lands in cfg, and ComfyUI loads it without
complaint. This is the same class of failure as reference-tag misnumbering, so it
gets the same treatment: asserted, not assumed.

nodes.py imports torch, so INPUT_TYPES is read with ast rather than by importing.
That keeps this test runnable in the same bare environment as the rest.
"""

import ast
import json
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
WORKFLOW_DIR = PROJECT_ROOT / "example_workflows"
# One graph ships, as of 2026-08-11. §15.4 and §15.9 asked for three -- the short
# path on this node's own sampler, the long path, and the long path with the patch
# chain -- and 3.0.0 released with all three. The short-path and Spectrum+Sage
# graphs were dropped afterwards; the long path is the one that shows what 3.0.0
# is for and the only one that needs no third-party pack to load clean.
#
# Every structural check still runs over the tuple rather than the single path, so
# putting a graph back is a one-line change here. A graph that loads as a silently
# disconnected mess is worse than one that fails to load.
LONG_FORM = WORKFLOW_DIR / "PulseSlate_LongForm.json"
SHIPPED_GRAPHS = (LONG_FORM,)
NODES_PY = PROJECT_ROOT / "nodes.py"

# Widget order is the order of keys in the "required" dict, minus the ones that
# are wired as connections rather than typed on the node face.
CONNECTION_TYPES = {"MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "AUDIO"}


def _required_widgets(class_name):
    """Widget names, in order, from a node class's INPUT_TYPES in nodes.py."""
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
                schema = stmt.value
                for key, value in zip(schema.keys, schema.values):
                    if getattr(key, "value", None) != "required":
                        continue
                    widgets = []
                    for wkey, wvalue in zip(value.keys, value.values):
                        name = wkey.value
                        first = wvalue.elts[0] if isinstance(wvalue, ast.Tuple) else None
                        # ("MODEL",) is a connection; ("STRING", {...}) / (["a","b"], {...})
                        # are widgets.
                        if isinstance(first, ast.Constant) and first.value in CONNECTION_TYPES:
                            continue
                        widgets.append(name)
                    return widgets
    raise AssertionError("could not find INPUT_TYPES for %s" % class_name)


class TestModelPathsMatchWhatComfyUIEmits(unittest.TestCase):
    """A model widget's stored value is compared to the model list *literally*.

    `folder_paths.get_filename_list()` builds names with `os.sep`, and
    `execution.py` rejects anything not in that list with `value_not_in_list`
    before the graph ever runs. So on Windows the only value that validates is
    `minimax\\name.safetensors` -- even though `get_full_path` would happily
    resolve the forward-slash form, because validation runs first.

    These graphs were authored, run and verified on Windows, so they carry the
    backslash. This was briefly "fixed" to forward slashes for portability,
    which made the shipped starter unqueueable on the platform it was built on:
    four red MISSING MODELS on load. There is no portable spelling -- the
    separator belongs to the host -- so the choice is which platform loads clean
    and which clicks the combo once. This test is here to stop the next
    well-meant portability fix from silently costing that again.
    """

    LOADERS = ("UNETLoader", "CLIPLoader", "VAELoader")

    def test_no_shipped_graph_uses_a_forward_slash_in_a_model_path(self):
        for path in WORKFLOW_DIR.glob("*.json"):
            with self.subTest(workflow=path.name):
                offenders = re.findall(r'"[^"]*/[^"]*\.safetensors"',
                                       path.read_text(encoding="utf-8"))
                self.assertEqual(offenders, [],
                                 "ComfyUI validates model values against a list built with "
                                 "os.sep; a forward slash fails validation on Windows:\n  "
                                 + "\n  ".join(sorted(set(offenders))))

    def test_every_loader_names_a_model_in_the_minimax_subfolder(self):
        for path in SHIPPED_GRAPHS:
            d = json.loads(path.read_text(encoding="utf-8"))
            loaders = [n for n in d["nodes"] if n["type"] in self.LOADERS]
            # Was `len(loaders) == 5` -- 2 DiT + 1 text encoder + 2 VAE. That is the
            # long-form graph's own shape, not a rule: a graph that never reaches
            # the anchored branch loads one DiT checkpoint, and as of this commit
            # the long-form graph is one of those. What every graph owes is that
            # the models it does load are the ones the README documents.
            self.assertTrue(loaders, "%s loads no model at all" % path.name)
            for node in loaders:
                with self.subTest(workflow=path.name, node=node["type"]):
                    value = node["widgets_values"][0]
                    self.assertTrue(value.startswith("minimax\\"),
                                    "%s: %r is not in the minimax\\ subfolder the README "
                                    "documents" % (node["type"], value))
                    self.assertTrue(value.endswith(".safetensors"), value)


class TestWorkflowShipsAndParses(unittest.TestCase):
    def test_workflow_exists_and_is_valid_json(self):
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                self.assertTrue(path.is_file(), "missing %s" % path)
                json.loads(path.read_text(encoding="utf-8"))

    def test_links_reference_real_nodes_and_slots(self):
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                by_id = {n["id"]: n for n in d["nodes"]}
                self.assertEqual(len(by_id), len(d["nodes"]), "duplicate node ids")
                seen = set()
                for link_id, src, src_slot, dst, dst_slot, _type in d["links"]:
                    self.assertNotIn(link_id, seen, "duplicate link id %s" % link_id)
                    seen.add(link_id)
                    self.assertIn(src, by_id)
                    self.assertIn(dst, by_id)
                    self.assertLess(src_slot, len(by_id[src]["outputs"]))
                    self.assertLess(dst_slot, len(by_id[dst]["inputs"]))

    def test_link_backfill_is_consistent_both_ways(self):
        """ComfyUI reads both the link table and the per-slot backfill; a
        mismatch loads as a silently disconnected graph."""
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                by_id = {n["id"]: n for n in d["nodes"]}
                for link_id, src, src_slot, dst, dst_slot, _type in d["links"]:
                    self.assertEqual(by_id[dst]["inputs"][dst_slot]["link"], link_id)
                    self.assertIn(link_id, by_id[src]["outputs"][src_slot]["links"])

    def test_no_slot_points_at_a_link_that_does_not_exist(self):
        """The mirror of the check above. Deleting a link and forgetting the
        backfill leaves an input holding a dead id, which loads as connected and
        renders with whatever was last in that slot."""
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                live = {l[0] for l in d["links"]}
                for node in d["nodes"]:
                    for slot in node.get("inputs") or []:
                        if slot.get("link") is not None:
                            self.assertIn(slot["link"], live,
                                          "%s input %s" % (node["type"], slot["name"]))
                    for slot in node.get("outputs") or []:
                        for link_id in slot.get("links") or []:
                            self.assertIn(link_id, live,
                                          "%s output %s" % (node["type"], slot["name"]))

    def test_the_director_is_in_the_graph(self):
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                types = {n["type"] for n in d["nodes"]}
                self.assertIn("PulseSlate", types)

    def test_no_shipped_graph_carries_a_muted_branch(self):
        """§14.6, and the reason the node was split at all.

        2.x shipped the short path and the long path as two parallel wired
        groups with one muted and a Ctrl+M instruction in the README. The two
        drifted, and a 15s timeline that split into two windows re-sampled its
        last window through the still-wired short branch and saved 7 seconds of
        video as the whole film. Mode 2 is 'never', mode 4 is 'bypass'.
        """
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                muted = [n["type"] for n in d["nodes"] if n.get("mode") in (2, 4)]
                self.assertEqual(muted, [],
                                 "%s ships muted nodes: %r" % (path.name, muted))
                titles = " ".join(g.get("title", "") for g in d.get("groups") or [])
                self.assertNotIn("muted", titles.lower())

    def test_the_long_graphs_wire_the_timeline_into_a_render_node(self):
        """§2.3. The short graph hands conditioning to the graph's own sampler;
        the long ones hand the timeline to PulseRender. Neither does both."""
        for path in (LONG_FORM,):
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                by_id = {n["id"]: n for n in d["nodes"]}
                slate = next(n for n in d["nodes"] if n["type"] == "PulseSlate")
                render = next(n for n in d["nodes"] if n["type"] == "PulseRender")
                timeline_links = [l for l in d["links"]
                                  if l[1] == slate["id"] and l[2] == 0]
                self.assertTrue(timeline_links, "the timeline output is not wired")
                self.assertEqual(by_id[timeline_links[0][3]]["type"], "PulseRender")
                # ...and the blocked conditioning outputs are wired to nothing.
                for slot in (1, 2):
                    self.assertEqual(
                        [l for l in d["links"] if l[1] == slate["id"] and l[2] == slot], [],
                        "a long graph must not wire positive/latent -- they are "
                        "blocked on that path and would stop the render")
                self.assertTrue(render["inputs"][0]["link"] is not None)

    def test_every_shot_node_is_actually_wired(self):
        """A shipped graph with a floating PulseShot renders a fraction of the
        film it appears to describe, and says nothing about it.

        This shipped with only shot 1 connected: three shot nodes on the canvas,
        one 6-second window rendered, no warning anywhere -- because a shot that
        is not connected is not a shot, it is a node sitting on a canvas.
        """
        for path in (LONG_FORM,):
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                shots = [n for n in d["nodes"] if n["type"] == "PulseShot"]
                self.assertTrue(shots, "a long-form example needs shot nodes")
                wired = {l[1] for l in d["links"] if l[5] == "PULSE_SHOT"}
                for node in shots:
                    self.assertIn(node["id"], wired,
                                  "%s: PulseShot %s is on the canvas but wired to "
                                  "nothing" % (path.name, node.get("title", node["id"])))

    def test_no_shipped_graph_has_a_node_wired_to_nothing(self):
        """Notes aside, a node with no links at all reads as a mistake."""
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                linked = {l[1] for l in d["links"]} | {l[3] for l in d["links"]}
                floating = [n.get("title") or n["type"] for n in d["nodes"]
                            if n["type"] != "MarkdownNote" and n["id"] not in linked]
                self.assertEqual(floating, [], "%s: %r" % (path.name, floating))

    def test_the_shot_driven_graphs_reference_no_asset_they_do_not_ship(self):
        """The long graphs carry an empty bin on purpose.

        The starter's bin points at example_character_*.png, which a new user has
        not got. Those assets are dropped before tags are assigned, and every
        `@Image1` in the prompt is then correctly reported as unresolved -- which
        is right, but is not what a graph demonstrating the render loop should
        open on.
        """
        for path in (LONG_FORM,):
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                slate = next(n for n in d["nodes"] if n["type"] == "PulseSlate")
                widgets = _required_widgets("PulseSlate")
                index = {name: n for n, name in enumerate(widgets)}
                document = json.loads(slate["widgets_values"][index["timeline_data"]])
                self.assertEqual(document["assets"], [])
                self.assertNotIn("@", slate["widgets_values"][index["global_prompt"]],
                                 "an empty bin cannot resolve an @reference")

    # Dropped with PulseSlate_Starter.json on 2026-08-11:
    # test_the_short_graph_has_no_render_node and
    # test_sigma_shift_is_upstream_of_the_sampler_not_the_director. Both asserted
    # things about the short path -- no PulseRender in the graph, and §1.1's
    # "model patches stay upstream" reaching the sampler through MiniMaxH3SigmaShift
    # rather than through the director. No shipped graph carries a sampler now, so
    # neither has a subject.
    #
    # Nothing replaces them. Together with TestThePatchChainVariant, dropped the
    # same day with the Spectrum+Sage graph, this leaves §1.1's upstream rule and
    # §12.3's Sol-after-Sage ordering trap unasserted by any test -- the only two
    # graphs that carried a model patch are both gone. The rules still hold; they
    # are just documented now rather than enforced. Restore the tests with the
    # graphs.


class TestWidgetOrderMatchesTheNode(unittest.TestCase):
    def _director(self):
        # Read from the long-form graph since 2026-08-11; it was the short
        # starter until that graph was dropped. Either works -- the check is on
        # the director's widget *array*, which is the same shape in every graph.
        d = json.loads(LONG_FORM.read_text(encoding="utf-8"))
        return next(n for n in d["nodes"] if n["type"] == "PulseSlate")

    def test_widget_count_matches_input_types(self):
        widgets = _required_widgets("PulseSlate")
        stored = self._director()["widgets_values"]
        # seed carries control_after_generate, which occupies one extra slot.
        self.assertEqual(
            len(stored), len(widgets) + 1,
            "workflow stores %d values but the node declares %d widgets (+1 for "
            "control_after_generate).\nnode order: %r" % (len(stored), len(widgets), widgets))

    def test_the_frozen_prefix_leads_the_widget_list(self):
        """Spec §3.1. schema_version must be readable before anything else is,
        because it is what says which layout the rest of the array is in."""
        widgets = _required_widgets("PulseSlate")
        self.assertEqual(widgets[:2], ["schema_version", "timeline_data"])

    def test_the_two_prompt_boxes_follow_the_frozen_prefix(self):
        widgets = _required_widgets("PulseSlate")
        self.assertEqual(widgets[2:4], ["global_prompt", "shot_prompt"],
                         "the prompt boxes must lead the visible node face")

    def test_every_node_in_the_pack_opens_with_schema_version(self):
        for name in ("PulseSlate", "PulseRetake", "PulseStill"):
            with self.subTest(node=name):
                self.assertEqual(_required_widgets(name)[0], "schema_version")

    def test_the_workflow_declares_the_current_schema(self):
        stored = self._director()["widgets_values"]
        self.assertEqual(stored[0], "3.0.0",
                         "slot 0 must carry the schema version the file was saved in")

    def test_stored_values_land_on_the_right_widgets(self):
        """Spot-check the values whose type makes a shift obvious."""
        widgets = _required_widgets("PulseSlate")
        stored = self._director()["widgets_values"]
        index = {name: n for n, name in enumerate(widgets)}
        self.assertIsInstance(stored[index["global_prompt"]], str)
        self.assertIsInstance(stored[index["shot_prompt"]], str)
        self.assertEqual(stored[index["aspect_ratio"]], "16:9 landscape")
        self.assertEqual(stored[index["sampler_name"]], "res_multistep")
        self.assertEqual(stored[index["scheduler"]], "simple")
        self.assertEqual(stored[index["cfg"]], 1.0)
        # Everything after seed is shifted one slot by control_after_generate.
        self.assertEqual(stored[index["seed"]], 0)
        self.assertEqual(stored[index["seed"] + 1], "fixed")
        for name, expected in (("partition_strategy", "balanced"), ("resize_method", "crop"),
                               ("carry_mode", "image"), ("ref_image_size", "match")):
            self.assertEqual(stored[index[name] + 1], expected,
                             "%s did not land on its own slot" % name)

    def test_timeline_data_is_valid_json_in_schema_2_form(self):
        widgets = _required_widgets("PulseSlate")
        self.assertEqual(widgets[1], "timeline_data")
        stored = self._director()["widgets_values"]
        document = json.loads(stored[1])
        self.assertIn("assets", document)
        # §3.1: shipped in schema 2 form now, so 1.1 adds to a key that exists.
        self.assertEqual(document["schema"], 2)
        self.assertIn("cast", document)

    def test_the_shipped_prompts_actually_compile(self):
        """The shipped text must produce a real plan, not just look plausible.

        Asserted on the director's own widgets only. The shipped graph is
        shot-node driven -- its per-shot text lives in `PulseShot` nodes that
        reach the director as `PULSE_SHOT` links at queue time, not in the
        `shot_prompt` box -- so the `[Shot N]` markers the short starter used to
        carry here legitimately are not in this array. Their presence in the
        compiled prompt is covered by the compiler's own tests.
        """
        from comfyui_pulse_studio.compiler import compile_timeline
        from comfyui_pulse_studio.widget_state import build_timeline
        widgets = _required_widgets("PulseSlate")
        stored = self._director()["widgets_values"]
        index = {name: n for n, name in enumerate(widgets)}
        timeline, _ = build_timeline(
            stored[index["timeline_data"]],
            global_prompt=stored[index["global_prompt"]],
            shot_prompt=stored[index["shot_prompt"]],
            duration_seconds=stored[index["duration_seconds"]])
        plan = compile_timeline(timeline)
        self.assertTrue(plan.ok, plan.problems)
        self.assertIn("overall_soundscape:", plan.windows[0].prompt)

    def test_no_shipped_text_hand_types_a_reference_ordinal(self):
        """The graph ships as an example, so it must not teach the wrong habit.

        `<Picture 3>` typed by hand is exactly what the asset bin exists to stop
        being necessary: the ordinal is computed from live bin order at compile
        time, so a hand-typed one goes stale the moment a reference is added or
        reordered. Checked across the director *and* every `PulseShot`, because
        the shot nodes are where the per-shot text lives in this graph.

        This does not assert that an `@name` is present. The shipped graph's bin
        is empty on purpose, so there is nothing for one to resolve to -- that
        half of the old assertion belonged to the short starter, which shipped a
        populated bin, and it went with it.
        """
        d = json.loads(LONG_FORM.read_text(encoding="utf-8"))
        by_type = {}
        for node in d["nodes"]:
            by_type.setdefault(node["type"], []).append(node)

        checked = 0
        for node_type, keys in (("PulseSlate", ("global_prompt", "shot_prompt")),
                                ("PulseShot", ("label", "visual", "audio_line"))):
            index = {name: n for n, name
                     in enumerate(_required_widgets(node_type))}
            for node in by_type.get(node_type, []):
                for key in keys:
                    text = node["widgets_values"][index[key]]
                    for bad in ("<Picture ", "<Video ", "<Audio "):
                        self.assertNotIn(
                            bad, text,
                            "%s %s hand-types a tag ordinal"
                            % (node_type, key))
                    checked += 1

        # Guards the loop above: a renamed node type would silently check nothing.
        self.assertGreaterEqual(checked, 11, "expected the director's two text "
                                             "boxes plus three fields on each of "
                                             "three PulseShot nodes")


if __name__ == "__main__":
    unittest.main()


class TestShippedGraphsCarryEveryWidget(unittest.TestCase):
    """A shipped graph must store a value for every widget its node declares.

    This is the guard that was missing when `ref_audio_mode` and
    `use_reference_audio` were appended: the cross-language parity test caught
    `ps_widget_order.js` drifting from `INPUT_TYPES`, and nothing at all noticed
    that four nodes across two shipped workflows were still storing the previous
    layout.

    Nothing *broke* -- a fresh frontend builds the node from `object_info` and the
    appended widget keeps its default. The failure it invites is quieter than
    that. `widgets_values` is applied positionally, so the moment anyone inserts
    rather than appends, a short array silently shifts every value past the
    insertion point, and a graph that loads without complaint renders with the
    wrong sampler or the wrong continuity mode. Storing the full array is what
    keeps the shipped graphs a faithful record of the layout that wrote them.
    """

    def _tables(self):
        from tests.test_js_guard import js_widget_names
        return {name: js_widget_names(name)
                for name in ("PulseSlate", "PulseShot", "PulseRender", "PulseBench")}

    def test_every_pulse_node_stores_a_full_widget_array(self):
        tables = self._tables()
        for path in WORKFLOW_DIR.glob("*.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for node in doc.get("nodes", []):
                names = tables.get(node.get("type"))
                if names is None:
                    continue
                stored = node.get("widgets_values") or []
                with self.subTest(workflow=path.name, node=node["type"], id=node["id"]):
                    self.assertGreaterEqual(
                        len(stored), len(names),
                        "%s node %s stores %d widget values but %s declares %d "
                        "(%r). Append the new value(s) to the shipped graph in the "
                        "same commit that appends the widget."
                        % (path.name, node["id"], len(stored), node["type"],
                           len(names), names[len(stored):]))

    def test_no_graph_stores_more_values_than_the_node_has_widgets(self):
        """The other direction: a stale extra value is a removed widget, and
        removing one is forbidden outright by the slot contract."""
        tables = self._tables()
        for path in WORKFLOW_DIR.glob("*.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for node in doc.get("nodes", []):
                names = tables.get(node.get("type"))
                if names is None:
                    continue
                stored = node.get("widgets_values") or []
                with self.subTest(workflow=path.name, node=node["type"], id=node["id"]):
                    # One spare is legitimate: LiteGraph appends a hidden
                    # control_after_generate value after every seed widget.
                    self.assertLessEqual(len(stored), len(names) + 1,
                                         "%s node %s stores %d values for %d widgets"
                                         % (path.name, node["id"], len(stored), len(names)))
