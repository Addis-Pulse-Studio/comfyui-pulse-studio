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
WORKFLOW = WORKFLOW_DIR / "PulseSlate_Starter.json"
# §15 ships two starter graphs. The second is the first with the documented
# patch chain wired in, so every structural check runs over both -- a graph that
# loads as a silently disconnected mess is worse than one that fails to load.
SHIPPED_GRAPHS = (WORKFLOW, WORKFLOW_DIR / "PulseSlate_Starter_SpectrumSage.json")
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
            self.assertEqual(len(loaders), 5, "%s: expected 2 DiT + 1 text encoder + 2 VAE"
                                              % path.name)
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


class TestThePatchChainVariant(unittest.TestCase):
    """§15's second graph exists to teach §10: patch *upstream* of the director.

    Path B samples internally with the model handed to the node, so a patch on
    the node's output accelerates the short path and does nothing for a long
    timeline. A graph that demonstrates the wrong wiring teaches it to everyone
    who opens it, so the direction is asserted rather than eyeballed.
    """

    PATCH_TYPES = ("SpectrumApplyMiniMaxH3", "PathchSageAttentionKJ")

    def setUp(self):
        path = WORKFLOW_DIR / "PulseSlate_Starter_SpectrumSage.json"
        self.d = json.loads(path.read_text(encoding="utf-8"))
        self.by_id = {n["id"]: n for n in self.d["nodes"]}

    def _source_chain(self, dst_slot_name):
        """Walk backwards from a director input, returning the node types."""
        director = next(n for n in self.d["nodes"] if n["type"] == "PulseSlate")
        slot = next(i for i in director["inputs"] if i["name"] == dst_slot_name)
        by_link = {l[0]: l for l in self.d["links"]}
        chain, link_id = [], slot["link"]
        while link_id is not None:
            src = self.by_id[by_link[link_id][1]]
            chain.append(src["type"])
            upstream = src.get("inputs") or []
            link_id = upstream[0]["link"] if upstream else None
        return chain

    def test_both_model_inputs_arrive_through_the_documented_chain(self):
        for slot in ("model", "model_fl2va"):
            with self.subTest(input=slot):
                self.assertEqual(
                    self._source_chain(slot),
                    ["PathchSageAttentionKJ", "SpectrumApplyMiniMaxH3", "UNETLoader"],
                    "spec §10's chain is UNETLoader -> Spectrum -> Sage -> PulseSlate")

    def test_no_patch_node_sits_downstream_of_the_director(self):
        director = next(n for n in self.d["nodes"] if n["type"] == "PulseSlate")
        downstream = {l[3] for l in self.d["links"] if l[1] == director["id"]}
        for node_id in downstream:
            self.assertNotIn(self.by_id[node_id]["type"], self.PATCH_TYPES,
                             "a patch downstream of the director does nothing for path B")

    def test_spectrum_keeps_its_history_in_system_ram(self):
        """§18.1: this is what makes a 362-frame window fit on a 32GB card. It
        is not a speed knob, and an example shipping `vram` would say it is."""
        found = [n for n in self.d["nodes"] if n["type"] == "SpectrumApplyMiniMaxH3"]
        self.assertTrue(found, "the variant must actually contain a Spectrum node")
        for node in found:
            self.assertIn("system_ram", node["widgets_values"],
                          "%s must store history in system_ram" % node.get("title"))

    def test_the_variant_is_otherwise_the_starter_graph(self):
        """Same director, same widget values. The two graphs teach one thing
        each; a drifted copy would teach two."""
        starter = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        a = next(n for n in starter["nodes"] if n["type"] == "PulseSlate")
        b = next(n for n in self.d["nodes"] if n["type"] == "PulseSlate")
        self.assertEqual(a["widgets_values"], b["widgets_values"])


class TestWidgetOrderMatchesTheNode(unittest.TestCase):
    def _director(self):
        d = json.loads(WORKFLOW.read_text(encoding="utf-8"))
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
        self.assertEqual(stored[0], "2.0.0",
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
        """The starter text must produce a real plan, not just look plausible."""
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
        prompt = plan.windows[0].prompt
        for n in (1, 2, 3):
            self.assertIn("[Shot %d]" % n, prompt)
        self.assertIn("overall_soundscape:", prompt)

    def test_starter_prompts_reference_assets_by_name_not_number(self):
        """The graph ships as an example, so it must not teach the wrong habit."""
        stored = self._director()["widgets_values"]
        widgets = _required_widgets("PulseSlate")
        index = {name: n for n, name in enumerate(widgets)}
        for key in ("global_prompt", "shot_prompt"):
            text = stored[index[key]]
            for bad in ("<Picture ", "<Video ", "<Audio "):
                self.assertNotIn(bad, text, "%s hand-types a tag ordinal" % key)
            self.assertIn("@", text)


if __name__ == "__main__":
    unittest.main()
