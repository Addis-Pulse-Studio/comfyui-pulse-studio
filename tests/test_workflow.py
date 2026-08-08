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
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
WORKFLOW = PROJECT_ROOT / "example_workflows" / "PulseSlate_Starter.json"
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


class TestWorkflowShipsAndParses(unittest.TestCase):
    def test_workflow_exists_and_is_valid_json(self):
        self.assertTrue(WORKFLOW.is_file(), "missing %s" % WORKFLOW)
        json.loads(WORKFLOW.read_text(encoding="utf-8"))

    def test_links_reference_real_nodes_and_slots(self):
        d = json.loads(WORKFLOW.read_text(encoding="utf-8"))
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
        d = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        by_id = {n["id"]: n for n in d["nodes"]}
        for link_id, src, src_slot, dst, dst_slot, _type in d["links"]:
            self.assertEqual(by_id[dst]["inputs"][dst_slot]["link"], link_id)
            self.assertIn(link_id, by_id[src]["outputs"][src_slot]["links"])

    def test_the_director_is_in_the_graph(self):
        d = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        types = {n["type"] for n in d["nodes"]}
        self.assertIn("PulseSlate", types)


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

    def test_the_two_prompt_boxes_come_first(self):
        widgets = _required_widgets("PulseSlate")
        self.assertEqual(widgets[:2], ["global_prompt", "shot_prompt"],
                         "the prompt boxes must be the first widgets on the node face")

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

    def test_timeline_data_is_valid_json_and_last(self):
        widgets = _required_widgets("PulseSlate")
        self.assertEqual(widgets[-1], "timeline_data")
        stored = self._director()["widgets_values"]
        document = json.loads(stored[-1])
        self.assertIn("assets", document)

    def test_the_shipped_prompts_actually_compile(self):
        """The starter text must produce a real plan, not just look plausible."""
        from comfyui_pulse_studio.compiler import compile_timeline
        from comfyui_pulse_studio.widget_state import build_timeline
        widgets = _required_widgets("PulseSlate")
        stored = self._director()["widgets_values"]
        index = {name: n for n, name in enumerate(widgets)}
        timeline, _ = build_timeline(
            stored[-1],
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
