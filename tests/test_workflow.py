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
# Four graphs ship, as of 2026-08-17. §15.4 and §15.9 asked for three -- the short
# path on this node's own sampler, the long path, and the long path with the patch
# chain -- and 3.0.0 released with all three. The short-path and Spectrum+Sage
# graphs were dropped on 2026-08-11, leaving the long path alone; the short path
# came back as PulseSlate_Single.json, with a cast-and-references graph and a
# retake graph beside it. The Spectrum+Sage variant did not come back and is not
# expected to: it needed three third-party packs installed or it opened red.
#
# Every structural check runs over the tuple rather than over any single path, so
# adding or putting back a graph is a one-line change here. A graph that loads as
# a silently disconnected mess is worse than one that fails to load.
#
# That promise was only half true until 2026-08-17: three tests iterated a fresh
# `(LONG_FORM,)` literal and two read LONG_FORM directly, so a restored graph would
# have skipped exactly the assertions written to catch a bad one. They all read the
# tuple now -- and the day the tuple grew, the two assertions still written around
# the long-form graph's own shape (five loaders, a director in every graph) failed
# on the graphs they were meant to cover. Both are stated as rules below.
LONG_FORM = WORKFLOW_DIR / "PulseSlate_LongForm.json"
CAST = WORKFLOW_DIR / "PulseSlate_Cast.json"
RETAKE = WORKFLOW_DIR / "PulseSlate_Retake.json"
SINGLE = WORKFLOW_DIR / "PulseSlate_Single.json"
SHIPPED_GRAPHS = (LONG_FORM, CAST, RETAKE, SINGLE)
ASSET_DIR = WORKFLOW_DIR / "assets"
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


def _first(document, node_type):
    """The first node of a type, or None. Not every shipped graph has a director.

    PulseRetake takes a finished film as an IMAGE input and compiles nothing, so a
    retake graph carries no PulseSlate at all. Returning None rather than raising
    lets the director-specific tests run over the whole tuple and skip the graphs
    that have no subject -- which keeps adding a graph a one-line change. Every
    such test asserts it found at least one director, so a rename cannot turn the
    whole class into a silent no-op.
    """
    return next((n for n in document["nodes"] if n["type"] == node_type), None)


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
            self.assertTrue(loaders, "%s loads no model at all" % path.name)
            for node in loaders:
                with self.subTest(workflow=path.name, node=node["type"]):
                    value = node["widgets_values"][0]
                    self.assertTrue(value.startswith("minimax\\"),
                                    "%s: %r is not in the minimax\\ subfolder the README "
                                    "documents" % (node["type"], value))
                    self.assertTrue(value.endswith(".safetensors"), value)

    def test_every_graph_loads_one_text_encoder_and_both_vaes(self):
        """The loader count was asserted as a flat `5` until 2026-08-17.

        That was the long-form graph's own shape written down as a rule, and the
        day three more graphs joined the tuple it failed on the first one: a graph
        that never touches the anchored branch loads one DiT checkpoint, not two.

        What is actually true of every path: one text encoder, and two VAEs --
        H3 decodes picture and audio through separate ones, so a graph carrying a
        single VAELoader has one of them wired to the wrong socket.
        """
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                types = [n["type"] for n in d["nodes"]]
                self.assertEqual(types.count("CLIPLoader"), 1,
                                 "%s: expected exactly one text encoder" % path.name)
                self.assertEqual(types.count("VAELoader"), 2,
                                 "%s: expected the video VAE and the audio VAE"
                                 % path.name)
                self.assertIn(types.count("UNETLoader"), (1, 2),
                              "%s loads %d DiT checkpoints; a graph uses the "
                              "reference branch, the anchored branch, or both"
                              % (path.name, types.count("UNETLoader")))

    def test_the_two_dit_sockets_never_share_a_loader(self):
        """`model` and `model_fl2va` are two different files, not two sockets for
        one checkpoint.

        ref2va and fl2va have disjoint inputs -- the anchored branch takes
        first/last frames where the reference branch takes references -- so one
        loader feeding both sockets is a graph that reads as correct and fails at
        sampling time. Cheap to write by accident when duplicating a wire.
        """
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                by_id = {n["id"]: n for n in d["nodes"]}
                sources = {}
                for link_id, src, _src_slot, dst, dst_slot, _type in d["links"]:
                    name = (by_id[dst].get("inputs") or [])[dst_slot].get("name")
                    if name in ("model", "model_fl2va"):
                        sources.setdefault(dst, {})[name] = src
                for dst, wired in sources.items():
                    if len(wired) < 2:
                        continue
                    self.assertNotEqual(
                        wired["model"], wired["model_fl2va"],
                        "%s: %s takes both DiT sockets from loader %s -- they are "
                        "different checkpoints" % (path.name, by_id[dst]["type"],
                                                   wired["model"]))


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

    PACK_NODES = ("PulseSlate", "PulseShot", "PulseRender", "PulseRetake",
                  "PulseStill", "PulseBench")

    def test_every_shipped_graph_demonstrates_a_node_from_this_pack(self):
        """This asked for `PulseSlate` in every graph until 2026-08-17.

        The retake graph has no director and never will: `PulseRetake` starts
        from a finished film rather than from a timeline, which is the whole point
        of it. What the check is for is unchanged -- a file in
        example_workflows/ that demonstrates nothing in this pack is one that got
        there by accident, and it ships to every user either way.
        """
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                types = {n["type"] for n in d["nodes"]}
                self.assertTrue(types & set(self.PACK_NODES),
                                "%s carries no node from this pack: %r"
                                % (path.name, sorted(types)))

    def test_the_director_is_in_at_least_one_shipped_graph(self):
        """The rule above is loose enough that a set of graphs could satisfy it
        without ever showing the node the pack is named for."""
        carriers = [path.name for path in SHIPPED_GRAPHS
                    if "PulseSlate" in {n["type"] for n
                                        in json.loads(path.read_text(encoding="utf-8"))
                                        ["nodes"]}]
        self.assertTrue(carriers, "no shipped graph carries PulseSlate")

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
        checked = 0
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                by_id = {n["id"]: n for n in d["nodes"]}
                slate = _first(d, "PulseSlate")
                render = _first(d, "PulseRender")
                if slate is None or render is None:
                    continue  # a retake graph, or the single-window graph's own sampler
                checked += 1
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
        self.assertTrue(checked, "no shipped graph pairs a PulseSlate with a "
                                 "PulseRender -- the long path is unasserted")

    def test_every_shot_node_is_actually_wired(self):
        """A shipped graph with a floating PulseShot renders a fraction of the
        film it appears to describe, and says nothing about it.

        This shipped with only shot 1 connected: three shot nodes on the canvas,
        one 6-second window rendered, no warning anywhere -- because a shot that
        is not connected is not a shot, it is a node sitting on a canvas.
        """
        seen = 0
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                shots = [n for n in d["nodes"] if n["type"] == "PulseShot"]
                seen += len(shots)
                wired = {l[1] for l in d["links"] if l[5] == "PULSE_SHOT"}
                for node in shots:
                    self.assertIn(node["id"], wired,
                                  "%s: PulseShot %s is on the canvas but wired to "
                                  "nothing" % (path.name, node.get("title", node["id"])))
        # A retake graph legitimately has no shots, so the "needs shot nodes"
        # assertion moved up here: some graph must exercise them.
        self.assertTrue(seen, "no shipped graph carries a PulseShot")

    def test_no_shipped_graph_has_a_node_wired_to_nothing(self):
        """Notes aside, a node with no links at all reads as a mistake."""
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                linked = {l[1] for l in d["links"]} | {l[3] for l in d["links"]}
                floating = [n.get("title") or n["type"] for n in d["nodes"]
                            if n["type"] != "MarkdownNote" and n["id"] not in linked]
                self.assertEqual(floating, [], "%s: %r" % (path.name, floating))

    def test_no_shipped_graph_wires_a_checkpoint_nothing_samples_with(self):
        """§18.1. A DiT checkpoint on the canvas is ~20 GB of resident memory
        whether or not a window ever samples through it.

        `model_fl2va` was wired into the long-form graph until 2026-08-17 beside a
        `continuity` of `none`, which is precisely the combination that never
        reaches the anchored branch: continuity is what maps a continuation window
        onto fl2va, and `none` leaves every window on the reference checkpoint.
        Harmless-looking, and on a 32 GB box with Spectrum offloading history to
        system RAM it is 20 GB the render wanted. The pack says so at queue time
        now, which made the flagship example ship a graph its own report told the
        user to change.

        The rule, stated so a graph cannot drift back into it: wire fl2va only
        where something uses it -- a retake, a continuity mode that pins a frame,
        or a shot with an anchor on it.
        """
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                by_id = {n["id"]: n for n in d["nodes"]}
                wired = [by_id[l[3]]["type"] for l in d["links"]
                         if (by_id[l[3]].get("inputs") or [])[l[4]].get("name")
                         == "model_fl2va"]
                if not wired:
                    continue

                # PulseRetake pins the frame either side of the cut, so the
                # anchored branch is the only one it can use.
                used = _first(d, "PulseRetake") is not None

                slate = _first(d, "PulseSlate")
                if slate is not None:
                    index = {name: n for n, name
                             in enumerate(_required_widgets("PulseSlate"))}
                    # Everything after seed is shifted one slot by
                    # control_after_generate.
                    used = used or slate["widgets_values"][index["continuity"] + 1] != "none"

                anchored = {"first_frame", "last_frame"}
                for node in d["nodes"]:
                    if node["type"] != "PulseShot":
                        continue
                    used = used or any(slot.get("link") is not None
                                       and slot.get("name") in anchored
                                       for slot in node.get("inputs") or [])

                self.assertTrue(
                    used,
                    "%s wires model_fl2va into %s, and nothing in the graph "
                    "reaches the anchored branch: no retake, no continuity mode "
                    "that pins a frame, no shot anchor. That is ~20 GB resident "
                    "for a checkpoint no window samples with, and the pack now "
                    "says so in the report." % (path.name, ", ".join(sorted(set(wired)))))

    def test_no_shipped_graph_references_an_asset_it_does_not_ship(self):
        """A shipped graph may only name a file a new user actually has.

        The starter's bin pointed at example_character_*.png, which a new user has
        not got. Those assets are dropped before tags are assigned, and every
        `@Image1` in the prompt is then correctly reported as unresolved -- which
        is right, but is not what a graph demonstrating the render loop should
        open on.

        The rule is "references only what ships", not "the bin is empty". A graph
        whose subject *is* the asset bin has to carry assets to show anything, so
        the placeholders in example_workflows/assets/ are installed into ComfyUI's
        input/ on load and a bin entry naming one of those is legitimate. A bin
        entry naming anything else is the old bug returning.
        """
        shipped = {p.name for p in ASSET_DIR.glob("*")} if ASSET_DIR.is_dir() else set()
        for path in SHIPPED_GRAPHS:
            with self.subTest(workflow=path.name):
                d = json.loads(path.read_text(encoding="utf-8"))
                slate = _first(d, "PulseSlate")
                if slate is None:
                    continue
                widgets = _required_widgets("PulseSlate")
                index = {name: n for n, name in enumerate(widgets)}
                document = json.loads(slate["widgets_values"][index["timeline_data"]])
                for asset in document["assets"]:
                    self.assertIn(
                        asset.get("file"), shipped,
                        "%s: bin asset %r names %r, which the pack does not ship"
                        % (path.name, asset.get("name"), asset.get("file")))
                if not document["assets"]:
                    self.assertNotIn(
                        "@", slate["widgets_values"][index["global_prompt"]],
                        "an empty bin cannot resolve an @reference")

    def test_the_long_form_graph_still_opens_on_an_empty_bin(self):
        """LongForm demonstrates the render loop, not the bin.

        Kept as its own assertion when the rule above was widened: the graph a new
        user opens to understand windows and carry-over should not also be asking
        them to reason about references.
        """
        d = json.loads(LONG_FORM.read_text(encoding="utf-8"))
        slate = _first(d, "PulseSlate")
        index = {name: n for n, name in enumerate(_required_widgets("PulseSlate"))}
        document = json.loads(slate["widgets_values"][index["timeline_data"]])
        self.assertEqual(document["assets"], [])

    # test_the_short_graph_has_no_render_node and
    # test_sigma_shift_is_upstream_of_the_sampler_not_the_director were dropped
    # with PulseSlate_Starter.json on 2026-08-11 and are restored below, because
    # PulseSlate_Single.json gives them a subject again: it is the only shipped
    # graph that samples the director's own conditioning, and the only one
    # carrying a model patch of any kind.
    #
    # §12.3's Sol-after-Sage ordering trap is still unasserted. It went with
    # TestThePatchChainVariant and the Spectrum+Sage graph the same day, and that
    # graph has not come back -- it needed three third-party packs installed or it
    # opened red. The rule still holds; it is documented in the README and in the
    # long-form graph's own note rather than enforced here.

    def _short_graphs(self):
        """Graphs that sample the director's own conditioning.

        Identified by a wired `positive` output rather than by filename, so the
        day a second short-path example lands it is covered without an edit here.
        """
        found = []
        for path in SHIPPED_GRAPHS:
            d = json.loads(path.read_text(encoding="utf-8"))
            slate = _first(d, "PulseSlate")
            if slate is None:
                continue
            if [l for l in d["links"] if l[1] == slate["id"] and l[2] == 1]:
                found.append((path, d, slate))
        self.assertTrue(found, "no shipped graph demonstrates the short path -- "
                               "PulseSlate's positive/latent outputs are unasserted")
        return found

    def test_the_short_graph_has_no_render_node(self):
        """§2.3 from the other side, and the reason the node was split.

        The two paths are alternatives. 2.x shipped both wired with one muted, and
        a 15-second timeline that split into two windows sampled internally *and*
        re-sampled its last window through the still-live short branch, saving 7
        seconds of video as the whole film without one error anywhere.
        """
        for path, d, _slate in self._short_graphs():
            with self.subTest(workflow=path.name):
                self.assertIsNone(
                    _first(d, "PulseRender"),
                    "%s samples the director's conditioning and also carries a "
                    "render node; they are alternatives, not a pair" % path.name)

    def test_sigma_shift_is_upstream_of_the_sampler_not_the_director(self):
        """§1.1. A model patch reaches the sampler by being applied to the model
        the sampler is handed -- never by being routed through this pack.

        `PulseSlate` lost its MODEL output in 3.0.0, so the original mistake is no
        longer expressible; what is still expressible is a sigma-shift node wired
        somewhere nothing samples through, which costs nothing at load time and
        renders on the wrong flow schedule.
        """
        for path, d, _slate in self._short_graphs():
            with self.subTest(workflow=path.name):
                shift = _first(d, "MiniMaxH3SigmaShift")
                self.assertIsNotNone(
                    shift, "%s: the short path samples directly, so it needs the "
                           "flow schedule H3 was trained on" % path.name)
                by_id = {n["id"]: n for n in d["nodes"]}
                upstream = {by_id[l[1]]["type"] for l in d["links"]
                            if l[3] == shift["id"]}
                self.assertEqual(upstream, {"UNETLoader"},
                                 "%s: the sigma shift takes its model from %r"
                                 % (path.name, sorted(upstream)))
                downstream = {by_id[l[3]]["type"] for l in d["links"]
                              if l[1] == shift["id"]}
                self.assertFalse(
                    downstream & {"PulseSlate", "PulseRender", "PulseStill",
                                  "PulseRetake"},
                    "%s: a model patch routed through this pack -- it belongs "
                    "between the loader and the sampler" % path.name)
                self.assertTrue(
                    downstream & {"BasicGuider", "BasicScheduler", "KSampler",
                                  "SamplerCustomAdvanced"},
                    "%s: the sigma shift feeds %r, none of which samples"
                    % (path.name, sorted(downstream)))


class TestWidgetOrderMatchesTheNode(unittest.TestCase):
    def _directors(self):
        """(path, PulseSlate node) for every shipped graph that has one.

        Read from the long-form graph alone until 2026-08-17; the check is on the
        director's widget *array*, which is the same shape in every graph, so it
        runs over all of them now. A retake graph has no director and drops out
        here rather than at each call site.
        """
        found = []
        for path in SHIPPED_GRAPHS:
            document = json.loads(path.read_text(encoding="utf-8"))
            slate = _first(document, "PulseSlate")
            if slate is not None:
                found.append((path, slate))
        self.assertTrue(found, "no shipped graph carries a PulseSlate")
        return found

    def test_widget_count_matches_input_types(self):
        widgets = _required_widgets("PulseSlate")
        for path, slate in self._directors():
            with self.subTest(workflow=path.name):
                stored = slate["widgets_values"]
                # seed carries control_after_generate, one extra slot.
                self.assertEqual(
                    len(stored), len(widgets) + 1,
                    "%s stores %d values but the node declares %d widgets (+1 for "
                    "control_after_generate).\nnode order: %r"
                    % (path.name, len(stored), len(widgets), widgets))

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
        for path, slate in self._directors():
            with self.subTest(workflow=path.name):
                self.assertEqual(
                    slate["widgets_values"][0], "3.0.0",
                    "slot 0 must carry the schema version the file was saved in")

    def test_stored_values_land_on_the_right_widgets(self):
        """Spot-check the values whose type makes a shift obvious."""
        widgets = _required_widgets("PulseSlate")
        index = {name: n for n, name in enumerate(widgets)}
        for path, slate in self._directors():
            with self.subTest(workflow=path.name):
                stored = slate["widgets_values"]
                self.assertIsInstance(stored[index["global_prompt"]], str)
                self.assertIsInstance(stored[index["shot_prompt"]], str)
                self.assertEqual(stored[index["sampler_name"]], "res_multistep")
                self.assertEqual(stored[index["scheduler"]], "simple")
                self.assertEqual(stored[index["cfg"]], 1.0)
                # Everything after seed is shifted one slot by control_after_generate.
                self.assertEqual(stored[index["seed"] + 1], "fixed")
                for name, expected in (("resize_method", "crop"),
                                       ("carry_mode", "image"),
                                       ("ref_image_size", "match")):
                    self.assertEqual(stored[index[name] + 1], expected,
                                     "%s did not land on its own slot" % name)
                # Checked against the vocabulary rather than one value: a graph
                # demonstrating shot_aligned windows is still a correct graph.
                self.assertIn(stored[index["partition_strategy"] + 1],
                              ("balanced", "fill", "shot_aligned"),
                              "partition_strategy did not land on its own slot")

    def test_the_stored_canvas_matches_the_preset_the_graph_selects(self):
        """The canvas is derived, so a stale width/height is invisible on the face.

        aspect_ratio is a label; resolution_for is what actually sizes the latent.
        Asserting only the label let 3.0.0 ship graphs storing 1344x736 for a
        preset that resolves to 1344x768 -- the widgets disagreed with the node and
        nothing said so, because the two are only compared at queue time.
        """
        from comfyui_pulse_studio.canvas import ASPECT_RATIOS, resolution_for

        widgets = _required_widgets("PulseSlate")
        index = {name: n for n, name in enumerate(widgets)}
        for path, slate in self._directors():
            with self.subTest(workflow=path.name):
                stored = slate["widgets_values"]
                aspect = stored[index["aspect_ratio"]]
                self.assertIn(aspect, ASPECT_RATIOS,
                              "%s selects %r, which is not a preset" % (path.name, aspect))
                width, height = resolution_for(aspect, stored[index["width"]],
                                               stored[index["height"]])
                self.assertEqual(
                    (stored[index["width"]], stored[index["height"]]), (width, height),
                    "%s stores %dx%d but %r resolves to %dx%d -- re-save the graph"
                    % (path.name, stored[index["width"]], stored[index["height"]],
                       aspect, width, height))

    def test_timeline_data_is_valid_json_in_schema_2_form(self):
        widgets = _required_widgets("PulseSlate")
        self.assertEqual(widgets[1], "timeline_data")
        for path, slate in self._directors():
            with self.subTest(workflow=path.name):
                document = json.loads(slate["widgets_values"][1])
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
        index = {name: n for n, name in enumerate(widgets)}
        for path, slate in self._directors():
            with self.subTest(workflow=path.name):
                stored = slate["widgets_values"]
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

        This does not assert that an `@name` is present. Most shipped graphs open
        on an empty bin, so there is nothing for one to resolve to; the graph that
        does carry a bin is covered by
        test_no_shipped_graph_references_an_asset_it_does_not_ship instead.
        """
        checked = 0
        for path in SHIPPED_GRAPHS:
            d = json.loads(path.read_text(encoding="utf-8"))
            by_type = {}
            for node in d["nodes"]:
                by_type.setdefault(node["type"], []).append(node)

            for node_type, keys in (("PulseSlate", ("global_prompt", "shot_prompt")),
                                    ("PulseShot", ("label", "visual", "audio_line")),
                                    ("PulseRetake", ("prompt",)),
                                    ("PulseStill", ("prompt",))):
                index = {name: n for n, name
                         in enumerate(_required_widgets(node_type))}
                for node in by_type.get(node_type, []):
                    for key in keys:
                        text = node["widgets_values"][index[key]]
                        for bad in ("<Picture ", "<Video ", "<Audio "):
                            self.assertNotIn(
                                bad, text,
                                "%s: %s %s hand-types a tag ordinal"
                                % (path.name, node_type, key))
                        checked += 1

        # Guards the loop above: a renamed node type would silently check nothing.
        self.assertGreaterEqual(checked, 11, "expected the director's two text "
                                             "boxes plus three fields on each of "
                                             "three PulseShot nodes")


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
        removing one is forbidden outright by the slot contract.

        One spare slot is legitimate. The js table already carries
        `control_after_generate`, so on PulseSlate the spare is `ps_asset_bin`, the
        panel's DOM widget: declared `serialize: false` in js/pulse_slate.js, and
        serialised anyway by frontend 1.49.6. Harmless only because it sits after
        every INPUT_TYPES widget -- which is precisely the invariant
        `checkWidgetOrder` now enforces, because the next widget appended to
        PulseSlate would otherwise land in front of it and saved files would feed
        the bin's JSON into it.
        """
        tables = self._tables()
        for path in WORKFLOW_DIR.glob("*.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for node in doc.get("nodes", []):
                names = tables.get(node.get("type"))
                if names is None:
                    continue
                stored = node.get("widgets_values") or []
                with self.subTest(workflow=path.name, node=node["type"], id=node["id"]):
                    self.assertLessEqual(len(stored), len(names) + 1,
                                         "%s node %s stores %d values for %d widgets"
                                         % (path.name, node["id"], len(stored), len(names)))

    def test_a_serialised_asset_bin_slot_stays_at_the_end(self):
        """If the bin's dead slot is present, it is last and it is the bin.

        A saved PulseSlate carries either 25 values (the js table, which already
        includes control_after_generate) or 26, the 26th being ps_asset_bin's
        value. Anything else in that position means a widget was inserted rather
        than appended, and every value past the insertion point is now off by one
        in a file that still loads without complaint.
        """
        names = self._tables()["PulseSlate"]
        for path in WORKFLOW_DIR.glob("*.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for node in doc.get("nodes", []):
                if node.get("type") != "PulseSlate":
                    continue
                stored = node.get("widgets_values") or []
                with self.subTest(workflow=path.name, id=node["id"]):
                    self.assertIn(len(stored), (len(names), len(names) + 1),
                                  "%s node %s stores %d values; expected %d or %d"
                                  % (path.name, node["id"], len(stored),
                                     len(names), len(names) + 1))
                    if len(stored) == len(names) + 1:
                        trailing = stored[-1]
                        self.assertNotIsInstance(
                            trailing, (int, float, bool),
                            "%s node %s: the trailing slot holds %r, which is a "
                            "widget value, not the asset bin's dead slot -- a "
                            "widget was inserted rather than appended"
                            % (path.name, node["id"], trailing))


if __name__ == "__main__":
    unittest.main()
