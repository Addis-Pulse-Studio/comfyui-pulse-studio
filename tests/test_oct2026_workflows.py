"""The ten Oct-2026 graphs `tools/build_oct2026_workflows.py` writes.

They are built into the user's own folder rather than shipped in
example_workflows, so the structural battery in test_workflow.py never sees them.
Three things went wrong in the first build and are pinned here:

* two checkpoints in one film -- a reference window followed by First/Last-Frame
  windows that take no references, so the faces and voices contradicted
  themselves after window 1;
* references attached in two places -- Load Image / Pulse Character and Load
  Audio / Pulse Voice wired around the slate, beside its Asset Bin;
* wires that the frontend could land on the wrong socket (the fix for that is in
  js/ps_sockets.js; what is checked here is that every saved wire is typed for
  the socket it is saved on).
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, AssetBin
from comfyui_pulse_studio.constants import AUDIO_ROLE_LIP_SYNC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILDER = PROJECT_ROOT / "tools" / "build_oct2026_workflows.py"

#: Diffusion checkpoints and the adapters stacked on them. One UNETLoader per graph.
CHECKPOINT = "UNETLoader"
#: Everything that used to load or carry a reference outside the slate's bin.
OUTSIDE_THE_BIN = {"LoadAudio", "PulseVoice", "PulseCharacter"}


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_oct2026_workflows", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TheTenGraphs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.builder = _load_builder()
        cls._dir = tempfile.TemporaryDirectory()
        cls.paths = [Path(p) for p in cls.builder.build(cls._dir.name)]
        cls.graphs = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in cls.paths}

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _each(self):
        for name, graph in self.graphs.items():
            with self.subTest(workflow=name):
                yield name, graph

    @staticmethod
    def _slate(graph):
        return next(n for n in graph["nodes"] if n["type"] == "PulseSlate")

    def test_there_are_ten_and_none_mixes_two_engines(self):
        self.assertEqual(len(self.graphs), 10)
        for name in self.graphs:
            self.assertNotIn("TaoMate_Take", name)

    def test_every_graph_renders_on_one_checkpoint(self):
        for name, graph in self._each():
            loaders = [n for n in graph["nodes"] if n["type"] == CHECKPOINT]
            self.assertEqual(len(loaders), 1, [n["widgets_values"][0] for n in loaders])

    def test_model_fl2va_is_only_ever_the_same_chain_as_model(self):
        for name, graph in self._each():
            for target in ("PulseSlate", "PulseRender"):
                node = next(n for n in graph["nodes"] if n["type"] == target)
                source = {}
                for link in graph["links"]:
                    if link[3] == node["id"]:
                        source[node["inputs"][link[4]]["name"]] = link[1]
                if "model_fl2va" in source:
                    self.assertEqual(source["model_fl2va"], source["model"],
                                     "%s.model_fl2va is a second model" % target)
                    self.assertIn("Vertical", name)

    def test_no_reference_is_attached_outside_the_bin(self):
        for name, graph in self._each():
            self.assertFalse({n["type"] for n in graph["nodes"]} & OUTSIDE_THE_BIN)
            slate = self._slate(graph)
            wired = [i["name"] for i in slate["inputs"] if i.get("link") is not None]
            for socket in wired:
                self.assertFalse(socket.startswith(("refs.", "voices.", "cast.", "ref_")),
                                 socket)

    def test_every_saved_wire_is_typed_for_the_socket_it_is_saved_on(self):
        for name, graph in self._each():
            by_id = {n["id"]: n for n in graph["nodes"]}
            for link_id, src, src_slot, dst, dst_slot, link_type in graph["links"]:
                out = by_id[src]["outputs"][src_slot]
                inp = by_id[dst]["inputs"][dst_slot]
                self.assertEqual(inp["link"], link_id)
                self.assertIn(link_id, out["links"])
                self.assertEqual(out["type"], link_type)
                if inp["type"] != "*":
                    self.assertEqual(inp["type"], link_type,
                                     "%s.%s" % (by_id[dst]["type"], inp["name"]))

    def test_shot_n_is_wired_to_shot_socket_n(self):
        for name, graph in self._each():
            by_id = {n["id"]: n for n in graph["nodes"]}
            slate = self._slate(graph)
            for link in graph["links"]:
                if link[3] != slate["id"]:
                    continue
                socket = slate["inputs"][link[4]]["name"]
                if socket.startswith("shots.shot_"):
                    title = by_id[link[1]]["title"]
                    self.assertTrue(title.startswith("Shot %s " % socket.rsplit("_", 1)[1]),
                                    (socket, title))

    def test_the_bins_are_legal_and_every_line_has_an_owner_and_a_shot(self):
        for name, graph in self._each():
            slate = self._slate(graph)
            document = json.loads(slate["widgets_values"][1])
            bin_ = AssetBin.from_list(document["assets"])
            if "Vertical" in name:
                self.assertEqual(len(bin_), 0)
                continue
            pictures = {a.asset_id for a in bin_ if a.kind == KIND_IMAGE}
            self.assertTrue(pictures)
            for asset in bin_:
                if asset.kind == KIND_IMAGE:
                    self.assertTrue(asset.description, asset.name)
                    continue
                self.assertEqual(asset.kind, KIND_AUDIO)
                self.assertEqual(asset.audio_role, AUDIO_ROLE_LIP_SYNC)
                self.assertIn(asset.voice_of, pictures)
            if "Dialogue" in name:
                shot = self.builder.SHOT
                for asset in bin_.by_kind(KIND_AUDIO):
                    start = asset.audio_offset
                    end = start + (asset.trim_end - asset.trim_start)
                    index = int(round(start / shot))
                    self.assertAlmostEqual(start, index * shot, places=9,
                                           msg="%s starts off its cut" % asset.name)
                    self.assertLessEqual(end, (index + 1) * shot + 1e-9,
                                         "%s runs into the next shot" % asset.name)

    def test_a_media_file_names_your_files_and_nothing_else_changes(self):
        mine = {"take_a": "mine_a.wav", "take_b": "mine_b.wav", "narration": "mine.wav",
                "narration_seconds": 31.5, "key_1": "k1.png", "key_2": "k2.png"}
        with tempfile.TemporaryDirectory() as other:
            for path in map(Path, self.builder.build(other, mine)):
                graph = json.loads(path.read_text(encoding="utf-8"))
                default = self.graphs[path.name]
                self.assertEqual(graph["links"], default["links"], path.name)
                slate = self._slate(graph)
                files = {a["file"] for a in json.loads(slate["widgets_values"][1])["assets"]}
                files |= {n["widgets_values"][0] for n in graph["nodes"]
                          if n["type"] == "LoadImage"}
                self.assertFalse(files & {"take_a.wav", "take_b.wav", "narration.wav",
                                          "keyframe_1.png", "keyframe_2.png"}, path.name)
                self.assertTrue(files <= set(mine.values()) | {
                    self.builder.MEDIA["face_a"], self.builder.MEDIA["face_b"]}, files)

    def test_an_unknown_media_key_is_refused(self):
        with tempfile.TemporaryDirectory() as other:
            with self.assertRaises(ValueError):
                self.builder.build(other, {"take_c": "x.wav"})

    def test_rebuilding_writes_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as again:
            for path in self.builder.build(again):
                path = Path(path)
                self.assertEqual(path.read_bytes(), (Path(self._dir.name) / path.name).read_bytes(),
                                 path.name)


if __name__ == "__main__":
    unittest.main()
