"""Nothing private ships in an example workflow.

WHY THIS IS A TEST

Example workflows are the one place client work leaks into a public repository
without anyone deciding to publish it. A ComfyUI workflow stores the *filename*
of every reference image, video and audio clip the graph was last run with --
inside node widgets, and again inside this pack's `timeline_data` string, which
is a JSON document embedded in a JSON document and therefore invisible to a
casual read of the file.

The starter graph shipped with four `Generated Image August 07, 2026 - 8_25PM.jpg`
references buried in exactly that second layer. Those are client-facing assets.
Publishing the repo would have published their names.

So: the shipped workflows may only name placeholder assets the user supplies
themselves. This is a scan rather than a convention because the failure is
silent, the reviewer sees valid JSON either way, and the cost of being wrong is
paid in public.

Model weights are checked here too. They are ~21 GB each, carry the MiniMax H3
Community License, and are linked from the README rather than redistributed --
so a weight file appearing in the tree is always a mistake.
"""

import json
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
WORKFLOW_DIR = PROJECT_ROOT / "example_workflows"

# The shapes an AI image tool stamps on its exports, and the shapes a person
# stamps on their own files. Neither belongs in a published example.
PRIVATE_PATTERNS = (
    re.compile(r"Generated (?:Image|Audio|Video)\b", re.I),
    re.compile(r"\bIMG_\d{4}", re.I),
    re.compile(r"\bDSC\d{4}", re.I),
    re.compile(r"\bScreenshot\b", re.I),
    re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.I),   # an absolute Windows home path
    re.compile(r"/home/[^/\"]+/", re.I),             # an absolute POSIX home path
)

MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".wav", ".mp3")
WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".bin")

FILENAME_RE = re.compile(r'"([^"\\]*?\.(?:png|jpg|jpeg|webp|mp4|mov|wav|mp3))"', re.I)


def workflows():
    return sorted(WORKFLOW_DIR.glob("*.json"))


class TestTheScanHasInput(unittest.TestCase):
    def test_example_workflows_are_present(self):
        found = workflows()
        self.assertTrue(found, "no example workflows found to scan")
        self.assertIn("PulseSlate_Starter.json", {p.name for p in found})


class TestNoPrivateAssetsShip(unittest.TestCase):
    def test_no_workflow_names_a_private_asset(self):
        violations = []
        for path in workflows():
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                for pattern in PRIVATE_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        violations.append("%s:%d: %s" % (path.name, lineno, match.group(0)))
        self.assertEqual(violations, [],
                         "an example workflow names a private asset. Replace it with a "
                         "placeholder the user supplies (example_character_a.png):\n  "
                         + "\n  ".join(sorted(set(violations))))

    def test_every_media_filename_is_a_placeholder(self):
        """Including the ones inside the embedded timeline_data document.

        The bin's JSON is stored as a *string* widget value, so a filename in
        there survives any scan that only walks the outer object.
        """
        violations = []
        for path in workflows():
            for name in FILENAME_RE.findall(path.read_text(encoding="utf-8")):
                if not Path(name).name.lower().startswith("example_"):
                    violations.append("%s: %s" % (path.name, name))
        self.assertEqual(violations, [],
                         "example workflows may only reference example_* placeholder "
                         "media:\n  " + "\n  ".join(sorted(set(violations))))

    def test_no_absolute_or_escaping_media_paths(self):
        violations = []
        for path in workflows():
            for name in FILENAME_RE.findall(path.read_text(encoding="utf-8")):
                if name.startswith(("/", "\\")) or ".." in name or re.match(r"^[A-Za-z]:", name):
                    violations.append("%s: %s" % (path.name, name))
        self.assertEqual(violations, [], "media path escapes the ComfyUI input dir:\n  "
                                          + "\n  ".join(sorted(set(violations))))


class TestNoWeightsInTheTree(unittest.TestCase):
    def test_no_model_weights_are_committed(self):
        """~21 GB each, separately licensed, linked from the README not shipped."""
        found = []
        for suffix in WEIGHT_SUFFIXES:
            for path in PROJECT_ROOT.rglob("*" + suffix):
                if ".git" in path.parts:
                    continue
                found.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(found, [], "model weights must never be committed:\n  "
                                     + "\n  ".join(found))

    def test_no_stray_media_files_in_the_package(self):
        """A dropped reference image left in the repo root ships with the pack."""
        found = []
        for suffix in MEDIA_SUFFIXES:
            for path in PROJECT_ROOT.rglob("*" + suffix):
                if ".git" in path.parts or "_scratch" in path.parts:
                    continue
                found.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(found, [],
                         "unexpected media in the tree. Screenshots belong in the README's "
                         "assets and client material belongs in _scratch/:\n  "
                         + "\n  ".join(found))


class TestWorkflowsStayParseable(unittest.TestCase):
    def test_every_workflow_is_valid_json(self):
        for path in workflows():
            with self.subTest(workflow=path.name):
                json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
