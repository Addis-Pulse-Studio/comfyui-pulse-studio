"""What `pyproject.toml` promises the outside world.

Two of these are invariants the rest of the suite cannot see, because they are
about the artefact rather than the code:

  * the package version and SCHEMA_VERSION are one number wearing two hats, and
    they drifted once already -- 3.0.0 shipped with `version = "2.0.0"` still in
    the manifest, which would have published a wheel labelled 2.0.0 containing
    three nodes 2.0.0 never had;
  * NOTICE has to be in the distribution. It is the only copy of upstream's MIT
    copyright notice in this project, and MIT grants its permissions on the
    condition that the notice travels with every copy. A build that ships
    LICENSE alone is a licence violation that no amount of correct prose in the
    repository fixes.

tomllib is 3.11+. CI runs 3.11 and 3.12 as well as 3.10, so these are covered on
the matrix; hand-rolling a TOML parser to also cover 3.10 would be testing the
parser rather than the manifest.
"""

import sys
import unittest
from pathlib import Path

from comfyui_pulse_studio.constants import SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


@unittest.skipUnless(sys.version_info >= (3, 11), "tomllib is 3.11+")
class TestPyproject(unittest.TestCase):
    def setUp(self):
        import tomllib
        with PYPROJECT.open("rb") as fh:
            self.cfg = tomllib.load(fh)
        self.project = self.cfg["project"]

    def test_version_matches_schema_version(self):
        self.assertEqual(
            self.project["version"], SCHEMA_VERSION,
            "pyproject version and constants.SCHEMA_VERSION must be the same "
            "number; bump both or neither")

    def test_notice_is_a_declared_licence_file(self):
        files = self.project.get("license-files") or []
        self.assertIn("NOTICE", files,
                      "NOTICE carries upstream's MIT copyright notice and must "
                      "ship with the distribution -- see NOTICE")
        self.assertIn("LICENSE", files)

    def test_declared_licence_files_exist(self):
        for name in self.project.get("license-files") or []:
            self.assertTrue((PROJECT_ROOT / name).is_file(),
                            "%s is declared but not in the tree" % name)

    def test_build_backend_is_pinned(self):
        # Absent this table pip picks setuptools' legacy backend and runs
        # flat-layout autodiscovery over a root that holds the GPL-3.0 reference
        # tree. .gitignore does not apply to a build.
        self.assertIn("build-system", self.cfg)
        self.assertEqual(self.cfg["build-system"]["build-backend"],
                         "setuptools.build_meta")

    def test_packages_are_listed_rather_than_discovered(self):
        tools = self.cfg.get("tool", {}).get("setuptools", {})
        self.assertIn("comfyui_pulse_studio", tools.get("packages", []))
        self.assertNotIn("tests", tools.get("packages", []))
        for name in tools.get("py-modules", []):
            self.assertTrue((PROJECT_ROOT / ("%s.py" % name)).is_file(),
                            "py-module %r has no source file" % name)

    def test_publisher_and_repository_are_the_confirmed_ones(self):
        # Both were unverified placeholders carried over from the pre-fork
        # manifest until 2026-08-09. Pinning them keeps a stale value from
        # publishing under someone else's namespace.
        self.assertEqual(self.cfg["tool"]["comfy"]["PublisherId"], "addis-pulse")
        self.assertEqual(self.project["urls"]["Repository"],
                         "https://github.com/Addis-Pulse-Studio/comfyui-addis-pulse")


if __name__ == "__main__":
    unittest.main()
