"""Hardening: import purity, the window floor, and Autogrow socket shape.

Written against the real implementations rather than mock data. A test that
asserts `[362, 38][-1] < 124` proves only that Python compares integers; these
call the actual partitioners and the actual socket builder, so they fail when
the shipping code regresses.

stdlib unittest, not pytest -- `run_tests.py` must work in a bare ComfyUI
environment with nothing pip-installed.
"""

import ast
import os
import unittest
from pathlib import Path

from omni_director.assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO, Asset, AssetBin
from omni_director.compiler import CarryPolicy, compile_timeline
from omni_director.constants import MAX_WINDOW_FRAMES, MIN_TRAINED_FRAMES
from omni_director.frames import is_on_grid, partition_windows, seconds_to_frames
from omni_director.sockets import (
    SocketGapError,
    assert_contiguous,
    check_socket_groups,
    drop_missing,
    socket_index,
)
from omni_director.timeline import Timeline

PROJECT_ROOT = Path(__file__).parent.parent
CORE = PROJECT_ROOT / "omni_director"

# The headless core must never reach for the runtime. This is what lets the whole
# compiler be tested with no GPU, no ComfyUI, and nothing pip-installed.
FORBIDDEN = {"torch", "comfy", "comfy_extras", "folder_paths", "nodes",
             "numpy", "PIL", "av", "server", "aiohttp"}


# ══════════════════════════════════════════════════════════════════════════
# TASK 6 — import purity, as a test rather than a one-time grep
# ══════════════════════════════════════════════════════════════════════════

class TestImportPurity(unittest.TestCase):
    def _core_files(self):
        found = []
        for root, _, files in os.walk(CORE):
            for name in files:
                if name.endswith(".py"):
                    found.append(Path(root) / name)
        return found

    def test_the_core_package_exists(self):
        """Guard against a false pass if the directory is ever moved."""
        self.assertTrue(CORE.is_dir(), "could not find %s" % CORE)
        self.assertTrue(self._core_files(), "no .py files found under %s" % CORE)

    def test_no_runtime_imports_anywhere_in_the_core(self):
        violations = []
        for path in self._core_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base = alias.name.split(".")[0]
                        if base in FORBIDDEN:
                            violations.append("%s:%d: import %s"
                                              % (path.relative_to(PROJECT_ROOT),
                                                 node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    # level > 0 is a relative import (from .frames import ...),
                    # which is always internal and never a runtime dependency.
                    if node.level:
                        continue
                    if node.module and node.module.split(".")[0] in FORBIDDEN:
                        violations.append("%s:%d: from %s import ..."
                                          % (path.relative_to(PROJECT_ROOT),
                                             node.lineno, node.module))
        self.assertEqual(violations, [], "headless core imports the runtime:\n  "
                                          + "\n  ".join(violations))

    def test_deferred_imports_are_caught_too(self):
        """A function-body `import torch` is still a violation.

        This is the shape the rule has to survive: phase 5's canvas will want to
        decode a preview frame, and the tempting fix is a local import inside one
        function. ast.walk descends into function bodies, so that is caught -- and
        this test asserts the walker really does descend.
        """
        source = "def f():\n    import torch\n    return torch\n"
        tree = ast.parse(source)
        found = [n for n in ast.walk(tree) if isinstance(n, ast.Import)]
        self.assertTrue(found, "ast.walk failed to descend into a function body")

    def test_core_imports_cleanly_with_nothing_installed(self):
        """The end the purity rule exists for: importing every core module must
        not require a single third-party package."""
        import importlib
        for path in self._core_files():
            if path.name == "__init__.py":
                continue
            importlib.import_module("omni_director.%s" % path.stem)


# ══════════════════════════════════════════════════════════════════════════
# TASK 5 — every emitted window >= 124 frames
# ══════════════════════════════════════════════════════════════════════════

class TestWindowFloor(unittest.TestCase):
    def _assert_sound(self, windows, total, note):
        self.assertGreaterEqual(sum(windows), total, note)
        for w in windows:
            self.assertTrue(is_on_grid(w), "%s: %d off-grid" % (note, w))
            self.assertLessEqual(w, MAX_WINDOW_FRAMES, "%s: %d over ceiling" % (note, w))
        if len(windows) > 1:
            self.assertGreaterEqual(
                min(windows), MIN_TRAINED_FRAMES,
                "%s: emitted %r, tail below the %d-frame floor"
                % (note, windows, MIN_TRAINED_FRAMES))

    def test_balanced_never_emits_a_sub_floor_window(self):
        for total in range(MIN_TRAINED_FRAMES, 4000, 7):
            w = partition_windows(total, MAX_WINDOW_FRAMES, policy="balanced")
            self._assert_sound(w, total, "balanced total=%d" % total)

    def test_fill_never_emits_a_sub_floor_window(self):
        for total in range(MIN_TRAINED_FRAMES, 4000, 7):
            w = partition_windows(total, MAX_WINDOW_FRAMES, policy="fill")
            self._assert_sound(w, total, "fill total=%d" % total)

    def test_both_partitioners_across_every_window_length(self):
        """The floor has to hold for user-chosen window lengths too, not just the
        362 default -- that is where cap and floor start fighting."""
        for cap in (124, 141, 192, 260, 300, 362):
            for total in range(MIN_TRAINED_FRAMES, 2000, 29):
                for policy in ("balanced", "fill"):
                    w = partition_windows(total, cap, policy=policy)
                    self._assert_sound(w, total, "%s cap=%d total=%d" % (policy, cap, total))

    def test_the_exact_case_from_the_brief(self):
        """[362, 38] must never escape. 400 frames is the canonical example."""
        for policy in ("balanced", "fill"):
            w = partition_windows(400, MAX_WINDOW_FRAMES, policy=policy)
            self.assertNotIn(38, w)
            self.assertGreaterEqual(min(w), MIN_TRAINED_FRAMES, "%s -> %r" % (policy, w))

    def test_fill_merges_a_short_tail_backwards(self):
        """A 39-frame tail behind a 192-frame window merges, because 231 snapped
        up to 238 still fits under the 362 ceiling."""
        notes = []
        total = 192 + 39
        w = partition_windows(total, 192, policy="fill", diagnostics=notes)
        self.assertEqual(len(w), 1, "expected a merge, got %r" % (w,))
        self.assertGreaterEqual(w[0], total)
        self.assertLessEqual(w[0], MAX_WINDOW_FRAMES)
        self.assertTrue(any("merged" in n for n in notes), notes)

    def test_fill_rebalances_when_a_merge_would_exceed_the_ceiling(self):
        """Merging is impossible when the predecessor is already at 362; the
        partition falls back to even windows rather than emitting an illegal one."""
        notes = []
        total = 362 * 2 + 39
        w = partition_windows(total, 362, policy="fill", diagnostics=notes)
        self._assert_sound(w, total, "fill ceiling case")
        self.assertLessEqual(max(w), MAX_WINDOW_FRAMES)

    def test_sub_floor_total_passes_through_as_one_short_window(self):
        """The documented exception."""
        notes = []
        w = partition_windows(seconds_to_frames(2.0), 362, diagnostics=notes)
        self.assertEqual(len(w), 1)
        self.assertLess(w[0], MIN_TRAINED_FRAMES)
        self.assertTrue(any("trained floor" in n for n in notes))

    def test_floor_outranks_a_conflicting_window_length(self):
        """200 frames cannot be two windows both >= 124. The floor wins and the
        override is reported rather than applied silently."""
        notes = []
        w = partition_windows(200, 124, policy="balanced", diagnostics=notes)
        self.assertEqual(len(w), 1)
        self.assertGreaterEqual(w[0], 200)
        self.assertTrue(any("cannot hold" in n for n in notes), notes)

    def test_the_compiler_inherits_the_floor(self):
        for seconds in (5.0, 16.0, 17.0, 20.0, 31.0, 45.0, 90.0):
            timeline = Timeline.from_dict({
                "shots": [{"id": "s", "start": 0, "duration": seconds, "prompt": "x"}],
                "duration_seconds": seconds,
            })
            plan = compile_timeline(timeline)
            if len(plan.windows) > 1:
                self.assertGreaterEqual(
                    min(w.frame_count for w in plan.windows), MIN_TRAINED_FRAMES,
                    "%.1fs -> %r" % (seconds, [w.frame_count for w in plan.windows]))


# ══════════════════════════════════════════════════════════════════════════
# TASK 4 — Autogrow dicts: contiguous, 0-based, gapless
# ══════════════════════════════════════════════════════════════════════════

class TestAutogrowShape(unittest.TestCase):
    def _bin(self):
        return AssetBin([
            Asset("i1", KIND_IMAGE, name="I1", file="i1.png"),
            Asset("i2", KIND_IMAGE, name="I2", file="i2.png"),
            Asset("i3", KIND_IMAGE, name="I3", file="i3.png"),
            Asset("v1", KIND_VIDEO, name="V1", file="v1.mp4", trim_end=5.0, include_audio=True),
            Asset("a1", KIND_AUDIO, name="A1", file="a1.wav"),
        ])

    def test_compiler_emits_a_dict_not_a_list(self):
        plan = compile_timeline(Timeline(
            assets=self._bin().to_list(),
            shots=[{"id": "s", "start": 0, "duration": 5, "prompt": "x"}],
            duration_seconds=5.0))
        groups = plan.windows[0].socket_kwargs()
        for name, mapping in groups.items():
            self.assertIsInstance(mapping, dict, "%s must be a dict, not a list" % name)

    def test_socket_groups_are_contiguous_and_zero_based(self):
        plan = compile_timeline(Timeline(
            assets=self._bin().to_list(),
            shots=[{"id": "s", "start": 0, "duration": 5, "prompt": "x"}],
            duration_seconds=5.0))
        check_socket_groups(plan.windows[0].socket_kwargs())

    def test_contiguity_holds_on_every_window_including_carry_over(self):
        timeline = Timeline(
            assets=self._bin().to_list(),
            shots=[{"id": "s", "start": 0, "duration": 40, "prompt": "x"}],
            duration_seconds=40.0, window_seconds=10.0)
        plan = compile_timeline(timeline, carry=CarryPolicy(mode="both", audio=True))
        self.assertGreater(len(plan.windows), 1)
        for window in plan.windows:
            check_socket_groups(window.socket_kwargs())

    def test_a_gap_is_detected(self):
        with self.assertRaises(SocketGapError) as ctx:
            assert_contiguous({"ref_image_0": 1, "ref_image_2": 2}, "ref_image")
        self.assertIn("not contiguous", str(ctx.exception))

    def test_out_of_order_keys_are_detected(self):
        """Every index is present but iteration order is wrong -- the tokenizer
        numbers by position, so this misnumbers just as badly as a gap."""
        from collections import OrderedDict
        bad = OrderedDict([("ref_image_1", 1), ("ref_image_0", 0)])
        with self.assertRaises(SocketGapError) as ctx:
            assert_contiguous(bad, "ref_image")
        self.assertIn("out of order", str(ctx.exception))

    def test_one_based_keys_are_detected(self):
        with self.assertRaises(SocketGapError):
            assert_contiguous({"ref_image_1": 1, "ref_image_2": 2}, "ref_image")

    def test_soundtrack_sockets_may_be_sparse_but_must_pair(self):
        """ref_video_audio_1 with no _0 is correct when only the second video
        carries sound -- but it must have a video to pair with."""
        check_socket_groups({
            "ref_videos": {"ref_video_0": "a", "ref_video_1": "b"},
            "ref_video_audios": {"ref_video_audio_1": "b_audio"},
        })
        with self.assertRaises(SocketGapError):
            check_socket_groups({
                "ref_videos": {"ref_video_0": "a"},
                "ref_video_audios": {"ref_video_audio_2": "orphan"},
            })

    def test_empty_groups_are_fine(self):
        check_socket_groups({"ref_images": {}, "ref_videos": None, "ref_audios": {}})

    def test_socket_index_parses(self):
        self.assertEqual(socket_index("ref_image_0"), 0)
        self.assertEqual(socket_index("ref_video_audio_2"), 2)


class TestMissingFilesDoNotCreateGaps(unittest.TestCase):
    """The real-world source of a gap: a file that fails to load at render time.

    Dropping the asset *before* the tag map is computed keeps ordinals dense by
    construction. Punching it out of the socket dict afterwards would leave a
    hole and silently shift every tag after it.
    """

    def _timeline(self):
        return Timeline(
            assets=[
                {"id": "i1", "kind": KIND_IMAGE, "name": "I1", "file": "present1.png"},
                {"id": "i2", "kind": KIND_IMAGE, "name": "I2", "file": "MISSING.png"},
                {"id": "i3", "kind": KIND_IMAGE, "name": "I3", "file": "present2.png"},
            ],
            shots=[{"id": "s", "start": 0, "duration": 5, "prompt": "@I3 walks"}],
            duration_seconds=5.0)

    def test_missing_asset_is_dropped_before_tags_are_assigned(self):
        timeline = self._timeline()
        notes = drop_missing(timeline, lambda f: "MISSING" not in f)
        self.assertEqual(len(notes), 1)
        self.assertIn("I2", notes[0])
        self.assertNotIn("i2", timeline.assets)

    def test_tags_stay_contiguous_after_a_drop(self):
        timeline = self._timeline()
        drop_missing(timeline, lambda f: "MISSING" not in f)
        plan = compile_timeline(timeline)
        window = plan.windows[0]
        check_socket_groups(window.socket_kwargs())
        self.assertEqual(window.tag_map.tag("i1"), "<Picture 1>")
        self.assertEqual(window.tag_map.tag("i3"), "<Picture 2>")

    def test_the_prompt_follows_the_renumbering(self):
        """@I3 was <Picture 3>; after the drop it is <Picture 2>, and the prompt
        text -- which never named a number -- follows without edit."""
        timeline = self._timeline()
        drop_missing(timeline, lambda f: "MISSING" not in f)
        prompt = compile_timeline(timeline).windows[0].prompt
        self.assertIn("<Picture 2>", prompt)
        self.assertNotIn("<Picture 3>", prompt)

    def test_the_drop_is_reported_not_silent(self):
        timeline = self._timeline()
        notes = drop_missing(timeline, lambda f: "MISSING" not in f)
        self.assertTrue(any("renumbered" in n for n in notes))

    def test_nothing_dropped_when_everything_loads(self):
        timeline = self._timeline()
        self.assertEqual(drop_missing(timeline, lambda f: True), [])
        self.assertEqual(len(timeline.assets), 3)


if __name__ == "__main__":
    unittest.main()
