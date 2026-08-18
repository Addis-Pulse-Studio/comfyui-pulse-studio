"""The compiler report and packed-sequence analysis. Spec §9, §12.5, §12.7."""

import json
import os
import shutil
import tempfile
import unittest

from comfyui_pulse_studio.bench import format_table, group_by_fingerprint, load_manifests
from comfyui_pulse_studio.fingerprint import describe_patches, patch_fingerprint
from comfyui_pulse_studio.pulse_timeline import (
    build_timeline,
    global_block,
    ref_descriptor,
    shot_block,
    window_block,
)
from comfyui_pulse_studio.report import (
    build_report,
    distinct_packed_lengths,
    frame_rows,
    packed_rows,
    packed_signature,
)
from comfyui_pulse_studio.segcache import SegmentDecision


def document(windows=None, shots=None, refs=None, budget=None, warnings=()):
    shots = shots or [shot_block("s1", 0, label="The Delivery", visual="she walks in",
                                 duration_seconds=7.5)]
    windows = windows or [window_block(0, ["s1"], 362, width=1344, height=736, seed=42)]
    return build_timeline(global_block(style="noir"), refs or [], shots, windows,
                          budget or {"images": 0, "videos": 0, "audio": 0, "total": 0},
                          warnings=warnings)


class TestPackedGeometry(unittest.TestCase):
    """§9.7 -- row arithmetic taken from comfy/ldm/minimax/model.py, not invented.

    PackedLayout builds `latent_t * frame_rows` video rows and `audio_t * 2` audio
    rows, where `_frame_grid(h, w)` yields (h//2)*(w//2) rows for a latent frame of
    h = height//16, w = width//16 -- so frame_rows is (height//32)*(width//32).
    """

    def test_frame_rows_matches_cores_grid(self):
        self.assertEqual(frame_rows(1344, 736), (736 // 32) * (1344 // 32))
        self.assertEqual(frame_rows(1344, 736), 23 * 42)

    def test_packed_rows_grows_with_frame_count(self):
        short = packed_rows(window_block(0, [], 124, width=1344, height=736))
        long = packed_rows(window_block(0, [], 362, width=1344, height=736))
        self.assertGreater(long, short)

    def test_a_uniform_plan_has_one_packed_length(self):
        windows = [window_block(i, [], 362, width=1344, height=736) for i in range(4)]
        self.assertEqual(len(distinct_packed_lengths(windows)), 1)

    def test_a_ragged_plan_has_more_than_one(self):
        windows = [window_block(0, [], 362, width=1344, height=736),
                   window_block(1, [], 192, width=1344, height=736)]
        self.assertEqual(len(distinct_packed_lengths(windows)), 2)

    def test_signature_keys_on_geometry_only(self):
        a = window_block(0, ["x"], 362, width=1344, height=736, seed=1)
        b = window_block(9, ["y"], 362, width=1344, height=736, seed=2)
        self.assertEqual(packed_signature(a), packed_signature(b))


class TestReportSections(unittest.TestCase):
    """§9.1-7 -- every numbered section is present and says something."""

    def report(self, **kwargs):
        return build_report(document(**kwargs.pop("doc", {})), **kwargs)

    def test_all_seven_sections_appear(self):
        text = build_report(document())
        for heading in ("1. WINDOWS", "2. ORDINAL MAP", "3. UNRESOLVED ALIASES",
                        "4. REFERENCE BUDGET", "5. UPSTREAM PATCH CHAIN",
                        "6. ESTIMATES", "7. PACKED SEQUENCE LENGTHS"):
            self.assertIn(heading, text)

    def test_the_window_table_carries_labels_frames_and_seed(self):
        text = build_report(document())
        self.assertIn("The Delivery", text)
        self.assertIn("362", text)
        self.assertIn("42", text)

    def test_cache_status_appears_only_when_decisions_are_supplied(self):
        """PulseSlate cannot know it -- the key needs a model fingerprint and the
        compiler holds no model."""
        self.assertNotIn("will render", build_report(document()))
        text = build_report(document(), decisions={
            0: SegmentDecision("render", "k", reason="no cached segment")})
        self.assertIn("will render", text)

        text = build_report(document(), decisions={
            0: SegmentDecision("reuse", "k", {"video_path": "x"}, reason="cached")})
        self.assertIn("will reuse", text)

    def test_the_ordinal_map_lists_global_references(self):
        text = build_report(document(
            refs=[ref_descriptor(1, "image", "Mimi", "bin", file="m.png", sha256="a"),
                  ref_descriptor(2, "image", "Kaleb", "bin", file="k.png", sha256="b")]))
        self.assertIn("Mimi", text)
        self.assertIn("<Picture 1>", text)
        self.assertIn("<Picture 2>", text)

    def test_scene_local_references_are_reported_against_their_own_shot(self):
        """§10 -- the same alias can legitimately be a different number in two
        shots, so a project-wide table would flatten what matters."""
        shot = shot_block("s1", 0, label="The Delivery", visual="x")
        shot["local_refs"] = [ref_descriptor(4, "image", "Prop1", "socket", sha256="z")]
        text = build_report(document(shots=[shot]))
        self.assertIn("Prop1", text)
        self.assertIn("scene-local", text)

    def test_unresolved_aliases_name_the_shot_they_were_written_in(self):
        shot = shot_block("s1", 0, label="The Delivery", visual="@Ghost waits")
        shot["unresolved_aliases"] = ["@Ghost"]
        text = build_report(document(shots=[shot]))
        self.assertIn("@Ghost", text)
        self.assertIn("The Delivery", text)

    def test_no_unresolved_aliases_says_so_plainly(self):
        self.assertIn("every @Alias resolved", build_report(document()))

    def test_the_budget_line_reports_against_the_documented_limits(self):
        text = build_report(document(budget={"images": 7, "videos": 2, "audio": 1,
                                             "total": 9}))
        self.assertIn("7/9 images", text)
        self.assertIn("9/12 files", text)

    def test_an_empty_patch_chain_is_warned_about_in_section_five(self):
        text = build_report(document(), patch_descriptor=describe_patches({}))
        self.assertIn("(none detected)", text)
        self.assertIn("Segment cache cannot protect", text)

    def test_a_detected_chain_is_listed_with_its_fingerprint(self):
        descriptor = describe_patches({"spectrum_h3_binding": {"history_storage": "system_ram"}})
        text = build_report(document(), patch_descriptor=descriptor,
                            patch_fingerprint=patch_fingerprint(descriptor))
        self.assertIn("spectrum", text)
        self.assertIn(patch_fingerprint(descriptor), text)

    def test_estimates_say_so_when_nothing_has_been_measured(self):
        text = build_report(document())
        self.assertIn("no completed segment has been timed", text)

    def test_estimates_use_measured_seconds_per_frame_when_available(self):
        text = build_report(document(), seconds_per_frame=1.5)
        self.assertIn("1.50s/frame", text)
        self.assertIn("measured on this box", text)

    def test_a_ragged_plan_warns_about_the_triton_sweep(self):
        windows = [window_block(0, ["s1"], 362, width=1344, height=736),
                   window_block(1, ["s1"], 192, width=1344, height=736)]
        text = build_report(document(windows=windows))
        self.assertIn("2 distinct packed sequence lengths", text)
        self.assertIn("autotunes", text)

    def test_a_uniform_plan_does_not_warn(self):
        windows = [window_block(i, ["s1"], 362, width=1344, height=736) for i in range(3)]
        self.assertNotIn("distinct packed sequence lengths", build_report(
            document(windows=windows)))

    def test_dry_run_is_stated_at_the_top(self):
        self.assertIn("DRY RUN", build_report(document(), dry_run=True))

    def test_warnings_become_their_own_section(self):
        text = build_report(document(warnings=["something to know"]))
        self.assertIn("8. WARNINGS", text)
        self.assertIn("something to know", text)

    def test_the_report_is_plain_text(self):
        """It is read in PreviewAny, which renders no markup."""
        text = build_report(document())
        self.assertNotIn("<", text.replace("<Picture", "").replace("<Video", "")
                         .replace("<Audio", ""))


class TestBench(unittest.TestCase):
    """§12.7 -- which chain is actually faster on this box."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="pulse-bench-")
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def write(self, name, segments, run_id="r"):
        path = os.path.join(self.directory, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"schema": 1, "run_id": run_id, "segments": segments}, handle)
        return path

    def test_groups_timings_by_patch_fingerprint(self):
        a = self.write("a", [
            {"status": "complete", "frames": 362, "render_seconds": 362.0,
             "patch_fingerprint": "aaaa", "peak_vram_bytes": 20 * 1024 ** 3},
            {"status": "complete", "frames": 362, "render_seconds": 362.0,
             "patch_fingerprint": "aaaa"},
        ])
        b = self.write("b", [
            {"status": "complete", "frames": 362, "render_seconds": 181.0,
             "patch_fingerprint": "bbbb"},
        ])
        manifests, problems = load_manifests([a, b])
        self.assertEqual(problems, [])
        groups = group_by_fingerprint(manifests)
        self.assertAlmostEqual(groups["aaaa"]["seconds_per_frame"], 1.0)
        self.assertAlmostEqual(groups["bbbb"]["seconds_per_frame"], 0.5)

    def test_the_warmup_window_is_counted_separately_not_averaged_in(self):
        path = self.write("a", [
            {"status": "complete", "frames": 100, "render_seconds": 900.0,
             "patch_fingerprint": "aaaa", "warmup": True},
            {"status": "complete", "frames": 100, "render_seconds": 100.0,
             "patch_fingerprint": "aaaa"},
        ])
        manifests, _ = load_manifests([path])
        group = group_by_fingerprint(manifests)["aaaa"]
        self.assertAlmostEqual(group["seconds_per_frame"], 1.0)
        self.assertEqual(group["warmup_segments"], 1)

    def test_incomplete_segments_are_ignored(self):
        path = self.write("a", [{"status": "pending", "frames": 100,
                                 "render_seconds": 10.0, "patch_fingerprint": "aaaa"}])
        manifests, _ = load_manifests([path])
        self.assertEqual(group_by_fingerprint(manifests), {})

    def test_a_missing_manifest_is_reported_not_raised(self):
        manifests, problems = load_manifests([os.path.join(self.directory, "nope")])
        self.assertEqual(manifests, [])
        self.assertEqual(len(problems), 1)

    def test_the_table_explains_itself_when_empty(self):
        text = format_table({}, [])
        self.assertIn("No completed, timed segments found", text)

    def test_the_table_sorts_fastest_first(self):
        groups = {"slow": {"segments": 1, "frames": 1, "seconds": 2.0,
                           "seconds_per_frame": 2.0, "warmup_segments": 0,
                           "warmup_seconds": 0.0, "peak_vram_bytes": 0, "runs": []},
                  "fast": {"segments": 1, "frames": 1, "seconds": 1.0,
                           "seconds_per_frame": 1.0, "warmup_segments": 0,
                           "warmup_seconds": 0.0, "peak_vram_bytes": 0, "runs": []}}
        text = format_table(groups)
        self.assertLess(text.index("fast"), text.index("slow"))


if __name__ == "__main__":
    unittest.main()


class TestShotsAcrossASeam(unittest.TestCase):
    """A shot compiled into two windows renders twice, and nothing said so.

    `Timeline.shots_in` is an overlap test, so a shot crossing a seam is compiled
    into both windows on purpose -- dropping it from the second would leave that
    window with no direction and render as a stall. But in the second window its
    timestamp is clamped to 0, so it reads as starting again, and its shot_id
    hashes into both window seeds. All of that was invisible.

    It is also close to unavoidable by hand: windows are 17k+5 frames each, and N
    of them sum to a grid total only when N is 1 mod 17, so with equal shot
    durations something almost always straddles.
    """

    def _document(self, window_shots):
        shots = sorted({sid for ids in window_shots for sid in ids})
        return document(
            shots=[shot_block(sid, i, label=sid.upper(), visual="x",
                              duration_seconds=5.0) for i, sid in enumerate(shots)],
            windows=[window_block(i, ids, 328) for i, ids in enumerate(window_shots)])

    def test_a_shot_in_two_windows_is_reported(self):
        text = build_report(self._document([["s1", "s2"], ["s2", "s3"]]))
        self.assertIn("SHOTS ACROSS A SEAM", text)
        self.assertIn("S2", text)
        self.assertIn("compiled into windows 0, 1", text)

    def test_shots_that_do_not_straddle_are_not_reported(self):
        text = build_report(self._document([["s1"], ["s2"]]))
        self.assertNotIn("SHOTS ACROSS A SEAM", text)

    def test_a_single_window_never_straddles(self):
        self.assertNotIn("SHOTS ACROSS A SEAM", build_report(document()))

    def test_every_straddler_is_listed_and_nothing_else_is(self):
        from comfyui_pulse_studio.report import straddling_shots

        lines = straddling_shots(self._document(
            [["s1", "s2"], ["s2", "s3"], ["s3", "s4"]]))
        listed = "\n".join(lines)
        self.assertIn("S2", listed)
        self.assertIn("S3", listed)
        # s1 and s4 appear in one window each and must not be accused.
        self.assertNotIn("S1", listed)
        self.assertNotIn("S4", listed)

    def test_the_note_says_it_is_not_an_error(self):
        """It is deliberate behaviour with a cost, not a bug to be fixed."""
        text = build_report(self._document([["s1", "s2"], ["s2", "s3"]]))
        self.assertIn("not an error", text)

    def test_a_shot_id_with_no_shot_block_still_reports(self):
        """Windows are the source of truth here; a missing block must not crash."""
        text = build_report(document(
            shots=[shot_block("s1", 0, label="Only", visual="x", duration_seconds=5.0)],
            windows=[window_block(0, ["s1", "ghost"], 328),
                     window_block(1, ["ghost"], 328)]))
        self.assertIn("SHOTS ACROSS A SEAM", text)


class TestPerWindowReferenceBudget(unittest.TestCase):
    """The project-wide meter is a peak, and the peak is not what constrains anyone.

    Continuation windows prepend synthetic carry-over references -- the previous
    window's last frame, its tail clip, its audio tail -- ahead of everything the
    user chose. A project sitting at 9/9 images renders window 1 and silently
    drops a reference from window 2. The document cannot express this: nodes.py
    collapses the per-window bins before building it, and adding the numbers to
    window_block would change the hashed document and therefore every cache key.
    """

    def _plan(self, shot_count=4, seconds_each=10.0):
        from comfyui_pulse_studio.assets import KIND_IMAGE, Asset, AssetBin
        from comfyui_pulse_studio.compiler import compile_timeline
        from comfyui_pulse_studio.timeline import Shot, Timeline

        timeline = Timeline()
        timeline.shots = [Shot("s%d" % i, seconds_each * i, seconds_each, "shot %d" % i)
                          for i in range(shot_count)]
        timeline.duration_seconds = seconds_each * shot_count
        timeline.assets = AssetBin([Asset("i1", KIND_IMAGE, name="Mimi", file="m.png",
                                          description="a woman in a red scarf")])
        return compile_timeline(timeline)

    def _document_for(self, plan):
        return document(
            shots=[shot_block("s%d" % i, i, label="Shot %d" % i, visual="x",
                              duration_seconds=10.0) for i in range(4)],
            windows=[window_block(w.index, w.shot_ids, w.frame_count)
                     for w in plan.windows],
            budget={"images": 1, "max_images": 9, "videos": 0, "max_videos": 3,
                    "audio": 0, "max_audio": 3, "total": 1, "max_total": 12})

    def test_carried_slots_are_shown_per_window(self):
        plan = self._plan()
        self.assertGreater(len(plan.windows), 1, "needed a continuation window")
        text = build_report(self._document_for(plan), plan=plan)
        self.assertIn("PER WINDOW, AFTER CARRY-OVER", text)
        self.assertIn("carried", text)

    def test_the_first_window_carries_nothing(self):
        plan = self._plan()
        text = build_report(self._document_for(plan), plan=plan)
        rows = [line for line in text.splitlines() if line.strip().startswith("0 ")]
        self.assertTrue(rows, text)
        self.assertNotIn("carried", rows[0])

    def test_a_single_window_plan_shows_no_table(self):
        """Nothing is carried, so the per-window view says nothing new."""
        plan = self._plan(shot_count=1, seconds_each=6.0)
        self.assertEqual(len(plan.windows), 1)
        self.assertNotIn("PER WINDOW, AFTER CARRY-OVER",
                         build_report(self._document_for(plan), plan=plan))

    def test_omitting_the_plan_drops_the_block_rather_than_guessing(self):
        self.assertNotIn("PER WINDOW, AFTER CARRY-OVER", build_report(document()))

    def test_the_block_explains_what_carried_means(self):
        plan = self._plan()
        text = build_report(self._document_for(plan), plan=plan)
        self.assertIn("claimed before any reference you chose", text)
