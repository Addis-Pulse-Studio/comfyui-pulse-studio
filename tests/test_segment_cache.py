"""The segment cache: every acceptance bullet in spec §7.5.

WHAT THIS COVERS AND WHAT IT DOES NOT
-------------------------------------
The five bullets in §7.5 are all statements about *decisions*: given a timeline
and a run folder, which windows load from disk and which re-render. That decision
is made by exactly two functions -- `cache_key` and `plan_window` -- called in
that order, once per window, and `render.run` does nothing else to reach it. So
these tests drive those two directly, which is the real code path and not a
re-implementation of it.

What is deliberately not covered here is the tensor marshalling around them:
sampling, decoding and encoding need torch and a GPU, and the whole point of
keeping comfyui_pulse_studio/ import-pure is that the correctness properties do
not. The sampler is stubbed out of the picture entirely by testing the layer
below it.
"""

import json
import os
import shutil
import tempfile
import unittest

from comfyui_pulse_studio.pulse_timeline import (
    build_timeline,
    global_block,
    shot_block,
    text_shot_id,
    window_block,
    window_seed,
)
from comfyui_pulse_studio.segcache import (
    CACHE_FORCE,
    CACHE_REUSE_ONLY,
    Manifest,
    ReuseOnlyMiss,
    cache_key,
    cache_key_material,
    derive_run_id,
    plan_window,
    seconds_per_frame,
    segment_entry,
    segment_paths,
    segment_stem,
    stale_entries,
)

MODEL_FP = "0a1b2c3d4e5f6071"
PATCH_FP = "9988776655443322"


def make_timeline(shot_texts, windows_of=2, seed=1234, steps=20):
    """A document with `len(shot_texts)` shots grouped into windows of `windows_of`.

    Built with the same pure-core builders `PulseSlate._build_document` uses, so a
    change to the document's shape breaks these tests rather than silently
    changing what gets hashed.

    Labels are derived from the shot's own text rather than from its position,
    which is what a real project looks like: a label is the user's name for the
    shot ("The Delivery") and it travels with the shot when the shot moves. A
    positional label would re-key every shot below an insertion and would be
    testing the helper rather than the cache.
    """
    shots = [
        shot_block(text_shot_id(text[:16], text), i, label=text[:16],
                   visual=text, duration_seconds=7.5, resolved_prompt=text)
        for i, text in enumerate(shot_texts)
    ]
    windows = []
    for index, start in enumerate(range(0, len(shots), windows_of)):
        ids = [s["shot_id"] for s in shots[start:start + windows_of]]
        windows.append(window_block(
            index, ids, 362, seed=window_seed(seed, ids), steps=steps))
    return build_timeline(global_block(style="noir"), [], shots, windows,
                          {"images": 0, "videos": 0, "audio": 0, "total": 0})


def keys_of(timeline, model_fp=MODEL_FP, patch_fp=PATCH_FP):
    return [cache_key(timeline, w, model_fp, patch_fp) for w in timeline["windows"]]


def complete(manifest, timeline, key, window):
    """Write a segment's files and its manifest entry, as a finished render would."""
    paths = segment_paths(window["window_index"], key)
    for rel in paths.values():
        with open(os.path.join(manifest.directory, rel), "wb") as handle:
            handle.write(b"x")
    manifest.upsert(segment_entry(window, key, paths, render_seconds=400.0,
                                  patch_fingerprint=PATCH_FP))
    manifest.save("2026-08-08T00:00:00Z")


class CacheCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="pulse-cache-")
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def manifest(self):
        return Manifest.load(self.directory, run_id="r", node_version="3.0.0",
                             model_fingerprint=MODEL_FP, created_utc="2026-08-08T00:00:00Z")

    def render_all(self, timeline):
        """Simulate a complete first run."""
        manifest = self.manifest()
        for window, key in zip(timeline["windows"], keys_of(timeline)):
            complete(manifest, timeline, key, window)
        return manifest

    def decisions(self, timeline, cache_mode="auto"):
        manifest = self.manifest()
        return [plan_window(manifest, key, cache_mode, w["window_index"]).action
                for w, key in zip(timeline["windows"], keys_of(timeline))]


class TestResumeAfterCrash(CacheCase):
    """§7.5.1 -- kill at window 9 of 12, requeue: 0-8 load, only 9-11 render."""

    def test_partial_run_resumes_at_the_first_missing_window(self):
        timeline = make_timeline(["shot %d" % i for i in range(24)])
        self.assertEqual(len(timeline["windows"]), 12)

        manifest = self.manifest()
        for window, key in zip(timeline["windows"][:9], keys_of(timeline)[:9]):
            complete(manifest, timeline, key, window)

        actions = self.decisions(timeline)
        self.assertEqual(actions, ["reuse"] * 9 + ["render"] * 3)

    def test_a_manifest_entry_without_its_file_is_not_reused(self):
        """The manifest is a promise the bytes are on disk. It is verified."""
        timeline = make_timeline(["a", "b"])
        manifest = self.render_all(timeline)
        entry = manifest.segments[0]
        os.unlink(os.path.join(self.directory, entry["video_path"]))

        self.assertEqual(self.decisions(timeline), ["render"])


class TestEditingOneShot(CacheCase):
    """§7.5.2 -- edit shot 3's visual: only its window re-renders."""

    def test_only_the_window_holding_the_edited_shot_moves(self):
        texts = ["shot %d" % i for i in range(8)]
        timeline = make_timeline(texts)
        self.render_all(timeline)
        self.assertEqual(self.decisions(timeline), ["reuse"] * 4)

        texts[3] = "shot 3, but she turns away this time"
        edited = make_timeline(texts)
        self.assertEqual(self.decisions(edited), ["reuse", "render", "reuse", "reuse"])

    def test_inserting_a_shot_does_not_reroll_the_windows_it_did_not_touch(self):
        """§6. The failure this whole design exists to prevent.

        Position-derived seeds meant inserting one shot at the top changed the
        seed of every window after it, so an edit that added an establishing shot
        silently re-rendered -- and altered -- the entire rest of the film.
        """
        texts = ["shot %d" % i for i in range(8)]
        before = make_timeline(texts)
        after = make_timeline(["a brand new opening", "second new shot"] + texts)

        # Windows group in twos, so prepending exactly two shots leaves every
        # later window holding the same shot set. Their seeds must not move.
        seeds_before = [w["seed"] for w in before["windows"]]
        seeds_after = [w["seed"] for w in after["windows"]]
        self.assertEqual(seeds_before, seeds_after[1:])

        self.render_all(before)
        actions = self.decisions(after)
        self.assertEqual(actions[0], "render")          # the new window
        self.assertEqual(actions[1:], ["reuse"] * 4)    # everything else survived


class TestSettingsInvalidateEverything(CacheCase):
    """§7.5.3 and §7.5.4 -- a seed or steps change re-renders every window."""

    def test_changing_the_base_seed_re_renders_all(self):
        texts = ["shot %d" % i for i in range(8)]
        self.render_all(make_timeline(texts, seed=1234))
        self.assertEqual(self.decisions(make_timeline(texts, seed=9999)),
                         ["render"] * 4)

    def test_changing_steps_re_renders_all(self):
        texts = ["shot %d" % i for i in range(8)]
        self.render_all(make_timeline(texts, steps=20))
        self.assertEqual(self.decisions(make_timeline(texts, steps=30)),
                         ["render"] * 4)

    def test_changing_the_patch_chain_re_renders_all(self):
        """§12.4. The reason patch_fingerprint is mandatory rather than optional.

        Without it, a cache would hand back window 4 rendered dense and window 5
        rendered at tau=2.0 and call the result a film.
        """
        timeline = make_timeline(["a", "b", "c", "d"])
        manifest = self.manifest()
        for window, key in zip(timeline["windows"], keys_of(timeline)):
            complete(manifest, timeline, key, window)

        other = self.manifest()
        actions = [plan_window(other, key, "auto", i).action
                   for i, key in enumerate(keys_of(timeline, patch_fp="ffff000011112222"))]
        self.assertEqual(actions, ["render", "render"])


class TestNoChange(CacheCase):
    """§7.5.5 -- change nothing and requeue: nothing renders."""

    def test_requeue_is_a_no_op(self):
        timeline = make_timeline(["shot %d" % i for i in range(8)])
        self.render_all(timeline)
        self.assertEqual(self.decisions(timeline), ["reuse"] * 4)

    def test_cache_keys_are_stable_across_processes(self):
        """A key derived from dict ordering would empty the cache on a restart."""
        timeline = make_timeline(["a", "b"])
        again = json.loads(json.dumps(timeline))
        self.assertEqual(keys_of(timeline), keys_of(again))


class TestCacheModes(CacheCase):
    def test_force_rerender_ignores_a_complete_entry(self):
        timeline = make_timeline(["a", "b"])
        self.render_all(timeline)
        self.assertEqual(self.decisions(timeline, CACHE_FORCE), ["render"])

    def test_reuse_only_aborts_naming_the_first_missing_window(self):
        timeline = make_timeline(["shot %d" % i for i in range(8)])
        manifest = self.manifest()
        for window, key in zip(timeline["windows"][:2], keys_of(timeline)[:2]):
            complete(manifest, timeline, key, window)

        keys = keys_of(timeline)
        manifest = self.manifest()
        self.assertTrue(plan_window(manifest, keys[0], CACHE_REUSE_ONLY, 0).reuse)
        with self.assertRaises(ReuseOnlyMiss) as caught:
            plan_window(manifest, keys[2], CACHE_REUSE_ONLY, 2)
        self.assertIn("window 2", str(caught.exception))


class TestCacheKeyContent(unittest.TestCase):
    """§7.1 -- what is and is not in the key."""

    def test_the_key_is_not_derived_from_window_index(self):
        timeline = make_timeline(["same text", "same text", "same text", "same text"],
                                 windows_of=2)
        # Two windows holding shots with identical text still differ, because the
        # shot ids differ -- text_shot_id is content-derived and these shots have
        # the same content, so they collide by design. Assert on the material
        # instead: window_index appears nowhere in it.
        material = cache_key_material(timeline, timeline["windows"][1], MODEL_FP, PATCH_FP)
        flat = json.dumps(material)
        self.assertNotIn("window_index", flat)

    def test_every_field_the_spec_lists_is_present_in_order(self):
        timeline = make_timeline(["a", "b"])
        names = [pair[0] for pair in
                 cache_key_material(timeline, timeline["windows"][0], MODEL_FP, PATCH_FP)]
        self.assertEqual(names, [
            "global", "shots", "refs", "frames", "fps", "width", "height",
            "seed", "steps", "sampler", "scheduler", "cfg",
            "continuity_in", "continuity_out",
            "model_fingerprint", "patch_fingerprint", "node_version"])

    def test_an_empty_patch_fingerprint_is_refused(self):
        """§14.8. 'Nothing detected' still has a fingerprint; an empty one means
        the detector never ran, and that key would be a lie."""
        timeline = make_timeline(["a"])
        with self.assertRaises(ValueError):
            cache_key(timeline, timeline["windows"][0], MODEL_FP, "")

    def test_the_global_prompt_is_in_the_key(self):
        a = make_timeline(["a", "b"])
        b = make_timeline(["a", "b"])
        b["global"]["style"] = "shot on 16mm, blown highlights"
        self.assertNotEqual(keys_of(a), keys_of(b))


class TestRunId(unittest.TestCase):
    """§7.2 -- the same project resumes into the same folder."""

    def test_a_seed_change_keeps_the_same_run_folder(self):
        texts = ["a", "b", "c", "d"]
        self.assertEqual(derive_run_id(make_timeline(texts, seed=1)),
                         derive_run_id(make_timeline(texts, seed=2)))

    def test_a_content_change_moves_the_run_folder(self):
        self.assertNotEqual(derive_run_id(make_timeline(["a", "b"])),
                            derive_run_id(make_timeline(["a", "c"])))

    def test_run_id_is_twelve_hex_chars(self):
        run_id = derive_run_id(make_timeline(["a"]))
        self.assertEqual(len(run_id), 12)
        int(run_id, 16)


class TestManifestDurability(CacheCase):
    """§7.3, §7.4 -- the manifest survives, and never over-promises."""

    def test_round_trip(self):
        timeline = make_timeline(["a", "b"])
        manifest = self.render_all(timeline)
        reloaded = Manifest.load(self.directory)
        self.assertEqual(len(reloaded.segments), len(manifest.segments))
        self.assertEqual(reloaded.data["run_id"], "r")

    def test_a_corrupt_manifest_is_moved_aside_not_overwritten(self):
        """Those files may be hours of GPU time; the manifest is the only record
        of which of them are usable."""
        path = os.path.join(self.directory, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json at all")
        manifest = Manifest.load(self.directory, run_id="fresh")
        self.assertEqual(manifest.segments, [])
        self.assertTrue(os.path.exists(os.path.join(self.directory, "manifest.corrupt.json")))

    def test_the_write_is_atomic_and_leaves_no_temp_files(self):
        timeline = make_timeline(["a", "b"])
        self.render_all(timeline)
        leftovers = [n for n in os.listdir(self.directory) if n.startswith(".manifest-")]
        self.assertEqual(leftovers, [])

    def test_two_windows_with_identical_content_share_one_segment(self):
        manifest = self.manifest()
        timeline = make_timeline(["a", "b"])
        key = keys_of(timeline)[0]
        complete(manifest, timeline, key, timeline["windows"][0])
        complete(manifest, timeline, key, timeline["windows"][0])
        self.assertEqual(len(manifest.segments), 1)


class TestStaleEntries(CacheCase):
    """§7.4, §14.5 -- stale segments are reported, never deleted."""

    def test_entries_from_an_earlier_edit_are_kept(self):
        timeline = make_timeline(["a", "b"])
        manifest = self.render_all(timeline)
        edited = make_timeline(["a", "changed"])
        stale = stale_entries(manifest, keys_of(edited))
        self.assertEqual(len(stale), 1)
        self.assertTrue(os.path.exists(
            os.path.join(self.directory, stale[0]["video_path"])))


class TestTiming(unittest.TestCase):
    """§12.5.2 -- the warm-up window is excluded from the estimate."""

    def test_warmup_is_not_averaged_in(self):
        segments = [
            {"status": "complete", "frames": 100, "render_seconds": 900.0, "warmup": True,
             "warmup_seconds": 900.0},
            {"status": "complete", "frames": 100, "render_seconds": 100.0},
            {"status": "complete", "frames": 100, "render_seconds": 100.0},
        ]
        self.assertAlmostEqual(seconds_per_frame(segments), 1.0)

    def test_no_timed_segments_gives_none(self):
        self.assertIsNone(seconds_per_frame([]))
        self.assertIsNone(seconds_per_frame([{"status": "pending", "frames": 10,
                                              "render_seconds": 5}]))


class TestReusedWindowsStillCarryContinuity(unittest.TestCase):
    """A bug the cache itself introduces, guarded at the source.

    Before segments could be reused, window i had always just been decoded when
    window i+1 was conditioned, so the carry-over tensors were simply in memory.
    A *reused* window decodes nothing. If the executor then leaves those sockets
    empty, the compiler has already allocated ordinals around them and every
    reference tag behind the hole shifts by one -- the exact silent-wrong-output
    failure this package exists to prevent.

    render.py needs torch, so this is asserted over its source rather than by
    calling it. Crude, and it still fails loudly the moment someone deletes the
    reconstruction.
    """

    def source(self):
        import pathlib
        return (pathlib.Path(__file__).parent.parent / "render.py").read_text(
            encoding="utf-8")

    def test_the_reuse_branch_rebuilds_the_carry_signals_from_disk(self):
        source = self.source()
        branch = source.split("if decision.reuse:", 1)[1].split("if compiled is None", 1)[0]
        self.assertIn("carry_from_segment", branch,
                      "a reused window must reconstruct the next window's carry-over "
                      "tensors from the files it left on disk")
        self.assertNotIn("carry_audio_clip = None", branch,
                         "silently dropping the audio carry leaves a hole in ref_audios")

    def test_a_missing_carry_signal_is_a_failure_not_a_skip(self):
        source = self.source()
        block = source.split("if ref.synthetic:", 1)[1].split("slot = socket_slot_of", 1)[0]
        self.assertIn("failures.append(ref)", block,
                      "an unfillable carry socket must be raised, not skipped -- "
                      "skipping renumbers every tag after it")

    def test_the_video_tail_is_only_decoded_when_a_window_asks_for_it(self):
        """Decoding a segment to rebuild motion is expensive; the frame and the
        audio are not. Only `carry_mode` video/both should pay for it."""
        source = self.source()
        self.assertIn("wants_video=window_wants(following, CARRY_VIDEO_ID)", source)
        self.assertIn("wants_audio=window_wants(following, CARRY_AUDIO_ID)", source)


class TestAssemblySurvivesItsOwnFailure(unittest.TestCase):
    """A container-format problem must not destroy a completed render.

    This one bit for real: a twelve-window render finishes, every segment is on
    disk, and then joining them raises `av.error.ArgumentError` and the whole node
    dies with a traceback. The GPU time is gone from the user's point of view even
    though the bytes are all still sitting in the run folder.

    render.py and media.py need PyAV, so the contract is asserted over their
    source. Crude, and it still fails the moment someone removes the guard.
    """

    def source(self, name):
        import pathlib
        return (pathlib.Path(__file__).parent.parent / name).read_text(encoding="utf-8")

    def test_the_join_returns_empty_rather_than_raising(self):
        media = self.source("media.py")
        block = media.split("def concat_videos", 1)[1].split("def _rescale", 1)[0]
        self.assertIn("except Exception", block)
        self.assertIn('return ""', block)

    def test_the_executor_guards_the_assembly_step(self):
        render = self.source("render.py")
        block = render.split("# ── assembly (§8)", 1)[1].split("frames_out =", 1)[0]
        self.assertIn("try:", block)
        self.assertIn("except Exception", block)
        self.assertIn("concat_videos", block)

    def test_a_failed_join_tells_the_user_the_segments_are_safe(self):
        """The only thing that matters at that moment is whether the work is gone.

        A bare traceback answers "is my render lost?" with silence, and the true
        answer is no -- every window is on disk and a requeue reuses all of them.
        """
        render = self.source("render.py")
        block = render.split("# ── assembly (§8)", 1)[1].split("frames_out =", 1)[0]
        message = block.split("warnings.append(", 1)[1].split(")", 1)[0]
        self.assertIn("safe", message)
        self.assertIn("Requeue", message)

    def test_the_header_is_written_before_any_time_base_is_read(self):
        """libavformat may settle on a different time base than the template's."""
        media = self.source("media.py")
        block = media.split("def _remux_concat", 1)[1]
        self.assertIn("output.start_encoding()", block)
        self.assertLess(block.index("output.start_encoding()"),
                        block.index("time_base = float(video_out.time_base"))

    def test_the_placement_arithmetic_lives_where_it_can_be_tested(self):
        """It was wrong once and shipped, because nothing without PyAV could
        reach it. It is now a pure module driven by timestamps read out of real
        segment files -- see tests/test_concat.py."""
        media = self.source("media.py")
        block = media.split("def _remux_concat", 1)[1]
        self.assertIn(
            "from .comfyui_pulse_studio.concat import video_is_gapless, video_shifts",
            block)
        self.assertIn("video_shifts(frame_counts, fps, time_base)", block)

    def test_a_placement_that_would_stutter_is_never_written(self):
        """Checked before a byte is muxed, so the failure is a message rather
        than a file that freezes at every seam."""
        media = self.source("media.py")
        block = media.split("def _remux_concat", 1)[1]
        self.assertIn("if not video_is_gapless(", block)
        self.assertIn("raise RuntimeError", block)

    def test_the_audio_is_rebuilt_rather_than_copied(self):
        """Each segment's AAC opens on a priming delay that cannot be
        concatenated away; the lossless FLACs exist precisely so it need not be."""
        media = self.source("media.py")
        block = media.split("def _remux_concat", 1)[1]
        self.assertIn("_encode_audio_packets(output, audio)", block)
        self.assertNotIn('source.streams.audio[0]', block)

        render = self.source("render.py")
        assembly = render.split("# ── assembly (§8)", 1)[1].split("frames_out =", 1)[0]
        self.assertLess(assembly.index("audio_out ="), assembly.index("concat_videos"),
                        "the finished audio has to exist before the video is joined")


class TestFilenames(unittest.TestCase):
    def test_stem_is_padded_and_carries_twelve_key_chars(self):
        self.assertEqual(segment_stem(7, "abcdef0123456789"), "seg_0007_abcdef012345")

    def test_paths_share_the_stem(self):
        paths = segment_paths(0, "abcdef0123456789")
        self.assertEqual(paths["video_path"], "seg_0000_abcdef012345.mp4")
        self.assertEqual(paths["audio_path"], "seg_0000_abcdef012345.audio.flac")
        self.assertEqual(paths["last_frame_path"], "seg_0000_abcdef012345.last.png")


if __name__ == "__main__":
    unittest.main()
