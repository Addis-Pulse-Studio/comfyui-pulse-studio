"""A fl2va window still drives its mouth from the recording.

`last_frame_carry` renders every continuation window on fl2va, whose bin is empty:
that checkpoint takes no references at all. Until `drive_files` existed, that left
lock_source with nothing to hold and use_reference_audio with nothing to mux, so a
narrated one-take film was lip-synced for its first window and silent after it --
no error, just a mouth that tracks nothing.

The recordings ride beside `files`, never in it: fl2va's sockets, ordinals and
prompt are exactly what they were. Headless, like the rest of the suite.
"""

import unittest

from comfyui_pulse_studio.assets import KIND_AUDIO, Asset
from comfyui_pulse_studio.compiler import CarryPolicy, compile_timeline
from comfyui_pulse_studio.constants import (
    AUDIO_MODE_LOCK,
    AUDIO_MODE_REFERENCE,
    AUDIO_ROLE_LIP_SYNC,
    AUDIO_ROLE_TIMBRE,
    BRANCH_FL2VA,
    BRANCH_REF2VA,
)
from comfyui_pulse_studio.pulse_timeline import (
    build_timeline,
    global_block,
    ref_descriptor,
    shot_block,
    text_shot_id,
    window_block,
    window_seed,
)
from comfyui_pulse_studio.segcache import cache_key, cache_key_material
from comfyui_pulse_studio.timeline import Timeline

MODEL_FP = "0a1b2c3d4e5f6071"
PATCH_FP = "9988776655443322"


def film(continuation=BRANCH_FL2VA, roles=(AUDIO_ROLE_LIP_SYNC,), durations=(10.83, 10.83, 9.16)):
    """Three shots, one window each, every shot carrying the same narration."""
    shots, cursor = [], 0.0
    for i, duration in enumerate(durations, 1):
        shots.append({"id": "s%d" % i, "start": cursor, "duration": duration,
                      "prompt": "The presenter keeps talking to camera."})
        cursor += duration
    timeline = Timeline(assets=[], shots=shots, duration_seconds=cursor, fps=24,
                        window_seconds=15.0, branch=BRANCH_REF2VA,
                        continuation_branch=continuation)
    for i in range(1, len(durations) + 1):
        timeline.local_refs["s%d" % i] = [
            Asset("aud_s%d_%d" % (i, j), KIND_AUDIO, name="Voice%d" % j, file="", audio_role=role)
            for j, role in enumerate(roles, 1)]
    return compile_timeline(timeline, policy="shot_aligned",
                            carry=CarryPolicy(mode="image", audio=False))


class TestTheContinuationWindowsKeepTheirRecording(unittest.TestCase):
    def test_fl2va_windows_carry_the_recording_beside_their_files(self):
        plan = film()
        self.assertEqual([w.branch for w in plan.windows], [BRANCH_REF2VA, BRANCH_FL2VA, BRANCH_FL2VA])
        for window in plan.windows[1:]:
            # fl2va takes no references: nothing reaches a socket or an ordinal
            self.assertEqual([f for f in window.files if not f.synthetic], [])
            self.assertEqual([(f.asset_id, f.audio_role) for f in window.drive_files],
                             [("aud_s%d_1" % (window.index + 1), AUDIO_ROLE_LIP_SYNC)])
            self.assertTrue(all(f.socket.startswith("drive_audio_") for f in window.drive_files))
            self.assertEqual(window.to_dict()["drive_files"][0]["asset_id"], window.drive_files[0].asset_id)

    def test_ref2va_windows_are_unchanged(self):
        plan = film()
        first = plan.windows[0]
        self.assertEqual(first.drive_files, [])
        self.assertNotIn("drive_files", first.to_dict())
        self.assertIn("aud_s1_1", [f.asset_id for f in first.files])
        for window in film(continuation=BRANCH_REF2VA).windows:
            self.assertEqual(window.drive_files, [])

    def test_only_lip_sync_recordings_drive(self):
        plan = film(roles=(AUDIO_ROLE_TIMBRE, AUDIO_ROLE_LIP_SYNC))
        for window in plan.windows[1:]:
            self.assertEqual([f.audio_role for f in window.drive_files], [AUDIO_ROLE_LIP_SYNC])
        for window in film(roles=(AUDIO_ROLE_TIMBRE,)).windows[1:]:
            self.assertEqual(window.drive_files, [])

    def test_the_prompt_says_the_speech_is_supplied(self):
        """A driven window no longer asks for silence over the voice it follows.

        Without a recording a continuation window is told "no dialogue or
        vocalisation" -- H3 left alone on fl2va invents speech. With one, that
        sentence asked for a still mouth over the very take lock_source holds, so
        it is replaced by one saying the speech is a recording the lips match.
        """
        with_voice = film()
        without = film(roles=())
        for a, b in zip(with_voice.windows[1:], without.windows[1:]):
            self.assertIn("no dialogue or vocalisation", b.prompt)
            self.assertNotIn("no dialogue or vocalisation", a.prompt)
            self.assertIn("is a recorded voice", a.prompt)
            # and nothing about it became a reference tag: fl2va takes none
            self.assertNotIn("<Audio", a.prompt)


def document(branches):
    """A timeline document: one lip-sync recording on each shot, windows on `branches`."""
    shots, windows = [], []
    for i, branch in enumerate(branches):
        text = "Shot %d of the take." % i
        refs = [ref_descriptor(1, KIND_AUDIO, "Narration", "socket", sha256="aa",
                               audio_role=AUDIO_ROLE_LIP_SYNC)]
        shots.append(shot_block(text_shot_id(text[:16], text), i, label=text[:16], visual=text,
                                duration_seconds=10.83, resolved_prompt=text, local_refs=refs))
        windows.append(window_block(i, [shots[-1]["shot_id"]], 260,
                                    seed=window_seed(7, [shots[-1]["shot_id"]]), branch=branch))
    return build_timeline(global_block(style="studio"), [], shots, windows,
                          {"images": 0, "videos": 0, "audio": 1, "total": 1})


class TestTheCacheForgetsUnlockedFl2vaSegments(unittest.TestCase):
    def test_a_locked_fl2va_window_gets_a_new_key(self):
        doc = document([BRANCH_REF2VA, BRANCH_FL2VA])
        ref2va, fl2va = doc["windows"]
        self.assertNotIn("audio_lock_branch",
                         [m[0] for m in cache_key_material(doc, ref2va, MODEL_FP, PATCH_FP, AUDIO_MODE_LOCK)])
        self.assertIn(["audio_lock_branch", BRANCH_FL2VA],
                      cache_key_material(doc, fl2va, MODEL_FP, PATCH_FP, AUDIO_MODE_LOCK))

    def test_reference_only_keys_stay_on_disk(self):
        doc = document([BRANCH_REF2VA, BRANCH_FL2VA])
        for window in doc["windows"]:
            material = cache_key_material(doc, window, MODEL_FP, PATCH_FP, AUDIO_MODE_REFERENCE)
            self.assertNotIn("audio_lock_branch", [m[0] for m in material])
        # and the key is deterministic
        self.assertEqual(cache_key(doc, doc["windows"][1], MODEL_FP, PATCH_FP, AUDIO_MODE_LOCK),
                         cache_key(doc, doc["windows"][1], MODEL_FP, PATCH_FP, AUDIO_MODE_LOCK))


if __name__ == "__main__":
    unittest.main()
