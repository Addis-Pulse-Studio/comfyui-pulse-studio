"""The drive/final split: locking a lip-sync recording into the latent, muxing the
clean take, and correcting one character at a time.

`reference_only` has to render exactly what it rendered before these existed --
same latent, same cache key -- because it is the default and every saved project
is using it. The rest is arithmetic: which windows a lock changes, which frames
belong to one character's voice. It runs headless; the tensor halves are marked
and skipped without torch.
"""

import unittest

from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, Asset
from comfyui_pulse_studio.constants import (
    AUDIO_MODE_LOCK,
    AUDIO_MODE_REFERENCE,
    AUDIO_MODE_REMIX,
    AUDIO_ROLE_LIP_SYNC,
    AUDIO_ROLE_TIMBRE,
)
from comfyui_pulse_studio.lipsync import (
    active_spans,
    owner_voice_ids,
    resolve_owner,
    speaker_frames,
    spans_to_frames,
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

try:
    import torch
except ImportError:  # pragma: no cover - the headless suite
    torch = None

needs_torch = unittest.skipIf(torch is None, "needs torch")

MODEL_FP = "0a1b2c3d4e5f6071"
PATCH_FP = "9988776655443322"


def document(role=AUDIO_ROLE_LIP_SYNC, local_audio_on=None):
    """Two windows. A global recording with `role`, or -- with `local_audio_on` --
    a recording local to that one shot only."""
    texts = ["Ada asks the question.", "Ben answers it."]
    local = {}
    refs = []
    if role is not None and local_audio_on is None:
        refs.append(ref_descriptor(1, KIND_AUDIO, "AdaVoice", "socket", sha256="aa",
                                   audio_role=role))
    shots = []
    for i, text in enumerate(texts):
        shot_refs = []
        if local_audio_on == i:
            shot_refs = [ref_descriptor(1, KIND_AUDIO, "BenVoice", "socket", sha256="bb",
                                        audio_role=AUDIO_ROLE_LIP_SYNC)]
        shots.append(shot_block(text_shot_id(text[:16], text), i, label=text[:16],
                                visual=text, duration_seconds=7.5, resolved_prompt=text,
                                local_refs=shot_refs))
        local[i] = shot_refs
    windows = [window_block(i, [s["shot_id"]], 181, seed=window_seed(7, [s["shot_id"]]))
               for i, s in enumerate(shots)]
    return build_timeline(global_block(style="studio"), refs, shots, windows,
                          {"images": 0, "videos": 0, "audio": len(refs), "total": len(refs)})


def keys(doc, mode=AUDIO_MODE_REFERENCE, strength=None):
    return [cache_key(doc, w, MODEL_FP, PATCH_FP, mode, strength) for w in doc["windows"]]


class TestTheLockIsInTheKeyOnlyWhereItChangesSomething(unittest.TestCase):
    def test_reference_only_keeps_every_key_on_disk(self):
        doc = document()
        before = [cache_key(doc, w, MODEL_FP, PATCH_FP) for w in doc["windows"]]
        self.assertEqual(keys(doc, AUDIO_MODE_REFERENCE), before)
        self.assertNotIn("audio_mode", [m[0] for m in cache_key_material(
            doc, doc["windows"][0], MODEL_FP, PATCH_FP)])

    def test_lock_and_remix_move_a_lip_sync_window(self):
        doc = document()
        reference, lock = keys(doc), keys(doc, AUDIO_MODE_LOCK)
        remix, remix_more = keys(doc, AUDIO_MODE_REMIX, 0.35), keys(doc, AUDIO_MODE_REMIX, 0.6)
        for i in range(2):
            self.assertEqual(len({reference[i], lock[i], remix[i], remix_more[i]}), 4)
        # lock_source has no strength to record: changing it moves nothing
        self.assertEqual(lock, keys(doc, AUDIO_MODE_LOCK, 0.9))

    def test_windows_without_a_lip_sync_recording_keep_their_key(self):
        for doc in (document(role=None), document(role=AUDIO_ROLE_TIMBRE)):
            self.assertEqual(keys(doc, AUDIO_MODE_LOCK), keys(doc))
        doc = document(local_audio_on=1)
        reference, lock = keys(doc), keys(doc, AUDIO_MODE_LOCK)
        self.assertEqual(lock[0], reference[0])       # Ada's shot has no recording
        self.assertNotEqual(lock[1], reference[1])    # Ben's does


class TestActiveSpans(unittest.TestCase):
    def test_a_breath_is_bridged_and_a_click_is_dropped(self):
        quiet, loud = -80.0, -10.0
        levels = ([quiet] * 50 + [loud] * 100 + [quiet] * 20 + [loud] * 80
                  + [quiet] * 100 + [loud] * 5 + [quiet] * 40)
        spans = active_spans(levels, hop_seconds=0.01, threshold_db=-40.0,
                             hangover_seconds=0.3, min_seconds=0.1)
        self.assertEqual([(round(a, 2), round(b, 2)) for a, b in spans], [(0.5, 2.5)])

    def test_speech_running_to_the_end_is_closed(self):
        spans = active_spans([-80.0] * 10 + [-20.0] * 30, hop_seconds=0.01)
        self.assertEqual([(round(a, 2), round(b, 2)) for a, b in spans], [(0.1, 0.4)])

    def test_frames_are_exact_widened_merged_and_clipped(self):
        self.assertEqual(spans_to_frames([(0.0, 7.5)], 24, 362), [(0, 180)])
        self.assertEqual(spans_to_frames([(7.5, 15.0)], 24, 362, handle_seconds=0.25),
                         [(174, 362)])
        self.assertEqual(spans_to_frames([(1.0, 2.0), (2.1, 3.0)], 24, 362, 0.1),
                         [(21, 75)])
        self.assertEqual(spans_to_frames([(20.0, 21.0)], 24, 362), [])


class TestHandlesOnlyReachIntoSilence(unittest.TestCase):
    def test_a_cut_between_speakers_stops_the_handle(self):
        # Ada 0-7.5 s, Ben 7.5-15 s: neither handle crosses the cut
        self.assertEqual(speaker_frames([(0.0, 7.5)], [(7.5, 15.0)], 24, 362, 0.25), [(0, 180)])
        self.assertEqual(speaker_frames([(7.5, 15.0)], [(0.0, 7.5)], 24, 362, 0.25), [(180, 362)])

    def test_a_gap_is_filled_up_to_the_other_voice(self):
        self.assertEqual(speaker_frames([(0.0, 2.0)], [(3.0, 5.0)], 24, 362, 1.5), [(0, 72)])
        # own speech overlapping someone else's is still corrected
        self.assertEqual(speaker_frames([(1.0, 2.0)], [(1.5, 3.0)], 24, 362, 0.5), [(12, 48)])


class TestWhoseVoiceIsWhose(unittest.TestCase):
    def cast(self):
        timeline = Timeline(
            assets=[
                {"id": "ada", "kind": KIND_IMAGE, "name": "Ada", "file": "a.png"},
                {"id": "ben", "kind": KIND_IMAGE, "name": "Ben", "file": "b.png"},
                {"id": "ada_vo", "kind": KIND_AUDIO, "name": "AdaVoice", "file": "a.wav",
                 "audio_role": AUDIO_ROLE_LIP_SYNC, "voice_of": "ada"},
                {"id": "narration", "kind": KIND_AUDIO, "name": "Narration",
                 "file": "n.wav", "audio_role": AUDIO_ROLE_LIP_SYNC},
            ],
            shots=[{"id": "s1", "start": 0.0, "duration": 7.5, "prompt": "@Ada speaks",
                    "speakers": ["ada"]},
                   {"id": "s2", "start": 7.5, "duration": 7.5, "prompt": "@Ben speaks",
                    "speakers": ["ben"]}],
            duration_seconds=15.0, fps=24)
        timeline.local_refs["s2"] = [Asset("sock:shot.s2.voice", KIND_AUDIO, name="Voice",
                                           audio_role=AUDIO_ROLE_LIP_SYNC)]
        return timeline

    def test_bin_voice_by_voice_of_and_shot_voice_by_first_speaker(self):
        timeline = self.cast()
        self.assertEqual(owner_voice_ids(timeline, "ada"), {"ada_vo"})
        self.assertEqual(owner_voice_ids(timeline, "ben"), {"sock:shot.s2.voice"})

    def test_names_resolve_to_faces_never_to_voices(self):
        timeline = self.cast()
        self.assertEqual(resolve_owner(timeline, "@Ada").asset_id, "ada")
        self.assertEqual(resolve_owner(timeline, " ben ").asset_id, "ben")
        with self.assertRaisesRegex(ValueError, "No character named 'AdaVoice'"):
            resolve_owner(timeline, "@AdaVoice")
        with self.assertRaisesRegex(ValueError, "Name the character"):
            resolve_owner(timeline, "@")


@needs_torch
class TestCutAndPaste(unittest.TestCase):
    def setUp(self):
        import media
        self.media = media

    def test_levels(self):
        rate = 1000
        wave = torch.zeros(1, 2, rate)
        wave[..., 500:] = 0.5
        levels = self.media.rms_levels_db({"waveform": wave, "sample_rate": rate}, 0.01)
        self.assertEqual(len(levels), 100)
        self.assertLess(levels[10], -100)
        self.assertAlmostEqual(levels[80], -6.02, places=1)

    def test_one_character_goes_back_where_it_came_from(self):
        images = torch.rand(40, 24, 32, 3)
        frames, segment = self.media.cut_segment(images, [(2, 6), (30, 33)],
                                                 box=(0.5, 0.0, 0.5, 1.0))
        self.assertEqual(frames.shape, (7, 24, 16, 3))
        same, notes = self.media.paste_segment(images, frames, segment, feather=3)
        self.assertTrue(torch.allclose(same, images))
        self.assertEqual(notes, [])
        fixed, _ = self.media.paste_segment(images, torch.ones(7, 24, 16, 3), segment)
        self.assertTrue(torch.equal(fixed[31, :, 16:], torch.ones(24, 16, 3)))
        self.assertTrue(torch.equal(fixed[31, :, :16], images[31, :, :16]))
        self.assertTrue(torch.equal(fixed[10], images[10]))
        # LatentSync hands back its own frame count, at its own size
        _, notes = self.media.paste_segment(images, torch.ones(11, 48, 32, 3), segment)
        self.assertEqual(len(notes), 2)
        with self.assertRaisesRegex(ValueError, "not the ones"):
            self.media.paste_segment(images[:39], frames, segment)

    def test_the_audio_is_the_same_seconds(self):
        wave = torch.arange(40 * 100, dtype=torch.float32).view(1, 1, -1)
        audio = self.media.cut_audio({"waveform": wave, "sample_rate": 2400},
                                     [(2, 6), (30, 33)], 24)
        self.assertEqual(audio["waveform"].shape[-1], 700)
        self.assertEqual(float(audio["waveform"][0, 0, 400]), 3000.0)


if __name__ == "__main__":
    unittest.main()
