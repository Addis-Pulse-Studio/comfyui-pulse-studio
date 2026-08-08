"""The raisable audio ceiling, and everything that must move with it.

WHY THIS IS OPT-IN, AND WHY IT WORKS AT ALL

`comfy_extras/nodes_minimax_h3.py` declares `ref_audios` as an Autogrow socket
with `max=3`. That cap is enforced by graph validation in `execution.py`. This
pack calls `MiniMaxH3ReferenceToVideo.execute()` in-process, so validation never
runs against what it passes, and the model layer underneath has no cap at all --
`PackedLayout` appends one `ref_audio` segment per reference block in a loop,
and the tokenizer just increments a counter for `<Audio j>`.

So nine standalone audio references really are marshalled and rendered. They are
also outside what MiniMax documents and what ComfyUI declares, which is why the
default stays at 3 and why crossing the line is announced rather than silent.

The tests below pin three things: the ceiling reaches every place that counts
references, the documented default is genuinely unchanged, and a workflow saved
before the widget existed still loads.
"""

import unittest

from comfyui_pulse_studio.assets import (
    DEFAULT_LIMITS,
    Asset,
    AssetBin,
    BudgetError,
    RefLimits,
)
from comfyui_pulse_studio.binops import bin_state
from comfyui_pulse_studio.compiler import compile_timeline
from comfyui_pulse_studio.constants import (
    MAX_REF_AUDIOS,
    MAX_REF_AUDIOS_CEILING,
    MAX_REF_FILES_TOTAL,
)
from comfyui_pulse_studio.timeline import Timeline
from comfyui_pulse_studio.widget_state import apply_bin_operation, build_timeline


def audio(n):
    return Asset("a%d" % n, "audio", name="Voice%d" % n, file="example_voice_%d.wav" % n)


def image(n):
    return Asset("i%d" % n, "image", name="Face%d" % n, file="example_face_%d.png" % n)


class TestRefLimits(unittest.TestCase):
    def test_the_default_is_exactly_what_minimax_documents(self):
        self.assertEqual(RefLimits().audios, MAX_REF_AUDIOS)
        self.assertEqual(RefLimits().files, MAX_REF_FILES_TOTAL)
        self.assertFalse(RefLimits().beyond_spec)

    def test_the_file_total_rises_with_the_audio_ceiling(self):
        """Otherwise the audio fits and the total refuses it, which reads as a
        bug rather than as a budget."""
        for n in range(MAX_REF_AUDIOS, MAX_REF_AUDIOS_CEILING + 1):
            with self.subTest(audios=n):
                limits = RefLimits(n)
                self.assertEqual(limits.files, MAX_REF_FILES_TOTAL + n - MAX_REF_AUDIOS)
                # 9 images + n audio must always be admissible on file count alone.
                self.assertGreaterEqual(limits.files, limits.images + n)

    def test_only_the_audio_ceiling_moves(self):
        raised = RefLimits(MAX_REF_AUDIOS_CEILING)
        self.assertEqual(raised.images, DEFAULT_LIMITS.images)
        self.assertEqual(raised.videos, DEFAULT_LIMITS.videos)
        self.assertEqual(raised.soundtracks, DEFAULT_LIMITS.soundtracks)

    def test_it_refuses_a_ceiling_outside_the_range(self):
        with self.assertRaises(ValueError):
            RefLimits(MAX_REF_AUDIOS - 1)
        with self.assertRaises(ValueError):
            RefLimits(MAX_REF_AUDIOS_CEILING + 1)

    def test_beyond_spec_is_true_only_above_the_documented_budget(self):
        self.assertFalse(RefLimits(MAX_REF_AUDIOS).beyond_spec)
        self.assertTrue(RefLimits(MAX_REF_AUDIOS + 1).beyond_spec)


class TestTheBinRespectsTheCeiling(unittest.TestCase):
    def test_the_fourth_audio_is_refused_by_default(self):
        bin_ = AssetBin([audio(n) for n in range(MAX_REF_AUDIOS)])
        ok, reason = bin_.can_add(audio(99))
        self.assertFalse(ok)
        self.assertIn("too many reference audios", reason)

    def test_nine_audio_fit_when_the_ceiling_is_raised(self):
        limits = RefLimits(9)
        bin_ = AssetBin()
        for n in range(9):
            bin_.add(audio(n), limits=limits)
        report = bin_.budget(limits=limits)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.audios, 9)

    def test_the_tenth_is_still_refused_at_a_raised_ceiling(self):
        """Raised is not unlimited. The dial has a stop."""
        limits = RefLimits(9)
        bin_ = AssetBin([audio(n) for n in range(9)], limits=limits)
        ok, reason = bin_.can_add(audio(99), limits=limits)
        self.assertFalse(ok)
        self.assertIn("too many reference audios", reason)

    def test_nine_images_and_nine_audio_fit_together(self):
        """The whole point of the ask: audio matching images."""
        limits = RefLimits(9)
        bin_ = AssetBin([image(n) for n in range(9)] + [audio(n) for n in range(9)],
                        limits=limits)
        report = bin_.budget()
        self.assertTrue(report.ok, report.errors)
        self.assertEqual((report.images, report.audios, report.files), (9, 9, 18))

    def test_raising_the_ceiling_does_not_raise_the_image_cap(self):
        limits = RefLimits(9)
        bin_ = AssetBin([image(n) for n in range(9)], limits=limits)
        ok, reason = bin_.can_add(image(99))
        self.assertFalse(ok)
        self.assertIn("too many reference images", reason)

    def test_the_meter_reads_out_the_ceiling_in_force(self):
        bin_ = AssetBin([audio(0)])
        self.assertIn("1/3 audio", bin_.budget().meter())
        self.assertIn("1/9 audio", bin_.budget(limits=RefLimits(9)).meter())

    def test_the_panel_state_reports_whether_it_is_beyond_spec(self):
        bin_ = AssetBin([audio(0)])
        self.assertFalse(bin_state(bin_)["budget"]["beyond_spec"])
        raised = bin_state(bin_, limits=RefLimits(9))["budget"]
        self.assertTrue(raised["beyond_spec"])
        self.assertEqual(raised["max_audios"], 9)
        self.assertEqual(raised["max_files"], 18)


class TestTheCompilerHonoursTheCeiling(unittest.TestCase):
    def _timeline(self, ceiling, count):
        return Timeline(
            assets=[a.to_dict() for a in [audio(n) for n in range(count)]],
            shots=[{"id": "s1", "start": 0.0, "duration": 5.0, "prompt": "she walks in"}],
            duration_seconds=5.0,
            audio_ref_ceiling=ceiling,
        )

    def test_nine_audio_cannot_even_be_assembled_at_the_documented_ceiling(self):
        """The refusal lands while building the project, not at render time.

        An over-budget bin is not a state this package lets you hold: the bin
        object itself will not construct. That is why raising the ceiling has to
        reach the document loader and not only the compiler.
        """
        with self.assertRaises(BudgetError) as caught:
            self._timeline(MAX_REF_AUDIOS, 9)
        self.assertIn("too many reference audios", str(caught.exception))

    def test_three_audio_still_compile_to_three_tags_by_default(self):
        plan = compile_timeline(self._timeline(MAX_REF_AUDIOS, 3))
        self.assertTrue(plan.ok, plan.problems)
        tags = plan.windows[0].tag_map.by_id
        self.assertEqual(sum(1 for t in tags.values() if t.startswith("<Audio")), 3)

    def test_a_raised_ceiling_carries_all_nine_into_the_window(self):
        plan = compile_timeline(self._timeline(9, 9))
        self.assertTrue(plan.ok, plan.problems)
        tags = plan.windows[0].tag_map.by_id
        audio_tags = sorted(t for t in tags.values() if t.startswith("<Audio"))
        self.assertEqual(len(audio_tags), 9)
        # Ordinals stay dense and 1-based -- the property the whole bin exists
        # to hold. A gap here is a misnumbered render that still succeeds.
        self.assertEqual(audio_tags[0], "<Audio 1>")
        self.assertEqual(
            sorted(int(t[len("<Audio "):-1]) for t in audio_tags), list(range(1, 10)))

    def test_the_sockets_the_stock_node_will_receive_are_contiguous(self):
        """`ref_audio_0..8`, gapless. The tokenizer numbers by position in that
        dict, so a hole would shift every <Audio j> after it."""
        plan = compile_timeline(self._timeline(9, 9))
        names = sorted(k for k in plan.windows[0].tag_map.sockets if k.startswith("ref_audio_"))
        self.assertEqual(names, ["ref_audio_%d" % n for n in range(9)])


class TestTheCeilingTravelsWithTheProject(unittest.TestCase):
    def test_a_raised_ceiling_survives_a_document_round_trip(self):
        raised = Timeline(audio_ref_ceiling=9)
        self.assertEqual(Timeline.from_dict(raised.to_dict()).limits.audios, 9)

    def test_a_default_project_serialises_as_it_always_did(self):
        """No new key in the document unless the ceiling was actually raised, so
        an untouched project's timeline_data does not churn."""
        self.assertNotIn("audio_ref_ceiling", Timeline().to_dict())

    def test_the_widget_wins_over_the_stored_copy(self):
        """The ceiling belongs to the render being queued. The stored value only
        exists so a saved project reopens under the budget it was built with."""
        stored = '{"schema": 2, "assets": [], "cast": [], "audio_ref_ceiling": 9}'
        timeline, _ = build_timeline(stored, shot_prompt="[Shot 1] a", duration_seconds=5)
        self.assertEqual(timeline.limits.audios, 9)
        timeline, _ = build_timeline(stored, shot_prompt="[Shot 1] a", duration_seconds=5,
                                     audio_ref_ceiling=3)
        self.assertEqual(timeline.limits.audios, 3)

    def test_a_workflow_saved_before_the_widget_existed_gets_the_default(self):
        timeline, _ = build_timeline('{"schema": 2, "assets": []}',
                                     shot_prompt="[Shot 1] a", duration_seconds=5)
        self.assertEqual(timeline.limits.audios, MAX_REF_AUDIOS)
        self.assertFalse(timeline.limits.beyond_spec)


class TestTheEditPathAgreesWithTheRenderPath(unittest.TestCase):
    """The bin refusing a drop that the compiler would have accepted -- or worse,
    accepting one it will silently drop -- is the failure this guards."""

    def _doc_with(self, count):
        assets = ", ".join(
            '{"id": "a%d", "kind": "audio", "name": "V%d", "file": "example_v%d.wav"}'
            % (n, n, n) for n in range(count))
        return '{"schema": 2, "assets": [%s], "cast": []}' % assets

    def test_the_fourth_drop_is_refused_at_the_documented_ceiling(self):
        new_raw, error = apply_bin_operation(
            self._doc_with(3), "add",
            asset={"id": "a9", "kind": "audio", "name": "V9", "file": "example_v9.wav"})
        self.assertIsNotNone(error)
        self.assertIn("too many reference audios", error)
        self.assertEqual(new_raw, self._doc_with(3), "a refused edit must not touch the document")

    def test_the_fourth_drop_is_accepted_at_a_raised_ceiling(self):
        new_raw, error = apply_bin_operation(
            self._doc_with(3), "add", limits=RefLimits(9),
            asset={"id": "a9", "kind": "audio", "name": "V9", "file": "example_v9.wav"})
        self.assertIsNone(error)
        self.assertIn('"a9"', new_raw)

    def test_what_the_bin_accepts_is_what_the_window_carries(self):
        raw = self._doc_with(9)
        timeline, _ = build_timeline(raw, shot_prompt="[Shot 1] a", duration_seconds=5,
                                     audio_ref_ceiling=9)
        self.assertTrue(timeline.assets.budget(limits=timeline.limits).ok)
        plan = compile_timeline(timeline)
        carried = sum(1 for t in plan.windows[0].tag_map.by_id.values()
                      if t.startswith("<Audio"))
        self.assertEqual(carried, 9, "the bin accepted nine; the window must carry nine")


if __name__ == "__main__":
    unittest.main()
