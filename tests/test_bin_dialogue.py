"""A two-hander laid out in the Asset Bin: faces, and one lip-sync row per line.

The bin -- the panel at the bottom of Pulse Slate -- is the one place a graph
should have to attach a reference. A dialogue could not be expressed there: the
bin counted every line against the three <Audio j> references a window has, the
panel had no way to say where on the film clock a line plays, and the compiler
judged each voice alone against the whole window, so every voice in every window
was reported silent while the other one spoke. So dialogue was wired as a Load
Audio and a Pulse Voice per line around the node, beside a bin holding the faces
-- references attached in two places.
"""

import json
import unittest

from comfyui_pulse_studio.assets import KIND_AUDIO, KIND_IMAGE, Asset, AssetBin, BudgetError
from comfyui_pulse_studio.binops import apply_operation, bin_state
from comfyui_pulse_studio.compiler import CarryPolicy, compile_timeline
from comfyui_pulse_studio.constants import AUDIO_ROLE_LIP_SYNC, AUDIO_ROLE_TIMBRE
from comfyui_pulse_studio.frames import in_speech, partition_windows
from comfyui_pulse_studio.lipsync import seam_frames, spans_to_frames
from comfyui_pulse_studio.timeline import Timeline
from comfyui_pulse_studio.widget_state import apply_bin_operation

SHOT = 181 / 24.0   # two of these are one 362-frame window
LINE = SHOT - 0.4   # a line, and the beat of air before the cut


def faces(*names):
    return [Asset(n.lower(), KIND_IMAGE, name=n, file=n.lower() + ".png",
                  description="the character called %s" % n) for n in names]


def line(i, owner, at, seconds=LINE, role=AUDIO_ROLE_LIP_SYNC):
    return Asset("%s_line_%d" % (owner, i), KIND_AUDIO, name="%sLine%d" % (owner.title(), i),
                 file=owner + ".wav", audio_role=role, voice_of=owner, audio_offset=at,
                 trim_start=0.0, trim_end=seconds, source_seconds=seconds)


def two_hander(lines, shots=4):
    """`shots` 7.54 s shots, Ada and Ben alternating, and the bin's lines."""
    assets = faces("Ada", "Ben") + lines
    timeline = Timeline(
        assets=[a.to_dict() for a in assets],
        shots=[{"id": "s%d" % i, "start": (i - 1) * SHOT, "duration": SHOT,
                "prompt": "%s speaks; the other listens." % ("@Ada" if i % 2 else "@Ben"),
                "speakers": ["ada" if i % 2 else "ben"]} for i in range(1, shots + 1)],
        duration_seconds=SHOT * shots, fps=24, window_seconds=SHOT * 2)
    return compile_timeline(timeline, policy="shot_aligned",
                            carry=CarryPolicy(mode="image", audio=False))


def notes(compiled):
    lines = list(compiled.diagnostics)
    for window in compiled.windows:
        lines.extend(window.diagnostics)
    return " ".join(lines)


class TheBudgetCountsVoicesNotLines(unittest.TestCase):

    def test_six_lines_of_two_characters_are_two_references(self):
        bin_ = AssetBin(faces("Ada", "Ben") + [line(i, "ada" if i % 2 else "ben", i * SHOT)
                                               for i in range(6)])
        report = bin_.budget()
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.audios, 2)
        self.assertEqual(report.files, 4)  # two faces, two voices
        self.assertIn("2/3 audio", report.meter())

    def test_a_fourth_voice_is_still_refused(self):
        with self.assertRaises(BudgetError) as caught:
            AssetBin(faces("Ada", "Ben", "Cy", "Dee")
                     + [line(1, owner, 0.0) for owner in ("ada", "ben", "cy", "dee")])
        self.assertIn("too many reference audios", str(caught.exception))

    def test_unbound_lines_still_count_one_each(self):
        """Nothing folds a recording that belongs to nobody into anyone's track."""
        unbound = [Asset("a%d" % i, KIND_AUDIO, name="Take%d" % i, file="t.wav",
                         audio_role=AUDIO_ROLE_LIP_SYNC) for i in range(4)]
        with self.assertRaises(BudgetError):
            AssetBin(unbound)

    def test_a_timbre_reference_is_a_reference_of_its_own(self):
        """Only lip-sync recordings are laid on a voice track; timbre is not."""
        bin_ = AssetBin(faces("Ada"))
        for i in range(3):
            bin_.add(line(i, "ada", 0.0, role=AUDIO_ROLE_TIMBRE))
        ok, reason = bin_.can_add(line(9, "ada", 0.0, role=AUDIO_ROLE_TIMBRE))
        self.assertFalse(ok)
        self.assertIn("too many reference audios", reason)

    def test_the_panel_tags_every_line_of_a_voice_with_that_voices_number(self):
        bin_ = AssetBin(faces("Ada", "Ben") + [line(i, "ada" if i % 2 else "ben", i * SHOT)
                                               for i in range(1, 5)])
        tags = {r["name"]: r["tag"] for r in bin_state(bin_)["assets"]}
        self.assertEqual((tags["AdaLine1"], tags["AdaLine3"]), ("<Audio 1>", "<Audio 1>"))
        self.assertEqual((tags["BenLine2"], tags["BenLine4"]), ("<Audio 2>", "<Audio 2>"))

    def test_what_the_bin_admits_the_compiler_folds_into_one_slot_per_voice(self):
        compiled = two_hander([line(i, "ada" if i % 2 else "ben", (i - 1) * SHOT)
                               for i in range(1, 9)], shots=8)
        self.assertTrue(compiled.ok, compiled.problems)
        for window in compiled.windows:
            audio = [f for f in window.files if f.kind == KIND_AUDIO]
            self.assertEqual(len(audio), 2, window.index)


class ThePanelPlacesALine(unittest.TestCase):

    def setUp(self):
        self.bin = AssetBin(faces("Ada") + [line(1, "ada", 0.0)])

    def test_it_sets_where_the_line_starts_and_which_seconds_play(self):
        asset = apply_operation(self.bin, "set_audio_timing", asset_id="ada_line_1",
                                audio_offset="15.0833", trim_start="7.14",
                                trim_end="14.28")
        self.assertEqual((asset.audio_offset, asset.trim_start, asset.trim_end),
                         (15.0833, 7.14, 14.28))

    def test_empty_fields_are_the_film_start_and_the_whole_file(self):
        asset = apply_operation(self.bin, "set_audio_timing", asset_id="ada_line_1",
                                audio_offset="", trim_start="", trim_end="")
        self.assertEqual((asset.audio_offset, asset.trim_start, asset.trim_end),
                         (0.0, 0.0, None))

    def test_an_end_before_its_start_is_refused(self):
        with self.assertRaisesRegex(ValueError, "has to come after trim_start"):
            apply_operation(self.bin, "set_audio_timing", asset_id="ada_line_1",
                            audio_offset=0, trim_start=5, trim_end=4)

    def test_a_negative_second_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            apply_operation(self.bin, "set_audio_timing", asset_id="ada_line_1",
                            audio_offset=-1, trim_start=0, trim_end=None)

    def test_a_picture_has_no_place_on_the_film_clock(self):
        with self.assertRaisesRegex(ValueError, "only a recording"):
            apply_operation(self.bin, "set_audio_timing", asset_id="ada",
                            audio_offset=1, trim_start=0, trim_end=None)

    def test_the_panel_is_sent_the_timing_it_shows(self):
        row = next(r for r in bin_state(self.bin)["assets"] if r["id"] == "ada_line_1")
        self.assertEqual((row["audio_offset"], row["trim_start"], row["trim_end"]),
                         (0.0, 0.0, LINE))

    def test_a_refused_edit_leaves_the_document_untouched(self):
        raw = json.dumps({"schema": 2, "assets": self.bin.to_list(), "cast": []})
        new_raw, error = apply_bin_operation(raw, "set_audio_timing", asset_id="ada_line_1",
                                             audio_offset=0, trim_start=3, trim_end=1)
        self.assertIn("has to come after trim_start", error)
        self.assertEqual(new_raw, raw)


class AConversationIsJudgedAsOne(unittest.TestCase):
    """Each voice is silent while the other speaks. That is a conversation, not a
    recording that runs out, and the window is judged on all its voices at once."""

    def test_a_two_hander_laid_out_in_the_bin_reports_nothing(self):
        compiled = two_hander([line(i, "ada" if i % 2 else "ben", (i - 1) * SHOT)
                               for i in range(1, 5)])
        self.assertTrue(compiled.ok, compiled.problems)
        self.assertEqual(len(compiled.windows), 2)
        text = notes(compiled)
        for phrase in ("runs out", "starts", "do not meet"):
            self.assertNotIn(phrase, text)

    def test_dead_air_nobody_speaks_in_is_still_named(self):
        # Ben's first line is a second and a half, then nothing until the cut.
        compiled = two_hander([line(1, "ada", 0.0), line(2, "ben", SHOT, seconds=1.5),
                               line(3, "ada", 2 * SHOT), line(4, "ben", 3 * SHOT)])
        first = " ".join(compiled.windows[0].diagnostics)
        self.assertIn("runs out", first)
        self.assertIn("@AdaLine1", first)
        self.assertIn("@BenLine2", first)
        self.assertEqual(first.count("runs out"), 1, "said once for the window, not per voice")
        self.assertNotIn("runs out", " ".join(compiled.windows[1].diagnostics))


class TheSeamSummaryAddsUp(unittest.TestCase):

    def test_a_seam_on_a_cut_inside_speech_is_not_counted_twice(self):
        notes_ = []
        windows = partition_windows(600, 362, policy="speech_aware", diagnostics=notes_,
                                    speech=[(0, 600)], boundaries=[260])
        self.assertEqual(windows[0], 260)
        summary = next(n for n in notes_ if n.startswith("speech_aware put"))
        self.assertIn("1 of 1 seam(s) on a cut and 0 in a pause", summary)
        self.assertNotIn("-1", summary)


class TheSeamPlannerRoundsToTheNearestFrame(unittest.TestCase):

    def test_a_line_typed_at_the_cuts_displayed_time_starts_on_the_cut(self):
        # A cut at 15.0833 s is frame 362; the panel shows it as 15.08.
        self.assertEqual(seam_frames([(15.08, 22.0)], 24, 1000), [(362, 528)])
        self.assertFalse(in_speech(362, seam_frames([(15.08, 22.0)], 24, 1000)))

    def test_the_lock_keeps_rounding_outward(self):
        """The lip-sync lock holds every frame a word touches; only seams round in."""
        self.assertEqual(spans_to_frames([(15.08, 22.0)], 24, 1000), [(361, 528)])


if __name__ == "__main__":
    unittest.main()
