"""Prompt-box parsing: normalization, shot extraction, global fields."""

import unittest

from comfyui_pulse_studio.parsing import (
    normalize_prompt_text,
    parse_global_prompt,
    parse_shots,
)


class TestNormalization(unittest.TestCase):
    def test_intact_text_is_left_alone(self):
        text = "[Shot 1] she walks in\n[Shot 2] he looks up"
        self.assertEqual(normalize_prompt_text(text), text)

    def test_adjacent_markers_are_split(self):
        """The exact damage a widget that strips newlines does."""
        out = normalize_prompt_text("[Shot 1] she walks in [Shot 2] he looks up")
        self.assertEqual(out.split("\n"), ["[Shot 1] she walks in", "[Shot 2] he looks up"])

    def test_double_space_becomes_a_newline(self):
        out = normalize_prompt_text("she walks in  he looks up")
        self.assertEqual(out.split("\n"), ["she walks in", "he looks up"])

    def test_single_spaces_are_never_touched(self):
        """Ordinary prose must survive untouched, or every sentence would break."""
        text = "a quiet room with one lamp and a long shadow"
        self.assertEqual(normalize_prompt_text(text), text)

    def test_windows_line_endings(self):
        self.assertEqual(normalize_prompt_text("a\r\nb"), "a\nb")

    def test_normalization_is_idempotent(self):
        once = normalize_prompt_text("[Shot 1] a [Shot 2] b")
        self.assertEqual(normalize_prompt_text(once), once)

    def test_timecode_markers_split_too(self):
        out = normalize_prompt_text("[00:00.000] a [00:04.000] b")
        self.assertEqual(len(out.split("\n")), 2)

    def test_empty_input(self):
        self.assertEqual(normalize_prompt_text(""), "")
        self.assertEqual(normalize_prompt_text(None), "")


class TestShotParsing(unittest.TestCase):
    def test_shot_markers(self):
        shots, _ = parse_shots("[Shot 1] she walks in\n[Shot 2] he looks up", 10.0)
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0]["prompt"], "she walks in")
        self.assertEqual(shots[1]["prompt"], "he looks up")

    def test_timecode_markers_set_the_clock(self):
        shots, _ = parse_shots("[00:00.000] a\n[00:04.500] b", 10.0)
        self.assertAlmostEqual(shots[0]["start"], 0.0)
        self.assertAlmostEqual(shots[1]["start"], 4.5)

    def test_at_form_from_a_previous_compile_round_trips(self):
        """A user who pastes back a compiled prompt must still parse."""
        shots, _ = parse_shots("[Shot 1] a\n[Shot 2] At 00:06.000, b", 12.0)
        self.assertAlmostEqual(shots[1]["start"], 6.0)

    def test_bare_timecode(self):
        shots, _ = parse_shots("00:00 a\n00:03.250 b", 8.0)
        self.assertAlmostEqual(shots[1]["start"], 3.25)

    def test_plain_paragraph_becomes_one_shot(self):
        """Typing prose with no markers must not yield nothing."""
        shots, _ = parse_shots("a wide shot of the empty cafe", 5.0)
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0]["start"], 0.0)

    def test_flattened_text_still_parses(self):
        """Belt and braces: newlines stripped in transit, shots still found."""
        shots, _ = parse_shots("[Shot 1] she walks in [Shot 2] he looks up", 10.0)
        self.assertEqual(len(shots), 2)

    def test_multiline_shot_body_is_kept(self):
        shots, _ = parse_shots("[Shot 1] she walks in\nand shakes off the rain\n[Shot 2] b", 10.0)
        self.assertIn("shakes off the rain", shots[0]["prompt"])
        self.assertEqual(len(shots), 2)

    def test_unstamped_shots_spread_evenly(self):
        shots, _ = parse_shots("[Shot 1] a\n[Shot 2] b\n[Shot 3] c", 9.0)
        self.assertAlmostEqual(shots[0]["start"], 0.0)
        self.assertAlmostEqual(shots[1]["start"], 3.0)
        self.assertAlmostEqual(shots[2]["start"], 6.0)

    def test_partial_timing_is_respected(self):
        """Stamp the two moments you care about; the rest fall between."""
        shots, _ = parse_shots("[Shot 1] a\n[Shot 2] b\n[00:08.000] c", 12.0)
        self.assertAlmostEqual(shots[0]["start"], 0.0)
        self.assertAlmostEqual(shots[2]["start"], 8.0)
        self.assertGreater(shots[1]["start"], 0.0)
        self.assertLess(shots[1]["start"], 8.0)

    def test_starts_are_strictly_increasing(self):
        shots, _ = parse_shots("[Shot 1] a\n[Shot 2] b\n[Shot 3] c\n[Shot 4] d", 10.0)
        starts = [s["start"] for s in shots]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(starts), len(set(starts)))

    def test_backwards_timecode_is_reported_and_ignored(self):
        shots, notes = parse_shots("[00:05.000] a\n[00:02.000] b", 10.0)
        self.assertTrue(any("backwards" in n for n in notes))
        starts = [s["start"] for s in shots]
        self.assertEqual(starts, sorted(starts))

    def test_durations_tile_the_timeline(self):
        shots, _ = parse_shots("[Shot 1] a\n[Shot 2] b\n[Shot 3] c", 12.0)
        for a, b in zip(shots, shots[1:]):
            self.assertAlmostEqual(a["start"] + a["duration"], b["start"], places=6)
        self.assertAlmostEqual(shots[-1]["start"] + shots[-1]["duration"], 12.0, places=6)

    def test_shot_ordinals_follow_position_not_the_typed_number(self):
        """A user who reorders paragraphs without renumbering means the new order."""
        shots, _ = parse_shots("[Shot 5] first\n[Shot 2] second", 10.0)
        self.assertEqual(shots[0]["prompt"], "first")
        self.assertEqual(shots[1]["prompt"], "second")

    def test_empty_input_yields_no_shots(self):
        self.assertEqual(parse_shots("", 10.0)[0], [])
        self.assertEqual(parse_shots("   \n  ", 10.0)[0], [])

    def test_shot_ids_are_unique(self):
        shots, _ = parse_shots("[Shot 1] a\n[Shot 2] b\n[Shot 3] c", 9.0)
        ids = [s["id"] for s in shots]
        self.assertEqual(len(ids), len(set(ids)))

    def test_reference_tokens_survive_parsing(self):
        """@Name must reach the compiler untouched -- parsing must not eat it."""
        shots, _ = parse_shots("[Shot 1] @Mimi walks past @[Cafe wide]", 5.0)
        self.assertIn("@Mimi", shots[0]["prompt"])
        self.assertIn("@[Cafe wide]", shots[0]["prompt"])

    def test_quoted_dialogue_survives_parsing(self):
        shots, _ = parse_shots('[Shot 1] he says "you came"', 5.0)
        self.assertIn('"you came"', shots[0]["prompt"])


class TestGlobalPrompt(unittest.TestCase):
    def test_unlabelled_text_becomes_style(self):
        fields, _ = parse_global_prompt("Shot on 35mm, warm practical light.")
        self.assertEqual(fields["style_line"], "Shot on 35mm, warm practical light.")

    def test_labelled_blocks(self):
        fields, _ = parse_global_prompt(
            "style: 35mm, warm light\n"
            "identity: Mimi always wears the red scarf\n"
            "soundscape: rain outside\n"
            "music: sparse piano")
        self.assertEqual(fields["style_line"], "35mm, warm light")
        self.assertEqual(fields["identity_notes"], "Mimi always wears the red scarf")
        self.assertEqual(fields["overall_soundscape"], "rain outside")
        self.assertEqual(fields["non_diegetic_music"], "sparse piano")

    def test_label_aliases(self):
        fields, _ = parse_global_prompt("look: noir\nscore: strings\nsubjects: he is tall")
        self.assertEqual(fields["style_line"], "noir")
        self.assertEqual(fields["non_diegetic_music"], "strings")
        self.assertEqual(fields["identity_notes"], "he is tall")

    def test_labels_are_case_insensitive(self):
        fields, _ = parse_global_prompt("STYLE: noir\nMusic: strings")
        self.assertEqual(fields["style_line"], "noir")
        self.assertEqual(fields["non_diegetic_music"], "strings")

    def test_multiline_block_accumulates(self):
        fields, _ = parse_global_prompt("identity: line one\nline two\nmusic: piano")
        self.assertIn("line one", fields["identity_notes"])
        self.assertIn("line two", fields["identity_notes"])
        self.assertNotIn("piano", fields["identity_notes"])

    def test_leading_unlabelled_text_before_a_label(self):
        fields, _ = parse_global_prompt("noir, handheld\nmusic: strings")
        self.assertEqual(fields["style_line"], "noir, handheld")
        self.assertEqual(fields["non_diegetic_music"], "strings")

    def test_empty(self):
        fields, _ = parse_global_prompt("")
        self.assertEqual(fields["style_line"], "")

    def test_flattened_global_prompt_still_splits(self):
        fields, _ = parse_global_prompt("style: noir  music: strings")
        self.assertEqual(fields["style_line"], "noir")
        self.assertEqual(fields["non_diegetic_music"], "strings")


if __name__ == "__main__":
    unittest.main()
