"""Parsing the two authored prompt boxes into timeline structure.

The node face carries two multiline boxes:

  GLOBAL PROMPT   art style, lighting, camera rules, identity locks, score.
                  Compiles into subject_definitions and retention_analysis.
  SHOT PROMPT     timecoded shots, in [Shot N] / [MM:SS.mmm] form.
                  Compiles into detailed_description.

Both are plain text, which means both can arrive damaged. A browser widget that
strips line breaks turns

    [Shot 1] she walks in
    [Shot 2] he looks up

into "[Shot 1] she walks in [Shot 2] he looks up", and a naive parser sees one
shot containing a literal bracket. `normalize_prompt_text` repairs that before
parsing. It is belt-and-braces behind the multiline widgets, not a replacement
for them -- the widgets are supposed to preserve newlines, and there is a test
asserting the round trip.
"""

import re

from .frames import seconds_to_frames

__all__ = [
    "normalize_prompt_text",
    "parse_timecode",
    "parse_shots",
    "parse_global_prompt",
]

# [Shot 3]  /  [shot 3]  -- the ordinal is advisory; position in the text wins,
# because a user who reorders paragraphs without renumbering means the new order.
_SHOT_MARKER = re.compile(r"^\[\s*shot\s*(\d+)\s*\]\s*", re.IGNORECASE)

# [00:04.500]  /  [0:04]  /  [00:04.5]  as a leading marker
_TIME_MARKER = re.compile(r"^\[\s*(\d{1,3}):(\d{1,2}(?:\.\d{1,3})?)\s*\]\s*")

# "At 00:04.500," following a shot marker, which is the form the compiler emits,
# so a user who pasted a previous compile back in still parses.
_AT_TIME = re.compile(r"^at\s+(\d{1,3}):(\d{1,2}(?:\.\d{1,3})?)\s*,?\s*", re.IGNORECASE)

# A bare leading timecode with no brackets: "00:04.500 she sits"
_BARE_TIME = re.compile(r"^(\d{1,3}):(\d{2}(?:\.\d{1,3})?)\s+")

_LABELLED = re.compile(
    r"^\s*(style|look|identity|subjects?|retention|soundscape|overall[_ ]soundscape|"
    r"music|non[_ ]diegetic[_ ]music|score)\s*:\s*", re.IGNORECASE)

_LABEL_CANON = {
    "style": "style", "look": "style",
    "identity": "identity", "subject": "identity", "subjects": "identity",
    "retention": "retention",
    "soundscape": "soundscape", "overall_soundscape": "soundscape",
    "overall soundscape": "soundscape",
    "music": "music", "score": "music",
    "non_diegetic_music": "music", "non diegetic music": "music",
}


def normalize_prompt_text(text):
    """Restore line structure to text that lost it in transit.

    Three repairs, in order:
      1. line endings normalised to \\n
      2. "] [" -> "]\\n["    (adjacent bracketed markers collapsed onto one line)
      3. runs of two or more spaces -> newline

    Rule 3 is deliberately blunt, and it is safe here for a specific reason: this
    text is prose destined for a prompt, where a double space carries no meaning
    the model reads. Converting it costs nothing if the newlines survived, and
    recovers the shot structure if they did not. Single spaces are never touched,
    so ordinary sentences are unaffected.
    """
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # Two markers written back to back: "[Shot 1][Shot 2]" or "] ["
    text = re.sub(r"\][ \t]*\[", "]\n[", text)
    # The common case, and the one the brief describes: a shot or timecode marker
    # sitting inline after the previous shot's text, because the line break in
    # between was eaten. Only *marker-shaped* brackets trigger this, so ordinary
    # bracketed prose is left alone.
    text = re.sub(
        r"[ \t]+(?=\[\s*(?:shot\s*\d+|\d{1,3}:\d{1,2}(?:\.\d{1,3})?)\s*\])",
        "\n", text, flags=re.IGNORECASE)
    # A marker preceded by run-on spacing.
    text = re.sub(r"[ \t]{2,}(?=\[)", "\n", text)
    # Any remaining run of two or more spaces was almost certainly a line break.
    text = re.sub(r"[ \t]{2,}", "\n", text)
    # Collapse blank-line runs but keep single blank lines as paragraph breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_timecode(minutes, seconds):
    return int(minutes) * 60 + float(seconds)


def _strip_markers(line):
    """Pull any leading shot/time markers off a line.

    Returns (explicit_seconds_or_None, remaining_text).
    """
    explicit = None
    text = line.strip()

    matched = True
    while matched:
        matched = False
        m = _SHOT_MARKER.match(text)
        if m:
            text = text[m.end():]
            matched = True
            continue
        m = _TIME_MARKER.match(text)
        if m:
            explicit = parse_timecode(m.group(1), m.group(2))
            text = text[m.end():]
            matched = True
            continue
        m = _AT_TIME.match(text)
        if m:
            explicit = parse_timecode(m.group(1), m.group(2))
            text = text[m.end():]
            matched = True
            continue
        m = _BARE_TIME.match(text)
        if m:
            explicit = parse_timecode(m.group(1), m.group(2))
            text = text[m.end():]
            matched = True
    return explicit, text.strip()


def _is_shot_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_SHOT_MARKER.match(stripped) or _TIME_MARKER.match(stripped)
                or _BARE_TIME.match(stripped))


def parse_shots(text, total_duration=None, fps=24):
    """Parse the shot box into shot dicts. Returns (shots, diagnostics).

    A line beginning with a shot or time marker opens a new shot; everything
    after it, up to the next marker, belongs to it. Text before the first marker
    becomes an opening shot at t=0, so a user who just types a paragraph gets one
    shot rather than nothing.

    Shots without an explicit timecode are spread evenly between the ones that
    have them, so partial timing is usable -- you can stamp the two moments you
    care about and let the rest fall where they land.
    """
    diagnostics = []
    text = normalize_prompt_text(text)
    if not text:
        return [], diagnostics

    # ── split into blocks on marker lines ───────────────────────────────────
    blocks = []  # (explicit_seconds_or_None, [lines])
    for line in text.split("\n"):
        if _is_shot_line(line):
            explicit, rest = _strip_markers(line)
            blocks.append([explicit, [rest] if rest else []])
        elif blocks:
            blocks[-1][1].append(line)
        elif line.strip():
            blocks.append([None, [line]])

    shots = []
    for explicit, lines in blocks:
        body = "\n".join(lines).strip()
        if not body:
            continue
        shots.append({"start": explicit, "prompt": body})

    if not shots:
        return [], diagnostics

    # ── monotonicity of the explicit stamps ─────────────────────────────────
    last = None
    for shot in shots:
        if shot["start"] is None:
            continue
        if last is not None and shot["start"] < last:
            diagnostics.append(
                "timecode %s runs backwards (after %s); shots are compiled in written order "
                "and this stamp was ignored"
                % (_fmt(shot["start"]), _fmt(last)))
            shot["start"] = None
            continue
        last = shot["start"]

    # ── fill the gaps ───────────────────────────────────────────────────────
    if total_duration is None:
        known = [s["start"] for s in shots if s["start"] is not None]
        total_duration = (max(known) + 2.0) if known else float(len(shots) * 2)
    total_duration = max(float(total_duration), 0.001)

    if shots[0]["start"] is None:
        shots[0]["start"] = 0.0

    n = len(shots)
    i = 0
    while i < n:
        if shots[i]["start"] is not None:
            i += 1
            continue
        # Find the next anchored shot and spread the unstamped run between them.
        j = i
        while j < n and shots[j]["start"] is None:
            j += 1
        prev = shots[i - 1]["start"]
        nxt = shots[j]["start"] if j < n else total_duration
        span = max(nxt - prev, 0.0)
        step = span / float(j - i + 1) if span > 0 else 0.0
        for k in range(i, j):
            shots[k]["start"] = prev + step * (k - i + 1)
        i = j

    # ── durations, ids, and a final strict-increase pass ────────────────────
    out = []
    for idx, shot in enumerate(shots):
        start = float(shot["start"])
        end = float(shots[idx + 1]["start"]) if idx + 1 < len(shots) else total_duration
        if idx + 1 < len(shots) and end <= start:
            # Two shots landed on the same instant; nudge by one frame so the
            # compiler's strictly-increasing timestamp rule has something to work
            # with rather than having to invent the gap itself.
            end = start + 1.0 / fps
            shots[idx + 1]["start"] = end
            diagnostics.append(
                "shots %d and %d landed on the same instant; shot %d nudged forward one frame"
                % (idx + 1, idx + 2, idx + 2))
        out.append({
            "id": "shot_%d" % (idx + 1),
            "start": start,
            "duration": max(0.0, end - start),
            "prompt": shot["prompt"],
        })
    return out, diagnostics


def _fmt(seconds):
    seconds = max(0.0, float(seconds))
    return "%02d:%06.3f" % (int(seconds // 60), seconds - int(seconds // 60) * 60)


def parse_global_prompt(text):
    """Parse the global box into the timeline's project-level fields.

    Labelled blocks are recognised at the start of a line:

        style:      / look:                -> detailed_description's opening line
        identity:   / subject:             -> subject_definitions
        retention:                         -> retention_analysis
        soundscape: / overall_soundscape:  -> overall_soundscape
        music:      / score:               -> non_diegetic_music

    Anything before the first label is treated as style, so a user who types a
    paragraph of look-and-feel with no labels at all still gets the right thing.

    Returns (fields, diagnostics).
    """
    diagnostics = []
    text = normalize_prompt_text(text)
    fields = {"style_line": "", "identity_notes": "", "retention_notes": "",
              "overall_soundscape": "", "non_diegetic_music": ""}
    if not text:
        return fields, diagnostics

    buckets = {"style": [], "identity": [], "retention": [], "soundscape": [], "music": []}
    current = "style"
    for line in text.split("\n"):
        m = _LABELLED.match(line)
        if m:
            label = m.group(1).lower().replace(" ", "_")
            current = _LABEL_CANON.get(label, _LABEL_CANON.get(label.replace("_", " "), "style"))
            rest = line[m.end():].strip()
            if rest:
                buckets[current].append(rest)
            continue
        if line.strip():
            buckets[current].append(line.strip())

    fields["style_line"] = " ".join(buckets["style"]).strip()
    fields["identity_notes"] = "\n".join(buckets["identity"]).strip()
    fields["retention_notes"] = "\n".join(buckets["retention"]).strip()
    fields["overall_soundscape"] = " ".join(buckets["soundscape"]).strip()
    fields["non_diegetic_music"] = " ".join(buckets["music"]).strip()
    return fields, diagnostics
