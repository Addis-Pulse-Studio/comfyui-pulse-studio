"""Widget state: merging node-face values into a timeline without losing any.

THE ERASURE BUG THIS EXISTS TO KILL
-----------------------------------
The Asset Bin panel used to own the timeline JSON. It parsed the widget, edited
the asset list, and serialised the whole document back. Its parse step had a
fallback:

    try   { parsed = JSON.parse(widget.value || "{}") }
    catch { return { assets: [] } }          // <-- everything else is gone

Any path that reached that fallback -- an empty widget, a value the frontend had
momentarily swapped out, a single malformed character -- produced a document
containing *only* an asset list. The next write serialised that stripped object
over the real one, taking shots, duration, style and prompts with it. Queuing
triggered it because queuing serialises widgets, and the panel re-rendered from
whatever it saw at that instant. Guarding with "if the canvas is empty, skip"
would have hidden it; the fallback is the bug.

Three structural fixes, none of them a guard:

  1. Prompts live in their own dedicated widgets, never inside timeline_data.
     The bin panel cannot erase what it does not store.
  2. Parse failure is an error, never a default. `load_timeline_document` returns
     an error instead of an empty document, and callers refuse to write.
  3. Edits are merges, not replacements. `apply_bin_operation` changes the asset
     list and copies every other key through untouched, including keys this
     version has never heard of.

Rule, enforced by test: nothing in the execute path ever writes to a widget.
Compilation reads widgets and builds an in-memory Timeline. Widget values change
only when the user changes them.
"""

import json

from .assets import AssetBin, BudgetError
from .binops import apply_operation
from .constants import TIMELINE_SCHEMA
from .parsing import parse_global_prompt, parse_shots
from .timeline import Timeline

__all__ = [
    "DocumentError",
    "load_timeline_document",
    "dump_timeline_document",
    "apply_bin_operation",
    "build_timeline",
    "empty_document",
    "EMPTY_TIMELINE_DATA",
    "PROMPT_WIDGETS",
]

# Widgets the bin panel and the execute path must never write to. Named here so
# the guarantee is testable rather than merely intended.
PROMPT_WIDGETS = ("global_prompt", "shot_prompt")


def empty_document():
    """A new, empty asset-bin document in schema 2 form.

    The `cast` key is written from 1.0 even though nothing populates it until
    1.1. Shipping the key empty means 1.1 adds entries to a key that already
    exists; shipping without it means 1.1 has to migrate a file format that is
    already saved inside users' workflows. The first costs one line now.
    """
    return {"schema": TIMELINE_SCHEMA, "assets": [], "cast": []}


# The `timeline_data` widget default. A literal rather than a computed value so
# that what ComfyUI stamps into a fresh node is exactly what the tests assert.
EMPTY_TIMELINE_DATA = '{"schema": 2, "assets": [], "cast": []}'


class DocumentError(ValueError):
    """timeline_data that cannot be read. Never silently replaced with a default."""


def load_timeline_document(raw):
    """Parse timeline_data into a dict, preserving unknown keys.

    An empty or whitespace value is a legitimately new document and yields an
    empty schema 2 document. Anything that is present but unparseable raises,
    because the alternative -- returning a blank document -- is what destroyed
    the user's work.

    A schema 1 document (a bare `{"assets": [...]}` with no `schema` key, which
    is what every pre-2.0.0 build wrote) loads with an empty cast and is
    upgraded in place. Because `apply_bin_operation` dumps whatever this
    returns, the upgrade persists the next time the bin is edited -- there is no
    separate migration pass to forget to run.
    """
    if raw is None:
        return empty_document()
    text = raw.strip() if isinstance(raw, str) else raw
    if not text or text in ("{}", "null"):
        return empty_document()
    if isinstance(text, dict):
        document = dict(text)
    else:
        try:
            document = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise DocumentError(
                "timeline_data is present but could not be parsed (%s). Refusing to "
                "replace it with an empty document -- fix or clear the widget instead."
                % (exc,))
    if not isinstance(document, dict):
        raise DocumentError("timeline_data must be a JSON object, got %s"
                            % type(document).__name__)
    if not isinstance(document.get("assets"), list):
        document["assets"] = []
    # Schema 1 -> 2. Additive only: an absent `schema` key means a pre-2.0.0
    # file, and the upgrade is to say so and to give `cast` somewhere to live.
    # Nothing is removed and no existing key is rewritten, so a document from a
    # *future* schema passes through untouched rather than being downgraded.
    if "schema" not in document:
        document["schema"] = TIMELINE_SCHEMA
    if not isinstance(document.get("cast"), list):
        document["cast"] = []
    return document


def dump_timeline_document(document):
    return json.dumps(document, ensure_ascii=False)


def apply_bin_operation(raw, operation, **kwargs):
    """Apply one Asset Bin edit to a timeline document. Returns (new_raw, error).

    Every key other than "assets" is copied through untouched -- including keys
    from a future version this code does not recognise. On refusal the original
    string is returned unchanged, so a rejected edit can never damage the
    document.
    """
    try:
        document = load_timeline_document(raw)
    except DocumentError as exc:
        return raw, str(exc)

    try:
        bin_ = AssetBin.from_list(document["assets"])
        apply_operation(bin_, operation, **kwargs)
    except (BudgetError, KeyError, ValueError) as exc:
        return raw, (str(exc) or repr(exc))

    merged = dict(document)          # preserve shots, duration, style, unknowns
    merged["assets"] = bin_.to_list()
    return dump_timeline_document(merged), None


def build_timeline(raw, global_prompt="", shot_prompt="", duration_seconds=None,
                   fps=24, branch=None, continuation_branch=None,
                   first_frame=None, last_frame=None, window_seconds=None,
                   dialogue_language="English"):
    """Assemble the Timeline that will be compiled, without mutating anything.

    Reads the stored document for assets (and for canvas-authored shots, when the
    shot box is empty), parses the two prompt boxes, and returns a fresh Timeline.
    Nothing here writes back: the widgets are inputs to this function, never
    outputs of it.

    Shot precedence: typed shots win. If the shot box has text it defines the
    shots outright, because it is the thing the user was last looking at. An
    empty box falls back to whatever the canvas stored, so clearing the box does
    not silently discard timeline work.

    Returns (Timeline, diagnostics).
    """
    diagnostics = []
    try:
        document = load_timeline_document(raw)
    except DocumentError as exc:
        diagnostics.append(str(exc))
        document = {"assets": []}

    fields, global_notes = parse_global_prompt(global_prompt)
    diagnostics.extend(global_notes)

    duration = duration_seconds if duration_seconds else document.get("duration_seconds")

    shots = []
    if (shot_prompt or "").strip():
        shots, shot_notes = parse_shots(shot_prompt, duration, fps)
        diagnostics.extend(shot_notes)
        if not shots:
            diagnostics.append(
                "the shot box has text but no shots were parsed from it; check that each "
                "shot begins with [Shot N] or a [MM:SS.mmm] timecode")
    else:
        shots = document.get("shots") or []
        if not shots:
            diagnostics.append(
                "no shots: the shot box is empty and the timeline holds none, so this "
                "render has no direction beyond the global prompt")

    if not duration:
        duration = max([s.get("start", 0) + s.get("duration", 0) for s in shots], default=0.0)

    timeline = Timeline(
        assets=document["assets"],
        shots=shots,
        duration_seconds=duration,
        fps=fps,
        style_line=fields["style_line"] or document.get("style_line", ""),
        overall_soundscape=fields["overall_soundscape"] or document.get("overall_soundscape", ""),
        non_diegetic_music=fields["non_diegetic_music"] or document.get("non_diegetic_music", ""),
        dialogue_language=dialogue_language or document.get("dialogue_language", "English"),
        branch=branch or document.get("branch", "ref2va"),
        continuation_branch=continuation_branch or document.get("continuation_branch"),
        first_frame=first_frame if first_frame is not None else document.get("first_frame"),
        last_frame=last_frame if last_frame is not None else document.get("last_frame"),
        window_seconds=window_seconds or document.get("window_seconds"),
    )
    # Identity locks and retention notes from the global box ride on the timeline
    # so the compiler can put them in subject_definitions / retention_analysis.
    timeline.identity_notes = fields["identity_notes"] or document.get("identity_notes", "")
    timeline.retention_notes = fields["retention_notes"] or document.get("retention_notes", "")
    return timeline, diagnostics
