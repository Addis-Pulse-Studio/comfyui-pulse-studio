# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### A recording says where on the film clock it starts

`PulseVoice` is a new node: one recording, plus what it is for, whose it is, and
**where it begins**. Its `PULSE_VOICE` output goes into `PulseSlate`'s new
`voices.voice_1..3` group — visible to every window — or into one `PulseShot`'s
new `voice` input, for that shot alone.

#### The failure it closes

A `lip_sync` clip is cut to the window it rides in. That makes *which seconds of
the file* the question the whole feature turns on, and nothing in the graph could
answer it: `render._audio_for` trimmed at `window.start_seconds`, which is the
**film's** clock, and said so nowhere. The rule was documented in one example
workflow's MarkdownNote and contradicted by `PulseShot.ref_audio`'s own tooltip
("a voice or effect sample for this scene only" — true about scope, false about
time).

Wire a per-shot take, which is what that tooltip invites, and
`concat.audio_span_bounds` clamped the start to the end of the buffer: window 3
asked for the seconds from 24.50s of a 12-second recording and got padding.
Every one of them. The render succeeded, the report said nothing, and the mouth
did not move — which is indistinguishable from the model being bad at lip sync,
so the failure was attributed to H3 and the graph was never looked at again.

`Asset.audio_offset` is where a recording's first sample sits on the film clock.
The default is `0.0`, which is what every project written before it existed
means, so **no existing render changes** — `window.start_seconds - 0.0` is the
call `_audio_for` has always made. `PulseVoice.aligns_to` is how a graph says
otherwise: `film_clock` for a narration spanning the timeline, `shot_start` for a
take recorded for one shot, plus an `offset_seconds` on top for room tone in
front of it.

It is a combo rather than a number because "this recording starts when the shot
starts" should be something the graph states, not arithmetic the author redoes
every time a duration changes.

#### It is measured, not assumed

`concat.audio_span_bounds` already computed how many samples of a span were
silence and `media.audio_span` threw the number away. It is now returned as
`(start, stop, head, tail)` and reported:

> `@Voice` covers 0.00-12.40s on the film clock and window 3 covers 24.50-36.75s.
> The two do not meet, so nothing of this recording reaches this window.

with a partial form for a narration that merely runs out early.

What an uncovered span *looks* like depends on whether the shots still ask for
speech, and the first draft of this check got it wrong. With no dialogue the
mouth is genuinely still. With dialogue it is not still at all: the `<d>` block
instructs the model to say those words, so it generates speech and the mouth
moves while `use_reference_audio` mixes the reference's silence over the top.
Confirmed on a real render — a 7.6s narration in a 13.88s film, mouth visibly
mid-syllable at 10.0s and 11.7s with a peak-zero track underneath. Saying "the
mouth is still" there sends the author looking for a longer recording when the
fix may be to cut the words, so the two cases now read differently.

A trailing gap on the *last* window is held to a whole second rather than a
quarter, because the film ending is not a mistake. Both graphs this was first run
against tripped the quarter-second rule on a correct film — 0.46s on a 45s
explainer, 0.37s on an 8s one — and a warning channel that fires on correct films
is worth less than none. A gap anywhere else keeps the quarter second, and
**lead-in** silence is never relaxed at either end: a recording that starts late
is late wherever it happens.

`source_seconds` is measured in the node layer — from the tensor for a socket, from the container
header for a file — because `compiler.py` imports no torch and must not. A
recording nothing could measure is never diagnosed: `None` means unknown, and
unknown is not the same as empty.

A **negative** offset is no longer clamped. Clamping was silent: a recording
starting three seconds into the film was slid forward to the window's opening and
every lip in it then ran three seconds early. It now leads in with silence, which
is what "the recording has not started yet" actually sounds like.

#### The sub-frame straddle, fixed at its source

`PulseShot.duration_seconds` capped at `15.08` while a full 362-frame window is
`15.08333s`, so a chain of full-length shots could not put a single cut on a
seam. Every one landed 3ms short, `Shot.overlaps` placed the shot in both
windows, and — because a shot's reference audio travels with it — each window was
handed **two** lip-sync recordings describing the same seconds. Two "lip
movements match" directives, one mouth, nothing reported.

`Timeline.shots_in` now applies a one-frame floor: a window renders whole frames,
so a shot occupying less than one of them does not appear in it. Two fallbacks
keep it honest — with fewer than two shots there is nothing to trim towards, and
a window whose every shot is a sliver keeps them, because rendering on the style
line alone is worse than the duplicate. `report.straddling_shots` still names
whatever remains. The widget cap is now `MAX_WINDOW_FRAMES / FPS`, so a shot may
be exactly as long as the window it renders in.

Two lip-sync references surviving in one window is still reported, and still not
repaired: choosing which copy to discard means guessing which shot the author
meant, and guessing wrong desynchronises the mouth that was right.

#### Prompt and mux

- A `lip_sync` reference now gets its `fully_copy` `retention_analysis` line even
  when no `speaker` names its owner. The line was withheld from unbound
  recordings to stop `reference` leaking a voice between characters — a real
  concern for `voice_timbre`, and the wrong rule for `lip_sync`, where "use this
  exact track and move the mouth with it" is true whoever is speaking. The
  commonest graph there is — a one-hander, where nobody fills in `speaker`
  because there is nothing to confuse — was asking for lip sync in
  `subject_definitions` and saying nothing about it in `retention_analysis`.
- `PulseRender.use_reference_audio` mixes **every** lip-sync recording in a
  window, each at its own offset. It took the first one it found and `break`,
  so a two-hander silently lost a character from the finished film.
- The "quoted dialogue and a `lip_sync` reference" note read backwards. A quote
  that *is* the recording's transcript is the recommended shape — it hands the
  text encoder the words the waveform it cannot hear is carrying — so the old
  wording fired on every shot of a narration graph doing exactly what the
  documentation asks for. The correct case is stated first now.

#### Nine tests that had never run

`media.py` is a top-level module — `pyproject` declares it a py-module beside
`nodes` and `render`, because those three need torch and comfy and the stdlib-only
core must never see either. Three of its functions nonetheless reached into the
core with a *relative* import, done lazily inside the function body:

```python
from .comfyui_pulse_studio.concat import audio_span_bounds
```

Inside ComfyUI that resolves, because the pack is loaded as a package. Anywhere
else — including the test suite, which imports `media` top-level as the manifest
says it is — it raises `ImportError: attempted relative import with no known
parent package`. Being lazy is what made it invisible: importing the module
worked and *calling* those functions did not.

The tests that call them guard themselves with `import torch` in `setUp` and skip
when it is missing, which is every CI runner and every box without a GPU. So the
suite was green on three Python versions while nine of its tests had never once
executed, and they failed the moment it was run where it matters: inside a working
ComfyUI, where torch is present. Eleven errors, `audio_span` and
`treat_audio_seams` among them — the two functions the whole lip-sync trim rests
on.

Now one import at module scope, branching on `__package__` rather than catching
`ImportError`, so a genuine breakage inside `concat` still raises instead of being
swallowed by a fallback. The core is stdlib-only, so importing it eagerly costs
nothing. Two source-text guards in `tests/test_segment_cache.py` moved with it,
and one now asserts the *absence* of a relative import inside a function body.

With torch present the suite is 863 passed, **0 skipped**. Without it, 863 passed
and 16 skipped, as before. CI runs the second of those, so the tensor layer is
still unguarded there — a skip is invisible in a green run.

#### Also

- `js/ps_sockets.js` gains `voices.voice_` **appended after** `shots.shot_`, never
  inserted: the frontend rebuilds that tail in `GROUPS` order and a group placed
  ahead of an existing one lands a saved graph's wires on the wrong sockets. The
  same rule put `voice` after `ref_audio` on `PulseShot`.
- `tests/test_js_guard.py` did not know `PULSE_VOICE` was a connection type, so it
  read the new input as a widget and waved through exactly the insertion it exists
  to catch. It is in `CONNECTION_TYPES` now, and it caught the first draft of the
  shipped graph wiring `voices.voice_1` into `model_fl2va`.
- `audio_offset` reaches the segment cache key, appended after `voice_of` and only
  when non-zero. Two renders identical in every other field are different films,
  and the content digest cannot tell them apart — it hashes the whole file either
  way. A project whose recordings all start with the film keeps every key on disk.
- `example_workflows/PulseSlate_Voice.json`, the fifth shipped graph. Its
  placeholder is the one-second test tone against a 13.88s film, so the first
  thing it does is print the coverage diagnostic — which is the feature, and the
  note says so.

#### Not done here: the version bump

The package version is unchanged at `3.0.0`, and this section is the note for
whoever cuts the release.

**Nothing in this work requires a schema bump.** The widget contract forbids
inserting, reordering, removing, retyping or renaming a widget and permits
appending; none of those happened. `PulseVoice` is a new node that brings its own
`SPECS` table, and `PulseShot.voice` / `PulseSlate.voices.voice_N` are connection
*inputs*, not widgets. Every existing node's widget list is byte-identical.

**The bump is forced from outside.** `comfyui-pulse-studio 3.0.0` is on the
ComfyUI registry with status `NodeVersionStatusBanned`, so that version can never
be republished — and `tests/test_packaging.py` couples the package version to
`SCHEMA_VERSION` deliberately ("one number wearing two hats", after they drifted
once). So publishing at all means moving `SCHEMA_VERSION`, which means moving the
whole widget contract with it, for a release in which no widget changed.

The six edits, in an order that keeps the suite green at each step:

1. `pyproject.toml` — `version = "3.1.0"`.
2. `comfyui_pulse_studio/constants.py` — `SCHEMA_VERSION = "3.1.0"`.
   `tests/test_packaging.py::test_version_matches_schema_version` guards 1 against 2.
3. `js/ps_widget_order.js` — `export const CURRENT = "3.1.0"`.
   `tests/test_js_guard.py` guards 2 against 3.
4. `js/ps_widget_order.js` — add a `"3.1.0"` key to **all seven** entries in
   `SPECS` (`PulseSlate`, `PulseShot`, `PulseVoice`, `PulseRender`, `PulseBench`,
   `PulseRetake`, `PulseStill`), each a copy of that node's `"3.0.0"` list, which
   is unchanged. **Keep the `"3.0.0"` keys.** `readSavedValues` looks the saved
   file's own version up in this table, so deleting them makes every workflow
   already on disk unreadable — the table is a history, not a current state. A
   node with no entry at `CURRENT` reports as drifted rather than as untouched,
   which is why all seven move even though none of them changed.
5. The five graphs in `example_workflows/` — `PulseSlate`'s `widgets_values[0]`
   from `"3.0.0"` to `"3.1.0"`, and the same on every other Pulse node in them.
   `tests/test_workflow.py::test_the_workflow_declares_the_current_schema` asserts
   this, and its expected literal moves too.
6. This file — retitle `[Unreleased]` to `[3.1.0]` with the date.

Re-check `GET https://api.comfy.org/nodes/comfyui-pulse-studio/versions` before
publishing: 3.1.0 has to be free as well as 3.0.0 being unusable.


### A voice is bound to a face

`PulseShot` gains a `speaker` widget, appended after `ref_audio_mode`. Type the
character's `@Name` — a bin asset, or the shot's own `@Ref1` — and the compiler
emits the binding MiniMax's reference format actually specifies.

The sockets never carried this. `comfy/text_encoders/minimax.py` emits the
literal marker `<Audio j>: ` and the waveform never reaches Qwen, so a reference
audio arrives at the model as an ordinal with no owner. On a one-hander that is
enough — there is one mouth. On a two-hander it is the shape of
[Comfy-Org/ComfyUI#15454](https://github.com/Comfy-Org/ComfyUI/issues/15454):
the intended character's lips move, and the other character's accent comes out
of them. `Shot.speakers` has been in the data model and validated since 2.0.0
with nothing writing to it, and `README` said per-character binding "lands in
1.1". This is it.

What a named speaker changes:

- a **global speaker id**, `(S1)`, `(S2)`, assigned at that character's first
  line **on the clock** and unchanged for the rest of the film. Assigned once
  across the whole timeline, deliberately not inside `_compile_window`: the id is
  what tells the model that the person talking after a cut is the person who
  talked before it, and per-window numbering would turn every character into a
  new person at every seam while looking correct in any single-window test;
- the id is stamped only in the shots that character speaks in —
  `<Subject 1> (S1) crosses to the counter`. A character standing silently in
  another shot stays unstamped, because an id there reads as a cue to give them
  a line;
- the shot's `ref_audio` is named as theirs: `` `<Audio 1>` is the speech
  <Subject 1> (S1) is saying `` for `lip_sync`, `` `<Audio 2>` is the
  voice-timbre reference for <Subject 2> (S2) `` for `voice_timbre`;
- a `retention_analysis` line per bound audio, in MiniMax's audio vocabulary
  rather than the picture words — `fully_copy` for `lip_sync`, `reference` for
  `voice_timbre`. `AUDIO_ROLES` keeps its two values, so no saved graph changes
  meaning; the mapping lives in `AUDIO_ROLE_RETENTION`.

Deliberately narrow:

- **naming nobody produces the prompt this pack produced before the field
  existed**, byte for byte, and `speaker_binding` is omitted from the document
  rather than written as `""`. A project that does not use speakers keeps every
  cache key already on disk.
- **bin audio is left unbound.** It is shared across the film, so there is no
  shot to read a speaker off, and guessing one would bind the voice to whoever
  happened to talk first.
- **a name that matches nothing is reported, never guessed at.** Binding a voice
  to the wrong face is worse than leaving it unbound. Scene-local references are
  searched before the bin, so two shots may each name their own `@Ref1` and mean
  different people (§10).
- a speaker whose reference was pushed out of this window's budget by carry-over
  loses the id for that window and is diagnosed, rather than emitting `(S1)`
  against a picture that is not there.

`speaker_binding` — the *resolved* `"<Subject 2> (S1)"`, not the asset id —
joins the shot's cache key, appended only when set. The asset id alone would not
move when an earlier shot gains a character and renumbers everyone downstream,
and the number is what reaches the model. Same hole `audio_role` had to be
plugged for: the binding sentence lives in the window's subject definitions,
which no shot's `resolved_prompt` covers.

`PulseSlate_Cast.json` now demonstrates it — `@Mimi` is `(S1)` with a `lip_sync`
take, `@Kade` is `(S2)` with a `voice_timbre` one. `tests/test_speakers.py`
covers the assignment, the seam, the two audio roles, the unbound prompt, the
cache key and the name resolution.

### More than one voice in the bin

`speaker` answers "whose voice is this" for a shot's own `ref_audio` — the node
carrying the recording also carries the character. A recording dropped in the
Asset Bin has no shot to read a speaker off, so a film with three characters and
three voice files could bind none of them. `Asset.voice_of` closes that.

- **bound by asset id**, never by ordinal or slot. A binding to "reference 3"
  follows whatever lands in slot 3 after the next bin edit and reports nothing —
  the failure the whole asset module exists to prevent. Refused against an id
  that is not in the bin, and against another recording: a voice belongs to
  somebody the model can see.
- **two new bin operations**, `set_voice_of` and `set_audio_role`, with controls
  on the panel's audio rows. `audio_role` on a bin recording had no control at
  all, so every one of them compiled as a timbre reference and only a
  `PulseShot` socket could ever ask for lip sync. `bin_state` now serves the
  cast as `(id, name)` — the picker offers a name and stores an id, so a rename
  cannot desynchronise a binding.
- **supplying a voice makes its owner a speaker**, numbered after everyone who
  has a line, so adding a voice reference cannot renumber a character already on
  screen.
- **an explicit `voice_of` outranks the shot that carries the recording.** The
  author naming a character beats the wiring implying one.

`voice_of` joins the reference cache key, appended after `audio_role` and only
when set. While fixing that: the *global* reference descriptor never passed
`audio_role` either, so a bin recording's role changed the prompt and not the
key. Survivable only while nothing could set it; both are passed now.

### Carry-over stops evicting whichever voice was last

A continuation window's carry-over claims the front of the audio group, and the
overflow was dropped by bin position. On a three-voice cast that is a coin flip:
a character keeps their picture, keeps their lines, and loses their voice from
window 2 onward — which reads as drift, not as a missing reference.

Eviction is role-aware now: unbound clips go first, then `voice_timbre`, then
`lip_sync`. Losing an alignment desynchronises a mouth, which is visible; losing
a timbre reference only changes how a voice sounds. The drop is still reported,
and the **survivors keep their bin order** — ordinals are bin order, and
re-sorting them would renumber a window that is not over budget at all.

### The §12.6 sink warning was counting the wrong thing

`_paired_audio_count` returned `len(by_kind(KIND_AUDIO))`, and the warning it
feeds asserts "this timeline pairs N audio references with separate characters".
That assertion was never true: three ambience beds or three narration takes
tripped a per-character voice-drift warning at a film where no character had a
voice at all. A warning channel that cries wolf is worth less than no channel —
the rule `_inert_audio_modes` is built around.

It counts distinct characters carrying a bound voice now, from both paths — a
bin recording's `voice_of` and a shot's `speaker`. Two recordings bound to one
character count once; any number of unbound clips count for nothing, because
there is no pairing for a cheaper `sink_conditioning` to damage. The warning
text was reworded to match what it now measures.

### Not done, and why

- **`(Sx)` in fl2va.** Base mode has no `<Subject N>` layer and
  `_window_bin` returns an empty bin for it, so a speaker there cannot be cited
  by any tag — the id would have to be appended to the shot's `<d>` block
  instead. MiniMax's placement for that is not something this repo can verify
  against ComfyUI source the way the ref2va tags were, and guessing where a
  literal goes in a prompt format is how silent wrong output happens. Left
  alone, deliberately.
- **A retention line for an *unbound* recording.** MiniMax's worked example
  lists every reference in both sections, and this emits an audio retention line
  only when the voice is bound. `fully_copy` against an unattributed recording
  tells the model to reproduce a voice without saying whose mouth it comes out
  of, which is the leak the binding exists to close — and the window prompt is
  not itself hashed (§7.1 hashes shot text and reference descriptors), so
  emitting it unconditionally would change what every existing project is told
  without moving a single cache key.
- **Transcript-driven shot timing.** `audio_span` puts the waveform on the right
  span, but the `[Shot N] At MM:SS.mmm` breaks are still hand-authored, so a
  `lip_sync` shot can be asked to match words that land elsewhere in the clip.
  The fix is a sidecar transcript parsed with stdlib — SRT/VTT/JSON, arithmetic
  testable on a GPU-less box like `concat.audio_span_bounds` already is. Not
  started; it is a feature, not a correction.

`PulseSlate_Cast.json` is now the worked example for all of it, and shrank to a
single 12-second window to be one: at 16 seconds it spanned a seam, and
carry-over then claimed an audio slot that the three voice references could not
spare. It also exposed the eviction bug in the shipped graph — window 2 dropped
`@Kade`'s voice, not the unbound recording, so a character kept his picture and
his lines and lost his voice at the cut. The bin recording is bound to `@Mimi` as
her film-wide timbre reference; `@Mimi` also carries a per-shot lip-sync take and
`@Kade` a per-shot timbre one, so every audio reference in the graph now names a
character and every one carries a stable speaker id. Its `lip_sync` shot stopped
carrying a quoted line as well — the compiler reported that conflict, and a
shipped example should demonstrate the fix the README prescribes rather than the
conflict. Seams and the segment cache stay LongForm's job.

`tests/test_voice_binding.py` covers the field, two voices in one bin, the
speaker grant, the refusals, both bin operations, the cache key, the eviction
order and the count. It also asserts the case that renumbers everything —
enabling a video's soundtrack claims an `<Audio j>` ordinal ahead of every
standalone recording, so every bound sentence moves and must still name the same
character.

### The canvas is the shape it claims to be

`16:9 landscape` resolved to **1344x736**: 4.2% under H3's 1,032,192 px budget, at
an actual ratio of 1.826 against the 1.778 it promised. Neither as large as it
could be nor the shape written on the widget.

The cause was rounding each axis down to the /32 grid independently. Independent
flooring loses up to a full grid step on *both* axes and the two losses compound
rather than cancel. The long edge still floors; the short edge now takes whichever
grid step is **nearest** the ratio-implied ideal and still inside the budget.

| preset | was | is | budget |
|---|---|---|---|
| 16:9 landscape | 1344x736 | **1344x768** | 95.8% -> 100.0% |
| 9:16 portrait | 736x1344 | **768x1344** | 95.8% -> 100.0% |
| 21:9 ultrawide | 1536x640 | **1536x672** | 95.2% -> 100.0% |
| 1:1 square | 992x992 | 992x992 | unchanged, exactly square |
| 4:3 landscape | 1152x864 | 1152x864 | unchanged, exactly 4:3 |
| 3:4 portrait | 864x1152 | 864x1152 | unchanged |

Nearest, not simply the largest that fits: at 4:3 the largest short edge that fits
is 896, which hits the budget exactly and turns 1.3333 into 1.2857, and taking the
largest everywhere would make the 1:1 preset 992x1024. A preset is a promise about
shape, so shape wins and the budget is filled only where filling it does not break
the promise.

- **This changes what renders.** Width and height are the latent size, so a
  segment cached before this re-renders after it, and a 16:9 film continued across
  the change would step resolution mid-cut if it did not.
- `PulseStill`'s `canvas_from_reference` calls the same function now. The two
  paths reach the same node, and until this landed a 1920x1080 reference and the
  "16:9 landscape" preset resolved to different canvases — 1344x736 against
  1344x768 — purely because each had its own rounding.
- The `height` widget's default moved 736 -> 768 with it, and the shipped graphs
  were re-saved. They were storing 1344x736 for a preset that resolves to
  1344x768: the widgets disagreed with the node, and nothing said so, because the
  two are only compared at queue time. `tests/test_workflow.py` now compares them.
- The arithmetic lives in `comfyui_pulse_studio/canvas.py` rather than in
  `nodes.py`, because `nodes.py` imports torch and that made the canvas — which
  sets every render's latent size — unreachable from the headless suite.

### New — `shot_aligned` windows

A third `partition_strategy`, beside `balanced` and `fill`. It puts window seams
**on shot cuts** wherever the frame grid allows it.

`Timeline.shots_in` is an overlap test, so a shot crossing a seam is compiled into
*both* windows: described twice, timestamped from zero in the second, and hashed
into both window seeds, so editing it rerolls two windows. That is the right
behaviour once a seam falls inside a shot — the alternative is a window with no
direction, which renders as a stall — but it is a cost, and until now the only way
to avoid it was to solve for shot durations by hand.

Alignment is all-or-nothing per seam, never "snap to the nearest cut". Cumulative
position after k windows is 17K + 5k, so a seam can land on a cut at frame p only
when p == 5k (mod 17), and landing *near* a cut is worth exactly nothing: a seam
one frame inside a shot straddles it as much as a seam in the middle does. So the
policy walks the spans between cuts, takes the subset it can hit at once, and
**names the cuts it could not honour** in the report rather than reporting success.
It gives up `window_seconds` to do it, and falls back to `balanced` — saying so —
when no shot boundaries were supplied.

### New — `seam_treatment` on Pulse Render

The container-level seam has been solved since 3.0.0: segment placement is
computed from frame counts, never measured from timestamps, so the picture is
gapless by construction. What was never addressed is that the two sides of a seam
are two *independently generated takes* — a score that restarts, a level that
steps, a grade that drifts. None of that is a timestamp problem.

- **`audio+colour`** (default) — level-matches and tapers each audio join, and
  grades a window's opening towards the frame the previous window ended on,
  decaying over 12 frames. A flat correction across a whole window does not remove
  a cut; it moves it to the next seam.
- **`audio`** — the audio join only.
- **`off`** — both joins untreated. Offered because a treatment that changes
  rendered output should be A/B-able against the untreated join, and because
  colour matching is the kind of correction that occasionally makes things worse.

A window that follows a **cached** segment is never colour-matched: the segment
before it is already encoded, so only one side of that seam can still move, and
grading half a boundary is worse than grading none of it. The report says so and
names the window rather than quietly doing half the job.

### New — descriptions and retention, from the bin panel

`compiler._build_subjects` promotes an image or video to a `<Subject N>`
definition only if it carries a description — which is the shape MiniMax's own
reference format asks for, and what identity consistency is built on. Nothing in
the panel could write one, so the whole mechanism was reachable only by
hand-editing `timeline_data`.

Both fields are now edited on the row, through the same server-side evaluate-and-
apply route as every other bin edit, so the numbering rule keeps one home.

- **description** — empty is accepted rather than refused; clearing one demotes
  the asset back to a plain `<Picture i>` citation, which is a legitimate edit.
  Refused on audio, which has no `<Subject N>` form.
- **retention** — `fully_preserved` or `partially_copy`, sent to the panel from
  `constants.py` rather than hardcoded in JavaScript. The document schema still
  says `"retention": "string"`, because a project with a better phrase for MiniMax
  should be able to use it; the *selector* is closed, because the value is written
  verbatim into a sentence the model reads literally, and a misspelled retention
  instruction is not an error — it is a slightly worse render, which is the
  hardest kind of failure to notice.

### A checkpoint nobody samples with is not free

`check_single_checkpoint` returns `(warnings, notes)` — the only two-tier finding
in the pack, because these two cases differ in severity and collapsing them would
either hide the cheap one or cry wolf about it.

- **Warning, unchanged** — *sampling* through both DiT checkpoints in one
  execution forces an evict-and-reload mid-render on a 32 GB card.
- **Note, new** — `model_fl2va` connected while no window uses it was silent, on
  the reasoning that it is "just a graph ready for either path". That assumed a
  checkpoint nobody samples with is free. It is not: with Spectrum offloading
  history to system RAM on a 32 GB box, a 20 GB checkpoint nobody samples with is
  20 GB the render wanted. Not a mistake — keeping the socket wired is a
  reasonable way to work — so it is a note, it reaches the report rather than the
  node face, and it says what it costs instead of telling anyone what to do.

**`PulseSlate_LongForm.json` was the first thing it caught.** The graph wired the
fl2va checkpoint beside a `continuity` of `none`, which is precisely the
combination that never reaches the anchored branch, so the flagship example
shipped a graph its own report told the user to change. The loader is gone from
it; `PulseSlate_Retake.json` is where fl2va is demonstrated now, and it uses it on
every queue. A test asserts no shipped graph can drift back into that state.

### A mode with nothing to apply it to

`ref_audio_mode` on a `PulseShot` is inert with nothing wired to `ref_audio`: the
compiler emits a lip-sync directive only when there is an audio asset to name, so
the widget read `lip_sync` on the node face and the render came back with a mouth
doing whatever H3 felt like.

Reporting every such shot would have put three warnings on the long-form graph,
which is doing nothing wrong — the widget *defaults* to `lip_sync`, so "lip_sync
and no audio" is the resting state of every shot in every film that uses no
reference audio. The two cases are separated by what the rest of the timeline is
doing: `voice_timbre` with no audio is always reported, because somebody chose it;
`lip_sync` with no audio is reported only when some *other* shot does connect one,
which is a film doing per-shot audio with a shot missed.

### The report answers two more questions

- **Shots across a seam.** Which shots were compiled into two windows, and which
  windows those are. With equal shot durations some shot almost always straddles,
  because N windows sum to a grid total only when N == 1 (mod 17) — and until now
  nothing said which one, so "why did editing that shot re-render two windows"
  had no answer in the report.
- **Per window, after carry-over.** What is left of each per-type reference budget
  once carry-over has taken its slots. The budget meter on the node face counts
  the bin; carry-over spends from the same per-window ceilings, and the difference
  only appeared as a dropped reference at compile time.

Uncited references — a reference occupying a socket that no prompt names — are
listed too. They cost a slot in every window they are attached to.

### Three more example workflows

One graph has shipped since 2026-08-11. Four do now, and none of them needs a
third-party pack to open clean:

- **`PulseSlate_Single.json`** — the short path returns. `PulseSlate` hands back
  `positive` and `latent`, `ModelSamplingMiniMaxH3` carries the flow schedule, and
  the graph's own `SamplerCustomAdvanced` does the rest. It is the only shipped
  graph where those two outputs are live, and the only one carrying a model patch.
- **`PulseSlate_Cast.json`** — the Asset Bin, which is the feature the pack is
  named for: three named references with descriptions and retention values, cited
  by `@Name` and never by ordinal, plus per-shot `ref_audio` demonstrating
  `lip_sync` against `voice_timbre`, and a `PulseStill` feeding a generated
  opening frame into shot 1.
- **`PulseSlate_Retake.json`** — a finished film in, one bad span out. The graph
  that shows why `PulseRetake` takes `model_fl2va` and not the reference
  checkpoint.

The Cast graph opens on a populated bin, which means the files it names have to be
in ComfyUI's `input/` folder — a workflow addresses a reference by filename, so
shipping them inside the pack is not enough. Four placeholders (flat tones and a
sine tone, generated by `tools/make_example_assets.py`) are copied there the first
time the pack loads. Never overwritten: a file already sitting under one of those
names is the user's. The copy is best-effort, so a read-only or headless install
degrades to "Cast opens with unresolved references" rather than to a pack that
fails to load, and it says which.

`tests/test_shipped_assets.py` keeps the exemption narrow — placeholder names,
flat directory, small enough that a photograph could not fit — and regenerates
them from the script to prove the committed bytes and the generator have not
drifted.

### The asset bin's serialised slot must stay last

`ps_asset_bin` is declared `serialize: false`. Frontend 1.49.6 serialises it
anyway, so a saved `PulseSlate` carries one value for it after every
`INPUT_TYPES` widget. That is harmless only while the slot is last: append a
required widget to `PulseSlate` and it is created before `addDOMWidget` runs, so
it lands *in front of* the bin's slot, and every saved file then feeds the bin's
JSON document into the new widget — loading without complaint. `checkWidgetOrder`
now fails loudly when the bin is not last, and CONTRIBUTING.md says what to do
when appending a widget to that node specifically.

### Tests

Every structural check in `tests/test_workflow.py` now runs over all four shipped
graphs. It claimed to already: the tuple existed so that restoring a graph would
be a one-line change, but three tests iterated a fresh `(LONG_FORM,)` literal and
two read `LONG_FORM` directly, so a restored graph would have skipped exactly the
assertions written to catch a bad one. The day the tuple grew, two more
assertions written around the long-form graph's own shape — five loaders, a
director in every graph — failed on the graphs they were meant to cover, and are
stated as rules now.

`test_sigma_shift_is_upstream_of_the_sampler_not_the_director` and
`test_the_short_graph_has_no_render_node` are **restored**, dropped on 2026-08-11
with the graph that was their subject. §1.1's "model patches stay upstream" is
asserted by a test again. §12.3's Sol-after-Sage ordering is still documented
rather than enforced; that needs the Spectrum+Sage graph back, and it is not.

New rules, each of which failed on a real shipped graph the day it was written:
the stored canvas must match the preset the graph selects; the two DiT sockets
must never share a loader; no graph may wire a checkpoint nothing in it samples
with.

### Removed — two of the three example workflows

`PulseSlate_Starter.json` and `PulseSlate_Starter_SpectrumSage.json` are gone.
`PulseSlate_LongForm.json` is the only graph that ships.

It is the one that shows what 3.0.0 is for — `PulseSlate` → `PulseRender`,
`PulseShot` nodes, the segment cache — and the only one that needs no third-party
pack to load clean. The Spectrum+Sage variant needed three of them installed or it
opened to a screen of red nodes, which is a poor thing for a sole example to do.

**3.0.0 shipped with all three; its entry below is left as it was.** That release
is on the registry and its record should describe what was actually in it.

What went with them, stated plainly because none of it is free:

- **§1.1's upstream rule and §12.3's Sol-after-Sage ordering are no longer
  asserted by any test.** `TestThePatchChainVariant` walked the shipped chain and
  proved Sol sat downstream of Sage; the sigma-shift test proved a model patch
  reached the sampler without going through the director. Both graphs are gone, so
  both tests are gone. The rules still hold and are still documented — in the
  README and in the long-form graph's own note — but nothing checks them now.

  **Half of this was undone on 2026-08-17.** `PulseSlate_Single.json` carries a
  sampler and a model patch, so the sigma-shift test has a subject again and is
  restored. §12.3's Sol-after-Sage ordering is still unasserted: it needs the
  Spectrum+Sage graph, which has not come back.
- **The short path has no example.** `PulseSlate` still hands back `positive` and
  `latent` for a ≤15 s timeline; you now wire your own sampler from the README's
  description rather than from a graph.

  **Undone on 2026-08-17** — `PulseSlate_Single.json` is that graph.
- **The patch chain is wired by hand.** The order, the both-inputs detail and the
  `exact_kv_and_rows` setting moved into `PulseSlate_LongForm.json`'s note so the
  §12.3 knowledge survives its graph.

Restoring any of it is `git revert` plus a one-line change to `SHIPPED_GRAPHS` in
`tests/test_workflow.py`, which still loops over a tuple for exactly that reason.

### Still unverified — everything above

Same rule as the 3.0.0 checklist at the bottom of this file: what needs a GPU, a
real frontend or a pair of ears is listed rather than assumed, and stays listed
until somebody writes down what they saw.

- [ ] **A treated seam, heard against an untreated one.** `seam_treatment`
      changes rendered output. Its arithmetic is tested — the taper is symmetric,
      the gain is bounded, the ramp decays — and none of that says whether the
      join sounds better. `off` exists so the comparison can be made; nobody has
      made it.
- [ ] **A colour-matched seam, looked at.** Same, for the picture: the grade is
      applied over 12 frames from the previous window's exit frame, and the case
      where it declines to act (a window following a cached segment) is reported
      but has not been watched.
- [ ] **`shot_aligned` on a real render.** The invariant it exists for is
      asserted on the compiled plan — no shot handed to two windows, every window
      on the grid, the whole timeline covered — but no film has been rendered with
      it, so nobody has seen whether a seam on a cut is in fact less visible than
      a seam inside a shot.
- [ ] **The canvas change against real VRAM.** 16:9 is 1344×768 now: 4.2% more
      pixels per window than the 1344×736 every measurement on this box was taken
      at, including the 362-frame window that only fits with Spectrum offloading
      history to system RAM. It should still fit — the budget is H3's own — but
      "should" is not a measurement.
- [ ] **The placeholder references, installed by a real ComfyUI.** The copy into
      `input/` runs at import against `folder_paths`, which the headless suite
      does not have; what is tested here is that the files exist, regenerate, and
      are named by the Cast graph's bin. Whether they land in `input/` on a real
      install, and whether Cast then opens with every `@reference` resolved, has
      not been seen.
- [ ] **The bin panel's new fields in the frontend.** `description` and
      `retention` are covered on the server side, where the rule lives. The row
      that edits them is JavaScript in a real browser and has not been clicked.
- [ ] **A two-hander rendered with speaker ids.** The binding is asserted against
      the compiled prompt — the ids, the seam, the two audio roles — and the
      prompt is the entire mechanism, so what is untested is whether H3 acts on
      it: whether `@Mimi` and `@Kade` in `PulseSlate_Cast.json` come back with
      two different voices instead of the leak
      [#15454](https://github.com/Comfy-Org/ComfyUI/issues/15454) describes.
      Nobody has listened to it.
- [ ] **Three voices in one bin, rendered.** `voice_of` makes a multi-character
      cast expressible, and every assertion behind it is against the compiled
      prompt. Whether H3 keeps three bound voices apart in one call — and whether
      `sink_conditioning=exact_kv_and_rows` is in fact what makes it — is the
      measurement §12.6 has always asserted and nobody has taken.

## [3.0.0] — 2026-08-10

Compile and render are separate nodes now, and every window a render produces is
cached on disk and reused. Built against `PulseSlate_v3_BUILD_INSTRUCTIONS.md`.

### ⚠️ Breaking — `PulseSlate` no longer renders, and no longer outputs a model

`PulseSlate` compiles a `PULSE_TIMELINE` and stops. A timeline that fits in one
window still hands back `positive` and `latent` for your own sampler; anything
longer blocks those two outputs and tells you to add a **Pulse Render** node.

Output slot 0 changed from `MODEL` to `PULSE_TIMELINE`. Slots 1–5 (`positive`,
`latent`, `combined_audio`, `images`, `compiled_prompt`) kept their types and
positions, so a 2.x graph wired into an external sampler keeps working. The old
`model` link on slot 0 is dropped by the frontend as a type mismatch — loudly,
which is the intended half of that trade. Sigma shift is a model patch like any
other and belongs upstream: **UNETLoader → MiniMaxH3SigmaShift → your sampler**.

Widget values migrate cleanly. 3.0.0 appends exactly one widget (`continuity`)
and changes nothing else, and `js/ps_widget_order.js` keeps the full 2.0.0 table,
so a workflow saved by 2.0.0 loads by name with `continuity` taking its default.

### Why the split

2.x shipped the short path and the long path as two parallel wired groups with
one muted and a Ctrl+M instruction in the README. They drifted, and the drift was
silent: a 15-second timeline that split into 192 + 175 frames sampled internally,
and the still-wired short branch re-sampled the last window alone and saved a
7-second file as the whole film. No error anywhere — 175 frames is a perfectly
valid latent. Nothing is muted in any shipped graph now, and a test enforces it.

### New — `PulseShot`

One node per shot: its text, its length, its continuity override, its own
first/last frames and its own scene-local references. Because the frames are real
`IMAGE` inputs, a shot's opening frame can come from an upstream generator in the
same graph instead of only from a file dragged onto the bin.

Connecting any shot socket makes the shot **text box inactive**. The two are never
merged and neither is silently preferred; the compiled prompt says which won, at
the top, every time.

### New — `PulseRender`, and the segment cache

Every window is written to `ComfyUI/output/<run_dir>/<run_id>/` as it finishes,
and `manifest.json` is fsynced before the next window starts. The manifest is
written **after** the media files are closed, never before, so a crash between
the two cannot leave a manifest claiming a segment that is not on disk.

- Kill a render at window 9 of 12 and requeue → 0–8 load, only 9–11 render.
- Edit one shot → only the window holding it re-renders, at its original seed.
- Change the base seed, `steps`, or the upstream patch chain → all windows go.

The cache key is **content, never window index**. Index keying cannot tell "shot
3 was edited" from "a shot was inserted before shot 3", and gets both wrong in
the expensive direction.

`dry_run` produces the report and nothing else — no sampling, no decode, no file
writes. A wrong reference binding renders successfully and hands you a well-formed
film of the wrong person; the dry run is how you catch it before the GPU starts.

### New — stable per-shot seeds

A window's seed is derived from the **set of shots in it**, not from its position.
Inserting a shot at the top of a twelve-window film no longer rerolls every window
after it — which used to turn "add an establishing shot" into a new film.

`PulseShot` writes a UUID once and never changes it. Text-box shots derive an id
from their own content.

> **Deviation from §6.2, deliberate.** The spec gives the text-shot id as
> `sha1(label + "|" + visual)`. That concatenation is ambiguous: a label of `a|b`
> with a visual of `c` hashes identically to a label of `a` with a visual of
> `b|c`, so two different shots would share an id, a seed **and** a cached
> segment. The fields are joined through the canonical JSON encoder instead. No
> ids were persisted by any earlier build, so this costs no migration.

### New — `patch_fingerprint` (mandatory) and `PulseBench`

Sol-Attn, Spectrum and EasyCache all change what the same prompt at the same seed
produces. A cache that ignored them would hand back window 4 rendered dense and
window 5 rendered at `tau=2.0` and call it a film. So every approximation on the
incoming model is detected, canonicalised and folded into the cache key, and
`segcache.cache_key` **refuses to run** without one.

Detection is a scan over key names, module paths and callable identities rather
than a table of literal keys: `ComfyUI-sol-attn` was not installed on the machine
this was written on, so its key names could not be read out of its source, and a
guessed table would have missed it silently. Nothing that reaches the descriptor
carries a memory address — a fingerprint that differed between processes would
re-render everything on every restart.

`PulseBench` reads manifests back and prints seconds-per-frame and peak VRAM
grouped by fingerprint. §12.7 is explicit that the right handling of "is Spectrum
faster here" is measurement, not a recommendation baked into code.

### New — continuity modes and scene-local references

`continuity` on `PulseSlate`, overridable per shot: `none`, `last_frame_carry`,
`keyframe_pairs`. The latter two need `model_fl2va` and **fail at compile time**
without it rather than falling back silently.

References now have two scopes. Global ones (the bin, plus `refs.ref_image_1..8`,
`ref_video`, `ref_video_audio`, `ref_music`) hold the same ordinal in every shot.
Scene-local ones on a `PulseShot` are visible only to that shot.

> **Note on §10's ordinal wording.** H3 takes one flat `ref_items` list per call
> and the tokenizer numbers by position in it, so two shots sharing a window
> cannot both start their locals immediately after the global block. Ordinals are
> assigned in window socket order; what §10's "within that shot only" actually
> buys — that one shot's prose can never resolve an alias to another shot's
> reference — is enforced by scoping the resolver instead. A shot rendered alone
> in its window gets the literal numbering §10 describes.

### New — growing socket groups

`shots.shot_1..24` and `refs.ref_image_1..8` are declared in full on the Python
side and grown one free socket at a time by `js/ps_sockets.js`. Sockets are
ordered by numeric suffix, never by connection order, and links are restored by
**name** across a rebuild — a link stores `target_slot` as an index, so trimming a
socket would otherwise move every wire after it onto its neighbour.

### Memory

`low_memory` accumulates assembled frames as 8-bit and releases VRAM between
windows. The finished video is assembled by stream-copying the segment files
together, so a twelve-window film is never held in RAM at all. The `frames`
output is only materialised when it is actually wired — answered from the prompt
graph, since ComfyUI does not tell a node which of its outputs are used.

### Fixed — lip-sync to a supplied recording never worked

H3 can match a character's mouth to audio you provide. This project shipped the
socket for it and none of the three things that make it function, so a wired
recording produced a clean render, no error, and a mouth tracking nothing.

- **The prompt never asked for the match.** `<Audio j>` was introduced to the
  model as "a voice-timbre reference" and the tag was left as a bare token at the
  end of the shot line. The directive *is* the mechanism: the tokenizer emits only
  the marker `"<Audio j>: "` and `comfy/text_encoders/minimax.py` states outright
  that "audio never enters Qwen", so nothing about connecting a socket tells the
  model what the recording is for. A `lip_sync` reference is now introduced as the
  speech the character is saying, with their lip movements matching that tag
  precisely — phrasing taken from a reference workflow that demonstrably works.

- **The clip was never trimmed to the window.** Reference audio rows are packed
  against the target audio grid (`comfy/ldm/minimax/model.py`, `PackedLayout`), so
  a 30-second file against a 9.42-second window asks the model to align two
  different stretches of time. Each `lip_sync` clip is now cut to its window's
  exact span, `frames / fps`, and padded with silence rather than shortened when
  the recording runs out — a voice that stops should leave the mouth still, not
  desynchronise every window after it. The index arithmetic lives in
  `concat.audio_span_bounds`, stdlib-only, for the same reason the placement maths
  does: it is what has to be right, and nothing needing torch is reachable by the
  suite on a box with no GPU.

- **New `PulseShot.ref_audio_mode`,** `lip_sync` (default) or `voice_timbre`. Two
  different jobs needing different sentences and different handling: timbre asks
  no temporal question and is not trimmed. **It is in the cache key** — the
  directive lives in the window's subject definitions, which no shot's
  `resolved_prompt` covers, so without it flipping the widget would change what
  the model is told while the cache returned the segment rendered under the other
  instruction. Appended to the reference descriptor only when set, so a project
  with no audio reference keeps every key already on disk.

- **New `PulseRender.use_reference_audio`.** On a lip-sync shot the audio H3
  generates is a re-synthesis of your recording; this muxes the original in
  instead. Built from the timeline rather than from the render loop, because a
  reused window decodes nothing and never resolves its references — collecting
  these during rendering would leave holes exactly where the cache did its job.
  Off by default, and the generated track still goes to every segment's `.flac`,
  so it is reversible without re-rendering.

- **A quoted line beside a `lip_sync` reference is now reported.** The `<d>`
  block tells the model to speak those words while the recording says what it
  says, and nothing in the output reveals which one it followed. Named in the
  report rather than refused: a quote that is the recording's own transcript is
  legitimate, and only the author knows whether it is.

**Behaviour change:** a shot with a connected `ref_audio` and no explicit mode
loads as `lip_sync` and its segments re-render once. That is the intended
correction — the previous behaviour was a timbre reference the prompt asked
nothing of.

Still true, and worth stating because it looks like a bug: **a `lip_sync`
reference on a shot with no dialogue produces no mouth movement.** There is no
speech to match. That was the shape of the fault in the shipped 18-second graph,
where the only audio reference sat on the one shot that never speaks.

### Fixed — packaging, and the attribution that rides in it

- **The package version was still `2.0.0`.** The v3 work bumped
  `SCHEMA_VERSION` and the widget tables and left `pyproject.toml` behind, so a
  publish would have shipped an artefact labelled 2.0.0 containing three nodes
  2.0.0 never had. Both numbers are `3.0.0`, and `tests/test_packaging.py` fails
  if they drift again.

- **`NOTICE` was not going to ship.** `pyproject.toml` declared
  `license = { file = "LICENSE" }` and nothing else — no `license-files`, no
  `MANIFEST.in`. NOTICE holds the *only* copy of upstream's MIT copyright
  notice, and MIT grants its permissions on the condition that the notice
  travels with every copy, while Apache-2.0 §4(d) requires the same of NOTICE
  itself. Every word of the licence boundary was correct in the repository and
  absent from the thing users would actually receive.

  Fixed in three places, because one of them is a promise and the others are
  facts: PEP 639 `license-files = ["LICENSE", "NOTICE"]`, a `MANIFEST.in` that
  covers the sdist and older builders, and a new CI `distribution` job that
  builds both artefacts and opens them to check. Nothing in CI built anything
  before, so this class of fault had no way to be caught.

- **There was no `[build-system]` table.** pip would have fallen back to
  setuptools' legacy backend and run flat-layout autodiscovery over a root that
  holds `nodes.py`, `media.py`, `render.py`, `tests/`, `js/`, and the GPL-3.0
  reference tree. That either errors on the multiple top-level names or guesses,
  and the expensive guess sweeps the reference tree into an sdist — `.gitignore`
  keeps it out of git, which is not the same as keeping it out of a build. The
  backend and the package list are spelled out now, `MANIFEST.in` prunes the
  tree explicitly, and the CI job asserts it is absent from both artefacts.

- **`PublisherId` and the repository URL were unverified placeholders** carried
  over from the pre-fork manifest. Confirmed 2026-08-09 against the publisher at
  `registry.comfy.org/publishers/addis-pulse` and the repository at
  `github.com/Addis-Pulse-Studio/comfyui-pulse-studio`; both are pinned by test,
  so a stale value cannot publish under someone else's namespace.

### Fixed — from the first real run

- **`PulseSlate_LongForm.json` shipped with only one of its three shot nodes
  wired.** Three shots on the canvas, one 6-second window rendered, and no
  warning anywhere — an unconnected shot is not a shot, it is a node sitting on a
  canvas. All three are wired now, and a test fails if any shipped graph carries a
  `PulseShot` connected to nothing, or any node connected to nothing at all.
- **Joining the segments could kill a finished render** —
  `av.error.ArgumentError: Invalid argument ... returned 22`, raised out of the
  node after every window had already been rendered and written.

  Segments do not start at zero. Video opens on a negative decode timestamp
  because of B-frame reordering, and AAC opens earlier still because of its
  priming delay: on a real segment both opened at `dts = -1024`, but in different
  time bases (1/12288 and 1/32000), which is −83 ms of video against −32 ms of
  audio. The first implementation measured each stream's length from zero rather
  than from where it began, understating the audio by its priming delay, so the
  next segment's audio landed on a timestamp already used and libavformat rejected
  the non-monotonic DTS.

  Measuring the extent instead — the obvious correction — is monotonic and still
  wrong: a segment's *decode* extent is longer than its content, precisely because
  of those negative starts, so packing nose to tail spaced the segments 84 ms
  further apart than their duration. Two dropped frames of dead air at every seam,
  which no timestamp assertion catches and which is obvious on playback.

  So the placement is no longer derived from timestamps at all. Every segment is
  exactly `frames` frames at a known fps, so segment *i* goes at
  `sum(frames before it) / fps` — exact, gapless, and indifferent to what the
  encoder did with its priming delays. **The audio is no longer copied**: the
  lossless per-segment FLACs are joined as one waveform (no priming delay to
  repeat, no seam to align) and encoded once into the assembled file, which also
  makes its audio better than the re-muxed AAC would have been.

  The arithmetic now lives in `comfyui_pulse_studio/concat.py` — pure, and driven
  by timestamps read out of the real segment files with `ffprobe`. It was wrong
  twice because nothing without PyAV could reach it.

  The assembly step is also wrapped now, at both levels. It is convenience: the
  render is already complete and durable when it starts. A failure there logs the
  reason, says plainly that the segments are safe and that requeueing will reuse
  them, and returns a blocker on `video` — it can no longer take the render with
  it.
- **The long graphs opened on an unresolved-reference warning.** They inherited
  the starter's asset bin, which points at `example_character_*.png` files a new
  user has not got; those assets are dropped before tags are assigned and every
  `@Image1` in the prompt is then correctly reported as unresolved. The
  shot-driven graphs now ship with an empty bin and a global prompt that names no
  asset. The bin and `@`-references are what the *short* starter teaches.
- **`PulseBench` sat in the long graphs wired to nothing**, which reads as a
  mistake rather than as an output node. It now has its own `PreviewAny`.
- **Scene-local references were unusable in practice.** Name lookup searched the
  whole window's bin and applied the shot's scope *afterwards*, so two shots each
  calling their own local image `Ref1` made the name ambiguous and neither
  resolved — the scoping in §10 existed but could not be reached by name. The
  scope is now applied inside the lookup, and locals get short fixed handles
  (`@Ref1`..`@Ref4`, `@Voice`) instead of names derived from the shot's label. An
  out-of-scope name is also reported as *belonging to another shot* rather than as
  not existing, which sends the author somewhere different.
- **Socket references could collide with the Asset Bin.** The bin numbers its own
  drops `Image1`, `Image2`…, and `refs.ref_image_1` on PulseSlate was auto-named
  `Image1` too. Two assets of the same name make `@Image1` ambiguous, and an
  ambiguous name resolves to nothing — silently. Socket references now take the
  next free name instead.

### Fixed

- A **reused** window decodes nothing, so the next window's carry-over sockets had
  no tensors to fill. The compiler had already allocated ordinals around those
  sockets, so leaving them empty shifted every reference tag behind the hole. The
  executor now rebuilds the carry frame, audio and motion tail from the files the
  cached segment left on disk, reading only what the next window asks for — and an
  unfillable carry socket is raised rather than skipped.
- `ff_chunk` was attributed to `sol_attn` in the patch descriptor, because feed-
  forward chunking ships *inside* the Sol-Attn pack and its module path contains
  `sol_attn`. The chain was reported with a patch missing. Fragment matching is
  now most-specific-first.

### Workflows

Three graphs, none with a muted branch:

| file | shows |
|---|---|
| `PulseSlate_Starter.json` | short path — `PulseSlate` → your own sampler |
| `PulseSlate_LongForm.json` | long path — `PulseSlate` → `PulseRender`, with `PulseShot` nodes |
| `PulseSlate_Starter_SpectrumSage.json` | the same, with the patch chain and the §12.3 ordering trap |

> **The Sol-Attn nodes are pre-wired.** §12.3's order is wired on both model
> paths and stated in the graph's `MarkdownNote`, including the trap — the Sol
> node applied *before* the Sage patch is shadowed entirely and silently does
> nothing. Class ids and widget order come from the installed pack's
> `INPUT_TYPES`, never guessed: a workflow naming a guessed id loads as a red
> MISSING NODE even when the pack is present. They are the only nodes here from a
> pack Pulse Studio does not otherwise need, so if `ComfyUI-sol-attn` is absent
> they load red and can be deleted, with Sage wired straight through.

## [2.0.0] — unreleased

First release under the Pulse Studio name, and the last one that is free to
break the slot contract. Nothing had been published, so the break was taken now
rather than carried forever.

### ⚠️ Breaking — workflows saved by any earlier build are not loadable

Widget order changed. `schema_version` is now widget index 0 and `timeline_data`
is index 1; every other widget kept its name, type and default and shifted by
two. A workflow saved by a pre-2.0.0 build has no schema version in slot 0, so
it is **refused with a message naming it as unloadable rather than guessed at**.

There is no automatic migration and there will not be one. The pre-2.0.0 files
in circulation were written by the build that spliced DOM headers into the
widget list, so their stored values are already shifted — the text that landed
in a header slot was discarded at save time and cannot be recovered. Loading
such a file "successfully" would render something wrong rather than fail.

**Recreate affected workflows from `example_workflows/PulseSlate_Starter.json`.**

### Changed — names

| old | new |
|---|---|
| `MiniMaxH3OmniDirector` | `PulseSlate` |
| `MiniMaxH3RetakeScissor` | `PulseRetake` |
| `MiniMaxH3StillMode` | `PulseStill` |
| `omni_director/` | `comfyui_pulse_studio/` |
| `js/omni_asset_bin.js` | `js/pulse_slate.js` |
| `js/od_widget_order.js` | `js/ps_widget_order.js` |
| `js/od_widget_guard.js` | `js/ps_widget_guard.js` |
| `MiniMax H3/Omni-Director` (menu) | `AddisPulse/H3` |
| `/omni_director/*` (routes) | `/pulse_studio/*` |
| `OmniDirector_Starter.json` | `example_workflows/PulseSlate_Starter.json` |

`PulseRetake` and `PulseStill` are not named in the build spec's rename map, but
they ship in `NODE_CLASS_MAPPINGS` and are equally subject to the slot contract,
so they were renamed and frozen alongside `PulseSlate`.

### Added — the slot contract (spec §3)

- `schema_version`, a hidden `STRING` at widget index 0 on **every** node in the
  pack. Written on save, read on load.
- Name-based `onConfigure`. Positional loading is gone. Values are matched by
  name against the saved schema version's name table; widgets the live node has
  but the file lacks take their declared default, and widgets the file has but
  the node lacks are dropped with a console warning. This is what makes future
  appends safe.
- `onSerialize` stamps the current schema version into both the widget and the
  serialised array, so the two cannot disagree.
- `js/ps_widget_order.js` now exports `WIDGET_NAMES[nodeClass][version]` and
  `CURRENT`. Keyed by node class as well as version, because three nodes are
  under the contract; the spec's single flat table assumes one.
- The cross-language parity test was extended from `PulseSlate`'s widgets to all
  three nodes' widgets, connection inputs, `RETURN_TYPES` and `RETURN_NAMES`.
  Adding a widget on one side of the language boundary now fails CI.
- `checkWidgetOrder()` additionally asserts that the live native widget names
  equal `WIDGET_NAMES[node][CURRENT]` in order, at node construction.

### Added — `timeline_data` schema 2

The bin document is now `{"schema": 2, "assets": [], "cast": []}`. The `cast`
key ships empty in 1.0 even though nothing populates it until 1.1, so that 1.1
adds entries to a key that already exists rather than migrating a file format
already saved inside users' workflows.

Schema 1 documents — a bare `{"assets": [...]}` — load with an empty cast and
are upgraded in place. The upgrade is additive, so a document from a *future*
schema passes through untouched rather than being downgraded.

### Added — `audio_ref_ceiling`, a raisable audio budget

New widget on `PulseSlate`, 3 to 9, **default 3**. Appended last, which the slot
contract makes safe: a workflow saved before it existed loads with the declared
default, and there is a JS test round-tripping exactly that array.

The 3 is not the model's number. Read out of source:

| where | what it says |
|---|---|
| `comfy_extras/nodes_minimax_h3.py` | `ref_audios` is Autogrow with `max=3` |
| `comfy/ldm/minimax/model.py` | `PackedLayout` appends one `ref_audio` segment per block, in a loop. No count check. |
| `comfy/text_encoders/minimax.py` | the tokenizer increments a counter for `<Audio j>`. No cap. |

So the cap is a socket declaration, unlike the first/last keyframe rule, which
is RoPE position math no node can route around. And Autogrow's `max=` is
enforced by graph validation in `execution.py`, which never runs for this pack:
it calls `MiniMaxH3ReferenceToVideo.execute()` in-process. Nine standalone audio
references are marshalled and rendered.

That is a bypass of a declared contract, so it is opt-in and announced. Above 3
the meter turns amber and the node logs, once per render, that MiniMax documents
three and ComfyUI's socket declares three. It never blocks — the user chose it.

The 12-file total is a model-card number too; it appears nowhere in ComfyUI's
source. It rises with the ceiling, by the same amount, so that nine images and
nine audio fit together rather than the audio passing and the total refusing it.

The budget is now a `RefLimits` value passed explicitly through the bin, the
compiler, the panel routes and the document, rather than read from module scope
— so the ceiling belongs to one render, and two nodes in a graph may disagree.
The panel sends its node's widget with every budget request, because the server
holds no session state and a meter that disagrees with what the render accepts
is worse than no meter.

Not covered by anything published: no MiniMax or ComfyUI documentation describes
a render with more than three, and reference audio rides every sampling step.
Measure it before trusting it.

**Reachable without raising anything:** a reference video's soundtrack takes its
own `<Audio j>`, separate from the three standalone slots, so three videos with
soundtracks plus three standalone audio is six audio references inside the
documented budget.

### Added — environment invariants (spec §18.1)

- **No network, enforced by test.** A source scan over both languages bans
  outbound HTTP clients, sockets, and model auto-download in Python, and
  absolute-origin URLs, CDN loaders, webfonts and request constructors in
  JavaScript. Inbound aiohttp route registration is explicitly allowed — it is
  ComfyUI's own server, and registering a route is not egress.
- **No private assets ship.** A scan of `example_workflows/` rejects
  AI-tool-generated filenames, absolute home paths, committed weights and stray
  media — including inside the `timeline_data` string, which is a JSON document
  embedded in the workflow JSON and invisible to a casual read.

### Added — packaging and CI (spec §14)

- **GitHub Actions on 3.10, 3.11 and 3.12**, push and pull request. It runs
  `python run_tests.py` — the same command the README gives a user — rather
  than installing pytest to run the same unittest cases through a different
  front door.
- Node is installed in CI rather than left optional. The JS tests skip
  themselves when `node` is missing, so that a bare ComfyUI box can still run
  the Python suite; a skip is invisible in a green run, so both `.mjs` files
  are additionally invoked directly, where a missing runtime is a failure.
- **A second job with nothing installed** imports every module in
  `comfyui_pulse_studio/` and asserts that `torch`, `comfy`, `comfy_extras`,
  `folder_paths` and `nodes` never reach `sys.modules`. The AST test already
  bans the imports; this one proves the package actually loads without them,
  which is a different claim.
- `requirements.txt` ships empty, deliberately, and says why. The real
  requirement is a ComfyUI version, which pip cannot express for a custom
  node, so the tested floor — **0.30.0** — is documented in the README instead.

### Added — `PulseSlate_Starter_SpectrumSage.json`

The second graph §15 calls for: the starter with
`UNETLoader → Spectrum (system_ram) → Sage Attention → PulseSlate` wired in on
both DiT inputs. Generated from the starter rather than hand-edited, so the two
cannot drift apart.

Tests walk backwards from each of the director's model inputs and assert the
chain arrives in that order, that no patch node sits downstream of the director,
and that Spectrum ships set to `system_ram`. A shipped example teaches its
wiring to everyone who opens it, and the failure it would teach — patching the
node's *output* — speeds up the ≤15 s path while doing nothing for a long
timeline.

Spectrum and KJNodes are not dependencies. Deleting the patch column leaves the
plain starter graph.

### Changed — README restructured to §15

Now opens with install, the model table, the never-type-a-tag rule, the two
duration paths, the patch chain, `cfg`, the no-network guarantee, and known
issues — in that order, before any of the design prose. The two-path table
states plainly that **path B is not verified on hardware**.

`docs/` is now the one directory where an image may live, for the node-face
screenshot. A test keeps that exemption narrow: flat, small, and still subject
to the private-filename scan.

### Changed — licence

Relicensed from MIT to **Apache-2.0**. The studio deploys private systems onto
hardware clients own; a permissive licence keeps those deployments free of a
source-offer obligation, and Apache adds an explicit patent grant that MIT
lacks.

`NOTICE` states explicitly that muse-collective MIT code **is** present — this
project is a fork of `MiniMaxH3-Director-Seed-Hunt` — and reproduces that MIT
notice in full, which is what keeps MIT compatible with an Apache-2.0 project.
It also records that no `seesee75-commits/ComfyUI-MiniMaxH3-Director` (GPL-3.0)
code is present, verified against the full git history and not only the working
tree.

### Fixed — a 15-second timeline rendered 7 seconds

Four faults in a row, each of which looked correct alone. Every step succeeded;
the result was a 7.29-second file with no error anywhere.

1. **The window cap snapped down to the frame grid.** 15.0 s is 360 frames, the
   grid steps by 17, so the cap became 345 frames — 14.375 s. The trained
   ceiling is 362 (15.083 s), so the most natural number a user can type could
   not express a single window, and a 15.0 s render silently split in two over a
   0.7 s shortfall. The cap now rounds to the **nearest** grid point, in both
   `partition_windows` and `compile_timeline`. It still clamps to
   `MAX_WINDOW_FRAMES`, so rounding up can never leave the trained range.
2. **Two windows means the internal sampling path**, which handed back the last
   window's latent on `latent` and `positive`.
3. **The starter graph's single-window group sampled that latent.** 175 frames
   is a perfectly valid latent, so it rendered and saved as if it were the whole
   piece. Those two outputs are now an `ExecutionBlocker` on the multi-window
   path, naming the window count and telling the user to switch groups. A
   message beats a plausible-looking short video.
4. **`[00:00.000 - 00:05.000]` range markers did not parse.** `_TIME_MARKER`
   accepted only a bare `[MM:SS.mmm]`, so a three-shot storyboard collapsed into
   one shot whose prompt was the entire text — a render with no per-shot
   direction in it at all. Ranges now parse, with `-`, `--`, `–`, `—` or `to`.
   The stated end is read and checked rather than obeyed: shot spans still run
   to the next shot's start, and a range that disagrees is reported.

### Fixed

- Ten client-facing asset filenames were referenced by the shipped example
  workflows, four of them buried inside the `timeline_data` string. All are now
  `example_*` placeholders that the user supplies.
- **Model paths in the example workflows are back to `minimax\name`**, reverting
  a portability change that made every shipped graph unqueueable on Windows —
  five red MISSING MODELS on load, on the platform the graphs were built on.

  ComfyUI compares a stored model value against `get_filename_list()`, which is
  built with `os.sep`, and `execution.py` rejects a miss as `value_not_in_list`
  *before* the graph runs. `get_full_path` would resolve either separator, but
  validation runs first, so the forward-slash form never got that far.

  There is no portable spelling — the separator belongs to the host. So the
  question is only which platform loads clean and which clicks a combo once, and
  the answer is the platform this was authored, run and verified on. A test now
  pins it, with that reasoning attached, so the next well-meant portability fix
  fails CI instead of shipping.

### Name availability check — 2026-08-07

Checked on 2026-08-07. Every machine-verifiable namespace is clear:

| namespace | query | result |
|---|---|---|
| PyPI | `comfyui-pulse-studio` | 404 — available |
| PyPI | `pulse-studio` | 404 — available |
| ComfyUI registry | `comfyui-pulse-studio` | 404 — available |
| GitHub | `ComfyUI-PulseStudio`, `PulseSlate` | no collision found |

Nearest neighbours, neither a conflict: `krishnancr/ComfyUI-Pulse-MeshAudit` on
GitHub, and `minimaxh3-direct` (publisher: miaodl) in the ComfyUI registry — the
latter worth tracking as a same-niche project. "Pulse" alone is widely used as a
product name; `PulseStudio` and `PulseSlate` are not.

**A trademark search has not been done.** Registry and PyPI 404s establish
availability, not clearance, and the two are not the same question. §1 calls for
a plain trademark search; that remains outstanding and blocks the tag.

### Released to the registry — 2026-08-11

`3.0.0` is published, as `comfyui-pulse-studio` under publisher `addis-pulse`:

    id          comfyui-pulse-studio      status  NodeStatusActive
    version     3.0.0                     status  NodeVersionStatusPending
    licence     Apache-2.0
    download    cdn.comfy.org/addis-pulse/comfyui-pulse-studio/3.0.0/node.zip

The **version** status is `Pending`, not `Active` — that is the registry's own
scan of the uploaded archive, and it flips on its own once the scan clears. Until
it does, the listing exists but the version is not installable. Nothing to do
about it but check back.

The GitHub repository went public the same day, so the `repository` URL in the
listing resolves rather than 404ing.

### Still unverified in the shipped release

These did not block the release and are not closed by it. Everything here needs a
GPU and is sequenced into one sitting in
[`docs/HARDWARE_VERIFICATION.md`](docs/HARDWARE_VERIFICATION.md) — what to do,
what you should see, and a place to write down what you actually saw. Fill it in
and commit it; a verification nobody wrote down has to be done again.

- [x] **Long-path verification on hardware — 2026-08-10.** A >15 s render through
      `PulseSlate → PulseRender` completed on the user's box. This closes the item
      that had never been run at all; before today only the ≤15 s single-window
      path was verified.

      Recorded on the user's confirmation, not from a filled-in
      `docs/HARDWARE_VERIFICATION.md` — the per-seam observations that document
      asks for are still blank. So: the multi-window path is known to run end to
      end, and the seam-by-seam listening pass is not written down anywhere.
- [ ] **The five §7.5 cache behaviours, observed on a real render.** Their
      decision logic is covered by `tests/test_segment_cache.py`, which drives the
      same two functions the executor calls, in the same order. What no test here
      can prove is the half that touches disk and VRAM: that a killed render
      really does resume, that a reused segment stream-copies into an assembly
      that plays, and that the carry-over rebuilt from a cached segment's PNG and
      FLAC produces a seam that matches one carried in memory.
- [ ] **`PulseBench` against two real chains.** The table is only worth having if
      the numbers in it came from this box; nothing has been measured yet.
- [x] **Sol-Attn node ids — read and wired, 2026-08-10.** The ids were read on
      2026-08-09 from the installed pack's `/object_info`, and the §12.3 chain is
      now pre-wired in `PulseSlate_Starter_SpectrumSage.json` on both model paths
      rather than described in its note. `MiniMaxH3ScheduledSolAttentionPatch`
      and `MiniMaxH3ChunkFeedForward`, with widget order taken from the pack's
      `INPUT_TYPES` rather than guessed. Pinned by
      `tests/test_workflow.py::TestThePatchChainVariant`, which asserts the whole
      five-node chain and, separately, that Sol sits downstream of Sage.

      Wired on the user's confirmation that the pack is installed and locally
      tested, not on benchmark numbers — the `PulseBench` item below is still
      open, so the chain is shipped as the recommended order rather than as a
      measured win.
- [x] **Trademark search — cleared, 2026-08-09.** Zero results for
      `ComfyUI-PulseStudio` on the ComfyUI Registry and PyPI, zero on **USPTO**
      in Classes 009 and 042, and zero in the **WIPO Global Brand Database**.
      This is the clearance the availability check above could not establish;
      the two were tracked as separate questions and both are now answered.
- [x] **`PublisherId` confirmed** — 2026-08-09, against the publisher at
      `registry.comfy.org/publishers/addis-pulse`. `pyproject.toml` carried
      `behailu-ai` from the pre-fork manifest; it is `addis-pulse` now, pinned by
      `tests/test_packaging.py`.
- [~] **Node-face screenshot — dropped, 2026-08-10**, at the user's call. §15
      asked for one at the top of the README; the README ships without it and the
      placeholder comment is gone. `docs/` keeps its image exemption and the
      narrowness test around it, so adding one later needs no other change.
- [ ] **Every example workflow opened on a fresh ComfyUI** with no red nodes.
      Their structure is asserted by test — ids, slots, link backfill both ways,
      the patch chain's direction — but "no red nodes" is a claim about the host
      install, and only loading them proves it. Two graphs shipped when this was
      written and one when 3.0.0's set was cut back; **four** ship as of
      2026-08-17, and the new three have never been opened.
- [x] **CI observed green on GitHub** — run `31318598402`, 2026-08-09, the first
      push to that remote (then named `comfyui-addis-pulse`, renamed to
      `comfyui-pulse-studio` later the same day). All five jobs: `test`
      on 3.10, 3.11 and 3.12, `import-purity`, and `distribution`.

      The `distribution` job is the one that mattered, because it could not be
      run here at all — this box has setuptools 68 and no `build`, `wheel` or
      `pip`, so the PEP 639 manifest was written unbuilt. On the runner
      (setuptools 84) it built both artefacts and opened them:

          built: comfyui_pulse_studio-3.0.0-py3-none-any.whl
                 comfyui_pulse_studio-3.0.0.tar.gz
          sdist: NOTICE -> comfyui_pulse_studio-3.0.0/NOTICE
          wheel: NOTICE -> comfyui_pulse_studio-3.0.0.dist-info/licenses/NOTICE
          both artefacts carry LICENSE and NOTICE; no GPL reference tree

      Upstream's MIT notice now demonstrably ships, and the GPL-3.0 reference
      tree demonstrably does not. Both are checked against the artefacts on
      every push rather than asserted in prose.
