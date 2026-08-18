# Changelog

All notable changes to this project are documented here.

## [Unreleased]

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
