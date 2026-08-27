# Pulse Studio — MiniMax H3 for ComfyUI

Direct a [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) render from a
shot timeline and an asset bin, on one node, without ever typing a reference tag
number.

Three things it does that the existing directors don't:

- **An Asset Bin that cannot misnumber a reference tag.** You never type
  `<Picture 3>`; you write `@Mimi`, and the ordinal is computed from live bin
  order at compile time.
- **A Retake Scissor** that patches a bad three seconds without re-rendering
  fifteen.
- **Still Mode**, so you can make your reference images in the same window you
  use them.

Plus compiler correctness, which is invisible when it's right and destroys trust
when it's wrong.

> **POWERED BY MINIMAX H3.** This repository contains no model weights. The
> weights carry the
> [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3),
> a separate agreement governing their use and any output generated with them.

---

## Install

Clone into your ComfyUI installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Addis-Pulse-Studio/comfyui-pulse-studio
```

Restart ComfyUI. There is nothing to `pip install`: the pack has no
dependencies of its own, and the tensor layer uses `torch`, `numpy`, `Pillow`
and `av`, all of which a working ComfyUI already has. See
[requirements.txt](requirements.txt) for why that stays true.

**ComfyUI 0.30.0 or newer**, which is the version this was developed and tested
against. It has to be a build carrying `comfy_extras/nodes_minimax_h3.py` and
`comfy/ldm/minimax/` — without those, MiniMax H3 is not there to direct. There
is no mechanism for a custom node to enforce a host version, so this is
documentation rather than a check.

Then load one of the four graphs in `example_workflows/`:

| graph | what it shows |
|---|---|
| `PulseSlate_Single.json` | The short path: one window, sampled by your own graph. Start here. |
| `PulseSlate_LongForm.json` | The long path: many windows, `PulseRender`, the segment cache. |
| `PulseSlate_Cast.json` | The Asset Bin — named references, descriptions, retention, per-shot audio. |
| `PulseSlate_Retake.json` | Re-render a bad span of a finished film, in place. |

Every one of them opens with no third-party pack installed. The placeholder
references `PulseSlate_Cast.json` cites are copied into ComfyUI's `input/` folder
the first time this pack loads, so its bin opens resolved rather than pointing at
files you have not got — they are flat colour and a sine tone, and the point is
to replace them.

## Updating

New work lands on `main`, so updating is a pull in place:

```bash
cd ComfyUI/custom_nodes/comfyui-pulse-studio
git pull
```

Restart ComfyUI afterwards — the pack is read at startup, and a running server
goes on serving the code it loaded. Nothing else is needed: there are still no
dependencies to reinstall, and workflows you have already saved keep loading,
since widget values migrate by name and a slot that changed type is dropped
rather than silently mis-wired.

## Models

Not redistributed here. Download from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) and place
them as below — the `minimax/` subfolder is what the example workflows expect:

| file | goes in | approx |
|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/minimax/` | 20 GB |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/minimax/` | 20 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/minimax/` | 15 GB |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/minimax/` | 4.9 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/minimax/` | 578 MB |

Those are the exact files the example graphs were built and tested against, and
the sizes are measured from that install. Nothing here is tied to a precision —
if you run a different quantisation of the same weights, point the loader
widgets at it and the rest of the graph is unchanged.

**On Linux and macOS the five model widgets will load red.** ComfyUI compares a
saved model value against a list it builds with the host's path separator, so
the graphs — authored and verified on Windows — carry `minimax\name.safetensors`
and no spelling is portable. Click each loader and pick the same file; the graph
is unchanged otherwise. Windows users see no such prompt.

The two DiT files are the ref2va and fl2va branches from
[§ Verified model constraints](#verified-model-constraints) — you need
whichever branches your timeline actually uses, and **only** `ref2va` for a
plain reference-driven render. Loading both is ~42 GB.

**Weights carry the MiniMax H3 Community License**, a separate agreement from
this pack's Apache-2.0 that governs your use of the model and of what you
generate with it. Read it before shipping client work.

## Never type a tag number

> ### Drag assets onto the node body. Reference them by name.
>
> `@Mimi`, `@Image1`, `@[Cafe wide]` — never `<Picture 2>`.

H3's reference tags are ordinals assigned by socket position. Type one by hand
and it goes stale the moment the bin changes, and the render **still succeeds** —
describing the wrong picture, with no error anywhere. This node computes every
ordinal at compile time from live bin order, and reports a hand-typed tag as an
error rather than trusting it. The mechanism, and the reference-video soundtrack
case that makes it worse, is in [The tag problem](#the-tag-problem-and-why-this-node-exists).

## Two paths, by duration

Two paths and a graph for each, never one graph with a muted branch. Pick the
file that matches the length you are making.

| duration | graph | what happens | status |
|---|---|---|---|
| **≤ 15 s** — one window | `PulseSlate_Single.json` | `PulseSlate` hands back `positive` and `latent`. The graph carries sampler → decode → mux. | **Verified on hardware.** |
| **> 15 s** — many windows | `PulseSlate_LongForm.json` | `PulseSlate` hands a `PULSE_TIMELINE` to `PulseRender`, which renders one window per H3 call, caches each to disk, and stitches. | **Runs on hardware** — a >15 s multi-window render completed 2026-08-10. The seam-by-seam listening pass is still not written down, so treat seam quality as unconfirmed. |

That last column is stated plainly because a stitched seam is the kind of thing
that passes every test and still sounds wrong. The path runs end to end; what
nobody has written down is whether the seams hold up to listening.

Clearing it is a specific, ordered piece of work rather than a vague "try it":
[`docs/HARDWARE_VERIFICATION.md`](docs/HARDWARE_VERIFICATION.md) walks the long
render, every seam, the five cache behaviours and the `PulseBench` numbers in
one sitting, and has blanks to record what actually happened.

Past a single window, `PulseSlate` **blocks** `positive` and `latent` rather than
handing back the last window alone. Until 3.0.0 it did hand them back, and a
still-wired sampler quietly re-rendered that one window and saved it as the whole
film: a 15-second timeline that split into 192 + 175 frames produced a 7-second
file, with no error anywhere, because 175 frames is a perfectly valid latent.

### The segment cache

`PulseRender` writes every window to `ComfyUI/output/<run_dir>/<run_id>/` as it
finishes and fsyncs the manifest before starting the next one.

- **Kill it at window 9 of 12 and requeue** → 0–8 load from disk, 9–11 render.
- **Edit one shot** → only the window holding it re-renders, at its original seed.
- **Change the seed, the steps, or the upstream patch chain** → everything goes,
  because all three change what comes out.

Seeds are derived from the *set of shots in a window*, so inserting a shot at the
top no longer rerolls every window after it.

**Run `dry_run` first.** It produces the report and nothing else — no sampling, no
decode, no file writes. A wrong reference binding renders successfully and hands
you a well-formed film of the wrong person; the report's per-shot ordinal map is
how you catch that before spending the GPU time.

## Model patches — wire them upstream

The pack consumes a patched `MODEL`. It does not patch anything itself and takes
no dependency on the packs that do. The documented chain:

```
UNETLoader
  → SpectrumApplyMiniMaxH3        (history_storage: system_ram)
  → PathchSageAttentionKJ         (sets Sage as the fallback backend)
  → MiniMax H3 Scheduled Sol Attention Patch
  → MiniMax H3 Chunk FeedForward
  → PulseSlate / PulseRender
```

You wire this yourself — **no graph ships with it pre-wired.** A variant that did,
`PulseSlate_Starter_SpectrumSage.json`, shipped in 3.0.0 and was dropped
afterwards so that the one example left needs no third-party pack to load clean.
The chain needs [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3),
[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) and
[ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn); none is
required by Pulse Studio itself. `PulseSlate_LongForm.json`'s own note carries
the same chain and the same warning.

**The Sol-Attn node must come after the Sage patch.** Applied after it, Sol adopts
the Sage forward as its fallback, so ineligible steps and short sequences run
memory-efficient Sage while eligible steps run Sol-Attn. Applied *before* it, the
Sage patch shadows the Sol node entirely and it does nothing — and the graph still
runs, still produces output, and gives you none of the speedup. Nothing warns you,
which is why it is stated here and again in the workflow's own note.

Wire it into **both** model inputs: `model` carries your references and
`model_fl2va` carries the first/last anchors. Set the Sol node's
`sink_conditioning` to `exact_kv_and_rows` — Pulse Slate pairs each character
image with its own audio track, and that pairing lives in exactly the query rows
that setting keeps dense. Nothing else needs hand-management: `PulseRender`
detects whatever patches it is handed and folds them into the cache key by
itself, so changing `tau_start` re-renders the affected segments instead of
silently reusing segments rendered at the old sparsity.

This order is documented rather than enforced. Until 2026-08-11 a test asserted
it against the shipped variant; that graph is gone, and nothing checks the
ordering for you now.

Nothing in this pack vendors, wraps or reimplements any of those kernels. Its job
is to detect them, fold them into the cache key, report the chain, and ship
workflows wired in the right order.

**Upstream is not a style preference.** `PulseRender` samples with the model
handed *to* it, so a patch applied downstream of it does nothing at all — and a
patch that changes between two runs of the same timeline would otherwise let the
cache reuse segments across it, giving you a film whose shots do not match.

**`system_ram` is not a speed setting.** On a 32 GB card, Spectrum storing its
history in system RAM is what makes a 362-frame window *fit at all*. Set it to
`vram` and a full-length window is likely to fail with an out-of-memory error
rather than run slowly.

The node inspects the model it is handed and warns — on the node face and in the
console — when it finds no attention patch or no offload patch, naming both. It
never blocks: running unpatched is a legitimate choice, and the warning exists so
that it is a choice rather than an accident.

## cfg stays at 1.0

H3's own reference pipeline has no negative conditioning anywhere; it uses
`BasicGuider`, which takes a single conditioning input. `cfg = 1.0` is that
native path and it is the default here.

Above 1.0 the node switches to `CFGGuider` with an **empty** negative prompt,
because there is nowhere for a real one to come from. That is offered because it
was asked for, not because it is recommended. Leave it at 1.0 unless you are
deliberately experimenting.

## No network

**The pack makes no outbound request** — not at import, not during execution,
not from the widget layer. No telemetry, no model auto-download, no CDN script,
no webfont, no remote image.

This is enforced by a source scan over both languages that runs in CI, not by
policy: outbound HTTP clients, sockets and auto-download are banned in the
Python, and absolute-origin URLs, CDN loaders, webfonts and request constructors
are banned in the JavaScript. Registering an inbound aiohttp route is explicitly
allowed — that is ComfyUI's own server, and answering the frontend is not egress.

Everything you drop in the bin stays on your machine.

## Known issues

- **Workflows saved by any pre-2.0.0 build will not load.** Widget order changed
  once, deliberately, before anything was published; a file without a schema
  version in slot 0 is refused by name rather than guessed at. There is no
  migration and there will not be one — the values in those files are already
  shifted, so a "successful" load would render something wrong instead of
  failing. Rebuild from `example_workflows/PulseSlate_LongForm.json`. The full
  reasoning is in [CHANGELOG.md](CHANGELOG.md).
- **Path B's seams are unconfirmed** — the render runs; nobody has written down
  how the window boundaries sound. See the duration table above.

---

## Where you type

The Director node carries **two multiline boxes on its face**, plus an exposed
control panel — nothing is hidden behind a gear.

**GLOBAL PROMPT** — art style, lighting, camera rules, identity locks, score.
Compiles into `subject_definitions` and `retention_analysis`. Optional labels at
the start of a line: `style:` `identity:` `retention:` `soundscape:` `music:`.
Unlabelled text is treated as style.

**SHOT PROMPT** — one shot per line, each opening with `[Shot N]` or a
`[MM:SS.mmm]` timecode. Shots without a timecode spread evenly between the ones
that have them, so you can stamp only the moments you care about. Quoted `"text"`
becomes dialogue.

Both boxes preserve newlines through a workflow save/load, and **queuing never
writes back to them** — see the erasure note below.

If a widget ever does strip the line breaks, `normalize_prompt_text` restores
them before parsing (`] [` and runs of two or more spaces become newlines). That
is belt-and-braces behind the multiline widgets, not a replacement for them.

### Assets

Drag images, video or audio **onto the node body**. No external loader nodes.
Each drop gets a short alias (`Image1`, `Video1`, `Audio1`) you rename inline —
you never type a filename. The meter refuses a drop that would break the budget
rather than failing at queue time.

Two fields on each row decide how much the model is told about a reference:

- **description** — a one-line description promotes the reference from a bare
  `<Picture i>` citation to a `<Subject N>` definition, which is the shape
  MiniMax's own reference format asks for and what identity consistency is built
  on. Clear it and the same image is cited plainly; both are legal. Audio has no
  `<Subject N>` form, so the field is refused there rather than written where it
  can never reach the model.
- **retention** — `fully_preserved` or `partially_copy`, written *verbatim* into
  the prompt's retention section. A closed list rather than free text: the value
  is a sentence the model reads literally, and a typo in one is not an error, it
  is a slightly worse render.

---

## The tag problem, and why this node exists

H3's reference tags are positional. From `comfy/text_encoders/minimax.py`:

```python
counters = {"image": 0, "audio": 0, "video": 0}
for item in minimax_ref_items:
    counters[kind] += 1
    add_text("<Picture %d>: " % counters["image"])
```

An ordinal means nothing more than *"the nth of its type in socket order."* Insert
an image at the top of your bin and every `<Picture N>` after it shifts by one —
while the prompt text referencing them stays exactly as you typed it.

The render still succeeds. It just describes the wrong pictures, and nothing
anywhere reports an error.

The subtle case is worse. A reference video's soundtrack is appended to
`ref_items` **before** its own `<Video k>` entry and **before** every standalone
audio, so ticking "use its soundtrack" on one video silently renumbers every
`<Audio j>` in your prompt:

```
bin: [video A]  [audio VO]                →  VO is <Audio 1>
bin: [video A + soundtrack]  [audio VO]   →  VO is <Audio 2>
```

This node's answer is that **no human ever types an ordinal**. You reference
assets by name (`@Mimi`) or id (`{{mimi}}`); the compiler resolves them against
current bin order on every compile. Renumbering cannot desynchronise, because
there is only ever one number and it is computed, not stored. Hand-typed tags are
detected and reported rather than silently trusted.

---

## Verified model constraints

Everything below was read out of ComfyUI's source, not inferred. All of it is
encoded in `comfyui_pulse_studio/constants.py` with the file and symbol it came from.

**Keyframes exist at two positions only.** `comfy/ldm/minimax/model.py`,
`PackedLayout.__init__`:

```python
if pixel_index == 0:                                cond_t = float(text_len)
elif frame_count is not None and pixel_index == frame_count - 1:
                                                    cond_t = ...
else: raise ValueError("only first/last keyframe anchors are supported")
```

This is RoPE position math inside the DiT, not a node-level guard. **A custom node
cannot route around it.** Per-shot visual control in H3 is textual — `[Shot N]`
timestamps inside the prompt. Arbitrary-timecode image anchors are not a missing
feature here; they are impossible.

**References and anchors are mutually exclusive.** Two nodes, two checkpoints,
disjoint inputs:

| | anchors | references | checkpoint |
|---|---|---|---|
| `MiniMaxH3ImageToVideo` | `first_frame`, `last_frame` | — | fl2va |
| `MiniMaxH3ReferenceToVideo` | — | 9 img / 3 vid / 3 audio | ref2va |

You cannot hold a character by reference *and* anchor a frame in the same render.
Loading both checkpoints is ~42 GB, so the choice is a memory decision too.

**Budget.** ≤9 images, ≤3 videos (2–15s each), ≤3 standalone audio, **≤12 files
total**. The 12-file cap genuinely binds — 9+3+3 = 15 sockets exist.

A reference video's soundtrack takes its own `<Audio j>` ordinal, separate from
the three standalone slots, so **six audio references are reachable inside the
documented budget**: three videos with their soundtracks enabled, plus three
standalone. The soundtrack rides inside the video's own file and costs nothing
against the 12.

**The audio cap is the one limit here that is not the model's.** Images at 9 and
videos at 3 are Autogrow socket templates in
`comfy_extras/nodes_minimax_h3.py`; so is audio at 3 — but nothing below it
agrees. `PackedLayout` appends one `ref_audio` segment per reference block in a
plain loop, with no count check, and the tokenizer just increments a counter.
The socket cap is enforced by graph validation, and this pack calls the
reference encoder **in-process**, so it never applies to what the node passes.

`audio_ref_ceiling` (3–9, default 3) exposes that. At 3 nothing changes. Above
3 the bin accepts up to nine standalone audio references, the file total rises
by the same amount, the meter turns amber, and the node says once per render
that it is past what MiniMax and ComfyUI document. It will run. Nobody has
published a result at nine, and every reference rides all your sampling steps —
so treat it as an experiment you measure, not a setting you leave on.

**Frame grid.** Frames snap to `17k + 5`, floor 5, trained ceiling 362 (~15.08s at
24fps). `MAX_WINDOW_FRAMES` is configuration, not a constant: when MiniMax raises
the ceiling, one integer moves.

**Trained floor.** `MiniMaxH3ImageToVideo`'s own tooltip gives the trained range
as ~124–362. Every emitted window is therefore **≥124 frames**: a short tail is
merged backwards into its predecessor, and where that would exceed the ceiling
the partition is rebalanced into fewer, longer windows. The one exception is a
render whose *total* is under 124 frames, which passes through as a single short
window — there is nothing to merge it with. The floor outranks your chosen window
length, and any override is reported.

**Audio is reference, not sync.** `ref_audios` is standalone reference
conditioning — voice timbre and sonic character, with no frame-level alignment.
There is no audio-driven latent path, so **lip-sync and beat-matched motion are
not achievable through the public node API.** Nothing this node emits promises
either; there is a test asserting so.

---

## Nodes

### Pulse Slate · MiniMax H3

Compiles a timeline into a storyboard prompt and reference set, and stops. It
does not sample.

- **One window** → hands back `positive` / `latent` for your own sampler.
- **Longer than one window** → those two outputs are blocked; take the `timeline`
  output to a **Pulse Render** node.

Windows are partitioned `balanced` by default: 16s at a 15s ceiling becomes two
~8s windows, not 15s + 1s. A 1s trailing window is below H3's trained floor and
reliably looks broken. `fill` is available when you mean windows of a literal
length, and `shot_aligned` puts the seams **on shot cuts** wherever the frame
grid allows it — a shot that straddles a seam is compiled into *both* windows,
described twice and hashed into both window seeds, so aligning is what stops that
happening. It cannot hit every cut: cumulative position after k windows is
17K + 5k, so a seam lands on a cut at frame p only when p ≡ 5k (mod 17). It takes
the largest subset it can hit, gives up `window_seconds` to hit them, and names
the cuts it could not reach in the report.

`aspect_ratio` picks the canvas, and each preset is fitted to H3's 1,032,192 px
budget at load time rather than hardcoded, so the table cannot drift from the
budget: 16:9 is 1344×768, 9:16 is 768×1344, 21:9 is 1536×672, 1:1 is 992×992,
4:3 is 1152×864. `custom` uses the `width`/`height` widgets instead.

`continuity` chooses how windows join: `none`, `last_frame_carry`, or
`keyframe_pairs`. The latter two pin a frame, which is the fl2va checkpoint
specifically, so they need `model_fl2va` and **fail at compile time** without it
rather than falling back to something that looks like it worked.

### Pulse Shot

One node per shot. Its text, its length, its continuity override, its own
first/last frames and its own scene-local references — and because those frames
are real `IMAGE` inputs, a shot can open on a frame generated elsewhere in the
same graph rather than on a file you had to export first.

Connecting any shot socket makes the **shot text box inactive**. The two are never
merged and neither is silently preferred; `compiled_prompt` says which won, at the
top, every time.

Each shot's identity is written once and never changes, which is what keeps its
seed and its cached segment attached to it when you reorder the film.

#### `ref_audio`, and what `ref_audio_mode` decides

H3 **always** generates its own audio track. What a reference recording changes is
what the model does alongside it, and the two options are genuinely different jobs:

- **`lip_sync`** (default) — the character's mouth matches your recording. The
  prompt says so explicitly, naming the tag, because that sentence *is* the
  mechanism: the tokenizer emits only the marker `<Audio j>: ` and the waveform
  never reaches the language model, so wiring the socket tells the model nothing
  on its own. The clip is also trimmed to the window's exact span — a 30-second
  file against a 9.42-second window asks the model to align two different
  stretches of time, and the result is a mouth that tracks nothing.
- **`voice_timbre`** — the model speaks this shot's own dialogue and borrows only
  the character of the voice. No alignment, no trim.

The mode is part of the cache key, so switching it re-renders rather than handing
back a segment made under the other instruction.

On a `lip_sync` shot the audio H3 generates is a re-synthesis of your recording —
close, but not your take. `PulseRender.use_reference_audio` muxes the original
into the finished film instead. The generated track is still written to every
segment's `.flac`, so it is a mux choice you can flip and requeue without
re-rendering anything.

**A `lip_sync` reference on a shot with no dialogue produces no mouth movement**,
and nothing warns you: there is no speech to match. Put the recording on the shot
that speaks.

The inverse is reported. A shot carrying both a quoted line and a `lip_sync`
reference gives the model two answers to "what is she saying" — the `<d>` block
instructs it to speak those words, and the recording says whatever it says. The
compiler names it in the report rather than refusing, because a quote that *is*
the recording's transcript is legitimate and only you know that. Otherwise drop
the quote and let the audio carry the words; keep the `@Voice` tag in the line,
which is what tells the model whose shot the recording belongs to.

#### `speaker`, and why a two-hander needs it

The socket carries a waveform and an ordinal. It does not carry an owner — the
tokenizer emits `<Audio j>: ` and the waveform never reaches the language model
— so on a shot with one character there is nothing to confuse and on a shot with
two there is nothing to go on. ComfyUI ships a known bug of exactly that shape
([Comfy-Org/ComfyUI#15454](https://github.com/Comfy-Org/ComfyUI/issues/15454)):
the intended character's lips move, and the other character's accent comes out
of them.

MiniMax's reference format binds a voice to a face with a **global speaker id**.
Type the character's `@Name` into `speaker` — a bin asset, or this shot's own
`@Ref1` — and the compiler:

- assigns that character `(S1)`, `(S2)`, … **at their first line in the film**,
  and keeps it for every later shot and every later window. The id is what tells
  the model that the person talking after a cut is the person who talked before
  it, so it is assigned once across the whole timeline rather than per window;
- stamps it on that character in the shots they actually speak in, and nowhere
  else — `<Subject 1> (S1) crosses to the counter`. A silent character in the
  background stays unstamped; an id there reads as a cue to give them a line;
- binds this shot's `ref_audio` to them by name, so the prompt says
  `` `<Audio 1>` is the speech <Subject 1> (S1) is saying `` rather than "this
  character";
- adds a `retention_analysis` line in MiniMax's audio vocabulary —
  `fully_copy` for `lip_sync`, `reference` for `voice_timbre`.

A character with a description in the bin becomes `<Subject N> (S1)`. One
without becomes `<Picture N> (S1)`, which is much weaker and still binds the
voice to a face.

Leave it blank on a shot where nobody speaks, and on a one-hander where there is
nothing to confuse — the prompt is then byte-for-byte what it was before the
field existed, and your cache is not invalidated. A name that matches nothing is
reported, never guessed at: binding a voice to the wrong face is worse than
leaving it unbound.

#### A voice in the Asset Bin

A recording dropped in the bin has no shot to read a speaker off, so it says who
it belongs to itself. The audio row in the panel carries two controls:

- **role** — `lip_sync` or `voice_timbre`, the same two jobs `ref_audio_mode`
  picks between. Until this shipped there was no way to say it on a bin
  recording, so every one of them compiled as a timbre reference and only a
  `PulseShot`'s own socket could ask for lip sync.
- **whose voice** — the character, picked by name and **stored as an asset id**.
  A binding to "reference 3" would follow whatever landed in slot 3 after the
  next bin edit and report nothing; a binding to an id follows the character
  through renames, reorders and renumbering.

That is what makes a film with three characters and three voice files work:
each recording names its own owner, each owner keeps one speaker id, and the
prompt says which is which. Supplying somebody's voice makes them a speaker even
if no shot ever named them — they are numbered after everyone who has a line, so
adding a voice reference cannot renumber a character already on screen.

Binding to a picture that is not in the bin, or to another recording, is refused
rather than written into the prompt. A voice belongs to somebody the model can
see.

Where both apply — a shot's own `ref_audio` that also carries an explicit
`voice_of` — the explicit binding wins. The author naming a character beats the
wiring implying one.

**Carry-over will not evict a bound voice first.** A continuation window's
carry-over claims the front of the audio group, so the last user recording is
dropped when the group is full. Chosen by bin position that is a coin flip, and
a character keeping their picture and their lines while losing their voice from
window 2 onward reads as drift rather than as a missing reference. Unbound clips
go first, then `voice_timbre`, then `lip_sync` — losing an alignment
desynchronises a mouth, which is visible. The drop is still reported either way,
and the survivors keep their bin order so nothing renumbers that did not have to.

### Pulse Render

Executes a `PULSE_TIMELINE`, reusing every window already on disk. See
[the segment cache](#the-segment-cache).

- `cache_mode` — `auto` reuses what is unchanged; `force_rerender` ignores the
  cache; `reuse_only` refuses to render anything and aborts naming the first
  missing window, for assembling a final cut without regenerating a frame.
- `dry_run` — the report and nothing else.
- `low_memory` — 8-bit frame accumulation and VRAM released between windows. The
  finished video is assembled by stream-copying the segment files, so a
  twelve-window film is never held in RAM.
- `prune_unused` — off by default. Segments from earlier edits are what make
  flipping back to an earlier edit free, so nothing is ever deleted automatically.
- `seam_treatment` — what to do where two windows meet. The container-level seam
  is solved by construction (placement is computed from frame counts, never
  measured from timestamps), but the two sides are two independently generated
  takes: a score that restarts, a level that steps, a grade that drifts.
  `audio+colour`, the default, level-matches and tapers each audio join and grades
  a window's opening towards the frame the previous one ended on, decaying over 12
  frames — a flat correction across a whole window does not remove a cut, it moves
  it to the next seam. `audio` does the first only. `off` leaves both untreated,
  which is what you want when you are A/B-ing it, and colour matching is the kind
  of correction that occasionally makes things worse. A window following a
  *cached* segment is never colour-matched — the segment either side of it is
  already encoded, so only one side of that seam can move — and the report says so
  rather than quietly doing half the job.

Every approximation patched onto the incoming model is detected and folded into
the cache key. Sol-Attn, Spectrum and EasyCache all change what the same prompt at
the same seed produces, and a cache that ignored them would hand you a film whose
shots were rendered at different sparsities without saying so.

### Pulse Bench

Reads `PulseRender` manifests back and prints seconds-per-frame and peak VRAM
grouped by patch chain. Sol-Attn and Spectrum address *different* memory, and
community reports on Spectrum's speed effect disagree — so this pack measures
rather than recommends. Point it at a run folder after a render and compare rows.

### Pulse Retake · MiniMax H3

Mark a bad span; the node pins the exact frame **before** the cut as
`first_frame` and the exact frame **after** it as `last_frame`, renders only the
gap, and stitches.

This is the one job where H3's two-anchor limit is an advantage rather than a
compromise — two anchors is exactly what a patch needs.

Patch length must sit on the frame grid, so the **cut snaps to fit** rather than
being rejected after you've made it. The stitch is length-preserving by
construction and asserted at runtime: a mismatch would shift every frame after
the patch and desync the audio. `keep_base_audio` defaults on, because a
re-rendered patch invents its own score and will not match the surrounding track.

### Pulse Still · MiniMax H3

A still is a 5-frame render where you keep one frame — same conditioning nodes,
same compiler, same reference marshalling.

- `frame_pick` (0–4): your source is pinned at frame 0, so 0 hugs the original
  and 4 drifts furthest. This is the edit-strength dial, exposed rather than
  hidden.
- `canvas_from_reference`: fits the source's aspect into H3's 1,032,192 px budget
  on the /32 grid. The long edge rounds down; the short edge takes whichever grid
  step is *nearest* the true ratio and still inside the budget. Flooring both axes
  independently loses up to a step on each and the two losses compound rather than
  cancel — 16:9 came out 1344×736, 4% under budget at an actual ratio of 1.826.
  It is the same function the `aspect_ratio` presets use, so a 1920×1080 reference
  and the "16:9 landscape" preset cannot resolve to different canvases.

Deliberately a mode, not a second product. No layers, no masks, no inpainting.
The moment it grows a brush it has become a different application.

---

## Architecture

```
comfyui_pulse_studio/          headless core — stdlib only, no torch, no comfy
  constants.py          every value, with the source symbol it came from
  frames.py             the 17k+5 grid; window partitioning
  assets.py             the bin, the budget, and the ONE place ordinals are assigned
  binops.py             UI-facing ops; renumber preview
  timeline.py           the data model that rides in the hidden JSON widget
  compiler.py           timeline -> storyboard prompt + ordered file list
  parsing.py            the two prompt boxes -> shots and project fields
  widget_state.py       merge-not-replace document edits (the erasure fix)
  sockets.py            Autogrow dict shape: contiguous, 0-based, gapless
  retake.py             cut geometry, anchor legality, stitch integrity
  still.py              canvas fitting, frame_pick, branch selection
  patches.py            what the incoming model is missing — duck-typed, imports nothing
  pulse_timeline.py     the PULSE_TIMELINE document, shot ids, window seeds, continuity
  fingerprint.py        every approximation on the model → one canonical hash
  segcache.py           cache key, manifest, durable writes, reuse-vs-render
  report.py             the §9 report; packed-sequence geometry
  bench.py              manifest timings grouped by patch chain
media.py                torch/PIL/PyAV loading — the only tensor code
render.py               the executor: the window loop, the disk writes, the assembly
nodes.py                ComfyUI binding — node faces only
js/ps_widget_guard.js   the prompt-widget write trap (isolated so it is testable)
js/ps_widget_order.js   WIDGET_NAMES[node][version] — the slot contract's name table
js/ps_sockets.js        growing socket groups; links restored by name, not by index
js/ps_warnings.js       paints the patch warning on the node face; owns no widget slot
js/pulse_slate.js       the node face: prompt cards, asset bin, thumbnails
tests/js/               Node tests for the JavaScript that can abort a load
```

`segcache.py` is deliberately free of ComfyUI imports even though it writes files
(spec §15.3): the entire resume behaviour — cache keys, manifest durability,
reuse-vs-render — is exercised on a machine with no GPU. `render.py` holds the
half that cannot be, so the boundary between "tested" and "verified on hardware"
runs along a file rather than through one.

Nothing in `comfyui_pulse_studio/` imports torch or comfy — and that is enforced by an
AST test, not a convention, so it survives the phase where the timeline canvas
wants to decode a preview frame. It is what lets the correctness live somewhere
testable: the whole suite runs in a few seconds with no GPU, no ComfyUI, and no
pip install.

### Window continuity

Two mechanisms hold a multi-window film together. Both originate upstream; see
[Credits](#credits).

**Chunked windows.** A timeline longer than one H3 call is cut into chained
windows at a fixed frame ceiling. `frames.py` owns the partitioning policies, the
124-frame trained floor, the backwards tail merge and the rebalancing that keeps
window lengths even.

**Audio carry-over.** On each continuation window, the previous window's decoded
audio tail is fed back through the reference audio sockets, so the window
continues the existing score and room tone rather than generating an unrelated
one. The tail length is a tunable (`carry_audio_seconds`, default 4 s). A reused
window decodes nothing, so its successor's carry-over is rebuilt from the cached
segment's own PNG and FLAC rather than from tensors in memory.

### Two erasure classes, both closed structurally

**Prompt erasure.** The bin panel used to own the timeline JSON, with a
`catch { return {assets: []} }` fallback that discarded every other field. Any
path reaching it wrote an assets-only document over the real one. Fixed by
moving prompts into their own widgets, making parse failure an error rather than
a default, and doing every edit as a server-side *merge*. Queuing cannot write to
a prompt widget; there is a regression test that types into both boxes with an
empty timeline, queues, and asserts the text survives.

**Socket erasure.** `io.Autogrow` inputs arrive as a **dict** keyed
`ref_image_0`, `ref_image_1`, … and the tokenizer numbers by position in it. A
file that fails to load would leave a hole, and a hole shifts every tag after it.
Unloadable references are therefore dropped *before* tags are assigned, so
ordinals stay dense by construction; `check_socket_groups` is the assertion under
that, and a decode failure at render time stops the run rather than misnumbering.

The JS panel never computes an ordinal itself — it asks the server. A second
implementation of the numbering rule in JavaScript is exactly how the two would
drift apart, and a drifted tag renders successfully while describing the wrong
picture.

---

## Tests

```bash
python3 run_tests.py        # the whole suite, no Python dependencies
python3 run_tests.py -v
node tests/js/test_widget_guard.mjs   # also run by the suite when node exists
node tests/js/test_widget_order.mjs
node tests/js/test_sockets.mjs
```

CI runs exactly that, on Python 3.10, 3.11 and 3.12, on every push and pull
request — plus a job that imports the pure core on a machine with nothing
installed, so a stray `torch` import cannot hide behind another step. See
[.github/workflows/ci.yml](.github/workflows/ci.yml).

Covers frame quantisation (including behavioural parity with core's own
`align_frame_count`), timestamp monotonicity, tag renumbering under
add/remove/reorder/soundtrack-toggle, budget enforcement, window partitioning,
retake geometry over an exhaustive sweep of reachable cuts, canvas fitting, and
end-to-end invariants such as *every tag cited in a prompt must correspond to a
socket that window actually carries*.

Also asserted, over all four shipped graphs rather than over one of them: stored
widget values stay aligned with `INPUT_TYPES` (a widget inserted in the middle
would silently shift every value after it), the stored canvas matches the preset
the graph selects, every link is backfilled in both directions, every emitted
window is at least 124 frames, no graph carries a muted branch or a floating
`PulseShot`, no graph cites a reference file the pack does not ship, and no graph
wires a DiT checkpoint nothing in it ever samples with — ~20 GB resident for a
model no window uses.

The segment cache gets its own file, covering each acceptance behaviour in the
spec's §7.5 — resume after a kill, one-shot edits invalidating one window, seed
and `steps` changes invalidating all of them, a requeue with no changes rendering
nothing — plus the properties underneath them: that a cache key is stable across
processes, that a manifest entry without its file is not trusted, that a corrupt
manifest is moved aside rather than overwritten, and that inserting a shot does
not move the seed of any window whose shot set is unchanged.

Patch detection is tested against stub patchers for each of the four packs,
present and absent, including the one property that would quietly disable the
whole cache: **nothing in the descriptor may carry a memory address**, or two
runs of the same graph would key differently and re-render for ever.

The JavaScript that can abort a workflow load is tested too. `protectWidget` is
idempotent across reloads, declines a non-configurable descriptor rather than
throwing, and **wraps** any existing accessor instead of replacing it — replacing
it would sever the widget from ComfyUI's serialization and lose the typed prompt
on save. Every LiteGraph hook is wrapped in `try`, because a throw in
`onConfigure` is what produces *"Loading aborted due to error reloading workflow
data"*.

---

## Maintainers

- Behailu Weldeyohannes — [@Behailu-Weldeyohannes](https://github.com/Behailu-Weldeyohannes), [@behailuaisolutions](https://github.com/behailuaisolutions)

## Acknowledgements

Developed with assistance from Claude, Anthropic's AI assistant.

---

## Credits

This project is a fork of
[muse-collective-26/MiniMaxH3-Director-Seed-Hunt](https://github.com/muse-collective-26/MiniMaxH3-Director-Seed-Hunt),
licensed under the MIT License. The upstream copyright notice is reproduced in
[NOTICE](NOTICE), and derived modules carry an attribution header.

Two design elements originate upstream:

- Chunking a long timeline into chained fixed-ceiling windows.
- Audio carry-over across a window seam.

Their implementation in this project is described under
[Architecture](#window-continuity).

## License

**Apache-2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Upstream is MIT, which is compatible with an Apache-2.0 project provided
attribution survives; upstream's copyright notice is reproduced in full in
[NOTICE](NOTICE) as that licence requires. muse-collective MIT code **is**
present in this tree — stated plainly there so the question is settled on the
record.

Model weights are **not** redistributed here and carry the MiniMax H3 Community
License, which is a separate agreement governing the weights and their use.

Source provenance rules for contributors — what may be read, what may not be
copied, and the named GPL-3.0 projects that rule covers — are in
[CONTRIBUTING.md](CONTRIBUTING.md).
