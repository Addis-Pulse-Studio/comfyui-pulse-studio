# Pulse Studio — MiniMax H3 for ComfyUI

A ComfyUI director node for [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

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

> **POWERED BY MINIMAX H3.** This repository contains no model weights. See
> [NOTICE](NOTICE) for the model's separate license, which governs your use of it
> and of anything you generate.

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

Compiles a timeline into a storyboard prompt and reference set.

- **One window** → hands back `positive` / `latent` for your own sampler.
  Your graph carries sampler → decode → mux.
- **Longer than one window** → renders internally, one real H3 call per window,
  carrying the previous window's last frame and audio tail forward, and returns
  the stitched `images` and `combined_audio`. Nothing is ever pushed to the
  ComfyUI API.

Windows are partitioned `balanced` by default: 16s at a 15s ceiling becomes two
~8s windows, not 15s + 1s. A 1s trailing window is below H3's trained floor and
reliably looks broken. `fill` is available when you mean windows of a literal
length.

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
- `canvas_from_reference`: fits the source's aspect into H3's 1,032,192 px budget,
  rounding **down** to multiples of 32 — rounding down is what guarantees the
  result stays inside the budget.

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
media.py                torch/PIL/PyAV loading — the only tensor code
nodes.py                ComfyUI binding; the sampling loop
js/ps_widget_guard.js   the prompt-widget write trap (isolated so it is testable)
js/pulse_slate.js    the node face: prompt cards, asset bin, thumbnails
tests/js/               Node tests for the JavaScript that can abort a load
```

Nothing in `comfyui_pulse_studio/` imports torch or comfy — and that is enforced by an
AST test, not a convention, so it survives the phase where the timeline canvas
wants to decode a preview frame. It is what lets the correctness live somewhere
testable: 321 tests run in ~0.1s with no GPU, no ComfyUI, and no pip install.

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

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-PulseStudio
```

Needs `torch`, `numpy`, `Pillow`, `av` — all already present in a standard
ComfyUI install. Restart ComfyUI.

## Tests

```bash
python3 run_tests.py        # 326 tests, no Python dependencies
python3 run_tests.py -v
node tests/js/test_widget_guard.mjs   # also run by the suite when node exists
```

Covers frame quantisation (including behavioural parity with core's own
`align_frame_count`), timestamp monotonicity, tag renumbering under
add/remove/reorder/soundtrack-toggle, budget enforcement, window partitioning,
retake geometry over an exhaustive sweep of reachable cuts, canvas fitting, and
end-to-end invariants such as *every tag cited in a prompt must correspond to a
socket that window actually carries*.

Also asserted: the shipped workflow's stored widget values stay aligned with
`INPUT_TYPES` (a widget inserted in the middle would silently shift every value
after it), and every emitted window is at least 124 frames.

The JavaScript that can abort a workflow load is tested too. `protectWidget` is
idempotent across reloads, declines a non-configurable descriptor rather than
throwing, and **wraps** any existing accessor instead of replacing it — replacing
it would sever the widget from ComfyUI's serialization and lose the typed prompt
on save. Every LiteGraph hook is wrapped in `try`, because a throw in
`onConfigure` is what produces *"Loading aborted due to error reloading workflow
data"*.

---

## Credits

Pulse Studio is a fork of
[muse-collective-26/MiniMaxH3-Director-Seed-Hunt](https://github.com/muse-collective-26/MiniMaxH3-Director-Seed-Hunt)
(MIT). Two things came from there and are worth naming rather than burying in a
licence file:

- **Chunking a long timeline into chained fixed-ceiling windows.** The design is
  upstream's; the partitioning policies, the 124-frame floor and the tail merge
  in `comfyui_pulse_studio/frames.py` are this project's implementation of it.
- **Audio carry-over across a window seam.** Feeding the previous window's
  decoded audio tail back through the reference audio sockets, so each window
  does not invent its own score. That is an empirical finding — observed on real
  renders, not derived — and it is the difference between a seam you can hear and
  one you cannot.

Both derived modules carry an attribution header pointing here, so the credit
survives a refactor by someone who no longer remembers the reason for it.

## License

**Apache-2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Upstream is MIT, which is compatible with an Apache-2.0 project provided
attribution survives; upstream's copyright notice is reproduced in full in
[NOTICE](NOTICE) as that licence requires. muse-collective MIT code **is**
present in this tree — stated plainly there so the question is settled on the
record.

Model weights are **not** redistributed here and carry the MiniMax H3 Community
License, which is a separate agreement governing the weights and their use.

**No code from `seesee75-commits/ComfyUI-MiniMaxH3-Director` (GPL-3.0) is present
here.** That project was reviewed for feature ideas only. Copying from it would
relicense this entire project under GPL-3.0. Contributors must not introduce code
derived from it.
