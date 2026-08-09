# PulseSlate v3 — Build Instructions

Target repo: `ComfyUI-PulseStudio`
Current state: single node `PulseSlate` (schema 2, widget version `2.0.0`) that both compiles conditioning and renders internally.
Goal: split compile from render, add a per-shot node, add segment caching with resume, and fix seed and reference scoping.

Read every section before writing code. Do not skip Section 1.

---

## 1. Invariants — do not change these

These are already correct. Breaking any of them is a regression.

1. **Model patches stay upstream.** `model`, `clip`, `vae`, `audio_vae`, `model_fl2va` remain node **inputs**. Do not add checkpoint-name widgets that load models inside the node. Every attention and memory patch — Spectrum, Sage Attention, Sol-Attn, feed-forward chunking — is applied before the model reaches the node, because the long-form path samples internally with the model it is handed. See Section 12.
2. **Alias resolution stays.** Users reference assets by name (`@Image1`, `@Mimi`, `@[Cafe wide]`). Ordinals (`<Picture 3>`) are assigned at compile time from bin order. Never require or accept hand-typed ordinals as the primary binding.
3. **Frame math stays.** Frames snap to `17k + 5`. Ceiling per window is 362 frames. Minimum window is 124 frames; a short tail merges backwards. A total under 124 frames renders as one short window.
4. **cfg defaults to 1.0** and uses `BasicGuider`. Above 1.0 switches to `CFGGuider` with an empty negative.
5. **Asset budget stays.** 9 images, 3 videos, 3 audio, 12 files total.

---

## 2. Node split

Replace the single `PulseSlate` node with three nodes.

### 2.1 `PulseSlate` — compiler only

Display name: `Pulse Slate · MiniMax H3`

Keeps: global prompt box, shot text box, drag-drop asset bin, duration, aspect, width, height, steps, sampler, scheduler, cfg, seed, seed mode, and all existing compile-time widgets.

Removes: all internal rendering. This node no longer samples, decodes, or stitches.

Outputs:

| name | type | notes |
|---|---|---|
| `timeline` | `PULSE_TIMELINE` | new primary output, see Section 3 |
| `positive` | `CONDITIONING` | single-window path only; `None` if the compiled plan has more than one window |
| `latent` | `LATENT` | single-window path only; `None` if more than one window |
| `combined_audio` | `AUDIO` | unchanged |
| `images` | `IMAGE` | unchanged |
| `compiled_prompt` | `STRING` | expanded, see Section 9 |

When the plan has more than one window, `positive` and `latent` return `None` and `compiled_prompt` states plainly that the timeline requires `PulseRender`.

### 2.2 `PulseShot` — per-shot node

Display name: `Pulse Shot`

Output: `shot` of type `PULSE_SHOT`.

Inputs (all optional except where noted):

| name | type | notes |
|---|---|---|
| `start_image` | `IMAGE` | optional; first frame of this shot |
| `end_image` | `IMAGE` | optional; last frame, enables keyframe-pair continuity |
| `refs.ref_image_1..N` | `IMAGE` | dynamic group, scene-local references |
| `ref_audio` | `AUDIO` | scene-local voice or effect sample |

Widgets:

| name | type | default |
|---|---|---|
| `label` | STRING, single line | `""` |
| `visual` | STRING, multiline | `""` |
| `audio_line` | STRING, multiline | `""` |
| `duration_seconds` | FLOAT, 0.5–15.08, step 0.01 | `5.0` |
| `continuity` | COMBO: `inherit`, `none`, `last_frame_carry`, `keyframe_pairs` | `inherit` |
| `shot_id` | STRING, hidden | auto-generated, see Section 6 |

Accepting `IMAGE` inputs is the point of this node. It lets shot frames come from upstream generators in the same graph instead of only from files dragged onto the bin.

### 2.3 `PulseRender` — executor

Display name: `Pulse Render`

Inputs:

| name | type | required |
|---|---|---|
| `timeline` | `PULSE_TIMELINE` | yes |
| `model` | `MODEL` | yes |
| `vae` | `VAE` | yes |
| `audio_vae` | `VAE` | yes |
| `model_fl2va` | `MODEL` | optional |

Widgets:

| name | type | default |
|---|---|---|
| `cache_mode` | COMBO: `auto`, `force_rerender`, `reuse_only` | `auto` |
| `run_dir` | STRING | `pulseslate` |
| `run_id` | STRING | `""` (empty means derive from timeline hash) |
| `save_segments` | BOOLEAN | `True` |
| `low_memory` | BOOLEAN | `True` |
| `dry_run` | BOOLEAN | `False` |

Outputs:

| name | type |
|---|---|
| `video` | `VIDEO` |
| `frames` | `IMAGE` |
| `audio` | `AUDIO` |
| `segment_paths` | `STRING` |
| `report` | `STRING` |

**Reason for the split:** the current starter graph carries the short path and the long path as two parallel wired groups, one of them muted, with a Ctrl+M instruction in the README. That is fragile and lets the two paths drift apart. After the split, a short timeline wires `PulseSlate → BasicGuider/SamplerCustomAdvanced`, and a long timeline wires `PulseSlate → PulseRender`. Nothing is muted and nothing is duplicated.

---

## 3. `PULSE_TIMELINE` schema

`PULSE_TIMELINE` is a plain Python dict. Serialize it with `json.dumps(obj, sort_keys=True, separators=(",", ":"))` whenever it is hashed.

```json
{
  "schema": 3,
  "node_version": "3.0.0",
  "global": {
    "style": "string",
    "identity": "string",
    "retention": "string",
    "soundscape": "string",
    "music": "string",
    "raw": "string"
  },
  "refs": {
    "global": [
      {"ordinal": 1, "kind": "image", "alias": "Image1", "source": "bin", "file": "name.jpg", "sha256": "..."}
    ]
  },
  "shots": [
    {
      "shot_id": "stable-uuid-string",
      "index": 0,
      "label": "The Delivery",
      "visual": "string",
      "audio_line": "string",
      "duration_seconds": 5.0,
      "continuity": "last_frame_carry",
      "start_image_ref": "tensor_slot_0 | null",
      "end_image_ref": "tensor_slot_1 | null",
      "local_refs": [
        {"ordinal": 4, "kind": "image", "alias": "Prop1", "source": "socket", "sha256": "..."}
      ],
      "resolved_prompt": "string with <Picture N> substituted",
      "unresolved_aliases": []
    }
  ],
  "windows": [
    {
      "window_index": 0,
      "shot_ids": ["...", "..."],
      "frames": 362,
      "fps": 24,
      "width": 1344,
      "height": 736,
      "seed": 1234567,
      "steps": 20,
      "sampler": "res_multistep",
      "scheduler": "simple",
      "cfg": 1.0,
      "continuity_in": "none",
      "continuity_out": "last_frame_carry",
      "cache_key": "sha256-hex"
    }
  ],
  "budget": {"images": 3, "videos": 0, "audio": 0, "total": 3},
  "warnings": ["string"]
}
```

Image and audio tensors are **not** stored in the timeline dict. Keep them in a side channel object attached to the returned tuple, keyed by the `*_ref` slot names. The dict must stay JSON-serializable so it can be hashed and written to the manifest.

---

## 4. Dynamic sockets

`PulseSlate` gains a dynamic input group `shots.shot_1 .. shots.shot_N` of type `PULSE_SHOT`.
`PulseSlate` gains a dynamic input group `refs.ref_image_1 .. refs.ref_image_8` of type `IMAGE`, plus `ref_video` (IMAGE), `ref_video_audio` (AUDIO), `ref_music` (AUDIO).

Implement these the same way the reference storyboard node does it: dot-namespaced optional inputs declared under `optional` in `INPUT_TYPES`, rendered with the optional socket shape. Connecting the last free socket must cause a new empty socket to appear. Sockets are ordered by their numeric suffix, not by connection time.

---

## 5. Precedence rule — one source of truth

If **any** `shots.shot_i` socket is connected:

- the shot text box is ignored for compilation,
- the text box is still shown but the node prepends a visible warning to `compiled_prompt`: `Shot text box ignored: N PulseShot nodes are connected.`

If no `shots.shot_i` socket is connected, parse the text box exactly as today.

Never merge the two. Never silently prefer one.

---

## 6. Stable per-shot seeds

Current behaviour derives a window seed from position. That rerolls every downstream shot the moment a shot is inserted.

Replace it:

1. On creation, each `PulseShot` writes a UUID4 string into its hidden `shot_id` widget and never changes it. Copying a node generates a new `shot_id`.
2. Text-box shots derive a stable id as `sha1(label + "|" + visual)` truncated to 16 hex chars.
3. Window seed is:

```python
def window_seed(base_seed: int, shot_ids: list[str]) -> int:
    h = hashlib.sha256("|".join(shot_ids).encode("utf-8")).digest()
    return (base_seed ^ int.from_bytes(h[:4], "big")) & 0x7FFFFFFF
```

Inserting, deleting, or reordering shots must not change the seed of any window whose shot set is unchanged.

---

## 7. Segment cache, manifest, and resume

This is the highest-value item in the whole spec. Implement it fully.

### 7.1 Cache key

Per window, the cache key is `sha256` over the canonical JSON of exactly these fields, in this order:

```
global block (all five labelled fields plus raw)
shot blocks belonging to this window (shot_id, label, visual, audio_line, duration_seconds, continuity, resolved_prompt)
resolved reference descriptors for this window (ordinal, kind, alias, sha256)
window.frames, window.fps, window.width, window.height
window.seed, window.steps, window.sampler, window.scheduler, window.cfg
window.continuity_in, window.continuity_out
model_fingerprint
patch_fingerprint
node_version
```

`model_fingerprint` is a short stable string identifying the loaded checkpoint. Use the model's state-dict-independent identity if available; otherwise `sha256` of the sorted list of `(param_name, tuple(shape))` pairs, truncated to 16 hex chars. Compute it once per run and cache it in memory.

`patch_fingerprint` covers every approximation applied to the model upstream. It is mandatory, not optional — see Section 12.4. Sol-Attn, Spectrum, and EasyCache all change the output of the same prompt and seed, so a cache that ignores them will happily reuse a segment rendered at a different sparsity and hand back a film whose shots do not match each other.

Do **not** key on window index. Index keying cannot distinguish "shot 3 was edited" from "a shot was inserted before shot 3", and it invalidates work that is still valid.

Reference images arriving through sockets are hashed from their tensor bytes. Reference files from the bin are hashed from file contents.

### 7.2 On-disk layout

```
ComfyUI/output/<run_dir>/<run_id>/
  manifest.json
  seg_0000_<cachekey12>.mp4
  seg_0000_<cachekey12>.audio.flac
  seg_0001_<cachekey12>.mp4
  ...
```

`run_id`, when the widget is empty, is `sha256` of the timeline dict with the `windows[].seed` and `cache_key` fields removed, truncated to 12 hex chars. That makes the same project resume into the same folder across sessions while still allowing seed changes to reuse the folder.

### 7.3 `manifest.json`

```json
{
  "schema": 1,
  "run_id": "abc123def456",
  "node_version": "3.0.0",
  "created_utc": "2026-08-08T00:00:00Z",
  "updated_utc": "2026-08-08T00:00:00Z",
  "model_fingerprint": "0a1b2c3d4e5f6071",
  "segments": [
    {
      "window_index": 0,
      "cache_key": "full-sha256-hex",
      "shot_ids": ["..."],
      "frames": 362,
      "fps": 24,
      "video_path": "seg_0000_abc123def456.mp4",
      "audio_path": "seg_0000_abc123def456.audio.flac",
      "last_frame_path": "seg_0000_abc123def456.last.png",
      "render_seconds": 412.7,
      "status": "complete"
    }
  ]
}
```

### 7.4 Execution loop

For each window in order:

1. Compute `cache_key`.
2. If `cache_mode` is `force_rerender`, render.
3. Else if the manifest has an entry with this `cache_key`, `status == "complete"`, and every referenced file exists on disk, load it and skip rendering. Log `reused`.
4. Else if `cache_mode` is `reuse_only`, abort the run with a clear error naming the first missing window.
5. Else render.

After each window finishes rendering:

- write the segment video and audio to disk **immediately** when `save_segments` is on,
- write `last_frame_path` as PNG when `continuity_out` is `last_frame_carry`,
- append or update the manifest entry and `fsync` the manifest before starting the next window.

The manifest write must happen after the media files are closed, never before. A crash between the two must leave a manifest that does not claim a file that is absent.

Stale entries whose `cache_key` no longer appears in the current timeline are left on disk and left in the manifest. Do not delete them. Add a `prune_unused` boolean widget defaulting to `False` if cleanup is wanted later.

### 7.5 Acceptance behaviour

- Kill the process at window 9 of 12, requeue the identical workflow: windows 0–8 load from disk, only 9–11 render.
- Edit shot 3's `visual` text and requeue: only the window containing shot 3 re-renders. Every other window is reused.
- Change the base seed and requeue: every window re-renders.
- Change `steps` and requeue: every window re-renders.
- Change nothing and requeue: nothing renders, and the output video is byte-identical to the previous assembly.

---

## 8. Memory: 8-bit frame accumulation

When `low_memory` is on, accumulate assembled frames as `uint8` in the range 0–255 and convert to float only at the encode boundary. Do not hold the full float32 frame stack in RAM.

Preferred implementation: never assemble a full frame stack at all. Write each window to its own file and concatenate at the end with a stream copy. Fall back to the uint8 buffer only when the `frames` output socket is actually connected.

Target: a 12-window timeline must not exceed roughly one quarter of the current peak system RAM.

---

## 9. Compiler report and dry run

Expand `compiled_prompt` and add the `report` output on `PulseRender`. The report must be plain text, readable in `PreviewAny`, and must contain:

1. Window table: index, shot labels, frame count, duration, seed, cache status (`will render` / `will reuse`).
2. Ordinal map **per shot**: which alias resolved to which `<Picture N>` or `<Audio N>` in that shot's prompt.
3. Every unresolved `@Alias`, with the shot it appeared in.
4. Budget usage against the 9/3/3/12 limits.
5. The detected upstream patch chain and its `patch_fingerprint`, plus a warning if the chain is empty. See Section 12.4.
6. Estimated peak VRAM per window and estimated total render time, based on measured seconds-per-frame from the manifest when prior runs exist.
7. The set of distinct packed sequence lengths across all windows, with a warning when there is more than one. See Section 12.5.

When `dry_run` is `True`, `PulseRender` produces the report and nothing else: no sampling, no decode, no file writes, and `video`/`frames`/`audio` return `None`.

A wrong ordinal binding currently renders successfully and produces the wrong film. The dry run exists to catch that before any GPU time is spent.

---

## 10. Reference scoping

Two scopes, resolved in one pass.

1. **Global references** — the drag-drop bin plus `refs.ref_image_1..8`, `ref_video`, `ref_video_audio`, `ref_music` on `PulseSlate`. These receive ordinals `1..N` and hold those ordinals identically in every shot's prompt. `ref_music` is always the last audio ordinal.
2. **Scene-local references** — `refs.ref_image_*` and `ref_audio` on a `PulseShot`. These receive ordinals continuing after the global block, **within that shot only**, and are not visible to any other shot.

Ordinal assignment happens at compile time. The user still writes `@Alias` everywhere and never types a number. Emit a warning for any literal `<Picture N>` or `<Audio N>` found in user text, and leave it untouched.

---

## 11. Continuity modes

Add `continuity` as a `PulseSlate` widget (`none`, `last_frame_carry`, `keyframe_pairs`) and as a per-shot override (`inherit` plus the same three).

- `none` — each window samples independently.
- `last_frame_carry` — the decoded last frame of window *i* is passed as the start image of window *i+1*. Requires `model_fl2va`.
- `keyframe_pairs` — each shot's `start_image` and the next shot's `start_image` form a first/last-frame pair. Requires `model_fl2va` and requires `start_image` on every shot in the chain.

If a mode requires `model_fl2va` and it is not connected, fail at compile time with an explicit message. Do not fall back silently.

---

## 12. Attention and memory backends — Sol-Attn, Sage, Spectrum

### 12.1 What PulseStudio implements, and what it does not

**Do not vendor, wrap, or reimplement Sol-Attn.** It is a separate node pack with a vendored NVIDIA Triton kernel under Apache-2.0. Copying it into this repo creates a licensing and maintenance liability for no benefit, and it would fight the upstream repos on every kernel update.

PulseStudio's job is four things:

1. **Detect** which patches are on the incoming model.
2. **Fold** them into the cache key so approximate settings cannot silently poison cached segments.
3. **Report** the detected chain in the compiler report.
4. **Ship** starter workflows wired for the recommended chain.

Nothing in this section adds a kernel to this repo.

### 12.2 The three packs the user has installed

| Pack | Node used | What it does |
|---|---|---|
| `ComfyUI-sol-attn` (Saganaki22) | `MiniMax H3 Memory Efficient Sol Attention Patch`, `MiniMax H3 Scheduled Sol Attention Patch`, `MiniMax H3 Chunk FeedForward` | Sparse attention with a zero-copy H3 path; MLP peak-memory reduction |
| `ComfyUI-KJNodes` (kijai) | `PathchSageAttentionKJ` | Dense int8 attention; becomes Sol-Attn's fallback backend |
| `ComfyUI-Spectrum-MiniMax-H3` (xmarre) | `SpectrumApplyMiniMaxH3` | History offload to system RAM plus step skipping |

### 12.3 Chain order — this is not optional

```
UNETLoader
  → SpectrumApplyMiniMaxH3        (history_storage: system_ram)
  → PathchSageAttentionKJ         (sets Sage as the fallback backend)
  → MiniMax H3 Scheduled Sol Attention Patch
  → MiniMax H3 Chunk FeedForward
  → PulseSlate / PulseRender
```

**The Sol-Attn H3 node must come after the Sage patch.** Applied after it, Sol adopts the Sage forward as its fallback, so ineligible steps and short sequences run memory-efficient Sage while eligible steps run Sol-Attn. Applied *before* it, the Sage patch shadows the Sol node entirely and it does nothing. This ordering trap is silent — the graph runs, produces output, and gives none of the speedup.

Ship this order in the starter workflow. Do not leave it to the user to discover.

### 12.4 `patch_fingerprint` — mandatory cache-key component

`PulseRender` must inspect the incoming `MODEL` patcher and build a canonical descriptor of every approximation attached to it. At minimum, detect and record:

```json
{
  "sol_attn": {"present": true, "node": "scheduled", "tau_start": 2.0, "tau_end": 0.8,
               "curve": "linear", "dense_percent": 0.2, "min_tokens": 8192,
               "sink_conditioning": "exact_kv_and_rows", "int8_qk": false,
               "thresh_type": "diag"},
  "sage": {"present": true, "mode": "auto"},
  "spectrum": {"present": true, "history_storage": "system_ram", "...": "all widget values"},
  "ff_chunk": {"present": true, "chunks": 2, "min_tokens": 4096},
  "easycache": {"present": false}
}
```

`patch_fingerprint` is `sha256` of that dict serialized with `sort_keys=True`, truncated to 16 hex chars.

Detection strategy, in order of preference:

1. Read `model.model_options` and the patcher's attention override / object-patch registry for known keys.
2. Fall back to class-name and module-path inspection of anything registered on `optimized_attention_override` and on patched attention/MLP forwards.
3. If nothing is detectable, record `{"detected": false}` and put a plain warning in the report: `No attention or memory patches detected on the incoming model. Segment cache cannot protect against a patch change.`

Do not silently succeed with an empty fingerprint. An unlabelled cache is worse than no cache: it produces a film where shot 4 was rendered dense and shot 5 was rendered at `tau=2.0`, and nothing in the UI says so.

Feed-forward chunking is documented as bit-identical output, so it *could* be excluded from the fingerprint. Include it anyway. The cost of an unnecessary re-render is minutes; the cost of a wrong reuse is a broken deliverable.

### 12.5 Triton autotune and window uniformity

Sol-Attn's Triton kernel autotunes keyed on sequence length. **The first run at any new token count pays a JIT sweep inside the sampling loop.** For a windowed render this is a direct cost: a timeline whose windows have different frame counts pays that sweep once per distinct length.

Two required behaviours in `PulseRender`:

1. **Uniform window planning.** When the compiler must split a timeline, prefer a plan where every window has the same frame count over one that produces a ragged tail, as long as the 17k+5 snap and the 124-frame minimum still hold. If a ragged plan is unavoidable, say so in the report.
2. **Warmup accounting.** Do not include the first window's render time in the seconds-per-frame estimate used for time projection. Record it separately in the manifest as `warmup_seconds`.

### 12.6 Recommended defaults for this pipeline

Put these in the starter workflow. They are starting points to A/B, not settled facts.

| Setting | Value | Reason |
|---|---|---|
| Sol node | `Scheduled` | tau ramps sparse-early, dense-late — sparsity costs least on high-noise steps |
| `tau_start` / `tau_end` | `2.0` / `0.8` | pack defaults |
| `dense_percent` | `0.2` | the paper's recipe; keeps the first fifth of sampling fully dense |
| `sink_conditioning` | `exact_kv_and_rows` | **see below** |
| `min_tokens` | `8192` | below roughly 4K tokens Sage is faster than Sol; the default guard is correct |
| `int8_qk` | `false` | ~1% extra numerical error for a speed gain; leave off until the baseline is validated |
| `chunks` (FF chunking) | `2` | −37% MLP peak on the INT8 ConvRot checkpoint, output bit-identical |

**On `sink_conditioning`:** `exact_kv` (~3% cost) keeps H3's packed text/conditioning/reference/audio KV blocks exact. `exact_kv_and_rows` (~20% cost) additionally runs those query rows dense, which makes the *generated audio stream* exact. PulseSlate pairs each character image with its own audio track for multi-character sync. That pairing lives in exactly the rows `exact_kv_and_rows` protects. Take the 20%.

`PulseSlate` should emit a warning when it compiles a timeline with more than one paired audio asset and the upstream Sol node reports `sink_conditioning` of `exact_kv` or `off`.

### 12.7 What this does not solve

Sol-Attn reduces attention and MLP *activation* peak. Spectrum's `system_ram` history offload is a different mechanism, and it is what currently makes a 362-frame window fit on a 32GB card. **Do not present Sol-Attn as a replacement for Spectrum in any documentation or workflow note.** They address different memory, and community reports on Spectrum's speed effect are mixed — one report has it costing time rather than saving it.

The correct handling is measurement, not a recommendation baked into code. `PulseRender` already writes `render_seconds` and `patch_fingerprint` per segment into the manifest. Add a small `PulseBench` utility node that reads one or more manifests and prints a table of seconds-per-frame and peak VRAM grouped by `patch_fingerprint`. That turns "which chain is faster on this box" from an argument into a lookup.

Note also that Spectrum and EasyCache are reported to be mutually exclusive in practice. If both are detected on the same model, put a warning in the report. Do not block the run.

---

## 13. Migration

- `PulseSlate` schema 2 workflows must still load. Detect `node_version` starting with `2.` and run a converter: preserve the asset bin JSON, the global prompt, the shot text, and all sampler widgets.
- v2 graphs wired into an external sampler keep working, because `positive` and `latent` remain on the node.
- v2 graphs that relied on the internal render path must be told what to do. When the loaded workflow has more than one window and no `PulseRender` is present, emit: `This timeline needs a Pulse Render node. Add it and connect the timeline output.`
- Bump `node_version` to `3.0.0` and `schema` to `3`.

---

## 14. Do not do these

1. Do not add checkpoint or VAE name widgets that load models inside any node. See Section 1.1.
2. Do not key the segment cache on window index. See Section 7.1.
3. Do not derive seeds from position. See Section 6.
4. Do not merge the shot text box with connected `PulseShot` nodes. See Section 5.
5. Do not delete cached segments automatically.
6. Do not reintroduce a muted parallel branch in the starter workflow.
7. Do not vendor or reimplement the Sol-Attn kernel in this repository. See Section 12.1.
8. Do not compute a cache key without `patch_fingerprint`. See Section 12.4.
9. Do not place the Sol-Attn H3 node before the Sage patch in any shipped workflow. See Section 12.3.

---

## 15. Deliverables

1. `PulseSlate`, `PulseShot`, `PulseRender` implemented, registered, and documented.
2. `PULSE_TIMELINE` and `PULSE_SHOT` type definitions.
3. Cache and manifest module, isolated from ComfyUI imports so it can be unit-tested standalone.
4. Updated starter workflow JSON with no muted groups: one graph showing `PulseSlate → PulseRender`, one showing `PulseSlate → SamplerCustomAdvanced`.
5. Updated `MarkdownNote` in the workflow reflecting the new wiring, the cache behaviour, and the dry run.
6. Unit tests covering every bullet in Section 7.5, run without a GPU by stubbing the sampler.
7. Patch detection module and `patch_fingerprint`, with unit tests using stub model patchers for each of the four packs, present and absent.
8. `PulseBench` node that groups manifest timings by `patch_fingerprint`.
9. A second starter workflow variant wired with the full Sol-Attn chain in the order given in Section 12.3, with a `MarkdownNote` stating the ordering trap explicitly.
