# Hardware verification — the run that clears the 3.0.0 tag

Everything in `CHANGELOG.md`'s "Pending before the tag" that needs a GPU, in the
order that makes it one sitting instead of six. Each step says what to do, what
you should see, and where to write down what you actually saw.

**Why this document exists.** The whole cache is covered by
`tests/test_segment_cache.py`, which drives the same two functions the executor
calls, in the same order. What no test on a CI runner can reach is the half that
touches disk and VRAM: that a killed render really resumes, that a reused
segment stream-copies into an assembly that plays, and that a carry-over rebuilt
from a segment's PNG and FLAC produces a seam that matches one carried in
memory. Those are claims about your box, and only your box can answer them.

**Fill this file in as you go and commit it.** A verification nobody wrote down
is a verification that has to be done again.

- Box / GPU: `________________`  VRAM: `______`  Driver: `______`
- ComfyUI commit: `________________`  Date run: `____________`
- Pack commit: `________________`

---

## Before you start

1. Pull the pack and restart ComfyUI fully. The frontend serves `js/` from
   `/extensions/<folder-name>/`, and a soft reload will not pick up changed
   JavaScript.
2. The `custom_nodes` symlink is still named `H3_Omni-Director` **on purpose** —
   that folder name is what the extension routes are built from. Do not rename
   it here.
3. Have the input files ready: three reference images and one voice WAV in
   `ComfyUI/input/`.
4. Note your starting free VRAM with nothing loaded. Step 5 needs it.

Total GPU time is roughly **two long renders plus four short requeues**. Steps
1–2 are free. Budget the sitting around step 3.

---

## Step 1 — Sol-Attn node ids (no GPU, do this first)

This one unblocks a change I can make for you, so it comes before anything
expensive.

`example_workflows/PulseSlate_Starter_SpectrumSage.json` documents the two
Sol-Attn nodes in a `MarkdownNote` instead of pre-wiring them, because
[ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) was not
installed on the machine the graph was built on. A workflow naming a guessed
node id loads as a red MISSING NODE even when the pack *is* present, so guessing
was worse than describing.

With the pack installed, read the ids out of its source:

```bash
grep -rn "NODE_CLASS_MAPPINGS" -A 20 \
  /mnt/d/AI_Tools/ComfyUI_Conda/ComfyUI/custom_nodes/ComfyUI-sol-attn/__init__.py
```

Write down the **mapping keys** (the class ids), not the display names:

**Done, 2026-08-09** — the pack is installed and the ids were read from the
running server's `/object_info`, not from source:

| display name | class id | outputs |
|---|---|---|
| MiniMax H3 Scheduled Sol Attention Patch | `MiniMaxH3ScheduledSolAttentionPatch` | **`MODEL`, `IMAGE`** |
| MiniMax H3 Chunk FeedForward | `MiniMaxH3ChunkFeedForward` | `MODEL` |

Three more ship in the same pack: `SolAttentionPatch`,
`MiniMaxH3MemoryEfficientSolAttentionPatch`, `MiniMaxH3FusedModulation`.

**The scheduled patch has two outputs** — `model` on slot 0 and `tau_graph` on
slot 1. Every other node in the chain is single-output, so anything wiring by
assumption gets it wrong.

**Wired, 2026-08-10.** The chain is now in the graph on both model paths. This
went in on the confirmation that the pack is installed and locally tested, ahead
of step 5's numbers rather than because of them — so it ships as the recommended
order, not as a measured win. If the benchmark below comes back against it,
deleting the two nodes and wiring Sage straight through is the whole reversal.

```
UNETLoader → SpectrumApplyMiniMaxH3 → PathchSageAttentionKJ
           → <Sol-Attn H3 patch> → <chunk feed-forward> → PulseSlate / PulseRender
```

**The ordering is the trap.** Sol after Sage adopts the Sage forward as its
fallback, so ineligible steps run memory-efficient Sage while eligible steps run
Sol. Sol *before* Sage is shadowed entirely and does nothing — and the graph
still runs, still produces output, and warns you about none of it.

---

## Step 2 — Both graphs load with no red nodes (no GPU)

Their structure is asserted by test — node ids, slot positions, link backfill in
both directions, the patch chain's direction. "No red nodes" is a different
claim: it is about your install, and only loading proves it.

On a **freshly restarted** ComfyUI, load each and check the console as well as
the canvas:

| workflow | loads clean? | notes |
|---|---|---|
| `PulseSlate_Starter.json` | ☐ | |
| `PulseSlate_LongForm.json` | ☐ | |
| `PulseSlate_Starter_SpectrumSage.json` | ☐ | needs Spectrum + KJNodes |

A red node here is one of three things: a missing third-party pack (expected for
the SpectrumSage graph if you have not installed both), a model value that is
not in your `get_filename_list()`, or a real bug. Model paths in the shipped
graphs use the Windows separator (`minimax\name`) deliberately — see the
CHANGELOG entry, and note it is pinned by test so a portability "fix" fails CI.

**The node-face screenshot was dropped on 2026-08-10**, at the user's call. The
README ships without one. If you ever want to add it, the constraints below still
apply and nothing else has to change.

Three constraints on any image in `docs/`, all enforced by
`tests/test_shipped_assets.py::test_the_docs_exemption_stays_narrow` — which was
skipping until this directory existed and is now live:

- **flat in `docs/`**, no subfolders;
- **under 4 MB** — it is a README illustration, not a render;
- **not named after a client asset.** `docs/` is the one place in this repo an
  image may live, and the exemption is exactly wide enough for README
  screenshots. A frame dropped there is a client asset in a repo you intend to
  publish, same as anywhere else, and the private-name scan still applies.

---

## Step 3 — The long path, and the seams

This is the headline item. The >15 s multi-window path has **never been run**;
only the ≤15 s single-window path is verified.

**Partly done, 2026-08-09.** Nine complete 2-window renders exist under
`output/pulseslate/`, every one assembled: 452 frames and 18.85 s against the
18.83 s the compiler predicted, so no frames are dropped or duplicated at the
seam. What is still outstanding from this step is **3c — listening to the
seams**, which no measurement substitutes for.

Use `PulseSlate_LongForm_18s_LIPSYNC.json` from your Downloads for the next run:
an 18 s / 2-window timeline whose references all resolve, now on the lip-sync
path, so one render exercises the seam and the voice work together.

### 3a. Dry run first, always

Set `PulseRender.dry_run = true` and queue. Nothing is sampled, decoded or
written. Read the report out of the `report` output (wire it to a `PreviewAny`;
both long-form graphs already have one). It has eight numbered sections:

```
1. WINDOWS               2. ORDINAL MAP        3. UNRESOLVED ALIASES
4. REFERENCE BUDGET      5. UPSTREAM PATCH CHAIN
6. ESTIMATES             7. PACKED SEQUENCE LENGTHS      8. WARNINGS
```

Check three of them before you spend any GPU time:

- **§3 UNRESOLVED ALIASES** must say `(none -- every @Alias resolved)`. Anything
  listed there reached the model as literal text. A wrong reference binding
  renders successfully and hands you a well-formed film of the wrong person;
  this is the line that catches it beforehand.
- **§5 UPSTREAM PATCH CHAIN** should name the patches you actually wired and
  print a `patch_fingerprint`. `(none detected)` plus a warning means your
  patches are downstream of `PulseRender` and are doing nothing — `PulseRender`
  samples with the model handed *to* it.
- **§1 WINDOWS** should show more than one row. If it shows one, the timeline is
  under the ceiling and this step is not testing what it is meant to test.

Record: windows `____`, total frames `____`, `patch_fingerprint` `____________`

### 3b. The real render

Set `dry_run = false`, `save_segments = true`, `cache_mode = auto`. Queue.

Watch the console for one line per window:

```
[PulseStudio] window 1/2: ...
[PulseStudio] N window(s) rendered, M reused -> <directory>
```

- [ ] every window rendered, no exception
- [ ] final assembly plays end to end
- [ ] duration matches the timeline (not one window's worth — that was the
      2.x fault this release exists to fix)

### 3c. Listen to every seam

This is the part that cannot be automated and the reason the step exists. At
each window boundary, listen for:

- [ ] **no hard reset** in score or room tone — the carry-over feeds the
      previous window's audio tail back through `ref_audios` precisely so each
      window does not invent its own score
- [ ] **no click, gap or overlap.** Segment placement is computed from
      frames/fps rather than from container timestamps, because streams open on
      negative DTS by different amounts and measuring extent left 84 ms of dead
      air per seam. If you hear dead air, that arithmetic has regressed
- [ ] **no visual jump** in framing or pose across the cut

Seam at `____`s: ☐ clean ☐ audible — what: `________________`
Seam at `____`s: ☐ clean ☐ audible — what: `________________`

---

## Step 4 — The five §7.5 cache behaviours

Do these **in order**, on the same graph, straight after step 3. Each one is a
requeue, not a fresh render, so the whole step costs about one extra window of
GPU time. Keep `cache_mode = auto` throughout.

The run folder is `ComfyUI/output/<run_dir>/<run_id>/`, printed by the
`segment_paths` output and at the end of the console log. Open its
`manifest.json` alongside — you are checking two things every time, what the
console says and what the manifest says.

**4.1 — Kill and resume.** Requeue, and kill ComfyUI partway through (the spec
says window 9 of 12; on a 2-window timeline kill during window 2). Restart and
requeue the identical workflow.

- [ ] the completed windows log `reused from ...`
- [ ] only the unfinished ones render
- [ ] `manifest.json` never lists a segment whose file is absent — this is the
      whole reason the manifest is fsynced *after* the media files are closed

**4.2 — Edit one shot.** Change a `PulseShot`'s `visual` text and requeue.

- [ ] only the window containing that shot re-renders
- [ ] every other window logs `reused`
- [ ] the reused segments keep their original seeds

**4.3 — Change the base seed.** Requeue.

- [ ] every window re-renders

**4.4 — Change `steps`.** Requeue.

- [ ] every window re-renders

**4.5 — Change nothing.** Requeue.

- [ ] nothing renders — the log reads `0 window(s) rendered, N reused`
- [ ] the output video is byte-identical to the previous assembly:
      `sha256sum` it before and after and compare

**And the one the tests cannot reach at all:** in 4.2, the window *after* the
edited one has its carry-over rebuilt from the cached segment's own PNG and
FLAC, because a reused window decodes nothing and produces no fresh tensors.

- [ ] that seam sounds and looks the same as it did in step 3c, when the
      carry-over came from memory

If it does not, the rebuild-from-disk path is wrong, and that is a fault no
green test would have shown you.

### Also worth trying while you are here

- `cache_mode = force_rerender` → everything renders regardless of the manifest
- `cache_mode = reuse_only` on a timeline with an uncached window → aborts with
  an error naming the first missing window, rather than silently rendering it

---

## Step 5 — PulseBench, two chains

The table is only worth having if the numbers came from this box.

Render the **same timeline twice** under two different patch chains — the
obvious pair is Spectrum+Sage versus Spectrum+Sage+Sol-Attn, once you have
step 1's node ids. Use a **different `run_dir` for each**, or the second run
reuses the first's segments and measures nothing.

> The cache keys on `patch_fingerprint`, so a chain change already invalidates
> every window. Separate folders are belt and braces, and they make the two runs
> easy to hand to `PulseBench` as two lines.

Then point `PulseBench.run_dirs` at both folders, one path per line (absolute,
or relative to `ComfyUI/output`), and queue. It groups completed segments by
`patch_fingerprint` and prints seconds-per-frame and peak VRAM.

| chain | patch_fingerprint | segments | sec/frame | peak VRAM |
|---|---|---|---|---|
| | | | | |
| | | | | |

Paste the table into `README.md` where §12.7 asks for it. Sol-Attn and Spectrum
address *different* memory, and the community disagrees about their speed — the
point of this node is to make that a lookup instead of an argument.

**`system_ram` is not a speed setting.** On a 32 GB card, Spectrum storing its
history in system RAM is what makes a 362-frame window fit at all. Set it to
`vram` and a full-length window will likely OOM rather than run slowly.

---

## Not on this list

Both are settled as of 2026-08-09: the trademark search cleared, and the
repository was renamed to `comfyui-pulse-studio` so that the repo name, the
package name and the registry entry are one word.

---

## When every box above is ticked

Update the "Pending before the tag" checklist in `CHANGELOG.md` with what you
actually observed — including anything that failed — and then stop. §17 step 7
is "stop before tagging and confirm"; the tag is yours to cut, not mine.
