# Changelog

All notable changes to this project are documented here.

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

### Pending before the tag

- [ ] **Path B verification on hardware.** A >15 s render, chained and stitched,
      with the audio seam listened to at each window boundary. Findings go here.
      Path B has never been run; only the ≤15 s single-window path is verified.
- [ ] **Trademark search** for "Pulse Studio" / "Pulse Slate" in the relevant
      jurisdiction. Availability was checked (above); clearance was not.
- [ ] `PublisherId` confirmed against the publisher created at registry.comfy.org.
      `pyproject.toml` carries `behailu-ai`, carried over from the previous
      manifest and unverified. Blocking for `comfy node publish`, not for the tag.
- [ ] **Node-face screenshot** at `docs/node_face.png`, from a real graph rather
      than a mock-up. §15 puts it at the top of the README; the slot is marked
      there in a comment. Needs the box with the weights on it.
- [ ] **Both example workflows opened on a fresh ComfyUI** with no red nodes.
      Their structure is asserted by test — ids, slots, link backfill both ways,
      the patch chain's direction — but "no red nodes" is a claim about the host
      install, and only loading them proves it.
- [ ] **CI observed green on GitHub.** The workflow's every command was run
      locally on 3.12 and passes; the matrix on 3.10 and 3.11 has not run
      anywhere yet, and there is no remote to run it on.
