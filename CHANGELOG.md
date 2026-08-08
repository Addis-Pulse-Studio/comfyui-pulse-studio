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

### Fixed

- Ten client-facing asset filenames were referenced by the shipped example
  workflows, four of them buried inside the `timeline_data` string. All are now
  `example_*` placeholders that the user supplies.
- Windows-only `minimax\` model paths in the example workflows use forward
  slashes, so the graphs load on any platform.

### Pending before the tag

- [ ] **Path B verification on hardware.** A >15 s render, chained and stitched,
      with the audio seam listened to at each window boundary. Findings go here.
      Path B has never been run; only the ≤15 s single-window path is verified.
- [ ] **Name availability check.** GitHub, PyPI and ComfyUI registry namespaces
      plus a plain trademark search for "Pulse Studio" / "PulseSlate". Record the
      date here. This blocks the tag.
- [ ] `PublisherId` confirmed against the publisher created at registry.comfy.org.
