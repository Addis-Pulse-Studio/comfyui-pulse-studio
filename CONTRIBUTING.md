# Contributing

Thank you for your interest in this project. Please read the following before
opening a pull request.

## Development setup

```bash
git clone https://github.com/Addis-Pulse-Studio/comfyui-addis-pulse
cd comfyui-addis-pulse
pip install -r requirements.txt
pytest
node --test tests/js
```

The full suite must pass on Python 3.10, 3.11 and 3.12. CI runs all three.

## Source provenance

This project is licensed under Apache-2.0 and runs inside ComfyUI, which is
licensed under GPL-3.0. Other packages in the same ecosystem are GPL-licensed
as well. Importing and calling those packages at runtime is expected and
supported. Copying source from them is not.

Contributions must observe the following:

- Do not copy, paste or transcribe source from any GPL-licensed project into
  this repository. This includes ComfyUI core (`comfy/`, `comfy_extras/`, the
  web frontend), ComfyUI-KJNodes, ComfyUI-Spectrum-MiniMax-H3, and any other
  GPL-licensed node pack.
- `seesee75-commits/ComfyUI-MiniMaxH3-Director` (GPL-3.0) was reviewed for
  feature ideas only and no code from it is present. Do not introduce code
  derived from it. Doing so would require this project to be relicensed under
  GPL-3.0 in its entirety.
- Facts learned by reading those projects may be used freely. Frame grids,
  pixel budgets, ordinal orderings, rounding rules and other interface
  requirements are not subject to copyright. Their expression is.
- If you have read a reference implementation closely, write the equivalent
  from the specification rather than from memory of the code, and state in the
  pull request which references you consulted.
- Code derived from a permissively licensed project must carry an attribution
  header on the derived functions naming the project, its licence and its
  copyright holder. The licence text is added to `NOTICE` in the same pull
  request.
- Do not vendor third-party source into `js/`. Do not introduce build-time
  package downloads.

Pull requests that add or substantially modify source are checked for
similarity against the projects named above before merge.

## Network policy

This package makes no network requests: not at import, not during execution,
and not from the web extension. There is no telemetry, no update check, no
model download and no external asset or webfont.

`requests`, `urllib`, `httpx` and `socket` are not permitted in package source.
The web extension must not call `fetch` or `XMLHttpRequest` against an absolute
origin. Both rules are enforced by tests in CI.

## The slot contract

LiteGraph serialises widget values and link endpoints by position. Widget
order, input order and output order are therefore a public interface from the
moment a workflow is saved.

- New widgets, inputs and outputs are appended. Existing ones are never
  inserted before, reordered, removed or retyped.
- Renaming a widget requires a `schema_version` bump and a migration entry in
  `js/ps_widget_order.js`.
- Exactly one `addDOMWidget` call exists in the web extension, and it is
  appended. `node.widgets.splice` is prohibited.
- Node class names serialise into saved workflows and are not renamed.

These rules are enforced by tests in both languages, including a cross-language
check that compares the Python `INPUT_TYPES` against the JavaScript widget
table. A change to one without the other fails CI.

## Tests

Every pull request that adds a widget, input or output updates the
cross-language parity fixture in the same commit. Bug fixes include a
regression test that fails against the previous implementation.

## Commits and pull requests

Keep renames, refactors and behaviour changes in separate commits. Describe
what changed and why in the pull request body, and note any reference material
consulted.
