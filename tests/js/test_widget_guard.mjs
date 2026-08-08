/**
 * Tests for the prompt-widget write trap.
 *
 * The reported crash was `TypeError: Cannot redefine property: value`, aborting
 * "Loading workflow data" on every reload. These reproduce that exact sequence
 * and the two adjacent failures the same code had.
 *
 *   node tests/js/test_widget_guard.mjs
 */

import assert from "node:assert/strict";
import { protectWidget } from "../../js/od_widget_guard.js";

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
  } catch (err) {
    failures.push(`${name}\n    ${err.message.split("\n")[0]}`);
  }
}

const quietConsole = { error() {}, warn() {}, debug() {} };

// ── 1. the reported crash ───────────────────────────────────────────────────

test("a workflow reload re-runs setup without throwing", () => {
  const widget = { name: "global_prompt", value: "typed by the user" };
  assert.equal(protectWidget(widget, "global_prompt", { console: quietConsole }), "installed");
  // Reload: onConfigure fires and setup runs again on the SAME widget object.
  // The first version called defineProperty unconditionally here and threw.
  assert.doesNotThrow(() =>
    protectWidget(widget, "global_prompt", { console: quietConsole }));
  assert.equal(protectWidget(widget, "global_prompt", { console: quietConsole }), "already");
  assert.equal(widget.value, "typed by the user");
});

test("repeated reloads stay stable", () => {
  const widget = { name: "shot_prompt", value: "[Shot 1] a" };
  for (let i = 0; i < 10; i++) {
    assert.doesNotThrow(() => protectWidget(widget, "shot_prompt", { console: quietConsole }));
  }
  assert.equal(widget.value, "[Shot 1] a");
});

test("a non-configurable value descriptor is declined, not fatal", () => {
  // Some frontends define widget.value themselves and seal it. defineProperty
  // then throws on the FIRST call, which is the other way the crash happened.
  const widget = { name: "global_prompt" };
  Object.defineProperty(widget, "value", {
    value: "locked", writable: true, configurable: false, enumerable: true,
  });
  let result;
  assert.doesNotThrow(() => {
    result = protectWidget(widget, "global_prompt", { console: quietConsole });
  });
  assert.equal(result, "skipped");
  assert.equal(widget.value, "locked");
  widget.value = "still writable";
  assert.equal(widget.value, "still writable");
});

test("a hostile widget never propagates an exception", () => {
  const widget = new Proxy({ name: "global_prompt", value: "x" }, {
    defineProperty() { throw new TypeError("Cannot redefine property: value"); },
  });
  let result;
  assert.doesNotThrow(() => {
    result = protectWidget(widget, "global_prompt", { console: quietConsole });
  });
  assert.equal(result, "failed");
});

// ── 2. the quieter bug: replacing instead of wrapping ───────────────────────

test("an existing accessor is wrapped, not severed", () => {
  // ComfyUI may back `value` with its own accessor onto node.widgets_values.
  // Replacing it with a closure variable would disconnect serialization and
  // silently lose the user's prompt on save.
  const backing = { global_prompt: "from the workflow file" };
  const widget = { name: "global_prompt" };
  Object.defineProperty(widget, "value", {
    get: () => backing.global_prompt,
    set: (v) => { backing.global_prompt = v; },
    configurable: true,
    enumerable: true,
  });

  protectWidget(widget, "global_prompt", { console: quietConsole });

  assert.equal(widget.value, "from the workflow file");
  widget.value = "the user typed this";
  // The write must reach the ORIGINAL backing store, not a private variable.
  assert.equal(backing.global_prompt, "the user typed this",
               "the trap severed the widget from its backing store");
  assert.equal(widget.value, "the user typed this");
});

test("a plain data property still round-trips", () => {
  const widget = { name: "shot_prompt", value: "one" };
  protectWidget(widget, "shot_prompt", { console: quietConsole });
  widget.value = "two";
  assert.equal(widget.value, "two");
});

// ── 3. the trap still does its job ──────────────────────────────────────────

test("a write during a bin document operation is blocked", () => {
  const scope = {};
  const widget = { name: "global_prompt", value: "user text" };
  const errors = [];
  protectWidget(widget, "global_prompt",
                { scope, console: { ...quietConsole, error: (m) => errors.push(m) } });

  scope.__odBinWriting = true;
  widget.value = "";           // the erasure bug's signature move
  assert.equal(widget.value, "user text", "the erasure was not blocked");
  assert.equal(errors.length, 1);
  assert.match(errors[0], /blocked a write to global_prompt/);

  scope.__odBinWriting = false;
  widget.value = "user typing normally";
  assert.equal(widget.value, "user typing normally");
});

test("typing is never blocked when the bin is idle", () => {
  const scope = {};
  const widget = { name: "shot_prompt", value: "" };
  protectWidget(widget, "shot_prompt", { scope, console: quietConsole });
  for (const line of ["[", "[S", "[Shot 1]", "[Shot 1] she walks in"]) {
    widget.value = line;
    assert.equal(widget.value, line);
  }
});

test("the marker does not leak into serialization", () => {
  const widget = { name: "global_prompt", value: "x" };
  protectWidget(widget, "global_prompt", { console: quietConsole });
  assert.ok(!Object.keys(widget).includes("__odProtected"),
            "__odProtected must be non-enumerable");
  assert.ok(!("__odProtected" in JSON.parse(JSON.stringify(widget))));
});

test("a missing widget is a no-op", () => {
  assert.equal(protectWidget(null, "global_prompt", { console: quietConsole }), "skipped");
  assert.equal(protectWidget(undefined, "shot_prompt", { console: quietConsole }), "skipped");
});

// ── report ──────────────────────────────────────────────────────────────────

if (failures.length) {
  console.error(`\n${failures.length} JS guard test(s) failed:\n`);
  for (const f of failures) console.error("  ✗ " + f);
  process.exit(1);
}
console.log(`ok - ${passed} JS widget-guard tests passed`);
