/**
 * Tests for widget-slot ordering.
 *
 * Reproduces the reported failure: five DOM header widgets spliced at indices
 * 0, 2, 4, 5, 6 shifted every native value, and ComfyUI rejected the prompt with
 *
 *     duration_seconds <- 'res_multistep'
 *     width            <- 1
 *     seed             <- 'image'
 *
 *   node tests/js/test_widget_order.mjs
 */

import assert from "node:assert/strict";
import {
  NATIVE_WIDGETS,
  checkWidgetOrder,
  describeMisalignment,
  validateWidgetValues,
} from "../../js/ps_widget_order.js";

let passed = 0;
const failures = [];
function test(name, fn) {
  try { fn(); passed += 1; }
  catch (err) { failures.push(`${name}\n    ${err.message.split("\n")[0]}`); }
}

/** A correctly-ordered save, as the shipped starter workflow produces. */
const GOOD = [
  "style: noir", "[Shot 1] she walks in",
  12.0, "16:9 landscape", 1344, 736, 20,
  "res_multistep", "simple", 1.0, 0, "fixed",
  "balanced", 15.0, "crop", "image", true, 4.0, "match", 12.0, 3.0,
  '{"assets": []}',
];

/** The array the buggy build actually wrote, from the user's file. */
const CORRUPT = [
  "", "[Shot 1] @Image1 pushes through the cafe door", "", "16:9 landscape",
  "", "", "", "res_multistep", "simple", 1, 0, "fixed", "balanced", 15,
  "crop", "image", true, 4, "match", 12, 3, '{"assets": []}',
  4, "match", 12, 3, '{"assets": []}',
];

// ── ordering invariant ──────────────────────────────────────────────────────

test("appending custom widgets after native ones is safe", () => {
  const widgets = [
    { name: "global_prompt" }, { name: "shot_prompt" }, { name: "duration_seconds" },
    { name: "od_asset_bin" },
  ];
  assert.equal(checkWidgetOrder(widgets).ok, true);
});

test("the exact splice that broke the node is rejected", () => {
  // What placeBefore() produced: headers at 0, 2, 4, 5, 6.
  const widgets = [
    { name: "od_h_global" }, { name: "global_prompt" },
    { name: "od_h_shot" }, { name: "shot_prompt" },
    { name: "od_h_assets" }, { name: "od_asset_bin" }, { name: "od_h_controls" },
    { name: "duration_seconds" }, { name: "aspect_ratio" }, { name: "width" },
  ];
  const report = checkWidgetOrder(widgets);
  assert.equal(report.ok, false);
  assert.equal(report.firstCustom, "od_h_global");
  // Every native widget after the first custom one has a shifted slot.
  assert.deepEqual(report.offenders, [
    "global_prompt", "shot_prompt", "duration_seconds", "aspect_ratio", "width",
  ]);
});

test("one custom widget in the middle is still a failure", () => {
  const widgets = [
    { name: "global_prompt" }, { name: "od_asset_bin" }, { name: "shot_prompt" },
  ];
  assert.equal(checkWidgetOrder(widgets).ok, false);
});

test("an empty or absent widget list is fine", () => {
  assert.equal(checkWidgetOrder([]).ok, true);
  assert.equal(checkWidgetOrder(undefined).ok, true);
});

// ── value validation ────────────────────────────────────────────────────────

test("a correctly ordered save validates clean", () => {
  const report = validateWidgetValues(GOOD);
  assert.equal(report.ok, true, JSON.stringify(report.mismatches));
});

test("the user's corrupt file is detected as misaligned", () => {
  // Read against the NATIVE spec -- which is how the fixed build will read it,
  // since DOM widgets are now appended. Four numeric widgets receive empty
  // strings, which is unambiguous evidence the file was written by the buggy
  // build. That is enough to refuse it; the detector does not need to reproduce
  // every one of ComfyUI's thirteen downstream type errors.
  const report = validateWidgetValues(CORRUPT);
  assert.equal(report.ok, false);
  const byName = Object.fromEntries(report.mismatches.map((m) => [m.name, m]));
  for (const name of ["duration_seconds", "width", "height", "steps"]) {
    assert.ok(byName[name], `${name} was not detected`);
    assert.equal(byName[name].expected, "number");
    assert.equal(byName[name].value, "");
  }
});

test("the corrupt file is still rejected after the ordering fix", () => {
  // The fix stops NEW saves from being corrupted; it cannot repair one already
  // written, because the text that landed in a header slot was discarded at save
  // time. Loading it must fail loudly rather than render something wrong.
  const report = validateWidgetValues(CORRUPT);
  assert.equal(report.ok, false);
  assert.ok(report.mismatches.length >= 4);
});

test("the buggy layout maps values exactly as ComfyUI reported", () => {
  // Documents the root cause: the file was saved from a 27-widget list, so
  // index i belongs to node.widgets[i], not to native[i].
  const built = [
    "od_h_global", "global_prompt", "od_h_shot", "shot_prompt", "od_h_assets",
    "od_asset_bin", "od_h_controls", "duration_seconds", "aspect_ratio", "width",
    "height", "steps", "sampler_name", "scheduler", "cfg", "seed",
  ];
  const got = {};
  built.forEach((name, i) => { if (!name.startsWith("od_")) got[name] = CORRUPT[i]; });
  assert.equal(got.duration_seconds, "res_multistep");
  assert.equal(got.aspect_ratio, "simple");
  assert.equal(got.width, 1);
  assert.equal(got.height, 0);
  assert.equal(got.steps, "fixed");
  assert.equal(got.sampler_name, "balanced");
  assert.equal(got.scheduler, 15);
  assert.equal(got.cfg, "crop");
  assert.equal(got.seed, "image");
});

test("a shorter array from an older save is not flagged", () => {
  // Trailing widgets simply absent: valid, not misaligned.
  assert.equal(validateWidgetValues(GOOD.slice(0, 10)).ok, true);
});

test("nulls are treated as absent, not as a shift", () => {
  const values = GOOD.slice();
  values[4] = null;
  assert.equal(validateWidgetValues(values).ok, true);
});

test("a non-array is reported as unreadable rather than throwing", () => {
  const report = validateWidgetValues("not an array");
  assert.equal(report.ok, false);
  assert.equal(report.unreadable, true);
});

test("the explanation names the file to reload", () => {
  const text = describeMisalignment(validateWidgetValues(CORRUPT));
  assert.match(text, /PulseSlate_Starter\.json/);
  assert.match(text, /duration_seconds/);
});

test("the spec matches the node's declared widget count", () => {
  // 21 required widgets + control_after_generate.
  assert.equal(NATIVE_WIDGETS.length, 22);
  assert.equal(NATIVE_WIDGETS[0].name, "global_prompt");
  assert.equal(NATIVE_WIDGETS[NATIVE_WIDGETS.length - 1].name, "timeline_data");
});

if (failures.length) {
  console.error(`\n${failures.length} JS widget-order test(s) failed:\n`);
  for (const f of failures) console.error("  ✗ " + f);
  process.exit(1);
}
console.log(`ok - ${passed} JS widget-order tests passed`);
