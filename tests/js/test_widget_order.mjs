/**
 * Tests for widget-slot ordering and the schema-2 migration.
 *
 * Reproduces the reported failure: five DOM header widgets spliced at indices
 * 0, 2, 4, 5, 6 shifted every native value, and ComfyUI rejected the prompt with
 *
 *     duration_seconds <- 'res_multistep'
 *     width            <- 1
 *     seed             <- 'image'
 *
 * and then covers the permanent fix: schema_version at slot 0, name-based
 * loading, and append-only growth. Spec §3.
 *
 *   node tests/js/test_widget_order.mjs
 */

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  CURRENT,
  NATIVE_WIDGETS,
  NODE_IDS,
  WIDGET_NAMES,
  applySavedValues,
  checkWidgetOrder,
  describeMisalignment,
  describeUnloadable,
  isKnownVersion,
  readSavedValues,
  validateWidgetValues,
  widgetNames,
} from "../../js/ps_widget_order.js";

let passed = 0;
const failures = [];
function test(name, fn) {
  try { fn(); passed += 1; }
  catch (err) { failures.push(`${name}\n    ${err.message.split("\n")[0]}`); }
}

/** A correctly-ordered schema 2.0.0 save, as the shipped starter produces. */
const GOOD = [
  "2.0.0", '{"schema": 2, "assets": [], "cast": []}',
  "style: noir", "[Shot 1] she walks in",
  12.0, "16:9 landscape", 1344, 736, 20,
  "res_multistep", "simple", 1.0, 0, "fixed",
  "balanced", 15.0, "crop", "image", true, 4.0, "match", 12.0, 3.0,
];

/**
 * The array the buggy pre-2.0.0 build actually wrote, from the user's file.
 * Kept verbatim as a regression case (§3.4): it must never load, and it must
 * never be interpreted positionally against the current layout.
 */
const CORRUPT = [
  "", "[Shot 1] @Image1 pushes through the cafe door", "", "16:9 landscape",
  "", "", "", "res_multistep", "simple", 1, 0, "fixed", "balanced", 15,
  "crop", "image", true, 4, "match", 12, 3, '{"assets": []}',
  4, "match", 12, 3, '{"assets": []}',
];

/** A minimal fake LiteGraph node built from a name list. */
function fakeNode(names, values = {}, defaults = {}) {
  return {
    widgets: names.map((name) => ({
      name,
      value: values[name],
      options: name in defaults ? { default: defaults[name] } : {},
    })),
  };
}

// ── ordering invariant ──────────────────────────────────────────────────────

test("appending custom widgets after native ones is safe", () => {
  const widgets = [
    { name: "global_prompt" }, { name: "shot_prompt" }, { name: "duration_seconds" },
    { name: "ps_asset_bin" },
  ];
  assert.equal(checkWidgetOrder(widgets).ok, true);
});

test("the exact splice that broke the node is rejected", () => {
  // What placeBefore() produced: headers at 0, 2, 4, 5, 6.
  const widgets = [
    { name: "ps_h_global" }, { name: "global_prompt" },
    { name: "ps_h_shot" }, { name: "shot_prompt" },
    { name: "ps_h_assets" }, { name: "ps_asset_bin" }, { name: "ps_h_controls" },
    { name: "duration_seconds" }, { name: "aspect_ratio" }, { name: "width" },
  ];
  const report = checkWidgetOrder(widgets);
  assert.equal(report.ok, false);
  assert.equal(report.firstCustom, "ps_h_global");
  // Every native widget after the first custom one has a shifted slot.
  assert.deepEqual(report.offenders, [
    "global_prompt", "shot_prompt", "duration_seconds", "aspect_ratio", "width",
  ]);
});

test("one custom widget in the middle is still a failure", () => {
  const widgets = [
    { name: "global_prompt" }, { name: "ps_asset_bin" }, { name: "shot_prompt" },
  ];
  assert.equal(checkWidgetOrder(widgets).ok, false);
});

test("an empty or absent widget list is fine", () => {
  assert.equal(checkWidgetOrder([]).ok, true);
  assert.equal(checkWidgetOrder(undefined).ok, true);
});

// ── the live node must match the name table ─────────────────────────────────

test("live widget names matching the table pass the name check", () => {
  const widgets = widgetNames("PulseSlate").map((name) => ({ name }));
  widgets.push({ name: "ps_asset_bin" });
  const report = checkWidgetOrder(widgets, undefined, "PulseSlate");
  assert.equal(report.nameError, null);
  assert.equal(report.ok, true);
});

test("a widget added to the node but not the table is caught", () => {
  // The exact drift the cross-language test exists to prevent, seen from JS.
  const widgets = widgetNames("PulseSlate").map((name) => ({ name }));
  widgets.push({ name: "locale" });          // appended in Python, not here
  const report = checkWidgetOrder(widgets, undefined, "PulseSlate");
  assert.ok(report.nameError, "drift was not detected");
  assert.match(report.nameError, /drifted/);
  assert.equal(report.ok, false);
});

test("a reordered live node is caught even with the same names", () => {
  const names = widgetNames("PulseSlate").slice();
  [names[4], names[5]] = [names[5], names[4]];
  const report = checkWidgetOrder(names.map((name) => ({ name })), undefined, "PulseSlate");
  assert.ok(report.nameError, "a reorder was not detected");
});

test("a frontend without control_after_generate is not a failure", () => {
  // Not ours to control: some builds render the seed control differently.
  const names = widgetNames("PulseSlate").filter((n) => n !== "control_after_generate");
  const report = checkWidgetOrder(names.map((name) => ({ name })), undefined, "PulseSlate");
  assert.equal(report.nameError, null);
});

// ── value validation ────────────────────────────────────────────────────────

test("a correctly ordered save validates clean", () => {
  const report = validateWidgetValues(GOOD);
  assert.equal(report.ok, true, JSON.stringify(report.mismatches));
});

test("the user's corrupt file is detected as misaligned", () => {
  // Read against the NATIVE spec -- which is how the fixed build will read it,
  // since DOM widgets are now appended. Numeric widgets receive empty strings,
  // which is unambiguous evidence the file was written by the buggy build.
  const report = validateWidgetValues(CORRUPT);
  assert.equal(report.ok, false);
  assert.ok(report.mismatches.length >= 4);
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

// Both recovery messages named PulseSlate_Starter.json, which was dropped in
// 3.0.0 and does not ship -- so the advice sent users to a file that is not
// there. Asserting the name against the directory rather than against a
// literal means the next graph that goes fails here instead of in the wild.
const WORKFLOW_DIR = resolve(dirname(fileURLToPath(import.meta.url)),
                             "..", "..", "example_workflows");

function assertNamesAShippedGraph(text) {
  const named = [...text.matchAll(/example_workflows\/([\w.-]+\.json)/g)]
    .map((m) => m[1]);
  assert.ok(named.length, `no example_workflows/*.json named in:\n${text}`);
  for (const name of named) {
    assert.ok(existsSync(resolve(WORKFLOW_DIR, name)),
              `recovery advice names ${name}, which does not ship`);
  }
}

test("the explanation names a file that actually ships", () => {
  assertNamesAShippedGraph(describeMisalignment(validateWidgetValues(CORRUPT)));
});

// ── the frozen prefix ───────────────────────────────────────────────────────

test("schema_version is widget 0 and timeline_data is widget 1", () => {
  assert.equal(NATIVE_WIDGETS[0].name, "schema_version");
  assert.equal(NATIVE_WIDGETS[1].name, "timeline_data");
});

test("every node in the pack opens with schema_version", () => {
  for (const nodeId of NODE_IDS) {
    assert.equal(widgetNames(nodeId)[0], "schema_version", `${nodeId} slot 0`);
  }
});

test("the spec matches the node's declared widget count", () => {
  // 24 required widgets + control_after_generate. 3.0.0 appended `continuity`.
  assert.equal(NATIVE_WIDGETS.length, 25);
  assert.equal(NATIVE_WIDGETS[NATIVE_WIDGETS.length - 1].name, "continuity");
});

test("a file saved before audio_ref_ceiling existed still loads", () => {
  // The append case, tested directly rather than reasoned about: GOOD is a
  // 2.0.0 array written before this widget was added, so it is one short. Every
  // value it does carry must land on the widget it was written for, and the
  // widget it lacks must simply be absent from the map so the node's own default
  // stands. If appending ever shifts a value, this is where it shows.
  const read = readSavedValues("PulseSlate", GOOD);
  assert.equal(read.ok, true);
  assert.equal(read.values.has("audio_ref_ceiling"), false);
  assert.equal(read.values.get("shift_audio"), GOOD.at(-1));
  assert.equal(read.values.get("shift_video"), 12.0);
  assert.equal(read.values.get("ref_image_size"), "match");
});

// ── reading a saved file ────────────────────────────────────────────────────

test("a known schema version reads into a name -> value map", () => {
  const read = readSavedValues("PulseSlate", GOOD);
  assert.equal(read.ok, true);
  assert.equal(read.version, "2.0.0");
  assert.equal(read.values.get("duration_seconds"), 12.0);
  assert.equal(read.values.get("sampler_name"), "res_multistep");
  assert.equal(read.values.get("carry_audio"), true);
  assert.equal(read.values.get("shift_audio"), 3.0);
});

test("the corrupt pre-2.0.0 file is refused, not guessed at", () => {
  const read = readSavedValues("PulseSlate", CORRUPT);
  assert.equal(read.ok, false);
  assert.match(read.reason, /schema version/);
});

test("an unknown schema version is refused by name", () => {
  const future = GOOD.slice();
  future[0] = "9.9.9";
  const read = readSavedValues("PulseSlate", future);
  assert.equal(read.ok, false);
  assert.match(read.reason, /9\.9\.9/);
});

test("isKnownVersion answers for every node", () => {
  for (const nodeId of NODE_IDS) {
    assert.equal(isKnownVersion(nodeId, CURRENT), true);
    assert.equal(isKnownVersion(nodeId, "1.0.0"), false);
  }
});

test("the unloadable message names the node and a recovery path that ships", () => {
  const text = describeUnloadable("PulseSlate", "because reasons");
  assert.match(text, /PulseSlate/);
  assert.match(text, /CHANGELOG/);
  assertNamesAShippedGraph(text);
});

// ── restoring by name ───────────────────────────────────────────────────────

test("values land on the right widgets by name, not by position", () => {
  const node = fakeNode(widgetNames("PulseSlate"));
  const result = applySavedValues(node, "PulseSlate", GOOD, () => {});
  assert.equal(result.ok, true);
  const byName = Object.fromEntries(node.widgets.map((w) => [w.name, w.value]));
  assert.equal(byName.duration_seconds, 12.0);
  assert.equal(byName.aspect_ratio, "16:9 landscape");
  assert.equal(byName.width, 1344);
  assert.equal(byName.seed, 0);
  assert.equal(byName.cfg, 1.0);
  assert.equal(byName.shift_audio, 3.0);
  assert.equal(byName.schema_version, "2.0.0");
});

test("a widget appended since the file was saved takes its default", () => {
  // THE POINT OF THE WHOLE MECHANISM. A 1.1 build has `locale`; the 1.0 file
  // does not. Positional loading would have shifted; name loading defaults it.
  const node = fakeNode([...widgetNames("PulseSlate"), "locale"], {}, { locale: "en" });
  const result = applySavedValues(node, "PulseSlate", GOOD, () => {});
  assert.equal(result.ok, true);
  const byName = Object.fromEntries(node.widgets.map((w) => [w.name, w.value]));
  assert.equal(byName.locale, "en");
  assert.equal(byName.shift_audio, 3.0, "the appended widget shifted a real value");
  assert.deepEqual(result.defaulted, ["locale"]);
});

test("a widget the file has but this build dropped is reported, not applied", () => {
  const names = widgetNames("PulseSlate").filter((n) => n !== "ref_image_size");
  const node = fakeNode(names);
  const warnings = [];
  const result = applySavedValues(node, "PulseSlate", GOOD, (m) => warnings.push(m));
  assert.equal(result.ok, true);
  assert.deepEqual(result.dropped, ["ref_image_size"]);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /ref_image_size/);
});

test("the appended DOM widget is never assigned a value", () => {
  // Spec §3.3.4: trailing slots are the DOM widget's dead slot.
  const node = fakeNode([...widgetNames("PulseSlate"), "ps_asset_bin"]);
  applySavedValues(node, "PulseSlate", GOOD, () => {});
  const bin = node.widgets.find((w) => w.name === "ps_asset_bin");
  assert.equal(bin.value, undefined, "the DOM widget received a native value");
});

test("trailing junk beyond the native count is ignored", () => {
  const node = fakeNode(widgetNames("PulseSlate"));
  const withJunk = [...GOOD, "leftover", 999];
  const result = applySavedValues(node, "PulseSlate", withJunk, () => {});
  assert.equal(result.ok, true);
  const byName = Object.fromEntries(node.widgets.map((w) => [w.name, w.value]));
  assert.equal(byName.shift_audio, 3.0);
});

test("a refused file assigns nothing at all", () => {
  const node = fakeNode(widgetNames("PulseSlate"));
  const before = node.widgets.map((w) => w.value);
  const result = applySavedValues(node, "PulseSlate", CORRUPT, () => {});
  assert.equal(result.ok, false);
  assert.deepEqual(node.widgets.map((w) => w.value), before,
                   "an unloadable file still wrote values into the node");
});

test("the other two nodes restore by name too", () => {
  for (const nodeId of ["PulseRetake", "PulseStill"]) {
    const names = widgetNames(nodeId);
    const saved = names.map((n) => (n === "schema_version" ? CURRENT : `v_${n}`));
    const node = fakeNode(names);
    const result = applySavedValues(node, nodeId, saved, () => {});
    assert.equal(result.ok, true, nodeId);
    const byName = Object.fromEntries(node.widgets.map((w) => [w.name, w.value]));
    assert.equal(byName.seed, "v_seed", `${nodeId} seed`);
    assert.equal(byName.shift_audio, "v_shift_audio", `${nodeId} shift_audio`);
  }
});

// ── save -> reload round trip ───────────────────────────────────────────────

/** What LiteGraph writes: widgets_values[i] = node.widgets[i].value. */
function serialize(node) {
  return node.widgets.map((w) => w.value);
}

test("a populated bin survives save -> reload unchanged", () => {
  const BIN = JSON.stringify({
    schema: 2,
    assets: [{ id: "a1", kind: "image", name: "Mimi", file: "example_character_a.png" }],
    cast: [],
  });
  const names = widgetNames("PulseSlate");
  const values = Object.fromEntries(names.map((n, i) => [n, GOOD[i]]));
  values.timeline_data = BIN;

  const node = fakeNode(names, values);
  const saved = serialize(node);
  assert.equal(saved.length, names.length);

  const reloaded = fakeNode(names);
  const result = applySavedValues(reloaded, "PulseSlate", saved, () => {});
  assert.equal(result.ok, true);

  const byName = Object.fromEntries(reloaded.widgets.map((w) => [w.name, w.value]));
  assert.equal(byName.timeline_data, BIN, "the bin did not survive the round trip");
  assert.deepEqual(JSON.parse(byName.timeline_data).assets.length, 1);
  assert.equal(byName.global_prompt, "style: noir");
  assert.equal(byName.duration_seconds, 12.0);
});

test("the widgets_values length is stable across repeated round trips", () => {
  // The DOM widget occupies a trailing slot on some frontend builds. Whether it
  // does or not, the length must not creep on every save -- a growing array is
  // how a trailing slot eventually becomes a leading one for a future widget.
  const names = widgetNames("PulseSlate");
  let node = fakeNode(names, Object.fromEntries(names.map((n, i) => [n, GOOD[i]])));
  node.widgets.push({ name: "ps_asset_bin", value: undefined, options: {} });

  const first = serialize(node);
  for (let i = 0; i < 3; i++) {
    const next = fakeNode(names);
    next.widgets.push({ name: "ps_asset_bin", value: undefined, options: {} });
    applySavedValues(next, "PulseSlate", serialize(node), () => {});
    node = next;
  }
  assert.equal(serialize(node).length, first.length,
               "widgets_values grew across save/reload cycles");
});

test("a round trip through a build with an extra widget stays aligned", () => {
  // 1.0 saves; 1.1 (with `locale` appended) loads; 1.1 saves; 1.0 loads again.
  const v10 = widgetNames("PulseSlate");
  const v10Node = fakeNode(v10, Object.fromEntries(v10.map((n, i) => [n, GOOD[i]])));

  const v11Node = fakeNode([...v10, "locale"], {}, { locale: "en" });
  applySavedValues(v11Node, "PulseSlate", serialize(v10Node), () => {});
  assert.equal(v11Node.widgets.find((w) => w.name === "locale").value, "en");

  // 1.1's save has one more slot; 1.0 reads the names it knows and ignores it.
  const back = fakeNode(v10);
  const result = applySavedValues(back, "PulseSlate", serialize(v11Node), () => {});
  assert.equal(result.ok, true);
  const byName = Object.fromEntries(back.widgets.map((w) => [w.name, w.value]));
  assert.equal(byName.shift_audio, 3.0, "the extra 1.1 widget shifted a 1.0 value");
  assert.equal(byName.carry_audio, true);
});

// ── the table itself ────────────────────────────────────────────────────────

test("WIDGET_NAMES is keyed by node then version, as the loader expects", () => {
  for (const nodeId of NODE_IDS) {
    assert.ok(WIDGET_NAMES[nodeId], `${nodeId} missing from WIDGET_NAMES`);
    assert.ok(Array.isArray(WIDGET_NAMES[nodeId][CURRENT]), `${nodeId} missing ${CURRENT}`);
  }
});

test("no node declares a duplicate widget name", () => {
  for (const nodeId of NODE_IDS) {
    const names = widgetNames(nodeId);
    assert.equal(new Set(names).size, names.length, `${nodeId} has a duplicate name`);
  }
});

test("an unknown node id yields null rather than throwing", () => {
  assert.equal(widgetNames("NotANode"), null);
  assert.equal(readSavedValues("NotANode", GOOD).ok, false);
});

if (failures.length) {
  console.error(`\n${failures.length} JS widget-order test(s) failed:\n`);
  for (const f of failures) console.error("  ✗ " + f);
  process.exit(1);
}
console.log(`ok - ${passed} JS widget-order tests passed`);
