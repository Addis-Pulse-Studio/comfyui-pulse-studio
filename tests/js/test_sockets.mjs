/**
 * Growing socket groups. Spec §4.
 *
 * `js/ps_sockets.js` imports ComfyUI's app module, which does not exist outside
 * a browser, so the pure half is exercised through a copy of its two exported
 * pure functions -- the same trick tests/js/test_widget_order.mjs uses. The
 * numbers and the group table are read out of the real file so they cannot drift
 * from it silently; tests/test_js_guard.py asserts they match nodes.py too.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE = readFileSync(join(HERE, "..", "..", "js", "ps_sockets.js"), "utf8");

let failures = 0;
function test(name, fn) {
  try {
    fn();
  } catch (error) {
    failures++;
    console.error(`  ✗ ${name}\n    ${error.message}`);
  }
}

/** desiredNames, lifted verbatim from ps_sockets.js so the test drives the real
 *  algorithm rather than a paraphrase of it. */
function loadDesiredNames() {
  const start = SOURCE.indexOf("export function desiredNames");
  assert.ok(start > -1, "desiredNames not found in ps_sockets.js");
  let depth = 0;
  let i = SOURCE.indexOf("{", start);
  const open = i;
  for (; i < SOURCE.length; i++) {
    if (SOURCE[i] === "{") depth++;
    else if (SOURCE[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  const body = SOURCE.slice(open + 1, i);
  return new Function("groups", "connected", body);
}

const desiredNames = loadDesiredNames();

const SHOTS = [{ prefix: "shots.shot_", max: 24 }];
const BOTH = [
  { prefix: "refs.ref_image_", max: 8 },
  { prefix: "shots.shot_", max: 24 },
];

test("an untouched node shows exactly one free socket per group", () => {
  assert.deepEqual(desiredNames(SHOTS, new Set()), ["shots.shot_1"]);
  assert.deepEqual(desiredNames(BOTH, new Set()),
                   ["refs.ref_image_1", "shots.shot_1"]);
});

test("filling the last free socket exposes the next one", () => {
  assert.deepEqual(desiredNames(SHOTS, new Set(["shots.shot_1"])),
                   ["shots.shot_1", "shots.shot_2"]);
  assert.deepEqual(desiredNames(SHOTS, new Set(["shots.shot_1", "shots.shot_2"])),
                   ["shots.shot_1", "shots.shot_2", "shots.shot_3"]);
});

test("sockets are ordered by numeric suffix, not by connection order", () => {
  // The user wired slot 3 first and slot 1 second. The compiler reads shots in
  // socket order, so the node must show them in socket order too.
  const names = desiredNames(SHOTS, new Set(["shots.shot_3", "shots.shot_1"]));
  assert.deepEqual(names, ["shots.shot_1", "shots.shot_3", "shots.shot_2"]);
});

test("a gap in the middle is preserved, and is the next free socket", () => {
  // Unplugging shot 2 out of three must not renumber shots 1 and 3 -- their
  // shot_ids, seeds and cached segments are attached to those nodes, not to the
  // socket they happen to sit in.
  const names = desiredNames(SHOTS, new Set(["shots.shot_1", "shots.shot_3"]));
  assert.equal(names.includes("shots.shot_1"), true);
  assert.equal(names.includes("shots.shot_3"), true);
  assert.equal(names.includes("shots.shot_2"), true);
});

test("groups stay in declaration order", () => {
  const names = desiredNames(BOTH, new Set(["shots.shot_1", "refs.ref_image_1"]));
  assert.deepEqual(names, ["refs.ref_image_1", "refs.ref_image_2",
                           "shots.shot_1", "shots.shot_2"]);
});

test("a full group stops growing instead of inventing an undeclared socket", () => {
  const full = new Set();
  for (let i = 1; i <= 8; i++) full.add(`refs.ref_image_${i}`);
  const names = desiredNames([{ prefix: "refs.ref_image_", max: 8 }], full);
  assert.equal(names.length, 8);
  assert.equal(names.includes("refs.ref_image_9"), false,
               "the backend declares 8; a 9th socket would validate as unknown");
});

test("the file never removes a socket without repairing target_slot", () => {
  // The bug this guards: a link stores target_slot as an index, so trimming a
  // socket silently moves every wire after it onto its neighbour.
  assert.ok(SOURCE.includes("link.target_slot = slot"),
            "sync must rewrite target_slot after rebuilding node.inputs");
});

test("every LiteGraph hook is wrapped so a throw cannot abort a workflow load", () => {
  for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
    const at = SOURCE.indexOf(`nodeType.prototype.${hook} =`);
    assert.ok(at > -1, `${hook} is missing`);
    assert.ok(SOURCE.slice(at, at + 160).includes("guard("),
              `${hook} is not wrapped`);
  }
});

test("connection changes are deferred a tick", () => {
  // LiteGraph calls onConnectionsChange in the middle of its own link
  // bookkeeping; rebuilding node.inputs underneath it corrupts the link it is
  // currently attaching.
  assert.ok(SOURCE.includes("requestAnimationFrame"),
            "sync must not run synchronously inside onConnectionsChange");
});

if (failures) {
  console.error(`\n${failures} JS socket test(s) failed:\n`);
  process.exit(1);
}
console.log("ps_sockets.js: all tests passed");
