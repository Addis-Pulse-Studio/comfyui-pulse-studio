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

// ── sync() against frontend 1.53's graph, end to end ─────────────────────────
// The real module, with ComfyUI's app import stubbed out. Frontend 1.53 keeps
// graph.links as a Map and gives every widget an input slot of its own; a saved
// wire is resolved by the slot number on its link. A sync that could not find
// the links in that Map left every shot wire pointing at its old slot number,
// which by then held global_prompt, shot_prompt, duration_seconds, aspect_ratio
// and width -- the exact validation error a 6-shot PulseSlate graph produced.
globalThis.LiteGraph = { HollowCircle: 7 };
const moduleSource = SOURCE
  .replace(/import \{ app \} from "[^"]+";/, "const app = { registerExtension() {} };");
const sockets = await import("data:text/javascript;base64," +
                             Buffer.from(moduleSource).toString("base64"));

const FIXED = ["model", "clip", "vae", "audio_vae", "model_fl2va", "ref_video",
               "ref_video_audio", "ref_music"];
const WIDGETS = ["global_prompt", "shot_prompt", "duration_seconds", "aspect_ratio", "width"];

function slateNode(inputs, links) {
  return { comfyClass: "PulseSlate", id: 20, inputs, graph: { links } };
}

test("linkById reads a Map (frontend 1.53) and a plain object (older builds)", () => {
  const link = { id: 5 };
  assert.equal(sockets.linkById(new Map([[5, link]]), 5), link);
  assert.equal(sockets.linkById({ 5: link }, 5), link);
  assert.equal(sockets.linkById(new Map(), 5), undefined);
});

test("six shot wires stay on the shot sockets after a sync on a 1.53 graph", () => {
  // The creation-time layout frontend 1.53 leaves behind: widget slots sit ahead
  // of the growing groups, and every wire still carries its saved slot number.
  const links = new Map();
  const inputs = FIXED.map((name) => ({ name, type: "*", link: null }));
  for (const name of WIDGETS) inputs.push({ name, type: "STRING", link: null, widget: { name } });
  inputs.push({ name: "refs.ref_image_1", type: "IMAGE", link: null });
  for (let k = 1; k <= 6; k++) {
    const id = 32 + k;
    inputs.push({ name: `shots.shot_${k}`, type: "PULSE_SHOT", link: id });
    links.set(id, { id, type: "PULSE_SHOT", target_id: 20, target_slot: 8 + k });
  }
  inputs.push({ name: "voices.voice_1", type: "PULSE_VOICE", link: null });
  const node = slateNode(inputs, links);

  sockets.sync(node);

  for (const [id, link] of links) {
    const target = node.inputs[link.target_slot];
    assert.equal(target.link, id, `link ${id} points at slot ${link.target_slot}`);
    assert.ok(target.name.startsWith("shots.shot_"),
              `link ${id} landed on ${target.name}, not a shot socket`);
  }
  // The shot sockets come before the widget inputs, where the saved graph has them.
  assert.deepEqual(node.inputs.slice(8, 16).map((i) => i.name),
                   ["refs.ref_image_1", "shots.shot_1", "shots.shot_2", "shots.shot_3",
                    "shots.shot_4", "shots.shot_5", "shots.shot_6", "shots.shot_7"]);
  for (const name of WIDGETS) {
    const widget = node.inputs.find((i) => i.name === name);
    assert.equal(widget.link, null, `${name} must stay a widget, not a socket with a wire`);
  }
});

test("a saved layout laid over the default one keeps one of each socket", () => {
  // LiteGraph.configure copies the saved inputs over the creation-time array
  // index by index, so a shorter saved array leaves the default tail behind.
  const links = new Map([[40, { id: 40, target_id: 20, target_slot: 9 }]]);
  const inputs = [
    ...FIXED.map((name) => ({ name, type: "*", link: null })),
    { name: "refs.ref_image_1", type: "IMAGE", link: null },
    { name: "shots.shot_1", type: "PULSE_SHOT", link: 40 },
    { name: "shots.shot_2", type: "PULSE_SHOT", link: null },
    { name: "global_prompt", type: "STRING", link: null, widget: { name: "global_prompt" } },
    { name: "shots.shot_1", type: "PULSE_SHOT", link: null },
    { name: "global_prompt", type: "STRING", link: null, widget: { name: "global_prompt" } },
  ];
  const node = slateNode(inputs, links);
  sockets.sync(node);
  const names = node.inputs.map((i) => i.name);
  assert.equal(names.filter((n) => n === "shots.shot_1").length, 1);
  assert.equal(names.filter((n) => n === "global_prompt").length, 1);
  assert.equal(node.inputs[links.get(40).target_slot].name, "shots.shot_1");
});

// ── sync() against frontend 1.57's link store ────────────────────────────────
// Frontend 1.57 derives a socket's link from a store keyed by (node, slot
// number): `input.link` is a read-only view, assigning null to it disconnects
// whatever wire sits at that slot, and a link's target_slot setter refuses a
// slot another wire occupies. The fake below models exactly those three rules,
// which the live frontend was checked against (tests in a real 1.57.1 page:
// every shipped graph, and a slate wired by hand, then saved and reopened).

class StoreSlot {
  constructor(node, fields) {
    this.node = node;
    Object.assign(this, fields);
  }
  get link() {
    const slot = this.node.inputs.indexOf(this);
    for (const link of this.node.graph.links.values()) {
      if (link.target_id === this.node.id && link.target_slot === slot) return link.id;
    }
    return null;
  }
  set link(value) {
    if (value !== null) return;  // 1.57 ignores it
    throw new Error(`disconnected ${this.name}: sync must never assign input.link on 1.57`);
  }
}

class StoreLink {
  constructor(graph, id, targetId, targetSlot) {
    this.graph = graph;
    this.id = id;
    this.target_id = targetId;
    this._slot = targetSlot;
  }
  get target_slot() {
    return this._slot;
  }
  set target_slot(slot) {
    for (const other of this.graph.links.values()) {
      if (other !== this && other.target_id === this.target_id && other._slot === slot) {
        throw new Error(`link ${this.id} refused: slot ${slot} is occupied by link ${other.id}`);
      }
    }
    this._slot = slot;
  }
}

function storeSlate(names, wires) {
  /** A 1.57 slate: sockets in `names` order, wires as [linkId, slot]. */
  const graph = { links: new Map() };
  const node = { comfyClass: "PulseSlate", id: 20, graph, inputs: [] };
  node.getInputLink = (slot) => {
    const id = node.inputs[slot]?.link;
    return id == null ? undefined : graph.links.get(id);
  };
  node.inputs = names.map((name) => new StoreSlot(node, {
    name, type: "*", ...(WIDGETS.includes(name) ? { widget: { name } } : {}) }));
  for (const [id, slot] of wires) graph.links.set(id, new StoreLink(graph, id, 20, slot));
  return node;
}

const CREATED = [...FIXED, "refs.ref_image_1", "shots.shot_1", "voices.voice_1", "cast.cast_1",
                 ...WIDGETS];

test("1.57: a load puts every saved wire back on its named socket", () => {
  // Saved: four shots and two cast members. 1.57 hangs each saved wire on the
  // creation-time layout by its saved slot number -- shot_2's wire on voice_1,
  // the cast wires past the shot group -- before onConfigure runs.
  const saved = [...FIXED, "refs.ref_image_1", "shots.shot_1", "shots.shot_2", "shots.shot_3",
                 "shots.shot_4", "shots.shot_5", "voices.voice_1", "cast.cast_1", "cast.cast_2",
                 "cast.cast_3"];
  const owner = { 16: "shots.shot_1", 17: "shots.shot_2", 18: "shots.shot_3",
                  19: "shots.shot_4", 20: "cast.cast_1", 21: "cast.cast_2" };
  const info = { inputs: saved.map((name) => ({
    name, link: Number(Object.keys(owner).find((id) => owner[id] === name) ?? NaN) || null })) };
  const wires = Object.keys(owner).map((id) => [Number(id), saved.indexOf(owner[id])]);
  const node = storeSlate(CREATED, wires);

  sockets.sync(node, sockets.savedLinks(info));

  for (const [id, name] of Object.entries(owner)) {
    const link = node.graph.links.get(Number(id));
    assert.equal(node.inputs[link.target_slot]?.name, name, `link ${id} is on the wrong socket`);
  }
  assert.equal(node.graph.links.size, 6, "no wire may be lost on the way");
});

test("1.57: a socket inserted ahead of live wires does not move them", () => {
  // Cast and voice wired first; then the shot group grows a socket ahead of
  // them, which shifts their slot numbers by one.
  const node = storeSlate(CREATED, [[5, CREATED.indexOf("cast.cast_1")],
                                    [6, CREATED.indexOf("voices.voice_1")],
                                    [7, CREATED.indexOf("shots.shot_1")]]);
  sockets.sync(node);
  const on = (id) => node.inputs[node.graph.links.get(id).target_slot]?.name;
  assert.equal(on(5), "cast.cast_1");
  assert.equal(on(6), "voices.voice_1");
  assert.equal(on(7), "shots.shot_1");
  assert.ok(node.inputs.some((i) => i.name === "shots.shot_2"), "the shot group must grow");
});

test("a rebuild grows the node to fit its sockets but never shrinks it", () => {
  // computeSize is the minimum. Taken outright on every load it squeezed a slate
  // saved 760 wide down to its narrowest, crushing the Asset Bin panel.
  const sized = (fit) => {
    const node = storeSlate(CREATED, [[7, CREATED.indexOf("shots.shot_1")]]);
    node.size = [760, 1700];
    node.computeSize = () => fit;
    node.setSize = (size) => { node.size = size; };
    sockets.sync(node);
    return node.size;
  };
  assert.deepEqual(sized([240, 900]), [760, 1700]);
  assert.deepEqual(sized([800, 1800]), [800, 1800]);
});

test("1.57: storeBacked tells the two socket models apart", () => {
  const node = storeSlate(["model"], []);
  assert.equal(sockets.storeBacked(node.inputs[0]), true);
  assert.equal(sockets.storeBacked({ name: "model", link: null }), false);
});

if (failures) {
  console.error(`\n${failures} JS socket test(s) failed:\n`);
  process.exit(1);
}
console.log("ps_sockets.js: all tests passed");
