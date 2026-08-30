/**
 * Growing socket groups for PulseSlate and PulseShot. Spec §4.
 *
 * `shots.shot_1..N` and `refs.ref_image_1..N` are declared in full on the Python
 * side as dot-namespaced optional inputs -- which is the shape ComfyUI's own
 * io.Autogrow expands to, and therefore the shape the frontend already draws with
 * the hollow optional-socket outline. Showing all thirty-two at once would be
 * unusable, so this file keeps exactly one free socket visible per group and adds
 * the next one when it is filled.
 *
 * WHY LINKS ARE RESTORED BY NAME
 *
 * A LiteGraph link stores `target_slot` as an INDEX into `node.inputs`. Removing
 * a socket therefore renumbers every socket after it, and any link pointing past
 * the removal silently reattaches to its neighbour -- a wire that still looks
 * connected, to the wrong input. With two growing groups on one node, trimming
 * the shot group would do exactly that to the reference group sitting after it.
 *
 * So `sync` never edits `node.inputs` in place. It snapshots name -> link id,
 * rebuilds the whole dynamic tail in canonical order, then reattaches each link
 * by name and repairs `target_slot` on the link object itself. A socket that
 * disappears takes its link with it deliberately; every other wire lands back on
 * the input it was actually connected to.
 *
 * Everything here is wrapped: a throw inside onConfigure aborts the entire
 * workflow load, which is a far worse failure than a node drawn with too many
 * sockets. See tests/js/test_sockets.mjs.
 */

import { app } from "../../scripts/app.js";

/** Which groups each node class grows, in the order they are declared in Python.
 *  The order matters: the rebuilt tail is written back in exactly this sequence,
 *  and it must match INPUT_TYPES or a saved graph's slots will not line up. */
export const GROUPS = {
  PulseSlate: [
    { prefix: "refs.ref_image_", max: 8 },
    { prefix: "shots.shot_", max: 24 },
    // Appended, never inserted: this list is rebuilt in order and a group placed
    // ahead of an existing one would move a saved graph's wires onto the wrong
    // sockets. It must stay in step with PulseSlate.INPUT_TYPES in nodes.py.
    { prefix: "voices.voice_", max: 3 },
  ],
  PulseShot: [
    { prefix: "refs.ref_image_", max: 4 },
  ],
};

/** Sockets that are not part of any growing group keep their place at the front. */
export function isDynamic(name, groups) {
  return groups.some((g) => typeof name === "string" && name.startsWith(g.prefix));
}

/**
 * The socket names a node should be showing: every connected one, plus one free
 * socket per group, capped at the group's maximum.
 *
 * Exported for the tests, and pure -- it takes the set of connected names and
 * returns names, touching no LiteGraph object.
 */
export function desiredNames(groups, connected) {
  const wanted = [];
  for (const group of groups) {
    const filled = [];
    for (let i = 1; i <= group.max; i++) {
      if (connected.has(`${group.prefix}${i}`)) filled.push(i);
    }
    // Numeric suffix order, never connection order (§4). A user who fills slot 3
    // and then slot 1 still gets them in the order the compiler will read them.
    for (const i of filled) wanted.push(`${group.prefix}${i}`);

    // The first free slot, so there is always somewhere to drop the next wire.
    for (let i = 1; i <= group.max; i++) {
      if (!connected.has(`${group.prefix}${i}`)) {
        wanted.push(`${group.prefix}${i}`);
        break;
      }
    }
  }
  return wanted;
}

function inputDefinition(node, name) {
  /** The declared type and options for one input, from the node definition. */
  const optional = node?.constructor?.nodeData?.input?.optional ?? {};
  const spec = optional[name];
  if (Array.isArray(spec)) return { type: spec[0], options: spec[1] ?? {} };
  return { type: "*", options: {} };
}

function labelFor(name) {
  /** `shots.shot_3` reads as `shot_3` on the node face; the namespace is
   *  plumbing and repeating it on every socket is noise. */
  const dot = name.indexOf(".");
  return dot === -1 ? name : name.slice(dot + 1);
}

/**
 * Rebuild a node's dynamic socket tail. Safe to call at any time.
 */
export function sync(node) {
  const groups = GROUPS[node?.comfyClass ?? node?.type];
  if (!groups) return;

  const inputs = node.inputs ?? [];

  // Snapshot: which dynamic sockets carry a link, and which link.
  const links = new Map();
  const connected = new Set();
  for (const input of inputs) {
    if (!isDynamic(input?.name, groups)) continue;
    if (input.link != null) {
      links.set(input.name, input.link);
      connected.add(input.name);
    }
  }

  const wanted = desiredNames(groups, connected);
  const fixed = inputs.filter((i) => !isDynamic(i?.name, groups));
  const existing = new Map(inputs.filter((i) => isDynamic(i?.name, groups))
                                 .map((i) => [i.name, i]));

  const rebuilt = wanted.map((name) => {
    const previous = existing.get(name);
    if (previous) {
      previous.link = links.has(name) ? links.get(name) : null;
      return previous;
    }
    const { type, options } = inputDefinition(node, name);
    return { name, label: labelFor(name), type, link: null,
             shape: LiteGraph?.HollowCircle, ...(options.tooltip
               ? { tooltip: options.tooltip } : {}) };
  });

  const next = [...fixed, ...rebuilt];
  const unchanged = next.length === inputs.length
    && next.every((input, i) => input === inputs[i]);
  if (unchanged) return;

  node.inputs = next;

  // Repair every link's target_slot against the new indices. Without this a wire
  // keeps pointing at the slot number it had before the rebuild.
  const graph = node.graph;
  if (graph?.links) {
    node.inputs.forEach((input, slot) => {
      if (input.link == null) return;
      const link = graph.links[input.link];
      if (link) {
        link.target_id = node.id;
        link.target_slot = slot;
      }
    });
  }

  node.setSize?.(node.computeSize?.() ?? node.size);
  node.setDirtyCanvas?.(true, true);
}

function guard(fn, label) {
  /** A throw in a LiteGraph hook aborts workflow loading. Never let one out. */
  return function (...args) {
    try {
      return fn.apply(this, args);
    } catch (error) {
      console.error(`[PulseStudio] ${label} failed`, error);
      return undefined;
    }
  };
}

app.registerExtension({
  name: "comfyui_pulse_studio.dynamic_sockets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!GROUPS[nodeData?.name]) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = guard(function () {
      const result = onNodeCreated?.apply(this, arguments);
      sync(this);
      return result;
    }, `${nodeData.name}.onNodeCreated`);

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = guard(function (info) {
      const result = onConfigure?.apply(this, arguments);
      // After LiteGraph has restored the saved inputs and links. Rebuilding here
      // is what makes a workflow saved with six shots reopen with six shots and a
      // seventh free socket, rather than with whatever this build declares.
      sync(this);
      return result;
    }, `${nodeData.name}.onConfigure`);

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = guard(function (type, index, connected,
                                                             linkInfo, ioSlot) {
      const result = onConnectionsChange?.apply(this, arguments);
      // Deferred one tick: LiteGraph calls this *during* its own link bookkeeping,
      // and rebuilding the input array underneath it corrupts the link it is in
      // the middle of attaching.
      requestAnimationFrame(() => {
        try {
          sync(this);
        } catch (error) {
          console.error("[PulseStudio] socket sync failed", error);
        }
      });
      return result;
    }, `${nodeData.name}.onConnectionsChange`);
  },
});
