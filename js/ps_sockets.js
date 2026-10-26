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
 * by name and repairs `target_slot` on the link object itself. Only a free
 * socket is ever removed; every wire lands back on the input it was actually
 * connected to. On a load the snapshot comes from the saved node, not the live
 * sockets (savedLinks), and on frontend 1.57 the link object is the only thing
 * that can move a wire (storeBacked).
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
    // Appended after voices, for the same reason.
    { prefix: "cast.cast_", max: 6 },
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
 * The link object for an id, on any frontend. LiteGraph used to keep
 * `graph.links` as a plain object keyed by id; frontend 1.53 keeps a Map. Read
 * it the old way on a Map and every lookup is `undefined` -- which is how a
 * rebuild used to leave each wire pointing at its old slot number, and the
 * frontend, which resolves a wire by slot number, then delivered PULSE_SHOT to
 * whatever widget input had moved into that slot.
 */
export function linkById(links, id) {
  if (links == null || id == null) return undefined;
  if (typeof links.get === "function") return links.get(id) ?? links.get(Number(id));
  return links[id];
}

/**
 * Where each socket goes. Plain connection sockets first, then the growing
 * groups, then the widget inputs frontend 1.53 gives every widget -- the order
 * the frontend itself creates them in, and the layout every shipped graph
 * saves, so a saved slot number means the same socket on every build. Duplicate
 * names (a saved array laid over the node's default one leaves copies behind)
 * keep one entry, preferring the one that carries a wire.
 */
export function orderInputs(fixed, rebuilt) {
  const seen = new Map();
  for (const input of fixed) {
    const key = input?.name;
    if (key == null) continue;
    const kept = seen.get(key);
    if (!kept || (kept.link == null && input.link != null)) seen.set(key, input);
  }
  const unique = fixed.filter((input) => input?.name == null || seen.get(input.name) === input);
  const sockets = unique.filter((input) => !input?.widget);
  const widgets = unique.filter((input) => input?.widget);
  return [...sockets, ...rebuilt, ...widgets];
}

/**
 * Which wire each socket carried when the node was saved, by socket name.
 *
 * On a load this is the only record that still pairs a wire with its socket.
 * Frontend 1.57 hangs every saved wire on the node's *creation-time* layout by
 * slot number before `configure` runs -- the layout with one free socket per
 * group -- and `configure` then merges the saved sockets in by name, keeping the
 * misplaced wires. A slate saved with four shots and two cast members had its
 * cast wires on shot_5 and shot_6 by the time onConfigure ran, and the server
 * refused the prompt as an invalid connection. Null when the serialised node
 * carries no inputs (a paste, a clone), which leaves the live links in charge.
 */
export function savedLinks(info) {
  if (!Array.isArray(info?.inputs)) return null;
  const saved = new Map();
  for (const input of info.inputs) {
    if (input?.name != null && input.link != null) saved.set(input.name, input.link);
  }
  return saved;
}

/**
 * True where a socket's `link` is derived from the frontend's link store (1.57
 * and later) rather than stored on the socket. There, assigning a link id does
 * nothing and assigning null *disconnects* whatever wire sits at that slot
 * number -- so this file must never write `input.link` on such a socket. A wire
 * is moved by changing its own `target_slot` instead.
 */
export function storeBacked(input) {
  const proto = input == null ? null : Object.getPrototypeOf(input);
  const descriptor = proto ? Object.getOwnPropertyDescriptor(proto, "link") : undefined;
  return typeof descriptor?.set === "function";
}

/** Where a wire is parked while a rebuild moves the others. No node has a
 *  socket this far down, so a parked wire can never land on a live one. */
const PARKING_SLOT = 1 << 20;

/** name -> link id, read off the sockets as they stand -- before anything moves.
 *  On frontend 1.57 a socket's link is looked up by its slot number, so reading
 *  this after a reorder would hand each name its neighbour's wire. */
function liveLinks(node) {
  const wires = new Map();
  (node.inputs ?? []).forEach((input, slot) => {
    if (input?.name == null || wires.has(input.name)) return;
    const id = typeof node.getInputLink === "function" && storeBacked(input)
      ? node.getInputLink(slot)?.id
      : input.link;
    if (id != null) wires.set(input.name, id);
  });
  return wires;
}

/**
 * Put each wire back on the socket it belongs to, by name, after node.inputs
 * has been rebuilt.
 *
 * Every wire is parked first and placed second. Frontend 1.57's link store
 * refuses a move onto a slot another wire still occupies, so moving the wires
 * one at a time fails the moment two of them trade places -- which is exactly
 * what a socket inserted ahead of a connected one asks for.
 */
function reattach(node, wires) {
  const slotOf = new Map();
  node.inputs.forEach((input, slot) => {
    if (input?.name != null && !slotOf.has(input.name)) slotOf.set(input.name, slot);
  });

  if (!node.inputs.some(storeBacked)) {
    // Older frontends keep the wire on the socket itself.
    for (const input of node.inputs) {
      if (input?.name != null) input.link = wires.get(input.name) ?? null;
    }
  }

  const graph = node.graph;
  if (!graph?.links) return;
  const moves = [];
  for (const [name, id] of wires) {
    const link = linkById(graph.links, id);
    const slot = slotOf.get(name);
    if (link && slot != null) moves.push([link, slot]);
  }
  moves.forEach(([link], i) => {
    link.target_slot = PARKING_SLOT + i;
  });
  for (const [link, slot] of moves) {
    if (link.target_id !== node.id) link.target_id = node.id;
    link.target_slot = slot;
  }
}

/**
 * Rebuild a node's dynamic socket tail. Safe to call at any time.
 *
 * `saved`, from savedLinks(), says which wire belongs on which socket in place of
 * the live sockets -- on a load, the live ones are wrong (see savedLinks).
 */
export function sync(node, saved = null) {
  const groups = GROUPS[node?.comfyClass ?? node?.type];
  if (!groups) return;

  const inputs = node.inputs ?? [];
  const wires = saved ?? liveLinks(node);
  const connected = new Set([...wires.keys()].filter((name) => isDynamic(name, groups)));

  const wanted = desiredNames(groups, connected);
  const fixed = inputs.filter((i) => !isDynamic(i?.name, groups));
  const existing = new Map(inputs.filter((i) => isDynamic(i?.name, groups))
                                 .map((i) => [i.name, i]));

  const rebuilt = wanted.map((name) => {
    const previous = existing.get(name);
    if (previous) return previous;
    const { type, options } = inputDefinition(node, name);
    return { name, label: labelFor(name), type, link: null,
             shape: LiteGraph?.HollowCircle, ...(options.tooltip
               ? { tooltip: options.tooltip } : {}) };
  });

  const next = orderInputs(fixed, rebuilt);
  const unchanged = next.length === inputs.length
    && next.every((input, i) => input === inputs[i]);
  // Same sockets in the same order can still need every wire moved: on a load
  // it is the wires that are misplaced, not the sockets.
  if (unchanged && !saved) return;

  node.inputs = next;
  // Without this a wire keeps pointing at the slot number it had before the
  // rebuild, and the frontend resolves a wire by slot number.
  reattach(node, wires);

  // Grow to fit the sockets, never shrink: computeSize is the node's minimum, and
  // a load runs this rebuild, so taking it outright squeezed every saved slate to
  // its narrowest and crushed the Asset Bin panel along with it.
  const fit = node.computeSize?.();
  if (fit && node.size) {
    node.setSize?.([Math.max(node.size[0], fit[0]), Math.max(node.size[1], fit[1])]);
  }
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
      // seventh free socket, rather than with whatever this build declares. The
      // wires come from the saved record, not the live sockets -- see savedLinks.
      sync(this, savedLinks(info));
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
