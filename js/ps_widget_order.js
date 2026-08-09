/**
 * Widget ordering safety and workflow migration for the Pulse Studio nodes.
 *
 * THE BUG THIS EXISTS TO PREVENT
 *
 * LiteGraph serialises a node as `widgets_values[i] = node.widgets[i].value` --
 * a POSITIONAL array over the live widget list. A DOM widget added with
 * `addDOMWidget` still occupies an entry in `node.widgets`, and `serialize:false`
 * does not remove its slot on every frontend build.
 *
 * So inserting a decorative DOM widget *before* a native one shifts every native
 * value after it by one. Saving then writes the shifted array, and loading it
 * feeds the backend garbage:
 *
 *     duration_seconds <- 'res_multistep'
 *     width            <- 1
 *     seed             <- 'image'
 *     cfg              <- 'crop'
 *
 * which is exactly what happened: five headers spliced at indices 0,2,4,5,6 made
 * ComfyUI reject the prompt with thirteen type errors, and destroyed the text in
 * whichever native widget got pushed into a DOM slot.
 *
 * THE RULE: every custom DOM widget is APPENDED after all native widgets, never
 * inserted among them. Trailing extra slots are harmless -- a shorter saved array
 * simply leaves them undefined. Leading or interleaved slots are fatal.
 *
 * Section headings are therefore done with `widget.label` and CSS on the native
 * elements, which costs no slot at all.
 *
 * WHY POSITIONAL LOADING IS GONE
 *
 * Appending is only half the fix. It keeps the DOM widget out of the way, but it
 * does nothing about the *next* widget this node gains: the moment a workflow is
 * saved, widget order is a public API indexed by position, and every future
 * append shifts the DOM widget's own index.
 *
 * So loading no longer reads by position at all. Widget index 0 is
 * `schema_version`, a hidden string the node writes on every save. On load we
 * read it, look up the name list that version was written with, rebuild a
 * name -> value map, and assign BY NAME. A file that does not carry a known
 * schema version is refused rather than guessed at.
 *
 * That is what makes future appends safe. See spec §3.
 */

/** The schema version this build writes. Must match SCHEMA_VERSION in
 *  comfyui_pulse_studio/constants.py -- tests/test_js_guard.py asserts it. */
export const CURRENT = "3.0.0";

/**
 * Live widget layout, per node class, per schema version.
 *
 * These are the names in `node.widgets` order, which is INPUT_TYPES order with
 * one addition: the frontend inserts `control_after_generate` immediately after
 * any widget declared with `control_after_generate: true`. It is a real widget
 * and consumes a real slot, so it belongs in this table.
 *
 * Keyed by node class because this pack ships three nodes and all three are
 * under the contract. The spec sketches a single flat table; three nodes need
 * three, and a shared version key across them keeps one number to bump.
 *
 * APPEND ONLY. Adding a name anywhere but the end of a list, or removing or
 * retyping one, silently corrupts every workflow already saved.
 */
const SPECS = {
  PulseSlate: {
    "2.0.0": [
      { name: "schema_version", kind: "string" },
      { name: "timeline_data", kind: "string" },
      { name: "global_prompt", kind: "string" },
      { name: "shot_prompt", kind: "string" },
      { name: "duration_seconds", kind: "number" },
      { name: "aspect_ratio", kind: "string" },
      { name: "width", kind: "number" },
      { name: "height", kind: "number" },
      { name: "steps", kind: "number" },
      { name: "sampler_name", kind: "string" },
      { name: "scheduler", kind: "string" },
      { name: "cfg", kind: "number" },
      { name: "seed", kind: "number" },
      { name: "control_after_generate", kind: "string", optional: true },
      { name: "partition_strategy", kind: "string" },
      { name: "window_seconds", kind: "number" },
      { name: "resize_method", kind: "string" },
      { name: "carry_mode", kind: "string" },
      { name: "carry_audio", kind: "boolean" },
      { name: "carry_audio_seconds", kind: "number" },
      { name: "ref_image_size", kind: "string" },
      { name: "shift_video", kind: "number" },
      { name: "shift_audio", kind: "number" },
      { name: "audio_ref_ceiling", kind: "number" },
    ],
    // 3.0.0 appends `continuity` and nothing else. Every 2.0.0 name is still
    // here, in the same order, which is what lets a workflow saved by 2.0.0 load
    // into this build by name -- spec §13. The 2.0.0 entry above is not dead
    // weight; deleting it is what would make those files unreadable.
    "3.0.0": [
      { name: "schema_version", kind: "string" },
      { name: "timeline_data", kind: "string" },
      { name: "global_prompt", kind: "string" },
      { name: "shot_prompt", kind: "string" },
      { name: "duration_seconds", kind: "number" },
      { name: "aspect_ratio", kind: "string" },
      { name: "width", kind: "number" },
      { name: "height", kind: "number" },
      { name: "steps", kind: "number" },
      { name: "sampler_name", kind: "string" },
      { name: "scheduler", kind: "string" },
      { name: "cfg", kind: "number" },
      { name: "seed", kind: "number" },
      { name: "control_after_generate", kind: "string", optional: true },
      { name: "partition_strategy", kind: "string" },
      { name: "window_seconds", kind: "number" },
      { name: "resize_method", kind: "string" },
      { name: "carry_mode", kind: "string" },
      { name: "carry_audio", kind: "boolean" },
      { name: "carry_audio_seconds", kind: "number" },
      { name: "ref_image_size", kind: "string" },
      { name: "shift_video", kind: "number" },
      { name: "shift_audio", kind: "number" },
      { name: "audio_ref_ceiling", kind: "number" },
      { name: "continuity", kind: "string" },
    ],
  },
  PulseShot: {
    "3.0.0": [
      { name: "schema_version", kind: "string" },
      { name: "shot_id", kind: "string" },
      { name: "label", kind: "string" },
      { name: "visual", kind: "string" },
      { name: "audio_line", kind: "string" },
      { name: "duration_seconds", kind: "number" },
      { name: "continuity", kind: "string" },
    ],
  },
  PulseRender: {
    "3.0.0": [
      { name: "schema_version", kind: "string" },
      { name: "cache_mode", kind: "string" },
      { name: "run_dir", kind: "string" },
      { name: "run_id", kind: "string" },
      { name: "save_segments", kind: "boolean" },
      { name: "low_memory", kind: "boolean" },
      { name: "dry_run", kind: "boolean" },
      { name: "prune_unused", kind: "boolean" },
    ],
  },
  PulseBench: {
    "3.0.0": [
      { name: "schema_version", kind: "string" },
      { name: "run_dirs", kind: "string" },
    ],
  },
  PulseRetake: {
    "2.0.0": [
      { name: "schema_version", kind: "string" },
      { name: "prompt", kind: "string" },
      { name: "cut_start_seconds", kind: "number" },
      { name: "cut_end_seconds", kind: "number" },
      { name: "keep_base_audio", kind: "boolean" },
      { name: "fps", kind: "number" },
      { name: "seed", kind: "number" },
      { name: "control_after_generate", kind: "string", optional: true },
      { name: "steps", kind: "number" },
      { name: "sampler_name", kind: "string" },
      { name: "scheduler", kind: "string" },
      { name: "cfg", kind: "number" },
      { name: "shift_video", kind: "number" },
      { name: "shift_audio", kind: "number" },
    ],
    // Unchanged in 3.0.0. Listed anyway: the version key is shared across the
    // pack so there is one number to bump, and a node with no entry at CURRENT
    // reports as drifted rather than as untouched.
    "3.0.0": [
      { name: "schema_version", kind: "string" },
      { name: "prompt", kind: "string" },
      { name: "cut_start_seconds", kind: "number" },
      { name: "cut_end_seconds", kind: "number" },
      { name: "keep_base_audio", kind: "boolean" },
      { name: "fps", kind: "number" },
      { name: "seed", kind: "number" },
      { name: "control_after_generate", kind: "string", optional: true },
      { name: "steps", kind: "number" },
      { name: "sampler_name", kind: "string" },
      { name: "scheduler", kind: "string" },
      { name: "cfg", kind: "number" },
      { name: "shift_video", kind: "number" },
      { name: "shift_audio", kind: "number" },
    ],
  },
  PulseStill: {
    "2.0.0": [
      { name: "schema_version", kind: "string" },
      { name: "prompt", kind: "string" },
      { name: "aspect_ratio", kind: "string" },
      { name: "width", kind: "number" },
      { name: "height", kind: "number" },
      { name: "frame_pick", kind: "number" },
      { name: "canvas_from_reference", kind: "boolean" },
      { name: "seed", kind: "number" },
      { name: "control_after_generate", kind: "string", optional: true },
      { name: "steps", kind: "number" },
      { name: "sampler_name", kind: "string" },
      { name: "scheduler", kind: "string" },
      { name: "cfg", kind: "number" },
      { name: "shift_video", kind: "number" },
      { name: "shift_audio", kind: "number" },
    ],
    // Unchanged in 3.0.0; see the note on PulseRetake.
    "3.0.0": [
      { name: "schema_version", kind: "string" },
      { name: "prompt", kind: "string" },
      { name: "aspect_ratio", kind: "string" },
      { name: "width", kind: "number" },
      { name: "height", kind: "number" },
      { name: "frame_pick", kind: "number" },
      { name: "canvas_from_reference", kind: "boolean" },
      { name: "seed", kind: "number" },
      { name: "control_after_generate", kind: "string", optional: true },
      { name: "steps", kind: "number" },
      { name: "sampler_name", kind: "string" },
      { name: "scheduler", kind: "string" },
      { name: "cfg", kind: "number" },
      { name: "shift_video", kind: "number" },
      { name: "shift_audio", kind: "number" },
    ],
  },
};

/** Names only, in the shape spec §3.3 describes: WIDGET_NAMES[class][version]. */
export const WIDGET_NAMES = Object.fromEntries(
  Object.entries(SPECS).map(([nodeId, versions]) => [
    nodeId,
    Object.fromEntries(
      Object.entries(versions).map(([v, spec]) => [v, spec.map((w) => w.name)])),
  ]));

/** The primary node's current spec. Kept as a named export because the value
 *  validator and its tests default to it. */
export const NATIVE_WIDGETS = SPECS.PulseSlate[CURRENT];

export const NODE_IDS = Object.keys(SPECS);

/** The widget spec for a node class at a schema version, or null if unknown. */
export function widgetSpec(nodeId, version = CURRENT) {
  return SPECS[nodeId]?.[version] ?? null;
}

/** The widget names for a node class at a schema version, or null if unknown. */
export function widgetNames(nodeId, version = CURRENT) {
  return WIDGET_NAMES[nodeId]?.[version] ?? null;
}

/** Is this a schema version we know how to read for this node? */
export function isKnownVersion(nodeId, version) {
  return Boolean(widgetNames(nodeId, version));
}

function kindOf(value) {
  if (typeof value === "number") return "number";
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "string") return "string";
  return "other";
}

/**
 * Does a saved widgets_values array line up with the node's native widgets?
 *
 * Type-based rather than length-based, so it catches a misalignment however it
 * arose -- a stale workflow, a hand-edited file, or a future widget inserted in
 * the middle. Returns { ok, mismatches: [{index, name, expected, got, value}] }.
 */
export function validateWidgetValues(values, spec = NATIVE_WIDGETS) {
  const mismatches = [];
  if (!Array.isArray(values)) return { ok: false, mismatches, unreadable: true };

  for (let i = 0; i < spec.length && i < values.length; i++) {
    const want = spec[i];
    const got = kindOf(values[i]);
    if (got === "other") continue; // null/undefined: absent, not misaligned
    // A boolean landing in a number slot is a real shift; a number in a string
    // slot likewise. Only exact-kind agreement counts as aligned.
    if (got !== want.kind) {
      mismatches.push({ index: i, name: want.name, expected: want.kind,
                        got, value: values[i] });
    }
  }
  return { ok: mismatches.length === 0, mismatches, unreadable: false };
}

/**
 * Assert that no custom DOM widget sits before a native one, and -- when a node
 * class is named -- that the live native widget names are exactly
 * WIDGET_NAMES[nodeId][CURRENT], in order.
 *
 * The first check keeps saving safe. The second is what catches a widget added
 * to INPUT_TYPES without updating this file, which would otherwise leave the
 * loader assigning values to the wrong names for the rest of the node's life.
 *
 * Returns { ok, offenders, firstCustom, nameError }.
 */
export function checkWidgetOrder(widgets, isCustom = (w) => /^ps_/.test(w?.name ?? ""),
                                 nodeId = null) {
  const offenders = [];
  let seenCustom = null;
  const liveNames = [];
  for (const widget of widgets ?? []) {
    if (isCustom(widget)) {
      if (seenCustom === null) seenCustom = widget.name;
      continue;
    }
    liveNames.push(widget?.name);
    if (seenCustom !== null) {
      // A native widget appearing AFTER a custom one means the custom widget
      // occupies a slot ahead of it, and every value from here on will shift.
      offenders.push(widget.name);
    }
  }

  let nameError = null;
  if (nodeId) {
    const expected = widgetNames(nodeId, CURRENT);
    if (!expected) {
      nameError = `no widget name table for ${nodeId} at schema ${CURRENT}`;
    } else {
      // control_after_generate is inserted by the frontend and is absent on
      // builds that render the seed control differently. Compare without it
      // when the live node does not have it, rather than failing on a
      // difference that is not ours.
      const want = liveNames.includes("control_after_generate")
        ? expected
        : expected.filter((n) => n !== "control_after_generate");
      if (liveNames.length !== want.length
          || want.some((n, i) => n !== liveNames[i])) {
        nameError =
          `${nodeId} widget names have drifted from ps_widget_order.js.\n` +
          `    expected: ${want.join(", ")}\n` +
          `    live:     ${liveNames.join(", ")}`;
      }
    }
  }

  return { ok: offenders.length === 0 && !nameError, offenders,
           firstCustom: seenCustom, nameError };
}

/**
 * Rebuild a name -> value map from a saved widgets_values array. Spec §3.3.1-2.
 *
 * Returns { ok, version, values } or { ok: false, reason } -- never a guess. A
 * file whose slot 0 is not a known schema version cannot be read positionally
 * either, because we have no way to know which layout wrote it.
 */
export function readSavedValues(nodeId, savedArray) {
  if (!Array.isArray(savedArray)) {
    return { ok: false, reason: "widgets_values is not an array" };
  }
  const version = savedArray[0];
  const names = widgetNames(nodeId, version);
  if (!names) {
    return {
      ok: false,
      version,
      reason:
        `this workflow was saved with schema version ${JSON.stringify(version)}, ` +
        `which this build of ${nodeId} cannot read. Known versions: ` +
        `${Object.keys(SPECS[nodeId] ?? {}).join(", ") || "none"}.`,
    };
  }
  const values = new Map();
  for (let i = 0; i < names.length && i < savedArray.length; i++) {
    values.set(names[i], savedArray[i]);
  }
  return { ok: true, version, values };
}

/**
 * Restore a saved workflow into a live node, by name. Spec §3.3.
 *
 * Positional assignment is never used. Names the live node has but the file
 * does not fall back to their declared default; names the file has but the live
 * node does not are dropped with a warning. Trailing slots beyond the native
 * widget count are the DOM widget's dead slot and are never assigned.
 *
 * Returns { ok, version, applied, defaulted, dropped, reason }.
 */
export function applySavedValues(node, nodeId, savedArray, warn = console.warn) {
  const read = readSavedValues(nodeId, savedArray);
  if (!read.ok) return { ok: false, version: read.version, reason: read.reason,
                         applied: [], defaulted: [], dropped: [] };

  const applied = [];
  const defaulted = [];
  const live = new Set();

  for (const widget of node?.widgets ?? []) {
    const name = widget?.name;
    if (!name || /^ps_/.test(name)) continue; // the appended DOM widget
    live.add(name);
    if (read.values.has(name)) {
      widget.value = read.values.get(name);
      applied.push(name);
    } else if (widget.options && "default" in widget.options) {
      // Present in the live node, absent from the file: a widget appended by a
      // newer build than the one that saved this workflow. Its declared default
      // is exactly right, and is why appending is safe.
      widget.value = widget.options.default;
      defaulted.push(name);
    }
  }

  const dropped = [...read.values.keys()].filter((n) => !live.has(n));
  if (dropped.length) {
    warn(`[PulseStudio] ${nodeId}: saved values for widgets this build no longer ` +
         `has were dropped: ${dropped.join(", ")}`);
  }
  return { ok: true, version: read.version, applied, defaulted, dropped };
}

/** A human-readable explanation for the console when a load looks misaligned. */
export function describeMisalignment(report) {
  if (report.ok) return "";
  const lines = report.mismatches.slice(0, 6).map(
    (m) => `    ${m.name}: expected ${m.expected}, got ${m.got} (${JSON.stringify(m.value)})`);
  return (
    "[PulseStudio] This workflow's stored widget values do not line up with the " +
    "node's inputs:\n" + lines.join("\n") +
    (report.mismatches.length > 6 ? `\n    ...and ${report.mismatches.length - 6} more` : "") +
    "\n  This file was saved by a build that inserted panel headers into the widget " +
    "list, which shifted every value. The layout bug is fixed, but the saved values " +
    "cannot be recovered — the text that landed in a header slot was discarded at " +
    "save time.\n  Load a fresh copy of example_workflows/PulseSlate_Starter.json " +
    "and re-enter your prompts.");
}

/** The message shown on the node face when a file cannot be read at all. */
export function describeUnloadable(nodeId, reason) {
  return (
    `[PulseStudio] ${nodeId}: this workflow cannot be loaded.\n  ${reason}\n` +
    "  Workflows saved by any build before 2.0.0 are not loadable and must be " +
    "recreated — see CHANGELOG.md. Start from " +
    "example_workflows/PulseSlate_Starter.json.");
}
