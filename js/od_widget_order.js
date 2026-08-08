/**
 * Widget ordering safety for the Omni-Director node face.
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
 */

/** The Director's native widgets, in INPUT_TYPES order.
 *  `control_after_generate` is inserted by the frontend right after `seed`. */
export const NATIVE_WIDGETS = [
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
  { name: "timeline_data", kind: "string" },
];

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
 * Assert that no custom DOM widget sits before a native one.
 * Returns { ok, offenders: [names] } -- the invariant that keeps saving safe.
 */
export function checkWidgetOrder(widgets, isCustom = (w) => /^od_/.test(w?.name ?? "")) {
  const offenders = [];
  let seenCustom = null;
  for (const widget of widgets ?? []) {
    if (isCustom(widget)) {
      if (seenCustom === null) seenCustom = widget.name;
      continue;
    }
    if (seenCustom !== null) {
      // A native widget appearing AFTER a custom one means the custom widget
      // occupies a slot ahead of it, and every value from here on will shift.
      offenders.push(widget.name);
    }
  }
  return { ok: offenders.length === 0, offenders, firstCustom: seenCustom };
}

/** A human-readable explanation for the console when a load looks misaligned. */
export function describeMisalignment(report) {
  if (report.ok) return "";
  const lines = report.mismatches.slice(0, 6).map(
    (m) => `    ${m.name}: expected ${m.expected}, got ${m.got} (${JSON.stringify(m.value)})`);
  return (
    "[OmniDirector] This workflow's stored widget values do not line up with the " +
    "node's inputs:\n" + lines.join("\n") +
    (report.mismatches.length > 6 ? `\n    ...and ${report.mismatches.length - 6} more` : "") +
    "\n  This file was saved by a build that inserted panel headers into the widget " +
    "list, which shifted every value. The layout bug is fixed, but the saved values " +
    "cannot be recovered — the text that landed in a header slot was discarded at " +
    "save time.\n  Load a fresh copy of example workflow/OmniDirector_Starter.json " +
    "and re-enter your prompts.");
}
