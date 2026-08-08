/**
 * Node-face warnings for Pulse Studio (spec §10, §18.1).
 *
 * WHY THIS EXISTS SEPARATELY FROM THE PANEL
 *
 * The patch warning applies to every node in the pack, not just the one with an
 * Asset Bin, and it must work on a node that has no DOM widget at all.
 *
 * WHY IT PAINTS ON THE CANVAS INSTEAD OF ADDING A WIDGET
 *
 * §3.2 allows exactly one addDOMWidget in the entire JS layer, and the Asset Bin
 * has it. A warning banner is not worth a widgets_values slot -- and a slot added
 * here would land in the middle of the array for anyone whose saved workflow
 * predates it. Canvas painting in onDrawForeground costs no slot at all, which is
 * the same reason section headings use widget.label rather than header widgets.
 *
 * WHAT IT SHOWS
 *
 * Whether the incoming MODEL carries an attention patch and an offload patch.
 * Both are warnings and neither blocks: the user may be running unpatched
 * deliberately. The message the backend sends is the message shown -- this file
 * decides where it appears, never what it says, so the console and the node face
 * cannot drift apart.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EVENT = "pulse_studio.warnings";
const BAND = "#d19a3a";
const BAND_TEXT = "#1a1206";

/** Wrap text to a pixel width using the canvas' own measurement. */
function wrap(ctx, text, maxWidth) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (ctx.measureText(candidate).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/**
 * Draw the warning band across the top of the node body.
 *
 * Wrapped in try/finally around the canvas state: a throw mid-draw with a
 * modified transform or fillStyle corrupts every node painted afterwards, and a
 * cosmetic warning must never be able to break the graph's rendering.
 */
function drawWarnings(node, ctx) {
  const warnings = node.psWarnings;
  if (!warnings || !warnings.length || node.flags?.collapsed) return;

  ctx.save();
  try {
    const width = (node.size?.[0] ?? 300) - 12;
    ctx.font = "10px sans-serif";

    const lines = [];
    for (const warning of warnings) lines.push(...wrap(ctx, warning, width - 16));

    const lineHeight = 12;
    const height = lines.length * lineHeight + 14;
    const top = -height - 4;

    ctx.fillStyle = BAND;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(6, top, width, height, 4);
    else ctx.rect(6, top, width, height);
    ctx.fill();

    ctx.fillStyle = BAND_TEXT;
    ctx.textBaseline = "top";
    ctx.fillText("⚠ UNPATCHED MODEL", 14, top + 4);
    lines.forEach((line, i) => ctx.fillText(line, 14, top + 4 + (i + 1) * lineHeight));
  } catch (err) {
    console.warn("[PulseStudio] could not draw the warning band:", err);
  } finally {
    ctx.restore();
  }
}

function nodeById(id) {
  try {
    return app.graph?.getNodeById?.(Number(id)) ?? app.graph?.getNodeById?.(id) ?? null;
  } catch (err) {
    return null;
  }
}

app.registerExtension({
  name: "comfyui_pulse_studio.warnings",

  async setup() {
    // One listener for the whole pack. The backend addresses each message to a
    // node id, so a graph with three Pulse nodes shows three separate bands.
    api.addEventListener(EVENT, (event) => {
      try {
        const { node_id: nodeId, warnings } = event?.detail ?? {};
        const node = nodeById(nodeId);
        if (!node) return;
        node.psWarnings = Array.isArray(warnings) ? warnings : [];
        for (const warning of node.psWarnings) {
          console.warn(`[PulseStudio] ${warning}`);
        }
        app.graph?.setDirtyCanvas?.(true, true);
      } catch (err) {
        console.warn("[PulseStudio] warning event could not be handled:", err);
      }
    });

    // A new run re-derives everything, so last run's warnings are stale the
    // moment the queue starts. Clearing on execution_start rather than on
    // completion means a warning that stops applying actually disappears.
    api.addEventListener("execution_start", () => {
      try {
        for (const node of app.graph?._nodes ?? []) {
          if (node.psWarnings) node.psWarnings = null;
        }
        app.graph?.setDirtyCanvas?.(true, true);
      } catch (err) {
        /* nothing to clear */
      }
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!String(nodeData?.name ?? "").startsWith("Pulse")) return;

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const result = onDrawForeground?.apply(this, arguments);
      drawWarnings(this, ctx);
      return result;
    };
  },
});
