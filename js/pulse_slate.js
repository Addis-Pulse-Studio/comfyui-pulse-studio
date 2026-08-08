/**
 * Asset Bin panel + node-face layout for Pulse Slate.
 *
 * TWO RULES THIS FILE OBEYS
 *
 * 1. It never computes a reference ordinal. The numbering rule lives on the
 *    server, in comfyui_pulse_studio/assets.py, and is the tested one. A second
 *    implementation here is how the two would drift apart, and a drifted tag
 *    renders successfully while describing the wrong picture.
 *
 * 2. It never builds timeline JSON, and never writes to a prompt widget.
 *    Every edit is POSTed to /pulse_studio/apply, which merges server-side and
 *    returns the complete new string. This file only assigns the string it was
 *    handed back.
 *
 * NOTHING HERE MAY ABORT A WORKFLOW LOAD. Every hook is wrapped: a failure in
 * this extension must degrade the node's appearance, never stop ComfyUI from
 * opening the user's file.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { protectWidget } from "./ps_widget_guard.js";
import {
  CURRENT,
  NODE_IDS,
  applySavedValues,
  checkWidgetOrder,
  describeMisalignment,
  describeUnloadable,
  validateWidgetValues,
} from "./ps_widget_order.js";

const NODE_ID = "PulseSlate";
const SCHEMA_WIDGET = "schema_version";
const STORAGE_WIDGET = "timeline_data";
const PROMPT_WIDGETS = ["global_prompt", "shot_prompt"];
const CONTROLS_START = "duration_seconds"; // first widget of the generation panel

const EXT_KIND = {
  png: "image", jpg: "image", jpeg: "image", webp: "image", bmp: "image", gif: "image",
  mp4: "video", mov: "video", webm: "video", mkv: "video", avi: "video", m4v: "video",
  wav: "audio", mp3: "audio", flac: "audio", ogg: "audio", m4a: "audio", aac: "audio",
};
const ACCEPT = Object.keys(EXT_KIND).map((e) => "." + e).join(",");

const KIND_META = {
  image: { label: "IMAGE REFERENCES", accent: "#4ea36b", hint: "characters, props, locations" },
  video: { label: "VIDEO REFERENCES", accent: "#d19a3a", hint: "motion and camera reference, 2-15s" },
  audio: { label: "AUDIO REFERENCES", accent: "#d1743a", hint: "voice timbre and sonic character" },
};

const CSS = `
.ps-wrap { display:flex; flex-direction:column; gap:8px; font-size:11px;
  font-family:var(--font-family, sans-serif); color:#dcdcdc; box-sizing:border-box;
  height:100%; overflow:auto; padding:2px; }
.ps-card { border:1px solid #444; border-left-width:3px; border-radius:6px;
  background:#141414; padding:7px 8px; box-sizing:border-box; }
.ps-card.ps-filedrop { background:#18232c; box-shadow:inset 0 0 0 1px #6ab0ff; }
.ps-card-head { display:flex; justify-content:space-between; align-items:baseline;
  gap:8px; margin-bottom:6px; }
.ps-card-title { font-size:10px; font-weight:700; letter-spacing:.09em; }
.ps-card-hint { font-size:9px; color:#7d7d7d; font-style:italic; }
.ps-count { font-size:9px; color:#9a9a9a; font-variant-numeric:tabular-nums; }

.ps-meter { display:flex; gap:10px; flex-wrap:wrap; padding:5px 8px; border-radius:5px;
  background:#141414; border:1px solid #444; border-left:3px solid #6d7f92;
  font-variant-numeric:tabular-nums; font-size:10px; }
.ps-meter.ps-over { background:#3a1717; border-color:#8d3b3b; color:#ffb4b4; }
.ps-meter.ps-beyond { background:#33240c; border-color:#8a6a2a; color:#f0cf90; }

.ps-row { display:grid; grid-template-columns:44px 1fr auto auto; gap:7px;
  align-items:center; padding:4px; border-radius:4px; background:#1d1d1d;
  margin-bottom:4px; cursor:grab; }
.ps-row:last-child { margin-bottom:0; }
.ps-row.ps-drag { opacity:.35; }
.ps-row.ps-over-drop { box-shadow:inset 0 0 0 1px #6ab0ff; }

.ps-thumb { width:44px; height:34px; border-radius:3px; background:#0c0c0c;
  object-fit:cover; display:block; border:1px solid #333; }
.ps-thumb-box { width:44px; height:34px; border-radius:3px; background:#0c0c0c;
  border:1px solid #333; display:flex; align-items:center; justify-content:center;
  overflow:hidden; position:relative; }
.ps-thumb-box video { width:100%; height:100%; object-fit:cover; }
.ps-badge { position:absolute; right:1px; bottom:1px; font-size:7px; padding:0 2px;
  border-radius:2px; background:rgba(0,0,0,.7); color:#ddd; letter-spacing:.04em; }

.ps-name { background:#111; border:1px solid #3a3a3a; color:#eee; font:inherit;
  width:100%; padding:3px 5px; border-radius:3px; box-sizing:border-box; }
.ps-name:focus { border-color:#6ab0ff; outline:none; background:#161c22; }
.ps-tag { font-family:ui-monospace,Consolas,monospace; font-size:10px; color:#8fc7ff;
  white-space:nowrap; }
.ps-sub { grid-column:1/5; font-size:9px; color:#8a8a8a; display:flex;
  align-items:center; gap:4px; padding-left:2px; }
.ps-btn { background:#2e2e2e; border:1px solid #444; color:#ccc; border-radius:3px;
  cursor:pointer; padding:3px 7px; font-size:10px; }
.ps-btn:hover { background:#3c3c3c; color:#fff; border-color:#666; }
.ps-warn { color:#ffb4b4; font-size:10px; padding:2px 0; }
.ps-diff { background:#18242e; border-left:2px solid #6ab0ff; padding:4px 6px;
  border-radius:3px; font-size:10px; font-family:ui-monospace,Consolas,monospace;
  white-space:pre-wrap; margin-top:5px; }
.ps-empty { color:#6d6d6d; text-align:center; padding:10px 6px; font-style:italic;
  border:1px dashed #3a3a3a; border-radius:4px; font-size:10px; }
.ps-actions { display:flex; gap:5px; margin-top:6px; }

/* Section headers spliced between native widgets. */
.ps-header { font-size:10px; font-weight:700; letter-spacing:.09em; padding:5px 8px;
  border-radius:5px; background:#141414; border:1px solid #444; border-left-width:3px;
  box-sizing:border-box; display:flex; justify-content:space-between; align-items:baseline;
  gap:8px; font-family:var(--font-family, sans-serif); }
.ps-header small { font-weight:400; letter-spacing:0; color:#7d7d7d; font-style:italic;
  font-size:9px; }

/* Native multiline textareas, restyled in place. ComfyUI keeps owning the value. */
textarea.ps-prompt { background:#0f0f0f !important; color:#eaeaea !important;
  border:1px solid #444 !important; border-left-width:3px !important;
  border-radius:5px !important; padding:7px 8px !important;
  font-family:ui-monospace,Consolas,monospace !important; font-size:11px !important;
  line-height:1.45 !important; caret-color:#6ab0ff; }
textarea.ps-prompt:focus { outline:none !important; box-shadow:0 0 0 1px currentColor inset; }
textarea.ps-prompt::placeholder { color:#5c5c5c; font-style:italic; }
`;

function injectStyle() {
  if (document.getElementById("ps-bin-style")) return;
  const el = document.createElement("style");
  el.id = "ps-bin-style";
  el.textContent = CSS;
  document.head.appendChild(el);
}

async function post(route, payload) {
  try {
    const res = await api.fetchApi(`/pulse_studio/${route}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await res.json();
  } catch (err) {
    console.error(`[PulseStudio] ${route} failed`, err);
    return { status: "error", message: String(err) };
  }
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file); // the endpoint takes any file type despite the name
  body.append("overwrite", "false");
  const res = await api.fetchApi("/upload/image", { method: "POST", body });
  if (res.status !== 200) throw new Error(`upload failed: ${res.status}`);
  const data = await res.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function kindForFile(name) {
  return EXT_KIND[(name.split(".").pop() || "").toLowerCase()] || "image";
}

/** URL for a file already uploaded into ComfyUI's input directory. */
function viewURL(path) {
  if (!path) return "";
  const parts = String(path).split(/[\\/]/);
  const filename = parts.pop();
  const subfolder = parts.join("/");
  return api.apiURL(
    `/view?filename=${encodeURIComponent(filename)}` +
    `&type=input&subfolder=${encodeURIComponent(subfolder)}` +
    `&rand=${Math.random().toString(36).slice(2, 8)}`);
}

/** A deterministic soundwave icon, drawn from the filename so it stays stable. */
function waveformSVG(seedText) {
  let h = 0;
  for (let i = 0; i < seedText.length; i++) h = (h * 31 + seedText.charCodeAt(i)) >>> 0;
  const bars = [];
  for (let i = 0; i < 14; i++) {
    h = (h * 1103515245 + 12345) >>> 0;
    const mag = 2 + ((h >>> 16) % 13); // 2..14
    const y = 17 - mag / 2;
    bars.push(`<rect x="${2 + i * 3}" y="${y.toFixed(1)}" width="1.6" ` +
              `height="${mag}" rx="0.8" fill="#d1743a"/>`);
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 34">` +
              `<rect width="44" height="34" fill="#141010"/>${bars.join("")}</svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

// ── node-face layout ────────────────────────────────────────────────────────

function textareaOf(widget) {
  const el = widget?.element ?? widget?.inputEl ?? null;
  if (!el) return null;
  if (el.tagName === "TEXTAREA") return el;
  return el.querySelector?.("textarea") ?? null;
}

const PROMPT_STYLE = {
  global_prompt: {
    accent: "#4a86c8",
    title: "GLOBAL PROMPT",
    hint: "style · identity · score",
    placeholder:
      "Art style, lighting, camera rules, identity locks, score.\n\n" +
      "style: shot on 35mm, warm practical light\n" +
      "identity: @Image1 keeps the red scarf in every shot\n" +
      "retention: the scarf is never removed\n" +
      "soundscape: rain outside, espresso hiss\n" +
      "music: sparse piano, no percussion",
  },
  shot_prompt: {
    accent: "#8b62c4",
    title: "SHOT TIMELINE",
    hint: "one shot per line",
    placeholder:
      "Timecoded shots. Reference assets by @name, never by number.\n\n" +
      "[Shot 1] @Image1 pushes through the cafe door\n" +
      "[Shot 2] At 00:04.000, she says \"you're still open\"\n" +
      "[Shot 3] At 00:08.000, she sits by the window",
  },
};

/** Style a native multiline widget in place. Its value stays ComfyUI's. */
function decoratePrompt(node, name) {
  const widget = node.widgets?.find((w) => w.name === name);
  const area = textareaOf(widget);
  const meta = PROMPT_STYLE[name];
  if (!area || !meta) return false;
  area.classList.add("ps-prompt");
  area.style.borderLeftColor = meta.accent;
  area.style.color = "#eaeaea";
  area.placeholder = meta.placeholder;
  area.spellcheck = false;
  area.title = `${meta.title} — ${meta.hint}`;
  // Never let the canvas swallow a click meant for the editor.
  if (!area.__psFocusWired) {
    for (const ev of ["mousedown", "pointerdown", "wheel", "keydown"]) {
      area.addEventListener(ev, (e) => e.stopPropagation());
    }
    area.__psFocusWired = true;
  }
  if (widget) widget.computeSize = () => [node.size?.[0] ?? 400, 132];
  return true;
}

/** Section labels, applied to the NATIVE widgets themselves.
 *
 * Deliberately not DOM header widgets. A widget inserted before a native one
 * takes a slot in the positional `widgets_values` array and shifts every value
 * after it -- see js/ps_widget_order.js. `widget.label` is free.
 */
const WIDGET_LABELS = {
  global_prompt: "◆ GLOBAL PROMPT — style · identity · score",
  shot_prompt: "◆ SHOT TIMELINE — one shot per line",
  duration_seconds: "▬ GENERATION CONTROLS — total duration (s)",
  timeline_data: "asset bin storage",
};

function applyLabels(node) {
  for (const [name, label] of Object.entries(WIDGET_LABELS)) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (widget) widget.label = label;
  }
}

// ── the Asset Bin panel ─────────────────────────────────────────────────────

class AssetBinPanel {
  constructor(node, storage) {
    this.node = node;
    this.storage = storage;
    this.root = document.createElement("div");
    this.root.className = "ps-wrap";
    this.dragId = null;
    this.pendingDiff = null;
    this.busy = false;
    this.render();
  }

  /** The ONLY widget assignment in this file, and only ever the storage widget
   *  with a string the server produced. */
  commit(timelineData) {
    if (typeof timelineData !== "string") return;
    this.storage.value = timelineData;
    this.node.graph?.setDirtyCanvas(true, true);
  }

  get raw() {
    return this.storage.value ?? "{}";
  }

  /** The node's own audio ceiling, sent with every request that judges a budget.
   *
   *  The server holds no session state, so it cannot know which node is asking
   *  or what that node's widget says. Reading the live widget here -- never
   *  writing to it -- is what makes the meter agree with what the render will
   *  actually accept. A missing widget (an older saved node, before the append)
   *  simply sends nothing and the server falls back to the documented 3.
   */
  get audioCeiling() {
    const widget = this.node.widgets?.find((w) => w.name === "audio_ref_ceiling");
    const value = Number(widget?.value);
    return Number.isFinite(value) && value > 0 ? value : undefined;
  }

  /** Request body shared by every route that evaluates the budget. */
  body(extra) {
    return { timeline_data: this.raw, audio_ref_ceiling: this.audioCeiling, ...extra };
  }

  async apply(operation, kwargs) {
    if (this.busy) return;
    this.busy = true;
    try {
      const res = await post("apply", this.body({ operation, kwargs }));
      if (res.status !== "ok") { alert(res.message || "Asset Bin backend error."); return; }
      if (res.error) { alert(res.error); return; }
      this.commit(res.timeline_data);
    } finally {
      this.busy = false;
      this.pendingDiff = null;
      await this.render();
    }
  }

  // ── rendering ─────────────────────────────────────────────────────────────

  async render() {
    let result;
    try {
      result = await post("bin_state", this.body());
    } catch (err) {
      result = { status: "error", message: String(err) };
    }
    this.root.replaceChildren();

    if (result.status !== "ok") {
      const err = document.createElement("div");
      err.className = "ps-warn";
      err.textContent = result.message || "Asset Bin backend unavailable.";
      this.root.appendChild(err);
      return;
    }

    const state = result.state;
    this.root.appendChild(this.buildMeter(state.budget));
    for (const problem of state.name_problems) {
      const warn = document.createElement("div");
      warn.className = "ps-warn";
      warn.textContent = `⚠ ${problem.problem}`;
      this.root.appendChild(warn);
    }

    for (const kind of ["image", "video", "audio"]) {
      this.root.appendChild(this.buildKindCard(kind, state.assets.filter((a) => a.kind === kind)));
    }

    if (this.pendingDiff) {
      const diff = document.createElement("div");
      diff.className = "ps-diff";
      diff.textContent = this.pendingDiff;
      this.root.appendChild(diff);
    }
  }

  buildMeter(budget) {
    const meter = document.createElement("div");
    // Over budget wins the colour: a bin that will not queue matters more than
    // one that is merely past the documented ceiling.
    meter.className = "ps-meter" + (budget.ok ? (budget.beyond_spec ? " ps-beyond" : "")
                                              : " ps-over");
    for (const part of budget.meter.split("|")) {
      const span = document.createElement("span");
      span.textContent = part.trim();
      meter.appendChild(span);
    }
    if (!budget.ok) {
      meter.title = budget.errors.join("\n");
    } else if (budget.beyond_spec) {
      meter.title =
        `audio_ref_ceiling is ${budget.max_audios}. MiniMax documents 3 standalone ` +
        `audio references and ComfyUI's socket declares 3; this render goes past ` +
        `both. It will run, but nothing about the result is covered by anything ` +
        `published.`;
    }
    return meter;
  }

  buildKindCard(kind, rows) {
    const meta = KIND_META[kind];
    const card = document.createElement("div");
    card.className = "ps-card";
    card.style.borderLeftColor = meta.accent;

    const head = document.createElement("div");
    head.className = "ps-card-head";
    const title = document.createElement("span");
    title.className = "ps-card-title";
    title.style.color = meta.accent;
    title.textContent = meta.label;
    const hint = document.createElement("span");
    hint.className = "ps-card-hint";
    hint.textContent = meta.hint;
    head.append(title, hint);
    card.appendChild(head);

    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "ps-empty";
      empty.textContent = `drop ${kind} files here`;
      card.appendChild(empty);
    } else {
      for (const row of rows) card.appendChild(this.buildRow(row));
    }

    const actions = document.createElement("div");
    actions.className = "ps-actions";
    const add = document.createElement("button");
    add.className = "ps-btn";
    add.textContent = `+ ${kind}`;
    add.addEventListener("click", () => this.pickFiles(kind));
    actions.appendChild(add);
    card.appendChild(actions);

    this.wireFileDrop(card);
    return card;
  }

  buildThumb(row) {
    const box = document.createElement("div");
    box.className = "ps-thumb-box";
    const url = viewURL(row.file);

    if (row.kind === "image") {
      const img = document.createElement("img");
      img.className = "ps-thumb";
      img.loading = "lazy";
      img.src = url;
      img.alt = row.name;
      img.addEventListener("error", () => {
        box.replaceChildren();
        box.textContent = "🖼";
        box.style.fontSize = "16px";
      });
      box.appendChild(img);
    } else if (row.kind === "video") {
      // A real keyframe: seek slightly in, because frame 0 is often black.
      const video = document.createElement("video");
      video.src = url;
      video.muted = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.addEventListener("loadedmetadata", () => {
        try { video.currentTime = Math.min(0.5, (video.duration || 1) * 0.25); } catch {}
      });
      video.addEventListener("error", () => {
        box.replaceChildren();
        box.textContent = "🎬";
        box.style.fontSize = "16px";
      });
      box.appendChild(video);
      const badge = document.createElement("span");
      badge.className = "ps-badge";
      badge.textContent = "VID";
      box.appendChild(badge);
    } else {
      const img = document.createElement("img");
      img.className = "ps-thumb";
      img.src = waveformSVG(row.file || row.name || row.id);
      img.alt = "waveform";
      box.appendChild(img);
      const badge = document.createElement("span");
      badge.className = "ps-badge";
      badge.textContent = "AUD";
      box.appendChild(badge);
    }
    box.title = row.file || "";
    return box;
  }

  buildRow(row) {
    const el = document.createElement("div");
    el.className = "ps-row";
    el.draggable = true;
    el.dataset.id = row.id;

    el.appendChild(this.buildThumb(row));

    const name = document.createElement("input");
    name.className = "ps-name";
    name.value = row.name;
    name.title = row.file || "";
    name.addEventListener("change", () =>
      this.apply("rename", { asset_id: row.id, name: name.value }));
    for (const ev of ["mousedown", "pointerdown", "keydown"]) {
      name.addEventListener(ev, (e) => e.stopPropagation());
    }
    el.appendChild(name);

    const tag = document.createElement("span");
    tag.className = "ps-tag";
    tag.textContent = row.tag || "";
    tag.title = `Assigned from bin order. Write @${row.name} in a prompt, never the number.`;
    el.appendChild(tag);

    const remove = document.createElement("button");
    remove.className = "ps-btn";
    remove.textContent = "✕";
    remove.title = "Remove from bin";
    remove.addEventListener("click", () => this.apply("remove", { asset_id: row.id }));
    el.appendChild(remove);

    if (row.kind === "video") {
      const sub = document.createElement("label");
      sub.className = "ps-sub";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = !!row.include_audio;
      box.addEventListener("change", () =>
        this.apply("set_include_audio", { asset_id: row.id, include: box.checked }));
      sub.appendChild(box);
      sub.appendChild(document.createTextNode(
        row.include_audio ? `use its soundtrack → ${row.soundtrack_tag}`
                          : "use its soundtrack"));
      el.appendChild(sub);
    }

    this.wireReorder(el, row.id);
    return el;
  }

  // ── reorder, with a live renumber preview ─────────────────────────────────

  wireReorder(el, id) {
    el.addEventListener("dragstart", (e) => {
      this.dragId = id;
      el.classList.add("ps-drag");
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", id);
      e.stopPropagation();
    });
    el.addEventListener("dragend", () => {
      el.classList.remove("ps-drag");
      this.dragId = null;
    });
    el.addEventListener("dragover", async (e) => {
      if (!this.dragId || this.dragId === id) return;
      e.preventDefault();
      e.stopPropagation();
      el.classList.add("ps-over-drop");
      const diff = await this.previewMove(this.dragId, id);
      if (diff && diff !== this.pendingDiff) this.pendingDiff = diff;
    });
    el.addEventListener("dragleave", () => el.classList.remove("ps-over-drop"));
    el.addEventListener("drop", async (e) => {
      if (!this.dragId || this.dragId === id) return;
      e.preventDefault();
      e.stopPropagation();
      el.classList.remove("ps-over-drop");
      const index = await this.indexOf(id);
      if (index >= 0) await this.apply("move", { asset_id: this.dragId, new_index: index });
    });
  }

  async indexOf(assetId) {
    const res = await post("bin_state", this.body());
    if (res.status !== "ok") return -1;
    return res.state.assets.findIndex((a) => a.id === assetId);
  }

  async previewMove(assetId, targetId) {
    const index = await this.indexOf(targetId);
    if (index < 0) return "";
    const res = await post("preview_change", this.body({
      operation: "move",
      kwargs: { asset_id: assetId, new_index: index },
    }));
    if (res.status !== "ok" || res.error) return res.error || "";
    if (!res.deltas?.length) return "";
    return "this reorder renumbers:\n" + res.deltas
      .map((d) => `  ${d.name}: ${d.before || "—"} → ${d.after || "removed"}`)
      .join("\n");
  }

  // ── adding files ──────────────────────────────────────────────────────────

  pickFiles(kind) {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = kind
      ? Object.keys(EXT_KIND).filter((e) => EXT_KIND[e] === kind).map((e) => "." + e).join(",")
      : ACCEPT;
    input.addEventListener("change", () => this.addFiles([...input.files]));
    input.click();
  }

  async addFiles(files) {
    for (const file of files) {
      try {
        const kind = kindForFile(file.name);
        const path = await uploadFile(file);
        const alias = await post("suggest_name", { timeline_data: this.raw, kind });
        await this.apply("add", {
          asset: {
            id: `a${Date.now().toString(36)}${Math.floor(Math.random() * 1e4).toString(36)}`,
            kind,
            name: alias.name || kind,
            file: path,
          },
        });
      } catch (err) {
        alert(`Could not add ${file.name}: ${err}`);
      }
    }
  }

  wireFileDrop(element) {
    const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
    element.addEventListener("dragover", (e) => {
      if (!hasFiles(e) || this.dragId) return;
      e.preventDefault();
      e.stopPropagation();
      e.dataTransfer.dropEffect = "copy";
      element.classList.add("ps-filedrop");
    });
    element.addEventListener("dragleave", (e) => {
      if (e.target === element) element.classList.remove("ps-filedrop");
    });
    element.addEventListener("drop", (e) => {
      element.classList.remove("ps-filedrop");
      if (!hasFiles(e) || this.dragId) return;
      e.preventDefault();
      e.stopPropagation();
      this.addFiles([...e.dataTransfer.files]);
    });
  }
}

// ── extension registration ──────────────────────────────────────────────────

/** Take a widget off the node face without taking it out of node.widgets.
 *  The slot must survive -- it is what the saved array is indexed by. */
function hideWidget(node, name) {
  const widget = node.widgets?.find((w) => w.name === name);
  if (!widget) return null;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
  return widget;
}

function buildFace(node) {
  const storage = node.widgets?.find((w) => w.name === STORAGE_WIDGET);
  if (!storage) return;

  // Widget indices 0 and 1. Hidden on the face, present in every save.
  hideWidget(node, SCHEMA_WIDGET);
  hideWidget(node, STORAGE_WIDGET);

  for (const name of PROMPT_WIDGETS) {
    protectWidget(node.widgets?.find((w) => w.name === name), name);
  }
  for (const name of PROMPT_WIDGETS) decoratePrompt(node, name);
  applyLabels(node);

  if (!node.psAssetBin) {
    const panel = new AssetBinPanel(node, storage);
    // APPENDED, never inserted. addDOMWidget pushes onto node.widgets, and
    // LiteGraph serialises widgets_values positionally over that same list, so a
    // widget placed before a native one shifts every value after it. Trailing
    // slots are harmless; leading ones corrupt the whole node.
    const widget = node.addDOMWidget("ps_asset_bin", "div", panel.root, {
      serialize: false, hideOnZoom: false,
    });
    widget.computeSize = () => [node.size?.[0] ?? 460, 430];
    node.psAssetBin = panel;
    node.psBinWidget = widget;

    // Both invariants, checked at construction rather than merely intended:
    // nothing custom ahead of a native widget, and the live native names still
    // equal the table the loader reads workflows against.
    const order = checkWidgetOrder(node.widgets, undefined, NODE_ID);
    if (order.offenders.length) {
      console.error(
        `[PulseStudio] widget order is unsafe: ${order.offenders.join(", ")} follow the ` +
        `custom widget "${order.firstCustom}". Their saved values would be shifted. ` +
        `Custom widgets must be appended after every native one.`);
    }
    if (order.nameError) {
      console.error(`[PulseStudio] ${order.nameError}\n  ` +
        `Workflows will load values into the wrong widgets until ` +
        `js/ps_widget_order.js is updated to match INPUT_TYPES.`);
    }

    node.size = [Math.max(node.size?.[0] ?? 0, 480), Math.max(node.size?.[1] ?? 0, 1020)];
  } else {
    node.psAssetBin.storage = storage;
    node.psAssetBin.render();
  }
}

/**
 * Restore a saved workflow into this node BY NAME. Spec §3.3.
 *
 * LiteGraph has already done its positional assignment by the time onConfigure
 * runs; this overwrites that with the name-based result, which is the only one
 * that stays correct once the widget list grows. A file whose slot 0 is not a
 * known schema version is refused outright and marked on the node -- guessing
 * at its layout is what produced `duration_seconds = 'res_multistep'`.
 */
function restoreByName(node, nodeId, info) {
  const values = info?.widgets_values;
  if (!Array.isArray(values)) return;

  const result = applySavedValues(node, nodeId, values);
  if (result.ok) {
    if (result.defaulted.length) {
      console.info(`[PulseStudio] ${nodeId}: widgets absent from this ` +
                   `(schema ${result.version}) file took their defaults: ` +
                   `${result.defaulted.join(", ")}`);
    }
    return;
  }

  // Unloadable. Say so on the node face, not only in the console: the graph
  // will otherwise fail validation with a pile of confusing type errors.
  console.error(describeUnloadable(nodeId, result.reason));
  node.has_errors = true;

  // The pre-2.0.0 corruption has its own explanation, which names the widgets
  // that received the wrong kind of value. Worth printing when it applies.
  const report = validateWidgetValues(values);
  if (!report.ok && !report.unreadable) console.error(describeMisalignment(report));
}

/**
 * The slot contract, installed on every node in the pack.
 *
 * Separate from the Asset Bin extension because all three nodes are frozen
 * under §3 but only PulseSlate has a bin. `onSerialize` stamps the schema
 * version the file is being written in; `onConfigure` reads it back and
 * restores by name. Both are wrapped -- a throw in either aborts the whole
 * "Loading workflow data" step for the user's entire graph, not just this node.
 */
app.registerExtension({
  name: "comfyui_pulse_studio.slot_contract",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_IDS.includes(nodeData.name)) return;
    const nodeId = nodeData.name;

    const onSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (info) {
      const result = onSerialize?.apply(this, arguments);
      try {
        // Stamp the version this build writes, in the widget AND in the array
        // that has already been captured, so the two can never disagree.
        const widget = this.widgets?.find((w) => w.name === SCHEMA_WIDGET);
        if (widget) widget.value = CURRENT;
        if (Array.isArray(info?.widgets_values) && info.widgets_values.length) {
          info.widgets_values[0] = CURRENT;
        }
      } catch (err) {
        console.warn(`[PulseStudio] ${nodeId}: could not stamp schema version:`, err);
      }
      return result;
    };

    const onConfigureContract = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = onConfigureContract?.apply(this, arguments);
      try {
        restoreByName(this, nodeId, info);
      } catch (err) {
        console.warn(`[PulseStudio] ${nodeId}: name-based restore failed:`, err);
      }
      return result;
    };
  },
});

app.registerExtension({
  name: "comfyui_pulse_studio.asset_bin",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) return;
    injectStyle();

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      // Widgets are not all present until the frame settles on some frontends.
      try { buildFace(this); } catch (err) {
        console.warn("[PulseStudio] node face setup failed:", err);
      }
      setTimeout(() => {
        try { buildFace(this); } catch (err) {
          console.warn("[PulseStudio] deferred node face setup failed:", err);
        }
      }, 0);
    };

    // Files dropped on the node body go to the bin -- no loader nodes needed.
    const onDragOver = nodeType.prototype.onDragOver;
    nodeType.prototype.onDragOver = function (e) {
      if ([...(e?.dataTransfer?.types || [])].includes("Files")) return true;
      return onDragOver?.apply(this, arguments) ?? false;
    };

    const onDragDrop = nodeType.prototype.onDragDrop;
    nodeType.prototype.onDragDrop = function (e) {
      try {
        const files = [...(e?.dataTransfer?.files || [])];
        if (files.length && this.psAssetBin) {
          this.psAssetBin.addFiles(files);
          return true;
        }
      } catch (err) {
        console.warn("[PulseStudio] drop failed:", err);
      }
      return onDragDrop?.apply(this, arguments) ?? false;
    };

    // Re-render after a workflow load, when timeline_data arrives populated.
    // The name-based restore already ran in the slot_contract extension, which
    // registers first; this only rebuilds the face from the restored values.
    // Wrapped: a throw here is what aborts "Loading workflow data".
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = onConfigure?.apply(this, arguments);
      try {
        buildFace(this);
      } catch (err) {
        console.warn("[PulseStudio] re-init after workflow load failed:", err);
      }
      return result;
    };
  },
});
