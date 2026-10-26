#!/usr/bin/env python3
"""Build the Oct-2026 Pulse Studio workflows: 30 s and 60 s, five families.

    python3 tools/build_oct2026_workflows.py <out_dir> [media.json]

Writes ten ComfyUI UI-format graphs:

  PS_FastH3Omni_Dialogue_{30s,60s}        FastH3-Omni 8-step (ref2va, VSA) two-hander
  PS_FastH3Omni_ContinuousTake_{30s,60s}  FastH3-Omni 8-step, one speaker, 9:16
  PS_HyperFlow_Dialogue_{30s,60s}         HyperFlow 8-step two-hander
  PS_HyperFlow_ContinuousTake_{30s,60s}   HyperFlow 8-step, one speaker
  PS_TaoMate_Vertical_{30s,60s}           9:16 start/end-keyframe series on TaoMate

One checkpoint per graph. A film whose first window is a reference checkpoint
and whose later windows are a First/Last-Frame one sees the faces and voices
only in window 1 -- fl2va takes no references -- and the two halves contradict
each other. So each family renders every window on one model.

The faces and the recordings live in Pulse Slate's Asset Bin (the IMAGE / AUDIO
REFERENCES panel at the bottom of the node), not in Load Image, Load Audio,
Pulse Character and Pulse Voice nodes wired around it: one place to attach a
reference. Each line is a bin row of its own -- a take file, whose voice it is,
the film second it starts at and the seconds of the file it plays.

The two dialogue families share one script and one set of takes, and so do the
two continuous-take families, so each pair is an A/B of the engines and nothing
else.

Deterministic: shot ids are uuid5s of the workflow and shot index, so rebuilding
writes byte-identical files. The bins are written by the pack's own Asset and
AssetBin, so a bin the node would refuse cannot be written. Stock-node widget
layouts are copied from graphs already run on the target ComfyUI (the
FastH3-Omni R2V template, a TaoMate FL2VA 3-step graph, HyperFlow dialogue
graphs); Pulse
layouts follow js/ps_widget_order.js, which tests/test_js_guard.py keeps in step
with nodes.py.

Media is named, not shipped: every file a graph names must be in ComfyUI/input.
The names in MEDIA below are placeholders; a JSON file with the same keys names
your own files instead, so they never have to be written into this script. Or
replace the takes and pictures in the Asset Bin afterwards -- the graphs are built
so that is the only edit needed.
"""

import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comfyui_pulse_studio.assets import Asset, AssetBin  # noqa: E402

SCHEMA = "3.0.0"
EMPTY_BIN = '{"schema": 2, "assets": [], "cast": []}'

# ── models (paths as the Windows ComfyUI lists them) ────────────────────────
FASTH3 = "minimax\\FastVideoH3\\fasth3_omniref_8step_pruned_int8_convrot.safetensors"
REF2VA_FULL = "minimax\\higher\\minimax_h3_ref2va_int8_convrot.safetensors"
FL2VA_FULL = "minimax\\higher\\minimax_h3_fl2va_int8_convrot.safetensors"
TAOMATE = "minimaxh3\\TaoMate-H3-FL2VA-3Step-ComfyUI-BF16.safetensors"
HYPERFLOW = "minimaxh3\\minimax_h3_hyperflow_8step_v1.0.safetensors"
TEXT_ENCODER = "minimax\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax\\minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax\\minimax_h3_audio_vae_fp32.safetensors"

# ── media already in ComfyUI/input ──────────────────────────────────────────
#: What each graph names. Override any of them with `media.json` (same keys).
MEDIA = {
    "face_a": "example_pulse_subject_a.png",  # ships with the pack, copied to input/
    "face_b": "example_pulse_subject_b.png",
    "take_a": "take_a.wav",          # @Ada: two lines back to back, up to 7.14 s each
    "take_b": "take_b.wav",          # @Ben: the same
    "narration": "narration.wav",    # one speaker, about 30 s
    "narration_seconds": 30.0,       # its length: where the 60 s graph's second copy starts
    "key_1": "keyframe_1.png",       # the vertical series alternates two keyframes
    "key_2": "keyframe_2.png",
}

GRID_15 = 362 / 24.0  # 15.0833 s: one full H3 window, on the frame grid
# 181 frames: two of these shots are exactly one 362-frame window, so a window
# seam can land on every second cut -- on the 17k+5 grid, an arbitrary 7.5 s shot
# puts no cut where a seam can reach, and every seam then falls inside a shot.
SHOT = 181 / 24.0     # 7.5417 s
# A take trimmed to this, plus the 0.4 s of air a duration-0 shot adds after it
# (constants.AUTO_SHOT_PAD_SECONDS), is a SHOT long.
LINE = SHOT - 0.4     # 7.1417 s


# ── a minimal graph builder ─────────────────────────────────────────────────

class Graph:
    """Nodes, links and groups in ComfyUI's UI format (version 0.4)."""

    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.links = []
        self.groups = []

    def node(self, type_, widgets, inputs=(), outputs=(), pos=(0, 0), size=(360, 120),
             title=None, color=None):
        node = {
            "id": len(self.nodes) + 1,
            "type": type_,
            "pos": [int(pos[0]), int(pos[1])],
            "size": [int(size[0]), int(size[1])],
            "flags": {},
            "order": len(self.nodes),
            "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
            "outputs": [{"name": n, "type": t, "links": []} for n, t in outputs],
            "properties": {"Node name for S&R": type_},
            "widgets_values": list(widgets),
        }
        if title:
            node["title"] = title
        if color:
            node["color"], node["bgcolor"] = color
        self.nodes.append(node)
        return node

    def link(self, src, src_slot, dst, dst_input):
        """Wire output `src_slot` (index or name) into `dst`'s input named `dst_input`."""
        if isinstance(src_slot, str):
            src_slot = next(i for i, o in enumerate(src["outputs"]) if o["name"] == src_slot)
        slot = next(i for i, inp in enumerate(dst["inputs"]) if inp["name"] == dst_input)
        link_id = len(self.links) + 1
        link_type = src["outputs"][src_slot]["type"]
        self.links.append([link_id, src["id"], src_slot, dst["id"], slot, link_type])
        dst["inputs"][slot]["link"] = link_id
        src["outputs"][src_slot]["links"].append(link_id)
        return link_id

    def group(self, title, x, y, w, h, color="#3f789e"):
        self.groups.append({"id": len(self.groups) + 1, "title": title,
                            "bounding": [int(x), int(y), int(w), int(h)],
                            "color": color, "font_size": 24, "flags": {}})

    def to_json(self):
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "pulse-studio/oct2026/" + self.name)),
            "revision": 0,
            "last_node_id": len(self.nodes),
            "last_link_id": len(self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": 0.55, "offset": [40, 40]}},
            "version": 0.4,
        }


# ── stock nodes, laid out exactly as the tested graphs saved them ───────────

def unet(g, name, pos, title):
    return g.node("UNETLoader", [name, "default"], outputs=[("MODEL", "MODEL")],
                  pos=pos, size=(460, 82), title=title)


def sigma_shift(g, pos):
    return g.node("MiniMaxH3SigmaShift", [12, 3], inputs=[("model", "MODEL")],
                  outputs=[("MODEL", "MODEL")], pos=pos, size=(300, 82),
                  title="Shift 12 / 3 (every H3 recipe here)")


def attention_backend(g, pos):
    return g.node("ModelAttentionBackend", ["comfy kitchen attention"],
                  inputs=[("model", "MODEL")], outputs=[("model", "MODEL")], pos=pos,
                  size=(320, 60), title="Attention backend (dense fallback)")


def vsa(g, pos):
    return g.node("BlockSparseAttention",
                  ["vsa", 10, 0, 1, "", 12288, 256, "exact_kv_and_rows", False],
                  inputs=[("model", "MODEL")], outputs=[("model", "MODEL")], pos=pos,
                  size=(340, 250), title="VSA keep 10 -- FastH3-Omni is trained for it")


def sol_attn(g, pos):
    return g.node("BlockSparseAttention",
                  ["sol-attn", 1.3, 0, 1, "", 12288, 256, "exact_kv_and_rows", False],
                  inputs=[("model", "MODEL")], outputs=[("model", "MODEL")], pos=pos,
                  size=(340, 250), title="sol-attn tau 1.3 (as in the tested TaoMate graph)")


def lora_only(g, name, pos, title):
    return g.node("LoraLoaderModelOnly", [name, 1.0], inputs=[("model", "MODEL")],
                  outputs=[("MODEL", "MODEL")], pos=pos, size=(460, 82), title=title)


def hyperflow(g, pos, title):
    return g.node("HyperFlowLoRALoader", [HYPERFLOW, 1.0, "auto"], inputs=[("model", "MODEL")],
                  outputs=[("MODEL", "MODEL")], pos=pos, size=(460, 106), title=title)


def loaders(g, x, y):
    clip = g.node("CLIPLoader", [TEXT_ENCODER, "minimax", "default"],
                  outputs=[("CLIP", "CLIP")], pos=(x, y), size=(460, 106),
                  title="Qwen3-VL text encoder (nvfp4)")
    video_vae = g.node("VAELoader", [VIDEO_VAE], outputs=[("VAE", "VAE")],
                       pos=(x, y + 130), size=(460, 58), title="video VAE")
    audio_vae = g.node("VAELoader", [AUDIO_VAE], outputs=[("VAE", "VAE")],
                       pos=(x, y + 210), size=(460, 58), title="audio VAE")
    return clip, video_vae, audio_vae


def load_image(g, name, pos, title):
    return g.node("LoadImage", [name, "image"],
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], pos=pos, size=(320, 340),
                  title=title)


def note(g, text, pos, size=(560, 760), title="READ FIRST"):
    return g.node("MarkdownNote", [text], pos=pos, size=size, title=title,
                  color=("#432", "#653"))


# ── Pulse nodes, widget order per js/ps_widget_order.js ─────────────────────

def recipe(g, preset, pos, title):
    return g.node("PulseRecipe", [SCHEMA, preset, 20, "euler", "simple", 1.0, ""],
                  outputs=[("recipe", "PULSE_RECIPE")], pos=pos, size=(340, 200), title=title)


def face(asset_id, name, file, description):
    """An Asset Bin picture that defines a character: described, so it compiles to a
    <Subject N>, and fully preserved."""
    return Asset(asset_id, "image", name=name, file=file, description=description,
                 retention="fully_preserved")


def line(asset_id, name, file, voice_of, at, trim_start=0.0, trim_end=None):
    """An Asset Bin recording the mouth follows: `voice_of` the face it belongs to,
    starting at film second `at`, playing [trim_start, trim_end) of the file."""
    return Asset(asset_id, "audio", name=name, file=file, audio_role="lip_sync",
                 voice_of=voice_of, audio_offset=at, trim_start=trim_start,
                 trim_end=trim_end)


def bin_data(assets):
    """timeline_data holding these assets. Built through AssetBin, which refuses
    a bin over H3's budget exactly as Pulse Slate would at compile time."""
    bin_ = AssetBin(assets)
    return json.dumps({"schema": 2, "assets": bin_.to_list(), "cast": []},
                      ensure_ascii=False)


def shot(g, workflow, index, label, visual, audio_line, duration, speaker, pos):
    shot_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                             "pulse-studio/oct2026/%s/shot/%d" % (workflow, index)))
    return g.node("PulseShot",
                  [SCHEMA, shot_id, label, visual, audio_line, float(duration), "inherit",
                   "lip_sync", speaker],
                  inputs=[("start_image", "IMAGE"), ("end_image", "IMAGE"),
                          ("ref_audio", "AUDIO"), ("voice", "PULSE_VOICE"),
                          ("refs.ref_image_1", "IMAGE")],
                  outputs=[("shot", "PULSE_SHOT")], pos=pos, size=(420, 330),
                  title="Shot %d -- %s" % (index, label))


def slate(g, pos, global_prompt, duration, aspect, width, height, steps, sampler, scheduler,
          seed, continuity, n_shots, timeline_data=EMPTY_BIN):
    """PulseSlate, its sockets in the order js/ps_sockets.js rebuilds them to.

    References go in `timeline_data` -- the Asset Bin -- and nowhere else: every
    reference socket on the node is left free, one per group.
    """
    inputs = [("model", "MODEL"), ("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"),
              ("model_fl2va", "MODEL"), ("ref_video", "IMAGE"), ("ref_video_audio", "AUDIO"),
              ("ref_music", "AUDIO"), ("recipe", "PULSE_RECIPE"),
              ("recipe_fl2va", "PULSE_RECIPE"), ("refs.ref_image_1", "IMAGE")]
    inputs += [("shots.shot_%d" % i, "PULSE_SHOT") for i in range(1, n_shots + 2)]
    inputs += [("voices.voice_1", "PULSE_VOICE"), ("cast.cast_1", "PULSE_CAST")]
    widgets = [SCHEMA, timeline_data, global_prompt, "", float(duration), aspect, width,
               height, steps, sampler, scheduler, 1.0, seed, "fixed", "speech_aware", 15.0,
               "crop", "image", False, 4.0, "match", 12.0, 3.0, 3, continuity, ""]
    return g.node("PulseSlate", widgets, inputs=inputs,
                  outputs=[("timeline", "PULSE_TIMELINE"), ("positive", "CONDITIONING"),
                           ("latent", "LATENT"), ("combined_audio", "AUDIO"),
                           ("images", "IMAGE"), ("compiled_prompt", "STRING")],
                  pos=pos, size=(760, 1700), title="Pulse Slate -- references in its Asset Bin")


def render(g, pos, run_dir, use_reference_audio=True, audio_mode="lock_source",
           upscale="1080p_fit", seam="audio+colour"):
    return g.node("PulseRender",
                  [SCHEMA, "auto", run_dir, "", True, True, False, False,
                   bool(use_reference_audio), seam, audio_mode, 0.35, "keep_aspect", upscale],
                  inputs=[("timeline", "PULSE_TIMELINE"), ("model", "MODEL"), ("vae", "VAE"),
                          ("audio_vae", "VAE"), ("model_fl2va", "MODEL")],
                  outputs=[("video", "VIDEO"), ("frames", "IMAGE"), ("audio", "AUDIO"),
                           ("segment_paths", "STRING"), ("report", "STRING")],
                  pos=pos, size=(440, 420), title="Pulse Render")


def outputs(g, slate_node, render_node, prefix, x, y):
    save = g.node("SaveVideo", ["video/" + prefix, "auto", "auto", "auto"],
                  inputs=[("video", "VIDEO")], outputs=[("video", "VIDEO")], pos=(x, y),
                  size=(700, 620), title="Save -- the 1080 delivery file")
    g.link(render_node, "video", save, "video")
    report = g.node("PreviewAny", [], inputs=[("source", "*")],
                    outputs=[("STRING", "STRING")], pos=(x, y + 660), size=(700, 420),
                    title="Render report (lip sync per window, seams, recipes)")
    g.link(render_node, "report", report, "source")
    prompt = g.node("PreviewAny", [], inputs=[("source", "*")],
                    outputs=[("STRING", "STRING")], pos=(x + 740, y), size=(700, 1080),
                    title="Compiled prompt (read it before a long render)")
    g.link(slate_node, "compiled_prompt", prompt, "source")


def shared_wiring(g, slate_node, render_node, clip, video_vae, audio_vae):
    g.link(clip, "CLIP", slate_node, "clip")
    g.link(video_vae, "VAE", slate_node, "vae")
    g.link(audio_vae, "VAE", slate_node, "audio_vae")
    g.link(slate_node, "timeline", render_node, "timeline")
    g.link(video_vae, "VAE", render_node, "vae")
    g.link(audio_vae, "VAE", render_node, "audio_vae")


# ── the scripts ─────────────────────────────────────────────────────────────

DIALOGUE_GLOBAL = (
    "style: Photoreal live-action, cinematic 35mm look, warm practical lighting in a modern "
    "studio, shallow depth of field, locked-off camera, soft film grain.\n"
    "identity: @Ada stays on the left and @Ben on the right in every shot, with the same "
    "faces, hair and clothing throughout. No other people.\n"
    "soundscape: Quiet room tone under the dialogue; no third voice and no music under "
    "the lines.\n"
    "music: None.\n"
    "language: English")

ADA = ("a woman in her thirties with shoulder-length dark hair, a red wool coat over a "
       "cream sweater, small gold earrings")
BEN = ("a man in his forties with a short grey beard, round glasses and a navy "
       "corduroy jacket over a white shirt")

TWO_SHOT = ("Locked-off medium two-shot at eye level across a small wooden table in a warm "
            "studio: @Ada on the left third facing right, @Ben on the right third facing "
            "left, both faces clearly visible, lit and in focus.")

# (label, speaker, visual, MEDIA key of the take file, trim_start, trim_end)
DIALOGUE = [
    ("Ada opens", "@Ada", TWO_SHOT + " @Ada leans in and speaks; @Ben listens.",
     "take_a", 0.0, LINE),
    ("Ben answers", "@Ben", TWO_SHOT + " @Ben answers with a small open-hand gesture; "
     "@Ada watches him.", "take_b", 0.0, LINE),
    ("Ada presses", "@Ada", "Medium close-up on @Ada, @Ben's shoulder soft in the right "
     "foreground. @Ada asks the next question, eyebrows raised, hands folded on the table.",
     "take_a", LINE, 2 * LINE),
    ("Ben's reply", "@Ben", "Medium close-up on @Ben, @Ada's shoulder soft in the left "
     "foreground. @Ben thinks for a beat, then replies with a warm half-smile.",
     "take_b", LINE, 2 * LINE),
    ("Ada lands it", "@Ada", TWO_SHOT + " @Ada makes her final point, tapping the table "
     "once; @Ben nods slowly.", "take_a", LINE, 2 * LINE),
    ("Ben concedes", "@Ben", TWO_SHOT + " @Ben laughs quietly and concedes the point, "
     "opening his hands; @Ada smiles.", "take_b", LINE, 2 * LINE),
    ("Ada again", "@Ada", "Medium close-up on @Ada, warm light on her face. @Ada picks up "
     "the thread again, animated.", "take_a", 0.0, LINE),
    ("Ben's last word", "@Ben", "Medium close-up on @Ben. @Ben gives the last word and "
     "leans back in his chair, satisfied.", "take_b", 0.0, LINE),
]

TAKE_GLOBAL = (
    "style: Photoreal live-action, golden-hour light, smooth gimbal follow shot at walking "
    "pace, shallow depth of field, gentle film grain. One continuous take: no cuts, no "
    "jumps.\n"
    "identity: @Ada is the only person who speaks; same face, hair and clothing throughout.\n"
    "soundscape: Soft street ambience under her voice -- distant traffic, footsteps, "
    "birdsong. No music under the lines.\n"
    "music: None.\n"
    "language: English")

TAKE_SHOTS = [
    ("Street, walking", "Medium shot, the camera leading @Ada as she walks slowly along a "
     "sunlit market street, talking to camera, relaxed and smiling."),
    ("Market stall", "@Ada slows beside a fruit stall, still talking to camera, and picks "
     "up an orange as she makes her point."),
    ("Crossing", "@Ada puts the orange back and keeps walking, gesturing with one hand, "
     "eyes on the lens."),
    ("Fountain", "@Ada reaches a small square with a fountain and turns to face the "
     "camera fully, finishing her thought."),
    ("Bench", "@Ada sits on the edge of the fountain, the camera settling to her eye "
     "level, and carries on talking."),
    ("Looking away", "@Ada glances at the water, then back to the lens with a small "
     "laugh, still talking."),
    ("Standing", "@Ada stands and starts walking again, the camera leading her out of the "
     "square, talking."),
    ("Sign-off", "@Ada stops at the corner, smiles to camera and gives her closing line."),
]

VERTICAL_SHOTS = [
    ("Keyframe 1 -> 2", "Vertical 9:16. The scene in the opening keyframe comes alive: the "
     "character breathes, blinks and turns slowly toward the window light while the camera "
     "pushes in gently. The shot ends exactly on the framing of the next keyframe."),
    ("Keyframe 2 -> 3", "The character lifts their gaze and steps closer to the light; dust "
     "motes drift; the camera eases sideways. The shot ends on the next keyframe's framing."),
    ("Keyframe 3 -> 4", "A slow, calm beat: the character smooths their clothing and looks "
     "back over one shoulder. Gentle push-in, no fast motion."),
    ("Keyframe 4", "The character turns to face the camera and holds a quiet, steady look as "
     "the light warms. Slow push-in to a close-up."),
]


# ── the five families ───────────────────────────────────────────────────────

def _dialogue_bin(n, media):
    """The two faces, then one lip-sync row per shot: shot i's line placed at shot i."""
    assets = [face("ada", "Ada", media["face_a"], ADA),
              face("ben", "Ben", media["face_b"], BEN)]
    for i, (_, who, _, take, a, b) in enumerate(DIALOGUE[:n], 1):
        owner = who[1:].lower()
        assets.append(line("%s_line_%d" % (owner, i), "%sLine%d" % (who[1:], i), media[take],
                           owner, (i - 1) * SHOT, a, None if b < 0 else b))
    return bin_data(assets)


def _dialogue(name, engine, seconds, media):
    """Families 1 and 3: one two-hander script, each line a row in the Asset Bin."""
    g = Graph(name)
    n = 4 if seconds == 30 else 8
    clip, video_vae, audio_vae = loaders(g, 0, 620)
    if engine == "fasth3":
        base = unet(g, FASTH3, (0, 0), "FastH3-Omni 8-step (pruned int8, ref2va)")
        shift = sigma_shift(g, (0, 110))
        backend = attention_backend(g, (0, 220))
        sparse = vsa(g, (0, 310))
        g.link(base, "MODEL", shift, "model")
        g.link(shift, "MODEL", backend, "model")
        g.link(backend, "model", sparse, "model")
        model, model_slot = sparse, "model"
        chosen = recipe(g, "fasth3_omni_8step", (0, 900), "Recipe -- FastH3-Omni 8-step")
        steps, sampler, scheduler = 8, "euler", "simple"
    else:
        base = unet(g, REF2VA_FULL, (0, 0), "H3 ref2va (non-pruned int8) -- HyperFlow needs it")
        adapter = hyperflow(g, (0, 110), "HyperFlow 8-step adapter, strength 1.0")
        shift = sigma_shift(g, (0, 250))
        g.link(base, "MODEL", adapter, "model")
        g.link(adapter, "MODEL", shift, "model")
        model, model_slot = shift, "MODEL"
        chosen = recipe(g, "hyperflow_8step", (0, 900), "Recipe -- HyperFlow 8-step")
        steps, sampler, scheduler = 8, "euler", "hyperflow"
    g.group("1 · Model -- one checkpoint for every window", -20, -60, 500, 1180)

    shots = []
    for i, (label, who, visual, _, _, _) in enumerate(DIALOGUE[:n], 1):
        col, row = divmod(i - 1, 4)
        other = "@Ben" if who == "@Ada" else "@Ada"
        s = shot(g, name, i, label, visual, "%s speaks the line; %s listens." % (who, other),
                 SHOT, who, (520 + col * 460, row * 360))
        shots.append(s)
    g.group("2 · Shots -- 7.54 s each; who speaks is the shot's speaker", 500, -60,
            460 * ((n + 3) // 4) + 20, 1480)

    x = 520 + 460 * ((n + 3) // 4) + 60
    sl = slate(g, (x, 0), DIALOGUE_GLOBAL, SHOT * n, "16:9 landscape", 1344, 768, steps,
               sampler, scheduler, 20261020, "none", n_shots=n,
               timeline_data=_dialogue_bin(n, media))
    g.link(model, model_slot, sl, "model")
    g.link(chosen, "recipe", sl, "recipe")
    for i, s in enumerate(shots, 1):
        g.link(s, "shot", sl, "shots.shot_%d" % i)
    rn = render(g, (x + 800, 0), "ps_%s_dialogue_%ds" % (engine, seconds))
    g.link(model, model_slot, rn, "model")
    shared_wiring(g, sl, rn, clip, video_vae, audio_vae)
    g.group("3 · Slate (faces and lines in its Asset Bin) and render", x - 20, -60, 1280, 1780)
    outputs(g, sl, rn, name, x + 1300, 0)

    engine_text = {
        "fasth3": ("**FastH3-Omni 8-step** (ref2va, pruned int8 + VSA keep 10). Every window "
                   "is a reference window, so both faces and both voices are referenced in "
                   "every one. Sampled with the `fasth3_omni_8step` recipe: euler, simple, 8 "
                   "steps, cfg 1."),
        "hyperflow": ("**HyperFlow 8-step** on the non-pruned ref2va checkpoint. Sampled with the `hyperflow_8step` recipe: euler, the "
                      "`hyperflow` schedule, 8 steps, cfg 1. Dense attention -- HyperFlow was "
                      "validated dense."),
    }[engine]
    note(g, _dialogue_note(name, seconds, n, engine_text, media), (-620, 0))
    return g


def _line_table(n, media):
    rows = ["| Shot | Speaker | at (s) | from - to (s) in the file |", "|---|---|---|---|"]
    for i, (_, who, _, take, a, b) in enumerate(DIALOGUE[:n], 1):
        rows.append("| %d | %s | %.2f | %s: %.2f - %s |" % (
            i, who, (i - 1) * SHOT, media[take], a, "end" if b < 0 else "%.2f" % b))
    return "\n".join(rows)


def _dialogue_note(name, seconds, n, engine_text, media):
    return """# %s

%s

Each shot is **7.54 s** (181 frames): two shots fill one 15.08 s window exactly, so
every seam lands on a cut.%s

## Where the references are
**All in Pulse Slate's Asset Bin** -- the IMAGE / AUDIO REFERENCES panel at the bottom of
the node. Nothing else in this graph loads a picture or a recording.
* **Ada** and **Ben**: a face each, described, so each is one `<Subject N>`.
* **One audio row per line**, `lip_sync`, owned by its speaker. `at` is the film second
  the line starts (its shot's start); `from` / `to` cut the line out of the take file.

%s

## What this graph does
* **One model for every window** -- %s Faces and voices are referenced in every window,
  so nothing changes identity at a seam.
* **One voice track per character per window.** All of @Ada's lines in a window ride one
  `<Audio j>`; H3 takes three audio references per window, so three is the number of
  *voices*, not lines. The bin's meter counts them the same way.
* **speech_aware windows** (5-15 s): seams go on a cut where the frame grid reaches one,
  otherwise into a pause -- never through a word.
* **lock_source**: each window's real lines are written into the audio latent and held,
  so the mouth follows the waveform itself; `use_reference_audio` puts your exact takes
  in the film.
* **keep_aspect references**: faces reach H3 whole (no crop to 16:9).
* **upscale 1080p_fit**: rendered at 1344x768 inside H3's budget, delivered 1890x1080.

## Replace before you judge quality
1. In the Asset Bin, replace **Ada** and **Ben**: remove the row, drop your picture in,
   name it exactly Ada / Ben (the shots say @Ada and @Ben), describe it.
2. Replace the **line rows**: drop each take in, set its owner, `at`, `from` and `to`.
   A line plays from `at` for (`to` - `from`) seconds -- keep it inside its shot.
3. Optional: write each line's words in quotes in the shot's audio_line --
   `@Ada says "..."` -- it hands the text encoder the words it cannot hear.

## Settings that matter
* The Recipe node, not the slate's sampler widgets, decides how windows sample.
* `continuity` is **none**: every window is a fresh reference window and the seams sit
  on cuts. **ref_guided_carry** (experimental) also pins the previous window's last frame
  -- still one checkpoint, faces and voices still referenced.
* Turn **dry_run** on in Pulse Render to read the report (windows, seams, voice tracks)
  without rendering anything.
""" % (name, engine_text.split(".")[0] + ".",
       ("" if seconds == 30 else
        " The two placeholder take files hold two full lines each, so for 60 s the "
        "lines repeat -- your own takes will not."),
       _line_table(n, media), engine_text)


def _take_bin(seconds, media):
    """@Ada's face and the narration, on the film clock from 0 -- owned by @Ada."""
    assets = [face("ada", "Ada", media["face_a"], ADA),
              line("ada_narration", "AdaNarration", media["narration"], "ada", 0.0)]
    if seconds == 60:
        # Placeholder for the second half: the same narration again, from where the
        # first ends. Two rows, one voice -- still one <Audio j> per window.
        assets.append(line("ada_narration_2", "AdaNarration2", media["narration"], "ada",
                           float(media["narration_seconds"])))
    return bin_data(assets)


def _take(name, engine, seconds, media):
    """Families 2 and 4: one speaker, one walk-and-talk, every window on one model."""
    g = Graph(name)
    n = 4 if seconds == 30 else 8
    portrait = engine == "fasth3"
    clip, video_vae, audio_vae = loaders(g, 0, 620)
    if engine == "fasth3":
        base = unet(g, FASTH3, (0, 0), "FastH3-Omni 8-step (pruned int8, ref2va)")
        shift = sigma_shift(g, (0, 110))
        backend = attention_backend(g, (0, 220))
        sparse = vsa(g, (0, 310))
        g.link(base, "MODEL", shift, "model")
        g.link(shift, "MODEL", backend, "model")
        g.link(backend, "model", sparse, "model")
        model, model_slot = sparse, "model"
        rec = recipe(g, "fasth3_omni_8step", (0, 900), "Recipe -- FastH3-Omni 8-step")
        steps, sampler, scheduler = 8, "euler", "simple"
    else:
        base = unet(g, REF2VA_FULL, (0, 0), "H3 ref2va (non-pruned int8) -- HyperFlow needs it")
        adapter = hyperflow(g, (0, 110), "HyperFlow 8-step adapter, strength 1.0")
        shift = sigma_shift(g, (0, 250))
        g.link(base, "MODEL", adapter, "model")
        g.link(adapter, "MODEL", shift, "model")
        model, model_slot = shift, "MODEL"
        rec = recipe(g, "hyperflow_8step", (0, 900), "Recipe -- HyperFlow 8-step")
        steps, sampler, scheduler = 8, "euler", "hyperflow"
    g.group("1 · Model -- one checkpoint for every window", -20, -60, 500, 1180)

    shots = []
    for i, (label, visual) in enumerate(TAKE_SHOTS[:n], 1):
        col, row = divmod(i - 1, 4)
        s = shot(g, name, i, label, visual, "@Ada keeps talking to camera.", SHOT, "@Ada",
                 (520 + col * 460, row * 360))
        shots.append(s)
    g.group("2 · Shots -- one walk-and-talk, 7.54 s each", 500, -60,
            460 * ((n + 3) // 4) + 20, 1480)

    x = 520 + 460 * ((n + 3) // 4) + 60
    sl = slate(g, (x, 0), TAKE_GLOBAL, SHOT * n,
               "9:16 portrait" if portrait else "16:9 landscape",
               768 if portrait else 1344, 1344 if portrait else 768, steps, sampler, scheduler,
               20261020, "none", n_shots=n, timeline_data=_take_bin(seconds, media))
    g.link(model, model_slot, sl, "model")
    g.link(rec, "recipe", sl, "recipe")
    for i, s in enumerate(shots, 1):
        g.link(s, "shot", sl, "shots.shot_%d" % i)
    rn = render(g, (x + 800, 0), "ps_%s_take_%ds" % (engine, seconds),
                upscale="1080p_exact" if portrait else "1080p_fit")
    g.link(model, model_slot, rn, "model")
    shared_wiring(g, sl, rn, clip, video_vae, audio_vae)
    g.group("3 · Slate (face and narration in its Asset Bin) and render", x - 20, -60,
            1280, 1780)
    outputs(g, sl, rn, name, x + 1300, 0)

    if engine == "fasth3":
        engine_text = ("**FastH3-Omni 8-step** (ref2va, pruned int8 + VSA keep 10) on every "
                       "window, sampled with the `fasth3_omni_8step` recipe: euler, simple, "
                       "8 steps, cfg 1. 9:16 at 768x1344, delivered 1080x1920.")
    else:
        engine_text = ("**HyperFlow 8-step** on the non-pruned ref2va checkpoint, every "
                       "window, sampled with the "
                       "`hyperflow_8step` recipe: euler, the `hyperflow` schedule, 8 steps, "
                       "cfg 1. 16:9 at 1344x768, delivered 1890x1080. HyperFlow needs the "
                       "non-pruned checkpoint; it refuses pruned ones.")
    text = """# %s

%s

## Where the references are
**All in Pulse Slate's Asset Bin** -- the IMAGE / AUDIO REFERENCES panel at the bottom of
the node.
* **Ada**: her face, described, so she is one `<Subject N>`.
* **AdaNarration**: the narration, `lip_sync`, owned by Ada, `at` 0 -- it runs on the film
  clock from the first frame.%s

## What this graph does
* **One model for every window.** Ada's face and voice are referenced in every window,
  so neither changes at a seam. (Mixing a reference checkpoint with a First/Last-Frame
  one hands the later windows no references at all -- the face and voice drift.)
* **The seams are carried by reference**: each window after the first also sees the
  previous window's last frame as a picture (`carry_mode: image`), and `speech_aware`
  puts every seam into a pause in the narration, never through a word.
* **Shots of 7.54 s** (181 frames): two fill one 15.08 s window exactly, so the seams sit
  on cuts too. %d shots = %.2f s.
* **lock_source**: each window's seconds of the narration are written into the audio
  latent and held -- the mouth follows the recording itself.
* **keep_aspect** reference, **upscale** for delivery.

## Replace before you judge quality
1. In the Asset Bin, replace **Ada**: remove the row, drop your picture in, name it Ada
   (the shots say @Ada), describe her.
2. Replace the narration row: drop your recording in, owner Ada, `at` 0.%s
3. The shot texts -- keep the motion calm and continuous.

## Settings that matter
* `continuity` is **none**. **ref_guided_carry** (experimental) also pins the previous
  window's last frame at frame 0 -- still one checkpoint, face and voice still
  referenced. Compare it against none before trusting it.
* Turn **dry_run** on in Pulse Render to read the report without rendering.
""" % (name, engine_text,
       ("\n* **AdaNarration2**: the same file again from %.2f s -- a placeholder for the "
        "second half." % float(media["narration_seconds"]) if seconds == 60 else ""),
       n, SHOT * n,
       (" For 60 s, delete AdaNarration2 and use one 60 s recording." if seconds == 60
        else " The film is %.2f s; a longer narration is cut there." % (SHOT * n)))
    note(g, text, (-620, 0))
    return g


def _vertical(name, seconds, media):
    """Family 5: a vertical series -- start/end keyframes, TaoMate on every window."""
    g = Graph(name)
    n = 2 if seconds == 30 else 4
    clip, video_vae, audio_vae = loaders(g, 0, 700)
    base = unet(g, FL2VA_FULL, (0, 0), "H3 fl2va (non-pruned int8) -- TaoMate needs it")
    lora = lora_only(g, TAOMATE, (0, 110), "TaoMate FL2VA 3-step LoRA, strength 1.0")
    shift = sigma_shift(g, (0, 220))
    backend = attention_backend(g, (0, 310))
    sparse = sol_attn(g, (0, 400))
    g.link(base, "MODEL", lora, "model")
    g.link(lora, "MODEL", shift, "model")
    g.link(shift, "MODEL", backend, "model")
    g.link(backend, "model", sparse, "model")
    rec = recipe(g, "taomate_fl2va_3step", (0, 1000), "Recipe -- TaoMate 3-step (every window)")
    g.group("1 · Models -- one checkpoint, TaoMate on every window", -20, -60, 500, 1300)

    keys = []
    for i in range(n):
        file = media["key_1"] if i % 2 == 0 else media["key_2"]
        keys.append(load_image(g, file, (520, i * 370),
                               "Keyframe %d -- pick your picture" % (i + 1)))
    g.group("2 · Keyframes -- shot i starts on keyframe i, ends on keyframe i+1", 500, -60,
            360, 370 * n + 80)

    shots = []
    for i, (label, visual) in enumerate(VERTICAL_SHOTS[:n], 1):
        s = shot(g, name, i, label, visual, "Soft room tone; no dialogue.", GRID_15, "",
                 (900, (i - 1) * 360))
        g.link(keys[i - 1], "IMAGE", s, "start_image")
        shots.append(s)
    g.group("3 · Shots -- one full 15.08 s window each", 880, -60, 460, 360 * n + 80)

    x = 1400
    sl = slate(g, (x, 0),
               "style: Photoreal vertical drama, soft window light, gentle camera moves, "
               "calm motion only.\nsoundscape: Soft room tone and distant birdsong; no "
               "dialogue, no music.\nmusic: None.\nlanguage: English",
               GRID_15 * n, "9:16 portrait", 768, 1344, 3, "euler", "simple", 20261020,
               "keyframe_pairs", n_shots=n)
    g.link(sparse, "model", sl, "model")
    g.link(sparse, "model", sl, "model_fl2va")
    g.link(rec, "recipe", sl, "recipe_fl2va")
    for i, s in enumerate(shots, 1):
        g.link(s, "shot", sl, "shots.shot_%d" % i)
    rn = render(g, (x + 680, 0), "ps_taomate_vertical_%ds" % seconds,
                use_reference_audio=False, audio_mode="reference_only", upscale="1080p_exact")
    g.link(sparse, "model", rn, "model")
    g.link(sparse, "model", rn, "model_fl2va")
    shared_wiring(g, sl, rn, clip, video_vae, audio_vae)
    g.group("4 · Slate and render", x - 20, -60, 1160, 1480)
    outputs(g, sl, rn, name, x + 1180, 0)
    note(g, """# %s

A vertical keyframe series, driven by Pulse Slate instead of one hand-wired graph per
window: **%d shots x 15.08 s**, one full H3 window each, every window on **TaoMate FL2VA
3-step**.

## What this graph does
* **keyframe_pairs**: shot *i* starts on keyframe *i* and ends on keyframe *i+1* (the next
  shot's start image), so every cut lands on a frame both sides are pinned to.
* The `recipe_fl2va` input carries TaoMate's own schedule: euler, sigmas
  `1.0, 0.961165, 0.853333, 0.0`, cfg 1.
* Rendered at **768x1344** (inside H3's trained budget -- TaoMate's FL2VA runtime is fixed
  at 768x1376 and does not render 1080 natively), delivered **1080x1920** by
  `upscale: 1080p_exact`.
* The same fl2va chain feeds `model` and `model_fl2va`: one checkpoint in memory.

## Replace before you judge quality
1. The keyframes -- one LoadImage per shot (the placeholders alternate two pictures).
2. The shot texts: keep motion calm; 3 steps handle faces and slow moves well, fast
   action poorly.
3. To add speech: drop the recording into Pulse Slate's Asset Bin (AUDIO REFERENCES),
   `lip_sync`, `at` = the shot's start; set Pulse Render's audio_mode to **lock_source**
   and turn on **use_reference_audio**. fl2va takes no references, but a lip_sync
   recording in the bin still drives the mouth.
""" % (name, n), (-620, 0))
    return g


FAMILIES = [
    ("PS_FastH3Omni_Dialogue_%ds", lambda name, s, m: _dialogue(name, "fasth3", s, m)),
    ("PS_FastH3Omni_ContinuousTake_%ds", lambda name, s, m: _take(name, "fasth3", s, m)),
    ("PS_HyperFlow_Dialogue_%ds", lambda name, s, m: _dialogue(name, "hyperflow", s, m)),
    ("PS_HyperFlow_ContinuousTake_%ds", lambda name, s, m: _take(name, "hyperflow", s, m)),
    ("PS_TaoMate_Vertical_%ds", lambda name, s, m: _vertical(name, s, m)),
]


def build(out_dir, media=None):
    """Write the ten graphs. `media` overrides any MEDIA entry by key."""
    unknown = set(media or {}) - set(MEDIA)
    if unknown:
        raise ValueError("unknown media key(s): %s -- expected %s"
                         % (", ".join(sorted(unknown)), ", ".join(MEDIA)))
    media = dict(MEDIA, **(media or {}))
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for pattern, make in FAMILIES:
        for seconds in (30, 60):
            name = pattern % seconds
            graph = make(name, seconds, media)
            path = os.path.join(out_dir, name + ".json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(graph.to_json(), handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            written.append(path)
    return written


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    overrides = None
    if len(sys.argv) == 3:
        with open(sys.argv[2], encoding="utf-8") as handle:
            overrides = json.load(handle)
    for path in build(sys.argv[1], overrides):
        print(path)
