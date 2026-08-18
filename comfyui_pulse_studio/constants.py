"""Model constants, every one read out of ComfyUI's own source rather than inferred.

Each value below carries the file and symbol it was verified against. If ComfyUI
changes, re-verify here first — nothing else in this package hardcodes these.

Verified against ComfyUI core at:
  comfy_extras/nodes_minimax_h3.py
  comfy/ldm/minimax/model.py       (PackedLayout.__init__)
  comfy/text_encoders/minimax.py   (MiniMaxTokenizer.tokenize_with_weights)
"""

# ── Slot contract ───────────────────────────────────────────────────────────
# The one value in this file that is ours rather than ComfyUI's, and the one
# that must never be edited casually.
#
# LiteGraph serialises a node as widgets_values[i] = node.widgets[i].value -- a
# positional array over the live widget list. The moment one workflow is saved,
# widget order is a public API indexed by position. SCHEMA_VERSION is what lets
# a saved file say which order it was written in, so loading can go by name
# instead of by position (js/ps_widget_order.js).
#
# Bump it only alongside a new entry in WIDGET_NAMES and a CHANGELOG migration
# note. Appending a widget does NOT require a bump; inserting, reordering,
# removing, retyping or renaming one does -- and those are forbidden anyway.
SCHEMA_VERSION = "3.0.0"

# The empty asset-bin document, written in schema 2 form from 1.0 onward even
# though the cast UI ships in 1.1. An empty `cast` key costs nothing today and
# means 1.1 adds entries to a key that already exists, rather than migrating a
# file format that is already in the wild.
TIMELINE_SCHEMA = 2

# ── Frame grid ──────────────────────────────────────────────────────────────
# nodes_minimax_h3.align_frame_count():  while n % 17 != 5: n += 1
# Legal frame counts are therefore exactly {5, 22, 39, 56, ...} = 17k + 5.
# Note the core helper rounds *up* only; this package also needs a round-down
# and a round-nearest, which live in frames.py.
FRAME_GRID_MODULUS = 17
FRAME_GRID_REMAINDER = 5

# nodes_minimax_h3.temporal_shape():  frame_count = align_frame_count(max(5, length))
MIN_FRAMES = 5

# 362 = 17*21 + 5, exactly on-grid. This is the top of H3's *trained* range, not
# a hard error: the stock node's `length` input accepts up to 3600 and the model
# will run past 362, untested. Treated as configuration per the blueprint's
# duration-agnostic rule -- when MiniMax raises the ceiling, this one number moves.
MAX_WINDOW_FRAMES = 362

# Bottom of the trained range (~5.17s). Below this, output quality degrades; used
# as the default floor when deciding whether a trailing window is worth rendering.
MIN_TRAINED_FRAMES = 124

# nodes_minimax_h3.FPS
FPS = 24

# nodes_minimax_h3.AUDIO_LATENT_FPS
AUDIO_LATENT_FPS = 40

# ── Canvas ──────────────────────────────────────────────────────────────────
# nodes_minimax_h3.CANVAS_MULTIPLE / BASE_SHORT_EDGE / MAX_PIXELS
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344  # 1,032,192

# nodes_minimax_h3.REF_IMAGE_SHORT_EDGE -- the 'max' setting of ref_image_size
REF_IMAGE_SHORT_EDGE = 2048

# nodes_minimax_h3.MiniMaxH3ReferenceToVideo, ref_image_size combo
REF_IMAGE_SIZE_OPTIONS = ("match", "max")

# ── Keyframe anchors ────────────────────────────────────────────────────────
# comfy/ldm/minimax/model.py, PackedLayout.__init__:
#
#     if pixel_index == 0:                              cond_t = text_len
#     elif frame_count is not None and pixel_index == frame_count - 1:
#                                                       cond_t = ...
#     else: raise ValueError("only first/last keyframe anchors are supported")
#
# This is RoPE position math inside the DiT, not a node-level guard. There is no
# workaround at the custom-node layer: an image can be pinned to frame 0 or to
# frame_count-1 and nowhere else. Per-shot visual control is textual only.
ANCHOR_FIRST = 0  # resolved_frame_index for first_frame
# The last anchor index is frame_count - 1; see frames.last_anchor_index().

# ── Reference budget ────────────────────────────────────────────────────────
# nodes_minimax_h3.MiniMaxH3ReferenceToVideo, io.Autogrow template max= values.
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_VIDEO_AUDIOS = 3  # soundtracks, index-paired to ref_videos
MAX_REF_AUDIOS = 3        # standalone reference audio

# Cross-type cap. 9 + 3 + 3 = 15 addressable slots exist, but no more than 12
# distinct files may be presented in one render, so this cap genuinely binds.
MAX_REF_FILES_TOTAL = 12

# ── The audio ceiling is raisable, and every other one is not ───────────────
# Read out of source before believing it:
#
#   comfy_extras/nodes_minimax_h3.py   ref_audios is io.Autogrow with max=3
#   comfy/ldm/minimax/model.py         PackedLayout walks reference blocks in a
#                                      loop and appends one ref_audio segment
#                                      per block. No count check, no assert.
#   comfy/text_encoders/minimax.py     the tokenizer increments a counter for
#                                      <Audio j>. No cap there either.
#
# So the 3 is a *socket* cap, not model math -- unlike the first/last keyframe
# rule above, which is RoPE position math no node can route around. And this
# pack calls MiniMaxH3ReferenceToVideo.execute() in-process rather than through
# the graph, so Autogrow's max= (enforced by execution.py's validation pass)
# never runs against what we pass. Nine standalone audio references will
# therefore be marshalled and rendered.
#
# That is a bypass of a declared contract, so it is opt-in and it is not the
# default. MiniMax documents three, ComfyUI's socket declares three, and no one
# has published a render using more. Reference audio rides through every
# sampling step, so the cost is certain and the quality effect is not known.
#
# MAX_REF_FILES_TOTAL is a model-card number too -- it appears nowhere in
# ComfyUI's source. Raising the audio ceiling raises the total by the same
# amount, because the extra files are exactly the extra audio (see RefLimits).
MAX_REF_AUDIOS_CEILING = 9

# Reference video duration window. The stock node enforces only the lower bound
# in code (">= 5 frames"); 2-15s is the documented usable range and what the
# budget checker reports against.
REF_VIDEO_MIN_SECONDS = 2.0
REF_VIDEO_MAX_SECONDS = 15.0

# nodes_minimax_h3.MiniMaxH3ReferenceToVideo.execute(): a reference video is
# truncated to the render's frame_count, then quantised DOWN to the 17k+5 grid
# (`while n % 17 != 5: n -= 1`), and must retain at least 5 frames.
REF_VIDEO_MIN_FRAMES = 5

# ── Branches ────────────────────────────────────────────────────────────────
# Two nodes, two checkpoints, disjoint inputs. Mutually exclusive per render:
# you cannot hold a character by reference AND anchor a frame in the same call.
BRANCH_FL2VA = "fl2va"  # MiniMaxH3ImageToVideo -- first_frame/last_frame, no refs
BRANCH_REF2VA = "ref2va"  # MiniMaxH3ReferenceToVideo -- refs, no anchors
BRANCHES = (BRANCH_FL2VA, BRANCH_REF2VA)

# Which reference types each branch can carry at all.
BRANCH_ACCEPTS_REFERENCES = {BRANCH_FL2VA: False, BRANCH_REF2VA: True}
BRANCH_ACCEPTS_ANCHORS = {BRANCH_FL2VA: True, BRANCH_REF2VA: False}

# ── Tag vocabulary ──────────────────────────────────────────────────────────
# comfy/text_encoders/minimax.py:
#
#     counters = {"image": 0, "audio": 0, "video": 0}
#     for item in minimax_ref_items:
#         counters[kind] += 1
#         image -> "<Picture %d>: "   audio -> "<Audio %d>: "   video -> "<Video %d>: "
#
# Ordinals are 1-based, per type, assigned purely by position in ref_items --
# i.e. by SOCKET ORDER. Nothing about the tag is stable across an edit to the
# reference list, which is why this package never lets a human type one.
TAG_IMAGE = "Picture"
TAG_VIDEO = "Video"
TAG_AUDIO = "Audio"

# nodes_minimax_h3.MiniMaxH3ReferenceToVideo.execute() builds ref_items in this
# order, and the tokenizer numbers them in exactly that order:
#   1. every ref_image                          -> <Picture i>
#   2. per ref_video, in socket order:
#        if it has a paired soundtrack, an audio item is appended FIRST
#                                                 -> <Audio j>
#        then the video item                      -> <Video k>
#   3. every standalone ref_audio               -> <Audio j>
#
# The consequence that bites: a reference video's soundtrack consumes an
# <Audio j> ordinal BEFORE any standalone reference audio does. Adding a
# soundtrack to video 1 renumbers every standalone audio tag in the prompt.
REF_ITEM_ORDER = ("images", "videos_with_soundtracks", "standalone_audios")

# ── What a reference audio is FOR ───────────────────────────────────────────
# Two different jobs, and they need different prompt text and different handling.
#
# `lip_sync`: the model matches the character's mouth to this recording. The
# audio rows are frozen conditioning the DiT attends to (model.py PackedLayout
# marks them `audio_update=False`), and the prompt has to *ask* for the match --
# the tokenizer only ever sees the marker "<Audio j>: ", never the waveform, so
# nothing about the sockets alone tells the model what the audio is for. A
# working reference workflow phrases it as "lip movements matching <Audio 1>
# precisely", and that sentence is the whole mechanism.
#
# Because it is a temporal alignment, the clip must span exactly the window it
# rides in -- see AUDIO_LIP_SYNC_TRIM in render.py. An untrimmed 30s file
# against a 9.42s window is asking the model to align two different spans of
# time, which is the failure that reads as "lip sync doesn't work".
#
# `voice_timbre`: the model generates its own speech from the shot's dialogue
# and borrows only the voice's character from this sample. No frame alignment
# is required or wanted.
#
# H3 always synthesises its own audio track either way. On the lip_sync path
# that track is a re-synthesis of the reference, which is why PulseRender can
# mux the original instead -- see `use_reference_audio`.
AUDIO_ROLE_LIP_SYNC = "lip_sync"
AUDIO_ROLE_TIMBRE = "voice_timbre"
AUDIO_ROLES = (AUDIO_ROLE_LIP_SYNC, AUDIO_ROLE_TIMBRE)

# ── How much of a visual reference to hold on to ────────────────────────────
# These strings are written *verbatim* into the prompt's retention_analysis
# section, which the model reads literally -- they are content, not an enum the
# code branches on. The document schema says "retention": "string" and that stays
# true, because a project that has a better phrase for MiniMax should be able to
# use it by editing timeline_data.
#
# The panel's selector is closed over these two anyway. Free text in a dropdown
# invites a typo into a sentence nothing validates, and a misspelled retention
# instruction is not an error -- it is a slightly worse render, which is the
# hardest kind of failure to notice.
RETENTION_FULL = "fully_preserved"
RETENTION_PARTIAL = "partially_copy"
RETENTION_VALUES = (RETENTION_FULL, RETENTION_PARTIAL)
RETENTION_DEFAULT = RETENTION_FULL

# ── What to do where two windows meet ───────────────────────────────────────
# The container-level seam is solved: segment placement is computed from frame
# counts, not measured from timestamps, so the picture is gapless by
# construction. What is not solved is that the two sides are two *independently
# generated* takes -- a score that restarts, a level that steps, a grade that
# drifts. None of that is a timestamp problem and none of it was addressed
# anywhere.
#
# Off is offered because a treatment that changes rendered output should be
# A/B-able against the untreated join, and because colour matching is the kind of
# correction that occasionally makes things worse.
SEAM_OFF = "off"
SEAM_AUDIO = "audio"
SEAM_AUDIO_COLOUR = "audio+colour"
SEAM_MODES = (SEAM_OFF, SEAM_AUDIO, SEAM_AUDIO_COLOUR)
SEAM_DEFAULT = SEAM_AUDIO_COLOUR

#: How far into the next window a colour correction is carried before it has
#: fully decayed. A flat correction across a whole window does not remove a cut,
#: it moves it to the *next* seam.
SEAM_COLOUR_RAMP_FRAMES = 12

# ── Audio carry-over ────────────────────────────────────────────────────────
# Upstream (muse-collective, MIT) fed the previous window's last ~4s of decoded
# audio back through ref_audios slot 0, having observed an audible hard reset
# between windows without it. Kept as a tunable default, not a constant -- the
# blueprint flags the exact tail length as untested.
DEFAULT_AUDIO_CARRY_SECONDS = 4.0

# ── Still mode ──────────────────────────────────────────────────────────────
# A still is a video render with length=5 where one frame is kept. frame_pick 0
# hugs the pinned source frame; 4 drifts furthest -- this is the edit-strength dial.
STILL_FRAMES = 5
STILL_FRAME_PICK_MIN = 0
STILL_FRAME_PICK_MAX = 4
