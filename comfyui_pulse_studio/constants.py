"""Model constants, every one read out of ComfyUI's own source rather than inferred.

Each value below carries the file and symbol it was verified against. If ComfyUI
changes, re-verify here first — nothing else in this package hardcodes these.

Verified against ComfyUI core at:
  comfy_extras/nodes_minimax_h3.py
  comfy/ldm/minimax/model.py       (PackedLayout.__init__)
  comfy/text_encoders/minimax.py   (MiniMaxTokenizer.tokenize_with_weights)
"""

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
