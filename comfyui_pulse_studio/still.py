"""Still Mode: generate or edit a single image through the same H3 pipeline.

A still in H3 is a video render with `length = 5` where you keep one frame. Same
conditioning nodes, same compiler, same reference marshalling -- only the
terminal differs (SaveImage rather than CreateVideo). That is why this belongs in
the Director rather than in a separate tool: the reference images that drive a
video have to come from somewhere, and generating them *through the same
reference set* makes identity consistent by construction instead of by luck.

Two controls do the real work:

  frame_pick (0-4)
      The source image is pinned at frame 0, so frame 0 hugs the original and
      frame 4 has drifted furthest from it. This is the edit-strength dial. It is
      exposed rather than hidden because it is the only continuous control over
      how much the edit departs from the source.

  canvas_from_reference
      When editing, the source should set the canvas rather than being cropped to
      a preset. Its aspect is fitted into H3's pixel budget and rounded down to
      multiples of 32.

Kept deliberately as a mode, not a second product: no layers, no masks, no
inpainting. The moment it grows a brush it has become a different application.
"""

import math

from .constants import (
    BASE_SHORT_EDGE,
    BRANCH_FL2VA,
    BRANCH_REF2VA,
    CANVAS_MULTIPLE,
    MAX_PIXELS,
    STILL_FRAME_PICK_MAX,
    STILL_FRAME_PICK_MIN,
    STILL_FRAMES,
)

__all__ = ["canvas_from_reference", "adapt_canvas_core", "StillPlan", "plan_still", "StillError"]


class StillError(ValueError):
    """A still that cannot be rendered as asked."""


def canvas_from_reference(width, height, max_pixels=MAX_PIXELS, multiple=CANVAS_MULTIPLE):
    """Largest canvas with the source's aspect ratio that fits the pixel budget.

    Rounds DOWN to `multiple` on both axes, which is what keeps the result inside
    the budget: rounding to nearest, as core's own adapt_canvas does, can round
    both axes up and land slightly over. Rounding down can only ever undershoot,
    and undershooting the budget is harmless where exceeding it is not.

    Returns (width, height), each at least `multiple`.
    """
    width, height = float(width), float(height)
    if width <= 0 or height <= 0:
        raise StillError("reference dimensions must be positive, got %rx%r" % (width, height))

    ratio = width / height
    # Fill the budget exactly at this aspect, then round down to the grid.
    nominal_w = math.sqrt(max_pixels * ratio)
    nominal_h = math.sqrt(max_pixels / ratio)

    w = max(multiple, int(nominal_w // multiple) * multiple)
    h = max(multiple, int(nominal_h // multiple) * multiple)
    return w, h


def adapt_canvas_core(width, height):
    """Behavioural mirror of core's nodes_minimax_h3.adapt_canvas.

    Normalises the *short edge* to 768 and only then caps the area, rounding to
    nearest. This is what the stock nodes do to a reference video's canvas, so it
    is kept available for callers that want to match core exactly rather than
    fill the budget. It is not the Still Mode default -- for editing, filling the
    budget preserves more of the source.
    """
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nom_w, nom_h = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


class StillPlan:
    """A single-image render, fully specified."""

    __slots__ = ("branch", "width", "height", "length", "frame_pick", "prompt",
                 "source_asset", "files", "anchors", "canvas_source", "diagnostics")

    def __init__(self, branch, width, height, length, frame_pick, prompt,
                 source_asset, files, anchors, canvas_source, diagnostics):
        self.branch = branch
        self.width = width
        self.height = height
        self.length = length
        self.frame_pick = frame_pick
        self.prompt = prompt
        self.source_asset = source_asset
        self.files = files
        self.anchors = anchors
        self.canvas_source = canvas_source
        self.diagnostics = diagnostics

    @property
    def is_edit(self):
        """True when a source image is pinned -- an edit rather than a generation."""
        return self.source_asset is not None

    @property
    def megapixels(self):
        return (self.width * self.height) / 1_000_000.0

    def to_dict(self):
        return {"branch": self.branch, "width": self.width, "height": self.height,
                "length": self.length, "frame_pick": self.frame_pick,
                "prompt": self.prompt, "source_asset": self.source_asset,
                "files": [f.to_dict() for f in self.files], "anchors": dict(self.anchors),
                "canvas_source": self.canvas_source, "is_edit": self.is_edit,
                "diagnostics": list(self.diagnostics)}

    def __repr__(self):
        return "StillPlan(%s, %dx%d, pick %d, %s)" % (
            self.branch, self.width, self.height, self.frame_pick,
            "edit" if self.is_edit else "generate")


def plan_still(prompt="", width=None, height=None, frame_pick=0, source_asset=None,
               source_size=None, canvas_from_reference_enabled=True, bin_=None,
               reference_ids=None):
    """Plan a still render.

    Editing (a `source_asset` is given) pins that image at frame 0 and runs the
    fl2va branch, because pinning a frame and carrying references are mutually
    exclusive -- different checkpoints, disjoint inputs. Generating with
    references runs ref2va instead.

    `source_size` is the source image's (width, height); with
    `canvas_from_reference_enabled` it sets the canvas, so the source is not
    cropped to a preset that does not match it.
    """
    diagnostics = []

    frame_pick = int(frame_pick)
    if not (STILL_FRAME_PICK_MIN <= frame_pick <= STILL_FRAME_PICK_MAX):
        raise StillError(
            "frame_pick must be %d-%d (a still renders %d frames); got %r"
            % (STILL_FRAME_PICK_MIN, STILL_FRAME_PICK_MAX, STILL_FRAMES, frame_pick))

    # ── canvas ─────────────────────────────────────────────────────────────
    canvas_source = "explicit"
    if canvas_from_reference_enabled and source_size:
        width, height = canvas_from_reference(*source_size)
        canvas_source = "reference"
    elif width is None or height is None:
        width, height = canvas_from_reference(BASE_SHORT_EDGE * 16, BASE_SHORT_EDGE * 9)
        canvas_source = "default"
        diagnostics.append("no canvas given; defaulted to %dx%d (16:9 within the pixel budget)"
                           % (width, height))
    else:
        width, height = int(width), int(height)
        if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
            width = max(CANVAS_MULTIPLE, width // CANVAS_MULTIPLE * CANVAS_MULTIPLE)
            height = max(CANVAS_MULTIPLE, height // CANVAS_MULTIPLE * CANVAS_MULTIPLE)
            diagnostics.append("canvas rounded down to %dx%d (multiples of %d)"
                               % (width, height, CANVAS_MULTIPLE))
        if width * height > MAX_PIXELS:
            diagnostics.append(
                "canvas %dx%d is %.0f px, over H3's %d px budget; expect degraded output"
                % (width, height, width * height, MAX_PIXELS))

    # ── branch and inputs ──────────────────────────────────────────────────
    files = []
    anchors = {}
    reference_ids = list(reference_ids or [])

    if source_asset is not None:
        branch = BRANCH_FL2VA
        anchors["first_frame"] = source_asset
        anchors["first_frame_index"] = 0
        if reference_ids:
            diagnostics.append(
                "editing pins the source at frame 0, which is the fl2va branch -- it takes "
                "no references, so the %d reference(s) requested are ignored for this still"
                % (len(reference_ids),))
        if frame_pick == 0:
            diagnostics.append(
                "frame_pick 0 returns the frame the source is pinned to, so the result will "
                "closely reproduce the source; raise it to let the edit take effect")
    else:
        branch = BRANCH_REF2VA
        if bin_ is not None and reference_ids:
            tag_map = bin_.tag_map()
            from .compiler import FileRef
            for socket, asset_id in sorted(tag_map.sockets.items()):
                if asset_id not in reference_ids:
                    continue
                asset = bin_.get(asset_id)
                files.append(FileRef(socket, asset.kind, asset.asset_id, asset.name,
                                     asset.file, tag_map.by_id.get(asset_id),
                                     asset.trim_start, asset.trim_end, asset.synthetic))

    return StillPlan(branch, width, height, STILL_FRAMES, frame_pick, prompt,
                     source_asset, files, anchors, canvas_source, diagnostics)
