"""Which canvas an aspect ratio resolves to. Spec §4.

Here rather than in `nodes` for the same reason the frame grid and the concat
placement are: it is pure arithmetic, it is what every render's latent size
depends on, and `nodes` imports torch -- so a test that wants to check the canvas
in a bare environment cannot reach it there.

WHY THE OBVIOUS ROUNDING IS WRONG
---------------------------------
The first implementation fitted the ratio to the pixel budget and then floored
each axis to the grid independently::

    w = sqrt(MAX_PIXELS * ratio) // 32 * 32
    h = sqrt(MAX_PIXELS / ratio) // 32 * 32

Independent flooring loses up to one grid step on *both* axes, and the two losses
compound rather than cancel. At 16:9 that produced 1344x736: 4.2% under the
budget, and -- worse -- an actual ratio of 1.826 against a requested 1.778. The
canvas was neither as large as it could be nor the shape it claimed to be. The
same 1344 paired with 768 is on the grid, is *exactly* MAX_PIXELS, and is closer
to true 16:9 on top of that.

The rule that fixes it: the long edge is floored to the grid as before, and then
the short edge is the on-grid value NEAREST the ratio-implied ideal that still
fits the budget.

Nearest, and not simply the largest that fits, because those two are not the same
question and the difference is visible. At 4:3 the largest short edge that fits is
896, which hits the budget exactly but turns 1.3333 into 1.2857; the nearest is
864, which is exactly 4:3 and gives up 3.6% of the budget to stay that way. Taking
the largest everywhere would also make the 1:1 preset 992x1024, which is not
square. A preset is a promise about shape, so shape wins and the budget is filled
only where filling it does not break the promise:

    16:9      1344x736  ->  1344x768     95.8% -> 100.0% of budget
    9:16       736x1344 ->   768x1344    95.8% -> 100.0%
    21:9      1536x640  ->  1536x672     95.2% -> 100.0%
    1:1        992x992  ->   992x992     unchanged, exactly square
    4:3       1152x864  ->  1152x864     unchanged, exactly 4:3
    3:4        864x1152 ->   864x1152    unchanged

Pure stdlib beyond `math`. `tests/test_canvas.py` drives it.
"""

import math

from .constants import CANVAS_MULTIPLE, MAX_PIXELS

__all__ = ["ASPECT_RATIOS", "ASPECT_OPTIONS", "fit_canvas", "resolution_for"]


# H3's six documented aspect ratios, as ratio pairs only. Each is fitted into the
# pixel budget at import time rather than hardcoded, so the table cannot drift
# from MAX_PIXELS.
ASPECT_RATIOS = {
    "16:9 landscape": (16, 9),
    "9:16 portrait": (9, 16),
    "1:1 square": (1, 1),
    "4:3 landscape": (4, 3),
    "3:4 portrait": (3, 4),
    "21:9 ultrawide": (21, 9),
}
ASPECT_OPTIONS = ["custom"] + list(ASPECT_RATIOS)


def _floor_to(value, multiple):
    return max(multiple, int(value // multiple) * multiple)


def _short_edge(long_edge, ideal_short, max_pixels, multiple):
    """The on-grid short edge nearest `ideal_short` that keeps the area in budget.

    The budget is enforced by the cap rather than by checking afterwards, so the
    returned pair can only ever be inside it.
    """
    cap = max(multiple, int(max_pixels // long_edge) // multiple * multiple)
    below = min(max(multiple, _floor_to(ideal_short, multiple)), cap)
    above = min(below + multiple, cap)
    # Ties go to the larger edge: same shape error either way, more pixels.
    if abs(above - ideal_short) <= abs(ideal_short - below):
        return above
    return below


def fit_canvas(ratio, max_pixels=MAX_PIXELS, multiple=CANVAS_MULTIPLE):
    """Largest on-grid canvas at `ratio` that fits the budget without distorting it.

    `ratio` is width/height. Returns (width, height), each a multiple of
    `multiple` and at least `multiple`, with width*height <= max_pixels.
    """
    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError("ratio must be positive, got %r" % (ratio,))

    ideal_w = math.sqrt(max_pixels * ratio)
    ideal_h = math.sqrt(max_pixels / ratio)

    if ideal_w >= ideal_h:
        width = _floor_to(ideal_w, multiple)
        return width, _short_edge(width, width / ratio, max_pixels, multiple)
    height = _floor_to(ideal_h, multiple)
    return _short_edge(height, height * ratio, max_pixels, multiple), height


def resolution_for(aspect, width, height):
    """Canvas for an aspect-ratio choice.

    'custom' is left alone beyond snapping to the grid: the widgets are documented
    as being used directly there, and capping a typed number to the budget would
    silently override it. Over-budget custom canvases are refused where it
    matters, in `still.plan_still`, with a message.
    """
    if aspect == "custom" or aspect not in ASPECT_RATIOS:
        return (_floor_to(int(width), CANVAS_MULTIPLE),
                _floor_to(int(height), CANVAS_MULTIPLE))
    rw, rh = ASPECT_RATIOS[aspect]
    return fit_canvas(rw / rh)
