"""The 17k+5 frame grid, and how a requested duration is cut into render windows.

Pure stdlib -- no torch, no comfy. Everything here is testable headless.

The grid is not a limit we could relax; it is the shape of the model's latent
temporal packing (`video_latent_t` below is core's own formula). Every frame
count that reaches a stock H3 node must sit on it, so quantisation happens once,
here, and every caller uses the result rather than rounding on its own.

ATTRIBUTION
-----------
The idea of cutting a long timeline into chained fixed-ceiling windows comes
from muse-collective-26/MiniMaxH3-Director-Seed-Hunt (MIT), which this project
forks. The partitioning policies, the 124-frame trained floor, the backwards
tail merge and the rebalancing below are this project's own work; the design
they implement is upstream's. See NOTICE.
"""

import math

from .constants import (
    FPS,
    FRAME_GRID_MODULUS,
    FRAME_GRID_REMAINDER,
    MAX_WINDOW_FRAMES,
    MIN_FRAMES,
    MIN_TRAINED_FRAMES,
)

__all__ = [
    "is_on_grid",
    "align_frame_count",
    "snap_frames",
    "snap_frames_down",
    "snap_frames_nearest",
    "grid_step_index",
    "frames_to_seconds",
    "seconds_to_frames",
    "video_latent_t",
    "last_anchor_index",
    "partition_windows",
    "window_bounds",
]


def is_on_grid(n):
    """True when n is a legal H3 frame count (n == 17k + 5 for integer k >= 0)."""
    return isinstance(n, int) and n >= MIN_FRAMES and n % FRAME_GRID_MODULUS == FRAME_GRID_REMAINDER


def align_frame_count(n):
    """Round up to the next legal frame count.

    Byte-for-byte behavioural mirror of core's own `align_frame_count`, including
    its implicit floor: core calls it as `align_frame_count(max(5, length))`, so
    anything below 5 lands on 5 rather than on a negative grid point.
    """
    n = max(MIN_FRAMES, int(n))
    while n % FRAME_GRID_MODULUS != FRAME_GRID_REMAINDER:
        n += 1
    return n


def snap_frames_down(n):
    """Round down to the previous legal frame count, with 5 as the floor.

    Core uses this direction when trimming a *reference* video to the grid
    (`while n % 17 != 5: n -= 1`). Nothing may be shorter than 5 frames.
    """
    n = int(n)
    if n < MIN_FRAMES:
        return MIN_FRAMES
    while n % FRAME_GRID_MODULUS != FRAME_GRID_REMAINDER:
        n -= 1
    return n


def snap_frames_nearest(n):
    """Round to whichever legal frame count is closer; ties round up.

    Used where a UI is snapping a user's dragged cut point -- rounding a handle
    always-up would drift the cut visibly to one side.
    """
    n = int(n)
    if n <= MIN_FRAMES:
        return MIN_FRAMES
    down = snap_frames_down(n)
    up = align_frame_count(n)
    return down if (n - down) < (up - n) else up


def snap_frames(n, mode="up"):
    """Quantise `n` onto the frame grid. mode is 'up' | 'down' | 'nearest'."""
    if mode == "up":
        return align_frame_count(n)
    if mode == "down":
        return snap_frames_down(n)
    if mode == "nearest":
        return snap_frames_nearest(n)
    raise ValueError("snap mode must be 'up', 'down' or 'nearest', got %r" % (mode,))


def grid_step_index(n):
    """The k in n == 17k + 5. Raises if n is off-grid."""
    if not is_on_grid(n):
        raise ValueError("%r is not on the 17k+5 frame grid" % (n,))
    return (n - FRAME_GRID_REMAINDER) // FRAME_GRID_MODULUS


def frames_from_step(k):
    """Inverse of grid_step_index: the frame count for grid step k."""
    return FRAME_GRID_MODULUS * int(k) + FRAME_GRID_REMAINDER


def frames_to_seconds(frames, fps=FPS):
    return frames / float(fps)


def seconds_to_frames(seconds, fps=FPS, mode="up"):
    """Convert a duration to a legal frame count, quantising in `mode`.

    The float->int step follows `mode` rather than always rounding: rounding 7.3s
    to 175 frames and then finding 175 already on-grid yields 7.292s, i.e. a
    render *shorter* than asked for, which silently truncates the last shot.
    'up' therefore ceils, 'down' floors, and only 'nearest' rounds.
    """
    exact = float(seconds) * fps
    if mode == "up":
        n = math.ceil(exact - 1e-9)
    elif mode == "down":
        n = math.floor(exact + 1e-9)
    else:
        n = round(exact)
    return snap_frames(n, mode=mode)


def video_latent_t(frame_count):
    """Core's own latent-temporal-length formula, for planning only.

    nodes_minimax_h3.video_latent_t: 2 if frame_count <= 5 else ((fc-5)//17)*5+2
    """
    return 2 if frame_count <= MIN_FRAMES else ((frame_count - MIN_FRAMES) // FRAME_GRID_MODULUS) * 5 + 2


def last_anchor_index(frame_count):
    """The only legal non-zero keyframe index for a render of this length.

    PackedLayout accepts pixel_index 0 or frame_count-1. Anything else raises
    ValueError inside the DiT. Callers pinning a `last_frame` must use this.
    """
    if not is_on_grid(frame_count):
        raise ValueError("frame_count %r is not on the 17k+5 grid" % (frame_count,))
    return frame_count - 1


def _spread(total_frames, n):
    """Split `total_frames` across exactly n windows, as evenly as the grid allows.

    Solved in grid-step space -- sum(17*k_i + 5) >= total, i.e. 17*K + 5n >= total
    -- so every window is legal by construction rather than quantised afterwards
    and hoped to still sum correctly.
    """
    steps_needed = max(0, math.ceil((total_frames - FRAME_GRID_REMAINDER * n) / FRAME_GRID_MODULUS))
    base, extra = divmod(steps_needed, n)
    return [frames_from_step(base + 1)] * extra + [frames_from_step(base)] * (n - extra)


def _merge_short_tail(windows, floor, notes):
    """Fold sub-floor trailing windows backwards. Returns None if impossible.

    Two grid values do not sum to a grid value (17a+5 + 17b+5 = 17(a+b)+10), so
    the merged window is snapped up rather than simply added -- which is also why
    a merge can run past the model ceiling and have to give up.
    """
    windows = list(windows)
    while len(windows) > 1 and windows[-1] < floor:
        merged = align_frame_count(windows[-2] + windows[-1])
        if merged > MAX_WINDOW_FRAMES:
            return None
        notes.append(
            "trailing window of %d frames was below the %d-frame trained floor; merged "
            "into the window before it, which is now %d frames (~%.2fs)"
            % (windows[-1], floor, merged, frames_to_seconds(merged)))
        windows[-2] = merged
        windows.pop()
    if len(windows) > 1 and min(windows) < floor:
        return None
    return windows


def partition_windows(total_frames, window_frames=MAX_WINDOW_FRAMES,
                      policy="balanced", min_window_frames=MIN_TRAINED_FRAMES,
                      diagnostics=None):
    """Split a total frame count into per-render windows, all on the grid.

    Returns a list of frame counts whose sum is >= the requested total (never
    less -- a window cannot be shortened off-grid to make the sum exact, so the
    render slightly overshoots rather than truncating the user's last shot).

    Every emitted window is at least `min_window_frames`, H3's trained floor of
    124. Sub-floor windows are not merely discouraged, they are eliminated: a
    short tail is merged backwards into its predecessor, and where that is not
    possible the partition is rebalanced into fewer, longer windows.

    The single documented exception: a render whose *total* is itself below the
    floor passes through as one short window. There is nothing to merge it with,
    and refusing a deliberately short clip would be worse than rendering it.

    policy:
      'balanced' (default) -- near-equal windows, differing by at most one grid
          step. Asking for 16s at a 15s window gives two ~8s windows, not
          15s + 1s.
      'fill' -- pack `window_frames` repeatedly and let the remainder be the
          tail. Honours a literal "my windows are 10s" request. A sub-floor tail
          is merged into the window before it; if that would exceed the model
          ceiling, the partition falls back to balanced.

    The floor outranks `window_frames`. The two can conflict -- 200 frames in
    124-frame windows cannot give two windows that are both >= 124 -- and when
    they do, window length is stretched (never past MAX_WINDOW_FRAMES) and the
    override is reported through `diagnostics`.
    """
    notes = diagnostics if diagnostics is not None else []
    # The *total* is a request, not a render length, so it is deliberately not
    # snapped here -- only the individual windows must sit on the grid. Snapping
    # the total first would inflate it by up to a grid step before the split even
    # begins, and that overshoot compounds across windows.
    total_frames = max(MIN_FRAMES, int(total_frames))
    cap = max(MIN_FRAMES, min(snap_frames_down(window_frames), MAX_WINDOW_FRAMES))
    floor = max(MIN_FRAMES, int(min_window_frames))

    if policy not in ("balanced", "fill"):
        raise ValueError("policy must be 'balanced' or 'fill', got %r" % (policy,))

    # The exception: a total below the floor has nothing to merge into.
    if total_frames <= floor:
        window = align_frame_count(total_frames)
        if window < floor:
            notes.append(
                "the whole render is %d frames, below H3's trained floor of %d (~%.2fs); "
                "it renders as one short window, outside the range the model was trained on"
                % (window, floor, frames_to_seconds(floor)))
        return [window]

    if total_frames <= cap:
        return [align_frame_count(total_frames)]

    if policy == "fill":
        n_full, remainder = divmod(total_frames, cap)
        windows = [cap] * n_full
        if remainder:
            windows.append(align_frame_count(remainder))
        merged = _merge_short_tail(windows, floor, notes)
        if merged is not None:
            return merged
        notes.append(
            "the trailing window fell below the %d-frame trained floor and could not merge "
            "without exceeding the %d-frame model ceiling; rebalanced into even windows"
            % (floor, MAX_WINDOW_FRAMES))

    # ── balanced ────────────────────────────────────────────────────────────
    # Fewest windows that respect the cap; most that respect the floor.
    n_fewest = max(1, math.ceil(total_frames / cap))
    n_most = max(1, total_frames // floor)

    if n_fewest <= n_most:
        n = n_fewest
    else:
        # Cap and floor cannot both hold. The floor is a property of how the
        # model was trained; the cap is a preference. The floor wins.
        n = n_most
        stretched = _spread(total_frames, n)
        notes.append(
            "a window length of %d frames (~%.2fs) cannot hold alongside the %d-frame "
            "trained floor; windows stretched to %d frames (~%.2fs) so none falls below it"
            % (cap, frames_to_seconds(cap), floor, max(stretched),
               frames_to_seconds(max(stretched))))

    while True:
        windows = _spread(total_frames, n)
        if max(windows) > MAX_WINDOW_FRAMES:
            n += 1
            continue
        if min(windows) < floor and n > 1:
            n -= 1
            continue
        return windows



def window_bounds(windows, fps=FPS):
    """Per-window [start_seconds, end_seconds) pairs for a partition.

    Bounds run on the *rendered* timeline -- windows are concatenated head to
    tail, so each starts exactly where the previous ended. A shot is compiled
    into every window its own time range overlaps.
    """
    bounds = []
    cursor = 0.0
    for frames in windows:
        end = cursor + frames_to_seconds(frames, fps)
        bounds.append((cursor, end))
        cursor = end
    return bounds
