"""Pulse Studio — headless core.

Nothing in this package imports torch, comfy, or any ComfyUI module. That is
deliberate: the compiler, the frame grid, and the asset bin are where the
correctness lives, and they are fully testable without a GPU or a running
ComfyUI. The node layer that binds this to ComfyUI lives in ../nodes.py.
"""

from .assets import Asset, AssetBin, BudgetError, BudgetReport
from .compiler import CarryPolicy, CompiledPlan, CompiledWindow, compile_timeline
from .constants import BRANCH_FL2VA, BRANCH_REF2VA, MAX_WINDOW_FRAMES
from .frames import align_frame_count, partition_windows, snap_frames
from .timeline import Shot, Timeline, TimelineError

__version__ = "0.1.0"

__all__ = [
    "Asset", "AssetBin", "BudgetError", "BudgetReport",
    "CarryPolicy", "CompiledPlan", "CompiledWindow", "compile_timeline",
    "Shot", "Timeline", "TimelineError",
    "align_frame_count", "partition_windows", "snap_frames",
    "BRANCH_FL2VA", "BRANCH_REF2VA", "MAX_WINDOW_FRAMES",
    "__version__",
]
