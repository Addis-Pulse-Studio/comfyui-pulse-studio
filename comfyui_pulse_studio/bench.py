"""`PulseBench`: measured cost per patch chain, read back out of manifests. Spec §12.7.

Sol-Attn reduces attention and MLP *activation* peak. Spectrum's `system_ram`
history offload is a different mechanism entirely, and it is what currently makes
a 362-frame window fit on a 32 GB card. They are not substitutes, and community
reports on Spectrum's effect on *speed* disagree with each other -- one has it
costing time rather than saving it.

The spec's answer to that is not to pick a side in code. `PulseRender` already
writes `render_seconds`, `peak_vram_bytes` and `patch_fingerprint` into every
segment entry, so the question "which chain is faster on this box" has an answer
sitting on disk. This module groups those numbers by fingerprint and prints them,
which turns the argument into a lookup.

Pure stdlib -- reads JSON files, imports nothing from ComfyUI.
"""

import json
import os

from .segcache import MANIFEST_NAME, STATUS_COMPLETE

__all__ = ["load_manifests", "group_by_fingerprint", "format_table"]


def load_manifests(paths):
    """Load manifests from files or run directories. Returns (manifests, problems)."""
    manifests, problems = [], []
    for path in paths or []:
        path = str(path).strip()
        if not path:
            continue
        candidate = os.path.join(path, MANIFEST_NAME) if os.path.isdir(path) else path
        if not os.path.exists(candidate):
            problems.append("no manifest at %s" % (candidate,))
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (ValueError, OSError) as exc:
            problems.append("could not read %s: %s" % (candidate, exc))
            continue
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            data["_path"] = candidate
            manifests.append(data)
        else:
            problems.append("%s is not a segment manifest" % (candidate,))
    return manifests, problems


def group_by_fingerprint(manifests):
    """`patch_fingerprint -> aggregate`, over completed segments only.

    Warm-up segments are counted separately rather than dropped: the Triton
    autotune sweep is a real cost the user pays once per distinct sequence length
    (§12.5), and a table that hid it would understate the cost of a ragged plan.
    """
    groups = {}
    for manifest in manifests or []:
        for entry in manifest.get("segments") or []:
            if entry.get("status") != STATUS_COMPLETE:
                continue
            frames = int(entry.get("frames") or 0)
            seconds = float(entry.get("render_seconds") or 0.0)
            if frames <= 0 or seconds <= 0:
                continue
            key = entry.get("patch_fingerprint") or "(none recorded)"
            group = groups.setdefault(key, {
                "segments": 0, "frames": 0, "seconds": 0.0,
                "warmup_segments": 0, "warmup_seconds": 0.0,
                "peak_vram_bytes": 0, "runs": set(),
            })
            group["runs"].add(manifest.get("run_id") or manifest.get("_path", ""))
            if entry.get("warmup"):
                group["warmup_segments"] += 1
                group["warmup_seconds"] += seconds
                continue
            group["segments"] += 1
            group["frames"] += frames
            group["seconds"] += seconds
            group["peak_vram_bytes"] = max(group["peak_vram_bytes"],
                                           int(entry.get("peak_vram_bytes") or 0))

    for group in groups.values():
        group["runs"] = sorted(group["runs"])
        group["seconds_per_frame"] = (group["seconds"] / group["frames"]
                                      if group["frames"] else None)
    return groups


def format_table(groups, problems=()):
    """The plain-text table `PulseBench` outputs."""
    lines = []
    lines.append("=" * 92)
    lines.append("PulseBench -- measured cost grouped by patch_fingerprint")
    lines.append("=" * 92)
    if not groups:
        lines.append("")
        lines.append("No completed, timed segments found.")
        lines.append("Point run_dirs at a PulseRender run folder (the one holding "
                     "manifest.json) after at least one window has rendered.")
    else:
        lines.append("%-18s %8s %9s %11s %11s %9s"
                     % ("patch_fingerprint", "segments", "frames", "sec/frame",
                        "peak VRAM", "warmup"))
        lines.append("-" * 92)
        for key in sorted(groups, key=lambda k: (groups[k]["seconds_per_frame"] is None,
                                                 groups[k]["seconds_per_frame"] or 0)):
            group = groups[key]
            spf = group["seconds_per_frame"]
            vram = group["peak_vram_bytes"]
            lines.append("%-18s %8d %9d %11s %11s %9s"
                         % (key, group["segments"], group["frames"],
                            ("%.3f" % spf) if spf else "-",
                            ("%.1f GiB" % (vram / (1024 ** 3))) if vram else "-",
                            ("%.0fs x%d" % (group["warmup_seconds"],
                                            group["warmup_segments"]))
                            if group["warmup_segments"] else "-"))
        lines.append("")
        lines.append("Lower sec/frame is faster. Compare rows only within one box and one "
                     "canvas -- sequence length drives both columns, so a 736p row and a "
                     "1080p row are not comparable.")
        lines.append("The 'warmup' column is Sol-Attn's Triton autotune sweep, paid once "
                     "per distinct packed sequence length (spec §12.5).")

    if problems:
        lines.append("")
        lines.append("Problems:")
        for problem in problems:
            lines.append("    - %s" % (problem,))
    return "\n".join(lines)
