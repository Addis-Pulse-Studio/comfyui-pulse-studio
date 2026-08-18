"""Pulse Studio — ComfyUI custom node package.

Fork of muse-collective-26/MiniMaxH3-Director-Seed-Hunt (MIT). See NOTICE.
"""

import logging

log = logging.getLogger(__name__)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


# ── placeholder references for the Cast example ─────────────────────────────
# PulseSlate_Cast.json is the graph that demonstrates the asset bin, so it has to
# open on a populated bin. A ComfyUI workflow names its references by filename
# relative to the input directory, so shipping the files in the pack is not
# enough -- they have to be *in* input/ or every @mention opens unresolved, which
# is exactly the state tests/test_shipped_assets.py exists to keep out of a demo.
#
# Copied, never overwritten: a file the user has already put there under one of
# these names is theirs. Whole thing is best-effort -- a read-only or headless
# install must degrade to "Cast opens with unresolved references", never to a
# node pack that fails to load.

def _install_example_assets():
    import os
    import shutil

    source = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "example_workflows", "assets")
    if not os.path.isdir(source):
        return
    import folder_paths

    target = folder_paths.get_input_directory()
    os.makedirs(target, exist_ok=True)
    for name in sorted(os.listdir(source)):
        destination = os.path.join(target, name)
        if os.path.exists(destination):
            continue
        shutil.copyfile(os.path.join(source, name), destination)
        log.info("[PulseStudio] installed example reference %s", name)


try:
    _install_example_assets()
except Exception as exc:  # pragma: no cover - no ComfyUI, or a read-only input dir
    log.info("[PulseStudio] example references were not installed (%s); "
             "PulseSlate_Cast.json will open with unresolved references until the "
             "files in example_workflows/assets/ are copied into ComfyUI's input "
             "folder by hand.", exc)


# ── Asset Bin backend routes ────────────────────────────────────────────────
# The bin panel asks the server to evaluate and apply edits rather than
# reimplementing the numbering rule in JavaScript. There is exactly one
# implementation of that rule (comfyui_pulse_studio/assets.py), it is the tested one,
# and the UI reads its answers.
#
# Crucially, /apply returns a whole new timeline_data string produced by a MERGE
# on the server. The panel never builds that JSON itself, which is what stopped
# it from being able to erase the fields it does not know about.

try:
    from aiohttp import web
    from server import PromptServer

    from .comfyui_pulse_studio.assets import AssetBin, RefLimits
    from .comfyui_pulse_studio.binops import bin_state, preview_change, suggest_alias, suggest_name
    from .comfyui_pulse_studio.constants import MAX_REF_AUDIOS
    from .comfyui_pulse_studio.widget_state import (
        apply_bin_operation,
        load_timeline_document,
    )

    def _assets_of(data):
        """Assets from a posted timeline_data string, or from a raw list."""
        if "timeline_data" in data:
            return load_timeline_document(data.get("timeline_data"))["assets"]
        return data.get("assets") or []

    def _limits_of(data):
        """The reference budget this request is judged against.

        The panel sends the node's own `audio_ref_ceiling`, because the ceiling
        is a per-node widget and the server holds no session state. An absent or
        unusable value falls back to the documented budget rather than failing
        the request -- a bin panel that will not draw is worse than one drawing
        the conservative number.
        """
        try:
            return RefLimits(data.get("audio_ref_ceiling") or MAX_REF_AUDIOS)
        except (TypeError, ValueError) as exc:
            log.warning("[PulseStudio] ignoring bad audio_ref_ceiling %r: %s",
                        data.get("audio_ref_ceiling"), exc)
            return RefLimits()

    @PromptServer.instance.routes.post("/pulse_studio/bin_state")
    async def _bin_state(request):
        """Live tags, budget meter and name problems for a bin."""
        try:
            data = await request.json()
            bin_ = AssetBin.from_list(_assets_of(data))
            return web.json_response(
                {"status": "ok", "state": bin_state(bin_, limits=_limits_of(data))})
        except Exception as exc:
            log.warning("[PulseStudio] bin_state failed: %s", exc)
            return web.json_response({"status": "error", "message": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/pulse_studio/preview_change")
    async def _preview_change(request):
        """What a pending edit would renumber, without applying it."""
        try:
            data = await request.json()
            bin_ = AssetBin.from_list(_assets_of(data))
            kwargs = dict(data.get("kwargs") or {})
            deltas, error = preview_change(bin_, data.get("operation"),
                                           limits=_limits_of(data), **kwargs)
            return web.json_response({"status": "ok", "error": error,
                                      "deltas": [d.to_dict() for d in deltas]})
        except Exception as exc:
            log.warning("[PulseStudio] preview_change failed: %s", exc)
            return web.json_response({"status": "error", "message": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/pulse_studio/apply")
    async def _apply(request):
        """Apply an edit and return the merged timeline_data.

        The merge happens here, in tested Python, so that shots, duration, style
        and any key a future version adds survive an asset edit untouched.
        """
        try:
            data = await request.json()
            raw = data.get("timeline_data")
            kwargs = dict(data.get("kwargs") or {})
            new_raw, error = apply_bin_operation(raw, data.get("operation"),
                                                 limits=_limits_of(data), **kwargs)
            return web.json_response({"status": "ok", "error": error,
                                      "timeline_data": new_raw})
        except Exception as exc:
            log.warning("[PulseStudio] apply failed: %s", exc)
            return web.json_response({"status": "error", "message": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/pulse_studio/suggest_name")
    async def _suggest_name(request):
        try:
            data = await request.json()
            bin_ = AssetBin.from_list(_assets_of(data))
            if data.get("kind"):
                name = suggest_alias(bin_, data["kind"])
            else:
                name = suggest_name(bin_, data.get("base") or "Asset")
            return web.json_response({"status": "ok", "name": name})
        except Exception as exc:
            return web.json_response({"status": "error", "message": str(exc)}, status=400)

except Exception as exc:  # pragma: no cover - ComfyUI server not present
    log.info("[PulseStudio] Asset Bin routes not registered (%s); the node still works "
             "with a raw JSON timeline_data widget.", exc)
