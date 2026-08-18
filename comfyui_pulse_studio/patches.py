"""Detecting whether the incoming MODEL has been patched, and saying so.

WHY THIS NODE DOES NOT APPLY THE PATCHES ITSELF
-----------------------------------------------
Attention and offload patches belong to other node packs. This pack consumes a
patched `MODEL`; it does not apply one, and it must not take a dependency on the
packs that do -- a hard dependency would make this node uninstallable for anyone
who wants it without them, and a soft one would drift the moment their key names
change.

So detection is duck-typed over `model_options`, by key presence only. Nothing
here imports Spectrum, Sage Attention, or anything else. If a pack renames its
key, the worst case is a spurious warning, never a broken render.

WHY THE PATCH MUST BE UPSTREAM OF THIS NODE
-------------------------------------------
The documented chain is:

    UNETLoader -> Spectrum -> Sage Attention -> PulseSlate

The order matters and is not cosmetic. On a timeline longer than one window this
node **samples internally**, using the model handed to its input. A patch applied
to the node's `model` *output* would accelerate the pass-through path (a single
window, where the user's own sampler runs) and silently do nothing at all for
long timelines -- which is the exact case where it matters most. The failure is
invisible: the render works, it is just slow, or it runs out of memory on a
window that would otherwise have fitted.

WHY OFFLOAD IS NOT A SPEED KNOB
-------------------------------
On a 32 GB card, Spectrum's `system_ram` history storage is what makes a
362-frame window fit **at all**. Describing offload as an optimisation invites
the user to turn it off for speed and then meet an out-of-memory error they have
no reason to connect to it. The warning below says so in those terms.

No finding here ever blocks execution. The user may be running deliberately
unpatched -- on a bigger card, on CPU, or while measuring what the patches
actually buy them.

Almost everything here is a WARNING. The one exception is
`check_single_checkpoint`, which returns `(warnings, notes)`: a checkpoint wired
but never sampled with is a real cost on a 32 GB box and used to be reported
nowhere, but it is not a mistake, and giving it the same voice as a mid-render
evict-and-reload would devalue both.

Pure stdlib. No torch, no comfy, no folder_paths -- so the whole thing is
testable against a stub model with an empty `model_options`.
"""

__all__ = [
    "PatchReport",
    "inspect_model_options",
    "check_model_patches",
    "check_single_checkpoint",
    "ATTENTION_KEYS",
    "OFFLOAD_KEYS",
    "RECOMMENDED_CHAIN",
]

RECOMMENDED_CHAIN = "UNETLoader -> Spectrum -> Sage Attention -> PulseSlate"

# Keys that indicate an attention patch. Verified against ComfyUI core:
# comfy/ldm/modules/attention.py dispatches through
# transformer_options["optimized_attention_override"], which is the supported
# hook every attention pack routes through. `patches_replace` is the older
# block-replacement path.
ATTENTION_KEYS = (
    ("transformer_options", "optimized_attention_override"),
    ("transformer_options", "patches_replace"),
    ("model_function_wrapper",),
)

# Keys that indicate an offload / memory-management patch. Verified against:
#   comfy/ (multigpu_clones, multigpu_options, offload_device)
#   ComfyUI-Spectrum-MiniMax-H3 comfyui_spectrum_h3/sampling.py
#     BINDING_KEY = "spectrum_h3_binding"
OFFLOAD_KEYS = (
    ("spectrum_h3_binding",),
    ("multigpu_clones",),
    ("multigpu_options",),
    ("offload_device",),
)


class PatchReport:
    """What was found on an incoming model, and what to tell the user."""

    __slots__ = ("has_attention", "has_offload", "attention_evidence",
                 "offload_evidence", "warnings")

    def __init__(self, has_attention, has_offload, attention_evidence,
                 offload_evidence, warnings):
        self.has_attention = has_attention
        self.has_offload = has_offload
        self.attention_evidence = attention_evidence
        self.offload_evidence = offload_evidence
        self.warnings = warnings

    @property
    def ok(self):
        return not self.warnings

    def to_dict(self):
        return {
            "has_attention": self.has_attention,
            "has_offload": self.has_offload,
            "attention_evidence": self.attention_evidence,
            "offload_evidence": self.offload_evidence,
            "warnings": list(self.warnings),
        }

    def __repr__(self):  # pragma: no cover - debugging aid
        return "PatchReport(attention=%r, offload=%r, warnings=%d)" % (
            self.has_attention, self.has_offload, len(self.warnings))


def _lookup(options, path):
    """Walk a key path through nested dicts. Returns (found, key_string)."""
    node = options
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    # A key present but explicitly empty is not evidence of a patch: ComfyUI
    # itself creates transformer_options and patches_replace as empty dicts.
    if isinstance(node, (dict, list, tuple)) and not node:
        return False, None
    return True, ".".join(path)


def _find(options, key_paths):
    for path in key_paths:
        found, name = _lookup(options, path)
        if found:
            return name
    return None


def inspect_model_options(model_options, sage_attention_global=False):
    """Report which patches are present on a model, and what is missing.

    `model_options` is the dict from a ComfyUI ModelPatcher. `None` and any
    non-dict are treated as "no patches" rather than raising -- this runs inside
    a render, and a detector that can throw is worse than one that can be wrong.

    `sage_attention_global` covers ComfyUI's own `--use-sage-attention` launch
    flag, which patches attention process-wide and therefore leaves no trace in
    model_options at all. The caller reads it from
    comfy.model_management.sage_attention_enabled(); this module stays pure.
    """
    options = model_options if isinstance(model_options, dict) else {}

    attention_evidence = _find(options, ATTENTION_KEYS)
    if attention_evidence is None and sage_attention_global:
        attention_evidence = "--use-sage-attention"
    offload_evidence = _find(options, OFFLOAD_KEYS)

    has_attention = attention_evidence is not None
    has_offload = offload_evidence is not None

    warnings = []
    if not has_attention:
        warnings.append(
            "No attention patch detected on the incoming model. Sage Attention "
            "(or ComfyUI's own --use-sage-attention) is what makes a 362-frame "
            "window sample in a workable time. Recommended chain: %s."
            % RECOMMENDED_CHAIN)
    if not has_offload:
        # §18.1: offload is mandatory on this hardware, not tunable. Say that,
        # rather than describing it as an optimisation the user may skip.
        warnings.append(
            "No offload patch detected on the incoming model. This is not a "
            "speed setting: on a 32 GB card, Spectrum's system_ram history "
            "storage is what makes a 362-frame window fit at all. Without it a "
            "full-length window is likely to fail with an out-of-memory error "
            "rather than run slowly. Recommended chain: %s." % RECOMMENDED_CHAIN)
    if warnings:
        warnings.append(
            "Patch upstream of this node, not downstream of it. A timeline "
            "longer than one window samples internally with the model wired "
            "into this node's input, so a patch applied to its output would "
            "speed up the single-window path and do nothing for long "
            "timelines -- the case where it matters most.")

    return PatchReport(has_attention, has_offload, attention_evidence,
                       offload_evidence, warnings)


def check_model_patches(model, sage_attention_global=False):
    """Same, straight from a MODEL object. Never raises."""
    return inspect_model_options(getattr(model, "model_options", None),
                                 sage_attention_global=sage_attention_global)


def check_single_checkpoint(branches_used, fl2va_connected):
    """§18.1: the cost of having both DiT checkpoints in one graph.

    Returns (warnings, notes) -- the module's only two-tier finding, and the
    reason the tier exists at all. Everything else here is a warning; these two
    cases are genuinely different in severity and collapsing them would either
    hide the cheap one or cry wolf about it.

    ref2va and fl2va are ~21 GB each and the text encoder is large.

    WARNING -- *sampling* through both inside one execution forces an
    evict-and-reload mid-render on a 32 GB card, which costs more than the render
    itself.

    NOTE -- connecting model_fl2va without any window needing it was silent until
    2026-08-17, on the reasoning that it is "just a graph ready for either path".
    That reasoning assumed a checkpoint nobody samples with is free. It is not:
    with Spectrum offloading to system RAM on a 32 GB box, a 20 GB checkpoint
    nobody samples with is still 20 GB of system RAM, and the room it takes is the
    room the render wanted. Not an error -- keeping the socket wired is a
    perfectly reasonable way to work -- so it is a note, and it says what it
    costs rather than telling anyone to change anything.
    """
    used = {b for b in (branches_used or ()) if b}
    if not fl2va_connected:
        return [], []
    if len(used) < 2:
        return [], [
            "model_fl2va is connected but no window uses it -- every window on "
            "this timeline renders through %s. The checkpoint is still resident: "
            "~20 GB, and on a 32 GB box with Spectrum offloading to system RAM "
            "that is 20 GB the render cannot have. Disconnect it if this timeline "
            "is not going to need an anchored window."
            % (", ".join(sorted(used)) or "the reference branch")
        ]
    return [
        "Both DiT checkpoints are live in this render: %s. They are ~21 GB each, "
        "so on a 32 GB card this forces an evict-and-reload mid-render. Split the "
        "timeline so one graph uses one checkpoint." % ", ".join(sorted(used))
    ], []
