"""`patch_fingerprint`: what approximations are on the incoming model. Spec §12.4.

WHY THIS IS MANDATORY AND NOT AN OPTIMISATION
---------------------------------------------
Sol-Attn, Spectrum and EasyCache all change the output of the same prompt at the
same seed. A segment cache that ignores them will happily hand back window 4
rendered dense and window 5 rendered at tau=2.0, and the film's shots will not
match each other. Nothing in the UI would say so. An unlabelled cache is worse
than no cache, so §7.1 folds this fingerprint into every cache key and §14.8
forbids computing a key without it.

WHAT IS VERIFIED AND WHAT IS DUCK-TYPED
---------------------------------------
Two of the three packs are installed on the development machine and their keys
were read out of their source rather than guessed:

  * `ComfyUI-KJNodes` `PathchSageAttentionKJ.patch` sets
    `model_options["transformer_options"]["optimized_attention_override"]` to a
    closure named `attention_override_sage`. It records no mode anywhere, so the
    mode genuinely cannot be read back -- only presence can.
  * `ComfyUI-Spectrum-MiniMax-H3` `comfyui_spectrum_h3/sampling.py` sets
    `model_options["spectrum_h3_binding"]` (`BINDING_KEY`), and reads
    `transformer_options["easycache"]` when it checks for the conflict §12.7
    mentions.

`ComfyUI-sol-attn` is *not* installed here, so its key names are unverified. That
is exactly why detection below is written as a scan rather than as a table of
literal key names: every key on `model_options` and `transformer_options` whose
name carries a known pack fragment is folded in, whatever the pack chose to call
it, along with the identity of whatever callable is registered as the attention
override. A pack that renames a key still lands in the fingerprint; the worst
case is that it lands under a slightly different label, which changes the hash
once and re-renders once. The failure mode this module must not have -- silently
missing a patch and reusing a segment across it -- is not reachable that way.

DETERMINISM IS THE WHOLE POINT
------------------------------
Nothing here may put a memory address, an object id, a timestamp or a `repr()`
into the descriptor. A fingerprint that differs between two processes running the
identical graph would invalidate the cache on every restart, which is the same
bug as having no cache, only slower. Values are reduced to scalars and to
`module.qualname` strings, both of which are stable across runs.

Pure stdlib. No torch, no comfy -- the whole module is testable against stub
patchers, which is what tests/test_fingerprint.py does for each pack present and
absent.
"""

import hashlib

from .pulse_timeline import canonical_json

__all__ = [
    "PACK_FRAGMENTS",
    "EXTRA_FRAGMENTS",
    "describe_weight_patches",
    "describe_model_patches",
    "describe_patches",
    "patch_fingerprint",
    "fingerprint_from_shapes",
    "model_fingerprint",
    "sink_conditioning",
    "patch_chain_summary",
    "patch_warnings",
    "NO_PATCHES_WARNING",
]

#: Name fragments that identify a pack, checked case-insensitively against key
#: names, module paths and qualnames.
#:
#: Ordered MOST SPECIFIC FIRST, and the order is load-bearing twice over. Feed-
#: forward chunking ships *inside* the Sol-Attn pack, so its module path is
#: `comfyui_sol_attn.chunk_feedforward` and a Sol-first table would file it under
#: `sol_attn` and report the chain with one patch missing. Equally, Sol adopts the
#: Sage forward as its fallback backend (§12.3), so a Sol key naming Sage must
#: land under Sol. Both cases are resolved by putting the narrower fragment set
#: ahead of the broader one.
PACK_FRAGMENTS = (
    ("ff_chunk", ("chunk_feedforward", "chunk_ff", "ff_chunk", "feedforward_chunk",
                  "chunkfeedforward")),
    ("sol_attn", ("sol_attn", "sol-attn", "solattn", "sol_attention", "solattention")),
    ("easycache", ("easycache", "easy_cache")),
    ("spectrum", ("spectrum",)),
    ("sage", ("sage",)),
)

#: Patches recognised since the table above shipped. Checked the same way, but
#: written into a descriptor ONLY when present: every name in PACK_FRAGMENTS is
#: always listed (as `present: false` when absent), so adding one there would
#: change the fingerprint of every model on every box and empty every cache.
#:
#:   * ComfyUI core's own `BlockSparseAttention` node -- the VSA chain FastH3-Omni
#:     is trained for, and core's sol-attn -- installs an attention override from
#:     `comfy_extras.nodes_sparse_attention`. It used to land in the catch-all
#:     `attention_override` with its settings unread, so VSA at 10% and sol-attn
#:     at tau 1.3 hashed the same.
#:   * ComfyUI-HyperFlow records its adapter in `model_options["hyperflow"]`.
EXTRA_FRAGMENTS = (
    ("block_sparse", ("nodes_sparse_attention", "block_sparse_attention",
                      "sparseattnpatch")),
    ("hyperflow", ("hyperflow",)),
)

#: Attention override slot. Verified against ComfyUI core
#: (comfy/ldm/modules/attention.py dispatches through it) and against KJNodes.
OVERRIDE_PATH = ("transformer_options", "optimized_attention_override")

NO_PATCHES_WARNING = (
    "No attention or memory patches detected on the incoming model. Segment cache "
    "cannot protect against a patch change.")

# How deep to walk a settings object before giving up. Deep enough for a config
# dataclass holding a nested dict, shallow enough that a model reference caught
# in a closure cannot drag the whole graph into the hash.
_MAX_DEPTH = 4


def _fragment_owner(name):
    """Which pack a key name, module path or qualname belongs to, or None."""
    lowered = str(name).lower()
    for pack, fragments in PACK_FRAGMENTS + EXTRA_FRAGMENTS:
        if any(f in lowered for f in fragments):
            return pack
    return None


def _closure_settings(fn):
    """The scalar settings of the first configuration object a closure captured.

    Core's sparse attention registers a bare closure whose only readable state is
    the `SparseAttnPatch` it closed over -- method, keep ratio, tau, the sigma
    window, sink_conditioning. Reduced the same way `_jsonify` reduces any object:
    scalars and identities only, never an address.
    """
    for cell in (getattr(fn, "__closure__", None) or ()):
        try:
            value = cell.cell_contents
        except ValueError:  # pragma: no cover - empty cell
            continue
        if value is None or callable(value) or isinstance(
                value, (bool, int, float, str, dict, list, tuple, set)):
            continue
        settings = _jsonify(value)
        if isinstance(settings, dict) and len(settings) > 1:
            return settings
    return {}


def _shape_signature(value, depth=0):
    """Shapes of the tensors inside a weight patch, without importing torch."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(d) for d in shape]
        except Exception:  # pragma: no cover - exotic tensor wrapper
            return "<shape>"
    if depth >= 3:
        return None
    inner = getattr(value, "weights", None)
    if isinstance(inner, (list, tuple)):
        value = inner
    if isinstance(value, (list, tuple)):
        return [_shape_signature(v, depth + 1) for v in value
                if not isinstance(v, (str, bool, int, float)) or depth == 0]
    return None


def describe_weight_patches(patches):
    """LoRA-style weight patches on a ModelPatcher, as one stable entry, or None.

    `ModelPatcher.patches` maps a weight key to a list of
    `(strength_patch, patch, strength_model, offset, function)`. A LoRA changes
    every frame the model renders -- a TaoMate adapter at 1.0 against one at 0.6
    are different films -- and the shape-only model fingerprint cannot see it.
    Hashed over keys, strengths and patch shapes, never tensor contents: reading
    2.5 GB of adapter weights per queue to tell one LoRA from another of the same
    rank is not worth it, and two such files at the same strength are rare.
    """
    if not isinstance(patches, dict) or not patches:
        return None
    rows = []
    for key in sorted(patches, key=str):
        for entry in patches[key] or ():
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            strength_patch, patch, strength_model = entry[0], entry[1], entry[2]
            rows.append([str(key), _jsonify(strength_patch), _jsonify(strength_model),
                         _shape_signature(patch)])
    if not rows:
        return None
    digest = hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()[:16]
    return {"present": True, "count": len(rows), "digest": digest}


def _callable_identity(value):
    """`module.qualname` for a function or class. Never an address.

    A closure created inside `PathchSageAttentionKJ.patch` reports a qualname of
    `PathchSageAttentionKJ.patch.<locals>.attention_override_sage`, which names
    both the pack and the node -- everything the fingerprint needs, and stable
    across processes in a way `repr()` is not.
    """
    module = getattr(value, "__module__", "") or ""
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", "") or ""
    if not module and not qualname:
        return None
    return "%s.%s" % (module, qualname) if module else qualname


def _jsonify(value, depth=0):
    """Reduce an arbitrary patch setting to something canonical_json can hash.

    Scalars survive. Containers recurse. Callables and classes become their
    `module.qualname`. Anything else is reduced to its type's `module.qualname`
    plus whatever scalar attributes it exposes -- which is how a Sol-Attn config
    object's tau_start and dense_percent reach the hash without this module
    knowing that object's type exists.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # Rounded so that a settings float which round-trips through a widget at
        # a different precision does not invalidate every cached segment.
        return round(value, 6)
    if depth >= _MAX_DEPTH:
        return "<depth>"
    if isinstance(value, dict):
        return {str(k): _jsonify(v, depth + 1) for k, v in sorted(value.items(), key=_str_key)}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v, depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if callable(value):
        return _callable_identity(value) or "<callable>"

    identity = _callable_identity(type(value)) or "<object>"
    attrs = {}
    for key, attr in sorted(_public_attrs(value), key=_str_key):
        if isinstance(attr, (bool, int, float, str)) or attr is None:
            attrs[str(key)] = _jsonify(attr, depth + 1)
    return {"type": identity, **attrs} if attrs else {"type": identity}


def _str_key(item):
    return str(item[0])


def _public_attrs(value):
    """(name, value) for an object's own public scalar attributes. Never raises."""
    out = []
    slots = getattr(type(value), "__slots__", None)
    names = []
    try:
        names = list(getattr(value, "__dict__", {}) or {})
    except Exception:  # pragma: no cover - exotic objects
        names = []
    if isinstance(slots, str):
        names += [slots]
    elif slots:
        names += list(slots)
    for name in names:
        if str(name).startswith("_"):
            continue
        try:
            out.append((name, getattr(value, name)))
        except Exception:  # pragma: no cover - property that raises
            continue
    return out


def _scan(container, into, source):
    """Attribute every key in a dict to a pack, and record its settings."""
    if not isinstance(container, dict):
        return
    for key, value in container.items():
        pack = _fragment_owner(key)
        if pack is None and callable(value):
            # The key says nothing (`optimized_attention_override` is neutral by
            # design), so ask the callable who wrote it.
            pack = _fragment_owner(_callable_identity(value) or "")
        if pack is None:
            continue
        entry = into.setdefault(pack, {"present": True})
        entry["present"] = True
        entry.setdefault("evidence", []).append("%s.%s" % (source, key))
        settings = _jsonify(value)
        if isinstance(settings, dict):
            for k, v in settings.items():
                entry.setdefault(k, v)
        elif callable(value):
            # A patch registered as a bare closure -- KJNodes' Sage override is
            # exactly this -- carries its settings nowhere readable. Its identity
            # is all there is, and it is enough to tell one pack from another.
            entry.setdefault("callable", settings)
            # Except where the closure captured its configuration, which core's
            # sparse attention does. Read only for the packs recognised since
            # the original table, so no existing descriptor gains a field.
            if pack in dict(EXTRA_FRAGMENTS):
                for k, v in _closure_settings(value).items():
                    entry.setdefault(k, v)
        else:
            entry.setdefault("value", settings)


def describe_patches(model_options, sage_attention_global=False, object_patches=None,
                     weight_patches=None):
    """The canonical patch descriptor for one model. Spec §12.4.

    `model_options` is the dict off a ComfyUI ModelPatcher; `object_patches` is
    its object-patch registry, whose *keys* name the forwards a pack replaced
    (that is where feed-forward chunking shows up, since it patches an MLP rather
    than registering an attention override).

    Never raises. A detector that can throw inside a render is worse than one
    that can be wrong.
    """
    options = model_options if isinstance(model_options, dict) else {}
    transformer = options.get("transformer_options")
    transformer = transformer if isinstance(transformer, dict) else {}

    found = {}
    _scan(options, found, "model_options")
    _scan(transformer, found, "transformer_options")

    # Object patches: the keys are dotted module paths into the model
    # (`diffusion_model.blocks.0.ffn.forward`), so the pack name usually appears
    # in the *replacement*, not the key. Both are checked.
    if isinstance(object_patches, dict):
        for key, value in object_patches.items():
            pack = _fragment_owner(key) or _fragment_owner(_callable_identity(value) or "")
            if pack is None:
                continue
            entry = found.setdefault(pack, {"present": True})
            entry["present"] = True
            entry.setdefault("evidence", []).append("object_patches.%s" % (key,))

    # ComfyUI's own --use-sage-attention flag patches attention process-wide and
    # leaves nothing in model_options at all, so it has to be passed in.
    if sage_attention_global:
        entry = found.setdefault("sage", {"present": True})
        entry["present"] = True
        entry.setdefault("evidence", []).append("--use-sage-attention")

    # An attention override from a pack this module has no fragment for is still
    # an approximation, and still has to move the hash.
    override = transformer.get(OVERRIDE_PATH[1])
    if override is not None and not any(
            k in found for k in ("sol_attn", "sage", "block_sparse")):
        found["attention_override"] = {
            "present": True,
            "callable": _callable_identity(override) or "<callable>",
            "evidence": ["transformer_options.optimized_attention_override"],
        }

    descriptor = {name: {"present": False} for name, _ in PACK_FRAGMENTS}
    for pack, entry in found.items():
        entry["evidence"] = sorted(set(entry.get("evidence", [])))
        descriptor[pack] = entry
    # Only when there are any -- see EXTRA_FRAGMENTS for why absent is unlisted.
    weights = describe_weight_patches(weight_patches)
    if weights is not None:
        descriptor["weight_patches"] = weights

    descriptor["detected"] = any(v.get("present") for v in descriptor.values()
                                 if isinstance(v, dict))
    return descriptor


def describe_model_patches(model, sage_attention_global=False):
    """Same, straight from a MODEL object. Never raises."""
    return describe_patches(
        getattr(model, "model_options", None),
        sage_attention_global=sage_attention_global,
        object_patches=getattr(model, "object_patches", None),
        weight_patches=getattr(model, "patches", None))


def patch_fingerprint(descriptor):
    """sha256 of the descriptor, 16 hex chars. Spec §12.4.

    Feed-forward chunking is documented as bit-identical and could in principle
    be excluded. It is included anyway, per §12.4: an unnecessary re-render costs
    minutes and a wrong reuse costs the deliverable.
    """
    return hashlib.sha256(canonical_json(descriptor).encode("utf-8")).hexdigest()[:16]


def fingerprint_from_shapes(pairs):
    """`model_fingerprint` from `(param_name, shape_tuple)` pairs. Spec §7.1.

    The node layer produces the pairs (it needs torch to read a state dict); the
    hashing lives here so it is testable without one. Sorted before hashing
    because a state dict's iteration order is not part of the model's identity.
    """
    canonical = [[str(name), [int(d) for d in shape]] for name, shape in pairs]
    canonical.sort(key=lambda item: item[0])
    return hashlib.sha256(canonical_json(canonical).encode("utf-8")).hexdigest()[:16]


def model_fingerprint(model):
    """A short stable identity for the loaded checkpoint. Spec §7.1.

    Tries the cheap identities a patcher may already carry, then falls back to
    hashing parameter names and shapes. Returns `"unknown"` rather than raising:
    a run must not die because a future ComfyUI renamed an attribute, and
    `"unknown"` still hashes into the cache key consistently within a session.
    """
    for attr in ("model_fingerprint", "pulse_model_fingerprint"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value

    inner = getattr(model, "model", None) or model
    state_dict = getattr(inner, "state_dict", None)
    if not callable(state_dict):
        return "unknown"
    try:
        pairs = [(name, tuple(getattr(tensor, "shape", ()) or ()))
                 for name, tensor in state_dict().items()]
    except Exception:  # pragma: no cover - exotic model wrappers
        return "unknown"
    if not pairs:
        return "unknown"
    return fingerprint_from_shapes(pairs)


# ── reading a descriptor back ───────────────────────────────────────────────

def sink_conditioning(descriptor):
    """Sol-Attn's `sink_conditioning` setting, or None if not detectable.

    Read by §12.6's warning: `exact_kv_and_rows` is what keeps the *generated
    audio stream* exact, and PulseSlate pairs each character image with its own
    audio track for multi-character sync. That pairing lives in exactly the query
    rows the cheaper settings stop running dense.
    """
    for pack in ("sol_attn", "block_sparse"):
        value = ((descriptor or {}).get(pack) or {}).get("sink_conditioning")
        if isinstance(value, str):
            return value
    return None


def patch_chain_summary(descriptor):
    """One line per detected pack, for the §9.5 report section."""
    lines = []
    for pack, _ in PACK_FRAGMENTS + EXTRA_FRAGMENTS:
        entry = (descriptor or {}).get(pack) or {}
        if not entry.get("present"):
            continue
        settings = {k: v for k, v in entry.items()
                    if k not in ("present", "evidence", "type")}
        detail = ", ".join("%s=%s" % (k, settings[k]) for k in sorted(settings)) or "present"
        lines.append("%-10s %s" % (pack, detail))
    extra = (descriptor or {}).get("attention_override") or {}
    if extra.get("present"):
        lines.append("%-10s %s (unrecognised pack)" % ("attention", extra.get("callable", "")))
    weights = (descriptor or {}).get("weight_patches") or {}
    if weights.get("present"):
        lines.append("%-10s %d patched weight(s), %s" % ("lora", weights.get("count", 0),
                                                         weights.get("digest", "")))
    return lines


def patch_warnings(descriptor, paired_audio_count=0):
    """Everything §12 wants said about a detected chain. Warnings only, never fatal.

    `paired_audio_count` is the number of *characters* carrying a bound voice, not
    the number of audio files -- see `nodes._paired_audio_count`. The §12.6
    warning below asserts that this timeline pairs voices with separate
    characters, and that assertion has to be true before it is worth printing.
    """
    descriptor = descriptor or {}
    warnings = []

    if not descriptor.get("detected"):
        warnings.append(NO_PATCHES_WARNING)

    if (descriptor.get("spectrum") or {}).get("present") and \
            (descriptor.get("easycache") or {}).get("present"):
        # §12.7. Reported, never blocked -- "mutually exclusive in practice" is a
        # community finding, not a constraint this pack can verify.
        warnings.append(
            "Spectrum and EasyCache are both attached to this model. They are "
            "reported to be mutually exclusive in practice; if output degrades, "
            "disable one before investigating anything else.")

    sink = sink_conditioning(descriptor)
    if paired_audio_count > 1 and sink in ("exact_kv", "off"):
        # §12.6. This is the one setting whose wrong value damages precisely what
        # this pack is for.
        warnings.append(
            "This timeline binds a voice reference to each of %d separate "
            "characters, but "
            "the upstream Sol-Attn node reports sink_conditioning=%s. Only "
            "'exact_kv_and_rows' runs the conditioning query rows dense, and those "
            "are the rows that carry per-character audio sync. Expect the voices to "
            "drift between characters. The setting costs about 20%%; take it."
            % (paired_audio_count, sink))

    return warnings
