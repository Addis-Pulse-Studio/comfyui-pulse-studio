"""Asset Bin operations, as the UI drives them.

assets.py holds the data model and the numbering rule. This module is the layer
the canvas talks to: it answers "what does the bin look like right now", "may I
drop this", and -- the one that earns its keep -- "what will this edit renumber?"

That last question matters because reordering the bin silently rewrites every
ordinal downstream of the move. The user is entitled to see that before they let
go of the mouse, not to discover it in a render.
"""

from .assets import (
    KIND_AUDIO,
    KIND_IMAGE,
    KIND_VIDEO,
    Asset,
    AssetBin,
    BudgetError,
)
from .constants import RETENTION_DEFAULT, RETENTION_VALUES

__all__ = [
    "bin_state",
    "preview_change",
    "name_problems",
    "suggest_name",
    "apply_operation",
    "RenumberDelta",
    "RETENTION_DEFAULT",
    "RETENTION_VALUES",
]


def bin_state(bin_, limits=None):
    """Everything the Asset Bin panel needs to draw itself, in one dict.

    Tags are live: computed from current order, never stored. The panel shows
    them so the user can read the mapping, but no part of the system reads them
    back -- prompt text refers to assets by name or id.
    """
    tm = bin_.tag_map()
    report = bin_.budget(limits=limits)
    rows = []
    for asset in bin_:
        row = {
            "id": asset.asset_id,
            "kind": asset.kind,
            "name": asset.name,
            "file": asset.file,
            "description": asset.description,
            "retention": asset.retention,
            "tag": tm.by_id.get(asset.asset_id),
            "synthetic": asset.synthetic,
        }
        if asset.kind == KIND_VIDEO:
            row["include_audio"] = asset.include_audio
            row["soundtrack_tag"] = tm.by_id.get(asset.asset_id + "#soundtrack")
            row["trim_start"] = asset.trim_start
            row["trim_end"] = asset.trim_end
        rows.append(row)
    return {
        "assets": rows,
        "budget": report.to_dict(),
        "name_problems": name_problems(bin_),
        # Sent rather than hard-coded in the panel so the vocabulary has one
        # home. The panel renders whatever options it is given.
        "retention_values": list(RETENTION_VALUES),
        "retention_default": RETENTION_DEFAULT,
    }


class RenumberDelta:
    """One asset whose tag changes as a result of a pending edit."""

    __slots__ = ("asset_id", "name", "kind", "before", "after")

    def __init__(self, asset_id, name, kind, before, after):
        self.asset_id = asset_id
        self.name = name
        self.kind = kind
        self.before = before
        self.after = after

    @property
    def removed(self):
        return self.after is None

    @property
    def added(self):
        return self.before is None

    def to_dict(self):
        return {"asset_id": self.asset_id, "name": self.name, "kind": self.kind,
                "before": self.before, "after": self.after}

    def __repr__(self):
        return "RenumberDelta(%s: %s -> %s)" % (self.name, self.before, self.after)


def preview_change(bin_, operation, limits=None, **kwargs):
    """What would this edit renumber? Returns (deltas, error).

    Runs the operation against a copy, so the live bin is never touched. `error`
    is a string when the operation would be refused (over budget, unknown id),
    in which case `deltas` is empty.

    operation is 'add' | 'remove' | 'move' | 'set_include_audio' | 'rename'.
    """
    before_map = bin_.tag_map().by_id
    trial = AssetBin.from_list(bin_.to_list())
    try:
        apply_operation(trial, operation, limits=limits, **kwargs)
    except (BudgetError, KeyError, ValueError) as exc:
        return [], str(exc) or repr(exc)

    after_map = trial.tag_map().by_id
    deltas = []
    for key in sorted(set(before_map) | set(after_map)):
        b, a = before_map.get(key), after_map.get(key)
        if b == a:
            continue
        asset_id = key.split("#", 1)[0]
        asset = trial.get(asset_id) or bin_.get(asset_id)
        label = asset.name if asset else asset_id
        kind = asset.kind if asset else ""
        if key.endswith("#soundtrack"):
            label += " (soundtrack)"
        deltas.append(RenumberDelta(key, label, kind, b, a))
    return deltas, None


def apply_operation(bin_, operation, limits=None, **kwargs):
    """Dispatch a named bin edit. Raises on refusal, leaving the bin unchanged.

    `limits` is the reference budget this edit is judged against; only `add` can
    be refused by it. Passing it here rather than reading module scope is what
    lets a project running a raised audio ceiling accept its fourth audio drop
    in the panel, instead of accepting it at compile time and refusing it at the
    drop -- which would be the worst of both.
    """
    if operation == "add":
        asset = kwargs.get("asset")
        if not isinstance(asset, Asset):
            asset = Asset.from_dict(asset)
        return bin_.add(asset, index=kwargs.get("index"), limits=limits)
    if operation == "remove":
        return bin_.remove(kwargs["asset_id"])
    if operation == "move":
        return bin_.move(kwargs["asset_id"], kwargs["new_index"])
    if operation == "set_include_audio":
        return bin_.set_include_audio(kwargs["asset_id"], kwargs["include"])
    if operation == "rename":
        return _rename(bin_, kwargs["asset_id"], kwargs["name"])
    if operation == "set_description":
        return _set_description(bin_, kwargs["asset_id"], kwargs["description"])
    if operation == "set_retention":
        return _set_retention(bin_, kwargs["asset_id"], kwargs["retention"])
    raise ValueError("unknown bin operation %r" % (operation,))


def _set_description(bin_, asset_id, description):
    """Set the one-line description that promotes an asset to a <Subject N>.

    This is what the panel was missing. `compiler._build_subjects` promotes an
    image or video to `<Subject N>` only if it carries a description, and until
    2026-08-17 nothing in the panel could write one -- so MiniMax's own reference
    format was unreachable without hand-editing timeline_data.

    An empty description is accepted rather than refused. Clearing one demotes the
    asset back to a plain `<Picture i>` citation, which is a legitimate edit; the
    asset keeps its socket and its ordinal either way.
    """
    asset = bin_.get(asset_id)
    if asset is None:
        raise KeyError(asset_id)
    if asset.kind == KIND_AUDIO:
        raise ValueError("audio references have no <Subject N> form, so a description "
                         "on one would never reach the prompt")
    asset.description = (description or "").strip()
    return asset


def _set_retention(bin_, asset_id, retention):
    """Set how much of the reference the model is told to preserve.

    Free text in the document schema, but not free text here: the value is written
    verbatim into `retention_analysis`, which the model reads literally, and a typo
    there is a silent failure of exactly the kind this pack exists to catch. The
    vocabulary is the one the compiler already emits for carry-over anchors.
    """
    asset = bin_.get(asset_id)
    if asset is None:
        raise KeyError(asset_id)
    if asset.kind == KIND_AUDIO:
        raise ValueError("retention describes a visual subject; audio references "
                         "carry an audio_role instead")
    value = (retention or "").strip()
    if value not in RETENTION_VALUES:
        raise ValueError("retention must be one of %s, got %r"
                         % (", ".join(sorted(RETENTION_VALUES)), retention))
    asset.retention = value
    return asset


def _rename(bin_, asset_id, name):
    """Rename, refusing a collision.

    Names are how prompt text addresses assets (@Mimi), so two assets sharing one
    makes every reference to it ambiguous -- and an ambiguous reference resolves
    to nothing rather than guessing. Refusing the rename keeps that from ever
    happening.
    """
    asset = bin_.get(asset_id)
    if asset is None:
        raise KeyError(asset_id)
    new = (name or "").strip()
    if not new:
        raise ValueError("an asset name cannot be empty -- prompt text addresses assets by name")
    clash = bin_.find_by_name(new)
    if clash is not None and clash.asset_id != asset_id:
        raise ValueError("another asset is already named %r; names must be unique so that "
                         "@%s resolves to exactly one thing" % (new, new))
    # find_by_name returns None for an *ambiguous* match too, so check directly.
    existing = [a for a in bin_ if a.asset_id != asset_id
                and a.name.strip().casefold() == new.casefold()]
    if existing:
        raise ValueError("another asset is already named %r; names must be unique" % (new,))
    asset.name = new
    return asset


def name_problems(bin_):
    """Names that cannot be used in prompt text, with the reason.

    Reported rather than auto-corrected: renaming someone's character behind
    their back is worse than telling them the reference will not resolve.
    """
    problems = []
    seen = {}
    for asset in bin_:
        key = asset.name.strip().casefold()
        if not key:
            problems.append({"asset_id": asset.asset_id,
                             "problem": "has no name, so it cannot be referenced by @name"})
            continue
        seen.setdefault(key, []).append(asset)
    for key, group in seen.items():
        if len(group) > 1:
            problems.append({
                "asset_id": group[0].asset_id,
                "problem": "%d assets share the name %r; @%s will not resolve to any of them"
                           % (len(group), group[0].name, group[0].name),
            })
    return problems


def suggest_name(bin_, base):
    """A unique name derived from `base` ('Mimi' -> 'Mimi 2' when taken)."""
    base = (base or "Asset").strip() or "Asset"
    taken = {a.name.strip().casefold() for a in bin_}
    if base.casefold() not in taken:
        return base
    n = 2
    while ("%s %d" % (base, n)).casefold() in taken:
        n += 1
    return "%s %d" % (base, n)


ALIAS_STEMS = {KIND_IMAGE: "Image", KIND_VIDEO: "Video", KIND_AUDIO: "Audio"}


def suggest_alias(bin_, kind):
    """A short, typeable alias for a freshly dropped asset: Image1, Video2, Audio1.

    Assigned on drop so the user never has to type a filename. Aliases are per
    kind and count from 1, and they are only a starting point -- renaming to
    'Mimi' or 'Cafe wide' is the expected next step, and prompt text follows the
    name wherever it goes.
    """
    stem = ALIAS_STEMS.get(kind, "Asset")
    taken = {a.name.strip().casefold() for a in bin_}
    n = 1
    while ("%s%d" % (stem, n)).casefold() in taken:
        n += 1
    return "%s%d" % (stem, n)
