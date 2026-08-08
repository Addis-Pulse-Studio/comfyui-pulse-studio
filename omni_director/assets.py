"""The Asset Bin: named references, the budget that governs them, and the one
place where <Picture i> / <Video k> / <Audio j> ordinals are assigned.

Why this module exists at all
----------------------------
H3's reference tags are positional. `comfy/text_encoders/minimax.py` numbers them
with a running counter over `ref_items`, so an ordinal means nothing more than
"the nth of its type in socket order". Insert an image at the top of the bin and
every <Picture N> after it shifts by one -- while the prompt text that referenced
them stays exactly as the user typed it. The render still succeeds. It just
describes the wrong pictures. That is the failure this package is built to make
impossible: plausible-looking wrong output, with no error anywhere.

The fix is to never let a human type an ordinal. Authors reference assets by
stable id or by name; ordinals are derived here, at compile time, from the live
bin order. Renumbering then cannot desynchronise, because there is only ever one
number and it is computed, not stored.
"""

import re

from .constants import (
    MAX_REF_AUDIOS,
    MAX_REF_FILES_TOTAL,
    MAX_REF_IMAGES,
    MAX_REF_VIDEO_AUDIOS,
    MAX_REF_VIDEOS,
    REF_VIDEO_MAX_SECONDS,
    REF_VIDEO_MIN_SECONDS,
    TAG_AUDIO,
    TAG_IMAGE,
    TAG_VIDEO,
)

__all__ = [
    "Asset",
    "AssetBin",
    "BudgetError",
    "BudgetReport",
    "TagMap",
    "KIND_IMAGE",
    "KIND_VIDEO",
    "KIND_AUDIO",
]

KIND_IMAGE = "image"
KIND_VIDEO = "video"
KIND_AUDIO = "audio"
KINDS = (KIND_IMAGE, KIND_VIDEO, KIND_AUDIO)


class BudgetError(ValueError):
    """Raised when an operation would exceed H3's reference budget.

    Raised at edit time (on the drop that would break it), never at queue time --
    the whole point is that the bin refuses the twelfth-plus file while the user
    is still looking at it.
    """


class Asset:
    """One reference file in the bin.

    `asset_id` is stable for the lifetime of the asset and is what prompt text
    refers to. `name` is the human handle ("Mimi", "Cafe wide") and may be
    edited freely. Neither carries an ordinal -- see the module docstring.
    """

    __slots__ = ("asset_id", "kind", "name", "file", "description", "retention",
                 "trim_start", "trim_end", "include_audio", "synthetic")

    def __init__(self, asset_id, kind, name="", file="", description="",
                 retention=None, trim_start=0.0, trim_end=None, include_audio=False,
                 synthetic=False):
        if kind not in KINDS:
            raise ValueError("unknown asset kind %r (expected one of %r)" % (kind, KINDS))
        self.asset_id = str(asset_id)
        self.kind = kind
        self.name = name or self.asset_id
        self.file = file
        self.description = description
        self.retention = retention
        self.trim_start = float(trim_start or 0.0)
        self.trim_end = None if trim_end is None else float(trim_end)
        # Only meaningful on a video: pull that file's own embedded soundtrack in
        # as a paired ref_video_audio. It consumes an <Audio j> ordinal but not a
        # file slot, because it is the same file already counted as the video.
        self.include_audio = bool(include_audio) and kind == KIND_VIDEO
        # Synthetic assets are produced by the pipeline, not dropped in by the
        # user -- window carry-over frames and audio tails. They occupy a real
        # socket and a real ordinal, so every per-type limit still applies to
        # them, but they are not files and must not move the 12-file meter.
        self.synthetic = bool(synthetic)

    @property
    def duration(self):
        """Trimmed duration in seconds, or None when the trim is open-ended."""
        if self.trim_end is None:
            return None
        return max(0.0, self.trim_end - self.trim_start)

    def to_dict(self):
        d = {"id": self.asset_id, "kind": self.kind, "name": self.name, "file": self.file}
        if self.description:
            d["description"] = self.description
        if self.retention:
            d["retention"] = self.retention
        if self.trim_start:
            d["trim_start"] = self.trim_start
        if self.trim_end is not None:
            d["trim_end"] = self.trim_end
        if self.include_audio:
            d["include_audio"] = True
        if self.synthetic:
            d["synthetic"] = True
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(
            asset_id=d.get("id") or d.get("asset_id"),
            kind=d.get("kind") or d.get("type"),
            name=d.get("name", ""),
            file=d.get("file", ""),
            description=d.get("description", ""),
            retention=d.get("retention"),
            trim_start=d.get("trim_start", 0.0),
            trim_end=d.get("trim_end"),
            include_audio=d.get("include_audio", False),
            synthetic=d.get("synthetic", False),
        )

    def __repr__(self):
        return "Asset(%r, %s, %r)" % (self.asset_id, self.kind, self.name)


class BudgetReport:
    """A live snapshot of bin occupancy, for the meter and for validation."""

    __slots__ = ("images", "videos", "soundtracks", "audios", "files", "errors")

    def __init__(self, images, videos, soundtracks, audios, files, errors):
        self.images = images
        self.videos = videos
        self.soundtracks = soundtracks
        self.audios = audios
        self.files = files
        self.errors = errors

    @property
    def ok(self):
        return not self.errors

    def meter(self):
        """The one-line string the Asset Bin shows, e.g.
        '7/9 images | 2/3 videos | 1/3 audio | 9/12 files'."""
        return "%d/%d images | %d/%d videos | %d/%d audio | %d/%d files" % (
            self.images, MAX_REF_IMAGES,
            self.videos, MAX_REF_VIDEOS,
            self.audios, MAX_REF_AUDIOS,
            self.files, MAX_REF_FILES_TOTAL,
        )

    def to_dict(self):
        return {
            "images": self.images, "max_images": MAX_REF_IMAGES,
            "videos": self.videos, "max_videos": MAX_REF_VIDEOS,
            "soundtracks": self.soundtracks, "max_soundtracks": MAX_REF_VIDEO_AUDIOS,
            "audios": self.audios, "max_audios": MAX_REF_AUDIOS,
            "files": self.files, "max_files": MAX_REF_FILES_TOTAL,
            "ok": self.ok, "errors": list(self.errors), "meter": self.meter(),
        }

    def __repr__(self):
        return "BudgetReport(%s, ok=%s)" % (self.meter(), self.ok)


class TagMap:
    """Resolved ordinals for one bin state, plus the socket layout that produced them.

    Built by AssetBin.tag_map(). Holds no opinion of its own -- it is a pure
    function of bin order, recomputed on every compile, never cached across an edit.
    """

    __slots__ = ("by_id", "sockets", "order")

    def __init__(self, by_id, sockets, order):
        self.by_id = by_id      # asset_id -> "<Picture 3>"
        self.sockets = sockets  # socket-name -> asset_id, matching the stock node's kwargs
        self.order = order      # the ref_items sequence the tokenizer will walk

    def tag(self, asset_id):
        """The tag string for an asset, e.g. '<Picture 3>'. KeyError if absent."""
        return self.by_id[asset_id]

    def audio_tag_for_video(self, asset_id):
        """The <Audio j> tag belonging to a video's paired soundtrack, or None."""
        return self.by_id.get(asset_id + "#soundtrack")


class AssetBin:
    """An ordered collection of reference assets, budget-aware at edit time.

    Order is socket order is tag order -- one sequence, no second source of truth.
    Mutating operations (add/remove/move) either succeed and leave the bin legal,
    or raise BudgetError and leave it untouched.
    """

    def __init__(self, assets=None):
        self._assets = []
        self._by_id = {}
        for a in assets or []:
            self._insert(a if isinstance(a, Asset) else Asset.from_dict(a), None, validate=False)
        report = self.budget()
        if not report.ok:
            raise BudgetError("; ".join(report.errors))

    # ── reading ─────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._assets)

    def __iter__(self):
        return iter(self._assets)

    def __contains__(self, asset_id):
        return asset_id in self._by_id

    def get(self, asset_id):
        return self._by_id.get(asset_id)

    def by_kind(self, kind):
        """Assets of one kind, in bin order. This order is the socket order."""
        return [a for a in self._assets if a.kind == kind]

    def find_by_name(self, name):
        """Case-insensitive name lookup. Returns None if absent or ambiguous."""
        needle = (name or "").strip().casefold()
        hits = [a for a in self._assets if a.name.strip().casefold() == needle]
        return hits[0] if len(hits) == 1 else None

    def index_of(self, asset_id):
        for i, a in enumerate(self._assets):
            if a.asset_id == asset_id:
                return i
        raise KeyError(asset_id)

    # ── budget ──────────────────────────────────────────────────────────────

    def budget(self, extra=None):
        """Occupancy report for this bin, optionally as if `extra` were added.

        `extra` lets a UI ask "would this drop fit?" without mutating anything.
        """
        assets = list(self._assets) + ([extra] if extra is not None else [])
        images = sum(1 for a in assets if a.kind == KIND_IMAGE)
        videos = sum(1 for a in assets if a.kind == KIND_VIDEO)
        audios = sum(1 for a in assets if a.kind == KIND_AUDIO)
        soundtracks = sum(1 for a in assets if a.kind == KIND_VIDEO and a.include_audio)
        # A paired soundtrack rides inside its video's own file, so it consumes an
        # <Audio j> ordinal but not one of the 12 file slots. Synthetic carry-over
        # assets are not files either -- they never existed on disk.
        files = sum(1 for a in assets if not a.synthetic)

        errors = []
        if images > MAX_REF_IMAGES:
            errors.append("too many reference images: %d (max %d)" % (images, MAX_REF_IMAGES))
        if videos > MAX_REF_VIDEOS:
            errors.append("too many reference videos: %d (max %d)" % (videos, MAX_REF_VIDEOS))
        if audios > MAX_REF_AUDIOS:
            errors.append("too many reference audios: %d (max %d)" % (audios, MAX_REF_AUDIOS))
        if soundtracks > MAX_REF_VIDEO_AUDIOS:
            errors.append("too many video soundtracks: %d (max %d)" % (soundtracks, MAX_REF_VIDEO_AUDIOS))
        if files > MAX_REF_FILES_TOTAL:
            errors.append("too many reference files: %d (max %d)" % (files, MAX_REF_FILES_TOTAL))

        for a in assets:
            if a.kind != KIND_VIDEO:
                continue
            dur = a.duration
            if dur is None:
                continue
            if dur < REF_VIDEO_MIN_SECONDS or dur > REF_VIDEO_MAX_SECONDS:
                errors.append(
                    "reference video %r is %.2fs; H3 accepts %.0f-%.0fs"
                    % (a.name, dur, REF_VIDEO_MIN_SECONDS, REF_VIDEO_MAX_SECONDS))

        return BudgetReport(images, videos, soundtracks, audios, files, errors)

    def can_add(self, asset):
        """(bool, reason) -- whether this asset would fit, without adding it."""
        if asset.asset_id in self._by_id:
            return False, "an asset with id %r is already in the bin" % (asset.asset_id,)
        report = self.budget(extra=asset)
        return (True, "") if report.ok else (False, "; ".join(report.errors))

    # ── mutating ────────────────────────────────────────────────────────────

    def add(self, asset, index=None):
        """Add an asset, optionally at a position. Raises BudgetError if it won't fit."""
        if not isinstance(asset, Asset):
            asset = Asset.from_dict(asset)
        ok, reason = self.can_add(asset)
        if not ok:
            raise BudgetError(reason)
        self._insert(asset, index)
        return asset

    def _insert(self, asset, index, validate=True):
        if validate and asset.asset_id in self._by_id:
            raise BudgetError("duplicate asset id %r" % (asset.asset_id,))
        if index is None:
            self._assets.append(asset)
        else:
            self._assets.insert(max(0, min(int(index), len(self._assets))), asset)
        self._by_id[asset.asset_id] = asset

    def remove(self, asset_id):
        """Remove by id. Returns the removed Asset. KeyError if absent."""
        asset = self._by_id.pop(asset_id)
        self._assets = [a for a in self._assets if a.asset_id != asset_id]
        return asset

    def move(self, asset_id, new_index):
        """Reorder. Every downstream ordinal shifts -- which is exactly why prompt
        text never stores one."""
        idx = self.index_of(asset_id)
        asset = self._assets.pop(idx)
        self._assets.insert(max(0, min(int(new_index), len(self._assets))), asset)
        return asset

    def set_include_audio(self, asset_id, include):
        """Toggle a video's paired soundtrack.

        Worth its own method because it is the least obvious renumbering trigger
        in the whole system: turning this on inserts an <Audio j> ahead of every
        standalone audio tag, shifting them all by one.
        """
        asset = self._by_id[asset_id]
        if asset.kind != KIND_VIDEO:
            raise ValueError("only a video asset can carry a paired soundtrack")
        was = asset.include_audio
        asset.include_audio = bool(include)
        report = self.budget()
        if not report.ok:
            asset.include_audio = was
            raise BudgetError("; ".join(report.errors))
        return asset

    # ── the one place ordinals are assigned ─────────────────────────────────

    def tag_map(self):
        """Compute every ordinal for the current bin state.

        Reproduces MiniMaxH3ReferenceToVideo.execute()'s ref_items build order
        exactly, because the tokenizer numbers by position in that list:

            1. every ref_image, in socket order          -> <Picture i>
            2. per ref_video, in socket order:
                   its soundtrack, if any, appended FIRST -> <Audio j>
                   then the video itself                  -> <Video k>
            3. every standalone ref_audio, in socket order -> <Audio j>

        Step 2's ordering is the subtle one. A video's soundtrack is emitted
        *before* its own <Video k> tag and *before* every standalone audio, so
        soundtracks always claim the low <Audio j> numbers.
        """
        by_id = {}
        sockets = {}
        order = []
        n_image = n_video = n_audio = 0

        for slot, asset in enumerate(self.by_kind(KIND_IMAGE)):
            n_image += 1
            by_id[asset.asset_id] = "<%s %d>" % (TAG_IMAGE, n_image)
            sockets["ref_image_%d" % slot] = asset.asset_id
            order.append(("image", asset.asset_id, n_image))

        for slot, asset in enumerate(self.by_kind(KIND_VIDEO)):
            if asset.include_audio:
                n_audio += 1
                # Keyed with a suffix so a video's soundtrack tag is addressable
                # without colliding with the video's own <Video k> entry.
                by_id[asset.asset_id + "#soundtrack"] = "<%s %d>" % (TAG_AUDIO, n_audio)
                sockets["ref_video_audio_%d" % slot] = asset.asset_id
                order.append(("audio", asset.asset_id + "#soundtrack", n_audio))
            n_video += 1
            by_id[asset.asset_id] = "<%s %d>" % (TAG_VIDEO, n_video)
            sockets["ref_video_%d" % slot] = asset.asset_id
            order.append(("video", asset.asset_id, n_video))

        for slot, asset in enumerate(self.by_kind(KIND_AUDIO)):
            n_audio += 1
            by_id[asset.asset_id] = "<%s %d>" % (TAG_AUDIO, n_audio)
            sockets["ref_audio_%d" % slot] = asset.asset_id
            order.append(("audio", asset.asset_id, n_audio))

        return TagMap(by_id, sockets, order)

    # ── serialisation ───────────────────────────────────────────────────────

    def to_list(self):
        return [a.to_dict() for a in self._assets]

    @classmethod
    def from_list(cls, data):
        return cls([Asset.from_dict(d) for d in (data or [])])


# Prompt text refers to assets with {{asset_id}} or @Name / @[Name With Spaces].
# Both resolve through the TagMap at compile time. A literal "<Picture 3>" typed
# by hand is detected separately and reported as a hazard, never silently trusted.
REF_TOKEN_RE = re.compile(r"\{\{\s*([A-Za-z0-9_\-]+)\s*\}\}|@\[([^\]]+)\]|@([A-Za-z0-9_\-]+)")
LITERAL_TAG_RE = re.compile(r"<\s*(Picture|Video|Audio)\s+(\d+)\s*>", re.IGNORECASE)
