"""The timeline data model: shots on a clock, plus the project-level fields.

This is what lives in the node's single hidden JSON widget. It is deliberately
plain data -- no torch, no comfy, no file IO -- so the whole compiler is testable
headless and the canvas (Phase 5) has exactly one contract to write against.

A note on what a "shot" is and is not: H3 has no per-shot conditioning. A shot is
a span of prose with a timestamp, compiled into a `[Shot N] At MM:SS.mmm, ...`
line. It cannot carry its own reference image pinned at its own time -- see
constants.ANCHOR_FIRST for why. Per-shot visual control in H3 is textual.
"""

from .assets import KIND_AUDIO, KIND_IMAGE, KIND_VIDEO, AssetBin, RefLimits
from .constants import BRANCH_FL2VA, BRANCH_REF2VA, BRANCHES, FPS, MAX_REF_AUDIOS

__all__ = ["Shot", "Timeline", "TimelineError"]


class TimelineError(ValueError):
    """A timeline that cannot be compiled as written."""


class Shot:
    """One prompt span. `start` and `duration` are seconds on the project clock."""

    __slots__ = ("shot_id", "start", "duration", "prompt", "speakers", "seed")

    def __init__(self, shot_id, start=0.0, duration=0.0, prompt="", speakers=None, seed=None):
        self.shot_id = str(shot_id)
        self.start = float(start)
        self.duration = max(0.0, float(duration))
        self.prompt = prompt or ""
        # Asset ids of the characters speaking in this shot. Used to assign the
        # (Sx) speaker ids H3's prompt format wants, and to pair a voice
        # reference audio with the subject it belongs to.
        self.speakers = list(speakers or [])
        self.seed = seed

    @property
    def end(self):
        return self.start + self.duration

    def overlaps(self, start, end):
        """True when this shot's span intersects [start, end).

        A zero-length shot still belongs to the window containing its start --
        otherwise a degenerate shot would be silently dropped from the render.
        """
        if self.duration <= 0.0:
            return start <= self.start < end
        return self.start < end and self.end > start

    def to_dict(self):
        d = {"id": self.shot_id, "start": self.start, "duration": self.duration,
             "prompt": self.prompt}
        if self.speakers:
            d["speakers"] = list(self.speakers)
        if self.seed is not None:
            d["seed"] = self.seed
        return d

    @classmethod
    def from_dict(cls, d, index=0):
        return cls(
            shot_id=d.get("id") or d.get("shot_id") or "shot_%d" % (index + 1),
            start=d.get("start", 0.0),
            duration=d.get("duration", 0.0),
            prompt=d.get("prompt", ""),
            speakers=d.get("speakers") or d.get("speaker_ids"),
            seed=d.get("seed"),
        )

    def __repr__(self):
        return "Shot(%r, %.3fs+%.3fs)" % (self.shot_id, self.start, self.duration)


class Timeline:
    """A whole project: the bin, the shots, and the global prompt fields."""

    def __init__(self, assets=None, shots=None, duration_seconds=None, fps=FPS,
                 style_line="", overall_soundscape="", non_diegetic_music="",
                 dialogue_language="English", branch=BRANCH_REF2VA,
                 continuation_branch=None, first_frame=None, last_frame=None,
                 window_seconds=None, identity_notes="", retention_notes="",
                 audio_ref_ceiling=MAX_REF_AUDIOS):
        # Free-form lines from the global prompt box. They ride into
        # subject_definitions and retention_analysis alongside the definitions the
        # bin generates, so a user can lock identity in prose without inventing
        # an asset for it.
        self.identity_notes = identity_notes or ""
        self.retention_notes = retention_notes or ""
        # §10 scene-local references: shot_id -> [Asset]. Populated by the node
        # layer from each PulseShot's own reference sockets, and left empty for
        # every text-box project, where all references are global by definition.
        self.local_refs = {}
        # §11 keyframe_pairs: shot_id -> {"first": asset_id, "last": asset_id}.
        # Per-shot keyframe anchors, set by the node layer from each PulseShot's
        # start_image / end_image sockets. Empty for every project that pins its
        # anchors at the project level instead.
        self.shot_anchors = {}
        self.limits = RefLimits(audio_ref_ceiling)
        self.assets = (assets if isinstance(assets, AssetBin)
                       else AssetBin.from_list(assets, limits=self.limits))
        self.shots = [s if isinstance(s, Shot) else Shot.from_dict(s, i)
                      for i, s in enumerate(shots or [])]
        self.fps = int(fps)
        self.style_line = style_line or ""
        self.overall_soundscape = overall_soundscape or ""
        self.non_diegetic_music = non_diegetic_music or ""
        self.dialogue_language = (dialogue_language or "English").strip() or "English"

        if branch not in BRANCHES:
            raise TimelineError("branch must be one of %r, got %r" % (BRANCHES, branch))
        self.branch = branch
        # Which branch continuation windows use. ref2va keeps identity references
        # alive across the whole sequence; fl2va gives a harder visual lock on the
        # seam but drops every reference for that window. Per-project setting.
        self.continuation_branch = continuation_branch or branch
        if self.continuation_branch not in BRANCHES:
            raise TimelineError("continuation_branch must be one of %r, got %r"
                                % (BRANCHES, self.continuation_branch))

        # fl2va anchors, by asset id. Only meaningful on the fl2va branch.
        self.first_frame = first_frame
        self.last_frame = last_frame

        # None means "one window per H3's trained ceiling"; the planner fills it in.
        self.window_seconds = window_seconds
        self._duration_seconds = duration_seconds

    # ── derived ─────────────────────────────────────────────────────────────

    @property
    def duration_seconds(self):
        """Project length. Defaults to the end of the last shot when unset."""
        if self._duration_seconds is not None:
            return float(self._duration_seconds)
        return max([s.end for s in self.shots], default=0.0)

    @duration_seconds.setter
    def duration_seconds(self, value):
        self._duration_seconds = None if value is None else float(value)

    def ordered_shots(self):
        """Shots in clock order. Sorting is stable, so shots written at the same
        instant keep the order the author put them in rather than being shuffled."""
        return sorted(self.shots, key=lambda s: s.start)

    def shots_in(self, start, end):
        """Every shot overlapping [start, end) by at least one frame, in clock order.

        A shot spanning a window boundary is compiled into both windows. Dropping
        it from the second would leave that window with no direction at all --
        which renders as a stall rather than as continued action.

        THE ONE-FRAME FLOOR

        A window renders whole frames, so a shot occupying less than one of them
        does not appear in it -- there is no frame for it to appear in. Without
        that floor the overlap test is exact-real arithmetic against a grid that
        is not, and the two disagree constantly: `PulseShot.duration_seconds` used
        to cap at 15.08 while a full 362-frame window is 15.08333s, so a chain of
        full-length shots could not put a single cut on a seam. Every one landed
        3ms short, every shot was compiled into two windows, and -- because a
        shot's reference audio comes with it -- each window was handed two
        lip-sync recordings describing the same seconds. The mouth then has two
        answers and nothing says so.

        Two fallbacks, both deliberate. With fewer than two shots there is nothing
        to trim towards, and a window whose every shot is a sliver keeps them:
        rendering on the style line alone is worse than the duplicate this exists
        to prevent. `report.straddling_shots` still names the ones that remain.
        """
        kept = [s for s in self.ordered_shots() if s.overlaps(start, end)]
        if len(kept) < 2:
            return kept
        grain = 1.0 / float(self.fps or FPS)
        return [s for s in kept
                if min(s.end, end) - max(s.start, start) >= grain] or kept

    # ── validation ──────────────────────────────────────────────────────────

    def validate(self):
        """Structural problems, as a list of human-readable strings.

        Returns rather than raises: the UI wants to show all of them at once,
        and the compiler decides which are fatal.
        """
        problems = []
        ordered = self.ordered_shots()

        seen_ids = set()
        for s in self.shots:
            if s.shot_id in seen_ids:
                problems.append("duplicate shot id %r" % (s.shot_id,))
            seen_ids.add(s.shot_id)
            if s.start < 0:
                problems.append("shot %r starts before zero (%.3fs)" % (s.shot_id, s.start))

        # Strict monotonicity. H3's format wants strictly increasing timestamps;
        # two shots sharing a start would emit the same MM:SS.mmm twice, which
        # reads to the model as one contradictory instruction rather than two.
        for a, b in zip(ordered, ordered[1:]):
            if b.start == a.start:
                problems.append(
                    "shots %r and %r both start at %.3fs; timestamps must be strictly increasing"
                    % (a.shot_id, b.shot_id, a.start))

        for s in self.shots:
            for spk in s.speakers:
                if spk not in self.assets:
                    problems.append("shot %r names speaker %r, which is not in the bin"
                                    % (s.shot_id, spk))

        # A bin voice names its owner by asset id, and the same rule applies: an
        # id pointing at nothing binds a voice to nobody, silently. Pointing it at
        # another recording is refused too -- a voice belongs to a character, and
        # a character is something the model can see.
        for a in self.assets.by_kind(KIND_AUDIO):
            if not a.voice_of:
                continue
            owner = self.assets.get(a.voice_of)
            if owner is None:
                problems.append("audio %r is the voice of %r, which is not in the bin"
                                % (a.name, a.voice_of))
            elif owner.kind == KIND_AUDIO:
                problems.append("audio %r is the voice of %r, which is another audio "
                                "reference; a voice belongs to a character"
                                % (a.name, owner.name))

        report = self.assets.budget()
        problems.extend(report.errors)

        # Branch/input coherence. These are mutually exclusive per render because
        # they are different checkpoints, not different settings.
        if self.branch == BRANCH_FL2VA:
            if len(self.assets):
                problems.append(
                    "branch is fl2va (MiniMaxH3ImageToVideo), which takes no references, "
                    "but the bin holds %d asset(s). Switch to ref2va or empty the bin."
                    % (len(self.assets),))
            for label, aid in (("first_frame", self.first_frame), ("last_frame", self.last_frame)):
                if aid and aid not in self.assets:
                    problems.append("%s names %r, which is not in the bin" % (label, aid))
        else:
            if self.first_frame or self.last_frame:
                problems.append(
                    "branch is ref2va (MiniMaxH3ReferenceToVideo), which has no anchor "
                    "inputs, but first_frame/last_frame is set. Anchors and references "
                    "cannot coexist in one render.")

        if self.duration_seconds <= 0:
            problems.append("project duration is zero; nothing to render")

        return problems

    # ── serialisation ───────────────────────────────────────────────────────

    def to_dict(self):
        d = {
            "version": 1,
            "fps": self.fps,
            "duration_seconds": self._duration_seconds,
            "window_seconds": self.window_seconds,
            "branch": self.branch,
            "continuation_branch": self.continuation_branch,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "identity_notes": self.identity_notes,
            "retention_notes": self.retention_notes,
            "style_line": self.style_line,
            "overall_soundscape": self.overall_soundscape,
            "non_diegetic_music": self.non_diegetic_music,
            "dialogue_language": self.dialogue_language,
            "assets": self.assets.to_list(),
            "shots": [s.to_dict() for s in self.shots],
        }
        # Emitted only when raised, so a project running the documented budget
        # serialises byte-identically to one written before this option existed.
        if self.limits.beyond_spec:
            d["audio_ref_ceiling"] = self.limits.audios
        return d

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            assets=data.get("assets"),
            shots=data.get("shots"),
            duration_seconds=data.get("duration_seconds"),
            fps=data.get("fps", FPS),
            style_line=data.get("style_line", ""),
            overall_soundscape=data.get("overall_soundscape", ""),
            non_diegetic_music=data.get("non_diegetic_music", ""),
            dialogue_language=data.get("dialogue_language", "English"),
            branch=data.get("branch", BRANCH_REF2VA),
            continuation_branch=data.get("continuation_branch"),
            first_frame=data.get("first_frame"),
            last_frame=data.get("last_frame"),
            window_seconds=data.get("window_seconds"),
            identity_notes=data.get("identity_notes", ""),
            retention_notes=data.get("retention_notes", ""),
            audio_ref_ceiling=data.get("audio_ref_ceiling", MAX_REF_AUDIOS),
        )

    @classmethod
    def from_json(cls, text):
        import json
        if not text or not str(text).strip():
            return cls()
        return cls.from_dict(json.loads(text))

    def to_json(self, indent=None):
        import json
        return json.dumps(self.to_dict(), indent=indent)
