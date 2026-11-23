"""Identity state — persistent behavioural identity for the evolutionary renderer.

This module is the glue between the renderer's evolution loop and a
*persistent identity*: an evergreen personality that remembers what it has
seen (champions, extinct species, mutation outcomes, human feedback) and
biases future evolution accordingly.

Two public classes are exported:

* :class:`IdentityGenome` -- a lightweight dataclass encoding identity-level
  traits (energy, curiosity, stability, aggression, ...).  It is mutated when
  the identity's mood drifts significantly and feeds into mutation-strength
  and exploration-probability calculations.

* :class:`VisualIdentity` -- the full identity manager.  State is persisted as
  JSON files in the ``state_path`` directory so it survives restarts and can
  be inspected / edited by hand.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

# Default trait values for a freshly initialised identity genome.
_DEFAULT_GENOME_PARAMS: dict[str, float] = {
    "energy": 0.5,
    "curiosity": 0.5,
    "stability": 0.5,
    "aggression": 0.3,
    "mutation_bias": 0.5,
    "exploration_bias": 0.3,
}

# Default mood values (0-1 floats).
_DEFAULT_MOOD: dict[str, float] = {
    "energy": 0.5,
    "curiosity": 0.5,
    "stability": 0.5,
    "aggression": 0.3,
}

# Default identity lifecycle seasons.
_SEASONS: tuple[str, ...] = ("birth", "youth", "maturity", "senescence")

# Schema version written into every persisted JSON file.  Bump this when a
# structural change is made so loads can migrate instead of silently misreading.
SCHEMA_VERSION: int = 1

# Age thresholds (seconds of process age) that transition the lifecycle season.
# These are intentionally long: a session that runs for hours stays in
# "maturity"; "senescence" is a multi-hour state, not a 5-minute trap.
_SEASON_THRESHOLDS: tuple[float, ...] = (60.0, 600.0, 3600.0, 86400.0)

# How often (seconds) a routine ``tick`` memory event is recorded.  Recording
# every frame at 60 FPS fills the event log in minutes and buries the
# meaningful mutation/champion/extinction events.  See ``VisualIdentity.tick``.
_TICK_EVENT_INTERVAL: float = 1.0

# Slow decay applied to preference weights each save cycle so that old
# likes/dislikes do not dominate forever.  See ``_decay_preferences``.
PREFERENCE_DECAY_RATE: float = 0.98


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_genome_id() -> str:
    """Generate a fresh, stable genome identifier."""
    import uuid

    return "g-" + uuid.uuid4().hex[:12]


def _serialize_genome(genome: Any) -> dict[str, Any]:
    """Serialise *genome* into a JSON-safe dictionary.

    Mirrors the pattern used in ``evo_renderer.evolution_memory.GenomeArchive``
    -- if the object has a ``to_dict`` method, use it; otherwise fall back to
    ``dict()`` for dict-like objects, and ``str()`` for anything else.
    """
    if hasattr(genome, "to_dict"):
        return genome.to_dict()
    if isinstance(genome, dict):
        return dict(genome)
    return {"_raw": str(genome)}


def _load_json(path: Path, default: Any) -> Any:
    """Load JSON from *path*, returning *default* on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> bool:
    """Write *data* as JSON to *path*.  Returns True on success.

    Writes to a temp file in the same directory and renames it over the
    target, so a crash mid-write cannot leave a torn JSON file behind.
    """
    try:
        payload = json.dumps(data, indent=2, default=str, sort_keys=False)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save %s: %s", path, exc)
        return False


def _load_jsonl(path: Path, default: list[Any]) -> list[Any]:
    """Load newline-delimited JSON from *path*."""
    try:
        events: list[Any] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_jsonl(path: Path, events: list[Any]) -> bool:
    """Write *events* as newline-delimited JSON to *path*.

    Uses the same atomic temp-file + rename strategy as :func:`_save_json`.
    """
    try:
        lines = [json.dumps(e, default=str) for e in events]
        payload = "\n".join(lines) + ("\n" if lines else "")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save %s: %s", path, exc)
        return False


def _append_jsonl(path: Path, events: list[Any]) -> bool:
    """Append *events* to a JSONL file atomically.

    Unlike :func:`_save_jsonl` this appends only the new events to the
    existing file content (if any), avoiding a full rewrite of a large log
    on every save cycle.  Falls back to a full rewrite on read failure.
    """
    try:
        lines = [json.dumps(e, default=str) for e in events]
        if not lines:
            return True
        tmp = path.with_suffix(path.suffix + ".tmp")
        existing = ""
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
        tmp.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to append-save %s: %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# IdentityGenome
# ---------------------------------------------------------------------------


@dataclass
class IdentityGenome:
    """Genetic blueprint encoding behavioural traits and active modules.

    Attributes:
        modules: Active behavioural module names (e.g. ``"noise"``,
            ``"warp"``, ``"fractal"``).
        params: Float trait values in [0, 1].  Keys include *energy*,
            *curiosity*, *stability*, *aggression*, *mutation_bias*, and
            *exploration_bias*.
        generation: How many times this genome has been mutated.
        lineage: Identifier of the lineage this genome belongs to.
        parent_id: Identifier of the parent genome (for lineage tracking).
    """

    modules: list[str] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_GENOME_PARAMS))
    generation: int = 0
    lineage: str = ""
    parent_id: str | None = None
    # Stable identifier for this genome instance.  Used as the key when
    # matching lineage parents/children in the evolution tree, instead of
    # comparing whole serialised entries (which is fragile and almost never
    # matches because the parent stored on a record is usually just an id).
    genome_id: str = field(default_factory=lambda: _new_genome_id())

    def __post_init__(self) -> None:
        if not self.genome_id:
            self.genome_id = _new_genome_id()

    # -- mutation -----------------------------------------------------------

    def mutate(self, strength: float = 0.1) -> None:
        """Mutate genome trait parameters with the given *strength*.

        Args:
            strength: Mutation step size (0-1).  Higher values produce more
                dramatic trait shifts.
        """
        strength = max(0.0, float(strength))
        for key, value in self.params.items():
            self.params[key] = max(0.0, min(1.0, value + random.gauss(0.0, strength)))
        # The parent of this mutation is the genome's previous self-identity.
        previous_id = self.genome_id
        self.generation += 1
        self.parent_id = previous_id

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this genome."""
        return {
            "modules": list(self.modules),
            "params": dict(self.params),
            "generation": self.generation,
            "lineage": self.lineage,
            "parent_id": self.parent_id,
            "genome_id": self.genome_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityGenome:
        """Reconstruct an :class:`IdentityGenome` from a dictionary."""
        return cls(
            modules=list(data.get("modules", [])),
            params={k: float(v) for k, v in data.get("params", {}).items()},
            generation=int(data.get("generation", 0)),
            lineage=str(data.get("lineage", "")),
            parent_id=data.get("parent_id"),
            genome_id=str(data.get("genome_id", "") or ""),
        )

    # -- factories ----------------------------------------------------------

    @classmethod
    def random(cls, seed: int | None = None, lineage: str = "") -> IdentityGenome:
        """Create a genome with randomised trait values."""
        rng = random.Random(seed)
        params = {k: rng.uniform(0.0, 1.0) for k in _DEFAULT_GENOME_PARAMS}
        return cls(params=params, lineage=lineage)

    @classmethod
    def from_mood(cls, mood: dict[str, float]) -> IdentityGenome:
        """Create a genome whose trait values are seeded from *mood*.

        Mood traits (energy, curiosity, stability, aggression) are mapped
        directly; ``mutation_bias`` and ``exploration_bias`` are derived
        from mood.
        """
        params: dict[str, float] = {}
        for k, v in _DEFAULT_GENOME_PARAMS.items():
            if k in mood:
                params[k] = max(0.0, min(1.0, float(mood[k])))
            elif k == "mutation_bias":
                params[k] = max(0.0, min(1.0, float(mood.get("aggression", 0.3)) + 0.2))
            elif k == "exploration_bias":
                params[k] = max(0.0, min(1.0, float(mood.get("curiosity", 0.5))))
            else:
                params[k] = v
        return cls(params=params)


# ---------------------------------------------------------------------------
# VisualIdentity
# ---------------------------------------------------------------------------


class VisualIdentity:
    """Persistent identity that ages, learns preferences, and guides evolution.

    The identity is the *personality* layer between the audio-reactive
    renderer and the evolutionary algorithm.  It records what the renderer
    has seen -- champions, extinct species, mutations, human feedback -- so
    that future evolution is biased by historical memory rather than
    starting from scratch each session.

    State is persisted as JSON files in the ``state_path`` directory:

    ===================  ================================================
    File                 Contents
    ===================  ================================================
    ``preferences.json``  Likes / dislikes (weights per module / mutation)
    ``hall_of_fame.json`` Best-ever genomes (sorted by score)
    ``museum.json``       Curated specimens with tags and notes
    ``lineages.json``     Lineage records (each a genome + metadata)
    ``extinct_species.json`` Archived species that went extinct
    ``ancestor_replay.json`` Per-lineage chains of ancestor genomes
    ``memory_events.jsonl`` Append-only event log (mutations, ticks, etc.)
    ``genome.json``       The current :class:`IdentityGenome`
    ``mood.json``         The current identity mood
    ===================  ================================================
    """

    # ------------------------------------------------------------------
    # Construction & persistence
    # ------------------------------------------------------------------

    def __init__(
        self,
        state_path: str | Path | None = None,
        max_memory_events: int = 10000,
    ) -> None:
        """Initialise the identity.

        Args:
            state_path: Directory where identity state JSON files are
                stored.  If *None*, the current working directory is used.
                The directory is created if it does not exist.
            max_memory_events: Maximum number of entries in the
                ``memory_events`` log (older entries are pruned).
        """
        self.state_path: Path = Path(state_path) if state_path else Path.cwd()
        self.state_path.mkdir(parents=True, exist_ok=True)

        self.max_memory_events: int = max(100, int(max_memory_events))

        # Age & time
        self.age: float = 0.0
        self.last_tick: float = time.time()

        # Mood (0-1 floats)
        self.mood: dict[str, float] = dict(_DEFAULT_MOOD)

        # Lifecycle season
        self.season: str = "birth"

        # Identity genome
        self.identity_genome: IdentityGenome = IdentityGenome.random()

        # Preferences: mutation/module name -> weight
        self.likes: dict[str, float] = {}
        self.dislikes: dict[str, float] = {}

        # Memory event log
        self.memory_events: list[dict[str, Any]] = []

        # Genetic library
        self.champions: list[dict[str, Any]] = []
        # ``hall_of_fame`` is a separate, curated list from ``champions``:
        # they are *not* the same buffer (see ``record_champion`` /
        # ``record_hall_of_fame``).  ``champions`` is the renderer's
        # best-of list; ``hall_of_fame`` is the identity's curated archive.
        self.hall_of_fame: list[dict[str, Any]] = []
        self.museum: list[dict[str, Any]] = []
        self.lineages: list[dict[str, Any]] = []
        self.extinct_species: list[dict[str, Any]] = []
        self.ancestor_replay: dict[str, list[dict[str, Any]]] = {}

        # Module usage tracking: (module_name, delta)
        self.module_history: list[tuple[str, float]] = []

        # Dream buffer
        self._dreams: list[dict[str, Any]] = []

        # Tick event throttling: only record a ``tick`` memory event every
        # ``_tick_event_interval`` seconds so the event log is not flooded
        # with routine per-frame noise.
        self._tick_event_interval: float = _TICK_EVENT_INTERVAL
        self._last_tick_event: float = 0.0

        # Memory events that have not yet been persisted (so they can be
        # appended to the JSONL log instead of rewriting it wholesale).
        self._pending_memory_events: list[dict[str, Any]] = []

        # Load persisted state
        self.load()

    def load(self) -> None:
        """Load all persisted state from JSON files in ``state_path``."""
        sp = self.state_path

        mood_data = _load_json(sp / "mood.json", {})
        if mood_data:
            self.mood.update(
                {k: max(0.0, min(1.0, float(v))) for k, v in mood_data.get("mood", {}).items()}
            )
            self.age = float(mood_data.get("age", 0.0))
            self.season = str(mood_data.get("season", "birth"))

        genome_data = _load_json(sp / "genome.json", {})
        if genome_data:
            self.identity_genome = IdentityGenome.from_dict(genome_data)

        prefs = _load_json(sp / "preferences.json", {})
        if prefs:
            self.likes = dict(prefs.get("likes", {}))
            self.dislikes = dict(prefs.get("dislikes", {}))

        self.memory_events = _load_jsonl(sp / "memory_events.jsonl", [])
        # ``champions`` and ``hall_of_fame`` are *separate* lists.  They used
        # to be aliased on load (``self.hall_of_fame = self.champions``),
        # which let ``record_champion`` and ``record_hall_of_fame`` fight over
        # one buffer.  Load them independently.
        self.champions = _load_json(sp / "hall_of_fame.json", [])
        self.hall_of_fame = _load_json(sp / "hall_of_fame_curated.json", [])
        self.museum = _load_json(sp / "museum.json", [])
        self.lineages = _load_json(sp / "lineages.json", [])
        self.extinct_species = _load_json(sp / "extinct_species.json", [])
        self.ancestor_replay = _load_json(sp / "ancestor_replay.json", {})

        self._migrate_schema(mood_data, genome_data, prefs)

        logger.debug(
            "Loaded identity: age=%.1f mood=%s champions=%d hall_of_fame=%d",
            self.age,
            {k: round(v, 2) for k, v in self.mood.items()},
            len(self.champions),
            len(self.hall_of_fame),
        )

    def _migrate_schema(
        self,
        mood_data: dict[str, Any],
        genome_data: dict[str, Any],
        prefs: dict[str, Any],
    ) -> None:
        """Apply one-shot migrations for older schema versions.

        Each persisted file carries a ``version`` key (see
        :data:`SCHEMA_VERSION`).  Files without one are treated as version 0
        and migrated in place.  After migration the in-memory state is the
        current schema; the next :meth:`save` rewrites the files at the new
        version.
        """
        migrated = False

        # mood.json
        if not mood_data or mood_data.get("version", 0) < SCHEMA_VERSION:
            migrated = True

        # genome.json
        if genome_data and genome_data.get("version", 0) < SCHEMA_VERSION:
            # Old genomes had no genome_id; assign fresh ones on load.
            if not genome_data.get("genome_id"):
                genome_data["genome_id"] = _new_genome_id()
            migrated = True

        # preferences.json
        if prefs and prefs.get("version", 0) < SCHEMA_VERSION:
            migrated = True

        if migrated:
            logger.debug("Migrated identity state to schema version %d", SCHEMA_VERSION)

    def save(self) -> None:
        """Persist all identity state to JSON files in ``state_path``."""
        sp = self.state_path

        _save_json(sp / "genome.json", self._wrap(self.identity_genome.to_dict()))
        _save_json(
            sp / "mood.json",
            self._wrap(
                {
                    "age": self.age,
                    "season": self.season,
                    "mood": dict(self.mood),
                }
            ),
        )
        _save_json(
            sp / "preferences.json",
            self._wrap({"likes": dict(self.likes), "dislikes": dict(self.dislikes)}),
        )
        # Memory events are appended incrementally so a 30 s save does not
        # rewrite a growing multi-MB log from scratch.
        if self._pending_memory_events:
            _append_jsonl(sp / "memory_events.jsonl", self._pending_memory_events)
            self._pending_memory_events = []
        # Slowly decay preference weights so old likes/dislikes do not
        # dominate forever (recent evidence matters more).
        self._decay_preferences()
        _save_json(sp / "hall_of_fame.json", self._wrap(self.champions))
        _save_json(sp / "hall_of_fame_curated.json", self._wrap(self.hall_of_fame))
        _save_json(sp / "museum.json", self._wrap(self.museum))
        _save_json(sp / "lineages.json", self._wrap(self.lineages))
        _save_json(sp / "extinct_species.json", self._wrap(self.extinct_species))
        _save_json(
            sp / "ancestor_replay.json",
            self._wrap(dict(self.ancestor_replay)),
        )

        logger.debug("Identity state saved to %s", sp)

    @staticmethod
    def _wrap(data: Any) -> dict[str, Any]:
        """Wrap *data* with the current schema version."""
        if isinstance(data, dict):
            payload = dict(data)
            payload["version"] = SCHEMA_VERSION
            return payload
        return {"version": SCHEMA_VERSION, "data": data}

    # ------------------------------------------------------------------
    # Tick & mood
    # ------------------------------------------------------------------

    def tick(self, dt: float, audio: Any | None = None) -> None:
        """Advance the identity by one frame.

        Ages the identity, updates mood from audio features, and records a
        ``tick`` memory event at most once per ``_tick_event_interval``
        seconds (throttled -- recording every frame at 60 FPS floods the
        event log and buries meaningful mutation/champion/extinction
        events within minutes).

        Args:
            dt: Frame delta-time in seconds.
            audio: Optional audio analyse object exposing ``bass``, ``mid``,
                ``treble`` and/or ``treb`` attributes (as used by
                :class:`evo_renderer.AudioAnalyzer`).
        """
        dt = max(0.0, float(dt))
        self.age += dt
        self.last_tick = time.time()

        # Derive mood adjustments from audio.  Accept both ``treble`` and
        # ``treb`` -- the renderer's audio dict uses ``treb`` while the
        # docstring and much of the codebase use ``treble``.
        if audio is not None:
            bass = float(getattr(audio, "bass", 0.0))
            mid = float(getattr(audio, "mid", 0.0))
            treble = float(getattr(audio, "treble", getattr(audio, "treb", 0.0)))
            energy = float(getattr(audio, "energy", (bass + mid + treble) / 3.0))

            # Audio drives energy & aggression
            self.mood["energy"] = _lerp(self.mood["energy"], energy, 0.01 * dt)
            self.mood["aggression"] = _lerp(self.mood["aggression"], energy * 0.5, 0.005 * dt)
            # Complex audio raises curiosity
            spectral_centroid = float(getattr(audio, "spectral_centroid", 0.0))
            complexity = _normalize_brightness(spectral_centroid)
            self.mood["curiosity"] = _lerp(self.mood["curiosity"], complexity, 0.005 * dt)

        # Stability drifts toward 0.5 when audio is active,
        # away from quiet
        if audio is not None:
            audio_present = (bass + mid + treble) > 0.05
            target_stability = 0.5 if audio_present else 0.7
            self.mood["stability"] = _lerp(self.mood["stability"], target_stability, 0.003 * dt)

        # Update lifecycle season based on age
        self._update_season()

        # Mutate identity genome when mood volatility is high
        self._maybe_mutate_genome()

        # Record a throttled ``tick`` event.  Routine ticks are excluded
        # from the pruned log by gating on an interval rather than by
        # dropping them entirely (so the log still reflects passage of
        # time without filling up).
        now = time.time()
        if now - self._last_tick_event >= self._tick_event_interval:
            self._last_tick_event = now
            self._record_event(
                "tick",
                {
                    "age": self.age,
                    "season": self.season,
                    "mood": dict(self.mood),
                },
            )

    def apply_silence(self, silence_seconds: float) -> None:
        """Adjust mood after a period of audio silence.

        Extended silence increases curiosity (the identity wants novel
        stimuli) and decreases stability.

        Args:
            silence_seconds: How long the audio was silent.
        """
        silence_seconds = max(0.0, float(silence_seconds))
        bonus = min(1.0, silence_seconds * 0.05)
        self.mood["curiosity"] = _clamp(self.mood["curiosity"] + bonus * 0.1)
        self.mood["stability"] = _clamp(self.mood["stability"] - bonus * 0.05)
        self.mood["energy"] = _clamp(self.mood["energy"] - bonus * 0.02)

        self._record_event(
            "silence",
            {
                "duration": silence_seconds,
                "mood": dict(self.mood),
            },
        )

    def _update_season(self) -> None:
        """Update the identity lifecycle season based on age.

        Thresholds are long (seconds of process age) so that a session
        running for hours stays in "maturity" rather than falling straight
        into "senescence" after five minutes.  See :data:`_SEASON_THRESHOLDS`.
        """
        idx = 0
        for t in _SEASON_THRESHOLDS:
            if self.age < t:
                break
            idx += 1
        self.season = _SEASONS[min(idx, len(_SEASONS) - 1)]

    def _maybe_mutate_genome(self) -> None:
        """Mutate the identity genome when mood volatility is high."""
        volatility = (
            abs(self.mood["energy"] - 0.5)
            + abs(self.mood["curiosity"] - 0.5)
            + abs(self.mood["aggression"] - 0.3)
        )
        if volatility > 0.7 and random.random() < volatility * 0.001:
            strength = self.identity_mutation_strength()
            self.identity_genome.mutate(strength=strength)
            self._record_event(
                "genome_mutation",
                {
                    "generation": self.identity_genome.generation,
                    "strength": strength,
                },
            )

    # ------------------------------------------------------------------
    # Mutation strength & exploration parameters
    # ------------------------------------------------------------------

    def identity_mutation_strength(self) -> float:
        """Return the mutation strength for the *identity genome*.

        Driven by mood: high aggression -> larger mutations; low stability
        -> slightly larger mutations.
        """
        base = 0.05
        aggression = self.mood.get("aggression", 0.3)
        stability = self.mood.get("stability", 0.5)
        bias = self.identity_genome.params.get("mutation_bias", 0.5)
        return _clamp(base + aggression * 0.15 + (1.0 - stability) * 0.1 + (bias - 0.5) * 0.1)

    def mutation_strength(self) -> float:
        """Return the *visual* mutation strength suggested to the renderer."""
        base = 0.1
        energy = self.mood.get("energy", 0.5)
        stability = self.mood.get("stability", 0.5)
        aggression = self.mood.get("aggression", 0.3)
        # Calm during high energy / stability; volatile when unstable
        return _clamp(base + aggression * 0.2 - (stability - 0.5) * 0.1 + (0.5 - energy) * 0.1)

    def exploration_probability(self) -> float:
        """Return the probability (0-1) of exploring unknown modules."""
        curiosity = self.mood.get("curiosity", 0.5)
        energy = self.mood.get("energy", 0.5)
        bias = self.identity_genome.params.get("exploration_bias", 0.3)
        return _clamp(curiosity * 0.5 + energy * 0.3 + bias * 0.2)

    def prefer_known_species(self) -> bool:
        """Whether to prefer known (previously-seen) species."""
        stability = self.mood.get("stability", 0.5)
        return stability > 0.65

    def should_explore_unknown(self) -> bool:
        """Whether the identity should actively explore unknown species."""
        return random.random() < self.exploration_probability()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_experience(self, mutation_id: str, before: float, after: float) -> None:
        """Record the outcome of a mutation.

        Also updates preference weights: if the mutation improved the score
        (*after* > *before*), it is added to ``likes``; otherwise to
        ``dislikes``.

        Args:
            mutation_id: Identifier of the mutation operator (e.g.
                ``"add_noise"``).
            before: Fitness score before mutation.
            after: Fitness score after mutation.
        """
        before = float(before)
        after = float(after)
        delta = after - before

        entry = {
            "ts": time.time(),
            "mutation": mutation_id,
            "before": before,
            "after": after,
            "delta": delta,
            "accepted": delta > 0.0,
        }

        # Update preferences
        if delta > 0.0:
            self._adjust_preference(self.likes, mutation_id, abs(delta))
        elif delta < 0.0:
            self._adjust_preference(self.dislikes, mutation_id, abs(delta))

        self._prune_memory_events()
        self._record_event("mutation", entry)

    def record_champion(
        self,
        genome: Any,
        score: float,
        parent: Any | None = None,
    ) -> None:
        """Record a champion genome.

        Args:
            genome: The champion genome (serialisable via ``to_dict`` or
                :func:`_serialize_genome`).
            score: Fitness score of the champion.
            parent: Optional parent genome (for lineage tracking).
        """
        genome_dict = _serialize_genome(genome)
        modules = _extract_modules(genome)
        parent_id = _genome_id(parent)
        genome_id = _genome_id(genome) or ""

        entry = {
            "ts": time.time(),
            "score": float(score),
            "lineage_id": getattr(genome, "lineage", "") or "",
            "genome": genome_dict,
            "genome_id": genome_id,
            "modules": modules,
            "parent": parent_id,
        }
        # Deduplicate by stable genome id + score band so float noise and
        # key ordering cannot defeat the membership test.
        if not _hall_of_fame_has(self.champions, genome_id, score):
            self.champions.append(entry)
            self.champions.sort(key=lambda x: x["score"], reverse=True)
            # Keep top N champions
            del self.champions[200:]

        self._record_event(
            "champion",
            {
                "score": float(score),
                "modules": modules,
            },
        )

    def record_modules(self, modules: list[str], delta: float) -> None:
        """Record module usage with a fitness delta.

        Args:
            modules: List of active module names.
            delta: Fitness change associated with these modules.
        """
        delta = float(delta)
        # Store the full module list (frozenset key) so multi-module genomes
        # do not collapse to their first module name.
        self.module_history.append((tuple(modules) if modules else ("none",), delta))
        # Keep a bounded history
        if len(self.module_history) > 1000:
            self.module_history = self.module_history[-1000:]

        self._record_event(
            "module_usage",
            {
                "modules": list(modules),
                "delta": delta,
            },
        )

    def apply_human_feedback(self, sentiment: str) -> None:
        """Apply human feedback (like/dislike) to adjust preferences.

        Uses a single schema helper (:meth:`_last_mutation_target`) so the
        shape of the mutation event cannot drift away from what
        :meth:`record_experience` writes.

        Args:
            sentiment: One of ``"like"``, ``"dislike"``, ``"neutral"``,
                or a free-form string.  ``like`` boosts the last seen
                mutation/module toward ``likes``; ``dislike`` moves it to
                ``dislikes``.
        """
        sentiment = str(sentiment).lower().strip()

        target = self._last_mutation_target()

        if target:
            if sentiment in ("like", "positive", "good", "love"):
                self._adjust_preference(self.likes, target, 0.1)
            elif sentiment in ("dislike", "negative", "bad", "hate"):
                self._adjust_preference(self.dislikes, target, 0.1)
            elif sentiment in ("neutral", "skip", "ignore"):
                pass
            else:
                self._adjust_preference(
                    self.likes if "love" in sentiment or "good" in sentiment else self.dislikes,
                    target,
                    0.05,
                )

    def _last_mutation_target(self) -> str | None:
        """Return the mutation id recorded by the most recent mutation event.

        Centralises the ``event["data"]["mutation"]`` access pattern so a
        single place owns the schema.  Returns ``None`` when no mutation
        event exists yet.
        """
        for event in reversed(self.memory_events):
            if event.get("event_type") == "mutation":
                data = event.get("data")
                if isinstance(data, dict):
                    mutation = data.get("mutation")
                    if mutation:
                        return str(mutation)
        return None

    def record_hall_of_fame(
        self,
        genome: Any,
        score: float,
        lineage_id: str | None = None,
        modules: list[str] | None = None,
        parent: Any | None = None,
    ) -> None:
        """Record a genome in the hall of fame.

        ``hall_of_fame`` is a *separate* curated list from ``champions``;
        they are no longer aliased on load.  Entries are deduplicated by
        stable genome id + score band rather than by fragile dict equality
        (float noise / key order would otherwise prevent matches).

        Args:
            genome: The genome to archive.
            score: Its fitness score.
            lineage_id: Optional lineage identifier.
            modules: Optional explicit module list.
            parent: Optional parent genome.
        """
        if modules is None:
            modules = _extract_modules(genome)

        genome_id = _genome_id(genome) or ""
        entry = {
            "ts": time.time(),
            "score": float(score),
            "lineage_id": lineage_id or getattr(genome, "lineage", "") or "",
            "genome": _serialize_genome(genome),
            "genome_id": genome_id,
            "modules": list(modules),
            "parent": _genome_id(parent),
        }
        # Hall of fame is its own list; do not merge into champions.
        if not _hall_of_fame_has(self.hall_of_fame, genome_id, score):
            self.hall_of_fame.append(entry)
            self.hall_of_fame.sort(key=lambda x: x["score"], reverse=True)
            del self.hall_of_fame[200:]

    def record_museum_specimen(
        self,
        genome: Any,
        score: float,
        tags: list[str] | None = None,
        notes: str = "",
        lineage_id: str | None = None,
    ) -> None:
        """Curate a specimen in the museum.

        Args:
            genome: The genome to curate.
            score: Its fitness score.
            tags: Optional tags (e.g. ``["champion", "noise"]``).
            notes: Free-form notes about the specimen.
            lineage_id: Optional lineage identifier.
        """
        entry = {
            "ts": time.time(),
            "score": float(score),
            "lineage_id": lineage_id or "",
            "genome": _serialize_genome(genome),
            "tags": list(tags) if tags else [],
            "notes": notes,
        }
        self.museum.append(entry)
        self.museum.sort(key=lambda x: x["score"], reverse=True)
        del self.museum[100:]

    def record_ancestor_lineage(self, lineage_id: str, chain: list[dict[str, Any]]) -> None:
        """Store a chain of ancestor genomes for a lineage.

        Args:
            lineage_id: The lineage identifier.
            chain: Ordered list of ancestor genome records (each a dict
                with at least ``genome_id``, ``score``, and ``ts``).
        """
        self.ancestor_replay[lineage_id] = [dict(c) for c in chain]

        # Also record as lineage entries
        for entry in chain:
            lineage_entry = {
                "ts": float(entry.get("ts", time.time())),
                "score": float(entry.get("score", 0.0)),
                "genome": entry.get("genome", {}),
                "genome_id": entry.get("genome_id") or _genome_id(entry.get("genome")),
                "lineage_id": lineage_id,
                "parent": entry.get("parent") or entry.get("parent_id"),
                "descendants": int(entry.get("descendants", 0)),
            }
            if not _lineage_has(self.lineages, lineage_entry):
                self.lineages.append(lineage_entry)

    def archive_extinct_species(
        self,
        species_id: str,
        genome: Any,
        score: float,
        cause: str = "unknown",
        lineage_id: str | None = None,
        generation: int = 0,
        modules: list[str] | None = None,
    ) -> None:
        """Archive a species that has gone extinct.

        Args:
            species_id: Unique species identifier.
            genome: The extinct genome.
            score: Its final fitness score.
            cause: Reason for extinction (e.g. ``"competition"``,
                ``"starvation"``, ``"predation"``).
            lineage_id: Optional lineage identifier.
            generation: Generation number at extinction.
            modules: Optional list of active modules.
        """
        entry = {
            "ts": time.time(),
            "species_id": species_id,
            "genome": _serialize_genome(genome),
            "score": float(score),
            "cause": cause,
            "lineage_id": lineage_id or "",
            "generation": int(generation),
            "modules": (list(modules) if modules else _extract_modules(genome)),
        }
        self.extinct_species.append(entry)
        # Keep bounded
        if len(self.extinct_species) > 500:
            self.extinct_species = self.extinct_species[-500:]

        self._record_event(
            "extinction",
            {
                "species_id": species_id,
                "score": float(score),
                "cause": cause,
            },
        )

    # ------------------------------------------------------------------
    # Dream simulation
    # ------------------------------------------------------------------

    def dream(self, renderer: Any, max_steps: int = 20) -> None:
        """Run a dream simulation during audio silence.

        Generates mutated copies of the identity genome, optionally
        evaluates them against the renderer (if it exposes a
        ``score_genome`` method), and stores promising results for later
        consumption.

        When the renderer cannot evaluate an *identity* genome (e.g. it
        only scores visual/organism genomes), dreams fall back to a pure
        heuristic and every dream result is tagged ``"heuristic": True`` so
        consumers can distinguish real evaluations from guesses.

        Args:
            renderer: The renderer object (may expose ``score_genome`` or
                similar evaluation API).  If evaluation is not available,
                dreams are generated using pure heuristics.
            max_steps: Maximum number of mutation steps to simulate.
        """
        max_steps = max(1, int(max_steps))
        dreams: list[dict[str, Any]] = []

        # Detect whether the renderer can actually evaluate identity
        # genomes.  Many renderers only score visual/organism genomes, in
        # which case every dream below is heuristic-only.
        score_fn = getattr(renderer, "score_genome", None)
        score_fn = score_fn if callable(score_fn) else getattr(renderer, "score", None)
        score_fn = score_fn if callable(score_fn) else None

        for _ in range(max_steps):
            # Clone the identity genome and mutate it
            child = IdentityGenome(
                modules=list(self.identity_genome.modules),
                params=dict(self.identity_genome.params),
                generation=self.identity_genome.generation,
                lineage=self.identity_genome.lineage,
                parent_id=self.identity_genome.genome_id,
            )
            child.mutate(strength=self.identity_mutation_strength())

            score: float | None = None
            heuristic = True

            # Try to evaluate via the renderer if it supports it
            if score_fn is not None:
                try:
                    score = float(score_fn(child.to_dict()))
                    heuristic = False
                except Exception as exc:
                    logger.debug("Dream evaluation failed: %s", exc)

            # Heuristic fallback: score based on mood alignment
            if score is None:
                score = self._heuristic_dream_score(child)

            dreams.append(
                {
                    "genome": child.to_dict(),
                    "score": score,
                    "strength": self.identity_mutation_strength(),
                    "heuristic": heuristic,
                }
            )

        promising = [d for d in dreams if d["score"] >= 0.3]
        promising_count = len(promising)
        avg_score = sum(d["score"] for d in dreams) / max(1, len(dreams))

        self._dreams.extend(dreams)
        # Cap dream buffer
        if len(self._dreams) > max_steps * 10:
            self._dreams = self._dreams[-max_steps * 10 :]

        self._record_event(
            "dream",
            {
                "simulated": max_steps,
                "promising_score": avg_score,
                "promising_count": promising_count,
            },
        )

        logger.debug(
            "Dream completed: simulated=%d promising=%d avg_score=%.3f",
            max_steps,
            promising_count,
            avg_score,
        )

    def _heuristic_dream_score(self, genome: IdentityGenome) -> float:
        """Score a dream-genome using mood alignment (no renderer needed).

        Returns a value in [0, 1].
        """
        score = 0.0
        for trait, target in self.mood.items():
            if trait in genome.params:
                diff = abs(genome.params[trait] - target)
                score += 1.0 - diff
        return _clamp(score / max(1, len(self.mood)))

    def consume_dreams(self) -> list[dict[str, Any]]:
        """Return and clear the accumulated dream results."""
        dreams = list(self._dreams)
        self._dreams.clear()
        return dreams

    # ------------------------------------------------------------------
    # Lineage & evolution tree queries
    # ------------------------------------------------------------------

    def replay_ancestors(self, lineage_id: str) -> list[dict[str, Any]]:
        """Return the stored ancestor chain for a lineage.

        Args:
            lineage_id: The lineage identifier.

        Returns:
            A list of ancestor genome records (each a dict).
        """
        return list(self.ancestor_replay.get(lineage_id, []))

    def get_evolution_tree(self, lineage_id: str | None = None) -> dict[str, Any]:
        """Build an evolution tree for a lineage.

        Each node contains the genome, score, and links to ancestors and
        children.

        Args:
            lineage_id: If *None*, the entire tree (all lineages) is
                returned.

        Returns:
            A nested dictionary representing the evolution tree.
        """
        if lineage_id is None:
            # Return the full tree (all lineages as roots)
            roots = [e for e in self.lineages if e.get("parent") is None]
            return {
                "lineage_id": "all",
                "nodes": [self._build_tree_node(e) for e in roots],
                "total_lineages": len(self.lineages),
                "total_champions": len(self.champions),
            }

        # Find the root for this lineage
        lineage_entries = [e for e in self.lineages if e.get("lineage_id", "") == lineage_id]
        if not lineage_entries:
            # Fall back to ancestor_replay
            chain = self.ancestor_replay.get(lineage_id, [])
            if chain:
                return {
                    "lineage_id": lineage_id,
                    "nodes": self._build_tree_from_chain(chain),
                    "node_count": len(chain),
                }
            return {
                "lineage_id": lineage_id,
                "nodes": [],
                "node_count": 0,
            }

        roots = [e for e in lineage_entries if e.get("parent") is None]
        if not roots:
            roots = lineage_entries[:1]

        return {
            "lineage_id": lineage_id,
            "nodes": [self._build_tree_node(e) for e in roots],
            "node_count": len(lineage_entries),
        }

    def _build_tree_node(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Recursively build a tree node from a lineage entry.

        Children are matched by stable genome id (``genome_id`` on the child
        vs ``parent`` on the parent), *not* by comparing whole serialised
        entries -- the parent stored on a record is normally just an id
        string, so JSON-equality matching almost never succeeds.
        """
        node = {
            "genome": entry.get("genome", {}),
            "score": entry.get("score", 0.0),
            "ts": entry.get("ts", time.time()),
            "descendants": entry.get("descendants", 0),
            "genome_id": entry.get("genome_id"),
            "children": [],
        }
        entry_id = _entry_id(entry)
        for child in self.lineages:
            parent = _entry_parent(child)
            if parent is not None and parent == entry_id:
                node["children"].append(self._build_tree_node(child))
        return node

    def _build_tree_from_chain(self, chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build tree nodes from an ancestor replay chain."""
        nodes: list[dict[str, Any]] = []
        for i, entry in enumerate(chain):
            node = {
                "genome_id": entry.get("genome_id"),
                "genome": entry.get("genome", {}),
                "score": entry.get("score", 0.0),
                "ts": entry.get("ts", 0.0),
            }
            if i + 1 < len(chain):
                node["next"] = chain[i + 1].get("genome_id")
            nodes.append(node)
        return nodes

    def find_common_ancestor(self, lineage_a: str, lineage_b: str) -> dict[str, Any] | None:
        """Find the most recent common ancestor of two lineages.

        Args:
            lineage_a: First lineage identifier.
            lineage_b: Second lineage identifier.

        Returns:
            The common ancestor's record, or ``None`` if none found.
        """
        # Gather ancestor chains for both lineages
        chain_a = self.ancestor_replay.get(lineage_a, [])
        chain_b = self.ancestor_replay.get(lineage_b, [])

        # If both have replay chains, find common genome_id from the end
        if chain_a and chain_b:
            ids_a: list[str] = []
            for e in reversed(chain_a):
                gid = e.get("genome_id")
                if gid:
                    ids_a.append(gid)
            ids_b_set = {e.get("genome_id") for e in reversed(chain_b) if e.get("genome_id")}
            for gid in ids_a:
                if gid in ids_b_set:
                    for e in reversed(chain_a):
                        if e.get("genome_id") == gid:
                            return e

        # Fall back: search lineage entries by lineage_id
        entries_a = [e for e in self.lineages if e.get("lineage_id", "") == lineage_a]
        entries_b = [e for e in self.lineages if e.get("lineage_id", "") == lineage_b]

        # Try to find a shared parent lineage entry, matching by stable id.
        parents_a = {_entry_parent(e) for e in entries_a if _entry_parent(e) is not None}
        for entry in entries_b:
            parent = _entry_parent(entry)
            if parent is not None and parent in parents_a:
                return entry

        return None

    def extinct_lineage_chain(self, lineage_id: str) -> list[dict[str, Any]]:
        """Return all extinct species belonging to a lineage.

        Args:
            lineage_id: The lineage identifier.

        Returns:
            A list of extinct-species entries (dicts) sorted by score.
        """
        chain = [s for s in self.extinct_species if s.get("lineage_id", "") == lineage_id]
        chain.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return chain

    # ------------------------------------------------------------------
    # Summary & helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary of the current identity state."""
        return {
            "age": self.age,
            "season": self.season,
            "mood": dict(self.mood),
            "identity_genome": self.identity_genome.to_dict(),
            "preferences": {
                "likes": dict(self.likes),
                "dislikes": dict(self.dislikes),
            },
            "champions": len(self.champions),
            "museum": len(self.museum),
            "extinct": len(self.extinct_species),
            "lineages": len(self.lineages),
            "memory_events": len(self.memory_events),
            "mutation_strength": self.mutation_strength(),
            "identity_mutation_strength": self.identity_mutation_strength(),
            "exploration_probability": self.exploration_probability(),
            "prefer_known_species": self.prefer_known_species(),
            "should_explore_unknown": self.should_explore_unknown(),
        }

    # -- internal helpers ---------------------------------------------------

    def _record_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Append a memory event and prune if over the limit.

        Events are kept in memory (for queries) *and* queued for
        append-only persistence (see :meth:`save`).  The queue is drained on
        the next save so the JSONL log grows incrementally instead of being
        rewritten wholesale every 30 s.
        """
        event = {
            "ts": time.time(),
            "event_type": event_type,
            "data": data,
        }
        self.memory_events.append(event)
        self._pending_memory_events.append(event)
        self._prune_memory_events()

    def _prune_memory_events(self) -> None:
        """Drop oldest memory events when the buffer exceeds the limit."""
        if len(self.memory_events) > self.max_memory_events:
            self.memory_events = self.memory_events[-self.max_memory_events :]
            # Keep the pending-persistence queue in sync: anything that was
            # pruned from memory must also be dropped from the pending queue
            # (it is already on disk from a previous save).
            if len(self._pending_memory_events) > self.max_memory_events:
                self._pending_memory_events = self._pending_memory_events[
                    -self.max_memory_events :
                ]

    def _adjust_preference(self, target: dict[str, float], key: str, amount: float) -> None:
        """Adjust a preference weight for *key* in *target* dict."""
        current = float(target.get(key, 0.0))
        target[key] = _clamp(current + amount)
        # Remove negligible weights
        if target[key] < 0.001:
            target.pop(key, None)

    def _decay_preferences(self) -> None:
        """Slowly decay preference weights so recent evidence matters more.

        Without decay, likes/dislikes only ever increase (then clamp), so an
        operator that was popular early on dominates forever.  This applies
        a gentle multiplicative decay each save cycle; weights below the
        removal threshold are dropped entirely.
        """
        for store in (self.likes, self.dislikes):
            for key in list(store.keys()):
                store[key] = _clamp(store[key] * PREFERENCE_DECAY_RATE)
                if store[key] < 0.001:
                    store.pop(key, None)


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation clamped to ``[0, 1]``."""
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *v* to ``[lo, hi]``."""
    return max(lo, min(hi, v))


def _normalize_brightness(spectral_centroid: float) -> float:
    """Normalise a spectral centroid value into [0, 1]."""
    if spectral_centroid <= 0.0:
        return 0.0
    return _clamp(1.0 - math.exp(-spectral_centroid / 5000.0))


def _extract_modules(genome: Any) -> list[str]:
    """Extract a module list from a genome object."""
    modules = getattr(genome, "modules", None)
    if modules is not None:
        return list(modules)
    genome_dict = _serialize_genome(genome)
    if isinstance(genome_dict, dict) and "modules" in genome_dict:
        return list(genome_dict["modules"])
    return []


def _genome_id(genome: Any) -> str | None:
    """Return a stable identifier for a genome, or None."""
    if genome is None:
        return None
    gid = getattr(genome, "genome_id", None)
    if gid:
        return str(gid)
    gid = getattr(genome, "id", None)
    if gid:
        return str(gid)
    gid = getattr(genome, "lineage", None)
    if gid:
        return str(gid)
    return None


def _entry_id(entry: dict[str, Any]) -> str | None:
    """Return the stable genome id for a lineage/champion entry.

    Falls back to the parent id when the entry has no genome_id of its own
    (e.g. a chain entry that only records its parent).  Returns ``None``
    when no usable id can be derived.
    """
    gid = entry.get("genome_id")
    if gid:
        return str(gid)
    genome = entry.get("genome")
    if isinstance(genome, dict):
        gid = genome.get("genome_id") or genome.get("id") or genome.get("lineage")
        if gid:
            return str(gid)
    return None


def _entry_parent(entry: dict[str, Any]) -> str | None:
    """Return the stable parent id for a lineage entry, or None."""
    parent = entry.get("parent")
    if parent is None:
        parent = entry.get("parent_id")
    if parent is None:
        return None
    if isinstance(parent, dict):
        return _entry_id(parent)
    return str(parent)


def _lineage_has(lineages: list[dict[str, Any]], entry: dict[str, Any]) -> bool:
    """Return True if *lineages* already contains an entry with the same
    stable genome id as *entry*."""
    target = _entry_id(entry)
    if target is None:
        return False
    return any(_entry_id(e) == target for e in lineages)


def _hall_of_fame_has(entries: list[dict[str, Any]], genome_id: str, score: float) -> bool:
    """Return True if *entries* already has an entry for *genome_id* with
    the same score (within a small epsilon, so float noise does not defeat
    the match)."""
    for e in entries:
        if str(e.get("genome_id", "")) != str(genome_id):
            continue
        try:
            if abs(float(e.get("score", 0.0)) - float(score)) <= 1e-6:
                return True
        except (TypeError, ValueError):
            continue
    return False
