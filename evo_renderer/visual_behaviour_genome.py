"""evo_renderer.visual_behaviour_genome

Spec 16 — Evolving visual behaviours.

Instead of evolving appearance (colours, warp, noise), evolve *behaviour*.
The genome encodes the dynamics of an ecosystem: how particles move, how
structures grow, how forces interact.  The shader becomes a viewport into
that ecosystem rather than a static image.

Families (``behaviour_family``):
    0  school       — boids-like schooling, coherent group motion
    1  explosion     — radial burst with shockwave propagation
    2  coral_growth  — recursive branching with persistent skeleton
    3  crystal       — ordered lattice growth from seed points
    4  storm         — turbulent vortical flow
    5  galaxy        — gravitational collapse + orbital rotation
    6  lava          — viscous flow with thermal cooling
    7  swarm         — chaotic collective cloud
    8  vine          — directional climbing growth
    9  neuron        — branching with signal propagation
    10 fungal        — mycelial network expansion

Every gene is normalised to 0..1 and mutates/crosses over so the population
can evolve toward recognisable visual forms.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, ClassVar


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAMILY_COUNT = 11
FAMILY_NAMES = (
    "school",
    "explosion",
    "coral_growth",
    "crystal",
    "storm",
    "galaxy",
    "lava",
    "swarm",
    "vine",
    "neuron",
    "fungal",
)

# Per-family default parameter overrides (applied on top of global defaults).
FAMILY_DEFAULTS: dict[int, dict[str, float]] = {
    0: {"intensity": 0.6, "speed": 0.7, "cohesion": 0.8, "chaos": 0.3},
    1: {"intensity": 0.9, "speed": 0.9, "gravity": 0.2, "persistence": 0.4},
    2: {"intensity": 0.5, "speed": 0.3, "branching": 0.8, "persistence": 0.9},
    3: {"intensity": 0.4, "speed": 0.2, "symmetry": 0.9, "chaos": 0.1},
    4: {"intensity": 0.7, "speed": 0.8, "viscosity": 0.3, "chaos": 0.8},
    5: {"intensity": 0.6, "speed": 0.4, "gravity": 0.9, "symmetry": 0.6},
    6: {"intensity": 0.5, "speed": 0.2, "viscosity": 0.9, "persistence": 0.8},
    7: {"intensity": 0.7, "speed": 0.8, "chaos": 0.9, "scale": 0.7},
    8: {"intensity": 0.4, "speed": 0.5, "branching": 0.6, "gravity": 0.3},
    9: {"intensity": 0.5, "speed": 0.6, "branching": 0.7, "symmetry": 0.4},
    10: {"intensity": 0.5, "speed": 0.3, "branching": 0.9, "persistence": 0.9},
}


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------


@dataclass
class VisualBehaviourGenome:
    """Evolvable genome that encodes *how* a visual system behaves.

    Unlike a shader-parameter genome (warp, glow, contrast…) this genome
    controls dynamics: forces, growth rules, persistence, branching,
    symmetry.  The visual appearance emerges from those dynamics.

    All genes are 0..1 floats.  ``behaviour_family`` selects the dominant
    behaviour mode; the remaining genes tune its dynamics.
    """

    # --- behaviour mode ----------------------------------------------------
    behaviour_family: int = 0

    # --- dynamical parameters (all 0..1) ------------------------------------
    intensity: float = 0.5
    scale: float = 0.5
    speed: float = 0.5
    branching: float = 0.5
    symmetry: float = 0.5
    viscosity: float = 0.5
    gravity: float = 0.5
    persistence: float = 0.5
    chaos: float = 0.5
    audio_reactivity: float = 0.5
    feedback_coupling: float = 0.5
    cohesion: float = 0.5

    # --- evolution metadata ------------------------------------------------
    lineage: str = field(default_factory=lambda: _gen_id())
    generation: int = 0
    parent_id: str | None = None
    mutations: list[str] = field(default_factory=list)
    age: float = 0.0
    lifespan: float = 500.0

    # Per-gene bounds (class constant).
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "behaviour_family": (0.0, float(FAMILY_COUNT - 1)),
        "intensity": (0.0, 1.0),
        "scale": (0.0, 1.0),
        "speed": (0.0, 1.0),
        "branching": (0.0, 1.0),
        "symmetry": (0.0, 1.0),
        "viscosity": (0.0, 1.0),
        "gravity": (0.0, 1.0),
        "persistence": (0.0, 1.0),
        "chaos": (0.0, 1.0),
        "audio_reactivity": (0.0, 1.0),
        "feedback_coupling": (0.0, 1.0),
        "cohesion": (0.0, 1.0),
    }

    INT_KEYS: ClassVar[set[str]] = {"behaviour_family"}

    def __post_init__(self) -> None:
        self.clamp()

    # ------------------------------------------------------------------
    # Genome operations
    # ------------------------------------------------------------------

    def mutate(self, strength: float = 0.12) -> None:
        keys = list(self.BOUNDS.keys())
        weights = [1.0 if k == "behaviour_family" else 2.0 for k in keys]
        key = random.choices(keys, weights=weights)[0]
        lo, hi = self.BOUNDS[key]
        old = float(getattr(self, key))
        if key in self.INT_KEYS:
            step = random.choice([-1, 1])
            new = int(old) + step
            new = max(int(lo), min(int(hi), new))
        else:
            delta = random.uniform(-strength, strength)
            if key in (
                "branching",
                "symmetry",
                "viscosity",
                "gravity",
                "persistence",
                "chaos",
                "cohesion",
            ):
                delta *= 1.5
            new = _clamp(old + delta, lo, hi)
        setattr(self, key, new)
        desc = f"{key}{new - old:+.3f}"
        self.mutations.append(desc)
        if len(self.mutations) > 64:
            del self.mutations[:-64]
        self.clamp()

    def crossover(self, other: VisualBehaviourGenome) -> VisualBehaviourGenome:
        child = VisualBehaviourGenome()
        for key in self.BOUNDS:
            val = self.params()[key] if random.random() < 0.5 else other.params()[key]
            setattr(child, key, val)
        child.lineage = f"{self.lineage}x{other.lineage}"
        child.generation = max(self.generation, other.generation) + 1
        child.parent_id = self.lineage
        child.clamp()
        return child

    def clone(self) -> VisualBehaviourGenome:
        g = VisualBehaviourGenome()
        for key, val in self.params().items():
            setattr(g, key, val)
        g.lineage = self.lineage
        g.generation = self.generation
        g.parent_id = self.parent_id
        g.mutations = list(self.mutations)
        g.age = self.age
        g.lifespan = self.lifespan
        g.clamp()
        return g

    def clamp(self) -> None:
        for key, (lo, hi) in self.BOUNDS.items():
            val = float(getattr(self, key))
            if key in self.INT_KEYS:
                setattr(self, key, int(max(lo, min(hi, val))))
            else:
                setattr(self, key, _clamp(val, lo, hi))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def params(self) -> dict[str, Any]:
        return {
            "behaviour_family": self.behaviour_family,
            "intensity": self.intensity,
            "scale": self.scale,
            "speed": self.speed,
            "branching": self.branching,
            "symmetry": self.symmetry,
            "viscosity": self.viscosity,
            "gravity": self.gravity,
            "persistence": self.persistence,
            "chaos": self.chaos,
            "audio_reactivity": self.audio_reactivity,
            "feedback_coupling": self.feedback_coupling,
            "cohesion": self.cohesion,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.params()
        d["_lineage"] = self.lineage
        d["_generation"] = self.generation
        d["_parent_id"] = self.parent_id
        d["_mutations"] = list(self.mutations)
        d["_age"] = self.age
        d["_lifespan"] = self.lifespan
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualBehaviourGenome:
        g = cls()
        for key in g.BOUNDS:
            if key in data:
                setattr(g, key, data[key])
        g.lineage = data.get("_lineage", g.lineage)
        g.generation = int(data.get("_generation", 0))
        g.parent_id = data.get("_parent_id")
        g.mutations = list(data.get("_mutations", []))
        g.age = float(data.get("_age", 0.0))
        g.lifespan = float(data.get("_lifespan", 500.0))
        g.clamp()
        return g

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def family_name(self) -> str:
        idx = int(self.behaviour_family) % FAMILY_COUNT
        return FAMILY_NAMES[idx]

    def apply_family_defaults(self) -> None:
        """Overlay family-specific defaults onto the genome."""
        idx = int(self.behaviour_family) % FAMILY_COUNT
        defaults = FAMILY_DEFAULTS.get(idx, {})
        for key, val in defaults.items():
            if key in self.BOUNDS:
                setattr(self, key, val)
        self.clamp()

    def distance(self, other: VisualBehaviourGenome) -> float:
        total = 0.0
        count = 0
        for k, (lo, hi) in self.BOUNDS.items():
            if k not in other.params():
                continue
            rng = max(abs(hi - lo), 1e-6)
            total += abs(self.params()[k] - other.params()[k]) / rng
            count += 1
        return total / max(count, 1)

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={getattr(self, k):.2f}" for k in self.BOUNDS)
        return f"VisualBehaviourGenome({inner})"


# ---------------------------------------------------------------------------
# Random initialisation
# ---------------------------------------------------------------------------


def _gen_id() -> str:
    import uuid

    return str(uuid.uuid4())[:8]


def random_visual_behaviour(*, seed: int | None = None) -> VisualBehaviourGenome:
    rng = random.Random(seed)
    g = VisualBehaviourGenome()
    for name, (lo, hi) in g.BOUNDS.items():
        if name in g.INT_KEYS:
            setattr(g, name, rng.randint(int(lo), int(hi)))
        else:
            setattr(g, name, round(rng.uniform(lo, hi), 4))
    g.apply_family_defaults()
    g.clamp()
    return g
