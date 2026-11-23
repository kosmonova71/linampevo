"""Particles — entities and their DNA for the physics ecosystem.

Follows the same gene-mutation pattern used by the rest of the
codebase (``evo_renderer/genes``): a ``BOUNDS`` dict defines the
valid range for each parameter, ``random()`` generates random DNA, and
``mutate()`` perturbs one parameter.

``EntityDNA`` provides ``to_dict`` / ``from_dict`` for stable JSON
serialisation (preferred over ``__dict__`` so the format stays
robust if ``__slots__`` or private attributes are added later).
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .field import Field

# Each entry: param_name -> (min, max)
BOUNDS: dict[str, tuple[float, float]] = {
    "gravity": (0.0, 0.5),
    "attraction": (0.0, 1.0),
    "decay": (0.0, 1.0),
    "growth": (0.0, 1.0),
    "mass": (0.1, 2.0),
    "speed": (0.1, 5.0),
    "bass_affinity": (0.0, 1.0),
    "mid_affinity": (0.0, 1.0),
    "treble_affinity": (0.0, 1.0),
    "colour": (0.0, 1.0),
}


class EntityDNA:
    """DNA encoding the physics parameters of a single entity.

    Parameters
    ----------
    gravity
        Downward pull strength (0 = no gravity, 0.5 = strong).
    attraction
        Tendency to move towards high-energy cells.
    decay
        Per-second energy decay rate (0 = no decay, 1 = rapid decay).
    growth
        Efficiency of converting ambient energy into entity energy.
    mass
        Entity mass — heavier entities accelerate slower.
    speed
        Maximum movement speed.
    bass_affinity, mid_affinity, treble_affinity
        Sensitivity to each audio frequency band.
    colour
        Colour identifier (hue in 0-1 range).
    """

    def __init__(
        self,
        *,
        gravity: float = 0.1,
        attraction: float = 0.5,
        decay: float = 0.1,
        growth: float = 0.1,
        mass: float = 1.0,
        speed: float = 2.0,
        bass_affinity: float = 0.5,
        mid_affinity: float = 0.5,
        treble_affinity: float = 0.5,
        colour: float = 0.5,
    ) -> None:
        self.gravity = float(gravity)
        self.attraction = float(attraction)
        self.decay = float(decay)
        self.growth = float(growth)
        self.mass = float(mass)
        self.speed = float(speed)
        self.bass_affinity = float(bass_affinity)
        self.mid_affinity = float(mid_affinity)
        self.treble_affinity = float(treble_affinity)
        self.colour = float(colour)

    # ------------------------------------------------------------------
    # classmethod factories
    # ------------------------------------------------------------------
    @classmethod
    def random(cls) -> EntityDNA:
        """Create DNA with each parameter uniformly sampled from its range."""
        return cls(**{k: random.uniform(lo, hi) for k, (lo, hi) in BOUNDS.items()})

    # ------------------------------------------------------------------
    # mutation / clamping
    # ------------------------------------------------------------------
    def mutate(self, strength: float = 0.12) -> None:
        """Perturb one random gene by *strength* (fraction of its range).

        The new value is clamped to ``BOUNDS``.
        """
        key = random.choice(list(BOUNDS.keys()))
        lo, hi = BOUNDS[key]
        current = float(getattr(self, key))
        delta = random.uniform(-strength, strength) * (hi - lo)
        setattr(self, key, max(lo, min(hi, current + delta)))

    def clamp(self) -> None:
        """Clamp every gene to its valid range."""
        for key, (lo, hi) in BOUNDS.items():
            setattr(self, key, max(lo, min(hi, float(getattr(self, key)))))

    # ------------------------------------------------------------------
    # serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, float]:
        """Serialise DNA to a plain dict suitable for JSON."""
        return {k: float(getattr(self, k)) for k in BOUNDS}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> EntityDNA:
        """Reconstruct DNA from a dict, defaulting missing keys to mid-range."""
        values: dict[str, float] = {}
        for key, (lo, hi) in BOUNDS.items():
            values[key] = float(data[key]) if key in data else (lo + hi) / 2.0
        dna = cls(**values)
        dna.clamp()
        return dna

    def clone(self) -> EntityDNA:
        """Return a deep copy of this DNA."""
        return self.from_dict(self.to_dict())


class Entity:
    """A simulated particle with position, velocity, and DNA.

    Entities are created by :class:`~physics.Ecosystem` and driven
    by the :class:`~physics.Forces` model and their own lifecycle
    logic.
    """

    # Life / energy thresholds
    _SPLIT_ENERGY: float = 8.0
    _SPLIT_COOLDOWN: float = 5.0
    _MAX_LIFE: float = 120.0
    _MIN_ENERGY: float = 0.0
    _START_ENERGY: float = 3.0
    _START_LIFE: float = 100.0

    def __init__(
        self,
        *,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        dna: EntityDNA,
        life: float | None = None,
        age: float = 0.0,
        energy: float | None = None,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.dna = dna
        self.energy = float(energy) if energy is not None else self._START_ENERGY
        self.life = float(life) if life is not None else self._START_LIFE
        self.age = float(age)
        self._split_cooldown = 0.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def step_lifecycle(self, dt: float, *, field: Field | None = None) -> None:
        """Advance the entity's internal state by *dt* seconds.

        Handles energy decay and regrowth, ageing, position
        integration, and velocity drag.

        When *field* is provided the entity samples the local cell
        energy at its (wrapped) position and withdraws the consumed
        amount — modelling resource competition.  Position is wrapped
        toroidally into the grid so entities never drift off-map.
        """
        dt = float(dt)
        self.age += dt

        # --- split cooldown timer ------------------------------------
        if self._split_cooldown > 0.0:
            self._split_cooldown -= dt

        # --- position integration -------------------------------------
        self.x += self.vx * dt
        self.y += self.vy * dt

        # --- local energy uptake (resource competition) ---------------
        dna = self.dna
        if field is not None:
            self._wrap_position(field.width, field.height)
            cx = int(self.x) % field.width if field.width else 0
            cy = int(self.y) % field.height if field.height else 0
            local_energy = field.sample(cx, cy)
            growth_rate = float(dna.growth)
            uptake = local_energy * growth_rate * dt
            if uptake > 0.0:
                field.withdraw(cx, cy, uptake)
                self.energy += uptake

        # --- energy dynamics -------------------------------------------
        # Natural decay.
        decay_rate = float(dna.decay)
        self.energy *= 1.0 - decay_rate * dt
        # Floor at zero so is_dead() works cleanly.
        self.energy = max(self.energy, 0.0)

        # --- life decay -----------------------------------------------
        life_cost = 0.05 * dt + 0.02 * decay_rate * dt
        self.life -= life_cost

        # --- drag ------------------------------------------------------
        drag = 0.98**dt
        self.vx *= drag
        self.vy *= drag

    def _wrap_position(self, width: int, height: int) -> None:
        """Wrap position toroidally into ``[0, width) x [0, height)``."""
        if width > 0:
            self.x %= width
        if height > 0:
            self.y %= height

    def is_dead(self) -> bool:
        """Return ``True`` when the entity should be removed."""
        return self.energy <= self._MIN_ENERGY or self.life <= 0.0 or self.age >= self._MAX_LIFE

    def maybe_split(self, dt: float) -> float | None:
        """Return the energy to transfer to a child, or ``None``.

        When the entity has enough stored energy it pays a fixed
        reproduction cost (dropping its own energy below the split
        threshold) and signals that it should reproduce by returning
        the child's starting energy.  The actual child creation is
        handled by :class:`~physics.Ecosystem`.

        A cooldown prevents spamming children every frame: after
        splitting the entity must wait ``_SPLIT_COOLDOWN`` seconds
        before it can split again.
        """
        del dt  # splitting is energy-gated, not time-gated
        if self._split_cooldown > 0.0:
            return None
        if self.energy < self._SPLIT_ENERGY:
            return None
        # Parent drops to half the threshold (below split trigger).
        parent_remaining = self._SPLIT_ENERGY * 0.5
        transfer = max(0.5, self.energy - parent_remaining)
        self.energy = parent_remaining
        self._split_cooldown = self._SPLIT_COOLDOWN
        return transfer

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def speed(self) -> float:
        """Current movement speed."""
        return math.hypot(self.vx, self.vy)

    def fitness(self) -> float:
        """Composite fitness score used when culling the population."""
        return self.energy + self.life

    def __repr__(self) -> str:
        return (
            f"Entity(x={self.x:.2f}, y={self.y:.2f}, "
            f"energy={self.energy:.2f}, life={self.life:.2f}, "
            f"age={self.age:.2f})"
        )
