"""ParticleGenes — particle / glow effects genome module."""

from __future__ import annotations

import random

BOUNDS: dict[str, tuple[float, float]] = {
    "particle_density": (0.0, 5.0),
    "glow": (0.0, 2.0),
    "vortex": (-5.0, 5.0),
}

INT_KEYS: set = set()


class ParticleGenes:
    """Genes controlling particle density, glow intensity, and vortex effects.

    Attributes map to ``k_particle_density``, ``k_glow``, ``k_vortex`` in the shader.
    """

    __slots__ = ("glow", "particle_density", "vortex")

    def __init__(self, *, rng: random.Random | None = None):
        r = rng or random
        self.particle_density: float = r.uniform(0.2, 3.0)
        self.glow: float = r.uniform(0.1, 1.2)
        self.vortex: float = r.uniform(-2.0, 2.0)

    def mutate(self, strength: float = 0.12) -> str:
        """Mutate one random gene in this module. Returns the mutated key."""
        keys = list(BOUNDS.keys())
        key = random.choice(keys)
        lo, hi = BOUNDS[key]
        delta = random.uniform(-strength, strength) * (hi - lo)
        val = float(getattr(self, key)) + delta
        setattr(self, key, max(lo, min(hi, val)))
        return key

    def clamp(self) -> None:
        for key, (lo, hi) in BOUNDS.items():
            val = float(getattr(self, key))
            setattr(self, key, max(lo, min(hi, val)))

    def to_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in BOUNDS}

    def from_dict(self, data: dict[str, float]) -> None:
        for k in BOUNDS:
            if k in data:
                setattr(self, k, float(data[k]))
        self.clamp()

    def clone(self) -> ParticleGenes:
        c = ParticleGenes.__new__(ParticleGenes)
        for k in BOUNDS:
            setattr(c, k, getattr(self, k))
        return c

    def distance(self, other: ParticleGenes) -> float:
        total = 0.0
        for k, (lo, hi) in BOUNDS.items():
            rng = max(abs(hi - lo), 1e-6)
            total += abs(float(getattr(self, k)) - float(getattr(other, k))) / rng
        return total / max(len(BOUNDS), 1)
