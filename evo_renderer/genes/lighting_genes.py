"""LightingGenes — lighting / fog / environmental genome module."""

from __future__ import annotations

import random

BOUNDS: dict[str, tuple[float, float]] = {
    "fog_density": (0.0, 5.0),
    "brightness_base": (0.3, 1.5),
    "environment_albedo": (0.0, 1.0),
}

INT_KEYS: set = set()


class LightingGenes:
    """Genes controlling fog, brightness, and environmental lighting.

    Attributes map to ``k_fog_density``, ``u_brightness``, etc. in the shader.
    """

    __slots__ = ("brightness_base", "environment_albedo", "fog_density")

    def __init__(self, *, rng: random.Random | None = None):
        r = rng or random
        self.fog_density: float = r.uniform(0.2, 2.5)
        self.brightness_base: float = r.uniform(0.5, 1.2)
        self.environment_albedo: float = r.uniform(0.1, 0.9)

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

    def clone(self) -> LightingGenes:
        c = LightingGenes.__new__(LightingGenes)
        for k in BOUNDS:
            setattr(c, k, getattr(self, k))
        return c

    def distance(self, other: LightingGenes) -> float:
        total = 0.0
        for k, (lo, hi) in BOUNDS.items():
            rng = max(abs(hi - lo), 1e-6)
            total += abs(float(getattr(self, k)) - float(getattr(other, k))) / rng
        return total / max(len(BOUNDS), 1)
