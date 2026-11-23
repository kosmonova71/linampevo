"""ColourGenes — palette / colour appearance genome module."""

from __future__ import annotations

import math
import random

TWO_PI = 6.28318


BOUNDS: dict[str, tuple[float, float]] = {
    "colour_shift": (-TWO_PI, TWO_PI),
    "colour_palette": (0.0, 20.0),
    "symmetry": (1, 12),
    "contrast": (0.0, 3.0),
}

INT_KEYS = {"symmetry"}


class ColourGenes:
    """Genes controlling colour palette, symmetry, and contrast.

    Attributes map to ``k_colour_shift``, ``k_symmetry``, etc. in the shader.
    """

    __slots__ = ("colour_palette", "colour_shift", "contrast", "symmetry")

    def __init__(self, *, rng: random.Random | None = None):
        r = rng or random
        self.colour_shift: float = r.uniform(-2.0, 2.0)
        self.colour_palette: float = r.uniform(0.0, 8.0)
        self.symmetry: int = r.randint(1, 8)
        self.contrast: float = r.uniform(0.5, 2.0)

    def mutate(self, strength: float = 0.12) -> str:
        """Mutate one random gene in this module. Returns the mutated key."""
        keys = list(BOUNDS.keys())
        key = random.choice(keys)
        lo, hi = BOUNDS[key]
        if key in INT_KEYS:
            step = random.choice([-1, 1])
            val = int(getattr(self, key)) + step
            setattr(self, key, int(max(lo, min(hi, val))))
        else:
            delta = random.uniform(-strength, strength) * (hi - lo)
            val = float(getattr(self, key)) + delta
            setattr(self, key, max(lo, min(hi, val)))
        return key

    def clamp(self) -> None:
        for key, (lo, hi) in BOUNDS.items():
            val = getattr(self, key)
            if key in INT_KEYS:
                setattr(self, key, int(max(lo, min(hi, val))))
            else:
                v = float(val)
                if key == "colour_shift":
                    v = ((v + math.pi) % TWO_PI) - math.pi
                setattr(self, key, max(lo, min(hi, v)))

    def to_dict(self) -> dict[str, float | int]:
        return {k: getattr(self, k) for k in BOUNDS}

    def from_dict(self, data: dict[str, float | int]) -> None:
        for k in BOUNDS:
            if k in data:
                setattr(self, k, data[k])
        self.clamp()

    def clone(self) -> ColourGenes:
        c = ColourGenes.__new__(ColourGenes)
        for k in BOUNDS:
            setattr(c, k, getattr(self, k))
        return c

    def distance(self, other: ColourGenes) -> float:
        total = 0.0
        for k, (lo, hi) in BOUNDS.items():
            rng = max(abs(hi - lo), 1e-6)
            total += abs(float(getattr(self, k)) - float(getattr(other, k))) / rng
        return total / max(len(BOUNDS), 1)
