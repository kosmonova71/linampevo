"""FieldGenes — noise / field generation genome module."""

from __future__ import annotations

import random

BOUNDS: dict[str, tuple[float, float]] = {
    "field_type": (0, 8),
    "noise_scale": (0.1, 20.0),
    "fractal_layers": (1, 12),
    "domain_warp": (0.0, 2.0),
    "curl_strength": (0.0, 2.0),
    "turbulence": (0.0, 3.0),
    "flow_speed": (0.0, 3.0),
}

INT_KEYS = {"field_type", "fractal_layers"}


class FieldGenes:
    """Genes controlling the noise/field generation pipeline.

    Attributes map to ``k_field_type``, ``k_noise_scale``, etc. in the shader.
    """

    __slots__ = (
        "curl_strength",
        "domain_warp",
        "field_type",
        "flow_speed",
        "fractal_layers",
        "noise_scale",
        "turbulence",
    )

    def __init__(self, *, rng: random.Random | None = None):
        r = rng or random
        self.field_type: int = r.randint(0, 4)
        self.noise_scale: float = r.uniform(1.0, 8.0)
        self.fractal_layers: int = r.randint(2, 7)
        self.domain_warp: float = r.uniform(0.0, 1.2)
        self.curl_strength: float = r.uniform(0.1, 1.2)
        self.turbulence: float = r.uniform(0.3, 2.0)
        self.flow_speed: float = r.uniform(0.3, 2.0)

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
                setattr(self, key, max(lo, min(hi, float(val))))

    def to_dict(self) -> dict[str, float | int]:
        return {k: getattr(self, k) for k in BOUNDS}

    def from_dict(self, data: dict[str, float | int]) -> None:
        for k in BOUNDS:
            if k in data:
                setattr(self, k, data[k])
        self.clamp()

    def clone(self) -> FieldGenes:
        c = FieldGenes.__new__(FieldGenes)
        for k in BOUNDS:
            setattr(c, k, getattr(self, k))
        return c

    def distance(self, other: FieldGenes) -> float:
        total = 0.0
        for k, (lo, hi) in BOUNDS.items():
            rng = max(abs(hi - lo), 1e-6)
            total += abs(float(getattr(self, k)) - float(getattr(other, k))) / rng
        return total / max(len(BOUNDS), 1)
