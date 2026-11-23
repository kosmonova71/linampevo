"""FeedbackGenes — feedback / temporal persistence genome module."""

from __future__ import annotations

import random

TWO_PI = 6.28318


BOUNDS: dict[str, tuple[float, float]] = {
    "feedback": (0.0, 1.0),
    "feedback_mode": (0, 6),
    "mirror": (0, 1),
    "rotation": (-TWO_PI, TWO_PI),
}

INT_KEYS = {"feedback_mode", "mirror"}


class FeedbackGenes:
    """Genes controlling feedback persistence, mode, mirroring, and rotation.

    Attributes map to ``k_feedback``, ``k_feedback_mode``, ``k_mirror``, ``k_rotation``.
    """

    __slots__ = ("feedback", "feedback_mode", "mirror", "rotation")

    def __init__(self, *, rng: random.Random | None = None):
        r = rng or random
        self.feedback: float = r.uniform(0.4, 0.92)
        self.feedback_mode: int = r.randint(0, 4)
        self.mirror: int = r.randint(0, 1)
        self.rotation: float = r.uniform(-2.0, 2.0)

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
                if key == "feedback":
                    v = max(0.0, min(0.99, v))
                setattr(self, key, max(lo, min(hi, v)))

    def to_dict(self) -> dict[str, float | int]:
        return {k: getattr(self, k) for k in BOUNDS}

    def from_dict(self, data: dict[str, float | int]) -> None:
        for k in BOUNDS:
            if k in data:
                setattr(self, k, data[k])
        self.clamp()

    def clone(self) -> FeedbackGenes:
        c = FeedbackGenes.__new__(FeedbackGenes)
        for k in BOUNDS:
            setattr(c, k, getattr(self, k))
        return c

    def distance(self, other: FeedbackGenes) -> float:
        total = 0.0
        for k, (lo, hi) in BOUNDS.items():
            rng = max(abs(hi - lo), 1e-6)
            total += abs(float(getattr(self, k)) - float(getattr(other, k))) / rng
        return total / max(len(BOUNDS), 1)
