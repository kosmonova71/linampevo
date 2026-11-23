"""CameraGenes — cinematography / viewpoint genome module."""

from __future__ import annotations

import math
import random

TWO_PI = 6.28318


# Per-gene safe ranges: (lo, hi)
BOUNDS: dict[str, tuple[float, float]] = {
    "dimension": (0, 3),
    "camera_speed": (0.0, 5.0),
    "camera_orbit": (-6.28, 6.28),
    "camera_depth": (1.0, 10.0),
    "camera_roll": (-6.28, 6.28),
    "camera_fov": (0.5, 3.0),
    "depth_scale": (0.0, 10.0),
    "ray_steps": (8, 64),
    "camera_focus": (0.0, 1.0),
    "camera_aperture": (0.0, 1.0),
    "camera_exposure": (0.0, 1.5),
    "camera_autoexp": (0.0, 1.0),
    "camera_shake": (0.0, 1.5),
    "camera_shake_freq": (0.0, 2.0),
    "camera_tracking": (0.0, 1.0),
    "camera_framing": (0.0, 1.0),
    "camera_headroom": (0.0, 1.0),
    "camera_school": (0, 7),
    "camera_shot_length": (0.0, 2.0),
    "camera_transition": (0.0, 1.0),
    "camera_crane": (0.0, 1.0),
    "camera_parallax": (0.0, 1.0),
    "camera_drift": (0.0, 1.0),
    "camera_focus_pull": (0.0, 1.0),
    "camera_handheld": (0.0, 1.0),
    "camera_dolly": (0.0, 1.0),
    "camera_pan": (0.0, 1.0),
    "camera_tilt": (0.0, 1.0),
}

INT_KEYS = {"dimension", "ray_steps", "camera_school"}


class CameraGenes:
    """Camera / viewpoint genes controlling 3D projection, DoF, shake, framing.

    Attributes correspond directly to shader uniforms prefixed with ``k_``.
    """

    __slots__ = (
        "camera_aperture",
        "camera_autoexp",
        "camera_crane",
        "camera_depth",
        "camera_dolly",
        "camera_drift",
        "camera_exposure",
        "camera_focus",
        "camera_focus_pull",
        "camera_fov",
        "camera_framing",
        "camera_handheld",
        "camera_headroom",
        "camera_orbit",
        "camera_pan",
        "camera_parallax",
        "camera_roll",
        "camera_school",
        "camera_shake",
        "camera_shake_freq",
        "camera_shot_length",
        "camera_speed",
        "camera_tilt",
        "camera_tracking",
        "camera_transition",
        "depth_scale",
        "dimension",
        "ray_steps",
    )

    def __init__(self, *, rng: random.Random | None = None):
        r = rng or random
        self.dimension: int = r.randint(0, 3)
        self.camera_speed: float = r.uniform(0.5, 2.0)
        self.camera_orbit: float = r.uniform(-3.0, 3.0)
        self.camera_depth: float = r.uniform(2.0, 5.0)
        self.camera_roll: float = r.uniform(-2.0, 2.0)
        self.camera_fov: float = r.uniform(1.0, 2.0)
        self.depth_scale: float = r.uniform(0.5, 3.0)
        self.ray_steps: int = r.randint(16, 48)
        self.camera_focus: float = r.uniform(0.2, 0.8)
        self.camera_aperture: float = r.uniform(0.05, 0.6)
        self.camera_exposure: float = r.uniform(0.2, 1.0)
        self.camera_autoexp: float = r.uniform(0.0, 0.8)
        self.camera_shake: float = r.uniform(0.0, 0.8)
        self.camera_shake_freq: float = r.uniform(0.2, 1.5)
        self.camera_tracking: float = r.uniform(0.2, 0.8)
        self.camera_framing: float = r.uniform(0.2, 0.8)
        self.camera_headroom: float = r.uniform(0.2, 0.8)
        self.camera_school: int = r.randint(0, 7)
        self.camera_shot_length: float = r.uniform(0.3, 1.5)
        self.camera_transition: float = r.uniform(0.1, 0.7)
        self.camera_crane: float = r.uniform(0.0, 0.6)
        self.camera_parallax: float = r.uniform(0.2, 0.8)
        self.camera_drift: float = r.uniform(0.0, 0.5)
        self.camera_focus_pull: float = r.uniform(0.2, 0.9)
        self.camera_handheld: float = r.uniform(0.0, 0.6)
        self.camera_dolly: float = r.uniform(0.2, 0.8)
        self.camera_pan: float = r.uniform(0.1, 0.7)
        self.camera_tilt: float = r.uniform(0.1, 0.7)

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
        """Clamp all genes to valid ranges."""
        for key, (lo, hi) in BOUNDS.items():
            val = getattr(self, key)
            if key in INT_KEYS:
                setattr(self, key, int(max(lo, min(hi, val))))
            else:
                v = float(val)
                if key == "camera_orbit":
                    v = ((v + math.pi) % TWO_PI) - math.pi
                setattr(self, key, max(lo, min(hi, v)))

    def to_dict(self) -> dict[str, float | int]:
        return {k: getattr(self, k) for k in BOUNDS}

    def from_dict(self, data: dict[str, float | int]) -> None:
        for k in BOUNDS:
            if k in data:
                setattr(self, k, data[k])
        self.clamp()

    def clone(self) -> CameraGenes:
        c = CameraGenes.__new__(CameraGenes)
        for k in BOUNDS:
            setattr(c, k, getattr(self, k))
        return c

    def distance(self, other: CameraGenes) -> float:
        """0..1 normalized distance to another CameraGenes."""
        total = 0.0
        for k, (lo, hi) in BOUNDS.items():
            rng = max(abs(hi - lo), 1e-6)
            total += abs(float(getattr(self, k)) - float(getattr(other, k))) / rng
        return total / max(len(BOUNDS), 1)
