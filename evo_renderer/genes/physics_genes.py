from __future__ import annotations

import random

BOUNDS: dict[str, tuple[float, float]] = {
    "fluid_strength": (0.0, 2.0),
    "fluid_viscosity": (0.8, 1.0),
    "fluid_swirl": (0.0, 2.0),
    "fluid_audio_coupling": (0.0, 1.0),
    "wave_speed": (0.1, 3.0),
    "wave_damping": (0.9, 1.0),
    "wave_frequency": (0.5, 8.0),
    "wave_amplitude": (0.1, 1.0),
    "heat_diffusion": (0.05, 0.5),
    "heat_decay": (0.99, 1.0),
    "heat_sources": (1.0, 5.0),
    "heat_iterations": (1.0, 12.0),
    "ca_rule_a": (1.0, 4.0),
    "ca_rule_b": (2.0, 5.0),
    "ca_speed": (0.1, 3.0),
    "ca_density": (0.3, 0.9),
    "ca_steps": (1.0, 10.0),
    "slime_agents": (10.0, 200.0),
    "slime_deposit": (0.005, 0.05),
    "slime_decay": (0.99, 1.0),
    "slime_turn": (0.1, 1.0),
    "slime_speed": (0.5, 2.0),
    "boid_count": (20.0, 300.0),
    "boid_separation": (0.01, 0.3),
    "boid_alignment": (0.01, 0.2),
    "boid_cohesion": (0.001, 0.02),
    "boid_range": (0.05, 0.3),
    "boid_speed": (0.5, 3.0),
    "phys_particles": (32.0, 300.0),
    "phys_deposit": (0.005, 0.03),
    "phys_decay": (0.99, 1.0),
    "phys_turn": (0.1, 0.8),
    "phys_chemo": (0.1, 0.8),
    "erosion_rate": (0.01, 0.2),
    "deposit_rate": (0.005, 0.1),
    "erosion_gravity": (0.1, 1.0),
    "erosion_evap": (0.001, 0.05),
    "erosion_iterations": (1.0, 16.0),
}

INT_KEYS: set = {
    "heat_sources",
    "heat_iterations",
    "ca_rule_a",
    "ca_rule_b",
    "ca_steps",
    "slime_agents",
    "boid_count",
    "phys_particles",
    "erosion_iterations",
}


class PhysicsGenes:
    """Genes controlling rendering quality simulation patterns.

    Covers fluid feedback, wave equations, heat diffusion,
    cellular automata, slime mold, boids, physarum, and erosion.
    """

    __slots__ = tuple(BOUNDS.keys())

    def __init__(self, *, rng: random.Random | None = None):
        r = rng or random
        for key, (lo, hi) in BOUNDS.items():
            if key in INT_KEYS:
                setattr(self, key, r.randint(int(lo), int(hi)))
            else:
                setattr(self, key, r.uniform(lo, hi))

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

    def clone(self) -> PhysicsGenes:
        c = PhysicsGenes.__new__(PhysicsGenes)
        for k in BOUNDS:
            setattr(c, k, getattr(self, k))
        return c

    def distance(self, other: PhysicsGenes) -> float:
        total = 0.0
        for k, (lo, hi) in BOUNDS.items():
            rng = max(abs(hi - lo), 1e-6)
            total += abs(float(getattr(self, k)) - float(getattr(other, k))) / rng
        return total / max(len(BOUNDS), 1)
