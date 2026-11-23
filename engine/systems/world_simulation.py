"""engine/systems/world_simulation.py

World / physics simulation for the evolution system.

Provides a lightweight world simulation that audio drives, producing
scalar summaries that the renderer can use as shader uniforms. The
evolution system uses world state for environmental affinity scoring.
"""

from __future__ import annotations

from .interfaces import AudioState


class WorldSimulation:
    """Lightweight world simulation driven by audio.

    Tracks aggregate world state (energy, entity count, age) that can
    be fed to shaders as uniforms and used for environmental affinity
    scoring in evolution.

    This class knows nothing about rendering or genomes.
    """

    def __init__(self, width: int = 64, height: int = 64):
        self.width = int(width)
        self.height = int(height)
        self.age: float = 0.0
        self.entity_count: int = 0
        self.energy_mean: float = 0.0
        self.energy_max: float = 0.0
        self._energy_buffer: list = []

    def step(self, dt: float, audio: AudioState) -> None:
        """Advance the simulation by dt seconds using audio as driver."""
        dt = max(0.0, min(dt, 0.1))
        self.age += dt

        energy = float(getattr(audio, "energy", 0.0))
        bass = float(getattr(audio, "bass", 0.0))
        mid = float(getattr(audio, "mid", 0.0))
        treble = float(getattr(audio, "treble", 0.0))

        # Simulate entity count driven by audio energy
        target_count = int(10 + energy * 50 + bass * 20)
        self.entity_count += (target_count - self.entity_count) * min(1.0, dt * 3.0)

        # Energy stats
        combined = bass * 0.5 + mid * 0.3 + treble * 0.2
        self._energy_buffer.append(combined)
        if len(self._energy_buffer) > 120:
            self._energy_buffer = self._energy_buffer[-120:]

        if self._energy_buffer:
            self.energy_mean = sum(self._energy_buffer) / len(self._energy_buffer)
            self.energy_max = max(self._energy_buffer)

    def get_summaries(self) -> dict[str, float]:
        """Return scalar summaries for shader uniforms."""
        return {
            "world_energy_mean": float(self.energy_mean),
            "world_energy_max": float(self.energy_max),
            "world_entity_count": float(self.entity_count),
            "world_age": float(self.age),
        }

    def reset(self) -> None:
        """Reset simulation state."""
        self.age = 0.0
        self.entity_count = 0
        self.energy_mean = 0.0
        self.energy_max = 0.0
        self._energy_buffer.clear()
