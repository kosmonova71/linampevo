"""engine/systems/audio_system.py

Audio analysis and state management for the evolution system.

Provides smoothed audio features and derived signals (energy, beat) that
other systems consume without needing to know about audio analysis details.
"""

from __future__ import annotations

import random
from typing import Any

from .interfaces import AudioState, IAudioProvider


class AudioSystem(IAudioProvider):
    """Analyze and smooth raw audio into a stable state for downstream systems.

    Wraps the raw FFT data into smoothed AudioState with derived signals.
    The evolution system and fitness evaluator consume AudioState directly
    without any audio-analysis coupling.
    """

    def __init__(
        self,
        smoothing: float = 0.85,
        beat_smoothing: float = 0.80,
    ):
        self.smoothing = float(smoothing)
        self.beat_smoothing = float(beat_smoothing)
        self.state = AudioState()
        self._raw_levels: dict[str, float] = {}

    def update(self, levels: dict[str, Any]) -> None:
        """Update audio state from raw FFT-derived levels.

        Expected keys: bass, mid, treble/treb, beat (optional), smoothing (optional).
        """
        self._raw_levels = dict(levels)
        self.state.set_levels(levels)

    def get_state(self) -> AudioState:
        return self.state

    def get_levels(self) -> dict[str, float]:
        return {
            "bass": self.state.bass,
            "mid": self.state.mid,
            "treble": self.state.treble,
            "energy": self.state.energy,
            "beat": self.state.beat,
        }

    def get_audio_pressure(self) -> dict[str, float]:
        """Return audio features that drive mutation pressure."""
        return {
            "bass": self.state.bass,
            "mid": self.state.mid,
            "treble": self.state.treble,
            "energy": self.state.energy,
            "beat": self.state.beat,
            "is_silent": self._is_silent(),
        }

    def is_silent(self) -> bool:
        return self._is_silent()

    def _is_silent(self) -> bool:
        avg = (self.state.bass + self.state.mid + self.state.treble) / 3.0
        return avg < 0.08

    def get_random_concept_seed(self) -> dict[str, float]:
        """Return a random concept seed for novelty search (Stage 8)."""
        return {
            "seed": random.random(),
            "energy_bias": random.uniform(0.0, 1.0),
            "complexity_bias": random.uniform(0.0, 1.0),
        }
