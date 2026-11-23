"""engine/systems/fitness.py

Fitness evaluation for the evolution system.

Computes a scalar fitness score for a genome given audio context, time,
novelty, and GPU cost. The renderer never touches fitness logic.
"""

from __future__ import annotations

import math
from typing import Any

from .interfaces import AudioState, IGenome


class FitnessEvaluator:
    """Evaluate genome fitness using audio proxies + novelty + GPU cost.

    Stage 8 (Option 1): novelty via concept archaeology (optional).
    Stage 10: GPU cost penalty.
    """

    def __init__(
        self,
        concept_archive: Any = None,
        gpu_cost_weight: float = 0.25,
    ):
        self.concept_archive = concept_archive
        self.gpu_cost_weight = float(gpu_cost_weight)

    def evaluate(
        self,
        genome: IGenome,
        audio: AudioState,
        frame: float,
        *,
        novelty: float | None = None,
        gpu_cost: float = 0.0,
    ) -> float:
        bass = float(getattr(audio, "bass", 0.0))
        mid = float(getattr(audio, "mid", 0.0))
        treble = float(getattr(audio, "treble", 0.0))

        motion = (0.5 + 0.5 * math.sin(frame * 0.25)) * (0.4 + 0.6 * (bass + mid + treble) / 3.0)
        audio_sync = bass * (0.7 + 0.3 * math.sin(frame * 0.6 + treble * 2.0))
        complexity = 0.3 * mid + 0.5 * treble

        novelty_term = float(novelty) if novelty is not None else 0.0
        base = float(motion + audio_sync + complexity + novelty_term)

        gpu_penalty = float(self.gpu_cost_weight) * float(gpu_cost)
        return max(0.0, base - gpu_penalty)

    def estimate_gpu_cost(
        self,
        genome: IGenome,
        *,
        modules: list | None = None,
        passes: list | None = None,
        feedback_resolution: float = 1.0,
    ) -> float:
        """Estimate GPU cost from genome topology.

        This is a cheap proxy; real profiling happens on the GPU side.
        """
        cost = 0.0

        # Module complexity cost
        if modules:
            module_weights = {
                "noise": 0.10,
                "warp": 0.12,
                "fractal": 0.42,
                "particles": 0.22,
                "bloom": 0.25,
            }
            for m in modules:
                cost += module_weights.get(m, 0.15)
            cost /= 4.0

        # Pass count cost
        if passes:
            cost += 0.05 * len(passes)

        # Resolution cost (quadratic)
        rr = max(0.1, feedback_resolution)
        cost += (rr * rr) * 0.55

        return float(cost)
