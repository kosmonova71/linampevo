#!/usr/bin/env python3
"""
SINGULARITY MODE - Self-Evolving Visual Intelligence System

Transforms MilkDrop from static preset player into adaptive, generative system.
"""

import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Pipeline(Enum):
    SAFE = "safe"
    EXPERIMENTAL = "experimental"


@dataclass
class MutationConfig:
    """Configuration for mutation engine."""

    base_rate: float = 0.1
    audio_multiplier: float = 1.5
    chaos_factor: float = 0.3
    max_mutations_per_frame: int = 3
    stability_threshold: float = 1e3


class AudioIntelligence:
    """Maps audio signals to behavioral control parameters."""

    def __init__(self):
        self.bass = 0.0
        self.mid = 0.0
        self.treb = 0.0
        self.beat = 0.0
        self.beat_att = 0.0

        # Behavioral control signals
        self.structural_distortion = 0.0
        self.motion_complexity = 0.0
        self.detail_noise = 0.0
        self.mutation_trigger = 0.0

    def update(self, audio_data: dict[str, float]):
        """Update audio analysis and derive intelligence signals."""
        self.bass = audio_data.get("bass", 0.0)
        self.mid = audio_data.get("mid", 0.0)
        self.treb = audio_data.get("treb", 0.0)
        self.beat = audio_data.get("beat", 0.0)
        self.beat_att = audio_data.get("beat_att", 0.0)

        # Map to behavioral controls
        self.structural_distortion = self.bass * 0.8
        self.motion_complexity = self.mid * 0.6
        self.detail_noise = self.treb * 0.4
        self.mutation_trigger = (self.beat_att * 0.5 + self.beat * 0.5) * self.audio_multiplier

    def get_mutation_rate(self) -> float:
        """Calculate real-time mutation rate based on audio."""
        return min(self.mutation_trigger, 1.0)

    def get_chaos_factor(self) -> float:
        """Get chaos factor for blending safe/experimental."""
        base_chaos = self.structural_distortion * 0.5 + self.motion_complexity * 0.3
        return min(base_chaos * self.chaos_factor, 1.0)


class MutationEngine:
    """Core mutation engine for expressions."""

    def __init__(self, config: MutationConfig = None):
        self.config = config or MutationConfig()
        self.mutation_history: list[str] = []
        self.frame_counter = 0

        # Mutation operators
        self.mutations = [
            self._mutate_constants,
            self._mutate_operators,
            self._mutate_functions,
            self._mutate_structure,
            self._mutate_amplitude,
        ]

    def mutate_expression(self, expr: str, audio_intel: AudioIntelligence) -> str:
        """Apply mutations to an expression based on audio intelligence."""
        if not expr or not expr.strip():
            return expr

        mutated = expr
        num_mutations = 0
        mutation_rate = audio_intel.get_mutation_rate()

        # Apply mutations based on audio-driven probability
        for mutation in self.mutations:
            if num_mutations >= self.config.max_mutations_per_frame:
                break
            if random.random() < mutation_rate:
                mutated = mutation(mutated, audio_intel)
                num_mutations += 1
                self.mutation_history.append(f"{mutation.__name__}: {expr} -> {mutated}")

        return mutated

    def _mutate_constants(self, expr: str, audio_intel: AudioIntelligence) -> str:
        """Mutate numeric constants in expressions."""

        def replace_constant(match):
            try:
                value = float(match.group())
                # Scale based on structural distortion
                scale = 1.0 + audio_intel.structural_distortion * random.uniform(-0.5, 0.5)
                new_value = value * scale
                return f"{new_value:.6f}"
            except (ValueError, AttributeError):
                return match.group()

        return re.sub(r"\b\d+\.?\d*\b", replace_constant, expr)

    def _mutate_operators(self, expr: str, audio_intel: AudioIntelligence) -> str:
        """Mutate mathematical operators."""
        if random.random() < 0.3:  # 30% chance
            # Swap operators based on motion complexity
            if audio_intel.motion_complexity > 0.5:
                expr = expr.replace("+", "*") if random.random() < 0.5 else expr
            elif audio_intel.detail_noise > 0.3:
                expr = expr.replace("*", "+") if random.random() < 0.5 else expr
        return expr

    def _mutate_functions(self, expr: str, audio_intel: AudioIntelligence) -> str:
        """Mutate function calls."""
        functions = ["sin", "cos", "tan", "sqrt", "abs"]
        for func in functions:
            if func in expr and random.random() < 0.2:  # 20% chance per function
                # Swap to different trig function
                new_func = random.choice(["sin", "cos", "tan"])
                expr = expr.replace(f"{func}(", f"{new_func}(")
        return expr

    def _mutate_structure(self, expr: str, audio_intel: AudioIntelligence) -> str:
        """Mutate expression structure."""
        if "=" in expr and random.random() < 0.4:  # 40% chance
            parts = expr.split("=", 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                # Add complexity based on detail noise
                if audio_intel.detail_noise > 0.3:
                    # Wrap in additional function
                    rhs = f"({rhs}) * (1 + {audio_intel.detail_noise:.3f})"
                mutated = f"{lhs} = {rhs}"
                return mutated
        return expr

    def _mutate_amplitude(self, expr: str, audio_intel: AudioIntelligence) -> str:
        """Mutate amplitude/scaling factors."""
        if random.random() < 0.5:  # 50% chance
            chaos = audio_intel.get_chaos_factor()
            scale = 1.0 + chaos * random.uniform(-0.3, 0.3)

            # Find and scale multiplication factors
            def scale_factor(match):
                try:
                    value = float(match.group(2))
                    new_value = value * scale
                    return f"{match.group(1)}{new_value:.6f}{match.group(3)}"
                except (ValueError, AttributeError, IndexError):
                    return match.group()

            return re.sub(r"([+\-*/]\s*)(\d+\.?\d*)(\s*)", scale_factor, expr)

        return expr


class DualPipelineSystem:
    """Manages safe vs experimental execution pipelines."""

    def __init__(self):
        self.safe_expressions: list[Any] = []
        self.experimental_expressions: list[Any] = []
        self.current_pipeline = Pipeline.SAFE
        self.chaos_level = 0.0
        self.frame_count = 0

    def add_expression(self, compiled_expr, pipeline: Pipeline = Pipeline.SAFE):
        """Add expression to specific pipeline."""
        if pipeline == Pipeline.SAFE:
            self.safe_expressions.append(compiled_expr)
        else:
            self.experimental_expressions.append(compiled_expr)

    def evaluate_blended(
        self, state: dict[str, Any], audio_intel: AudioIntelligence
    ) -> dict[str, float]:
        """Evaluate both pipelines and blend results."""
        safe_results = {}
        experimental_results = {}

        # Evaluate safe pipeline
        for expr in self.safe_expressions:
            try:
                safe_results[expr.source] = expr.eval(state)
            except Exception:
                logger.exception("Safe pipeline evaluation failed")
                safe_results[expr.source] = 0.0

        # Evaluate experimental pipeline
        for expr in self.experimental_expressions:
            try:
                experimental_results[expr.source] = expr.eval(state)
            except Exception:
                logger.exception("Experimental pipeline evaluation failed")
                experimental_results[expr.source] = 0.0

        # Blend based on chaos level
        chaos = audio_intel.get_chaos_factor()
        final_results = {}

        all_keys = set(safe_results.keys()) | set(experimental_results.keys())

        for key in all_keys:
            safe_val = safe_results.get(key, 0.0)
            exp_val = experimental_results.get(key, 0.0)

            # Blend: final = safe * (1 - chaos) + experimental * chaos
            final_val = safe_val * (1.0 - chaos) + exp_val * chaos
            final_results[key] = final_val

        return final_results

    def set_pipeline_mode(self, mode: Pipeline):
        """Switch between safe and experimental modes."""
        self.current_pipeline = mode

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Get current pipeline statistics."""
        return {
            "safe_count": len(self.safe_expressions),
            "experimental_count": len(self.experimental_expressions),
            "current_mode": self.current_pipeline.value,
            "chaos_level": self.chaos_level,
            "frame_count": self.frame_count,
        }


class StabilityGuard:
    """Safety system to prevent Singularity Mode explosion."""

    def __init__(self, stability_threshold: float = 1e3):
        self.threshold = stability_threshold
        self.violation_count = 0
        self.last_reset_frame = 0

    def check_state(self, state: dict[str, Any]) -> bool:
        """Check if state is stable."""
        for value in state.values():
            if not isinstance(value, (int, float)):
                continue
            if not np.isfinite(value):
                return False
            if abs(value) > self.threshold:
                self.violation_count += 1
                return False
        return True

    def clamp_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Clamp state values to safe ranges."""
        clamped = {}
        for key, value in state.items():
            if isinstance(value, (int, float)):
                clamped[key] = max(min(value, self.threshold), -self.threshold)
            else:
                clamped[key] = value
        return clamped

    def should_reset(self, frame_count: int) -> bool:
        """Check if system should reset due to instability."""
        if (
            self.violation_count > 5 and frame_count - self.last_reset_frame < 30
        ):  # Too many violations recently
            self.last_reset_frame = frame_count
            self.violation_count = 0
            return True
        return False


class SingularityController:
    """Main controller for SINGULARITY MODE."""

    def __init__(self):
        self.audio_intel = AudioIntelligence()
        self.mutation_engine = MutationEngine()
        self.dual_pipeline = DualPipelineSystem()
        self.stability_guard = StabilityGuard()
        self.is_active = False
        self.frame_count = 0

    def activate(self):
        """Activate Singularity Mode."""
        self.is_active = True

    def deactivate(self):
        """Deactivate Singularity Mode."""
        self.is_active = False

    def update(
        self, audio_data: dict[str, float], preset_state: dict[str, Any]
    ) -> dict[str, float]:
        """Main update loop for Singularity Mode."""
        if not self.is_active:
            return preset_state

        self.frame_count += 1

        # Update audio intelligence
        self.audio_intel.update(audio_data)

        # Check stability
        if not self.stability_guard.check_state(preset_state):
            preset_state = self.stability_guard.clamp_state(preset_state)

        # Check for reset condition
        if self.stability_guard.should_reset(self.frame_count):
            preset_state = self._get_safe_state()

        # Apply mutations every N frames
        if self.frame_count % 30 == 0:  # Every 30 frames
            self._apply_mutations(preset_state)

        # Evaluate with dual pipeline blending
        final_state = self.dual_pipeline.evaluate_blended(preset_state, self.audio_intel)

        self.dual_pipeline.frame_count = self.frame_count
        self.dual_pipeline.chaos_level = self.audio_intel.get_chaos_factor()

        return final_state

    def _apply_mutations(self, state: dict[str, Any]):
        """Apply mutations to experimental expressions."""
        for expr in self.dual_pipeline.experimental_expressions:
            if random.random() < self.audio_intel.get_mutation_rate():
                mutated_source = self.mutation_engine.mutate_expression(
                    expr.source, self.audio_intel
                )
                # Recompile mutated expression
                try:
                    expr.source = mutated_source
                    expr._compile()  # Recompile with new source
                except Exception:
                    logger.debug("Mutation compilation failed", exc_info=True)

    def _get_safe_state(self) -> dict[str, Any]:
        """Get a safe default state."""
        return {
            "zoom": 1.0,
            "rot": 0.0,
            "warp": 0.0,
            "sx": 1.0,
            "sy": 1.0,
            "dx": 0.0,
            "dy": 0.0,
            "r": 1.0,
            "g": 1.0,
            "b": 1.0,
            "a": 1.0,
        }

    def get_status(self) -> dict[str, Any]:
        """Get current Singularity status."""
        return {
            "active": self.is_active,
            "frame_count": self.frame_count,
            "audio_intelligence": {
                "mutation_rate": self.audio_intel.get_mutation_rate(),
                "chaos_factor": self.audio_intel.get_chaos_factor(),
                "structural_distortion": self.audio_intel.structural_distortion,
                "motion_complexity": self.audio_intel.motion_complexity,
                "detail_noise": self.audio_intel.detail_noise,
            },
            "pipeline_stats": self.dual_pipeline.get_pipeline_stats(),
            "stability": {
                "violations": self.stability_guard.violation_count,
                "last_reset": self.stability_guard.last_reset_frame,
            },
        }
