from __future__ import annotations

from typing import Any

from .agent import VisualAgent


class StructureAgent(VisualAgent):
    """Structure controls geometry/complexity/pattern bias.

    Complexity is derived from warp strength and feedback.  Low complexity
    triggers fractal growth; strong bass adds particles for structural
    variability; otherwise the agent tunes params, biased by its own memory
    of which tuning mutations worked.
    """

    def __init__(self, name: str = "structure"):
        super().__init__(name=name)
        self.memory.setdefault("likes", {})

    def suggest(self, genome: Any, audio: Any) -> dict[str, Any] | None:
        from .agent import _audio_channel  # local import to avoid circular side-effects

        warp_strength = self._param(genome, "warp_strength", 0.5)
        feedback = self._param(genome, "feedback", 0.8)
        complexity = 0.55 * warp_strength + 0.45 * feedback
        likes = self.memory.setdefault("likes", {})

        if complexity < 0.5:
            return self._make_suggestion(
                "add_fractal",
                weight=1.2,
                reason=f"complexity={complexity:.3f} < 0.5 -> add_fractal",
            )
        bass = _audio_channel(audio, "bass")
        if bass > 0.7:
            return self._make_suggestion(
                "add_particles",
                weight=1.0,
                reason=f"bass={bass:.3f} -> add_particles (structure variability)",
            )
        tune_w = 0.7 + 0.3 * float(likes.get("tune_params", 0.0))
        return self._make_suggestion(
            "tune_params",
            weight=tune_w,
            reason="maintain structure via params tuning",
        )

    @staticmethod
    def _param(genome: Any, name: str, default: float) -> float:
        """Return a numeric genome parameter, tolerating dict or attribute access."""
        params = getattr(genome, "params", None)
        if params is None:
            params = genome if isinstance(genome, dict) else {}
        try:
            return float(params.get(name, default))  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return float(default)

    def record_feedback(self, mutation_id: str, delta: float) -> None:
        """Record whether a structure mutation helped or hurt."""
        likes = self.memory.setdefault("likes", {})
        if delta > 0.0:
            likes[mutation_id] = float(likes.get(mutation_id, 0.0)) + abs(delta)
        elif delta < 0.0:
            likes[mutation_id] = max(
                0.0, float(likes.get(mutation_id, 0.0)) - abs(delta)
            )
