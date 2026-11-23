from __future__ import annotations

from typing import Any

from .agent import VisualAgent


class MotionAgent(VisualAgent):
    """Motion controls feedback/flow/camera-like dynamics (proxy via params).

    Strong bass pushes feedback toward its ceiling; strong treble triggers the
    warp module; an overly high mutation rate is stabilised instead.
    """

    def __init__(self, name: str = "motion"):
        super().__init__(name=name)
        self.memory.setdefault("likes", {})

    def suggest(self, genome: Any, audio: Any) -> dict[str, Any] | None:
        from .agent import _audio_channel  # local import to avoid circular side-effects

        bass = _audio_channel(audio, "bass")
        treble = _audio_channel(audio, "treble", "treb")
        feedback = self._param(genome, "feedback", 0.85)
        mutation_rate = self._param(genome, "mutation_rate", 0.1)
        likes = self.memory.setdefault("likes", {})

        if bass > 0.8 and feedback < 0.98:
            return self._make_suggestion(
                "increase_feedback",
                weight=1.25,
                reason=f"bass={bass:.3f} high + feedback={feedback:.3f} -> increase_feedback",
            )
        if treble > 0.7:
            return self._make_suggestion(
                "warp_module",
                weight=1.05,
                reason=f"treble={treble:.3f} high -> warp_module",
            )
        if mutation_rate > 0.2:
            return self._make_suggestion(
                "tune_params",
                weight=0.95,
                reason=f"mutation_rate={mutation_rate:.3f} too high -> tune_params (stabilize)",
            )
        tune_w = 0.85 + 0.15 * float(likes.get("tune_params", 0.0))
        return self._make_suggestion(
            "tune_params",
            weight=tune_w,
            reason="default smooth motion tuning",
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
        """Record whether a motion mutation helped or hurt."""
        likes = self.memory.setdefault("likes", {})
        if delta > 0.0:
            likes[mutation_id] = float(likes.get(mutation_id, 0.0)) + abs(delta)
        elif delta < 0.0:
            likes[mutation_id] = max(
                0.0, float(likes.get(mutation_id, 0.0)) - abs(delta)
            )
