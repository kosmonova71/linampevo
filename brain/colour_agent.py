from __future__ import annotations

from typing import Any

from .agent import VisualAgent


class ColourAgent(VisualAgent):
    """Colour controls palettes/contrast/emotional tone (proxy via colour genes).

    The agent reacts to overall musical energy: low energy pushes toward deep,
    smooth colour; high energy pushes toward high contrast.  It also records
    which colour mutations it preferred so future suggestions can bias toward
    them.
    """

    def __init__(self, name: str = "colour"):
        super().__init__(name=name)
        self.memory.setdefault("likes", {})

    def suggest(self, genome: Any, audio: Any) -> dict[str, Any] | None:
        from .agent import _audio_channel  # local import to avoid circular side-effects

        bass = _audio_channel(audio, "bass")
        treble = _audio_channel(audio, "treble", "treb")
        energy = (bass + treble) / 2.0
        likes = self.memory.setdefault("likes", {})

        if energy < 0.35:
            return self._make_suggestion(
                "change_colour",
                weight=1.15,
                reason=f"energy={energy:.3f} low -> deep/smooth colour via change_colour",
            )
        if energy > 0.7:
            return self._make_suggestion(
                "change_colour",
                weight=1.25,
                reason=f"energy={energy:.3f} high -> high contrast via change_colour",
            )
        # Moderate energy: lean into previously-liked colour mutations.
        tune_w = 0.8 + 0.2 * float(likes.get("change_colour", 0.0))
        return self._make_suggestion(
            "tune_params",
            weight=tune_w,
            reason="maintain palette; light tuning",
        )

    def record_feedback(self, mutation_id: str, delta: float) -> None:
        """Record whether a colour mutation helped or hurt.

        Mirrors the pattern used by ``identity.state.VisualIdentity.record_experience``
        so the agent can learn independently of the global identity.
        """
        likes = self.memory.setdefault("likes", {})
        if delta > 0.0:
            likes[mutation_id] = float(likes.get(mutation_id, 0.0)) + abs(delta)
        elif delta < 0.0:
            likes[mutation_id] = max(
                0.0, float(likes.get(mutation_id, 0.0)) - abs(delta)
            )

