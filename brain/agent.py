from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Shared audio channel helper used by all agents.
# Kept in the base module so that ColourAgent, MotionAgent, and StructureAgent
# can import it without duplication (see each agent's suggest() for the import).
def _audio_channel(audio: Any, *names: str) -> float:
    """Return the first available audio channel from *names* (default 0.0).

    The renderer's audio dict sometimes uses ``treb`` and sometimes
    ``treble``; this helper accepts both without forcing callers to care.
    """
    for name in names:
        value = getattr(audio, name, None)
        if value is None and isinstance(audio, dict):
            value = audio.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


class VisualAgent:
    """Base class for visual mutation agents.

    Each agent inspects the current genome and audio state and proposes a
    mutation suggestion.  Suggestions carry a ``mutation_id`` (the operator
    to apply), a ``weight`` (priority 0-1, higher wins ties) and a human
    readable ``reason``.  Returning ``None`` means "no action".

    Subclasses are expected to be cheap to construct and to hold a small
    amount of learned state in :attr:`memory`.  Both :meth:`to_dict` and
    :meth:`from_dict` are provided so agent state can be persisted alongside
    the genome archive, mirroring the pattern used by
    ``evo_renderer.evolution_memory.GenomeArchive``.

    Optimisation note: ``to_dict`` / ``from_dict`` avoid ``copy.deepcopy``
    because the memory dict typically contains only numeric values (floats),
    so a shallow copy is sufficient and much faster.
    """

    __slots__ = ("memory", "name")

    def __init__(self, name: str = "agent"):
        self.name = name
        self.memory: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def suggest(self, genome: Any, audio: Any) -> dict[str, Any] | None:
        """Suggest a mutation based on genome and audio state.

        Args:
            genome: Current genome being evaluated.
            audio: Audio state information.

        Returns:
            A suggestion dict or None if no mutation is suggested.
        """
        return None

    # ------------------------------------------------------------------
    # Suggestion helpers
    # ------------------------------------------------------------------
    def _make_suggestion(
        self,
        mutation_id: str,
        weight: float,
        reason: str,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build a suggestion dict that satisfies the GuardianAgent.

        The guardian requires a numeric ``score`` to compare against its
        ``min_score`` floor and a ``delta`` (optional) for the minimum
        improvement threshold.  Agents express priority via ``weight``;
        we mirror that into ``score`` so the guardian does not reject every
        candidate outright.  Callers can override either field via *extra*.
        """
        suggestion: dict[str, Any] = {
            "mutation_id": mutation_id,
            "weight": float(weight),
            "score": float(weight),
            "reason": reason,
        }
        suggestion.update(extra)
        return suggestion

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise agent state into a JSON-safe dictionary.

        Uses a shallow copy of the memory dict since it typically contains
        only numeric values; this is much faster than ``copy.deepcopy``.
        """
        return {"name": self.name, "memory": dict(self.memory)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualAgent:
        """Reconstruct an agent from a serialised dictionary.

        Unknown keys are ignored so persisted state from future versions
        does not break deserialisation.
        """
        agent = cls(name=str(data.get("name", cls.__name__)))
        mem = data.get("memory")
        if isinstance(mem, dict):
            agent.memory = dict(mem)  # shallow copy
        else:
            agent.memory = {}
        return agent

    def clone(self) -> VisualAgent:
        """Return a copy of this agent.

        Avoids the deepcopy overhead of the original implementation by
        using ``to_dict`` / ``from_dict`` which perform shallow copies.
        """
        return self.from_dict(self.to_dict())

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}(name={self.name!r})"
