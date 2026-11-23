from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .agent import VisualAgent
from .colour_agent import ColourAgent
from .guardian_agent import GuardianAgent
from .motion_agent import MotionAgent
from .structure_agent import StructureAgent

logger = logging.getLogger(__name__)


class BrainCoordinator:
    """Run a team of agents and select the best suggestion.

    The coordinator is the integration point between the ``brain`` package
    and the renderer's evolution loop.  Each frame it:

    1. Asks every registered :class:`VisualAgent` for a suggestion.
    2. Filters the results through a :class:`GuardianAgent` so low-quality
       or low-fitness candidates are rejected.
    3. Picks the suggestion with the highest weight (ties broken by the
       order agents were registered).
    4. Records the outcome back into the agents that produced the winner so
       they can learn.

    The coordinator itself is serialisable via :meth:`to_dict` /
    :meth:`from_dict` so it can be persisted alongside the genome archive.
    """

    def __init__(
        self,
        agents: Sequence[VisualAgent] | None = None,
        guardian: GuardianAgent | None = None,
    ):
        self.agents: list[VisualAgent] = list(agents) if agents else [
            ColourAgent(),
            MotionAgent(),
            StructureAgent(),
        ]
        self.guardian = guardian if guardian is not None else GuardianAgent()
        self.last_suggestion: dict[str, Any] | None = None
        self.last_winner: VisualAgent | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def suggest(self, genome: Any, audio: Any) -> dict[str, Any] | None:
        """Return the best approved suggestion, or None if all rejected."""
        candidates: list[tuple[float, int, VisualAgent, dict[str, Any]]] = []
        for idx, agent in enumerate(self.agents):
            try:
                suggestion = agent.suggest(genome, audio)
            except Exception as exc:  # agent must never crash the loop
                logger.debug("Agent %s raised: %s", agent.name, exc)
                continue
            if not isinstance(suggestion, dict):
                continue
            weight = float(suggestion.get("weight", 1.0))
            candidates.append((weight, idx, agent, suggestion))

        if not candidates:
            self.last_suggestion = None
            self.last_winner = None
            return None

        # Highest weight wins; registration order breaks ties deterministically.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        _best_weight, _idx, winner, suggestion = candidates[0]

        parent_score = getattr(genome, "fitness", None)
        decision = self.guardian.approve(suggestion, parent_score=parent_score)
        if not decision.approved:
            logger.debug(
                "Guardian rejected %s: %s", suggestion.get("mutation_id"), decision.reason
            )
            self.last_suggestion = None
            self.last_winner = None
            return None

        suggestion = dict(suggestion)
        suggestion.setdefault("approved", True)
        suggestion["guardian_score"] = decision.score
        self.last_suggestion = suggestion
        self.last_winner = winner
        return suggestion

    def record_outcome(self, delta: float, mutation_id: str | None = None) -> None:
        """Feed a fitness delta back into the winning agent (if any)."""
        if self.last_winner is None:
            return
        mid = mutation_id or (self.last_suggestion or {}).get("mutation_id")
        if mid is None:
            return
        try:
            self.last_winner.record_feedback(mid, delta)
        except AttributeError:
            # Agent does not implement record_feedback; nothing to learn.
            pass
        except Exception as exc:  # learning must never crash the loop
            logger.debug("record_feedback on %s failed: %s", self.last_winner.name, exc)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.to_dict() for a in self.agents],
            "guardian": self.guardian.to_dict() if hasattr(self.guardian, "to_dict") else {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrainCoordinator:
        agents_data = data.get("agents") or []
        agents: list[VisualAgent] = []
        for entry in agents_data:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).lower()
            klass = _AGENT_REGISTRY.get(name)
            if klass is None:
                continue
            agents.append(klass.from_dict(entry))
        guardian = None
        gd = data.get("guardian")
        if isinstance(gd, dict) and hasattr(GuardianAgent, "from_dict"):
            guardian = GuardianAgent.from_dict(gd)
        return cls(agents=agents if agents else None, guardian=guardian)

    def clone(self) -> BrainCoordinator:
        return self.from_dict(self.to_dict())


# Internal registry used by from_dict to reconstruct the right subclass.
_AGENT_REGISTRY: dict[str, type[VisualAgent]] = {
    cls.__name__.lower(): cls
    for cls in (ColourAgent, MotionAgent, StructureAgent)
}
