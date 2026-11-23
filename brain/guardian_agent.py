from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GuardianDecision:
    approved: bool
    reason: str
    score: float | None = None


class GuardianAgent:
    """Guardian maintains quality by rejecting low-fitness candidates.

    Two thresholds are enforced:

    * ``min_score`` -- an absolute floor; candidates below it are rejected
      outright.
    * ``min_delta`` -- when a ``parent_score`` is supplied, the candidate's
      ``delta`` must be at least this large to be approved.  This prevents
      the population from accepting near-neutral mutations that would
      merely add noise.
    """

    def __init__(self, min_score: float = 0.0, min_delta: float = 0.0):
        self.min_score = float(min_score)
        self.min_delta = float(min_delta)

    def approve(
        self, candidate: Any, *, parent_score: float | None = None
    ) -> GuardianDecision:
        score = None
        delta = None
        if isinstance(candidate, dict):
            score = candidate.get("score")
            delta = candidate.get("delta")
        else:
            score = getattr(candidate, "score", None)
            delta = getattr(candidate, "delta", None)
        if score is None:
            return GuardianDecision(False, "no score provided", score=None)
        score = float(score)
        if score < self.min_score:
            return GuardianDecision(
                False,
                f"score {score:.3f} < min_score {self.min_score:.3f}",
                score=score,
            )
        if parent_score is not None and delta is not None:
            delta = float(delta)
            if delta < self.min_delta:
                return GuardianDecision(
                    False,
                    f"delta {delta:.3f} < min_delta {self.min_delta:.3f}",
                    score=score,
                )
        return GuardianDecision(True, "approved", score=score)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "min_score": self.min_score,
            "min_delta": self.min_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardianAgent:
        if not isinstance(data, dict):
            return cls()
        return cls(
            min_score=float(data.get("min_score", 0.0)),
            min_delta=float(data.get("min_delta", 0.0)),
        )

    def clone(self) -> GuardianAgent:
        return self.from_dict(self.to_dict())
