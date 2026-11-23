#!/usr/bin/env python3

"""evo_bio.pareto

Multi-objective Pareto optimisation for the evolution system.

Replaces the single scalar fitness with eight independent objective axes:

    Beauty        — visual aesthetic quality (entropy, edges, colour, symmetry)
    Novelty       — difference from previously seen phenotypes
    Audio_sync    — alignment between visual change and audio energy
    Performance   — GPU/runtime cost (lower cost = higher score)
    Species_health — species diversity and ecosystem stability
    Motion        — visual movement and kinetic energy
    Complexity    — structural and computational depth
    Memory        — temporal stability and coherence

No single objective can dominate. Selection uses Pareto ranking: organisms
on the non-dominated frontier survive preferentially, but dominated
organisms still breed occasionally to maintain genetic diversity.

Usage
-----
    from evo_bio.pareto import ParetoFront, evaluate_objectives

    objectives = evaluate_objectives(genome, image_data, audio_data, ...)
    front = ParetoFront()
    front.add(organism_id, objectives)
    rank = front.rank_of(organism_id)
    parents = front.select(k=2)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

OBJECTIVES = [
    "beauty",
    "novelty",
    "audio_sync",
    "performance",
    "species_health",
    "motion",
    "complexity",
    "memory",
]


@dataclass
class Objectives:
    """Eight-axis objective vector. All values clamped 0..1."""

    beauty: float = 0.0
    novelty: float = 0.0
    audio_sync: float = 0.0
    performance: float = 0.0
    species_health: float = 0.0
    motion: float = 0.0
    complexity: float = 0.0
    memory: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "beauty": self.beauty,
            "novelty": self.novelty,
            "audio_sync": self.audio_sync,
            "performance": self.performance,
            "species_health": self.species_health,
            "motion": self.motion,
            "complexity": self.complexity,
            "memory": self.memory,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Objectives:
        return cls(
            beauty=float(d.get("beauty", 0.0)),
            novelty=float(d.get("novelty", 0.0)),
            audio_sync=float(d.get("audio_sync", 0.0)),
            performance=float(d.get("performance", 0.0)),
            species_health=float(d.get("species_health", 0.0)),
            motion=float(d.get("motion", 0.0)),
            complexity=float(d.get("complexity", 0.0)),
            memory=float(d.get("memory", 0.0)),
        )

    def dominates(self, other: Objectives) -> bool:
        """True if this vector is >= on all axes and strictly better on at least one."""
        at_least_one = False
        for k in OBJECTIVES:
            a = getattr(self, k)
            b = getattr(other, k)
            if a < b:
                return False
            if a > b:
                at_least_one = True
        return at_least_one

    def scalar(self, weights: dict[str, float] | None = None) -> float:
        """Weighted sum for display / tie-breaking only."""
        w = weights or {k: 1.0 / len(OBJECTIVES) for k in OBJECTIVES}
        return sum(w.get(k, 0.0) * getattr(self, k) for k in OBJECTIVES)

    def __repr__(self) -> str:
        return f"Objectives({self.scalar():.3f})"


class ParetoFront:
    """Maintains a population's Pareto ranking and selection.

    Parameters
    ----------
    rng : random.Random, optional
        Random source for stochastic selection.

    Attributes
    ----------
    _entries : list of (id, Objectives)
        Registered organisms.
    _ranks : dict mapping id -> front index
    """

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()
        self._entries: list[tuple[Any, Objectives]] = []
        self._ranks: dict[Any, int] = {}
        self._dominators: dict[Any, set] = {}

    def clear(self) -> None:
        self._entries.clear()
        self._ranks.clear()
        self._dominators.clear()

    def add(self, obj_id: Any, objectives: Objectives) -> None:
        self._entries.append((obj_id, objectives))

    def build(self) -> None:
        """Compute Pareto ranks for all registered entries (O(n^2))."""
        n = len(self._entries)
        self._ranks.clear()
        self._dominators.clear()

        dominated_count = [0] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._entries[j][1].dominates(self._entries[i][1]):
                    dominated_count[i] += 1

        frontier = [i for i in range(n) if dominated_count[i] == 0]
        level = 0
        remaining = set(range(n))

        while frontier:
            for i in frontier:
                oid, _ = self._entries[i]
                self._ranks[oid] = level
                self._dominators[oid] = set()
                remaining.discard(i)

            next_frontier = []
            for i in remaining:
                # Check if still dominated by anyone in the remaining set
                for j in remaining:
                    if i == j:
                        continue
                    if self._entries[j][1].dominates(self._entries[i][1]):
                        break
                else:
                    next_frontier.append(i)
            frontier = next_frontier
            level += 1

    def rank_of(self, obj_id: Any) -> int:
        return self._ranks.get(obj_id, 999)

    def front(self, level: int = 0) -> list[Any]:
        return [oid for oid, r in self._ranks.items() if r == level]

    def select(self, k: int = 2, pressure: float = 2.0) -> list[Any]:
        """Select *k* parents, favouring lower Pareto ranks.

        Parameters
        ----------
        k : int
            Number of parents to select.
        pressure : float
            Selection pressure; higher = stronger preference for front 0.
        """
        if not self._entries:
            return []

        ids = [oid for oid, _ in self._entries]
        weights = []
        for oid, _ in self._entries:
            rank = self._ranks.get(oid, 999)
            w = math.exp(-pressure * rank)
            weights.append(w)

        total = sum(weights)
        if total <= 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total for w in weights]

        chosen = []
        available = list(range(len(ids)))
        avail_weights = list(weights)
        for _ in range(min(k, len(ids))):
            if not available:
                break
            pick = self._rng.choices(available, weights=avail_weights, k=1)[0]
            pos = available.index(pick)
            chosen.append(ids[pick])
            available.pop(pos)
            avail_weights.pop(pos)

        return chosen

    def best(self, key: str = "beauty") -> Any:
        """Return the organism with the highest value for *key* among front-0."""
        front0 = self.front(0)
        if not front0:
            front0 = list(self._ranks.keys())
        lookup = {oid: obj for oid, obj in self._entries}
        return max(front0, key=lambda oid: getattr(lookup[oid], key, 0.0))


def pareto_dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """Return True if *a* dominates *b* on all objective axes."""
    at_least_one = False
    for k in OBJECTIVES:
        va = float(a.get(k, 0.0))
        vb = float(b.get(k, 0.0))
        if va < vb:
            return False
        if va > vb:
            at_least_one = True
    return at_least_one


def pareto_rank(objectives: list[dict[str, float]]) -> list[int]:
    """Assign Pareto front indices (0 = best front) to each vector."""
    n = len(objectives)
    if n == 0:
        return []

    dominated = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pareto_dominates(objectives[j], objectives[i]):
                dominated[i] += 1

    ranks = [0] * n
    frontier = [i for i in range(n) if dominated[i] == 0]
    level = 0
    remaining = set(range(n))

    while frontier:
        for i in frontier:
            ranks[i] = level
            remaining.discard(i)
        next_frontier = []
        for i in remaining:
            still_dominated = False
            for j in remaining:
                if i != j and pareto_dominates(objectives[j], objectives[i]):
                    still_dominated = True
                    break
            if not still_dominated:
                next_frontier.append(i)
        frontier = next_frontier
        level += 1

    return ranks
