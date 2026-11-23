"""evo_bio.gene_usage_tracking

Tracks the full gene lifecycle:
    mutation -> visual change -> fitness change -> survival

Genes that never influence visuals become dormant.
Genes repeatedly helping become protected.
Mutation operators that succeed get higher weight (evolution evolves itself).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class GeneStatus(StrEnum):
    DORMANT = "dormant"
    ACTIVE = "active"
    PROTECTED = "protected"
    HARMFUL = "harmful"


@dataclass
class GeneRecord:
    name: str
    mutations: int = 0
    visual_changes: int = 0
    fitness_deltas: list[float] = field(default_factory=list)
    survived: int = 0
    last_visual_mag: float = 0.0
    created: float = field(default_factory=time.time)

    @property
    def visual_rate(self) -> float:
        return self.visual_changes / max(self.mutations, 1)

    @property
    def success_rate(self) -> float:
        if not self.fitness_deltas:
            return 0.0
        positive = sum(1 for d in self.fitness_deltas if d > 0)
        return positive / len(self.fitness_deltas)

    @property
    def survival_rate(self) -> float:
        return self.survived / max(self.mutations, 1)

    @property
    def avg_fitness_delta(self) -> float:
        if not self.fitness_deltas:
            return 0.0
        return sum(self.fitness_deltas) / len(self.fitness_deltas)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mutations": self.mutations,
            "visual_changes": self.visual_changes,
            "visual_rate": self.visual_rate,
            "avg_fitness_delta": self.avg_fitness_delta,
            "success_rate": self.success_rate,
            "survival_rate": self.survival_rate,
        }


class GeneUsageTracker:
    """Tracks per-gene lifecycle across mutation, visual impact, fitness, and survival.

    Usage
    -----
    >>> tracker = GeneUsageTracker()
    >>> tracker.record_mutation(["camera_orbit", "warp"], visual_changed=True)
    >>> tracker.record_fitness_change(["camera_orbit"], 0.05)
    >>> tracker.record_survival(["camera_orbit"], survived=True)
    >>> tracker.status("camera_orbit")
    <GeneStatus.ACTIVE: 'active'>
    """

    def __init__(
        self,
        visual_threshold: float = 0.01,
        dormant_after: int = 20,
        protected_after: int = 5,
        harmful_after: int = 8,
    ):
        self._records: dict[str, GeneRecord] = {}
        self.visual_threshold = float(visual_threshold)
        self.dormant_after = max(4, int(dormant_after))
        self.protected_after = max(2, int(protected_after))
        self.harmful_after = max(2, int(harmful_after))
        self._total_mutations = 0
        self._total_visual_changes = 0
        self._total_accepted = 0

    def _get(self, name: str) -> GeneRecord:
        if name not in self._records:
            self._records[name] = GeneRecord(name=name)
        return self._records[name]

    def record_mutation(
        self,
        gene_names: list[str],
        visual_changed: bool = True,
        visual_magnitude: float = 0.0,
    ) -> None:
        for name in gene_names:
            rec = self._get(name)
            rec.mutations += 1
            if visual_changed:
                rec.visual_changes += 1
            rec.last_visual_mag = float(visual_magnitude)
        self._total_mutations += len(gene_names)
        if visual_changed:
            self._total_visual_changes += 1

    def record_fitness_change(self, gene_names: list[str], delta: float) -> None:
        for name in gene_names:
            rec = self._get(name)
            rec.fitness_deltas.append(float(delta))
            if len(rec.fitness_deltas) > 64:
                rec.fitness_deltas = rec.fitness_deltas[-64:]

    def record_survival(self, gene_names: list[str], survived: bool) -> None:
        if not survived:
            return
        for name in gene_names:
            rec = self._get(name)
            rec.survived += 1

    def status(self, name: str) -> GeneStatus:
        rec = self._records.get(name)
        if rec is None:
            return GeneStatus.ACTIVE
        if rec.mutations >= self.dormant_after and rec.visual_changes == 0:
            return GeneStatus.DORMANT
        if rec.mutations >= self.protected_after and rec.success_rate >= 0.65:
            return GeneStatus.PROTECTED
        if rec.mutations >= self.harmful_after and rec.success_rate <= 0.25:
            return GeneStatus.HARMFUL
        return GeneStatus.ACTIVE

    def is_dormant(self, name: str) -> bool:
        return self.status(name) == GeneStatus.DORMANT

    def is_protected(self, name: str) -> bool:
        return self.status(name) == GeneStatus.PROTECTED

    def is_harmful(self, name: str) -> bool:
        return self.status(name) == GeneStatus.HARMFUL

    def weights(self, all_gene_names: list[str] | None = None) -> dict[str, float]:
        """Return normalized weights for all known/queried genes."""
        names = set(self._records)
        if all_gene_names:
            names.update(all_gene_names)
        w: dict[str, float] = {}
        for name in names:
            s = self.status(name)
            if s == GeneStatus.PROTECTED:
                w[name] = 2.5
            elif s == GeneStatus.DORMANT:
                w[name] = 0.02
            elif s == GeneStatus.HARMFUL:
                w[name] = 0.08
            else:
                w[name] = 1.0
        return w

    def module_weight(self, module_type: str, bound_keys: list[str]) -> float:
        """Average gene weight for a module, used to bias module selection."""
        weights = self.weights([f"{module_type}.{k}" for k in bound_keys])
        vals = [weights.get(f"{module_type}.{k}", 1.0) for k in bound_keys]
        return sum(vals) / max(len(vals), 1)

    def summary(self, top_k: int = 10) -> dict:
        counts: dict[GeneStatus, int] = {s: 0 for s in GeneStatus}
        for name in self._records:
            counts[self.status(name)] += 1
        top_protected = sorted(
            [n for n, r in self._records.items() if self.status(n) == GeneStatus.PROTECTED],
            key=lambda n: (self._records[n].success_rate, self._records[n].avg_fitness_delta),
            reverse=True,
        )[:top_k]
        top_dormant = sorted(
            [n for n, r in self._records.items() if self.status(n) == GeneStatus.DORMANT],
            key=lambda n: self._records[n].mutations,
            reverse=True,
        )[:top_k]
        top_harmful = sorted(
            [n for n, r in self._records.items() if self.status(n) == GeneStatus.HARMFUL],
            key=lambda n: self._records[n].avg_fitness_delta,
        )[:top_k]
        return {
            "total_genes": len(self._records),
            "total_mutations": self._total_mutations,
            "total_visual_changes": self._total_visual_changes,
            "total_accepted": self._total_accepted,
            "status_counts": {s.value: c for s, c in counts.items()},
            "top_protected": top_protected,
            "top_dormant": top_dormant,
            "top_harmful": top_harmful,
        }

    def reset(self) -> None:
        self._records.clear()
        self._total_mutations = 0
        self._total_visual_changes = 0
        self._total_accepted = 0


class MutationOperatorTracker:
    """Tracks which mutation operators/strategies work best.

    This is the mechanism by which "evolution evolves its own mutation operators".
    """

    def __init__(self, operator_names: list[str] | None = None):
        self._attempts: dict[str, int] = {}
        self._accepts: dict[str, int] = {}
        self._fitness_deltas: dict[str, list[float]] = {}
        self._names = set(operator_names or [])

    def record(self, operator: str, accepted: bool, fitness_delta: float = 0.0) -> None:
        self._names.add(operator)
        self._attempts[operator] = self._attempts.get(operator, 0) + 1
        self._accepts[operator] = self._accepts.get(operator, 0) + (1 if accepted else 0)
        if accepted:
            d = self._fitness_deltas.setdefault(operator, [])
            d.append(float(fitness_delta))
            if len(d) > 64:
                self._fitness_deltas[operator] = d[-64:]

    def success_rate(self, operator: str) -> float:
        att = self._attempts.get(operator, 0)
        if att == 0:
            return 0.5
        return self._accepts.get(operator, 0) / att

    def avg_fitness_delta(self, operator: str) -> float:
        deltas = self._fitness_deltas.get(operator, [])
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def score(self, operator: str) -> float:
        sr = self.success_rate(operator)
        avg_d = self.avg_fitness_delta(operator)
        return sr * 0.6 + max(0.0, avg_d) * 0.4

    def weights(self) -> dict[str, float]:
        if not self._names:
            return {}
        raw = {op: max(0.02, self.score(op)) for op in self._names}
        total = sum(raw.values()) or 1.0
        return {op: w / total for op, w in raw.items()}

    def choose(self) -> str | None:
        w = self.weights()
        if not w:
            return None
        import random

        return random.choices(list(w.keys()), weights=list(w.values()), k=1)[0]

    def stats(self) -> dict:
        return {
            "operators": {
                op: {
                    "attempts": self._attempts.get(op, 0),
                    "accepts": self._accepts.get(op, 0),
                    "success_rate": self.success_rate(op),
                    "avg_fitness_delta": self.avg_fitness_delta(op),
                }
                for op in self._names
            },
            "weights": self.weights(),
        }
