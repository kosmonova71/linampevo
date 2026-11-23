"""engine/systems/population.py

Population management for the evolution system.

Maintains a list of genomes, tracks the champion, and provides selection
utilities. Completely independent of rendering.
"""

from __future__ import annotations

import random

from .interfaces import IGenome


class Population:
    """Manage a population of evolved genomes.

    Tracks the champion, provides fitness-proportionate selection, and
    supports elite preservation.

    This class knows nothing about shaders, OpenGL, or rendering. It
    operates purely on IGenome instances.
    """

    def __init__(
        self,
        initial_genomes: list[IGenome] | None = None,
        size: int = 3,
    ):
        self._genomes: list[IGenome] = list(initial_genomes or [])
        self._size = max(1, int(size))
        self._champion: IGenome | None = None
        self._best_score: float = float("-inf")

    @property
    def champion(self) -> IGenome | None:
        return self._champion

    @property
    def best_score(self) -> float:
        return self._best_score

    def add(self, genome: IGenome) -> None:
        self._genomes.append(genome)
        self._update_champion(genome)

    def remove(self, genome: IGenome) -> None:
        try:
            self._genomes.remove(genome)
        except ValueError:
            pass
        if self._champion is genome:
            self._recompute_champion()

    def size(self) -> int:
        return len(self._genomes)

    def all(self) -> list[IGenome]:
        return list(self._genomes)

    def get_champion(self) -> IGenome | None:
        return self._champion

    def select_parents(self, n: int) -> list[IGenome]:
        """Select n parents using fitness-proportionate (roulette wheel) selection.

        Parents may be selected multiple times (with replacement).
        """
        n = int(n)
        if not self._genomes:
            return []

        scores = [max(0.0, g.fitness) for g in self._genomes]
        total = sum(scores)
        if total <= 0.0:
            return [random.choice(self._genomes) for _ in range(n)]

        weights = [s / total for s in scores]
        return random.choices(self._genomes, weights=weights, k=n)

    def select_elites(self, elite_frac: float = 0.25) -> list[IGenome]:
        """Return the top elite_frac fraction of genomes by fitness."""
        if not self._genomes:
            return []
        ranked = sorted(self._genomes, key=lambda g: g.fitness, reverse=True)
        count = max(1, round(elite_frac * len(ranked)))
        return ranked[:count]

    def replace(self, new_genomes: list[IGenome]) -> None:
        """Replace the entire population."""
        self._genomes = list(new_genomes)
        self._recompute_champion()

    def record_fitness(self, genome: IGenome, score: float) -> None:
        """Update a genome's fitness and champion tracking."""
        genome.fitness = float(score)
        self._update_champion(genome)

    def _update_champion(self, genome: IGenome) -> None:
        if genome.fitness > self._best_score:
            self._best_score = genome.fitness
            self._champion = genome

    def _recompute_champion(self) -> None:
        if not self._genomes:
            self._champion = None
            self._best_score = float("-inf")
            return
        best = max(self._genomes, key=lambda g: g.fitness)
        self._champion = best
        self._best_score = best.fitness
