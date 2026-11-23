"""
ASCEND V9 - Autonomous System Hooks (Evolution)

Evolution engine for autonomous shader generation and optimization.

Features:
- NSGA-II multi-objective optimization
- Simulated Binary Crossover (SBX)
- Polynomial mutation
- Fitness sharing for diversity maintenance
- Adaptive mutation rates
- Elite preservation with crowding distance
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from engine.ir.ssa import IRBlock


@dataclass
class Genome:
    """Genome representing shader parameters."""

    genes: dict[str, float]
    fitness: float = 0.0
    objectives: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0
    lineage: str = field(default_factory=lambda: f"gen_{random.randint(1000, 9999)}")
    generation: int = 0
    parent_ids: tuple[str, str] | None = None

    def mutate(self, rate: float = 0.1, eta: float = 20.0) -> Genome:
        """Mutate genome with polynomial mutation.

        Args:
            rate: Mutation rate (0-1) per gene
            eta: Distribution index for polynomial mutation (higher = smaller mutations)

        Returns:
            New mutated genome
        """
        new_genes = self.genes.copy()
        for key in new_genes:
            if random.random() < rate:
                # Polynomial mutation
                u = random.random()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                new_genes[key] = max(0.0, min(1.0, new_genes[key] + delta))

        return Genome(
            new_genes,
            0.0,
            lineage=f"{self.lineage}_mut",
            generation=self.generation + 1,
            parent_ids=(self.lineage, self.lineage)
        )

    def crossover(self, other: Genome, rate: float = 0.9, eta: float = 20.0) -> Genome:
        """Crossover with another genome using SBX.

        Args:
            other: Other genome to crossover with
            rate: Crossover probability
            eta: Distribution index for SBX (higher = more similar to parents)

        Returns:
            New child genome
        """
        new_genes = {}
        for key in self.genes:
            if key in other.genes and random.random() < rate:
                # Simulated Binary Crossover (SBX)
                x1 = self.genes[key]
                x2 = other.genes[key]
                u = random.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                child_val = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2)
                new_genes[key] = max(0.0, min(1.0, child_val))
            elif key in other.genes:
                # Uniform crossover
                new_genes[key] = self.genes[key] if random.random() < 0.5 else other.genes[key]
            else:
                new_genes[key] = self.genes[key]

        return Genome(
            new_genes,
            0.0,
            lineage=f"{self.lineage}x{other.lineage}",
            generation=max(self.generation, other.generation) + 1,
            parent_ids=(self.lineage, other.lineage)
        )

    def clone(self) -> Genome:
        """Create a copy of this genome."""
        return Genome(
            self.genes.copy(),
            self.fitness,
            self.objectives.copy(),
            self.rank,
            self.crowding_distance,
            self.lineage,
            self.generation,
            self.parent_ids
        )

    def distance(self, other: Genome) -> float:
        """Calculate normalized Euclidean distance to another genome."""
        common_keys = set(self.genes.keys()) & set(other.genes.keys())
        if not common_keys:
            return 1.0
        total = sum((self.genes[k] - other.genes[k]) ** 2 for k in common_keys)
        return math.sqrt(total / len(common_keys))

    def params(self) -> dict[str, float]:
        """Get genome parameters."""
        return self.genes.copy()


class EvolutionEngine:
    """Evolution engine for autonomous shader optimization using NSGA-II.

    Uses multi-objective genetic algorithms to evolve shader parameters.
    """

    def __init__(
        self,
        population_size: int = 50,
        crossover_rate: float = 0.9,
        crossover_eta: float = 20.0,
        mutation_rate: float = 0.1,
        mutation_eta: float = 20.0,
        elite_fraction: float = 0.1,
        fitness_sharing_sigma: float = 0.15,
        fitness_sharing_alpha: float = 1.0,
        min_mutation_rate: float = 0.05,
        max_mutation_rate: float = 0.3,
        diversity_target: float = 0.3,
    ):
        """Initialize evolution engine.

        Args:
            population_size: Size of population
            crossover_rate: Probability of crossover
            crossover_eta: SBX distribution index
            mutation_rate: Per-gene mutation probability
            mutation_eta: Polynomial mutation distribution index
            elite_fraction: Fraction of population to preserve as elite
            fitness_sharing_sigma: Sharing radius for fitness sharing
            fitness_sharing_alpha: Sharing function exponent
            min_mutation_rate: Minimum adaptive mutation rate
            max_mutation_rate: Maximum adaptive mutation rate
            diversity_target: Target diversity level
        """
        self.population_size = max(4, population_size)
        self.crossover_rate = max(0.0, min(1.0, crossover_rate))
        self.crossover_eta = max(1.0, crossover_eta)
        self.mutation_rate = max(0.0, min(1.0, mutation_rate))
        self.mutation_eta = max(1.0, mutation_eta)
        self.elite_fraction = max(0.0, min(0.5, elite_fraction))
        self.fitness_sharing_sigma = max(0.01, fitness_sharing_sigma)
        self.fitness_sharing_alpha = max(0.1, fitness_sharing_alpha)
        self.min_mutation_rate = max(0.0, min(1.0, min_mutation_rate))
        self.max_mutation_rate = max(0.0, min(1.0, max_mutation_rate))
        self.diversity_target = max(0.0, min(1.0, diversity_target))

        self.population: list[Genome] = []
        self.generation = 0
        self.best_genome: Genome | None = None
        self.current_mutation_rate = mutation_rate
        self._objective_names = [
            "beauty", "novelty", "complexity", "motion", "audio_sync", "stability"
        ]

    def initialize(self, base_genes: dict[str, float]) -> None:
        """Initialize population with base genome."""
        self.population = []
        for i in range(self.population_size):
            genes = base_genes.copy()
            # Add random variation
            for key in genes:
                genes[key] += random.gauss(0, 0.05)
                genes[key] = max(0.0, min(1.0, genes[key]))
            self.population.append(Genome(
                genes,
                0.0,
                lineage=f"init_{i}",
                generation=0
            ))

        self.generation = 0

    def evaluate_fitness(self, fitness_fn: Callable[[Genome], dict[str, float]]) -> None:
        """Evaluate multi-objective fitness of all genomes.

        Args:
            fitness_fn: Function returning dict of objective name -> value (0-1)
        """
        for genome in self.population:
            genome.objectives = fitness_fn(genome)
            # Scalar fitness for backward compatibility (weighted sum)
            weights = {
                "beauty": 0.40,
                "novelty": 0.15,
                "complexity": 0.10,
                "motion": 0.15,
                "audio_sync": 0.10,
                "stability": 0.10,
            }
            genome.fitness = sum(
                genome.objectives.get(k, 0.0) * w for k, w in weights.items()
            )

    def _non_dominated_sort(self, population: list[Genome]) -> list[list[Genome]]:
        """Fast non-dominated sorting (NSGA-II).

        Returns list of fronts, where front 0 is the Pareto front.
        """
        n = len(population)
        domination_count = [0] * n
        dominated_sets = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                rel = self._dominates(population[i], population[j])
                if rel == 1:
                    dominated_sets[i].append(j)
                    domination_count[j] += 1
                elif rel == -1:
                    dominated_sets[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])

        front_idx = 0
        while front_idx < len(fronts) and fronts[front_idx]:
            next_front = []
            for ind in fronts[front_idx]:
                idx_i = population.index(ind)
                for j in dominated_sets[idx_i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = front_idx + 1
                        next_front.append(population[j])
            front_idx += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _dominates(self, a: Genome, b: Genome) -> int:
        """Check if a dominates b.
        Returns: 1 if a dominates b, -1 if b dominates a, 0 otherwise.
        """
        obj_a = a.objectives
        obj_b = b.objectives

        better_in_any = False
        worse_in_any = False

        for key in self._objective_names:
            if key not in obj_a or key not in obj_b:
                continue
            if obj_a[key] > obj_b[key]:
                better_in_any = True
            elif obj_a[key] < obj_b[key]:
                worse_in_any = True

        if better_in_any and not worse_in_any:
            return 1
        if worse_in_any and not better_in_any:
            return -1
        return 0

    def _crowding_distance(self, front: list[Genome]) -> None:
        """Calculate crowding distance for a front."""
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for obj_name in self._objective_names:
            # Sort front by this objective
            front.sort(key=lambda ind: ind.objectives.get(obj_name, 0.0))

            # Boundary points get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Normalize by objective range
            obj_min = front[0].objectives.get(obj_name, 0.0)
            obj_max = front[-1].objectives.get(obj_name, 0.0)
            obj_range = obj_max - obj_min

            if obj_range > 1e-6:
                for i in range(1, len(front) - 1):
                    prev_val = front[i - 1].objectives.get(obj_name, 0.0)
                    next_val = front[i + 1].objectives.get(obj_name, 0.0)
                    front[i].crowding_distance += (next_val - prev_val) / obj_range

    def _fitness_sharing(self, population: list[Genome]) -> None:
        """Apply fitness sharing to maintain diversity."""
        n = len(population)
        if n < 2:
            return

        for i in range(n):
            niche_count = 0.0
            for j in range(n):
                if i == j:
                    continue
                dist = population[i].distance(population[j])
                if dist < self.fitness_sharing_sigma:
                    sh = 1.0 - (dist / self.fitness_sharing_sigma) ** self.fitness_sharing_alpha
                    niche_count += sh

            # Adjust fitness (for selection purposes)
            population[i].fitness /= (1.0 + niche_count)

    def _calculate_diversity(self, population: list[Genome]) -> float:
        """Calculate average pairwise distance in population."""
        n = len(population)
        if n < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += population[i].distance(population[j])
                count += 1

        return total / count if count > 0 else 0.0

    def _adapt_mutation_rate(self) -> None:
        """Adapt mutation rate based on population diversity."""
        diversity = self._calculate_diversity(self.population)

        if diversity < self.diversity_target * 0.5:
            # Low diversity - increase mutation
            self.current_mutation_rate = min(
                self.max_mutation_rate,
                self.current_mutation_rate * 1.1
            )
        elif diversity > self.diversity_target * 1.5:
            # High diversity - decrease mutation
            self.current_mutation_rate = max(
                self.min_mutation_rate,
                self.current_mutation_rate * 0.95
            )
        else:
            # Near target - slow adaptation toward center
            center = (self.min_mutation_rate + self.max_mutation_rate) / 2.0
            if self.current_mutation_rate > center:
                self.current_mutation_rate *= 0.99
            else:
                self.current_mutation_rate *= 1.01

    def evolve(self, mutation_rate: float | None = None) -> None:
        """Evolve population to next generation using NSGA-II.

        Args:
            mutation_rate: Optional override for mutation rate
        """
        if not self.population:
            return

        # Use adaptive mutation rate if not overridden
        mut_rate = mutation_rate if mutation_rate is not None else self.current_mutation_rate

        # Combine current population with offspring for selection
        # (In standard NSGA-II, we create offspring first, then select from combined pool)

        # Create offspring
        offspring = []
        while len(offspring) < self.population_size:
            # Binary tournament selection on rank + crowding
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # Crossover
            if random.random() < self.crossover_rate:
                child = parent1.crossover(parent2, self.crossover_rate, self.crossover_eta)
            else:
                child = parent1.clone()

            # Mutation
            child = child.mutate(mut_rate, self.mutation_eta)

            offspring.append(child)

        # Evaluate offspring (will be done by caller in next generation)
        # For now, just combine and select
        combined = self.population + offspring

        # Non-dominated sorting
        fronts = self._non_dominated_sort(combined)

        # Crowding distance for each front
        for front in fronts:
            self._crowding_distance(front)

        # Fitness sharing
        self._fitness_sharing(combined)

        # Select next generation
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # Sort front by crowding distance (descending)
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                needed = self.population_size - len(new_population)
                new_population.extend(front[:needed])
                break

        self.population = new_population
        self.generation += 1

        # Update best genome (rank 0, highest crowding)
        front0 = [ind for ind in self.population if ind.rank == 0]
        if front0:
            self.best_genome = max(front0, key=lambda ind: ind.crowding_distance)
        elif self.population:
            self.best_genome = min(self.population, key=lambda ind: ind.rank)

        # Adapt mutation rate
        self._adapt_mutation_rate()

    def _tournament_select(self) -> Genome:
        """Binary tournament selection based on rank and crowding distance."""
        a = random.choice(self.population)
        b = random.choice(self.population)

        # Lower rank is better
        if a.rank != b.rank:
            return a if a.rank < b.rank else b

        # Same rank: higher crowding distance is better
        return a if a.crowding_distance > b.crowding_distance else b

    def get_best(self) -> Genome | None:
        """Get best genome from population."""
        if self.population:
            # Prefer rank 0 with highest crowding
            front0 = [g for g in self.population if g.rank == 0]
            if front0:
                return max(front0, key=lambda g: g.crowding_distance)
            return min(self.population, key=lambda g: g.rank)
        return None

    def get_pareto_front(self) -> list[Genome]:
        """Get all non-dominated solutions (rank 0)."""
        return [g for g in self.population if g.rank == 0]

    def stats(self) -> dict[str, Any]:
        """Return population statistics."""
        if not self.population:
            return {
                "size": 0,
                "generation": self.generation,
                "best_fitness": 0.0,
            }

        diversity = self._calculate_diversity(self.population)
        front_sizes = {}
        for g in self.population:
            front_sizes[g.rank] = front_sizes.get(g.rank, 0) + 1

        return {
            "size": len(self.population),
            "generation": self.generation,
            "best_fitness": self.best_genome.fitness if self.best_genome else 0.0,
            "diversity": diversity,
            "mutation_rate": self.current_mutation_rate,
            "front_distribution": front_sizes,
            "pareto_front_size": front_sizes.get(0, 0),
        }


def mutate_ir(block: IRBlock, rate: float = 0.1) -> IRBlock:
    """Mutate IR block by modifying constants.

    Args:
        block: SSA IR block
        rate: Mutation rate

    Returns:
        Mutated IR block
    """
    new_block = block.copy()

    for instr in new_block.instructions:
        if instr.op == "const" and random.random() < rate:
            # Mutate constant value
            value = float(instr.args[0])
            value += random.gauss(0, 0.1 * abs(value))
            instr.args[0] = str(value)

    return new_block


def score_ir(block: IRBlock, frame_output: bytes) -> float:
    """Score IR block based on frame output.

    Args:
        block: SSA IR block
        frame_output: Rendered frame output

    Returns:
        Fitness score (0-1)
    """
    # In a real implementation, this would analyze:
    # - Visual complexity
    # - Color variety
    # - Motion
    # - Audio synchronization

    # Placeholder: random score
    return random.random()
