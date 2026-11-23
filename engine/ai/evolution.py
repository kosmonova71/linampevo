"""
ASCEND V9 - Autonomous System Hooks (Evolution)

Evolution engine for autonomous shader generation and optimization.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from engine.ir.ssa import IRBlock


@dataclass
class Genome:
    """Genome representing shader parameters."""

    genes: dict[str, float]
    fitness: float = 0.0

    def mutate(self, rate: float = 0.1) -> Genome:
        """Mutate genome with given rate.

        Args:
            rate: Mutation rate (0-1)

        Returns:
            New mutated genome
        """
        new_genes = self.genes.copy()
        for key in new_genes:
            if random.random() < rate:
                # Gaussian mutation
                new_genes[key] += random.gauss(0, 0.1)
                # Clamp to reasonable range
                new_genes[key] = max(0.0, min(1.0, new_genes[key]))

        return Genome(new_genes, 0.0)

    def crossover(self, other: Genome) -> Genome:
        """Crossover with another genome.

        Args:
            other: Other genome to crossover with

        Returns:
            New child genome
        """
        new_genes = {}
        for key in self.genes:
            if key in other.genes:
                # Uniform crossover
                new_genes[key] = self.genes[key] if random.random() < 0.5 else other.genes[key]
            else:
                new_genes[key] = self.genes[key]

        return Genome(new_genes, 0.0)


class EvolutionEngine:
    """Evolution engine for autonomous shader optimization.

    Uses genetic algorithms to evolve shader parameters.
    """

    def __init__(self, population_size: int = 20):
        """Initialize evolution engine.

        Args:
            population_size: Size of population
        """
        self.population_size = population_size
        self.population: list[Genome] = []
        self.generation = 0
        self.best_genome: Genome | None = None

    def initialize(self, base_genes: dict[str, float]) -> None:
        """Initialize population with base genome.

        Args:
            base_genes: Base gene values
        """
        self.population = []
        for _ in range(self.population_size):
            genes = base_genes.copy()
            # Add random variation
            for key in genes:
                genes[key] += random.gauss(0, 0.05)
                genes[key] = max(0.0, min(1.0, genes[key]))
            self.population.append(Genome(genes, 0.0))

        self.generation = 0

    def evaluate_fitness(self, fitness_fn: Callable[[Genome], float]) -> None:
        """Evaluate fitness of all genomes.

        Args:
            fitness_fn: Function to evaluate genome fitness
        """
        for genome in self.population:
            genome.fitness = fitness_fn(genome)

        # Track best
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        self.best_genome = self.population[0]

    def evolve(self, mutation_rate: float = 0.1) -> None:
        """Evolve population to next generation.

        Args:
            mutation_rate: Mutation rate
        """
        if not self.population:
            return

        # Selection: keep top 50%
        survivors = self.population[: self.population_size // 2]

        # Reproduction
        new_population = []
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)

            # Crossover
            child = parent1.crossover(parent2)

            # Mutate
            child = child.mutate(mutation_rate)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def get_best(self) -> Genome | None:
        """Get best genome from population.

        Returns:
            Best genome, or None if population is empty
        """
        if self.population:
            self.population.sort(key=lambda g: g.fitness, reverse=True)
            return self.population[0]
        return None


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
