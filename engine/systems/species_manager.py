"""engine/systems/species_manager.py

Species tracking and management for the evolution system.

Clusters genomes into species by phenotypic distance, tracks species
formation events, and manages species-level knowledge sharing.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .interfaces import IGenome


class SpeciesRecord:
    """Record for a single species."""

    def __init__(self, species_id: str, representative: IGenome):
        self.species_id = species_id
        self.representative = representative
        self.members: list[IGenome] = [representative]
        self.total_fitness: float = 0.0
        self.count: int = 1
        self.best_fitness: float = representative.fitness

    def add_member(self, genome: IGenome) -> None:
        self.members.append(genome)
        self.total_fitness += genome.fitness
        self.count += 1
        self.best_fitness = max(self.best_fitness, genome.fitness)

    def avg_fitness(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_fitness / self.count


class SpeciesManager:
    """Track species formation and manage species-level operations.

    Uses phenotypic distance to cluster genomes into species. Genomes
    closer than the species threshold to an existing representative
    join that species; otherwise a new species is created.

    This class knows nothing about rendering or OpenGL.
    """

    def __init__(self, distance_threshold: float = 0.5):
        self.distance_threshold = float(distance_threshold)
        self._species: dict[str, SpeciesRecord] = {}
        self._genome_to_species: dict[str, str] = {}

    def classify(self, genome: IGenome) -> str:
        """Return the species ID for a genome, creating a new species if needed."""
        gid = getattr(genome, "lineage", str(id(genome)))

        if gid in self._genome_to_species:
            return self._genome_to_species[gid]

        # Try to match existing species
        best_species: str | None = None
        best_dist = float("inf")

        for sid, record in self._species.items():
            try:
                dist = genome.distance(record.representative)
                if dist < best_dist:
                    best_dist = dist
                    best_species = sid
            except Exception:
                continue

        if best_species is not None and best_dist <= self.distance_threshold:
            species_id = best_species
        else:
            species_id = f"sp_{random.randint(10000, 99999)}"
            self._species[species_id] = SpeciesRecord(species_id, genome)

        self._genome_to_species[gid] = species_id
        self._species[species_id].add_member(genome)
        return species_id

    def record(self, genome: IGenome, fitness: float) -> None:
        """Record a genome observation for species tracking."""
        self.classify(genome)

    def get_species_summary(self) -> dict[str, int]:
        """Return counts of genomes per species."""
        return {sid: rec.count for sid, rec in self._species.items()}

    def get_species_count(self) -> int:
        return len(self._species)

    def share_brains(self, genomes: list[IGenome], share_prob: float = 0.12) -> None:
        """Species-level knowledge sharing.

        High-fitness organisms broadcast their brain weights to nearby
        lower-fitness organisms in the same species.
        """
        if not genomes:
            return

        by_species: dict[str, list[IGenome]] = defaultdict(list)
        for g in genomes:
            by_species[self.classify(g)].append(g)

        for members in by_species.values():
            if len(members) < 2:
                continue
            scores = [max(1e-3, g.fitness) for g in members]
            avg_score = sum(scores) / len(scores)
            for org in members:
                if org.fitness > avg_score:
                    for neighbour in members:
                        if neighbour is org and random.random() < share_prob:
                            try:
                                neighbour_brain = getattr(neighbour, "brain", None)
                                if neighbour_brain is not None:
                                    neighbour_brain.genome = org.brain.genome.clone()
                            except Exception:
                                pass

    def reset(self) -> None:
        """Clear all species data."""
        self._species.clear()
        self._genome_to_species.clear()
