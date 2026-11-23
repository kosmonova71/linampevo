"""Population and MemoryDNA for evolutionary simulation."""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

from evo_visualizer.genomes import VisualGenome

logger = logging.getLogger(__name__)


class Population:
    """Manages a population of evolving genomes.

    Genome-type agnostic: it only relies on the :class:`GenomeBase` contract
    (``fitness``, ``lineage``, ``mutate``, ``clone``, ``crossover``), so the
    same population works for both ``VisualGenome`` and ``ModularGenome``.
    Thread-safe for concurrent access.

    Features:
    - NSGA-II selection with Pareto dominance and crowding distance
    - Fitness sharing / niching for diversity maintenance
    - Speciation support with species tracking
    - Adaptive mutation rates based on population diversity
    - Elite preservation with diversity pressure
    """

    def __init__(
        self,
        size: int = 32,
        crossover_rate: float = 0.3,
        genome_factory=None,
        elite_fraction: float = 0.15,
        tournament_size: int = 4,
        # New parameters for diversity maintenance
        fitness_sharing_sigma: float = 0.15,
        fitness_sharing_alpha: float = 1.0,
        min_mutation_rate: float = 0.05,
        max_mutation_rate: float = 0.3,
        speciation_threshold: float = 0.3,
        diversity_target: float = 0.4,
    ):
        self.size = max(1, int(size))
        self.crossover_rate = float(crossover_rate)
        self._elite_fraction = max(0.0, min(0.5, float(elite_fraction)))
        self._tournament_size = max(2, int(tournament_size))
        self.organisms: list[Any] = []
        self._generation = 0
        self._best_fitness = 0.0
        self._lineage: dict[str, int] = {}
        self._genome_factory = genome_factory
        self._lock = threading.RLock()

        # Diversity maintenance parameters
        self._fitness_sharing_sigma = max(0.01, float(fitness_sharing_sigma))
        self._fitness_sharing_alpha = max(0.1, float(fitness_sharing_alpha))
        self._min_mutation_rate = max(0.0, min(1.0, float(min_mutation_rate)))
        self._max_mutation_rate = max(0.0, min(1.0, float(max_mutation_rate)))
        self._speciation_threshold = max(0.0, min(1.0, float(speciation_threshold)))
        self._diversity_target = max(0.0, min(1.0, float(diversity_target)))

        # Species tracking
        self._species: dict[int, dict] = {}  # species_id -> {members, centroid, age}
        self._next_species_id = 0

        # Adaptive mutation rate
        self._current_mutation_rate = (self._min_mutation_rate + self._max_mutation_rate) / 2.0

        def _default_factory():
            return VisualGenome(strength=0.1, generation=0)

        if genome_factory is None:
            genome_factory = _default_factory

        for _ in range(self.size):
            g = genome_factory()
            self.organisms.append(g)
            self._record_lineage(g.lineage)

    def evolve(self, fitness_fn) -> None:
        with self._lock:
            from evo_bio.pareto import Objectives, ParetoFront

            for g in self.organisms:
                if hasattr(g, "age_one_generation"):
                    g.age_one_generation()
                else:
                    g.age = getattr(g, "age", 0) + 1

            # Build a lineage -> organism lookup once so parent selection
            # never raises StopIteration when lineages collide or the Pareto
            # front returns ids that are not present in the current list.
            lineage_map: dict[str, Any] = {}
            for g in self.organisms:
                lin = getattr(g, "lineage", None) or str(id(g))
                lineage_map[lin] = g

            # Evaluate all organisms and build Pareto front.
            front = ParetoFront()
            scored: list[tuple[float, Any]] = []
            for g in self.organisms:
                if hasattr(g, "objectives") and isinstance(g.objectives, Objectives):
                    objectives = g.objectives
                else:
                    scalar = fitness_fn(g)
                    objectives = Objectives(
                        beauty=float(scalar) * 0.4,
                        novelty=float(scalar) * 0.15,
                        audio_sync=float(scalar) * 0.1,
                        performance=1.0,
                        species_health=float(scalar) * 0.1,
                        motion=float(scalar) * 0.1,
                        complexity=float(scalar) * 0.1,
                        memory=float(scalar) * 0.05,
                    )
                    if hasattr(g, "objectives"):
                        g.objectives = objectives
                front.add(getattr(g, "lineage", str(id(g))), objectives)
                scored.append((objectives.scalar(), g))

            front.build()

            # Apply fitness sharing for diversity maintenance
            self._apply_fitness_sharing(scored, front)

            # Re-sort after fitness sharing
            scored.sort(
                key=lambda x: (
                    front.rank_of(getattr(x[1], "lineage", str(id(x[1])))),
                    -x[0],
                )
            )

            best_scalar = scored[0][0] if scored else 0.0
            self._best_fitness = max(self._best_fitness, best_scalar)

            # Update species tracking
            self._update_species(scored)

            # Adaptive mutation rate based on diversity
            self._adapt_mutation_rate(scored)

            elite_count = max(1, int(self.size * self._elite_fraction))
            elite = scored[:elite_count]
            new_organisms: list[Any] = [g.clone() for _, g in elite]
            for g in new_organisms:
                if hasattr(g, "compute_entropy"):
                    g.compute_entropy()

            while len(new_organisms) < self.size:
                # Use adaptive mutation rate
                mutation_rate = self._current_mutation_rate

                if random.random() < self.crossover_rate and len(new_organisms) >= 2:
                    parent_ids = front.select(k=2)
                    parent_a = lineage_map.get(parent_ids[0])
                    parent_b = lineage_map.get(parent_ids[1])
                    if parent_a is None or parent_b is None:
                        parent_a = self._tournament_select(scored)[1]
                        parent_b = self._tournament_select(scored)[1]
                    child = parent_a.crossover(parent_b)
                    child.mutate(strength=mutation_rate)
                    if hasattr(child, "compute_entropy"):
                        child.compute_entropy()
                else:
                    parent_ids = front.select(k=1)
                    parent = lineage_map.get(parent_ids[0])
                    if parent is None:
                        parent = self._tournament_select(scored)[1]
                    child = parent.clone()
                    child.mutate(strength=mutation_rate)
                    if hasattr(child, "parent_id"):
                        child.parent_id = parent.lineage
                    if hasattr(child, "compute_entropy"):
                        child.compute_entropy()

                child.generation = self._generation + 1
                new_organisms.append(child)
                self._record_lineage(child.lineage)

            self.organisms = new_organisms[: self.size]
            self._generation += 1

    def _apply_fitness_sharing(self, scored: list[tuple[float, Any]], front: Any) -> None:
        """Apply fitness sharing to maintain diversity.

        Fitness sharing reduces the fitness of individuals that are similar
        to many others, encouraging the population to spread across the
        Pareto front.
        """
        if len(scored) < 2:
            return

        # Calculate pairwise distances
        n = len(scored)
        shared_fitness = [0.0] * n

        for i in range(n):
            _, org_i = scored[i]
            params_i = org_i.params if hasattr(org_i, 'params') else {}

            niche_count = 0.0
            for j in range(n):
                if i == j:
                    continue
                _, org_j = scored[j]
                params_j = org_j.params if hasattr(org_j, 'params') else {}

                # Calculate normalized distance
                dist = self._genome_distance(params_i, params_j)
                if dist < self._fitness_sharing_sigma:
                    # Sharing function: 1 - (dist/sigma)^alpha
                    sh = 1.0 - (dist / self._fitness_sharing_sigma) ** self._fitness_sharing_alpha
                    niche_count += sh

            # Shared fitness = raw_fitness / (1 + niche_count)
            raw_fitness = scored[i][0]
            shared_fitness[i] = raw_fitness / (1.0 + niche_count)

        # Update scored with shared fitness
        for i in range(n):
            scored[i] = (shared_fitness[i], scored[i][1])

    def _genome_distance(self, params1: dict, params2: dict) -> float:
        """Calculate normalized distance between two genome parameter sets."""
        if not params1 or not params2:
            return 1.0

        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 1.0

        total_dist = 0.0
        count = 0
        for k in common_keys:
            v1 = params1[k]
            v2 = params2[k]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                total_dist += abs(float(v1) - float(v2))
                count += 1

        return total_dist / max(count, 1)

    def _update_species(self, scored: list[tuple[float, Any]]) -> None:
        """Update species tracking for speciation support."""
        # Assign organisms to species based on genetic distance
        new_species: dict[int, dict] = {}
        next_sid = 0

        for _, org in scored:
            assigned = False

            # Try to assign to existing species
            for sid, sp_data in self._species.items():
                centroid = sp_data.get('centroid')
                if centroid is not None:
                    dist = org.distance(centroid) if hasattr(org, 'distance') else 1.0
                    if dist < self._speciation_threshold:
                        # Assign to this species
                        if sid not in new_species:
                            new_species[sid] = {'members': [], 'centroid': None, 'age': sp_data.get('age', 0) + 1}
                        new_species[sid]['members'].append(org)
                        assigned = True
                        break

            if not assigned:
                # Create new species
                new_species[next_sid] = {'members': [org], 'centroid': None, 'age': 0}
                next_sid += 1

        # Recompute centroids
        for sid, sp_data in new_species.items():
            members = sp_data['members']
            if len(members) > 1:
                centroid = members[0].clone()
                param_sums: dict[str, float] = {}
                param_counts: dict[str, int] = {}
                for m in members:
                    params = m.params if hasattr(m, 'params') else {}
                    for k, v in params.items():
                        if isinstance(v, (int, float)):
                            param_sums[k] = param_sums.get(k, 0.0) + float(v)
                            param_counts[k] = param_counts.get(k, 0) + 1
                avg_params = {}
                for k in param_sums:
                    if param_counts[k] > 0:
                        avg_params[k] = param_sums[k] / param_counts[k]
                # Use params setter to update centroid
                centroid.params = avg_params
                if hasattr(centroid, 'clamp'):
                    centroid.clamp()
                sp_data['centroid'] = centroid
            elif len(members) == 1:
                sp_data['centroid'] = members[0].clone()

        self._species = new_species

    def _adapt_mutation_rate(self, scored: list[tuple[float, Any]]) -> None:
        """Adapt mutation rate based on population diversity."""
        if len(scored) < 2:
            return

        # Calculate average pairwise distance (diversity)
        total_dist = 0.0
        count = 0
        for i in range(len(scored)):
            _, org_i = scored[i]
            params_i = org_i.params if hasattr(org_i, 'params') else {}
            for j in range(i + 1, len(scored)):
                _, org_j = scored[j]
                params_j = org_j.params if hasattr(org_j, 'params') else {}
                dist = self._genome_distance(params_i, params_j)
                total_dist += dist
                count += 1

        if count == 0:
            return

        avg_diversity = total_dist / count

        # Adjust mutation rate: increase if diversity is low, decrease if high
        if avg_diversity < self._diversity_target * 0.5:
            # Low diversity - increase mutation
            self._current_mutation_rate = min(
                self._max_mutation_rate,
                self._current_mutation_rate * 1.1
            )
        elif avg_diversity > self._diversity_target * 1.5:
            # High diversity - decrease mutation
            self._current_mutation_rate = max(
                self._min_mutation_rate,
                self._current_mutation_rate * 0.95
            )
        else:
            # Near target - slow adaptation toward center
            center = (self._min_mutation_rate + self._max_mutation_rate) / 2.0
            if self._current_mutation_rate > center:
                self._current_mutation_rate *= 0.99
            else:
                self._current_mutation_rate *= 1.01

    def _tournament_select(self, scores):
        contestants = random.sample(scores, min(self._tournament_size, len(scores)))
        return max(contestants, key=lambda x: x[0])[1]

    def _record_lineage(self, lineage: str) -> None:
        """Increment the lineage counter, keeping the dict bounded.

        Without a cap the set of unique lineage strings grows forever over long
        sessions; we keep the most recent entries only.
        Thread-safe: caller must hold lock.
        """
        self._lineage[lineage] = self._lineage.get(lineage, 0) + 1
        if len(self._lineage) > 1024:
            # Drop the oldest recorded lineage (dicts preserve insertion order).
            self._lineage.pop(next(iter(self._lineage)))

    def best(self) -> Any:
        """Return the organism with highest Pareto rank (closest to front 0)."""
        with self._lock:
            from evo_bio.pareto import ParetoFront

            if not self.organisms:
                return None
            front = ParetoFront()
            for g in self.organisms:
                obj = getattr(g, "objectives", None)
                if obj is None:
                    from evo_bio.pareto import Objectives

                    obj = Objectives()
                front.add(getattr(g, "lineage", str(id(g))), obj)
            front.build()
            best_id = front.best()
            for g in self.organisms:
                if getattr(g, "lineage", str(id(g))) == best_id:
                    return g
            return self.organisms[0]

    def stats(self) -> dict:
        """Return population statistics."""
        with self._lock:
            # Calculate diversity
            diversity = 0.0
            if len(self.organisms) > 1:
                total_dist = 0.0
                count = 0
                for i in range(len(self.organisms)):
                    params_i = self.organisms[i].params if hasattr(self.organisms[i], 'params') else {}
                    for j in range(i + 1, len(self.organisms)):
                        params_j = self.organisms[j].params if hasattr(self.organisms[j], 'params') else {}
                        dist = self._genome_distance(params_i, params_j)
                        total_dist += dist
                        count += 1
                if count > 0:
                    diversity = total_dist / count

            return {
                "size": len(self.organisms),
                "generation": self._generation,
                "best_fitness": self._best_fitness,
                "unique_lineages": len(self._lineage),
                "num_species": len(self._species),
                "avg_species_age": sum(s['age'] for s in self._species.values()) / max(len(self._species), 1),
                "diversity": diversity,
                "mutation_rate": self._current_mutation_rate,
            }


class MemoryDNA:
    """Tracks lineage and mutation history for hereditary presets."""

    def __init__(self, preset_dir: Path | None = None):
        self.preset_dir = preset_dir or Path.home() / ".evo_renderer" / "lineage"
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        self._lineage_file = self.preset_dir / "lineage.json"
        self._records: dict[str, dict] = {}
        self._last_save = 0.0
        self._save_interval = 30.0
        self._load()

    def _load(self) -> None:
        """Load lineage records from disk."""
        try:
            if self._lineage_file.exists():
                with open(self._lineage_file) as f:
                    self._records = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Failed to load lineage records: %s", e)
            self._records = {}

    def _save(self) -> None:
        """Save lineage records to disk."""
        try:
            with open(self._lineage_file, "w") as f:
                json.dump(self._records, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save lineage: {e}")

    def record(self, genome: Any) -> None:
        """Record a genome in the lineage database.

        Args:
            genome: Genome to record
        """
        objectives = getattr(genome, "objectives", None)
        record = {
            "parent": genome.parent_id,
            "generation": genome.generation,
            "objectives": objectives.to_dict() if objectives else {},
            "mutations": list(genome.mutations[-10:]),
            "timestamp": time.time(),
            "params": genome.to_dict(),
        }
        self._records[genome.lineage] = record
        # Bound the in-memory lineage log so it cannot grow without limit over
        # long sessions (one entry per unique lineage). Disk is the durable copy.
        if len(self._records) > 2000:
            oldest = min(
                self._records,
                key=lambda k: self._records[k].get("timestamp", 0.0),
            )
            del self._records[oldest]
        now = time.time()
        if now - self._last_save >= self._save_interval:
            self._last_save = now
            self._save()

    def get_lineage(self, lineage_id: str) -> dict | None:
        """Get lineage history for a genome.

        Args:
            lineage_id: Lineage identifier

        Returns:
            Lineage record or None
        """
        return self._records.get(lineage_id)

    def trim_to(self, max_records: int) -> int:
        """Drop the oldest lineage records until ``len <= max_records``.

        Used by the proactive memory watchdog so the in-memory lineage log
        (one entry per unique lineage, retained for heredity) cannot grow
        without bound over a long session. Disk remains the durable copy.
        Returns the number of records dropped.
        """
        max_records = max(1, int(max_records))
        dropped = 0
        while len(self._records) > max_records:
            oldest = min(
                self._records,
                key=lambda k: self._records[k].get("timestamp", 0.0),
            )
            del self._records[oldest]
            dropped += 1
        return dropped

    def export_preset(self, genome: Any, path: Path | None = None) -> bool:
        """Export genome as hereditary preset.

        Args:
            genome: Genome to export
            path: Export path

        Returns:
            True if successful
        """
        try:
            if path is None:
                path = self.preset_dir / f"lineage_{genome.lineage}.json"

            preset = {
                "genome": genome.to_dict(),
                "parent": genome.parent_id,
                "generation": genome.generation,
                "fitness": genome.fitness,
                "mutations": list(genome.mutations),
                "timestamp": time.time(),
            }

            with open(path, "w") as f:
                json.dump(preset, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.warning(f"Failed to export preset: {e}")
            return False


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)


# Complexity tiers (spec 57): evolution acquires complexity over time.
COMPLEXITY_TIERS = [
    ("fbm", "none"),  # 0 simple noise
    ("fbm", "curl"),  # 1 fbm
    ("fbm", "vortex"),  # 2 curl
    ("voronoi", "none"),  # 3 voronoi
    ("reaction_diffusion", "curl"),  # 4 reaction diffusion
    ("mandelbrot", "twist"),  # 5 volumetric-ish
    ("sdf", "vortex"),  # 6 raymarch
    ("sdf", "curl"),  # 7 hybrid
]

SIGNAL_TYPES = ("danger", "food", "mate", "music")
