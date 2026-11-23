#!/usr/bin/env python3

"""evo_advanced.py

Advanced evolutionary systems layered on top of the existing spatial
ecosystem (evo_visualizer.py).

This module implements the "developmental biology" tier of the renderer:
organisms are no longer static genomes that get replaced every N seconds.
They grow, they influence their own genes, they are rewarded for being
*different*, they remember, they feel, they repair, they choose mates, they
invent renderers, they occupy niches, they are replayed, and eventually they
begin to modify their own evolutionary machinery (while you keep the safety
constraints).

Features implemented (from the spec list):
  16. Gene Regulatory Networks (GRN)  -- genes influence other genes
  17. Development Instead of Instant Rendering -- egg/embryo/juvenile/adult/elder
  18. Continuous Mutation             -- tiny mutation every frame
  19. Multi-objective Evolution        -- Pareto fitness, no single best
  20. Novelty Search ★                -- reward distance from previous organisms
  21. Memory-Based Evolution           -- organisms remember survivors/predict
  22. Emotional State                  -- fear/curiosity/stress/... modify behaviour
  23. Genome Repair                    -- repair_rate / mutation_filter / redundancy
  24. Sexual Selection                 -- organisms choose mates
  25. Evolution of Rendering Algorithms-- organisms invent the renderer
  26. Living Materials                 -- shaders become tissues
  27. Ecosystem Niches                 -- habitats auto-discovered
  28. Evolution Replay                 -- record birth/death/split/extinction
  29. Genetic Archaeology              -- family/species/world tree
  30. Self-Modifying Evolution ★       -- evolution rewrites evolution (bounded)

All of these are CPU-side supervisors that feed into the genome + shader
pipeline already present in the renderer. The GPU stays under your control:
every generated shader is still produced by the existing shader factory, and
the self-modifying stage can only pick from a curated, validated set of
operators so invalid GLSL can never reach the GPU.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

# Curated parameter bounds for discretisation and normalisation.
# Keep this authoritative and in sync with the genome schema.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "warp": (0.0, 3.0),
    "glow": (0.0, 2.0),
    "rotation": (0.0, 6.2832),
    "noise_scale": (0.01, 5.0),
    "domain_warp": (0.0, 3.0),
    "turbulence": (0.0, 2.0),
    "vortex": (0.0, 3.0),
    "curl_strength": (0.0, 3.0),
    "feedback": (0.0, 1.0),
    "symmetry": (1, 12),
    "particle_density": (0.0, 1.0),
    "contrast": (0.0, 2.0),
    "flow_speed": (0.0, 2.0),
    "scale": (0.1, 3.0),
}

# Multi-objective fitness: True = higher is better, False = lower is better.
# This controls the direction in pareto_dominates.
OBJECTIVE_DIRECTIONS: dict[str, bool] = {
    "beauty": True,
    "movement": True,
    "stability": True,
    "gpu_cost": True,  # now inverted: higher = more efficient
    "novelty": True,
    "survival": True,
    "music_sync": True,
    "complexity": True,
    "symmetry": True,
    "glow": True,
}

OBJECTIVE_WEIGHTS = {
    "beauty": 1.0,
    "movement": 1.0,
    "stability": 1.0,
    "gpu_cost": 1.0,
    "novelty": 1.0,
    "survival": 1.0,
    "music_sync": 1.0,
    "complexity": 1.0,
    "symmetry": 1.0,
    "glow": 1.0,
}

# Centralised magic constants for easy tuning.
# Mutation / GRN
DEFAULT_MUTATION_STRENGTH = 0.15
DEFAULT_CONTINUOUS_MUTATION_RATE = 0.004
MAX_CONTINUOUS_MUTATION_RATE = 0.05
GRN_EDGE_DENSITY = 0.25
GRN_MUTATE_ADD_REMOVE_PROB = 0.2
GRN_MUTATE_REMOVE_PROB = 0.15
GRN_BEAT_TRIGGER_MUTATE_PROB = 0.1
GRN_KIND_CHANGE_PROB = 0.03  # 0.15 * 0.2
GRN_MULTIPLY_SCALE = 0.15
GRN_ADD_SCALE = 0.05
GRN_REDUCE_SCALE = 0.08
GRN_ACTIVATE_THRESHOLD = 0.6
GRN_ACTIVATE_SCALE = 0.2

# Continuous mutation
CONTINUOUS_MUTATE_KEYS_PER_FRAME = 3
CONTINUOUS_BEAT_BOOST = 0.5
CONTINUOUS_MUTATION_SCALE = 0.5

# Emotion
EMOTION_EXCITEMENT_DECAY = 0.9
EMOTION_EXCITEMENT_BOOST = 0.1
EMOTION_STRESS_DECAY = 0.95
EMOTION_STRESS_BOOST = 0.05
EMOTION_FEAR_STRESS_FACTOR = 0.8
EMOTION_FEAR_CONFIDENCE_FACTOR = 0.3
EMOTION_CONFIDENCE_DECAY = 0.98
EMOTION_CONFIDENCE_BOOST = 0.02
EMOTION_CURIOSITY_DECAY = 0.95
EMOTION_CURIOSITY_BOOST = 0.1
EMOTION_FATIGUE_INCREMENT = 0.002
EMOTION_FATIGUE_RELIEF = 0.01
EMOTION_HIGH_EXCITEMENT_THRESHOLD = 0.7

EMOTION_MUTATION_BASE = 0.5
EMOTION_MUTATION_CURIOSITY = 0.8
EMOTION_MUTATION_FATIGUE = 0.3
EMOTION_MOVE_SPEED_BASE = 0.6
EMOTION_MOVE_SPEED_EXCITEMENT = 0.6
EMOTION_MOVE_SPEED_FEAR = 0.4
EMOTION_EXPLORE_CURIOSITY = 0.7
EMOTION_EXPLORE_CONFIDENCE = 0.3
EMOTION_SHADER_CHANGE_EXCITEMENT = 0.5
EMOTION_SHADER_CHANGE_STRESS = 0.2

# Genome repair
REPAIR_BACKUP_PROB = 1.0  # multiplied by redundancy
REPAIR_FORBIDDEN_CENTRE_PULL = 1.0  # multiplied by repair_rate
REPAIR_RESTORE_PROB = 0.1  # multiplied by repair_rate

# Sexual selection
SEXUAL_CROSS_SPECIES_BASE = 0.3
SEXUAL_CROSS_SPECIES_MAX = 0.7

# Self-modifying evolution
MUTATION_OPERATORS = {
    "drift": 1.0,
    "leap": 2.5,
    "creep": 0.4,
    "oscillate": 1.0,
    "pulse": 1.6,
    "conservative": 0.6,
    "explosive": 3.0,
}

FITNESS_OBJECTIVES = (
    "beauty",
    "movement",
    "stability",
    "novelty",
    "survival",
    "music_sync",
    "complexity",
)

SHADER_NODES = (
    "neon_edge",
    "bloom",
    "reaction_diffusion",
    "polar_galaxy",
    "nonlinear_trails",
    "audio_topology",
    "depth_cloud",
    "cells",
    "kaleidoscope",
    "voronoi",
    "metaballs",
    "wave",
    "caustics",
)

# Predictive mutation bias clamp range
BIAS_CLAMP_MIN = 0.3
BIAS_CLAMP_MAX = 2.0

# Survival / repeat tracker
SURVIVAL_DISCRETISATION_BANDS = 10
REPEAT_DETECTOR_BANDS = 8
REPEAT_MIN_COUNT = 3

# Reaction scorer
REACTION_MIN_DELTA = 0.01
REACTION_SCORE_SCALE = 2.0

# Preference model
PREFERENCE_DECAY = 0.95
PREFERENCE_MIN_SIGNALS = 5
PREFERENCE_K_NEIGHBORS = 10
PREFERENCE_MAX_AGE_SECONDS = 3600.0
PREFERENCE_MAX_SIGNALS = 200
PREFERENCE_BIAS_PRESERVE = 0.6
PREFERENCE_BIAS_EXPLORE = 1.4
PREFERENCE_BIAS_SCALE = 0.8

# Novelty
NOVELTY_ARCHIVE_CAPACITY = 400
NOVELTY_K_NEIGHBORS = 15

# Replay / archive
REPLAY_CAPACITY = 5000
GENETIC_ARCHIVE_CAPACITY = 4000

# Maps
MAP_DECAY_TERRITORY = 0.985
MAP_DECAY_FOOD = 0.94
MAP_DECAY_PHEROMONE = 0.9
MAP_DECAY_DANGER = 0.92
MAP_DIFFUSE_TERRITORY = 0.05
MAP_DIFFUSE_FOOD = 0.12
MAP_DIFFUSE_PHEROMONE = 0.18
MAP_DIFFUSE_DANGER = 0.1
MAP_DEPOSIT_TERRITORY_BASE = 0.15
MAP_DEPOSIT_TERRITORY_ENERGY = 0.25
MAP_DEPOSIT_FOOD_FIELD = 0.6
MAP_DEPOSIT_FOOD_DECAY = 0.5
MAP_DEPOSIT_PHEROMONE_BASE = 0.2
MAP_DEPOSIT_PHEROMONE_EXCITE = 0.4
MAP_DEPOSIT_DANGER_ENERGY = 0.1
MAP_DEPOSIT_DANGER_FEAR = 0.2
MAP_DEPOSIT_DANGER_NEIGHBOUR = 0.06
MAP_DEPOSIT_DANGER_SIGNAL = 0.15
MAP_DEPOSIT_TERRITORY_COMPETITION = -0.02
MAP_RANGE_MIN = 0.05
MAP_RANGE_MAX = 0.6

from evo_performance import (  # noqa: E402
    BindlessTextureArray,
    FrustumCuller,
    GpuProfilerTimer,
    GpuTimerQuery,
    MultiDrawIndirect,
    PersistentMappedPBO,
    PersistentSSBO,
    ShaderBinaryCache,
    TextureAtlasPacker,
)

logger = __import__("logging").getLogger(__name__)

STAGE_NAMES = ("egg", "embryo", "juvenile", "adult", "elder")


# ===========================================================================
# 16. Gene Regulatory Networks (GRN)
# ===========================================================================
# Instead of genes acting independently, genes regulate each other. A gene is a
# node that, given the current value of the genes it reads, writes a new value
# to the genes it drives. The GRN is itself evolved, so the network topology
# (which gene activates which) is part of the genome.

GRN_REGULATORS = (
    "warp",
    "glow",
    "rotation",
    "noise_scale",
    "domain_warp",
    "turbulence",
    "vortex",
    "curl_strength",
    "feedback",
    "symmetry",
    "particle_density",
    "contrast",
)


@dataclass
class GeneEdge:
    """A single regulatory edge: source gene -> target gene."""

    src: str
    tgt: str
    weight: float = 1.0  # strength of the regulation
    kind: str = "multiply"  # "multiply" | "add" | "reduce" | "activate"

    def mutate(self, strength: float = DEFAULT_MUTATION_STRENGTH) -> None:
        if random.random() < strength:
            self.weight = max(
                -3.0, min(3.0, self.weight + random.uniform(-strength, strength) * 1.5)
            )
        if random.random() < strength * 0.2:
            self.kind = random.choice(("multiply", "add", "reduce", "activate"))

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "tgt": self.tgt, "weight": self.weight, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeneEdge:
        return cls(d["src"], d["tgt"], float(d.get("weight", 1.0)), d.get("kind", "multiply"))


class GeneRegulatoryNetwork:
    """Evolves gene-to-gene regulation so the phenotype behaves as a system."""

    def __init__(self, edges: list[GeneEdge] | None = None, density: float = GRN_EDGE_DENSITY):
        if edges is not None:
            self.edges = list(edges)
        else:
            self.edges = []
            genes = list(GRN_REGULATORS)
            for tgt in genes:
                for src in genes:
                    if src == tgt:
                        continue
                    if random.random() < density:
                        kind = random.choice(("multiply", "add", "reduce", "activate"))
                        self.edges.append(GeneEdge(src, tgt, random.uniform(-1.5, 1.5), kind))
        # Beat-gated activations (spec 16: Gene C activates Gene D only on beats)
        self.beat_triggers: dict[str, str] = {}
        for _ in range(random.randint(0, 2)):
            a = random.choice(GRN_REGULATORS)
            b = random.choice(GRN_REGULATORS)
            if a != b:
                self.beat_triggers[a] = b

    def apply(self, params: dict[str, Any], audio: dict[str, float]) -> None:
        """Run one regulatory pass in place over ``params``.

        Beat gating: edges that are explicitly beat-gated (src -> tgt in beat_triggers)
        only fire when the music has an active beat.
        """
        beat = float(audio.get("beat", 0.0))
        delta = {}
        for e in self.edges:
            if e.tgt not in params or e.src not in params:
                continue
            # Beat-gated edges: only fire when beat is active.
            # beat_triggers maps source -> target for beat-gated edges.
            if e.src in self.beat_triggers and self.beat_triggers[e.src] == e.tgt and beat < 0.5:
                continue
            src_val = float(params[e.src])
            cur = float(params[e.tgt])
            if e.kind == "multiply":
                delta[e.tgt] = (
                    delta.get(e.tgt, 0.0) + (src_val * e.weight - 1.0) * GRN_MULTIPLY_SCALE * cur
                )
            elif e.kind == "add":
                delta[e.tgt] = delta.get(e.tgt, 0.0) + src_val * e.weight * GRN_ADD_SCALE
            elif e.kind == "reduce":
                delta[e.tgt] = delta.get(e.tgt, 0.0) - abs(src_val) * e.weight * GRN_REDUCE_SCALE
            else:  # activate
                if src_val > GRN_ACTIVATE_THRESHOLD:
                    delta[e.tgt] = (
                        delta.get(e.tgt, 0.0) + (1.0 - cur) * e.weight * GRN_ACTIVATE_SCALE
                    )
        for k, v in delta.items():
            try:
                params[k] = float(params[k]) + v
            except (TypeError, ValueError):
                pass

    def mutate(self, strength: float = DEFAULT_MUTATION_STRENGTH) -> None:
        for e in self.edges:
            e.mutate(strength)
        # Occasionally add/remove an edge.
        if random.random() < GRN_MUTATE_ADD_REMOVE_PROB:
            genes = list(GRN_REGULATORS)
            a, b = random.sample(genes, 2)
            self.edges.append(GeneEdge(a, b, random.uniform(-1.5, 1.5)))
        if self.edges and random.random() < GRN_MUTATE_REMOVE_PROB:
            self.edges.pop(random.randrange(len(self.edges)))
        if self.beat_triggers and random.random() < GRN_BEAT_TRIGGER_MUTATE_PROB:
            self.beat_triggers.pop(next(iter(self.beat_triggers)))
        elif random.random() < GRN_BEAT_TRIGGER_MUTATE_PROB:
            a = random.choice(GRN_REGULATORS)
            b = random.choice(GRN_REGULATORS)
            if a != b:
                self.beat_triggers[a] = b

    def clone(self) -> GeneRegulatoryNetwork:
        g = GeneRegulatoryNetwork([copy.deepcopy(e) for e in self.edges])
        g.beat_triggers = dict(self.beat_triggers)
        return g

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "beat_triggers": dict(self.beat_triggers),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeneRegulatoryNetwork:
        g = cls([GeneEdge.from_dict(e) for e in d.get("edges", [])])
        g.beat_triggers = dict(d.get("beat_triggers", {}))
        return g


# ===========================================================================
# 17. Development Instead of Instant Rendering
# ===========================================================================
# Each organism has a life stage. The same genome renders with different shader
# parameters at each stage, so the organism visibly *matures* from egg to elder.

# Per-stage parameter multipliers (visual maturity curve).
STAGE_PARAMS: dict[str, dict[str, float]] = {
    "egg": {"scale": 0.35, "glow": 0.3, "contrast": 0.5, "turbulence": 0.4, "feedback": 0.6},
    "embryo": {"scale": 0.6, "glow": 0.5, "contrast": 0.7, "turbulence": 0.7, "feedback": 0.75},
    "juvenile": {"scale": 0.85, "glow": 0.8, "contrast": 0.9, "turbulence": 0.9, "feedback": 0.9},
    "adult": {"scale": 1.0, "glow": 1.0, "contrast": 1.0, "turbulence": 1.0, "feedback": 1.0},
    "elder": {"scale": 0.8, "glow": 1.2, "contrast": 1.1, "turbulence": 1.3, "feedback": 1.05},
}


def stage_for_age(age: float, life_span: float) -> str:
    """Return the life-stage name for a normalized age 0..life_span."""
    frac = max(0.0, min(1.0, age / max(life_span, 1e-6)))
    if frac < 0.1:
        return "egg"
    if frac < 0.3:
        return "embryo"
    if frac < 0.65:
        return "juvenile"
    if frac < 0.9:
        return "adult"
    return "elder"


def mature_params(params: dict[str, Any], stage: str) -> dict[str, Any]:
    """Return a copy of ``params`` adjusted for a life stage.

    The same genome therefore renders differently as the organism ages --
    the organism literally grows.
    """
    m = STAGE_PARAMS[stage]
    out = dict(params)
    if "noise_scale" in out:
        out["noise_scale"] = float(out["noise_scale"]) * m["scale"]
    if "glow" in out:
        out["glow"] = float(out["glow"]) * m["glow"]
    if "contrast" in out:
        out["contrast"] = float(out["contrast"]) * m["contrast"]
    if "turbulence" in out:
        out["turbulence"] = float(out["turbulence"]) * m["turbulence"]
    if "feedback" in out:
        out["feedback"] = max(0.0, min(1.0, float(out["feedback"]) * m["feedback"]))
    return out


# ===========================================================================
# 18. Continuous Mutation
# ===========================================================================
# Instead of mutating on a 10-second boundary, organisms take a tiny mutation
# every frame. The drift is small so the look stays coherent, but over time the
# organism wanders through phenotype space -- evolution becomes continuous.


class ContinuousMutator:
    """Applies a tiny per-frame mutation to a genome's numeric params."""

    def __init__(self, per_frame: float = DEFAULT_CONTINUOUS_MUTATION_RATE):
        self.per_frame = max(0.0, min(MAX_CONTINUOUS_MUTATION_RATE, per_frame))
        self._param_bias: dict[str, float] = {}

    def step(
        self, params: dict[str, Any], audio: dict[str, float], dt: float = 1.0 / 60.0
    ) -> None:
        amt = self.per_frame * dt * 60.0
        beat = float(audio.get("beat", 0.0))
        # Beats give evolution a small kick so it pulses with the music.
        amt *= 1.0 + beat * CONTINUOUS_BEAT_BOOST
        keys = [k for k in params if isinstance(params[k], (int, float))]
        if not keys:
            return
        # Mutate a few keys per frame (up to 3) so per-parameter bias is actually used.
        n_mutate = min(CONTINUOUS_MUTATE_KEYS_PER_FRAME, len(keys))
        for k in random.sample(keys, n_mutate):
            v = float(params[k])
            bias = self._param_bias.get(k, 1.0)
            key_amt = amt * bias
            params[k] = v + random.uniform(-key_amt, key_amt) * (
                CONTINUOUS_MUTATION_SCALE + abs(v)
            )

    def to_dict(self) -> dict[str, Any]:
        return {"per_frame": self.per_frame, "_param_bias": dict(self._param_bias)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContinuousMutator:
        m = cls(d.get("per_frame", DEFAULT_CONTINUOUS_MUTATION_RATE))
        m._param_bias = dict(d.get("_param_bias", {}))
        return m

    def clone(self) -> ContinuousMutator:
        m = ContinuousMutator(self.per_frame)
        m._param_bias = dict(self._param_bias)
        return m


# ===========================================================================
# 19. Multi-objective Evolution (Pareto)
# ===========================================================================
# Fitness is no longer a single scalar. An organism is scored on many axes and
# the population is ranked by Pareto dominance: one organism is "better" only
# if it is at least as good on every axis AND strictly better on one. The
# result is a frontier of trade-off organisms; there is no single "best".

OBJECTIVE_WEIGHTS = {
    "beauty": 1.0,
    "movement": 1.0,
    "stability": 1.0,
    "gpu_cost": 1.0,
    "novelty": 1.0,
    "survival": 1.0,
    "music_sync": 1.0,
    "complexity": 1.0,
}


def evaluate_objectives(
    genome: Any,
    fitness: float,
    audio: dict[str, float],
    gpu_cost: float = 0.0,
    novelty: float = 0.0,
    complexity: float = 0.0,
) -> dict[str, float]:
    """Compute the multi-objective score vector for one organism."""
    params = getattr(genome, "params", None) or getattr(genome, "to_dict", dict)()
    bass = float(audio.get("bass", 0.0))
    treb = float(audio.get("treb", 0.0))
    beat = float(audio.get("beat", 0.0))
    sym = float(params.get("symmetry", 1))
    glow = float(params.get("glow", 0.5))

    return {
        "beauty": float(fitness),
        "movement": min(
            1.0,
            abs(float(params.get("vortex", 0.0))) * 0.3
            + float(params.get("flow_speed", 1.0)) * 0.2,
        ),
        "stability": min(1.0, 1.0 - gpu_cost * 0.5 + 0.2),
        "gpu_cost": min(1.0, 1.0 - gpu_cost),  # inverted: higher = more efficient (lower cost)
        "novelty": min(1.0, novelty),
        "survival": min(1.0, float(getattr(genome, "energy", 1.0)) / 3.0),
        "music_sync": min(1.0, (bass + treb + beat) / 3.0),
        "complexity": min(1.0, complexity),
        # Symmetry and glow are cheap stand-ins for an aesthetic prior.
        "symmetry": min(1.0, sym / 12.0),
        "glow": min(1.0, glow / 2.0),
    }


def pareto_dominates(
    a: dict[str, float],
    b: dict[str, float],
    weights: dict[str, float] | None = None,
    directions: dict[str, bool] | None = None,
) -> bool:
    """Return True if ``a`` dominates ``b`` (>= on all, > on at least one).

    Supports per-objective directions: True = higher is better, False = lower is better.
    """
    w = weights or OBJECTIVE_WEIGHTS
    d = directions or OBJECTIVE_DIRECTIONS
    at_least_one = False
    for key, weight in w.items():
        if key not in a or key not in b:
            continue
        va = a[key] * weight
        vb = b[key] * weight
        if d.get(key, True):  # default: higher is better
            if va > vb:
                at_least_one = True
            elif va < vb:
                return False
        else:  # lower is better
            if va < vb:
                at_least_one = True
            elif va > vb:
                return False
    return at_least_one


def pareto_rank(
    objectives: list[dict[str, float]],
    weights: dict[str, float] | None = None,
    directions: dict[str, bool] | None = None,
) -> list[int]:
    """Assign each organism a Pareto front index (0 = best front).

    Lower rank = closer to the Pareto-optimal frontier.
    Uses the standard fast non-dominated sorting algorithm (NSGA-II style).
    """
    n = len(objectives)
    if n == 0:
        return []

    ranks = [0] * n
    dominated_count = [0] * n
    dominates_list: list[list[int]] = [[] for _ in range(n)]

    # Build dominance relationships
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pareto_dominates(objectives[i], objectives[j], weights, directions):
                dominates_list[i].append(j)
            elif pareto_dominates(objectives[j], objectives[i], weights, directions):
                dominated_count[i] += 1

    # Find first front (dominated_count == 0)
    current_front = [i for i in range(n) if dominated_count[i] == 0]
    front_rank = 0

    while current_front:
        next_front = []
        for i in current_front:
            ranks[i] = front_rank
            # Decrement dominated_count for all solutions dominated by i
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        front_rank += 1

    return ranks


# ===========================================================================
# 20. Novelty Search ★
# ===========================================================================
# Reward *difference* from every previous organism rather than a fixed target.
# Fitness becomes the distance from the archive of past phenotypes, so
# evolution keeps inventing new visual species instead of converging.


class NoveltyArchive:
    """Stores past phenotype signatures and measures behavioural novelty."""

    def __init__(self, capacity: int = NOVELTY_ARCHIVE_CAPACITY, k: int = NOVELTY_K_NEIGHBORS):
        self.capacity = max(16, int(capacity))
        self.k = max(1, int(k))
        self._archive: list[dict[str, float]] = []
        self._add_counter = 0

    def _signature(self, genome: Any) -> dict[str, float]:
        params = getattr(genome, "params", None) or getattr(genome, "to_dict", dict)()
        sig = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
        return sig

    def distance(self, a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        total = 0.0
        for k in keys:
            va = a.get(k, 0.0)
            vb = b.get(k, 0.0)
            lo, hi = PARAM_BOUNDS.get(k, (0.0, 1.0))
            if hi > lo:
                total += abs(va - vb) / (hi - lo)
            else:
                total += abs(va - vb)
        return total / len(keys)

    def novelty(self, genome: Any) -> float:
        sig = self._signature(genome)
        if not self._archive:
            return 1.0
        dists = sorted(self.distance(sig, past) for past in self._archive)
        nearest = dists[: self.k]
        return sum(nearest) / max(len(nearest), 1)

    def add(self, genome: Any) -> None:
        self._archive.append(self._signature(genome))
        self._add_counter += 1
        if len(self._archive) > self.capacity:
            # Bounded: drop oldest so memory cannot grow without limit.
            self._archive.pop(0)

    def to_dict(self) -> dict[str, Any]:
        return {"archive": self._archive[-self.capacity :], "capacity": self.capacity, "k": self.k}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NoveltyArchive:
        a = cls(capacity=d.get("capacity", 400), k=d.get("k", 15))
        a._archive = list(d.get("archive", []))
        return a


# ===========================================================================
# 21. Memory-Based Evolution
# ===========================================================================
# Every organism stores experiences (audio context -> survived or not). Later,
# when a similar audio context arrives, it can *prepare* before the beat hits.
# This produces anticipation: the organism acts on predicted events.


@dataclass
class Experience:
    bass: float
    treb: float
    beat: float
    survived: bool
    reward: float = 0.0


class OrganismMemory:
    """Per-organism associative memory of (audio context -> outcome)."""

    def __init__(self, capacity: int = 64):
        self.capacity = max(8, int(capacity))
        self._mem: list[Experience] = []

    def record(self, audio: dict[str, float], survived: bool, reward: float = 0.0) -> None:
        self._mem.append(
            Experience(
                float(audio.get("bass", 0.0)),
                float(audio.get("treb", 0.0)),
                float(audio.get("beat", 0.0)),
                survived,
                reward,
            )
        )
        if len(self._mem) > self.capacity:
            self._mem.pop(0)

    def anticipation(self, audio: dict[str, float]) -> float:
        """Return 0..1 confidence that a *high-bass beat* will arrive.

        Built from how often high-bass contexts were followed by survival in
        the past. An organism with good memory "prepares" by returning high
        confidence just before the beat.
        """
        bass = float(audio.get("bass", 0.0))
        beat = float(audio.get("beat", 0.0))
        recent = [m for m in self._mem if m.bass > 0.6]
        if not recent:
            return beat
        # If high-bass historically meant survival, get ready early.
        good = sum(1.0 for m in recent if m.survived)
        conf = good / len(recent)
        # Anticipate: ramp up as bass rises, peak right before beat.
        return max(beat, conf * bass)

    def clone(self) -> OrganismMemory:
        m = OrganismMemory(self.capacity)
        m._mem = [copy.copy(e) for e in self._mem]
        return m

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "mem": [vars(e) for e in self._mem],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OrganismMemory:
        m = cls(d.get("capacity", 64))
        for e in d.get("mem", []):
            m._mem.append(
                Experience(
                    e["bass"],
                    e["treb"],
                    e["beat"],
                    bool(e["survived"]),
                    float(e.get("reward", 0.0)),
                )
            )
        return m


# ===========================================================================
# 22. Emotional State
# ===========================================================================
# Organisms own internal variables that modify behaviour. Genetically identical
# organisms behave differently because of experience.

EMOTIONS = ("fear", "curiosity", "confidence", "stress", "fatigue", "excitement")


@dataclass
class EmotionalState:
    fear: float = 0.0
    curiosity: float = 0.0
    confidence: float = 0.0
    stress: float = 0.0
    fatigue: float = 0.0
    excitement: float = 0.0

    def update(self, audio: dict[str, float], energy: float, novelty: float = 0.0) -> None:
        bass = float(audio.get("bass", 0.0))
        beat = float(audio.get("beat", 0.0))
        self.excitement = min(
            1.0,
            self.excitement * EMOTION_EXCITEMENT_DECAY + (beat + bass) * EMOTION_EXCITEMENT_BOOST,
        )
        self.stress = min(
            1.0,
            max(
                0.0,
                self.stress * EMOTION_STRESS_DECAY + (1.0 - energy / 3.0) * EMOTION_STRESS_BOOST,
            ),
        )
        self.fear = max(
            0.0,
            min(
                1.0,
                self.stress * EMOTION_FEAR_STRESS_FACTOR
                - self.confidence * EMOTION_FEAR_CONFIDENCE_FACTOR,
            ),
        )
        self.confidence = max(
            0.0,
            min(
                1.0,
                self.confidence * EMOTION_CONFIDENCE_DECAY
                + energy / 3.0 * EMOTION_CONFIDENCE_BOOST,
            ),
        )
        self.curiosity = max(
            0.0,
            min(1.0, self.curiosity * EMOTION_CURIOSITY_DECAY + novelty * EMOTION_CURIOSITY_BOOST),
        )
        self.fatigue = min(1.0, self.fatigue + EMOTION_FATIGUE_INCREMENT)
        if self.excitement > EMOTION_HIGH_EXCITEMENT_THRESHOLD:
            self.fatigue = max(0.0, self.fatigue - EMOTION_FATIGUE_RELIEF)

    def modifier(self) -> dict[str, float]:
        """Behaviour modifiers derived from the emotional vector."""
        return {
            "mutation_rate": EMOTION_MUTATION_BASE
            + self.curiosity * EMOTION_MUTATION_CURIOSITY
            - self.fatigue * EMOTION_MUTATION_FATIGUE,
            "move_speed": EMOTION_MOVE_SPEED_BASE
            + self.excitement * EMOTION_MOVE_SPEED_EXCITEMENT
            - self.fear * EMOTION_MOVE_SPEED_FEAR,
            "explore": self.curiosity * EMOTION_EXPLORE_CURIOSITY
            + self.confidence * EMOTION_EXPLORE_CONFIDENCE,
            "shader_change": self.excitement * EMOTION_SHADER_CHANGE_EXCITEMENT
            + self.stress * EMOTION_SHADER_CHANGE_STRESS,
            "rest": self.fatigue,
        }

    def clone(self) -> EmotionalState:
        return copy.copy(self)

    def to_dict(self) -> dict[str, float]:
        return {
            "fear": self.fear,
            "curiosity": self.curiosity,
            "confidence": self.confidence,
            "stress": self.stress,
            "fatigue": self.fatigue,
            "excitement": self.excitement,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> EmotionalState:
        return cls(
            d.get("fear", 0.0),
            d.get("curiosity", 0.0),
            d.get("confidence", 0.0),
            d.get("stress", 0.0),
            d.get("fatigue", 0.0),
            d.get("excitement", 0.0),
        )


# ===========================================================================
# 23. Genome Repair
# ===========================================================================
# Real DNA repairs itself. Organisms evolve a repair apparatus: a repair rate
# (how aggressively drift is corrected), a mutation filter (forbidden regions),
# redundancy (backed-up genes), and a backup genome. Some lineages become
# extremely stable; others become mutation machines.


class GenomeRepair:
    """Evolved self-repair mechanism for a genome."""

    def __init__(
        self, repair_rate: float = 0.1, redundancy: float = 0.5, mutation_filter: set | None = None
    ):
        self.repair_rate = max(0.0, min(1.0, repair_rate))
        self.redundancy = max(0.0, min(1.0, redundancy))
        self.mutation_filter = set(mutation_filter or set())
        self._backup: dict[str, Any] | None = None

    def backup(self, params: dict[str, Any]) -> None:
        if random.random() < self.redundancy * REPAIR_BACKUP_PROB:
            self._backup = dict(params)

    def repair(self, params: dict[str, Any], bounds: dict[str, tuple[float, float]]) -> None:
        """Correct drift: pull values toward a healthy centre and, when a gene
        is in the backup, occasionally restore it."""
        if self._backup is None:
            return
        for k, (lo, hi) in bounds.items():
            if k not in params:
                continue
            if k in self.mutation_filter:
                # Forbidden region: hard-reset back toward centre.
                centre = (lo + hi) / 2.0
                cur = float(params[k])
                params[k] = cur + (centre - cur) * self.repair_rate * REPAIR_FORBIDDEN_CENTRE_PULL
                continue
            if random.random() < self.repair_rate * REPAIR_RESTORE_PROB and k in self._backup:
                params[k] = float(self._backup[k])

    def mutate(self, strength: float = DEFAULT_MUTATION_STRENGTH) -> None:
        if random.random() < strength:
            self.repair_rate = max(
                0.0, min(1.0, self.repair_rate + random.uniform(-strength, strength))
            )
        if random.random() < strength:
            self.redundancy = max(
                0.0, min(1.0, self.redundancy + random.uniform(-strength, strength))
            )
        if random.random() < strength * 0.3:
            gene = random.choice(list(GRN_REGULATORS))
            if gene in self.mutation_filter:
                self.mutation_filter.discard(gene)
            else:
                self.mutation_filter.add(gene)

    def clone(self) -> GenomeRepair:
        g = GenomeRepair(self.repair_rate, self.redundancy, set(self.mutation_filter))
        g._backup = dict(self._backup) if self._backup else None
        return g

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_rate": self.repair_rate,
            "redundancy": self.redundancy,
            "mutation_filter": list(self.mutation_filter),
            "_backup": dict(self._backup) if self._backup else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenomeRepair:
        g = cls(
            d.get("repair_rate", 0.1), d.get("redundancy", 0.5), set(d.get("mutation_filter", []))
        )
        g._backup = dict(d.get("_backup", {})) if d.get("_backup") else None
        return g


# ===========================================================================
# 24. Sexual Selection
# ===========================================================================
# Instead of reproducing with random survivors, organisms *choose* mates. The
# criteria are evolved traits (brightness, movement, symmetry, novelty,
# species, territory, energy) -- this produces runaway evolution (peacock
# feathers).

MATING_TRAITS = ("brightness", "movement", "symmetry", "novelty", "species", "territory", "energy")


class SexualSelector:
    """Each organism carries a preference vector over mating traits."""

    def __init__(self, prefs: dict[str, float] | None = None):
        if prefs is not None:
            self.prefs = dict(prefs)
        else:
            self.prefs = {t: random.uniform(0.0, 1.0) for t in MATING_TRAITS}

    def attractiveness(self, org: Any, trait_cache: dict[str, float] | None = None) -> float:
        """0..1 score of how attractive ``org`` is to this selector's owner."""
        if trait_cache is not None:
            tv = trait_cache
        else:
            tv = organism_traits(org)
        score = 0.0
        wsum = 0.0
        for trait, w in self.prefs.items():
            if trait == "species":
                continue
            score += w * tv.get(trait, 0.0)
            wsum += w
        return score / max(wsum, 1e-6)

    def mate_score(
        self, candidate: Any, owner: Any, trait_cache: dict[str, float] | None = None
    ) -> float:
        """Combine trait attractiveness with species preference."""
        tv = trait_cache if trait_cache is not None else organism_traits(candidate)
        base = self.attractiveness(candidate, tv)
        # Prefer same species unless the selector's owner is a cross-species
        # experimenter.
        if getattr(candidate, "species", -1) != getattr(owner, "species", -2):
            cross = self.prefs.get("species", 0.0)
            base *= SEXUAL_CROSS_SPECIES_BASE + cross * SEXUAL_CROSS_SPECIES_MAX
        return base

    def mutate(self, strength: float = 0.1) -> None:
        for t in self.prefs:
            if random.random() < strength:
                self.prefs[t] = max(
                    0.0, min(1.0, self.prefs[t] + random.uniform(-strength, strength))
                )

    def clone(self) -> SexualSelector:
        return SexualSelector(dict(self.prefs))

    def to_dict(self) -> dict[str, Any]:
        return {"prefs": dict(self.prefs)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SexualSelector:
        return cls(d.get("prefs"))


def organism_traits(org: Any) -> dict[str, float]:
    """Derive the 0..1 mating-trait vector for an organism."""
    params = getattr(org, "params", None) or getattr(org, "to_dict", dict)()
    return {
        "brightness": min(1.0, float(params.get("glow", 0.0)) / 2.0),
        "movement": min(
            1.0,
            abs(float(params.get("vortex", 0.0))) * 0.3
            + float(params.get("flow_speed", 1.0)) * 0.2,
        ),
        "symmetry": min(1.0, float(params.get("symmetry", 1)) / 12.0),
        "novelty": min(1.0, float(getattr(org, "_novelty", 0.0))),
        "species": 1.0,
        "territory": min(1.0, float(getattr(org, "radius", 0.1)) * 6.0),
        "energy": min(1.0, float(getattr(org, "energy", 1.0)) / 3.0),
    }


def choose_mate(owner: Any, selector: SexualSelector, candidates: list[Any]) -> Any | None:
    """Return the best-matching mate among ``candidates`` for ``owner``."""
    best = None
    best_score = 0.0
    for c in candidates:
        if c is owner or not getattr(c, "alive", True):
            continue
        s = selector.mate_score(c, owner)
        if s > best_score:
            best_score = s
            best = c
    return best


# ===========================================================================
# 25. Evolution of Rendering Algorithms ★
# ===========================================================================
# Evolution is allowed to *invent* the rendering method, not just tune noise /
# palette / feedback. An organism carries a renderer-genome choosing among a
# curated, validated set of algorithms. Invalid operators are impossible by
# construction (this is the GPU safety boundary for the self-modifying stage).

RENDER_ALGORITHMS = (
    "fbm",
    "raymarch",
    "distance_field",
    "reaction_diffusion",
    "particle_fluid",
    "voronoi",
    "metaballs",
    "fractal",
    "wave_equation",
    "cellular_automata",
    "signed_distance_field",
)


class RendererGenome:
    """The organism's choice of rendering algorithm (evolved, bounded)."""

    def __init__(self, algorithm: str = "fbm", blend: float = 0.5):
        self.algorithm = algorithm if algorithm in RENDER_ALGORITHMS else "fbm"
        self.blend = max(0.0, min(1.0, blend))

    @staticmethod
    def random() -> RendererGenome:
        return RendererGenome(random.choice(RENDER_ALGORITHMS), random.uniform(0.0, 1.0))

    def mutate(self, strength: float = 0.15) -> None:
        if random.random() < strength:
            self.algorithm = random.choice(RENDER_ALGORITHMS)
        if random.random() < strength:
            self.blend = max(0.0, min(1.0, self.blend + random.uniform(-strength, strength)))

    def crossover(self, other: RendererGenome) -> RendererGenome:
        g = self.algorithm if random.random() < 0.5 else other.algorithm
        b = (self.blend + other.blend) / 2.0
        return RendererGenome(g, b)

    def clone(self) -> RendererGenome:
        return RendererGenome(self.algorithm, self.blend)

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, "blend": self.blend}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RendererGenome:
        return cls(d.get("algorithm", "fbm"), float(d.get("blend", 0.5)))


# ===========================================================================
# 26. Living Materials
# ===========================================================================
# Shaders become tissues. The material layer evolves separately from geometry
# so an organism can keep its shape but grow a new skin.

MATERIAL_TISSUES = (
    "skin",
    "shell",
    "crystal",
    "fur",
    "fire",
    "liquid",
    "plasma",
    "glass",
    "metal",
    "smoke",
)


class LivingMaterial:
    """Evolvable material/tissue assigned to an organism's surface."""

    def __init__(
        self,
        tissue: str = "skin",
        roughness: float = 0.5,
        emission: float = 0.3,
        thickness: float = 0.5,
    ):
        self.tissue = tissue if tissue in MATERIAL_TISSUES else "skin"
        self.roughness = max(0.0, min(1.0, roughness))
        self.emission = max(0.0, min(1.0, emission))
        self.thickness = max(0.0, min(1.0, thickness))

    @staticmethod
    def random() -> LivingMaterial:
        return LivingMaterial(
            random.choice(MATERIAL_TISSUES), random.random(), random.random(), random.random()
        )

    def mutate(self, strength: float = 0.15) -> None:
        if random.random() < strength:
            self.tissue = random.choice(MATERIAL_TISSUES)
        for attr in ("roughness", "emission", "thickness"):
            if random.random() < strength:
                v = getattr(self, attr) + random.uniform(-strength, strength)
                setattr(self, attr, max(0.0, min(1.0, v)))

    def clone(self) -> LivingMaterial:
        return LivingMaterial(self.tissue, self.roughness, self.emission, self.thickness)

    def shader_modifier(self) -> dict[str, float]:
        """Convert the material into shader-uniform multipliers."""
        base = {
            "skin": (0.9, 0.1, 0.8),
            "shell": (0.7, 0.05, 1.0),
            "crystal": (0.2, 0.4, 0.9),
            "fur": (1.1, 0.0, 0.7),
            "fire": (1.3, 0.9, 0.4),
            "liquid": (0.8, 0.2, 0.9),
            "plasma": (1.2, 0.8, 0.6),
            "glass": (0.4, 0.1, 1.0),
            "metal": (0.5, 0.05, 1.1),
            "smoke": (0.6, 0.3, 0.7),
        }.get(self.tissue, (1.0, 0.1, 0.8))
        glow_m, _emission_m, contrast_m = base
        return {
            "glow": glow_m * self.emission,
            "contrast": contrast_m,
            "roughness": self.roughness,
            "thickness": self.thickness,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "tissue": self.tissue,
            "roughness": self.roughness,
            "emission": self.emission,
            "thickness": self.thickness,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LivingMaterial:
        return cls(
            d.get("tissue", "skin"),
            d.get("roughness", 0.5),
            d.get("emission", 0.3),
            d.get("thickness", 0.5),
        )


# ===========================================================================
# 27. Ecosystem Niches
# ===========================================================================
# Instead of random habitats, the world auto-discovers niches from the music:
# deep field, bright field, chaotic region, calm region, music hotspots,
# resource valleys. Species specialise to a niche and compete to occupy it.

NICHE_TYPES = (
    "deep_field",
    "bright_field",
    "chaotic_region",
    "calm_region",
    "music_hotspot",
    "resource_valley",
)


@dataclass
class Niche:
    ntype: str
    cx: float
    cy: float
    radius: float
    affinity: float = 0.5

    def in_range(self, x: float, y: float) -> bool:
        return math.hypot(x - self.cx, y - self.cy) <= self.radius

    def suitability(self, audio: dict[str, float], intensity: float) -> float:
        """How well this niche matches the current music (0..1)."""
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        beat = float(audio.get("beat", 0.0))
        mid = float(audio.get("mid", 0.0))
        if self.ntype == "deep_field":
            return 1.0 - min(1.0, (bass + beat) / 2.0)
        if self.ntype == "bright_field":
            return min(1.0, treb + mid * 0.5)
        if self.ntype == "chaotic_region":
            return min(1.0, beat * 0.7 + abs(bass - treb) * 0.5)
        if self.ntype == "calm_region":
            return 1.0 - min(1.0, beat + abs(bass - treb))
        if self.ntype == "music_hotspot":
            return min(1.0, (bass + treb + beat) / 2.0)
        if self.ntype == "resource_valley":
            return min(1.0, intensity)
        return 0.5


class NicheMap:
    """Discovers and maintains ecosystem niches over time."""

    def __init__(self, target: int = 4):
        self.target = max(2, int(target))
        self.niches: list[Niche] = []
        self._spawn_timer = 0.0
        for _ in range(self.target):
            self.niches.append(self._random_niche())

    def _random_niche(self) -> Niche:
        ntype = random.choice(NICHE_TYPES)
        return Niche(ntype, random.random(), random.random(), random.uniform(0.2, 0.45))

    def step(self, audio: dict[str, float], dt: float, intensity: float) -> None:
        self._spawn_timer -= dt
        # Drift niche centres slowly with the music current.
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        for n in self.niches:
            n.cx = max(0.05, min(0.95, n.cx + (treb - bass) * 0.02 * dt))
            n.cy = max(0.05, min(0.95, n.cy + (bass - treb) * 0.02 * dt))
            n.affinity = 0.8 * n.affinity + 0.2 * n.suitability(audio, intensity)
        if self._spawn_timer <= 0.0 and len(self.niches) < self.target + 2:
            self._spawn_timer = random.uniform(15.0, 40.0)
            self.niches.append(self._random_niche())
        # Retire niches that are persistently unsuitable.
        self.niches = [n for n in self.niches if n.affinity > 0.1 or random.random() < 0.99]
        if len(self.niches) < self.target:
            self.niches.append(self._random_niche())

    def best_for(
        self, x: float, y: float, audio: dict[str, float], intensity: float
    ) -> Niche | None:
        best = None
        best_fit = -1.0
        for n in self.niches:
            if not n.in_range(x, y):
                continue
            fit = n.suitability(audio, intensity)
            if fit > best_fit:
                best_fit = fit
                best = n
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "niches": [vars(n) for n in self.niches],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NicheMap:
        m = cls(d.get("target", 4))
        m.niches = [
            Niche(n["ntype"], n["cx"], n["cy"], n["radius"], n.get("affinity", 0.5))
            for n in d.get("niches", [])
        ]
        return m


# ===========================================================================
# 28. Evolution Replay
# ===========================================================================
# Record mutation / birth / death / species-split / extinction events so the
# whole history can later be replayed (e.g. 100 hours of evolution in 5
# minutes). The recorder is bounded: only a rolling window is kept in memory,
# the durable copy being the fossil record / lineage files.

REPLAY_EVENT_TYPES = ("mutation", "birth", "death", "species_split", "extinction")


class EvolutionRecorder:
    """Append-only event log for replaying evolution."""

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.capacity = max(256, int(capacity))
        self.events: list[dict[str, Any]] = []

    def record(
        self, etype: str, lineage: str, data: dict[str, Any] | None = None, t: float | None = None
    ) -> None:
        if etype not in REPLAY_EVENT_TYPES:
            etype = "mutation"
        ev = {
            "t": t if t is not None else time.time(),
            "type": etype,
            "lineage": lineage,
            "data": data or {},
        }
        self.events.append(ev)
        if len(self.events) > self.capacity:
            # Keep a downsampled history: drop oldest but keep anchors so the
            # replay still spans the whole run.
            self.events.pop(0)

    def segment(self, start_idx: int = 0, count: int = 100) -> list[dict[str, Any]]:
        return self.events[start_idx : start_idx + count]

    def to_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "events": self.events[-self.capacity :]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionRecorder:
        r = cls(d.get("capacity", REPLAY_CAPACITY))
        r.events = list(d.get("events", []))
        return r


# ===========================================================================
# 29. Genetic Archaeology
# ===========================================================================
# Keep every successful genome and build a family / species / world tree.
# Clicking any ancestor recreates it instantly (handled by the renderer UI).


@dataclass
class TreeNode:
    lineage: str
    parent: str | None
    genome: dict[str, Any]
    fitness: float
    generation: int
    species: int = 0
    born_at: float = 0.0


class GeneticArchive:
    """Persistent family/species tree across the whole run."""

    def __init__(self, capacity: int = GENETIC_ARCHIVE_CAPACITY):
        self.capacity = max(256, int(capacity))
        self.nodes: dict[str, TreeNode] = {}
        self.children: dict[str, list[str]] = {}
        self.species_roots: dict[int, str] = {}

    def add(self, node: TreeNode) -> None:
        self.nodes[node.lineage] = node
        if node.parent and node.parent in self.nodes:
            self.children.setdefault(node.parent, []).append(node.lineage)
        if node.species not in self.species_roots:
            self.species_roots[node.species] = node.lineage
        if len(self.nodes) > self.capacity:
            # Bound in-memory tree; oldest redundant leaves are dropped.
            oldest = min(self.nodes, key=lambda k: self.nodes[k].born_at)
            self.nodes.pop(oldest, None)
            self.children.pop(oldest, None)

    def ancestors(self, lineage: str) -> list[str]:
        out = []
        cur = self.nodes.get(lineage)
        seen = 0
        while cur and cur.parent and seen < self.capacity:
            out.append(cur.parent)
            cur = self.nodes.get(cur.parent)
            seen += 1
        return out

    def species_tree(self) -> dict[int, list[str]]:
        tree: dict[int, list[str]] = {}
        for n in self.nodes.values():
            tree.setdefault(n.species, []).append(n.lineage)
        return tree

    def reconstruct(self, lineage: str) -> dict[str, Any] | None:
        n = self.nodes.get(lineage)
        return dict(n.genome) if n else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "nodes": {k: vars(v) for k, v in self.nodes.items()},
            "species_roots": self.species_roots,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeneticArchive:
        a = cls(d.get("capacity", 4000))
        for k, v in d.get("nodes", {}).items():
            a.nodes[k] = TreeNode(
                v["lineage"],
                v.get("parent"),
                v.get("genome", {}),
                float(v.get("fitness", 0.0)),
                int(v.get("generation", 0)),
                int(v.get("species", 0)),
                float(v.get("born_at", 0.0)),
            )
        a.species_roots = {int(k): v for k, v in d.get("species_roots", {}).items()}
        for n in a.nodes.values():
            if n.parent in a.nodes:
                a.children.setdefault(n.parent, []).append(n.lineage)
        return a


# ===========================================================================
# 30. Self-Modifying Evolution ★ (the final stage)
# ===========================================================================
# Evolution writes new evolution code. Instead of a fixed ``mutation()``, the
# population evolves *how mutation works*. Instead of a fixed ``fitness()``,
# it evolves *what fitness means*. And instead of a fixed shader graph, it
# evolves *which shader nodes* are available. Crucially, every operator the
# self-modifier can choose from is curated and validated, so invalid GLSL or
# unsafe GPU behaviour can never be produced -- you keep the safety constraints
# (resource limits, valid shader generation) while the system explores its own
# mechanics.

# Curated, safe mutation operators. Each returns a delta-multiplier on the
# base mutation step. No operator can exceed bounded ranges.
MUTATION_OPERATORS = {
    "drift": 1.0,  # baseline gaussian drift
    "leap": 2.5,  # occasional large jumps
    "creep": 0.4,  # very slow refinement
    "oscillate": 1.0,  # oscillate a gene over time
    "pulse": 1.6,  # mutate harder on beats
    "conservative": 0.6,  # protect good traits (less mutation)
    "explosive": 3.0,  # high exploration (bounded)
}

# Curated fitness objectives the self-modifier may weight (subset of 19).
FITNESS_OBJECTIVES = (
    "beauty",
    "movement",
    "stability",
    "novelty",
    "survival",
    "music_sync",
    "complexity",
)

# Curated shader-node families the self-modifier may enable. These map to
# existing, validated shader operators -- it can toggle/weight them, never
# invent raw GLSL.
SHADER_NODES = (
    "neon_edge",
    "bloom",
    "reaction_diffusion",
    "polar_galaxy",
    "nonlinear_trails",
    "audio_topology",
    "depth_cloud",
    "cells",
    "kaleidoscope",
    "voronoi",
    "metaballs",
    "wave",
    "caustics",
)


@dataclass
class EvolutionPolicy:
    """The self-modifiable evolutionary machinery.

    Everything here is chosen from fixed, bounded sets so the system cannot
    generate an invalid or unsafe renderer. ``weights`` are clamped on use.
    """

    mutation_operator: str = "drift"
    mutation_scale: float = 0.12
    fitness_weights: dict[str, float] = field(default_factory=dict)
    enabled_nodes: tuple[str, ...] = ("neon_edge", "bloom")
    crossover_bias: float = 0.5
    selection_pressure: float = 1.0

    def __post_init__(self):
        if self.mutation_operator not in MUTATION_OPERATORS:
            self.mutation_operator = "drift"
        for n in list(self.enabled_nodes):
            if n not in SHADER_NODES:
                self.enabled_nodes = tuple(x for x in self.enabled_nodes if x != n)
        if not self.fitness_weights:
            self.fitness_weights = {o: 1.0 for o in FITNESS_OBJECTIVES}

    # --- derived helpers (all bounded) ---
    def mutation_step(self, audio: dict[str, float]) -> float:
        base = self.mutation_scale * MUTATION_OPERATORS[self.mutation_operator]
        if self.mutation_operator == "pulse":
            base *= 1.0 + float(audio.get("beat", 0.0)) * 1.5
        return max(0.001, min(0.6, base))

    def weighted_fitness(self, objectives: dict[str, float]) -> float:
        total = 0.0
        wsum = 0.0
        for o, w in self.fitness_weights.items():
            if o not in objectives:
                continue
            w = max(0.0, min(5.0, w))
            total += objectives[o] * w
            wsum += w
        return total / max(wsum, 1e-6)

    def nodes_set(self) -> set:
        return set(self.enabled_nodes)

    # --- self-modification (bounded mutation of the policy itself) ---
    def self_modify(self, strength: float = 0.25) -> None:
        """Mutate the evolutionary machinery. Only curated elements change."""
        if random.random() < strength:
            # Switch mutation operator (curated, bounded).
            self.mutation_operator = random.choice(list(MUTATION_OPERATORS))
        if random.random() < strength:
            self.mutation_scale = max(
                0.01, min(0.4, self.mutation_scale + random.uniform(-strength, strength) * 0.05)
            )
        if random.random() < strength:
            # Reweight a fitness objective.
            o = random.choice(FITNESS_OBJECTIVES)
            self.fitness_weights[o] = max(
                0.0,
                min(
                    5.0,
                    self.fitness_weights.get(o, 1.0) + random.uniform(-strength, strength) * 0.5,
                ),
            )
        if random.random() < strength:
            # Toggle a shader node (curated, validated names).
            n = random.choice(SHADER_NODES)
            cur = set(self.enabled_nodes)
            if n in cur:
                if len(cur) > 1:
                    cur.discard(n)
            else:
                cur.add(n)
            self.enabled_nodes = tuple(sorted(cur))
        if random.random() < strength:
            self.crossover_bias = max(
                0.0, min(1.0, self.crossover_bias + random.uniform(-strength, strength) * 0.1)
            )
        if random.random() < strength:
            self.selection_pressure = max(
                0.2, min(3.0, self.selection_pressure + random.uniform(-strength, strength) * 0.2)
            )

    def clone(self) -> EvolutionPolicy:
        p = EvolutionPolicy(
            self.mutation_operator,
            self.mutation_scale,
            dict(self.fitness_weights),
            tuple(self.enabled_nodes),
            self.crossover_bias,
            self.selection_pressure,
        )
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_operator": self.mutation_operator,
            "mutation_scale": self.mutation_scale,
            "fitness_weights": dict(self.fitness_weights),
            "enabled_nodes": list(self.enabled_nodes),
            "crossover_bias": self.crossover_bias,
            "selection_pressure": self.selection_pressure,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionPolicy:
        return cls(
            d.get("mutation_operator", "drift"),
            float(d.get("mutation_scale", 0.12)),
            dict(d.get("fitness_weights", {})),
            tuple(d.get("enabled_nodes", ("neon_edge", "bloom"))),
            float(d.get("crossover_bias", 0.5)),
            float(d.get("selection_pressure", 1.0)),
        )


class SelfModifyingEngine:
    """Holds and evolves the population's collective EvolutionPolicy.

    A population of policies competes: the policy that produces the fittest
    organisms gets copied and self-modified. This is the loop where evolution
    rewrites its own rules -- while only ever touching curated, safe operators.
    """

    def __init__(self, n_policies: int = 6):
        self.policies: list[EvolutionPolicy] = [
            EvolutionPolicy() for _ in range(max(2, int(n_policies)))
        ]
        self.scores: list[float] = [0.0] * len(self.policies)
        self.generation = 0
        self.active = self.policies[0]

    def select(self) -> EvolutionPolicy:
        """Return the currently best-scoring policy as the active one."""
        i = max(range(len(self.policies)), key=lambda k: self.scores[k])
        self.active = self.policies[i]
        return self.active

    def report(self, policy_idx: int, fitness: float) -> None:
        if 0 <= policy_idx < len(self.scores):
            # Rolling average of how good the organisms this policy produced were.
            self.scores[policy_idx] = self.scores[policy_idx] * 0.8 + fitness * 0.2

    def evolve_policies(self) -> None:
        """Compete policies: keep the best, spawn mutated variants of it."""
        self.generation += 1
        ranked = sorted(range(len(self.policies)), key=lambda k: self.scores[k], reverse=True)
        best = ranked[0]
        new_policies = [self.policies[best].clone()]
        while len(new_policies) < len(self.policies):
            p = self.policies[best].clone()
            p.self_modify(strength=0.3)
            new_policies.append(p)
        self.policies = new_policies
        self.scores = [self.scores[best]] + [0.0] * (len(self.policies) - 1)
        self.active = self.policies[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "policies": [p.to_dict() for p in self.policies],
            "scores": list(self.scores),
            "active_idx": max(range(len(self.policies)), key=lambda k: self.scores[k]),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SelfModifyingEngine:
        e = cls(n_policies=max(2, len(d.get("policies", [{}]))))
        e.generation = int(d.get("generation", 0))
        e.policies = [EvolutionPolicy.from_dict(p) for p in d.get("policies", [])]
        e.scores = list(d.get("scores", [0.0] * len(e.policies)))
        if e.policies:
            e.active = e.policies[int(d.get("active_idx", 0)) % len(e.policies)]
        return e


# ===========================================================================
# 31. Per-Organism Spatial Maps ★ (the "spatial GPU ecosystem" tier)
# ===========================================================================
# Going further than the single global PersistentWorldTextures: every organism now
# owns its *own* little world -- four small maps centred on itself:
#
#   * territory  -- how strongly it claims each cell of its neighbourhood
#   * food       -- where it has found / expects resources
#   * pheromone  -- trail it lays down as it moves (self + kin signalling)
#   * danger     -- memory of predators / low-energy / danger signals
#
# The maps are CPU-side (cheap, tiny, per-organism) and evolve every tick.
# They are then *packed* into one shared GPU texture atlas (one tile per
# organism, RGBA = food / pheromone / danger / territory) so a single texture
# upload feeds every organism's shader. This keeps the GPU cost O(1) textures
# regardless of population size -- exactly the "CPU maps + packed GPU atlas"
# design -- while making evolution literally visible inside each territory.

MAP_CHANNELS = ("territory", "food", "pheromone", "danger")


class OrganismMaps:
    """Four small square maps an organism owns and continuously updates.

    Each map is a flat ``res*res`` list of floats in ``0..1``. Cell (r,r/2)
    at the centre is the organism itself; the maps describe the organism's
    immediate neighbourhood in *world* space, scaled by its perception range.
    """

    __slots__ = (
        "_decay",
        "_diffuse",
        "danger",
        "food",
        "grad_danger",
        "grad_food",
        "pheromone",
        "range",
        "res",
        "territory",
    )

    def __init__(self, res: int = 12, perception: float = 0.22):
        self.res = max(4, min(32, int(res)))
        self.range = max(MAP_RANGE_MIN, min(MAP_RANGE_MAX, float(perception)))
        n = self.res * self.res
        self.territory = [0.0] * n
        self.food = [0.0] * n
        self.pheromone = [0.0] * n
        self.danger = [0.0] * n
        # Per-map dynamics. Different organisms can evolve different rates so
        # some leave long-lived trails and others forget fast.
        self._decay = {
            "territory": MAP_DECAY_TERRITORY,
            "food": MAP_DECAY_FOOD,
            "pheromone": MAP_DECAY_PHEROMONE,
            "danger": MAP_DECAY_DANGER,
        }
        self._diffuse = {
            "territory": MAP_DIFFUSE_TERRITORY,
            "food": MAP_DIFFUSE_FOOD,
            "pheromone": MAP_DIFFUSE_PHEROMONE,
            "danger": MAP_DIFFUSE_DANGER,
        }
        # Cached steering gradients (recomputed once per update, read by the
        # organism's movement code to avoid recomputing per frame).
        self.grad_food = (0.0, 0.0)
        self.grad_danger = (0.0, 0.0)

    # -- coordinate helpers -------------------------------------------------
    def _world_to_cell(self, org_pos, wx: float, wy: float):
        """Map a world coord to a cell index, or None if outside the map."""
        half = self.range
        lx = (wx - org_pos[0]) / (2.0 * half) + 0.5
        ly = (wy - org_pos[1]) / (2.0 * half) + 0.5
        if not (0.0 <= lx <= 1.0 and 0.0 <= ly <= 1.0):
            return None
        cx = int(lx * (self.res - 1))
        cy = int(ly * (self.res - 1))
        return cy * self.res + cx

    def _deposit(self, arr, idx: int, amount: float) -> None:
        if idx is None:
            return
        r = self.res
        cx = idx % r
        cy = idx // r
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < r and 0 <= y < r:
                    w = 1.0 if (dx == 0 and dy == 0) else 0.35
                    j = y * r + x
                    arr[j] = min(1.0, arr[j] + amount * w)

    # -- per-tick update ----------------------------------------------------
    def update(
        self,
        host: Any,
        audio: dict[str, float],
        neighbours: list[Any] | None = None,
        emotion: Any | None = None,
    ) -> None:
        """Evolve the four maps one step from the organism's situation."""
        pos = getattr(host, "pos", (0.5, 0.5))
        r = self.res
        centre = (r // 2) * r + (r // 2)

        # -- territory: strongest at self, claimed by presence & energy.
        energy = max(0.0, min(1.0, float(getattr(host, "energy", 1.0)) / 3.0))
        self._deposit(
            self.territory,
            centre,
            MAP_DEPOSIT_TERRITORY_BASE + MAP_DEPOSIT_TERRITORY_ENERGY * energy,
        )

        # -- food: deposit where resources actually are (from world field).
        eco = getattr(host, "_eco_ref", None)
        field = getattr(eco, "field", None)
        if field is not None and hasattr(field, "sample"):
            for gx in range(0, r, 2):
                for gy in range(0, r, 2):
                    wx = pos[0] + ((gx / (r - 1)) - 0.5) * 2.0 * self.range
                    wy = pos[1] + ((gy / (r - 1)) - 0.5) * 2.0 * self.range
                    if not (0.0 <= wx <= 1.0 and 0.0 <= wy <= 1.0):
                        continue
                    try:
                        val = (field.sample("bass", wx, wy) + field.sample("treb", wx, wy)) * 0.5
                    except (TypeError, ValueError):
                        val = 0.0
                    self.food[gy * r + gx] = min(
                        1.0,
                        MAP_DEPOSIT_FOOD_DECAY * self.food[gy * r + gx]
                        + MAP_DEPOSIT_FOOD_FIELD * min(1.0, val),
                    )
        else:
            # No field: fall back to audio-driven food estimate at the centre.
            self._deposit(
                self.food,
                centre,
                (float(audio.get("bass", 0.0)) + float(audio.get("treb", 0.0))) * 0.15,
            )

        # -- pheromone: laid at self, stronger when excited / moving.
        beat = float(audio.get("beat", 0.0))
        excite = float(getattr(emotion, "excitement", 0.0)) if emotion else beat
        self._deposit(
            self.pheromone,
            centre,
            MAP_DEPOSIT_PHEROMONE_BASE + MAP_DEPOSIT_PHEROMONE_EXCITE * excite,
        )

        # -- danger: from nearby predators, danger signals, and low energy.
        danger_amt = (1.0 - energy) * MAP_DEPOSIT_DANGER_ENERGY
        if getattr(host, "emotional_state", "") in ("fear",):
            danger_amt += MAP_DEPOSIT_DANGER_FEAR
        self._deposit(self.danger, centre, danger_amt)
        if neighbours:
            for nb in neighbours:
                if nb is host or not getattr(nb, "alive", True):
                    continue
                npos = getattr(nb, "pos", None)
                if npos is None:
                    continue
                idx = self._world_to_cell(pos, npos[0], npos[1])
                if idx is None:
                    continue
                # A neighbour that eats our species is a danger; others crowd
                # our territory (competition).
                self._deposit(self.territory, idx, MAP_DEPOSIT_TERRITORY_COMPETITION)
                if getattr(nb, "energy", 0.0) > float(getattr(host, "energy", 0.0)):
                    self._deposit(self.danger, idx, MAP_DEPOSIT_DANGER_NEIGHBOUR)
        # External danger signals in the world.
        for sig in getattr(eco, "signals", []) or []:
            if getattr(sig, "stype", "") != "danger":
                continue
            idx = self._world_to_cell(pos, getattr(sig, "x", 0.0), getattr(sig, "y", 0.0))
            self._deposit(
                self.danger, idx, MAP_DEPOSIT_DANGER_SIGNAL * float(getattr(sig, "strength", 1.0))
            )

        self._step_dynamics()
        # Refresh cached steering gradients once per update.
        self.grad_food = self.gradient("food")
        self.grad_danger = self.gradient("danger")

    def _step_dynamics(self) -> None:
        """Decay + light diffusion for each map (in place)."""
        r = self.res
        for name in MAP_CHANNELS:
            arr = getattr(self, name)
            decay = self._decay[name]
            diff = self._diffuse[name]
            if diff > 0.0:
                src = list(arr)
                for y in range(r):
                    for x in range(r):
                        i = y * r + x
                        acc = 0.0
                        cnt = 0
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                xx = x + dx
                                yy = y + dy
                                if 0 <= xx < r and 0 <= yy < r:
                                    acc += src[yy * r + xx]
                                    cnt += 1
                        blurred = acc / max(1, cnt)
                        arr[i] = (src[i] * (1.0 - diff) + blurred * diff) * decay
            else:
                for i in range(len(arr)):
                    arr[i] *= decay

    def mutate(self, strength: float = 0.1) -> None:
        """Evolve the map dynamics (resolution stays fixed; rates drift)."""
        if random.random() < strength:
            self.range = max(
                MAP_RANGE_MIN,
                min(MAP_RANGE_MAX, self.range + random.uniform(-strength, strength) * 0.1),
            )
        for name in MAP_CHANNELS:
            if random.random() < strength:
                self._decay[name] = max(
                    0.8, min(0.999, self._decay[name] + random.uniform(-strength, strength) * 0.05)
                )
            if random.random() < strength:
                self._diffuse[name] = max(
                    0.0, min(0.4, self._diffuse[name] + random.uniform(-strength, strength) * 0.1)
                )

    # -- behaviour hooks ----------------------------------------------------
    def gradient(self, name: str) -> tuple[float, float]:
        """World-space gradient (points toward higher values) of a map.

        Lets the organism steer using its own maps: climb food, flee danger.
        """
        arr = getattr(self, name, None)
        if not arr:
            return (0.0, 0.0)
        r = self.res
        gx = gy = 0.0
        wsum = 0.0
        for y in range(r):
            for x in range(r):
                v = arr[y * r + x]
                if v <= 1e-4:
                    continue
                dx = (x / (r - 1)) - 0.5
                dy = (y / (r - 1)) - 0.5
                gx += dx * v
                gy += dy * v
                wsum += v
        if wsum <= 1e-6:
            return (0.0, 0.0)
        return (gx / wsum * 2.0 * self.range, gy / wsum * 2.0 * self.range)

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        # Maps are transient/regenerating: persist only shape + evolved rates.
        return {
            "res": self.res,
            "range": self.range,
            "decay": dict(self._decay),
            "diffuse": dict(self._diffuse),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OrganismMaps:
        m = cls(int(d.get("res", 12)), float(d.get("range", 0.22)))
        m._decay.update({k: float(v) for k, v in d.get("decay", {}).items()})
        m._diffuse.update({k: float(v) for k, v in d.get("diffuse", {}).items()})
        return m

    def clone(self) -> OrganismMaps:
        m = OrganismMaps(self.res, self.range)
        m._decay = dict(self._decay)
        m._diffuse = dict(self._diffuse)
        return m


class OrganismMapAtlas:
    """Packs every organism's four maps into one RGBA CPU buffer for the GPU.

    Layout: a near-square grid of tiles, one tile per organism, each tile
    ``res*res`` texels. Channels per texel are:

        R = food, G = pheromone, B = danger, A = territory

    The renderer uploads this single buffer to one texture; each organism's
    shader samples its own tile via ``uv_min`` / ``uv_size`` uniforms. This is
    what turns per-organism CPU maps into a single, scalable GPU resource.
    """

    def __init__(self, tile_res: int = 12, max_tiles: int = 256):
        self.tile_res = max(4, min(32, int(tile_res)))
        self.max_tiles = max(1, int(max_tiles))
        self.cols = 1
        self.rows = 1
        self.width = self.tile_res
        self.height = self.tile_res
        self._buffer: Any | None = None
        self._tile_of: dict[int, int] = {}  # organism id -> tile index

    def _ensure_grid(self, n_tiles: int) -> None:
        n = max(1, min(self.max_tiles, n_tiles))
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, math.ceil(n / cols))
        if cols != self.cols or rows != self.rows:
            self.cols = cols
            self.rows = rows
            self.width = cols * self.tile_res
            self.height = rows * self.tile_res
            self._buffer = None  # force reallocation

    def uv_for(self, tile_idx: int) -> tuple[float, float, float, float]:
        """Return (min_u, min_v, size_u, size_v) for a tile in the atlas."""
        col = tile_idx % self.cols
        row = tile_idx // self.cols
        su = self.tile_res / float(self.width)
        sv = self.tile_res / float(self.height)
        return (col * su, row * sv, su, sv)

    def pack(self, entries: list[tuple[int, OrganismMaps]]):
        """Fill the atlas buffer from ``(org_id, OrganismMaps)`` pairs.

        Returns a flat float RGBA buffer (``array``/``list``) of length
        ``width*height*4`` and refreshes ``_tile_of`` mapping.
        """
        try:
            import numpy as _np
        except ImportError:
            _np = None
        n = min(len(entries), self.max_tiles)
        self._ensure_grid(n)
        tr = self.tile_res
        if _np is not None:
            if self._buffer is None or self._buffer.shape[0] != self.height * self.width * 4:
                self._buffer = _np.zeros(self.height * self.width * 4, dtype=_np.float32)
            buf = self._buffer
            buf.fill(0.0)
            self._tile_of.clear()
            # View the flat buffer as (H, W, 4) so each tile is one slice
            # assignment per channel instead of a per-texel Python loop.
            view = buf.reshape(self.height, self.width, 4)
            for tile_idx, (oid, maps) in enumerate(entries[:n]):
                self._tile_of[oid] = tile_idx
                col = tile_idx % self.cols
                row = tile_idx // self.cols
                bx = col * tr
                by = row * tr
                tile = view[by : by + tr, bx : bx + tr, :]
                tile[:, :, 0] = _np.asarray(maps.food, dtype=_np.float32).reshape(tr, tr)
                tile[:, :, 1] = _np.asarray(maps.pheromone, dtype=_np.float32).reshape(tr, tr)
                tile[:, :, 2] = _np.asarray(maps.danger, dtype=_np.float32).reshape(tr, tr)
                tile[:, :, 3] = _np.asarray(maps.territory, dtype=_np.float32).reshape(tr, tr)
            return buf
        # Pure-python fallback.
        buf = [0.0] * (self.width * self.height * 4)
        self._tile_of.clear()
        for tile_idx, (oid, maps) in enumerate(entries[:n]):
            self._tile_of[oid] = tile_idx
            col = tile_idx % self.cols
            row = tile_idx // self.cols
            base_x = col * tr
            base_y = row * tr
            for cy in range(tr):
                for cx in range(tr):
                    src = cy * tr + cx
                    px = base_x + cx
                    py = base_y + cy
                    di = (py * self.width + px) * 4
                    buf[di] = maps.food[src]
                    buf[di + 1] = maps.pheromone[src]
                    buf[di + 2] = maps.danger[src]
                    buf[di + 3] = maps.territory[src]
        self._buffer = buf
        return buf

    def tile_index_of(self, org_id: int) -> int | None:
        return self._tile_of.get(org_id)


# ===========================================================================
# Integration glue: per-organism advanced state + ecosystem driver
# ===========================================================================
# These two classes bind every feature above onto the existing
# ``EnhancedSpatialEcosystem`` / ``SpatialOrganism`` in evo_visualizer.py.
# They are deliberately thin: the heavy lifting lives in the systems above.


class StageOrganism:
    """Attaches all advanced developmental systems to one SpatialOrganism.

    The host organism keeps its existing genome/energy; ``StageOrganism`` adds
    the GRN, continuous mutator, novelty memory, emotions, repair, sexual
    selector, renderer choice, living material, and per-organism replay/lineage.
    """

    def __init__(self, host: Any, config: Any, model: PredictiveMutationModel | None = None):
        self.host = host
        self.config = config
        self.model = model
        self.grn = GeneRegulatoryNetwork() if config.enable_gene_regulatory_network else None
        self.cont = ContinuousMutator() if config.enable_continuous_mutation else None
        self.memory = OrganismMemory() if config.enable_memory_evolution else None
        self.emotion = EmotionalState() if config.enable_emotional_state else None
        self.repair = GenomeRepair() if config.enable_genome_repair else None
        self.selector = SexualSelector() if config.enable_sexual_selection else None
        self.renderer = RendererGenome.random() if config.enable_renderer_evolution else None
        self.material = LivingMaterial.random() if config.enable_living_materials else None
        self.maps = (
            OrganismMaps(res=getattr(config, "organism_map_res", 12))
            if getattr(config, "enable_organism_maps", False)
            else None
        )
        self.stage = "egg"
        self._novelty = 0.0
        self._policy_idx = 0

    # --- developmental life-stage rendering params ---
    def render_params(self) -> dict[str, Any]:
        params = (
            getattr(self.host.genome, "params", None)
            or getattr(self.host.genome, "to_dict", dict)()
        )
        if self.config.enable_development:
            return mature_params(dict(params), self.stage)
        return dict(params)

    def shader_nodes(self) -> set:
        """Nodes this organism's renderer enables (for the shader factory)."""
        if self.renderer is None:
            return set()
        # Map the renderer-algorithm choice to a validated node family.
        mapping = {
            "fbm": "neon_edge",
            "raymarch": "depth_cloud",
            "distance_field": "metaballs",
            "reaction_diffusion": "reaction_diffusion",
            "particle_fluid": "caustics",
            "voronoi": "voronoi",
            "metaballs": "metaballs",
            "fractal": "kaleidoscope",
            "wave_equation": "wave",
            "cellular_automata": "cells",
            "signed_distance_field": "voronoi",
        }
        nodes = (
            set(getattr(self.renderer, "_nodes", ()))
            if hasattr(self.renderer, "_nodes")
            else set()
        )
        nodes.add(mapping.get(self.renderer.algorithm, "bloom"))
        return nodes

    def step(
        self,
        dt: float,
        audio: dict[str, float],
        novelty_archive: NoveltyArchive | None = None,
        neighbours: list[Any] | None = None,
        update_maps: bool = True,
    ) -> None:
        host = self.host
        p = getattr(host.genome, "params", None) or {}
        if not isinstance(p, dict):
            return

        # Apply AI-guided predictive bias before mutation.
        if self.model is not None and self.cont is not None:
            apply_predictive_bias(self.cont, p, audio, self.model)

        # 16 GRN
        if self.grn is not None:
            self.grn.apply(p, audio)
        # 18 continuous mutation
        if self.cont is not None:
            self.cont.step(p, audio, dt)
        # 23 repair (re-establish healthy drift after mutation)
        if self.repair is not None:
            bounds = getattr(host.genome, "PARAM_BOUNDS", {})
            if bounds:
                self.repair.repair(p, bounds)
            else:
                self.repair.backup(p)
        # 17 development stage from age
        if self.config.enable_development:
            self.stage = stage_for_age(host.age, host.life_span)
        # 20 novelty
        if novelty_archive is not None:
            self._novelty = novelty_archive.novelty(host.genome)
            host._novelty = self._novelty
        # 22 emotion
        if self.emotion is not None:
            self.emotion.update(audio, getattr(host, "energy", 1.0), self._novelty)
        # 21 memory
        if self.memory is not None:
            self.memory.record(audio, getattr(host, "alive", True), getattr(host, "fitness", 0.0))
        # 31 per-organism spatial maps (territory/food/pheromone/danger).
        if self.maps is not None and update_maps:
            self.maps.update(host, audio, neighbours, self.emotion)
        # 32 AI-guided evolution signals.
        if self.model is not None:
            record_ai_signals(
                self.model,
                host.genome,
                p,
                audio,
                float(getattr(host, "fitness", 0.0)),
                getattr(host, "alive", True),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grn": self.grn.to_dict() if self.grn else None,
            "memory": self.memory.to_dict() if self.memory else None,
            "emotion": self.emotion.to_dict() if self.emotion else None,
            "repair": self.repair.to_dict() if self.repair else None,
            "selector": self.selector.to_dict() if self.selector else None,
            "renderer": self.renderer.to_dict() if self.renderer else None,
            "material": self.material.to_dict() if self.material else None,
            "maps": self.maps.to_dict() if self.maps else None,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, host: Any, config: Any, d: dict[str, Any]) -> StageOrganism:
        s = cls(host, config)
        if d.get("grn") and s.grn:
            s.grn = GeneRegulatoryNetwork.from_dict(d["grn"])
        if d.get("memory") and s.memory:
            s.memory = OrganismMemory.from_dict(d["memory"])
        if d.get("emotion") and s.emotion:
            s.emotion = EmotionalState.from_dict(d["emotion"])
        if d.get("repair") and s.repair:
            s.repair = GenomeRepair.from_dict(d["repair"])
        if d.get("selector") and s.selector:
            s.selector = SexualSelector.from_dict(d["selector"])
        if d.get("renderer") and s.renderer:
            s.renderer = RendererGenome.from_dict(d["renderer"])
        if d.get("material") and s.material:
            s.material = LivingMaterial.from_dict(d["material"])
        if d.get("maps") and s.maps:
            s.maps = OrganismMaps.from_dict(d["maps"])
        s.stage = d.get("stage", "egg")
        return s


class AdvancedEcosystem:
    """Drives all developmental systems for a host ecosystem each tick.

    Also owns the performance optimisation layer: frustum culling,
    texture atlas packing, persistent GPU buffers, shader binary
    caching, bindless textures, multi-draw indirect, persistent SSBOs,
    and GPU timestamp queries.  The GPU-side objects are initialised
    by the renderer (``evo_visualizer.py``) which calls ``init_gl()``
    on each feature after the GL context is current.
    """

    def __init__(self, host_eco: Any, config: Any):
        self.host = host_eco
        self.config = config
        self.novelty = NoveltyArchive() if config.enable_novelty_search else None
        self.niches = NicheMap() if config.enable_niches else None
        self.recorder = EvolutionRecorder() if config.enable_evolution_replay else None
        self.archive = GeneticArchive() if config.enable_genetic_archaeology else None
        self.engine = SelfModifyingEngine() if config.enable_self_modifying else None
        self.map_atlas = (
            OrganismMapAtlas(
                tile_res=getattr(config, "organism_map_res", 12),
                max_tiles=getattr(config, "organism_map_max_tiles", 256),
            )
            if getattr(config, "enable_organism_maps", False)
            else None
        )
        # 32 AI-guided evolution model.
        self.ai_model = (
            PredictiveMutationModel()
            if getattr(config, "enable_ai_guided_evolution", False)
            else None
        )
        self._stage: dict[int, StageOrganism] = {}
        self._next_policy_step = 0.0
        self._frame = 0
        self._map_interval = max(1, int(getattr(config, "organism_map_update_interval", 3)))

        # ------------------------------------------------------------------
        # Performance optimisation layer (spec items 1-8)
        # ------------------------------------------------------------------
        # 1. Persistent mapped PBOs — CPU-side readback buffer that stays
        #    mapped for the lifetime of the GL context; the renderer calls
        #    init_gl() to create and map the buffers.
        self._perf_pbo = PersistentMappedPBO(
            res=getattr(config, "feedback_res", 512),
            n_buffers=3,
        )
        # 2. Shader binary cache — glProgramBinary to avoid recompilation
        #    between runs; the renderer calls init_gl() and then
        #    try_load()/store() around shader linking.
        self._perf_shader_bin_cache = ShaderBinaryCache(
            max_entries=getattr(config, "shader_cache_max_entries", 512),
        )
        # 3. Bindless texture array — per-organism ecosystem maps as a
        #    single bindless texture; the renderer calls init_gl() with
        #    the correct layer count and dimensions.
        self._perf_bindless_atlas = BindlessTextureArray()
        # 4. Multi-draw indirect — batches per-organism draw calls into
        #    one glMultiDrawArraysIndirect call; the renderer calls
        #    init_gl() and then build_commands()/draw().
        self._perf_multi_draw = MultiDrawIndirect(
            max_organisms=getattr(config, "max_organisms", 64),
        )
        # 5. Persistent SSBO — organism parameter data that stays mapped;
        #    the renderer calls init_gl() and then write_organisms()/bind().
        self._perf_org_ssbo = PersistentSSBO(
            max_organisms=getattr(config, "max_organisms", 64),
            floats_per_org=getattr(config, "floats_per_org", 16),
        )
        # 6. GPU timestamp queries — measure per-organism rendering cost;
        #    the renderer calls init_gl() and then begin()/end()/collect().
        self._perf_gpu_timer = GpuTimerQuery(
            max_queries=getattr(config, "gpu_timer_max_queries", 128),
        )
        #    Per-stage GPU timer for measuring the cost of each render pass.
        self._perf_stage_timer = GpuProfilerTimer()
        # 7. Frustum/visibility culling — skip organisms outside the
        #    visible screen region; the renderer calls set_mvp() each
        #    frame and cull_organisms() before draw calls.
        self._perf_frustum = FrustumCuller()
        # 8. Texture atlas packing — shelf-based packing of ecosystem maps
        #    into a single GPU texture; the renderer calls init_gl() and
        #    then pack_tile()/update_tile()/upload().
        self._perf_atlas_packer = TextureAtlasPacker(
            max_tiles=getattr(config, "organism_map_max_tiles", 256),
            tile_res=getattr(config, "organism_map_res", 12),
        )

    def _stage_for(self, org: Any) -> StageOrganism:
        key = id(org)
        st = self._stage.get(key)
        if st is None or st.host is not org:
            st = StageOrganism(org, self.config, self.ai_model)
            # Spec 31: inherit + mutate the parent's evolved map dynamics so
            # decay/diffusion/range actually evolve across generations and
            # selection can favour long-trail vs fast-forget lineages.
            if st.maps is not None:
                parent_maps = getattr(org, "_parent_maps", None)
                if parent_maps is not None:
                    try:
                        st.maps = parent_maps.clone()
                        st.maps.mutate(strength=0.12)
                    except (AttributeError, TypeError) as exc:
                        logger.debug("Failed to clone/mutate parent_maps: %s", exc)
                    try:
                        org._parent_maps = None
                    except (AttributeError, TypeError):
                        pass
            self._stage[key] = st
        return st

    def _nearby(
        self, org: Any, alive: list[Any], radius: float = 0.3, limit: int = 8
    ) -> list[Any]:
        """Cheap neighbour set within ``radius`` for the maps to consume."""
        ox, oy = org.pos[0], org.pos[1]
        out = []
        for o in alive:
            if o is org:
                continue
            if abs(o.pos[0] - ox) > radius or abs(o.pos[1] - oy) > radius:
                continue
            if math.hypot(o.pos[0] - ox, o.pos[1] - oy) <= radius:
                out.append(o)
                if len(out) >= limit:
                    break
        return out

    def map_atlas_frame(self) -> tuple[Any, OrganismMapAtlas] | None:
        """Pack all living organisms' maps into the atlas and return it.

        Returns ``(buffer, atlas)`` or ``None`` if maps are disabled. The
        renderer uploads ``buffer`` to a single GPU texture and uses
        ``atlas.tile_index_of(id(org))`` + ``atlas.uv_for(idx)`` to locate each
        organism's tile.

        When the GPU-side :class:`TextureAtlasPacker` has been initialised
        by the renderer, the CPU-side ``OrganismMapAtlas`` packing is
        skipped and the renderer uses the packer directly.
        """
        if self.map_atlas is None:
            return None
        if self._perf_atlas_packer.available:
            return None
        entries: list[tuple[int, OrganismMaps]] = []
        for org in self.host.alive_organisms():
            st = self._stage.get(id(org))
            if st is not None and st.maps is not None:
                entries.append((id(org), st.maps))
        buf = self.map_atlas.pack(entries)
        return (buf, self.map_atlas)

    def set_mvp(self, mvp: list[float]) -> None:
        """Pass the combined model-view-projection matrix to the frustum
        culler so that visibility tests use the current camera."""
        self._perf_frustum.set_mvp(mvp)

    def cull_organisms(self, organisms: list, radius: float = 0.15) -> list:
        """Return the subset of *organisms* whose bounding sphere is
        inside the frustum.  Organisms outside the visible region are
        skipped by the renderer, saving per-organism draw overhead.

        ``radius`` is the organism's bounding-sphere radius in
        normalised world space (0..1).
        """
        visible = []
        for org in organisms:
            pos = getattr(org, "pos", (0.5, 0.5))
            if self._perf_frustum.sphere_visible(
                pos[0],
                pos[1],
                0.0,
                radius,
            ):
                visible.append(org)
        return visible

    def step(self, dt: float, audio: dict[str, float]) -> None:
        eco = self.host
        alive = eco.alive_organisms()

        # 7 Frustum/visibility culling — skip off-screen organisms.
        visible = self.cull_organisms(alive)

        # 27 niches drift with the music.
        if self.niches is not None:
            intensity = max(
                0.0,
                min(
                    1.0, float(audio.get("bass", 0.0)) * 0.5 + float(audio.get("beat", 0.0)) * 0.5
                ),
            )
            self.niches.step(audio, dt, intensity)

        # 19 multi-objective + 20 novelty + 22 emotion + etc per organism.
        objectives = []
        want_maps = self.map_atlas is not None
        self._frame += 1
        maps_due = want_maps and (self._frame % self._map_interval == 0)
        for org in visible:
            st = self._stage_for(org)
            neighbours = None
            # Skip map work for distant (LOD 0) organisms; they contribute
            # little visually and their maps can update less often.
            org_maps_due = maps_due and getattr(org, "_lod_tier", 2) > 0
            if want_maps and org_maps_due:
                # Give each organism a reference to the world so its maps can
                # sample the resource field / signals, and a small neighbour set.
                try:
                    org._eco_ref = eco
                except (AttributeError, TypeError):
                    pass
                neighbours = self._nearby(org, visible)
            st.step(dt, audio, self.novelty, neighbours, update_maps=org_maps_due)
            if want_maps and st.maps is not None:
                # Expose the maps on the host so its movement code can steer by
                # its own food/danger gradients on the next tick.
                try:
                    org._own_maps = st.maps
                except (AttributeError, TypeError):
                    pass
            # 27 niche suitability feeds survival.
            if self.niches is not None:
                nic = self.niches.best_for(org.pos[0], org.pos[1], audio, 0.5)
                if nic is not None:
                    org.fitness = org.fitness * 0.9 + nic.affinity * 0.1
            # 19 compute multi-objective vector and (optionally) let the
            # self-modifying policy decide fitness meaning.
            obj = evaluate_objectives(
                org.genome,
                float(org.fitness),
                audio,
                gpu_cost=st.renderer.blend if st.renderer else 0.2,
                novelty=st._novelty,
                complexity=float(getattr(org, "complexity_level", 0)) / 7.0,
            )
            if self.engine is not None:
                pol = self.engine.select()
                obj["_policy_idx"] = self.engine.policies.index(pol)
                # 30: fitness "means" whatever the policy weights.
                org.fitness = pol.weighted_fitness(obj)
            else:
                org.fitness = obj.get("beauty", org.fitness)
            objectives.append(obj)

        # 19 Pareto ranking replaces single-score survival ordering.
        if self.config.enable_multi_objective and objectives:
            ranks = pareto_rank(objectives)
            for org, rank in zip(visible, ranks):
                # Lower rank (closer to frontier) = small survival bonus.
                org.fitness = float(org.fitness) * 0.95 + (1.0 / (1.0 + rank)) * 0.05

        # 24 sexual selection: occasionally replace random replacement with a
        # chosen-mate child.
        if self.config.enable_sexual_selection and self.engine is None:
            self._sexual_reproduction(visible, audio)

        # 30 self-modifying engine competes its policies periodically.
        if self.engine is not None:
            now = time.time()
            if now - self._next_policy_step > 20.0:
                self._next_policy_step = now
                best_fit = max((float(o.fitness) for o in visible), default=0.0)
                self.engine.report(
                    max(range(len(self.engine.policies)), key=lambda k: self.engine.scores[k]),
                    best_fit,
                )
                self.engine.evolve_policies()

        # 28/29 record + archive events.
        for org in visible:
            if self.novelty is not None:
                self.novelty.add(org.genome)
            if self.recorder is not None and random.random() < 0.05:
                self.recorder.record(
                    "mutation",
                    getattr(org, "lineage", str(id(org))),
                    {"stage": self._stage_for(org).stage},
                )
            if self.archive is not None and random.random() < 0.01:
                self.archive.add(
                    TreeNode(
                        getattr(org, "lineage", str(id(org))),
                        getattr(org, "parent_id", None),
                        getattr(org.genome, "to_dict", dict)(),
                        float(org.fitness),
                        getattr(org, "generation", 0),
                        getattr(org, "species", 0),
                        time.time(),
                    )
                )

    def _sexual_reproduction(self, alive: list, audio: dict[str, float]) -> None:
        if len(alive) < 2:
            return
        for org in alive:
            st = self._stage_for(org)
            if st.selector is None:
                continue
            if random.random() < 0.02 and getattr(org, "energy", 0.0) > 1.3:
                mate = choose_mate(org, st.selector, alive)
                if mate is not None and hasattr(org.genome, "crossover"):
                    try:
                        child = org.genome.crossover(mate.genome)
                        child.mutate()
                        org.energy -= 0.3
                        if self.recorder is not None:
                            self.recorder.record(
                                "birth",
                                getattr(child, "lineage", str(id(child))),
                                {
                                    "parents": [
                                        getattr(org, "lineage", "?"),
                                        getattr(mate, "lineage", "?"),
                                    ]
                                },
                            )
                    except (AttributeError, TypeError, ValueError):
                        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "novelty": self.novelty.to_dict() if self.novelty else None,
            "niches": self.niches.to_dict() if self.niches else None,
            "recorder": self.recorder.to_dict() if self.recorder else None,
            "archive": self.archive.to_dict() if self.archive else None,
            "engine": self.engine.to_dict() if self.engine else None,
            "ai_model": self.ai_model.to_dict() if self.ai_model else None,
        }

    @classmethod
    def from_dict(cls, host_eco: Any, config: Any, d: dict[str, Any]) -> AdvancedEcosystem:
        a = cls(host_eco, config)
        if d.get("novelty") and a.novelty:
            a.novelty = NoveltyArchive.from_dict(d["novelty"])
        if d.get("niches") and a.niches:
            a.niches = NicheMap.from_dict(d["niches"])
        if d.get("recorder") and a.recorder:
            a.recorder = EvolutionRecorder.from_dict(d["recorder"])
        if d.get("archive") and a.archive:
            a.archive = GeneticArchive.from_dict(d["archive"])
        if d.get("engine") and a.engine:
            a.engine = SelfModifyingEngine.from_dict(d["engine"])
        if d.get("ai_model") and a.ai_model:
            a.ai_model = PredictiveMutationModel.from_dict(d["ai_model"])
        return a


# ===========================================================================
# 32. AI-Guided Evolution ★
# ===========================================================================
# Instead of relying solely on objective fitness, the system learns from
# four signals and uses them to build a predictive model that biases
# mutation *before* rendering:
#
#   1. "What visuals survive?"  -- VisualSurvivalTracker
#   2. "What visuals repeat?"   -- VisualRepeatDetector
#   3. "What visuals react best?" -- ReactionScorer
#   4. "What visuals users like?" -- UserPreferenceModel
#
# The PredictiveMutationModel fuses these into per-parameter mutation
# multipliers, so the next generation is biased toward visuals that
# have proven robust, recurrent, reactive, or preferred -- without
# ever leaving the curated, bounded mutation operators already in place.

# ---------------------------------------------------------------------------
# 32a. VisualSurvivalTracker
# ---------------------------------------------------------------------------
# Records the genome parameters of organisms that survive across
# generations and computes per-parameter persistence scores. A parameter
# that consistently appears in survivors is "what survives" and should
# be preserved (lower mutation probability) rather than explored.


@dataclass
class SurvivalRecord:
    """One snapshot of a genome's params at the moment it survived."""

    params: dict[str, float]
    fitness: float = 0.0
    generation: int = 0


class VisualSurvivalTracker:
    """Tracks which genome parameter values persist across generations."""

    def __init__(self, capacity: int = 200, survival_threshold: float = 0.5):
        self.capacity = max(16, int(capacity))
        self.survival_threshold = max(0.0, min(1.0, survival_threshold))
        self._records: list[SurvivalRecord] = []
        self._generation = 0
        # Per-parameter persistence: how often a param value band is
        # occupied by survivors. Keys are (param_name, discretised_bucket).
        self._persistence: dict[tuple[str, int], float] = {}

    def record_survivor(self, genome: Any, fitness: float) -> None:
        params = getattr(genome, "params", None) or getattr(genome, "to_dict", dict)()
        if not isinstance(params, dict):
            return
        numeric = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
        if not numeric:
            return
        self._records.append(SurvivalRecord(numeric, fitness, self._generation))
        if len(self._records) > self.capacity:
            self._records.pop(0)
        # Update persistence counts for survivors above threshold.
        if fitness >= self.survival_threshold:
            for k, v in numeric.items():
                bucket = self._discretise(k, v)
                key = (k, bucket)
                self._persistence[key] = self._persistence.get(key, 0.0) + 1.0

    def _discretise(
        self, key: str, value: float, bands: int = SURVIVAL_DISCRETISATION_BANDS
    ) -> int:
        """Map a continuous value to a discrete bucket for frequency counting."""
        lo, hi = PARAM_BOUNDS.get(key, (0.0, 1.0))
        if hi == lo:
            return 0
        frac = (value - lo) / (hi - lo)
        frac = max(0.0, min(1.0, frac))
        return int(frac * bands) % bands

    def persistence(self, param: str, value: float) -> float:
        """Return 0..1 how often this param value band has been occupied by survivors."""
        bucket = self._discretise(param, value)
        key = (param, bucket)
        total = sum(v for (k, _), v in self._persistence.items() if k == param)
        if total <= 0.0:
            return 0.5  # Neutral: no data yet.
        return min(1.0, self._persistence.get(key, 0.0) / total)

    def survival_bias(self, params: dict[str, Any]) -> dict[str, float]:
        """Return per-parameter multipliers: >1 means preserve (less mutation),
        <1 means explore more. Survivors bias toward preservation."""
        out: dict[str, float] = {}
        for k, v in params.items():
            if not isinstance(v, (int, float)):
                continue
            p = self.persistence(k, float(v))
            # Map persistence 0..1 -> bias 0.5..1.5 (preserved params mutate less).
            out[k] = 1.5 - p
        return out

    def next_generation(self) -> None:
        self._generation += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "threshold": self.survival_threshold,
            "generation": self._generation,
            "records": [
                {"params": r.params, "fitness": r.fitness, "generation": r.generation}
                for r in self._records
            ],
            "persistence": {f"{k[0]}:{k[1]}": v for k, v in self._persistence.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VisualSurvivalTracker:
        t = cls(capacity=d.get("capacity", 200), survival_threshold=d.get("threshold", 0.5))
        t._generation = int(d.get("generation", 0))
        for r in d.get("records", []):
            t._records.append(
                SurvivalRecord(
                    r.get("params", {}), float(r.get("fitness", 0.0)), int(r.get("generation", 0))
                )
            )
        for k_str, v in d.get("persistence", {}).items():
            parts = k_str.rsplit(":", 1)
            if len(parts) == 2:
                t._persistence[(parts[0], int(parts[1]))] = float(v)
        return t

    def clone(self) -> VisualSurvivalTracker:
        t = VisualSurvivalTracker(self.capacity, self.survival_threshold)
        t._records = [
            SurvivalRecord(dict(r.params), r.fitness, r.generation) for r in self._records
        ]
        t._persistence = dict(self._persistence)
        t._generation = self._generation
        return t


# ---------------------------------------------------------------------------
# 32b. VisualRepeatDetector
# ---------------------------------------------------------------------------
# Identifies which visual patterns (genome parameter combinations) recur
# frequently across the current population. Recurring patterns are
# "what repeats" and signal a convergent visual motif that the predictive
# model can reinforce.


@dataclass
class RepeatPattern:
    """A recurring combination of genome parameter values."""

    signature: tuple[tuple[str, int], ...]
    count: int = 0
    avg_fitness: float = 0.0


class VisualRepeatDetector:
    """Detects recurring visual patterns in a population of genomes."""

    def __init__(
        self,
        bands: int = REPEAT_DETECTOR_BANDS,
        min_count: int = REPEAT_MIN_COUNT,
        capacity: int = 50,
    ):
        self.bands = max(2, int(bands))
        self.min_count = max(2, int(min_count))
        self.capacity = max(4, int(capacity))
        self._patterns: list[RepeatPattern] = []
        self._param_keys: list[str] = []

    def _signature(self, params: dict[str, Any]) -> tuple[tuple[str, int], ...]:
        """Discretise every numeric param into a bucket tuple for hashing."""
        keys = sorted(k for k, v in params.items() if isinstance(v, (int, float)))
        sig: list[tuple[str, int]] = []
        for k in keys:
            v = float(params[k])
            lo, hi = PARAM_BOUNDS.get(k, (0.0, 1.0))
            if hi == lo:
                bucket = 0
            else:
                frac = max(0.0, min(1.0, (v - lo) / (hi - lo)))
                bucket = int(frac * self.bands) % self.bands
            sig.append((k, bucket))
        return tuple(sig)

    def observe(self, genome: Any, fitness: float) -> None:
        """Record one organism's genome for pattern detection."""
        params = getattr(genome, "params", None) or getattr(genome, "to_dict", dict)()
        if not isinstance(params, dict):
            return
        numeric = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
        if not numeric:
            return
        if not self._param_keys:
            self._param_keys = sorted(numeric.keys())
        sig = self._signature(numeric)
        # Find an existing pattern or create a new one.
        for p in self._patterns:
            if p.signature == sig:
                p.count += 1
                p.avg_fitness = (p.avg_fitness * (p.count - 1) + fitness) / p.count
                return
        self._patterns.append(RepeatPattern(sig, 1, fitness))
        # Bound the pattern list.
        if len(self._patterns) > self.capacity:
            self._patterns.sort(key=lambda p: p.count, reverse=True)
            self._patterns = self._patterns[: self.capacity]

    def recurring(self) -> list[RepeatPattern]:
        """Return patterns that appear at least min_count times."""
        return [p for p in self._patterns if p.count >= self.min_count]

    def repeat_bias(self, params: dict[str, Any]) -> dict[str, float]:
        """Return per-parameter multipliers: >1 means the value belongs to a
        recurring pattern (reinforce), <1 means it's an outlier (explore)."""
        sig = self._signature(params)
        recurring_sigs = {p.signature for p in self.recurring()}
        if sig in recurring_sigs:
            # This genome matches a recurring pattern -- reinforce it.
            return {k: 0.6 for k, v in params.items() if isinstance(v, (int, float))}
        # No match: encourage exploration.
        return {k: 1.4 for k, v in params.items() if isinstance(v, (int, float))}

    def to_dict(self) -> dict[str, Any]:
        return {
            "bands": self.bands,
            "min_count": self.min_count,
            "capacity": self.capacity,
            "patterns": [
                {"signature": list(p.signature), "count": p.count, "avg_fitness": p.avg_fitness}
                for p in self._patterns
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VisualRepeatDetector:
        det = cls(
            bands=d.get("bands", 8),
            min_count=d.get("min_count", 3),
            capacity=d.get("capacity", 50),
        )
        for p in d.get("patterns", []):
            sig = tuple(tuple(e) for e in p.get("signature", []))
            det._patterns.append(
                RepeatPattern(sig, p.get("count", 0), float(p.get("avg_fitness", 0.0)))
            )
        return det

    def clone(self) -> VisualRepeatDetector:
        d = VisualRepeatDetector(self.bands, self.min_count, self.capacity)
        d._patterns = [RepeatPattern(p.signature, p.count, p.avg_fitness) for p in self._patterns]
        d._param_keys = list(self._param_keys)
        return d


# ---------------------------------------------------------------------------
# 32c. ReactionScorer
# ---------------------------------------------------------------------------
# Measures how well a visual "reacts" to audio changes. An organism that
# produces dramatically different visuals when the audio shifts is
# considered highly reactive. The scorer tracks per-organism visual
# change magnitude against audio delta and produces a reaction score.


@dataclass
class ReactionSample:
    """One observation: audio state -> visual params -> fitness."""

    audio_vec: list[float]  # numeric audio features
    params_vec: list[float]  # numeric genome params
    fitness: float
    audio_delta: float = 0.0
    visual_delta: float = 0.0


class ReactionScorer:
    """Scores how well visuals react to audio and user input changes.

    Uses L1 distance on numeric feature vectors instead of hash differences,
    so nearby continuous states are properly measured as close.
    """

    def __init__(self, window: int = 64, min_delta: float = REACTION_MIN_DELTA):
        self.window = max(8, int(window))
        self.min_delta = max(0.0, float(min_delta))
        self._samples: list[ReactionSample] = []
        # Per-param reactivity: how much does this param change when audio changes?
        self._param_reactivity: dict[str, float] = {}
        # Running average of audio delta and visual delta.
        self._avg_audio_delta = 0.0
        self._avg_visual_delta = 0.0

    @staticmethod
    def _extract_numeric_vec(d: dict[str, Any]) -> list[float]:
        """Extract numeric values in sorted key order for stable vector."""
        return [float(v) for k, v in sorted(d.items()) if isinstance(v, (int, float))]

    @staticmethod
    def _l1_dist(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        return sum(abs(a[i] - b[i]) for i in range(n)) / max(1, n)

    def record(self, params: dict[str, Any], audio: dict[str, float], fitness: float) -> None:
        """Record the current state. Called each frame or on audio change."""
        audio_vec = self._extract_numeric_vec(audio)
        params_vec = self._extract_numeric_vec(params)
        audio_delta = 0.0
        visual_delta = 0.0
        if self._samples:
            last = self._samples[-1]
            audio_delta = self._l1_dist(audio_vec, last.audio_vec)
            visual_delta = self._l1_dist(params_vec, last.params_vec)
        sample = ReactionSample(audio_vec, params_vec, fitness, audio_delta, visual_delta)
        self._samples.append(sample)
        if len(self._samples) > self.window:
            self._samples.pop(0)
        # Update running averages.
        if self._samples:
            self._avg_audio_delta = sum(s.audio_delta for s in self._samples) / len(self._samples)
            self._avg_visual_delta = sum(s.visual_delta for s in self._samples) / len(
                self._samples
            )

    def reaction_score(self, params: dict[str, Any], audio: dict[str, float]) -> float:
        """Return a 0..1 score: how reactive this visual is to the current audio."""
        if self._avg_audio_delta < self.min_delta:
            return 0.5  # No audio change to react to.
        # Higher visual delta per audio delta = more reactive.
        visual_delta = self._avg_visual_delta
        ratio = visual_delta / max(self.min_delta, self._avg_audio_delta)
        return min(1.0, ratio / REACTION_SCORE_SCALE)

    def reactivity_bias(self, params: dict[str, Any], audio: dict[str, float]) -> dict[str, float]:
        """Return per-parameter multipliers: params that changed a lot when
        audio changed get lower mutation (they're already reactive), while
        static params get higher mutation (they need to become reactive)."""
        score = self.reaction_score(params, audio)
        out: dict[str, float] = {}
        for k, v in params.items():
            if not isinstance(v, (int, float)):
                continue
            # High reactivity -> preserve (0.6). Low reactivity -> push (1.4).
            out[k] = PREFERENCE_BIAS_EXPLORE - score * PREFERENCE_BIAS_SCALE
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "min_delta": self.min_delta,
            "avg_audio_delta": self._avg_audio_delta,
            "avg_visual_delta": self._avg_visual_delta,
            "param_reactivity": dict(self._param_reactivity),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReactionScorer:
        s = ReactionScorer(
            window=d.get("window", 64), min_delta=d.get("min_delta", REACTION_MIN_DELTA)
        )
        s._avg_audio_delta = float(d.get("avg_audio_delta", 0.0))
        s._avg_visual_delta = float(d.get("avg_visual_delta", 0.0))
        s._param_reactivity = dict(d.get("param_reactivity", {}))
        return s

    def clone(self) -> ReactionScorer:
        sc = ReactionScorer(self.window, self.min_delta)
        sc._samples = [
            ReactionSample(
                list(s.audio_vec), list(s.params_vec), s.fitness, s.audio_delta, s.visual_delta
            )
            for s in self._samples
        ]
        sc._param_reactivity = dict(self._param_reactivity)
        sc._avg_audio_delta = self._avg_audio_delta
        sc._avg_visual_delta = self._avg_visual_delta
        return sc


# ---------------------------------------------------------------------------
# 32d. UserPreferenceModel
# ---------------------------------------------------------------------------
# Learns what visuals the user prefers from implicit signals:
#   - dwell time (how long the user watches a particular organism)
#   - save/bookmark actions
#   - explicit rating if available
#   - which organisms the user returns to after switching away
#
# The model maintains a preference vector over genome parameters and
# produces a "preference score" for any genome, which the predictive
# mutation model uses to bias toward user-liked visuals.


@dataclass
class PreferenceSignal:
    """One implicit or explicit signal about user preference.

    Stores the actual numeric parameter vector for similarity-based
    per-parameter learning, plus a hash for exact-match lookup.
    """

    params_vec: list[float]  # numeric params in sorted key order
    param_keys: list[str]  # corresponding parameter names
    signal_type: str  # "dwell", "save", "rate", "return"
    strength: float = 1.0
    timestamp: float = 0.0
    params_hash: int = 0  # for exact-match rating lookup


class UserPreferenceModel:
    """Learns user preferences from implicit and explicit signals.

    Uses similarity-weighted k-nearest-neighbour inference over stored
    signals to produce per-parameter preference estimates.
    """

    def __init__(
        self,
        decay: float = PREFERENCE_DECAY,
        min_signals: int = PREFERENCE_MIN_SIGNALS,
        k: int = PREFERENCE_K_NEIGHBORS,
    ):
        self.decay = max(0.5, min(1.0, float(decay)))
        self.min_signals = max(1, int(min_signals))
        self.k = max(1, int(k))
        self._signals: list[PreferenceSignal] = []
        # Explicit ratings: params_hash -> rating 0..1.
        self._ratings: dict[int, float] = {}

    @staticmethod
    def _extract_numeric_vec(params: dict[str, Any]) -> tuple[list[str], list[float]]:
        items = [(k, float(v)) for k, v in sorted(params.items()) if isinstance(v, (int, float))]
        return [k for k, _ in items], [v for _, v in items]

    @staticmethod
    def _l1_dist(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        return sum(abs(a[i] - b[i]) for i in range(n)) / max(1, n)

    def _hash_params(self, params: dict[str, Any]) -> int:
        h = 0
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, (int, float)):
                h = (h * 31 + hash(round(float(v), 4))) & 0xFFFFFFFF
        return h

    def _add_signal(self, params: dict[str, Any], signal_type: str, strength: float) -> None:
        keys, vec = self._extract_numeric_vec(params)
        h = self._hash_params(params)
        self._signals.append(PreferenceSignal(vec, keys, signal_type, strength, time.time(), h))
        self._prune()

    def signal_dwell(
        self, params: dict[str, Any], duration: float, max_duration: float = 30.0
    ) -> None:
        """User watched this visual for ``duration`` seconds."""
        strength = min(1.0, duration / max_duration)
        self._add_signal(params, "dwell", strength)

    def signal_save(self, params: dict[str, Any]) -> None:
        """User saved/bookmarked this visual."""
        self._add_signal(params, "save", 1.0)

    def signal_rate(self, params: dict[str, Any], rating: float) -> None:
        """User gave an explicit rating (0..1)."""
        h = self._hash_params(params)
        self._ratings[h] = max(0.0, min(1.0, float(rating)))
        self._add_signal(params, "rate", max(0.0, min(1.0, float(rating))))

    def signal_return(self, params: dict[str, Any]) -> None:
        """User returned to this visual after switching away."""
        self._add_signal(params, "return", 0.8)

    def _prune(self) -> None:
        """Keep only recent signals to prevent unbounded growth."""
        now = time.time()
        self._signals = [
            s for s in self._signals if now - s.timestamp < PREFERENCE_MAX_AGE_SECONDS
        ]
        if len(self._signals) > PREFERENCE_MAX_SIGNALS:
            self._signals = self._signals[-PREFERENCE_MAX_SIGNALS:]

    def _infer_per_param(self, params: dict[str, Any]) -> dict[str, float]:
        """K-NN inference of per-parameter preference from stored signals."""
        _, vec = self._extract_numeric_vec(params)
        if not vec or len(self._signals) < self.min_signals:
            return {}

        # Find k nearest neighbours by L1 distance on param vectors.
        now = time.time()
        scored: list[tuple[float, PreferenceSignal]] = []
        for s in self._signals:
            # Only compare against signals with compatible param keys.
            if len(s.params_vec) != len(vec):
                continue
            age = now - s.timestamp
            decayed = s.strength * (self.decay ** (age / 60.0))
            dist = self._l1_dist(vec, s.params_vec)
            # Weight by decayed strength and inverse distance.
            weight = decayed / (1.0 + dist)
            scored.append((weight, s))

        if not scored:
            return {}

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = scored[: self.k]

        # Aggregate per-parameter: for each param, weight by neighbour similarity.
        param_scores: dict[str, list[float]] = {}
        for weight, s in top_k:
            for i, pk in enumerate(s.param_keys):
                if i < len(s.params_vec):
                    param_scores.setdefault(pk, []).append(s.params_vec[i] * weight)

        # Normalise: convert weighted average param values to 0..1 preference.
        out: dict[str, float] = {}
        for pk, vals in param_scores.items():
            wsum = sum(abs(v) for v in vals)  # rough normalisation
            if wsum > 0:
                # Map to 0..1: this is a heuristic; could use PARAM_BOUNDS.
                out[pk] = min(1.0, max(0.0, sum(vals) / wsum))
        return out

    def preference_score(self, params: dict[str, Any]) -> float:
        """Return 0..1 how much the user prefers this visual."""
        h = self._hash_params(params)
        if h in self._ratings:
            return self._ratings[h]
        if len(self._signals) < self.min_signals:
            return 0.5
        # Average of per-param preferences as overall score.
        per_param = self._infer_per_param(params)
        if not per_param:
            return 0.5
        return sum(per_param.values()) / len(per_param)

    def preference_bias(self, params: dict[str, Any]) -> dict[str, float]:
        """Return per-parameter multipliers: preferred visuals get lower
        mutation (preserve), disliked visuals get higher mutation (explore)."""
        per_param = self._infer_per_param(params)
        out: dict[str, float] = {}
        for k, v in params.items():
            if not isinstance(v, (int, float)):
                continue
            # High preference -> preserve (0.6). Low preference -> explore (1.4).
            pref = per_param.get(k, 0.5)
            out[k] = PREFERENCE_BIAS_EXPLORE - pref * PREFERENCE_BIAS_SCALE
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "min_signals": self.min_signals,
            "k": self.k,
            "signals": [
                {
                    "params_vec": s.params_vec,
                    "param_keys": s.param_keys,
                    "signal_type": s.signal_type,
                    "strength": s.strength,
                    "timestamp": s.timestamp,
                    "params_hash": s.params_hash,
                }
                for s in self._signals
            ],
            "ratings": {str(k): v for k, v in self._ratings.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UserPreferenceModel:
        m = UserPreferenceModel(
            decay=d.get("decay", PREFERENCE_DECAY),
            min_signals=d.get("min_signals", PREFERENCE_MIN_SIGNALS),
            k=d.get("k", PREFERENCE_K_NEIGHBORS),
        )
        for s in d.get("signals", []):
            m._signals.append(
                PreferenceSignal(
                    s.get("params_vec", []),
                    s.get("param_keys", []),
                    s.get("signal_type", "dwell"),
                    float(s.get("strength", 1.0)),
                    float(s.get("timestamp", 0.0)),
                    int(s.get("params_hash", 0)),
                )
            )
        for k_str, v in d.get("ratings", {}).items():
            try:
                m._ratings[int(k_str)] = float(v)
            except (ValueError, TypeError):
                pass
        return m

    def clone(self) -> UserPreferenceModel:
        m = UserPreferenceModel(self.decay, self.min_signals, self.k)
        m._signals = [
            PreferenceSignal(
                list(s.params_vec),
                list(s.param_keys),
                s.signal_type,
                s.strength,
                s.timestamp,
                s.params_hash,
            )
            for s in self._signals
        ]
        m._ratings = dict(self._ratings)
        return m


# ---------------------------------------------------------------------------
# 32e. PredictiveMutationModel
# ---------------------------------------------------------------------------
# Fuses the four learned signals into per-parameter mutation multipliers
# that bias the continuous mutator *before* rendering. The model produces
# a multiplier per genome parameter: values < 1 reduce mutation (preserve
# what works), values > 1 increase mutation (explore what doesn't).
#
# The fusion is a weighted average of the four biases, with weights that
# can be evolved by the self-modifying engine or tuned by the user.


@dataclass
class MutationBias:
    """Per-parameter mutation multiplier for one genome."""

    param: str
    bias: float = 1.0  # 1.0 = neutral, <1 = preserve, >1 = explore
    source: str = "neutral"  # Which signal dominated this bias.


class PredictiveMutationModel:
    """AI-guided predictive model that biases mutation before rendering.

    The model learns from four signals and produces per-parameter mutation
    multipliers. These are applied to the ``ContinuousMutator`` so that
    the next frame's mutation is biased toward visuals that survive, repeat,
    react well, or are user-preferred -- without ever leaving the bounded,
    curated mutation operator set.
    """

    def __init__(
        self,
        enable_survival: bool = True,
        enable_repeat: bool = True,
        enable_reaction: bool = True,
        enable_preference: bool = True,
        survival_weight: float = 0.25,
        repeat_weight: float = 0.25,
        reaction_weight: float = 0.25,
        preference_weight: float = 0.25,
    ):
        self.enable_survival = bool(enable_survival)
        self.enable_repeat = bool(enable_repeat)
        self.enable_reaction = bool(enable_reaction)
        self.enable_preference = bool(enable_preference)
        # Normalised weights (sum to 1).
        total = survival_weight + repeat_weight + reaction_weight + preference_weight
        if total > 0:
            self.survival_weight = survival_weight / total
            self.repeat_weight = repeat_weight / total
            self.reaction_weight = reaction_weight / total
            self.preference_weight = preference_weight / total
        else:
            self.survival_weight = self.repeat_weight = self.reaction_weight = (
                self.preference_weight
            ) = 0.25

        self.survival = VisualSurvivalTracker()
        self.repeat = VisualRepeatDetector()
        self.reaction = ReactionScorer()
        self.preference = UserPreferenceModel()

        # Cached last bias so we don't recompute every frame unnecessarily.
        self._last_bias: dict[str, float] = {}
        self._last_params_hash: int = 0
        self._bias_stale: bool = True

    def observe_survival(self, genome: Any, fitness: float) -> None:
        """Record a genome as a survivor for the survival tracker."""
        self.survival.record_survivor(genome, fitness)

    def observe_repeat(self, genome: Any, fitness: float) -> None:
        """Record a genome for pattern repetition detection."""
        self.repeat.observe(genome, fitness)

    def observe_reaction(
        self, params: dict[str, Any], audio: dict[str, float], fitness: float
    ) -> None:
        """Record the current visual state for reaction scoring."""
        self.reaction.record(params, audio, fitness)

    def observe_preference(
        self, params: dict[str, Any], signal_type: str, strength: float = 1.0
    ) -> None:
        """Record a user preference signal."""
        if signal_type == "dwell":
            self.preference.signal_dwell(params, strength)
        elif signal_type == "save":
            self.preference.signal_save(params)
        elif signal_type == "rate":
            self.preference.signal_rate(params, strength)
        elif signal_type == "return":
            self.preference.signal_return(params)

    def _compute_bias(self, params: dict[str, Any], audio: dict[str, float]) -> dict[str, float]:
        """Fuse all four signals into a single per-parameter bias dict."""
        # Start with neutral bias of 1.0 for every numeric param.
        bias: dict[str, float] = {}
        for k, v in params.items():
            if not isinstance(v, (int, float)):
                continue
            bias[k] = 1.0

        # Fuse survival bias.
        if self.enable_survival:
            surv = self.survival.survival_bias(params)
            for k, v in bias.items():
                if k in surv:
                    bias[k] = v * (0.5 + 0.5 * surv[k]) * self.survival_weight + v * (
                        1.0 - self.survival_weight
                    )

        # Fuse repeat bias.
        if self.enable_repeat:
            rep = self.repeat.repeat_bias(params)
            for k, v in bias.items():
                if k in rep:
                    bias[k] = v * (0.5 + 0.5 * rep[k]) * self.repeat_weight + v * (
                        1.0 - self.repeat_weight
                    )

        # Fuse reaction bias.
        if self.enable_reaction:
            rej = self.reaction.reactivity_bias(params, audio)
            for k, v in bias.items():
                if k in rej:
                    bias[k] = v * (0.5 + 0.5 * rej[k]) * self.reaction_weight + v * (
                        1.0 - self.reaction_weight
                    )

        # Fuse preference bias.
        if self.enable_preference:
            pref = self.preference.preference_bias(params)
            for k, v in bias.items():
                if k in pref:
                    bias[k] = v * (0.5 + 0.5 * pref[k]) * self.preference_weight + v * (
                        1.0 - self.preference_weight
                    )

        # Clamp biases to a safe range so mutation is never
        # completely suppressed or wildly amplified.
        for k, v in bias.items():
            bias[k] = max(BIAS_CLAMP_MIN, min(BIAS_CLAMP_MAX, v))

        return bias

    def mutation_multiplier(
        self, params: dict[str, Any], audio: dict[str, float], param: str
    ) -> float:
        """Return the mutation multiplier for a single parameter.

        Values < 1.0 reduce mutation (preserve), > 1.0 increase it (explore).
        The result is cached for the same params hash to avoid recomputation
        every frame.
        """
        params_hash = hash(
            tuple(
                sorted(
                    (k, round(float(v), 4))
                    for k, v in params.items()
                    if isinstance(v, (int, float))
                )
            )
        )
        if params_hash != self._last_params_hash:
            self._last_bias = self._compute_bias(params, audio)
            self._last_params_hash = params_hash
            self._bias_stale = False
        if self._bias_stale:
            self._last_bias = self._compute_bias(params, audio)
            self._bias_stale = False
        return self._last_bias.get(param, 1.0)

    def bias_continuous_mutator(
        self, mutator: ContinuousMutator, params: dict[str, Any], audio: dict[str, float]
    ) -> None:
        """Apply the predictive bias to a ContinuousMutator's per-frame step.

        This is the main integration point: call this before the mutator
        steps, and it will adjust the mutation amount per parameter based
        on the learned model.
        """
        apply_predictive_bias(mutator, params, audio, self)

    def step(self, dt: float, audio: dict[str, float]) -> None:
        """Advance the model's internal state (decay signals, update weights)."""
        self.survival.next_generation()
        # Decay preference signals over time (already handled by _prune).
        # Update reaction scorer averages.
        self.reaction._avg_audio_delta *= 0.95
        self.reaction._avg_visual_delta *= 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_survival": self.enable_survival,
            "enable_repeat": self.enable_repeat,
            "enable_reaction": self.enable_reaction,
            "enable_preference": self.enable_preference,
            "survival_weight": self.survival_weight,
            "repeat_weight": self.repeat_weight,
            "reaction_weight": self.reaction_weight,
            "preference_weight": self.preference_weight,
            "survival": self.survival.to_dict(),
            "repeat": self.repeat.to_dict(),
            "reaction": self.reaction.to_dict(),
            "preference": self.preference.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PredictiveMutationModel:
        m = cls(
            enable_survival=d.get("enable_survival", True),
            enable_repeat=d.get("enable_repeat", True),
            enable_reaction=d.get("enable_reaction", True),
            enable_preference=d.get("enable_preference", True),
            survival_weight=d.get("survival_weight", 0.25),
            repeat_weight=d.get("repeat_weight", 0.25),
            reaction_weight=d.get("reaction_weight", 0.25),
            preference_weight=d.get("preference_weight", 0.25),
        )
        if d.get("survival"):
            m.survival = VisualSurvivalTracker.from_dict(d["survival"])
        if d.get("repeat"):
            m.repeat = VisualRepeatDetector.from_dict(d["repeat"])
        if d.get("reaction"):
            m.reaction = ReactionScorer.from_dict(d["reaction"])
        if d.get("preference"):
            m.preference = UserPreferenceModel.from_dict(d["preference"])
        return m

    def clone(self) -> PredictiveMutationModel:
        m = PredictiveMutationModel(
            enable_survival=self.enable_survival,
            enable_repeat=self.enable_repeat,
            enable_reaction=self.enable_reaction,
            enable_preference=self.enable_preference,
            survival_weight=self.survival_weight,
            repeat_weight=self.repeat_weight,
            reaction_weight=self.reaction_weight,
            preference_weight=self.preference_weight,
        )
        m.survival = self.survival.clone()
        m.repeat = self.repeat.clone()
        m.reaction = self.reaction.clone()
        m.preference = self.preference.clone()
        return m


# ---------------------------------------------------------------------------
# 32f. Integration helpers
# ---------------------------------------------------------------------------
# These functions connect the AI-guided model to the existing
# StageOrganism and ContinuousMutator so the predictive bias is applied
# automatically each frame.


def apply_predictive_bias(
    mutator: ContinuousMutator,
    params: dict[str, Any],
    audio: dict[str, float],
    model: PredictiveMutationModel | None = None,
) -> None:
    """Apply the predictive mutation bias to a ContinuousMutator before it steps.

    If ``model`` is None or has no learned signals, the mutator proceeds
    with its default per-frame drift (no bias). Otherwise, the mutator's
    per-frame amount is scaled per-parameter by the model's bias, so
    parameters that belong to survivors, recurring patterns, reactive
    visuals, or user-preferred visuals are mutated less (preserved) while
    others are explored more aggressively.
    """
    if model is None:
        return
    bias = model._compute_bias(params, audio)
    # Store the bias on the mutator so ContinuousMutator.step() can use it.
    mutator._param_bias = bias


def record_ai_signals(
    model: PredictiveMutationModel,
    genome: Any,
    params: dict[str, Any],
    audio: dict[str, float],
    fitness: float,
    survived: bool,
) -> None:
    """Record all four AI signals for one organism in one step."""
    if model is None:
        return
    model.observe_survival(genome, fitness)
    model.observe_repeat(genome, fitness)
    model.observe_reaction(params, audio, fitness)
    if survived:
        model.observe_preference(params, "dwell", strength=min(1.0, fitness))
