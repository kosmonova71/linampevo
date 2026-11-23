"""Visual genome and gene dataclass re-exports.

VisualGenome is the numeric genome subclassing GenomeBase. The gene dataclasses
(CameraGenes, FieldGenes, etc.) are re-exported from evo_renderer.genes.
"""

from __future__ import annotations

import copy
import logging
import random
import sys

from evo.genome_base import GenomeBase
from evo_bio.gene_usage_tracking import GeneUsageTracker
from evo_bio.pareto import Objectives
from evo_renderer.genes import (
    BehaviourGenes,
    CameraGenes,
    ColourGenes,
    FeedbackGenes,
    FieldGenes,
    LightingGenes,
    ParticleGenes,
)
from evo_visualizer.constants import MAX_MUTATION_HISTORY

logger = logging.getLogger(__name__)


def _bound_keys(module: object) -> list[str]:
    """Return the gene keys bounded by the module class's ``BOUNDS`` table.

    ``BOUNDS`` is a *module-level* constant in ``evo_renderer.genes.*``, not a
    class attribute, so it must be looked up in the defining module.  Falling
    back to the instance's own ``to_dict()`` keeps gene-usage weighting working
    for any module that follows the same contract.
    """
    cls = type(module)
    bounds = getattr(sys.modules.get(cls.__module__), "BOUNDS", None)
    if isinstance(bounds, dict) and bounds:
        return list(bounds.keys())
    to_dict = getattr(module, "to_dict", None)
    if callable(to_dict):
        try:
            return list(to_dict().keys())
        except Exception:
            logger.debug("to_dict() failed for %s", cls.__name__, exc_info=True)
    return []


__all__ = [
    "MAX_MUTATION_HISTORY",
    "BehaviourGenes",
    "CameraGenes",
    "ColourGenes",
    "FeedbackGenes",
    "FieldGenes",
    "GeneUsageTracker",
    "LightingGenes",
    "Objectives",
    "ParticleGenes",
    "VisualGenome",
]


class VisualGenome(GenomeBase):
    """Represents a visual organism's genetic parameters.

    Tracks mutation acceptance history to adaptively adjust mutation strength,
    """

    __slots__ = (
        "_best_fitness_history",
        "_gene_usage_tracker",
        "_mutated_genes_last",
        "_recent_accepts",
        "_stagnation_counter",
        "_strength",
        "_window_size",
        "age",
        "behaviour",
        "camera",
        "colour",
        "entropy",
        "feedback",
        "field",
        "generation",
        "lifespan",
        "lighting",
        "lineage",
        "mutations",
        "objectives",
        "parent_id",
        "particle",
    )

    def __init__(
        self,
        strength: float = 0.1,
        lineage: str | None = None,
        generation: int = 0,
        parent_id: str | None = None,
    ):
        """Initialize a genome with default parameters.

        Args:
            strength: Initial mutation step size
            lineage: Lineage identifier for memory DNA
            generation: Generation number
            parent_id: Parent genome identifier
        """
        self._recent_accepts: list[int] = []
        self._window_size = 8
        self._strength = float(strength)
        self.lineage = lineage or self._generate_id()
        self.generation = generation
        self.objectives = Objectives()
        self.parent_id = parent_id
        self.mutations: list[str] = []
        self.age = 0
        self.lifespan = 500
        self.entropy = 0.5
        self._best_fitness_history: list[float] = []
        self._stagnation_counter = 0
        self._gene_usage_tracker = GeneUsageTracker()
        self._mutated_genes_last: list[str] = []

        self.camera = CameraGenes()
        self.field = FieldGenes()
        self.colour = ColourGenes()
        self.particle = ParticleGenes()
        self.feedback = FeedbackGenes()
        self.lighting = LightingGenes()
        self.behaviour = BehaviourGenes()

        defaults = {
            "warp": 0.5,
            "noise_scale": 3.0,
            "feedback": 0.85,
            "rotation": 0.0,
            "symmetry": 4,
            "colour_shift": 0.0,
            "field_type": 0,
            "flow_speed": 1.0,
            "curl_strength": 0.5,
            "domain_warp": 0.5,
            "fractal_layers": 4,
            "turbulence": 1.0,
            "vortex": 0.0,
            "mirror": 0,
            "particle_density": 1.0,
            "glow": 0.5,
            "contrast": 1.0,
            "colour_palette": 0.0,
            "feedback_mode": 0,
            "dimension": 0,
            "camera_speed": 1.0,
            "camera_orbit": 1.0,
            "camera_depth": 3.0,
            "camera_roll": 0.0,
            "camera_fov": 1.5,
            "depth_scale": 1.0,
            "ray_steps": 32,
            "fog_density": 1.0,
            "camera_focus": 0.5,
            "camera_aperture": 0.2,
            "camera_exposure": 0.5,
            "camera_autoexp": 0.4,
            "camera_shake": 0.15,
            "camera_shake_freq": 0.5,
            "camera_tracking": 0.5,
            "camera_framing": 0.5,
            "camera_headroom": 0.5,
            "camera_school": 0,
            "camera_shot_length": 1.0,
            "camera_transition": 0.3,
            "camera_crane": 0.0,
            "camera_parallax": 0.3,
            "camera_drift": 0.0,
            "camera_focus_pull": 0.3,
            "camera_handheld": 0.2,
            "camera_dolly": 0.5,
            "camera_pan": 0.3,
            "camera_tilt": 0.3,
        }
        for module in (
            self.camera,
            self.field,
            self.colour,
            self.particle,
            self.feedback,
            self.lighting,
            self.behaviour,
        ):
            module.from_dict(defaults)

    @property
    def params(self) -> dict[str, float | int]:
        """Backward-compatible flat parameter view over sub-modules."""
        result: dict[str, float | int] = {}
        for module in (
            self.camera,
            self.field,
            self.colour,
            self.particle,
            self.feedback,
            self.lighting,
            self.behaviour,
        ):
            result.update(module.to_dict())
        return result

    @params.setter
    def params(self, value: dict[str, float | int]) -> None:
        for module in (
            self.camera,
            self.field,
            self.colour,
            self.particle,
            self.feedback,
            self.lighting,
            self.behaviour,
        ):
            module.from_dict(value)

    def _generate_id(self) -> str:
        """Generate a unique genome identifier."""
        import uuid

        return str(uuid.uuid4())[:8]

    @property
    def fitness(self) -> float:
        return self.objectives.scalar()

    @fitness.setter
    def fitness(self, value: float) -> None:
        self.objectives = Objectives(
            beauty=float(value) * 0.4,
            novelty=float(value) * 0.15,
            audio_sync=float(value) * 0.1,
            performance=1.0,
            species_health=float(value) * 0.1,
            motion=float(value) * 0.1,
            complexity=float(value) * 0.1,
            memory=float(value) * 0.05,
        )

    def mutate(self, strength: float | None = None, large: bool = False) -> None:
        """Mutate a random parameter by the current strength.

        Entropy drives the mutation regime:
        * high entropy (entropy >= entropy_high) -> ``invent`` mode:
          strength is amplified so the genome explores distant regions
          of the parameter space.
        * low entropy (entropy <= entropy_low) -> ``refine`` mode:
          strength is damped so the genome fine-tunes around the
          current good region.
        * mid entropy -> standard strength.

        Args:
            strength: Optional override for mutation strength
            large: When True, apply a stronger / multi-step mutation used when
                an organism outlives its lifespan to force succession.
        """
        if strength is not None:
            self._strength = float(strength)
        effective = self._strength * self._entropy_factor() * self._age_mutation_factor()
        if large:
            effective = min(effective * 2.0 + 0.1, 1.0)
        self._mutate_once(effective)
        if large:
            self._mutate_once(effective)
        if not self._mutated_genes_last:
            self._mutated_genes_last = []

    def _entropy_factor(self) -> float:
        """Return a multiplier for mutation strength based on entropy.

        High entropy amplifies (invent), low entropy dampens (refine).
        """
        hi = 0.7
        lo = 0.2
        if self.entropy >= hi:
            return 1.0 + (self.entropy - hi) / max(1.0 - hi, 1e-6) * 2.0
        if self.entropy <= lo:
            return 1.0 - (lo - self.entropy) / max(lo, 1e-6) * 0.7
        return 1.0

    def _age_mutation_factor(self) -> float:
        """Return a multiplier for mutation strength based on organism age.

        Young organisms explore more aggressively (factor > 1) to escape
        their initial basin; as they mature the factor decays toward unity so
        refinement is driven by :meth:`_entropy_factor` rather than age.
        Falls back to 1.0 when age/lifespan attributes are unavailable.
        """
        age = float(getattr(self, "age", 0))
        lifespan = max(1.0, float(getattr(self, "lifespan", 500)))
        return 1.5 - 0.5 * min(1.0, age / lifespan)

    def compute_entropy(self) -> float:
        """Compute this genome's entropy score in [0, 1].

        Entropy is derived from three signals:
        1. Mutation diversity — how many distinct parameter classes
           have been mutated recently (high = broad exploration).
        2. Fitness variance over the recent history (high = volatile
           exploration, low = convergence).
        3. Parameter distance from the initial defaults — a genome
           that has drifted far from its origin is more "invented".

        The result is clamped to [0, 1].
        """
        # 1. Mutation diversity: count unique mutation descriptions
        #    in the recent window as a proxy for breadth.
        recent = self.mutations[-MAX_MUTATION_HISTORY:] if self.mutations else []
        if recent:
            unique_classes = set()
            for m in recent:
                cls = m.split(".")[0] if "." in m else m
                unique_classes.add(cls)
            diversity = len(unique_classes) / max(len(self.mutations), 1)
        else:
            diversity = 0.0

        # 2. Fitness variance: how much the fitness has fluctuated
        #    in the recent history.
        if len(self._best_fitness_history) >= 4:
            window = self._best_fitness_history[-MAX_MUTATION_HISTORY:]
            mean_f = sum(window) / len(window)
            variance = sum((f - mean_f) ** 2 for f in window) / len(window)
            fitness_volatility = min(variance * 10.0, 1.0)
        else:
            fitness_volatility = 0.0

        # 3. Parameter drift from defaults — how far has the genome
        #    moved from its initial state.  We approximate this by
        #    the number of unique module mutation classes divided by
        #    the total number of modules.
        param_drift = len(set(recent)) / max(MAX_MUTATION_HISTORY, 1)

        # Combine the three signals with weights.
        entropy = 0.40 * diversity + 0.30 * fitness_volatility + 0.30 * param_drift
        self.entropy = max(0.0, min(1.0, entropy))
        return self.entropy

    def _detect_stagnation(self, fitness: float) -> bool:
        """Return True if fitness has stagnated (no improvement).

        Records the current fitness and checks whether the best
        fitness in the history window has improved.  If not, the
        stagnation counter increments; when it exceeds the window
        threshold the method returns True (and entropy is raised).

        Args:
            fitness: Current scalar fitness value.

        Returns:
            True if stagnation is detected.
        """
        prev_best = max(self._best_fitness_history) if self._best_fitness_history else fitness
        self._best_fitness_history.append(fitness)
        if len(self._best_fitness_history) > MAX_MUTATION_HISTORY:
            self._best_fitness_history.pop(0)

        if len(self._best_fitness_history) < 4:
            return False

        improvement = (fitness - prev_best) / max(abs(prev_best), 1e-6)

        if improvement > 1e-4:
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

        return self._stagnation_counter >= 30

    def raise_entropy(self, amount: float = 0.15) -> None:
        """Increase entropy to escape a local minimum.

        Called when stagnation is detected.  Higher entropy drives
        the genome into an inventive mutation regime.
        """
        self.entropy = min(1.0, self.entropy + amount)
        self._stagnation_counter = 0

    def _mutate_once(self, strength: float | None = None) -> None:
        """Single mutation pass using weighted sub-module selection.

        Gene usage tracking biases module selection:
        - Modules with mostly dormant genes get lower weight
        - Modules with protected genes get higher weight

        Args:
            strength: Override for the mutation step size.  When
                provided it takes priority over ``self._strength``.
        """
        s = float(strength) if strength is not None else self._strength
        modules = [
            (self.camera, 3),
            (self.field, 3),
            (self.colour, 3),
            (self.particle, 2),
            (self.feedback, 3),
            (self.lighting, 2),
            (self.behaviour, 2),
        ]
        tracker = getattr(self, "_gene_usage_tracker", None)
        if tracker is not None:
            weights = []
            for module, base_w in modules:
                mod_name = type(module).__name__.replace("Genes", "").lower()
                avg_w = tracker.module_weight(mod_name, _bound_keys(module))
                weights.append(base_w * avg_w)
        else:
            weights = [w for _, w in modules]
        if not any(w > 0.0 for w in weights):
            # A degenerate tracker must never make selection impossible.
            weights = [w for _, w in modules]
        module, _ = random.choices(modules, weights=weights)[0]
        mutated_key = module.mutate(strength=s)
        mutation_desc = f"{type(module).__name__}.mutate"
        if mutated_key:
            mutation_desc = f"{type(module).__name__}.{mutated_key}"
            mod_name = type(module).__name__.replace("Genes", "").lower()
            self._mutated_genes_last.append(f"{mod_name}.{mutated_key}")
            if len(self._mutated_genes_last) > 32:
                self._mutated_genes_last = self._mutated_genes_last[-32:]
        self.mutations.append(mutation_desc)
        if len(self.mutations) > MAX_MUTATION_HISTORY:
            del self.mutations[:-MAX_MUTATION_HISTORY]
        self.clamp()

    def record_accept(self, accepted: bool) -> None:
        """Record whether a mutation was accepted and recompute entropy.

        After each mutation the genome's entropy is recalculated so
        the invent/refine regime stays in sync with the latest
        mutation history.  Stagnation is also detected here: when
        the fitness has not improved over the history window the
        entropy is raised automatically to escape local minima.

        Args:
            accepted: Whether the mutation was accepted
        """
        self._recent_accepts.append(1 if accepted else 0)
        if len(self._recent_accepts) > self._window_size:
            self._recent_accepts.pop(0)
        self.compute_entropy()

    def adapt_strength(self, lo: float, hi: float) -> None:
        """Adaptively adjust mutation strength based on acceptance rate
        and entropy regime.

        When entropy is high (invent mode) strength is pushed toward
        the upper bound to encourage exploration.  When entropy is
        low (refine mode) strength is pulled toward the lower bound
        to avoid disrupting converged parameters.  The acceptance
        rate then fine-tunes within that range.

        Args:
            lo: Minimum mutation strength bound
            hi: Maximum mutation strength bound
        """
        if not self._recent_accepts:
            return
        rate = sum(self._recent_accepts) / len(self._recent_accepts)
        # Baseline adjustment from acceptance rate.
        if rate < 0.2:
            self._strength = max(lo, self._strength * 0.7)
        elif rate > 0.6:
            self._strength = min(hi, self._strength * 1.3)
        # Entropy-based modulation: high entropy pushes strength up,
        # low entropy pulls it down, reinforcing the invent/refine regime.
        entropy_mod = (self.entropy - 0.5) * 0.4
        if entropy_mod > 0:
            self._strength = min(hi, self._strength * (1.0 + entropy_mod))
        else:
            self._strength = max(lo, self._strength * (1.0 + entropy_mod))
        self._strength = max(lo, min(hi, self._strength))

    def clamp(self) -> None:
        """Clamp all parameters to valid ranges."""
        for module in (
            self.camera,
            self.field,
            self.colour,
            self.particle,
            self.feedback,
            self.lighting,
            self.behaviour,
        ):
            module.clamp()

    def crossover(self, other: VisualGenome) -> VisualGenome:
        """Create a child genome by crossover with another genome.

        The child inherits a blended entropy from both parents,
        with a small random perturbation to seed diversity.

        Args:
            other: Parent genome to crossover with
        Returns:
            New child genome
        """
        child = VisualGenome(
            strength=(self._strength + other._strength) / 2.0,
            lineage=f"{self.lineage}x{other.lineage}",
            generation=max(self.generation, other.generation) + 1,
            parent_id=self.lineage,
        )
        for attr in ("camera", "field", "colour", "particle", "feedback", "lighting", "behaviour"):
            if random.random() < 0.5:
                setattr(child, attr, getattr(self, attr).clone())
            else:
                setattr(child, attr, getattr(other, attr).clone())
        child.objectives = self.objectives
        child.entropy = (self.entropy + other.entropy) / 2.0 + random.uniform(-0.05, 0.05)
        child.entropy = max(0.0, min(1.0, child.entropy))
        child.clamp()
        return child

    def clone(self) -> VisualGenome:
        """Create a deep copy of this genome.

        Returns:
            Copy of this genome
        """
        g = VisualGenome(
            strength=self._strength,
            lineage=self.lineage,
            generation=self.generation,
            parent_id=self.parent_id,
        )
        for attr in ("camera", "field", "colour", "particle", "feedback", "lighting", "behaviour"):
            setattr(g, attr, getattr(self, attr).clone())
        g._recent_accepts = list(self._recent_accepts)
        g.objectives = self.objectives
        g.mutations = list(self.mutations)
        g.entropy = self.entropy
        g._best_fitness_history = list(self._best_fitness_history)
        g._stagnation_counter = self._stagnation_counter
        g._gene_usage_tracker = copy.deepcopy(self._gene_usage_tracker)
        g._mutated_genes_last = list(self._mutated_genes_last)
        return g

    def shader_signature(self) -> dict[str, float | int]:
        """Return only the genome fields that affect the generated shader.
        Used as the :class:`ShaderCache` key so an unchanged phenotype reuses
        its compiled GL program. Must NOT include lineage, generation, fitness
        or mutation history -- those churn every generation and would defeat
        the cache (forcing recompiles and eviction leaks).
        """
        return self.params

    def to_dict(self) -> dict[str, float | int]:
        """Export genome as a dictionary."""
        data = self.params
        data["_lineage"] = self.lineage
        data["_generation"] = self.generation
        data["_objectives"] = self.objectives.to_dict()
        data["_parent_id"] = self.parent_id
        data["_mutations"] = list(self.mutations)
        data["_entropy"] = self.entropy
        data["_best_fitness_history"] = self._best_fitness_history
        data["_stagnation_counter"] = self._stagnation_counter
        return data

    def from_dict(self, data: dict[str, float | int]) -> None:
        """Load genome from a dictionary."""
        self.params = data
        self.lineage = data.get("_lineage", self._generate_id())
        self.generation = int(data.get("_generation", 0))
        obj_data = data.get("_objectives", {})
        self.objectives = Objectives.from_dict(obj_data) if obj_data else Objectives()
        self.parent_id = data.get("_parent_id")
        self.mutations = list(data.get("_mutations", []))
        self.entropy = float(data.get("_entropy", 0.5))
        self._best_fitness_history = list(data.get("_best_fitness_history", []))
        self._stagnation_counter = int(data.get("_stagnation_counter", 0))
        self.clamp()

    def distance(self, other: VisualGenome) -> float:
        """Normalized 0..1 phenotypic distance to another genome.
        Each parameter is compared relative to its allowed range so that genes
        with wide bounds do not dominate the metric.
        """
        total = 0.0
        count = 0
        pairs = [
            (self.camera, other.camera),
            (self.field, other.field),
            (self.colour, other.colour),
            (self.particle, other.particle),
            (self.feedback, other.feedback),
            (self.lighting, other.lighting),
            (self.behaviour, other.behaviour),
        ]
        for a, b in pairs:
            total += a.distance(b)
            count += 1
        return total / max(count, 1)

    def environmental_affinity(self, audio: dict[str, float]) -> float:
        """Bonus for how well this genome suits the current audio.
        Bass-heavy music favours vortex motion; treble favours particle
        density. Mirrors the selection pressure in the modular genome.
        """
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        score = 0.0
        if bass > 0.7:
            score += float(self.particle.vortex) * 0.2
        if treb > 0.7:
            score += float(self.particle.particle_density) * 0.1
        return score
