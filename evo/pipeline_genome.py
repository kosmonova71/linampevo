from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from evo.genome_base import GenomeBase
from evo.modular_genome import ModularGenome

logger = logging.getLogger(__name__)

MAX_MUTATION_HISTORY = 64

#: Valid successors for each pass type. Used to bias mutations toward
#: topologically sensible pipeline graphs.
PASS_GATES: dict[str, list[str]] = {
    "field": [
        "feedback",
        "post",
        "sdf",
        "reaction_diffusion",
        "fluid_feedback",
        "wave_equation",
        "heat_diffusion",
        "cellular_automata",
        "slime_mold",
        "boids",
        "physarum",
        "erosion",
        "lighting",
        "particle",
        "volume",
        "composite",
    ],
    "sdf": [
        "feedback",
        "post",
        "lighting",
        "particle",
        "volume",
        "composite",
    ],
    "reaction_diffusion": [
        "feedback",
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "fluid_feedback": [
        "feedback",
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "wave_equation": [
        "feedback",
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "heat_diffusion": [
        "feedback",
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "cellular_automata": [
        "feedback",
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "slime_mold": [
        "feedback",
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "boids": [
        "feedback",
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "physarum": [
        "feedback",
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "erosion": [
        "feedback",
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "feedback": [
        "post",
        "lighting",
        "particle",
        "volume",
        "composite",
    ],
    "post": [
        "post",
        "lighting",
        "particle",
        "volume",
        "composite",
    ],
    "lighting": [
        "post",
        "particle",
        "volume",
        "composite",
    ],
    "particle": [
        "post",
        "volume",
        "composite",
    ],
    "volume": [
        "post",
        "composite",
    ],
    "composite": [],
}

#: Default module selections for each pass type when constructing new passes.
DEFAULT_PASS_MODULES: dict[str, dict[str, Any]] = {
    "field": {
        "field_type": "fbm",
        "warp": "curl",
        "palette": "cosine",
        "feedback_mode": "spiral",
        "post": ["glow"],
    },
    "sdf": {
        "field_type": "sdf",
        "warp": "twist",
        "palette": "spectral",
        "feedback_mode": "none",
        "post": ["bloom"],
    },
    "reaction_diffusion": {
        "field_type": "reaction_diffusion",
        "warp": "none",
        "palette": "plasma",
        "feedback_mode": "spiral",
        "post": [],
    },
    "fluid_feedback": {
        "field_type": "fluid_feedback",
        "warp": "curl",
        "palette": "cosine",
        "feedback_mode": "spiral",
        "post": ["glow"],
    },
    "wave_equation": {
        "field_type": "wave_equation",
        "warp": "none",
        "palette": "spectral",
        "feedback_mode": "mirror",
        "post": ["glow"],
    },
    "heat_diffusion": {
        "field_type": "heat_diffusion",
        "warp": "none",
        "palette": "thermal",
        "feedback_mode": "spiral",
        "post": [],
    },
    "cellular_automata": {
        "field_type": "cellular_automata",
        "warp": "none",
        "palette": "binary",
        "feedback_mode": "mirror",
        "post": [],
    },
    "slime_mold": {
        "field_type": "slime_mold",
        "warp": "none",
        "palette": "plasma",
        "feedback_mode": "spiral",
        "post": ["glow"],
    },
    "boids": {
        "field_type": "boids",
        "warp": "vortex",
        "palette": "cosine",
        "feedback_mode": "drift",
        "post": ["glow"],
    },
    "physarum": {
        "field_type": "physarum",
        "warp": "none",
        "palette": "plasma",
        "feedback_mode": "spiral",
        "post": ["glow"],
    },
    "erosion": {
        "field_type": "erosion",
        "warp": "curl",
        "palette": "spectral",
        "feedback_mode": "none",
        "post": ["bloom"],
    },
    "feedback": {
        "field_type": "fbm",
        "warp": "none",
        "palette": "cosine",
        "feedback_mode": "spiral",
        "post": [],
    },
    "post": {
        "field_type": "fbm",
        "warp": "none",
        "palette": "cosine",
        "feedback_mode": "none",
        "post": ["bloom", "chromatic_aberration"],
    },
    "lighting": {
        "field_type": "mandelbrot",
        "warp": "curl",
        "palette": "cosine",
        "feedback_mode": "none",
        "post": ["glow"],
    },
    "particle": {
        "field_type": "fbm",
        "warp": "vortex",
        "palette": "cosine",
        "feedback_mode": "drift",
        "post": ["glow"],
    },
    "volume": {
        "field_type": "voronoi",
        "warp": "none",
        "palette": "cosine",
        "feedback_mode": "none",
        "post": [],
    },
    "composite": {
        "field_type": "fbm",
        "warp": "none",
        "palette": "cosine",
        "feedback_mode": "none",
        "post": ["glow"],
    },
}

#: Named pipeline archetypes for randomized initialization. Each list is an
#: ordered sequence of pass types. Organisms drawn from these archetypes
#: demonstrate semantically meaningful render graphs:
#:   A: minimal field → feedback → bloom
#:   B: rich RD → particles → lighting → post
#:   C: SDF → bloom → volume → composite
#:   D: fluid feedback → wave → erosion → composite
#:   E: slime mold → boids → physarum → post
PIPELINE_ARCHETYPES: list[list[str]] = [
    ["field", "feedback", "post"],
    ["reaction_diffusion", "particle", "lighting", "post"],
    ["sdf", "post", "volume", "composite"],
    ["field", "volume", "composite"],
    ["reaction_diffusion", "feedback", "post", "composite"],
    ["sdf", "lighting", "post", "composite"],
    ["particle", "post"],
    ["field", "feedback", "post", "lighting", "volume", "composite"],
    ["fluid_feedback", "wave_equation", "erosion", "composite"],
    ["slime_mold", "boids", "physarum", "post"],
    ["cellular_automata", "heat_diffusion", "post"],
]


# ============================================================================
# Pass-level genome (one render pass)
# ============================================================================

AVAILABLE_INPUT_SOURCES = ["previous", "feedback", "first", "none"]
AVAILABLE_BLEND_MODES = ["replace", "alpha", "add", "multiply", "screen"]
AVAILABLE_FEEDBACK_ROUTES = ["none", "self", "previous", "first"]


@dataclass
class PassGenome(GenomeBase):
    """A single render pass with its own module selections and graph topology.

    Each pass is a self-contained ``ModularGenome`` plus pass type tag and
    render-graph wiring:
      - input_source: where this pass reads its input from
      - blend_mode: how this pass blends with its input
      - feedback_route: whether this pass participates in feedback

    Passes mutate independently, so the search space is multiplicative.
    """

    PASS_TYPES = (
        "field",  # Base procedural field (FBM, voronoi, etc.)
        "sdf",  # Signed-distance field raymarch
        "reaction_diffusion",  # Reaction-diffusion pattern
        "fluid_feedback",  # Fluid advection feedback
        "wave_equation",  # Wave equation simulation
        "heat_diffusion",  # Heat diffusion simulation
        "cellular_automata",  # Cellular automata patterns
        "slime_mold",  # Slime mold / transport networks
        "boids",  # Boids flocking simulation
        "physarum",  # Physarum transport networks
        "erosion",  # Hydraulic erosion simulation
        "feedback",  # Temporal feedback / echo
        "post",  # Post-processing effects
        "lighting",  # 3D surface lighting
        "particle",  # Particle overlay
        "volume",  # Atmosphere / fog / volumetric
        "composite",  # Final compositor / output
    )

    pass_type: str = "field"
    modules: ModularGenome = dc_field(default_factory=ModularGenome)

    # Multi-pass render graph wiring
    input_source: str = "previous"
    blend_mode: str = "replace"
    feedback_route: str = "none"

    lineage: str = ""
    generation: int = 0
    parent_id: str | None = None
    _objectives: dict[str, float] | None = None
    mutations: list[str] = dc_field(default_factory=list)

    age: int = 0
    lifespan: int = 500

    _strength: float = 0.3
    _recent_accepts: list[int] = dc_field(default_factory=list)
    _window_size: int = 8

    def __post_init__(self):
        if not self.lineage:
            self.lineage = str(uuid.uuid4())[:8]
        if self.pass_type not in self.PASS_TYPES:
            self.pass_type = "field"
        if self.input_source not in AVAILABLE_INPUT_SOURCES:
            self.input_source = "previous"
        if self.blend_mode not in AVAILABLE_BLEND_MODES:
            self.blend_mode = "replace"
        if self.feedback_route not in AVAILABLE_FEEDBACK_ROUTES:
            self.feedback_route = "none"

    @property
    def objectives(self):
        from evo_bio.pareto import Objectives

        if self._objectives is None:
            self._objectives = Objectives().to_dict()
        if isinstance(self._objectives, dict):
            return Objectives.from_dict(self._objectives)
        return self._objectives

    @objectives.setter
    def objectives(self, value):
        from evo_bio.pareto import Objectives

        if isinstance(value, Objectives):
            self._objectives = value.to_dict()
        elif isinstance(value, dict):
            self._objectives = dict(value)
        else:
            self._objectives = Objectives().to_dict()

    @property
    def fitness(self) -> float:
        return self.objectives.scalar()

    @fitness.setter
    def fitness(self, value: float) -> None:
        from evo_bio.pareto import Objectives

        self._objectives = Objectives(
            beauty=float(value) * 0.4,
            novelty=float(value) * 0.15,
            audio_sync=float(value) * 0.1,
            performance=1.0,
            species_health=float(value) * 0.1,
            motion=float(value) * 0.1,
            complexity=float(value) * 0.1,
            memory=float(value) * 0.05,
        ).to_dict()

    def mutate(self, strength: float | None = None, large: bool = False) -> None:
        if strength is not None:
            self._strength = float(strength)
        if large:
            self._strength = min(self._strength * 2.0 + 0.1, 1.0)

        self._mutate_once()
        if large:
            self._mutate_once()

    def _mutate_once(self) -> None:
        # Delegate to the inner ModularGenome; it handles module swaps and
        # parameter drift.
        try:
            self.modules.mutate(strength=self._strength)
            desc = f"{self.pass_type}:{random.choice(self.modules.mutations[-3:]) if self.modules.mutations else 'param'}"
        except Exception:
            desc = f"{self.pass_type}:noop"

        # Occasionally swap pass type itself (rare structural mutation).
        if random.random() < 0.05:
            old_type = self.pass_type
            new_type = random.choice(self.PASS_TYPES)
            self.pass_type = new_type
            desc = f"pass_type:{old_type}->{new_type}"

        # Mutate render graph wiring
        wiring_mutated = False
        if random.random() < 0.15:
            old = self.input_source
            choices = [c for c in AVAILABLE_INPUT_SOURCES if c != old]
            if choices:
                self.input_source = random.choice(choices)
                desc = f"input:{old}->{self.input_source}"
                wiring_mutated = True

        if random.random() < 0.10:
            old = self.blend_mode
            choices = [c for c in AVAILABLE_BLEND_MODES if c != old]
            if choices:
                self.blend_mode = random.choice(choices)
                if not wiring_mutated:
                    desc = f"blend:{old}->{self.blend_mode}"
                else:
                    desc += f"+blend:{old}->{self.blend_mode}"
                wiring_mutated = True

        if random.random() < 0.08:
            old = self.feedback_route
            choices = [c for c in AVAILABLE_FEEDBACK_ROUTES if c != old]
            if choices:
                self.feedback_route = random.choice(choices)
                if not wiring_mutated:
                    desc = f"fb:{old}->{self.feedback_route}"
                else:
                    desc += f"+fb:{old}->{self.feedback_route}"

        if desc:
            self.mutations.append(desc)
            if len(self.mutations) > MAX_MUTATION_HISTORY:
                del self.mutations[:-MAX_MUTATION_HISTORY]

    def record_accept(self, accepted: bool) -> None:
        self._recent_accepts.append(1 if accepted else 0)
        if len(self._recent_accepts) > self._window_size:
            self._recent_accepts.pop(0)
        try:
            self.modules.record_accept(accepted)
        except Exception:
            pass

    def adapt_strength(self, lo: float, hi: float) -> None:
        try:
            self.modules.adapt_strength(lo, hi)
        except Exception:
            pass

    def clone(self) -> PassGenome:
        g = PassGenome(
            pass_type=self.pass_type,
            modules=self.modules.clone(),
            input_source=self.input_source,
            blend_mode=self.blend_mode,
            feedback_route=self.feedback_route,
            lineage=self.lineage,
            generation=self.generation,
            parent_id=self.parent_id,
        )
        g.fitness = self.fitness
        g.mutations = list(self.mutations)
        g._strength = self._strength
        g._recent_accepts = list(self._recent_accepts)
        return g

    def crossover(self, other: PassGenome) -> PassGenome:
        child_modules = self.modules.crossover(other.modules)
        return PassGenome(
            pass_type=random.choice([self.pass_type, other.pass_type]),
            modules=child_modules,
            input_source=random.choice([self.input_source, other.input_source]),
            blend_mode=random.choice([self.blend_mode, other.blend_mode]),
            feedback_route=random.choice([self.feedback_route, other.feedback_route]),
            generation=max(self.generation, other.generation) + 1,
            parent_id=self.lineage,
        )

    def distance(self, other: PassGenome) -> float:
        d = 0.0
        n = 0
        if self.pass_type != other.pass_type:
            d += 1.0
        n += 1
        if self.input_source != other.input_source:
            d += 0.5
        n += 1
        if self.blend_mode != other.blend_mode:
            d += 0.3
        n += 1
        if self.feedback_route != other.feedback_route:
            d += 0.3
        n += 1
        try:
            d += self.modules.distance(other.modules)
            n += 1
        except Exception:
            pass
        return d / max(n, 1)

    def shader_signature(self) -> dict[str, Any]:
        return {
            "pass_type": self.pass_type,
            "modules": self.modules.shader_signature(),
            "input_source": self.input_source,
            "blend_mode": self.blend_mode,
            "feedback_route": self.feedback_route,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_type": self.pass_type,
            "modules": self.modules.to_dict(),
            "input_source": self.input_source,
            "blend_mode": self.blend_mode,
            "feedback_route": self.feedback_route,
            "_lineage": self.lineage,
            "_generation": self.generation,
            "_objectives": self.objectives.to_dict() if hasattr(self, "objectives") else {},
            "_parent_id": self.parent_id,
            "_mutations": list(self.mutations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PassGenome:
        g = cls()
        g.pass_type = data.get("pass_type", "field")
        g.input_source = data.get("input_source", "previous")
        g.blend_mode = data.get("blend_mode", "replace")
        g.feedback_route = data.get("feedback_route", "none")
        try:
            g.modules = ModularGenome()
            g.modules.from_dict(data.get("modules", {}))
        except Exception:
            g.modules = ModularGenome()
        g.lineage = data.get("_lineage", g.lineage or str(uuid.uuid4())[:8])
        g.generation = int(data.get("_generation", 0))
        obj_data = data.get("_objectives", data.get("_fitness"))
        if isinstance(obj_data, dict) and obj_data:
            from evo_bio.pareto import Objectives

            g.objectives = Objectives.from_dict(obj_data)
        elif isinstance(obj_data, (int, float)):
            g.fitness = float(obj_data)
        g.parent_id = data.get("_parent_id")
        g.mutations = list(data.get("_mutations", []))[-MAX_MUTATION_HISTORY:]
        return g


# ============================================================================
# Pipeline-level genome (multiple passes evolved together)
# ============================================================================


@dataclass
class PipelineGenome(GenomeBase):
    """An entire render pipeline / render graph, composed of independent passes.

    Each pass has its own PassGenome with full render graph wiring:
      - input_source: previous, feedback, first, none
      - blend_mode: replace, alpha, add, multiply, screen
      - feedback_route: none, self, previous, first

    The pipeline mutates at three levels:
      1. Graph topology: add / remove / reorder passes (structural)
      2. Per-pass wiring: mutate input_source, blend_mode, feedback_route
      3. Module-level: each PassGenome mutates its own modules (parametric)

    This is the top-level genome the evolution engine should drive.
    """

    passes: list[PassGenome] = dc_field(default_factory=list)
    global_params: dict[str, float | int] = dc_field(default_factory=dict)

    lineage: str = ""
    generation: int = 0
    parent_id: str | None = None
    _objectives: dict[str, float] | None = None
    mutations: list[str] = dc_field(default_factory=list)

    age: int = 0
    lifespan: int = 500

    _strength: float = 0.3
    _recent_accepts: list[int] = dc_field(default_factory=list)
    _window_size: int = 8

    def __post_init__(self):
        if not self.lineage:
            self.lineage = str(uuid.uuid4())[:8]
        if not self.passes:
            self.passes = self._default_pipeline()
        if not self.global_params:
            self.global_params = {
                "brightness": 0.85,
                "feedback_alpha": 0.85,
                "feedback_res": 512,
                "ray_steps": 32,
                "fog_density": 1.0,
            }
            if self._objectives is None:
                self._objectives = {
                    "beauty": 0.5,
                    "novelty": 0.0,
                    "audio_sync": 0.0,
                    "performance": 1.0,
                    "species_health": 0.5,
                    "motion": 0.0,
                    "complexity": 0.5,
                    "memory": 0.5,
                }

    @property
    def objectives(self):
        from evo_bio.pareto import Objectives

        if self._objectives is None:
            self._objectives = Objectives().to_dict()
        if isinstance(self._objectives, dict):
            return Objectives.from_dict(self._objectives)
        return self._objectives

    @objectives.setter
    def objectives(self, value):
        from evo_bio.pareto import Objectives

        if isinstance(value, Objectives):
            self._objectives = value.to_dict()
        elif isinstance(value, dict):
            self._objectives = dict(value)
        else:
            self._objectives = Objectives().to_dict()

    @property
    def fitness(self) -> float:
        return self.objectives.scalar()

    @fitness.setter
    def fitness(self, value: float) -> None:
        from evo_bio.pareto import Objectives

        self._objectives = Objectives(
            beauty=float(value) * 0.4,
            novelty=float(value) * 0.15,
            audio_sync=float(value) * 0.1,
            performance=1.0,
            species_health=float(value) * 0.1,
            motion=float(value) * 0.1,
            complexity=float(value) * 0.1,
            memory=float(value) * 0.05,
        ).to_dict()

    def update_animation(self, audio: dict[str, float], t: float, beat: float) -> None:
        """Drive Milkdrop-style per-frame animation for every pass.

        Each :class:`PassGenome`'s :class:`ModularGenome` carries the animated
        parameters; nudge every pass so field passes share a consistent
        zoom/rot/warp state driven by live audio and time. The renderer calls
        this once per frame (via the shared ``update_animation`` contract)
        before rendering.
        """
        for p in self.passes:
            modules = getattr(p, "modules", None)
            if modules is not None and hasattr(modules, "update_animation"):
                modules.update_animation(audio, t, beat)

    def get_animation_uniforms(self) -> dict[str, float]:
        """Return live animation uniforms from the first animated pass.

        Mirrors the renderer's primary-genome selection: the first pass drives
        the transform, and because every pass is updated identically in
        :meth:`update_animation`, its values are valid for all field passes.
        """
        for p in self.passes:
            modules = getattr(p, "modules", None)
            if modules is not None and hasattr(modules, "get_animation_uniforms"):
                return modules.get_animation_uniforms()
        return {}

    @staticmethod
    def _default_pipeline() -> list[PassGenome]:
        return [
            PassGenome(
                pass_type="field",
                modules=ModularGenome(
                    field_type="fbm",
                    warp="curl",
                    palette="cosine",
                    feedback_mode="spiral",
                    post=["glow"],
                ),
                input_source="none",
                blend_mode="replace",
                feedback_route="none",
            ),
            PassGenome(
                pass_type="feedback",
                modules=ModularGenome(
                    field_type="fbm",
                    warp="none",
                    palette="cosine",
                    feedback_mode="spiral",
                    post=[],
                ),
                input_source="feedback",
                blend_mode="alpha",
                feedback_route="self",
            ),
            PassGenome(
                pass_type="post",
                modules=ModularGenome(
                    field_type="fbm",
                    warp="none",
                    palette="cosine",
                    feedback_mode="none",
                    post=["bloom", "chromatic_aberration"],
                ),
                input_source="previous",
                blend_mode="add",
                feedback_route="none",
            ),
            PassGenome(
                pass_type="composite",
                modules=ModularGenome(
                    field_type="fbm",
                    warp="none",
                    palette="cosine",
                    feedback_mode="none",
                    post=["glow"],
                ),
                input_source="previous",
                blend_mode="screen",
                feedback_route="none",
            ),
        ]

    @staticmethod
    def _make_pass(ptype: str) -> PassGenome:
        dm = DEFAULT_PASS_MODULES.get(ptype, DEFAULT_PASS_MODULES["field"])
        modules = ModularGenome(**dm)
        input_src = {
            "field": "none",
            "sdf": "none",
            "reaction_diffusion": "none",
            "fluid_feedback": "feedback",
            "wave_equation": "none",
            "heat_diffusion": "none",
            "cellular_automata": "none",
            "slime_mold": "feedback",
            "boids": "previous",
            "physarum": "feedback",
            "erosion": "none",
            "feedback": "feedback",
            "post": "previous",
            "lighting": "previous",
            "particle": "previous",
            "volume": "previous",
            "composite": "previous",
        }.get(ptype, "previous")
        blend = {
            "field": "replace",
            "sdf": "replace",
            "reaction_diffusion": "replace",
            "fluid_feedback": "alpha",
            "wave_equation": "add",
            "heat_diffusion": "add",
            "cellular_automata": "replace",
            "slime_mold": "alpha",
            "boids": "screen",
            "physarum": "alpha",
            "erosion": "replace",
            "feedback": "alpha",
            "post": "add",
            "lighting": "screen",
            "particle": "screen",
            "volume": "alpha",
            "composite": "screen",
        }.get(ptype, "replace")
        return PassGenome(
            pass_type=ptype,
            modules=modules,
            input_source=input_src,
            blend_mode=blend,
            feedback_route="self" if ptype == "feedback" else "none",
        )

    @staticmethod
    def random_initial() -> PipelineGenome:
        g = PipelineGenome()
        g.passes = []

        # 70% archetype-based, 30% fully random
        if random.random() < 0.7:
            arch = random.choice(PIPELINE_ARCHETYPES)
            for ptype in arch:
                g.passes.append(PipelineGenome._make_pass(ptype))
        else:
            n_passes = random.randint(3, 6)
            types = list(PassGenome.PASS_TYPES)
            random.shuffle(types)
            for i in range(n_passes):
                ptype = types[i % len(types)]
                modules = ModularGenome(
                    field_type=random.choice(
                        [
                            "fbm",
                            "voronoi",
                            "reaction_diffusion",
                            "mandelbrot",
                            "sdf",
                            "fluid_feedback",
                            "wave_equation",
                            "heat_diffusion",
                            "cellular_automata",
                            "slime_mold",
                            "boids",
                            "physarum",
                            "erosion",
                        ]
                    ),
                    warp=random.choice(["curl", "vortex", "twist", "none"]),
                    palette=random.choice(["cosine", "spectral", "plasma"]),
                    feedback_mode=random.choice(["drift", "mirror", "spiral", "zoom", "kaleido"]),
                    post=random.sample(
                        [
                            "bloom",
                            "feedback",
                            "distortion",
                            "glow",
                            "chromatic_aberration",
                            "polar_galaxy",
                            "cells",
                            "nonlinear_trails",
                            "audio_topology",
                            "depth_cloud",
                            "neon_edge",
                        ],
                        k=random.randint(0, 3),
                    ),
                )
                g.passes.append(
                    PassGenome(
                        pass_type=ptype,
                        modules=modules,
                        input_source=random.choice(AVAILABLE_INPUT_SOURCES),
                        blend_mode=random.choice(AVAILABLE_BLEND_MODES),
                        feedback_route=random.choice(AVAILABLE_FEEDBACK_ROUTES),
                    )
                )

        for k in g.global_params:
            if k in ("brightness", "feedback_alpha"):
                g.global_params[k] = random.uniform(0.5, 0.95)
            elif k == "feedback_res":
                g.global_params[k] = random.choice([256, 512, 1024])
            elif k == "ray_steps":
                g.global_params[k] = random.randint(16, 64)
            elif k == "fog_density":
                g.global_params[k] = random.uniform(0.0, 3.0)
        return g

    # ------------------------------------------------------------------ mutation

    def mutate(self, strength: float | None = None, large: bool = False) -> None:
        if strength is not None:
            self._strength = float(strength)
        if large:
            self._strength = min(self._strength * 2.0 + 0.1, 1.0)

        self._mutate_once()
        if large:
            self._mutate_once()

    def _mutate_once(self) -> None:
        roll = random.random()
        if roll < 0.15:
            self._mutate_structure()
        elif roll < 0.25:
            self._mutate_order()
        elif roll < 0.40:
            self._mutate_connections()
        elif roll < 0.55:
            self._mutate_global()
        else:
            self._mutate_passes()

    def _mutate_structure(self) -> None:
        """Add / remove / duplicate a pass."""
        if random.random() < 0.5 and len(self.passes) > 2:
            idx = random.randrange(len(self.passes))
            removed = self.passes.pop(idx)
            self.mutations.append(f"remove_pass:{removed.pass_type}")
        elif len(self.passes) < 8:
            ptype = self._pick_compatible_pass_type()
            modules = ModularGenome(
                field_type=random.choice(
                    ["fbm", "voronoi", "reaction_diffusion", "mandelbrot", "sdf"]
                ),
                warp=random.choice(["curl", "vortex", "twist", "none"]),
                palette=random.choice(["cosine", "spectral", "plasma"]),
                feedback_mode="none",
                post=[],
            )
            new_pass = PassGenome(
                pass_type=ptype,
                modules=modules,
                input_source=random.choice(AVAILABLE_INPUT_SOURCES),
                blend_mode=random.choice(AVAILABLE_BLEND_MODES),
                feedback_route=random.choice(AVAILABLE_FEEDBACK_ROUTES),
            )
            # Prefer to insert after a valid predecessor
            insert_idx = self._find_valid_insert_position(ptype, len(self.passes))
            self.passes.insert(insert_idx, new_pass)
            self.mutations.append(f"add_pass:{ptype}")
        else:
            self._mutate_passes()

    def _pick_compatible_pass_type(self) -> str:
        """Choose a pass type that is compatible with the current terminal passes."""
        if not self.passes:
            return random.choice(list(PassGenome.PASS_TYPES))
        last = self.passes[-1].pass_type
        # Composite must always be last — skip it if already present
        if "composite" in {p.pass_type for p in self.passes}:
            # Only allow volume/composite/feedback to be added as final supplement
            candidates = [
                t
                for t in PassGenome.PASS_TYPES
                if t in ("volume", "feedback", "post", "particle", "composite")
                and t not in {p.pass_type for p in self.passes}
            ]
        else:
            # All types that CAN follow the current last pass
            gates = PASS_GATES.get(last, list(PassGenome.PASS_TYPES))
            candidates = [t for t in gates if t in PassGenome.PASS_TYPES]
        if not candidates:
            candidates = list(PassGenome.PASS_TYPES)
        return random.choice(candidates)

    def _find_valid_insert_position(self, ptype: str, limit: int) -> int:
        """Find an index where inserting `ptype` after that index is valid."""
        if limit == 0:
            return 0
        # Collect indices where the current pass's valid successors include ptype
        valid_indices = []
        for i in range(limit):
            cur = self.passes[i].pass_type
            gates = PASS_GATES.get(cur, list(PassGenome.PASS_TYPES))
            if ptype in gates or cur == ptype:
                valid_indices.append(i)
        if valid_indices:
            return random.choice(valid_indices) + 1
        return limit

    def _mutate_order(self) -> None:
        """Shuffle pass ordering (structural mutation)."""
        if len(self.passes) < 2:
            return
        for _ in range(3):
            i, j = random.sample(range(len(self.passes)), 2)
            # Validate against PASS_GATES around swap positions
            if self._order_valid(i, j):
                self.passes[i], self.passes[j] = self.passes[j], self.passes[i]
                self.mutations.append(f"reorder:{i}<->{j}")
                return

    def _order_valid(self, i: int, j: int) -> bool:
        """Return True if swapping positions i and j preserves valid transitions."""
        passes = self.passes
        # Check predecessor of i (before swap)
        if i > 0:
            prev = passes[i - 1].pass_type
            if passes[j].pass_type not in PASS_GATES.get(prev, list(PassGenome.PASS_TYPES)):
                return False
        # Check predecessor of j
        if j > 0:
            prev = passes[j - 1].pass_type
            if passes[i].pass_type not in PASS_GATES.get(prev, list(PassGenome.PASS_TYPES)):
                return False
        # Check successor constraints
        if i + 1 < len(passes):
            succ_i = passes[i + 1].pass_type
            if passes[j].pass_type not in PASS_GATES.get(succ_i, list(PassGenome.PASS_TYPES)):
                return False
        if j + 1 < len(passes):
            succ_j = passes[j + 1].pass_type
            if passes[i].pass_type not in PASS_GATES.get(succ_j, list(PassGenome.PASS_TYPES)):
                return False
        return True

    def _mutate_connections(self) -> None:
        """Mutate render graph wiring of a random pass."""
        if not self.passes:
            return
        pass_idx = random.randrange(len(self.passes))
        p = self.passes[pass_idx]

        wiring_choices = ["input_source", "blend_mode", "feedback_route"]
        choice = random.choice(wiring_choices)

        if choice == "input_source":
            old = p.input_source
            candidates = [c for c in AVAILABLE_INPUT_SOURCES if c != old]
            if candidates:
                p.input_source = random.choice(candidates)
                self.mutations.append(f"pass[{pass_idx}]_input:{old}->{p.input_source}")
        elif choice == "blend_mode":
            old = p.blend_mode
            candidates = [c for c in AVAILABLE_BLEND_MODES if c != old]
            if candidates:
                p.blend_mode = random.choice(candidates)
                self.mutations.append(f"pass[{pass_idx}]_blend:{old}->{p.blend_mode}")
        elif choice == "feedback_route":
            old = p.feedback_route
            candidates = [c for c in AVAILABLE_FEEDBACK_ROUTES if c != old]
            if candidates:
                p.feedback_route = random.choice(candidates)
                self.mutations.append(f"pass[{pass_idx}]_fb:{old}->{p.feedback_route}")

    def _mutate_global(self) -> None:
        key = random.choice(list(self.global_params.keys()))
        val = float(self.global_params[key])
        delta = random.uniform(-self._strength, self._strength) * max(abs(val), 0.1)
        new_val = val + delta
        if key == "ray_steps":
            new_val = max(8, min(128, round(new_val)))
        elif key == "feedback_res":
            new_val = max(64, min(2048, round(new_val)))
        else:
            new_val = max(0.0, new_val)
        self.global_params[key] = new_val
        self.mutations.append(f"global:{key}{delta:+.3f}")

    def _mutate_passes(self) -> None:
        if not self.passes:
            return
        pass_idx = random.randrange(len(self.passes))
        pass_genome = self.passes[pass_idx]
        pass_genome.mutate(strength=self._strength)
        self.mutations.append(f"pass[{pass_idx}]:{pass_genome.pass_type}")

    def record_accept(self, accepted: bool) -> None:
        self._recent_accepts.append(1 if accepted else 0)
        if len(self._recent_accepts) > self._window_size:
            self._recent_accepts.pop(0)
        for p in self.passes:
            try:
                p.record_accept(accepted)
            except Exception:
                pass

    def adapt_strength(self, lo: float, hi: float) -> None:
        if not self._recent_accepts:
            return
        rate = sum(self._recent_accepts) / len(self._recent_accepts)
        if rate < 0.2:
            self._strength = max(lo, self._strength * 0.7)
        elif rate > 0.6:
            self._strength = min(hi, self._strength * 1.3)
        for p in self.passes:
            try:
                p.adapt_strength(lo, hi)
            except Exception:
                pass

    def clone(self) -> PipelineGenome:
        g = PipelineGenome(
            passes=[p.clone() for p in self.passes],
            global_params=dict(self.global_params),
            lineage=self.lineage,
            generation=self.generation,
            parent_id=self.parent_id,
        )
        g.fitness = self.fitness
        g.mutations = list(self.mutations)
        g._strength = self._strength
        g._recent_accepts = list(self._recent_accepts)
        return g

    def crossover(self, other: PipelineGenome) -> PipelineGenome:
        child = PipelineGenome.random_initial()
        child.global_params = (
            dict(self.global_params) if random.random() < 0.5 else dict(other.global_params)
        )

        # Crossover passes: take some from each parent, preserving order roughly.
        child.passes = []
        max_passes = max(len(self.passes), len(other.passes))
        for i in range(max_passes):
            if i < len(self.passes) and i < len(other.passes):
                child.passes.append(self.passes[i].crossover(other.passes[i]))
            elif i < len(self.passes):
                child.passes.append(self.passes[i].clone())
            else:
                child.passes.append(other.passes[i].clone())

        # Trim to reasonable size
        if len(child.passes) > 8:
            child.passes = child.passes[:8]
        if len(child.passes) < 2:
            child.passes = self.passes[:2] + other.passes[:2]

        return child

    def distance(self, other: PipelineGenome) -> float:
        d = 0.0
        n = 0
        mine_types = [p.pass_type for p in self.passes]
        theirs_types = [p.pass_type for p in other.passes]
        d += len(set(mine_types) ^ set(theirs_types))
        n += 1

        for i in range(max(len(self.passes), len(other.passes))):
            if i < len(self.passes) and i < len(other.passes):
                d += self.passes[i].distance(other.passes[i])
            else:
                d += 1.0
            n += 1

        for k in set(self.global_params) | set(other.global_params):
            a = self.global_params.get(k)
            b = other.global_params.get(k)
            if a is None or b is None:
                d += 1.0
            else:
                denom = max(abs(float(a)), abs(float(b)), 1e-6)
                d += min(abs(float(a) - float(b)) / denom, 1.0)
            n += 1

        return d / max(n, 1)

    def shader_signature(self) -> dict[str, Any]:
        return {
            "passes": [p.shader_signature() for p in self.passes],
            "global_params": dict(self.global_params),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes": [p.to_dict() for p in self.passes],
            "global_params": dict(self.global_params),
            "_lineage": self.lineage,
            "_generation": self.generation,
            "_objectives": self.objectives.to_dict() if hasattr(self, "objectives") else {},
            "_parent_id": self.parent_id,
            "_mutations": list(self.mutations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineGenome:
        g = cls()
        g.passes = [PassGenome.from_dict(p) for p in data.get("passes", [])]
        g.global_params = dict(data.get("global_params", g.global_params))
        g.lineage = data.get("_lineage", g.lineage or str(uuid.uuid4())[:8])
        g.generation = int(data.get("_generation", 0))
        obj_data = data.get("_objectives", data.get("_fitness"))
        if isinstance(obj_data, dict) and obj_data:
            from evo_bio.pareto import Objectives

            g.objectives = Objectives.from_dict(obj_data)
        elif isinstance(obj_data, (int, float)):
            g.fitness = float(obj_data)
        g.parent_id = data.get("_parent_id")
        g.mutations = list(data.get("_mutations", []))[-MAX_MUTATION_HISTORY:]
        return g

    def age_one_generation(self) -> None:
        self.age += 1
        if self.age > self.lifespan:
            try:
                self.mutate(large=True)
            except Exception:
                self.mutate()
            self.age = 0
            return

        ratio = self.age / max(self.lifespan, 1)
        if ratio < 0.25:
            try:
                self.mutate(large=True)
            except Exception:
                self.mutate()
        elif ratio < 0.75:
            try:
                self.mutate()
            except Exception:
                pass
        else:
            try:
                self.mutate(large=True)
                self.mutate(large=True)
            except Exception:
                try:
                    self.mutate()
                except Exception:
                    pass
        for p in self.passes:
            try:
                p.age_one_generation()
            except Exception:
                pass

    def metamorphosis(self) -> None:
        """Genetic mutation event: a rare dramatic jump to a new species."""
        for p in self.passes:
            try:
                p.modules.metamorphosis()
            except Exception:
                pass
        self.mutations.append("metamorphosis")
