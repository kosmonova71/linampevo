#!/usr/bin/env python3

"""modular_genome.py

Modular visual genome that chooses shader modules instead of numeric constants.

Genome structure:
{
    "field": "fbm",           # field module name
    "warp": "curl",           # warp module name
    "palette": "cosine",      # colour module name
    "feedback_mode": "spiral",# feedback module mode
    "post": ["glow", "chromatic_aberration"],  # post-processing modules

    # Module-specific parameters
    "field_params": {
        "fractal_layers": 4,
        "lacunarity": 2.0,
        ...
    },
    ...
}
"""

from __future__ import annotations

import copy
import logging
import math
import random
import uuid
from dataclasses import dataclass
from dataclasses import field as dc_field

from evo.genome_base import GenomeBase

logger = logging.getLogger(__name__)

# Mutation history is bounded so genomes do not grow without limit across
# generations (every clone deep-copies the list).
MAX_MUTATION_HISTORY = 64

AVAILABLE_FIELDS = [
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
AVAILABLE_WARPS = ["curl", "vortex", "twist", "none"]
AVAILABLE_PALETTES = ["cosine", "spectral", "plasma"]
# Feedback transforms applied to the feedback-sample UV (see evo/feedback/*.glsl).
AVAILABLE_FEEDBACK = ["drift", "mirror", "spiral", "zoom", "kaleido"]
# Post operators are an evolving stack: evolution discovers combinations of these.
# Block 1 are the original operators; block 2 are the new shader operators
# (kaleidoscope feedback is handled via AVAILABLE_FEEDBACK instead).
AVAILABLE_POST = [
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
]


@dataclass
class ModularGenome(GenomeBase):
    """A genome that selects shader modules rather than numeric constants.

    The genome consists of:
    - Module selections (field, warp, palette, post)
    - Module-specific parameter dicts
    - Evolutionary tracking (lineage, generation, fitness)
    """

    field_type: str = "fbm"
    warp: str = "curl"
    palette: str = "cosine"
    feedback_mode: str = "spiral"
    lighting: str = "none"
    camera: str = "none"
    particles: str = "none"
    post: list[str] = dc_field(default_factory=lambda: ["glow"])

    # Milkdrop-style per-frame animation parameters
    anim_zoom: float = 1.0
    anim_rot: float = 0.0
    anim_cx: float = 0.5
    anim_cy: float = 0.5
    anim_dx: float = 0.0
    anim_dy: float = 0.0
    anim_sx: float = 1.0
    anim_sy: float = 1.0
    anim_warp: float = 0.0
    anim_wave_r: float = 0.5
    anim_wave_g: float = 0.5
    anim_wave_b: float = 0.5
    anim_wave_x: float = 0.5
    anim_wave_y: float = 0.5
    anim_ob_size: float = 0.0
    anim_ob_r: float = 0.0
    anim_ob_g: float = 0.0
    anim_ob_b: float = 0.0
    anim_ob_a: float = 0.0
    anim_ib_size: float = 0.0
    anim_ib_r: float = 0.0
    anim_ib_g: float = 0.0
    anim_ib_b: float = 0.0
    anim_ib_a: float = 0.0
    anim_mv_x: float = 0.0
    anim_mv_y: float = 0.0
    anim_mv_l: float = 0.0
    anim_mv_r: float = 0.0
    anim_mv_g: float = 0.0
    anim_mv_b: float = 0.0
    anim_mv_a: float = 0.0

    # Per-frame equation targets (computed each frame from audio/time)
    anim_zoom_target: float = 1.0
    anim_rot_target: float = 0.0
    anim_cx_target: float = 0.5
    anim_cy_target: float = 0.5
    anim_warp_target: float = 0.0

    # Animation smoothing factors (0-1, higher = faster response)
    anim_smooth_zoom: float = 0.02
    anim_smooth_rot: float = 0.02
    anim_smooth_cx: float = 0.01
    anim_smooth_cy: float = 0.01
    anim_smooth_warp: float = 0.03
    anim_smooth_wave: float = 0.1

    field_params: dict[str, float | int] = dc_field(default_factory=dict)
    warp_params: dict[str, float | int] = dc_field(default_factory=dict)
    palette_params: dict[str, float | int] = dc_field(default_factory=dict)
    feedback_params: dict[str, float | int] = dc_field(default_factory=dict)
    lighting_params: dict[str, float | int] = dc_field(default_factory=dict)
    camera_params: dict[str, float | int] = dc_field(default_factory=dict)
    particle_params: dict[str, float | int] = dc_field(default_factory=dict)
    post_params: dict[str, float | int] = dc_field(default_factory=dict)

    extra_fields: list[str] = dc_field(default_factory=list)
    extra_warps: list[str] = dc_field(default_factory=list)
    extra_palettes: list[str] = dc_field(default_factory=list)
    extra_feedbacks: list[str] = dc_field(default_factory=list)
    extra_field_params: list[dict[str, float | int]] = dc_field(default_factory=list)
    extra_warp_params: list[dict[str, float | int]] = dc_field(default_factory=list)
    extra_palette_params: list[dict[str, float | int]] = dc_field(default_factory=list)
    extra_feedback_params: list[dict[str, float | int]] = dc_field(default_factory=list)

    lineage: str = ""
    generation: int = 0
    parent_id: str | None = None
    _objectives: dict[str, float] = None
    mutations: list[str] = dc_field(default_factory=list)

    age: int = 0
    lifespan: int = 500

    _strength: float = 0.3
    _recent_accepts: list[int] = dc_field(default_factory=list)
    _window_size: int = 8

    def __post_init__(self):
        if not self.lineage:
            self.lineage = str(uuid.uuid4())[:8]
        if not self.field_params:
            self.field_params = self._default_field_params()
        if not self.warp_params:
            self.warp_params = self._default_warp_params()
        if not self.palette_params:
            self.palette_params = self._default_palette_params()
        if not self.feedback_params:
            self.feedback_params = self._default_feedback_params()
        if not self.lighting_params:
            self.lighting_params = self._default_lighting_params()
        if not self.camera_params:
            self.camera_params = self._default_camera_params()
        if not self.particle_params:
            self.particle_params = self._default_particle_params()
        if not self.post_params:
            self.post_params = self._default_post_params()
        if not hasattr(self, "anim_zoom"):
            self.__dict__.update(self._default_animation_params())
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

    @property
    def animation(self) -> dict[str, float]:
        """Return all animation parameters as a dict for shader uniform binding."""
        return {k: getattr(self, k) for k in self.__dict__ if k.startswith("anim_")}

    def _default_field_params(self) -> dict[str, float | int]:
        params = {
            "fbm": {
                "fractal_layers": 4,
                "lacunarity": 2.0,
                "gain": 0.5,
                "domain_warp": 0.5,
                "noise_scale": 3.0,
            },
            "voronoi": {"noise_scale": 5.0, "jitter": 1.0, "animation_speed": 1.0},
            "reaction_diffusion": {
                "feed_rate": 0.055,
                "kill_rate": 0.062,
                "diffusion_u": 1.0,
                "diffusion_v": 0.5,
                "iterations": 32,
            },
            "mandelbrot": {
                "max_iter": 64,
                "escape_radius": 4.0,
                "zoom": 1.0,
                "julia_mode": 0,
                "julia_cx": 0.355,
                "julia_cy": 0.355,
            },
            "sdf": {
                "primitive_type": 0,
                "sdf_shape": 0,
                "sdf_roundness": 0.5,
                "surface_noise": 0.8,
                "sdf_curl": 0.5,
                "sdf_turbulence": 0.5,
                "sdf_dom_warp": 0.5,
                "camera_distance": 3.5,
                "camera_orbit_speed": 1.0,
                "light_rotation": 1.0,
                "radius": 1.0,
                "twist": 0.5,
                "march_steps": 48,
                "camera_z": 3.0,
                "sdf_combination": 0,
                "sdf_smooth_k": 0.25,
            },
        }
        return copy.deepcopy(params.get(self.field_type, {}))

    def _default_warp_params(self) -> dict[str, float | int]:
        params = {
            "curl": {"strength": 0.5, "scale": 1.0, "octaves": 4},
            "vortex": {"strength": 0.3, "arms": 3.0, "twist_rate": 0.5},
            "twist": {"amount": 0.5, "iterations": 2},
            "none": {},
        }
        return copy.deepcopy(params.get(self.warp, {}))

    def _default_palette_params(self) -> dict[str, float | int]:
        params = {
            "cosine": {"palette_id": 0, "colour_shift": 0.0, "shift_speed": 0.1},
            "spectral": {"wavelength_shift": 0.0, "dispersion": 0.0},
            "plasma": {"speed": 1.0, "complexity": 1.0, "saturation": 1.0},
        }
        return copy.deepcopy(params.get(self.palette, {}))

    def _default_post_params(self) -> dict[str, float | int]:
        return {
            "bloom_threshold": 0.6,
            "bloom_intensity": 0.5,
            "bloom_radius": 1.0,
            "feedback_mode": 0,
            "feedback_speed": 1.0,
            "feedback_amount": 0.5,
            "distortion_amount": 0.5,
            "distortion_mode": 1,
            "glow_intensity": 0.5,
            "glow_colour_shift": 0.5,
            "chromatic_aberration_amount": 0.003,
            "chromatic_aberration_direction": 0.0,
        }

    def _default_feedback_params(self) -> dict[str, float | int]:
        return {"strength": 0.5, "speed": 1.0}

    def _default_lighting_params(self) -> dict[str, float | int]:
        return {"light_intensity": 0.8, "light_ambient": 0.2}

    def _default_camera_params(self) -> dict[str, float | int]:
        return {"camera_distance": 3.5, "camera_orbit_speed": 1.0, "camera_z": 3.0}

    def _default_particle_params(self) -> dict[str, float | int]:
        return {"particle_density": 1.0, "particle_size": 0.02}

    def _default_animation_params(self) -> dict[str, float | int]:
        return {
            "anim_zoom": 1.0,
            "anim_rot": 0.0,
            "anim_cx": 0.5,
            "anim_cy": 0.5,
            "anim_dx": 0.0,
            "anim_dy": 0.0,
            "anim_sx": 1.0,
            "anim_sy": 1.0,
            "anim_warp": 0.0,
            "anim_wave_r": 0.5,
            "anim_wave_g": 0.5,
            "anim_wave_b": 0.5,
            "anim_wave_x": 0.5,
            "anim_wave_y": 0.5,
            "anim_ob_size": 0.0,
            "anim_ob_r": 0.0,
            "anim_ob_g": 0.0,
            "anim_ob_b": 0.0,
            "anim_ob_a": 0.0,
            "anim_ib_size": 0.0,
            "anim_ib_r": 0.0,
            "anim_ib_g": 0.0,
            "anim_ib_b": 0.0,
            "anim_ib_a": 0.0,
            "anim_mv_x": 0.0,
            "anim_mv_y": 0.0,
            "anim_mv_l": 0.0,
            "anim_mv_r": 0.0,
            "anim_mv_g": 0.0,
            "anim_mv_b": 0.0,
            "anim_mv_a": 0.0,
            "anim_zoom_target": 1.0,
            "anim_rot_target": 0.0,
            "anim_cx_target": 0.5,
            "anim_cy_target": 0.5,
            "anim_warp_target": 0.0,
            "anim_smooth_zoom": 0.02,
            "anim_smooth_rot": 0.02,
            "anim_smooth_cx": 0.01,
            "anim_smooth_cy": 0.01,
            "anim_smooth_warp": 0.03,
            "anim_smooth_wave": 0.1,
        }

    def mutate(self, strength: float | None = None, large: bool = False) -> None:
        """Mutate the genome: change modules or mutate parameters.

        Args:
            strength: Optional override for the mutation step size.
            large: When True, apply a boosted, multi-step mutation used when an
                organism outlives its lifespan to force succession.
        """
        if strength is not None:
            self._strength = float(strength)
        if large:
            self._strength = min(self._strength * 2.0 + 0.1, 1.0)

        self._mutate_once()
        if large:
            self._mutate_once()

    def _mutate_once(self) -> None:
        mutation_type = random.choice(
            [
                "replace_module",
                "duplicate_module",
                "split_module",
                "fuse_modules",
                "invent_module",
                "replace_module",
            ]
        )
        mutation_desc = ""

        if mutation_type == "replace_module":
            mutation_desc = self._mutate_replace_module()
        elif mutation_type == "duplicate_module":
            mutation_desc = self._mutate_duplicate_module()
        elif mutation_type == "split_module":
            mutation_desc = self._mutate_split_module()
        elif mutation_type == "fuse_modules":
            mutation_desc = self._mutate_fuse_modules()
        elif mutation_type == "invent_module":
            mutation_desc = self._mutate_invent_module()

        if mutation_desc:
            self.mutations.append(mutation_desc)
            if len(self.mutations) > MAX_MUTATION_HISTORY:
                del self.mutations[:-MAX_MUTATION_HISTORY]

    def _mutate_replace_module(self) -> str:
        module_choices = [
            ("field_type", AVAILABLE_FIELDS, self.field_params, self._default_field_params),
            ("warp", AVAILABLE_WARPS, self.warp_params, self._default_warp_params),
            ("palette", AVAILABLE_PALETTES, self.palette_params, self._default_palette_params),
            (
                "feedback_mode",
                AVAILABLE_FEEDBACK,
                self.feedback_params,
                self._default_feedback_params,
            ),
            (
                "lighting",
                ["none", "lighting"],
                self.lighting_params,
                self._default_lighting_params,
            ),
            ("camera", ["none", "camera"], self.camera_params, self._default_camera_params),
            (
                "particles",
                ["none", "particle"],
                self.particle_params,
                self._default_particle_params,
            ),
        ]
        attr, choices, params, default_fn = random.choice(module_choices)
        old = getattr(self, attr)
        new_choices = [c for c in choices if c != old]
        if not new_choices:
            return ""
        new_val = random.choice(new_choices)
        setattr(self, attr, new_val)
        if params is not None and default_fn is not None:
            setattr(self, attr + "_params", default_fn())
        return f"replace:{attr}={new_val}"

    def _mutate_duplicate_module(self) -> str:
        targets = [
            ("post", self.post, self.post_params),
            ("field", self.extra_fields, self.extra_field_params),
            ("warp", self.extra_warps, self.extra_warp_params),
            ("palette", self.extra_palettes, self.extra_palette_params),
            ("feedback", self.extra_feedbacks, self.extra_feedback_params),
        ]
        target_name, target_list, target_params_list = random.choice(targets)
        source_map = {
            "post": (self.post, self.post_params),
            "field": (self.field_type, self.field_params),
            "warp": (self.warp, self.warp_params),
            "palette": (self.palette, self.palette_params),
            "feedback": (self.feedback_mode, None),
        }
        src_name, src = source_map[target_name]
        if target_name == "post":
            if len(self.post) >= 4:
                return ""
            available = [m for m in AVAILABLE_POST if m not in self.post]
            if not available:
                return ""
            new_mod = random.choice(available)
            self.post.append(new_mod)
            return f"duplicate:post+{new_mod}"
        else:
            if len(target_list) >= 2:
                return ""
            module_name = src_name
            target_list.append(module_name)
            new_params = copy.deepcopy(src) if src else {}
            for k in new_params:
                new_params[k] = float(new_params[k]) * random.uniform(0.8, 1.2)
            target_params_list.append(new_params)
            return f"duplicate:{target_name}+{module_name}"

    def _mutate_split_module(self) -> str:
        if len(self.post) >= 4:
            return ""
        if not self.post:
            return ""
        idx = random.randrange(len(self.post))
        removed = self.post.pop(idx)
        available = [m for m in AVAILABLE_POST if m not in self.post]
        if not available:
            self.post.insert(idx, removed)
            return ""
        new_mod = random.choice(available)
        self.post.append(new_mod)
        if True:
            split_params = copy.deepcopy(self.post_params)
            for k in split_params:
                split_params[k] = float(split_params[k]) * random.uniform(0.5, 0.8)
            return f"split:{removed}->{new_mod}"
        return f"split:post+{new_mod}"

    def _mutate_fuse_modules(self) -> str:
        if len(self.post) <= 1:
            return ""
        idx_a = random.randrange(len(self.post))
        idx_b = random.randrange(len(self.post))
        while idx_b == idx_a:
            idx_b = random.randrange(len(self.post))
        name_a = self.post[idx_a]
        name_b = self.post[idx_b]
        if random.random() < 0.5:
            removed = self.post.pop(max(idx_a, idx_b))
            return f"fuse:{name_a}+{name_b}={removed}"
        else:
            kept = random.choice([name_a, name_b])
            self.post[idx_a] = kept
            removed = self.post.pop(idx_b)
            return f"fuse:{name_a}+{name_b}={kept}"

    def _mutate_invent_module(self) -> str:
        module_choices = [
            ("field_type", AVAILABLE_FIELDS, self.field_params, self._default_field_params),
            ("warp", AVAILABLE_WARPS, self.warp_params, self._default_warp_params),
            ("palette", AVAILABLE_PALETTES, self.palette_params, self._default_palette_params),
            (
                "feedback_mode",
                AVAILABLE_FEEDBACK,
                self.feedback_params,
                self._default_feedback_params,
            ),
            (
                "lighting",
                ["none", "lighting"],
                self.lighting_params,
                self._default_lighting_params,
            ),
            ("camera", ["none", "camera"], self.camera_params, self._default_camera_params),
            (
                "particles",
                ["none", "particle"],
                self.particle_params,
                self._default_particle_params,
            ),
        ]
        attr, choices, _, _ = random.choice(module_choices)
        new_val = random.choice(choices)
        setattr(self, attr, new_val)
        if attr == "field_type":
            self.field_params = self._default_field_params()
        elif attr == "warp":
            self.warp_params = self._default_warp_params()
        elif attr == "palette":
            self.palette_params = self._default_palette_params()
        elif attr == "feedback_mode":
            self.feedback_params = self._default_feedback_params()
        elif attr == "lighting":
            self.lighting_params = self._default_lighting_params()
        elif attr == "camera":
            self.camera_params = self._default_camera_params()
        elif attr == "particles":
            self.particle_params = self._default_particle_params()
        if random.random() < 0.4 and len(self.post) < 4:
            available = [m for m in AVAILABLE_POST if m not in self.post]
            if available:
                self.post.append(random.choice(available))
                return f"invent:{attr}={new_val},post+..."
        return f"invent:{attr}={new_val}"

    def record_accept(self, accepted: bool) -> None:
        self._recent_accepts.append(1 if accepted else 0)
        if len(self._recent_accepts) > self._window_size:
            self._recent_accepts.pop(0)

    def adapt_strength(self, lo: float, hi: float) -> None:
        if not self._recent_accepts:
            return
        rate = sum(self._recent_accepts) / len(self._recent_accepts)
        if rate < 0.2:
            self._strength = max(lo, self._strength * 0.7)
        elif rate > 0.6:
            self._strength = min(hi, self._strength * 1.3)

    def crossover(self, other: ModularGenome) -> ModularGenome:
        child = ModularGenome(
            field_type=random.choice([self.field_type, other.field_type]),
            warp=random.choice([self.warp, other.warp]),
            palette=random.choice([self.palette, other.palette]),
            feedback_mode=random.choice([self.feedback_mode, other.feedback_mode]),
            lighting=random.choice([self.lighting, other.lighting]),
            camera=random.choice([self.camera, other.camera]),
            particles=random.choice([self.particles, other.particles]),
            post=random.choice([copy.deepcopy(self.post), copy.deepcopy(other.post)]),
            extra_fields=random.choice(
                [copy.deepcopy(self.extra_fields), copy.deepcopy(other.extra_fields)]
            ),
            extra_warps=random.choice(
                [copy.deepcopy(self.extra_warps), copy.deepcopy(other.extra_warps)]
            ),
            extra_palettes=random.choice(
                [copy.deepcopy(self.extra_palettes), copy.deepcopy(other.extra_palettes)]
            ),
            extra_feedbacks=random.choice(
                [copy.deepcopy(self.extra_feedbacks), copy.deepcopy(other.extra_feedbacks)]
            ),
            generation=max(self.generation, other.generation) + 1,
            parent_id=self.lineage,
        )
        child.field_params = self._merge_params(self.field_params, other.field_params)
        child.warp_params = self._merge_params(self.warp_params, other.warp_params)
        child.palette_params = self._merge_params(self.palette_params, other.palette_params)
        child.feedback_params = self._merge_params(self.feedback_params, other.feedback_params)
        child.lighting_params = self._merge_params(self.lighting_params, other.lighting_params)
        child.camera_params = self._merge_params(self.camera_params, other.camera_params)
        child.particle_params = self._merge_params(self.particle_params, other.particle_params)
        child.post_params = self._merge_params(self.post_params, other.post_params)
        child.extra_field_params = self._merge_extra_params(
            self.extra_field_params, other.extra_field_params
        )
        child.extra_warp_params = self._merge_extra_params(
            self.extra_warp_params, other.extra_warp_params
        )
        child.extra_palette_params = self._merge_extra_params(
            self.extra_palette_params, other.extra_palette_params
        )
        child.extra_feedback_params = self._merge_extra_params(
            self.extra_feedback_params, other.extra_feedback_params
        )
        return child

    def _merge_extra_params(self, a: list[dict], b: list[dict]) -> list[dict]:
        merged = copy.deepcopy(a)
        for i, v in enumerate(b):
            if i < len(merged):
                for k in v:
                    if random.random() < 0.5:
                        merged[i][k] = v[k]
            else:
                merged.append(copy.deepcopy(v))
        return merged

    def metamorphosis(self) -> None:
        """Genetic mutation event (trick 10): a rare dramatic jump to a new
        species. Resets the module selections (field/warp/palette/feedback_mode),
        swaps one post operator and reseeds all module parameters, mimicking a
        metamorphosis / evolution jump rather than a small local mutation.
        """
        self.field_type = random.choice(AVAILABLE_FIELDS)
        self.warp = random.choice(AVAILABLE_WARPS)
        self.palette = random.choice(AVAILABLE_PALETTES)
        self.feedback_mode = random.choice(AVAILABLE_FEEDBACK)
        self.lighting = random.choice(["none", "lighting"])
        self.camera = random.choice(["none", "camera"])
        self.particles = random.choice(["none", "particle"])
        if self.post and random.random() < 0.5:
            self.post.pop(random.randrange(len(self.post)))
        available = [m for m in AVAILABLE_POST if m not in self.post]
        if available:
            self.post.append(random.choice(available))
        self.field_params = self._default_field_params()
        self.warp_params = self._default_warp_params()
        self.palette_params = self._default_palette_params()
        self.feedback_params = self._default_feedback_params()
        self.lighting_params = self._default_lighting_params()
        self.camera_params = self._default_camera_params()
        self.particle_params = self._default_particle_params()
        self.post_params = self._default_post_params()
        self.mutations.append("metamorphosis")

    def _merge_params(self, a: dict, b: dict) -> dict:

        merged = copy.deepcopy(a)
        for k, v in b.items():
            if (k in merged and random.random() < 0.5) or k not in merged:
                merged[k] = v
        return merged

    def clone(self) -> ModularGenome:
        g = ModularGenome(
            field_type=self.field_type,
            warp=self.warp,
            palette=self.palette,
            feedback_mode=self.feedback_mode,
            lighting=self.lighting,
            camera=self.camera,
            particles=self.particles,
            post=copy.deepcopy(self.post),
            extra_fields=copy.deepcopy(self.extra_fields),
            extra_warps=copy.deepcopy(self.extra_warps),
            extra_palettes=copy.deepcopy(self.extra_palettes),
            extra_feedbacks=copy.deepcopy(self.extra_feedbacks),
            generation=self.generation,
            parent_id=self.parent_id,
        )
        g.field_params = copy.deepcopy(self.field_params)
        g.warp_params = copy.deepcopy(self.warp_params)
        g.palette_params = copy.deepcopy(self.palette_params)
        g.feedback_params = copy.deepcopy(self.feedback_params)
        g.lighting_params = copy.deepcopy(self.lighting_params)
        g.camera_params = copy.deepcopy(self.camera_params)
        g.particle_params = copy.deepcopy(self.particle_params)
        g.post_params = copy.deepcopy(self.post_params)
        g.extra_field_params = copy.deepcopy(self.extra_field_params)
        g.extra_warp_params = copy.deepcopy(self.extra_warp_params)
        g.extra_palette_params = copy.deepcopy(self.extra_palette_params)
        g.extra_feedback_params = copy.deepcopy(self.extra_feedback_params)

        # Copy animation parameters
        anim_attrs = [k for k in self.__dict__ if k.startswith("anim_")]
        for attr in anim_attrs:
            setattr(g, attr, getattr(self, attr))

        g.fitness = self.fitness
        g.mutations = list(self.mutations)
        g._strength = self._strength
        g._recent_accepts = list(self._recent_accepts)
        return g

    def shader_signature(self) -> dict:
        """Return only the genome fields that affect the generated shader.

        Used as the :class:`ShaderCache` key so an unchanged phenotype reuses
        its compiled GL program. Excludes lineage / generation / fitness /
        mutation history, which churn every generation and would defeat the
        cache (forcing recompiles and eviction leaks).
        """
        anim_keys = [
            "anim_zoom", "anim_rot", "anim_cx", "anim_cy", "anim_dx", "anim_dy",
            "anim_sx", "anim_sy", "anim_warp", "anim_wave_r", "anim_wave_g",
            "anim_wave_b", "anim_wave_x", "anim_wave_y", "anim_ob_size",
            "anim_ob_r", "anim_ob_g", "anim_ob_b", "anim_ob_a", "anim_ib_size",
            "anim_ib_r", "anim_ib_g", "anim_ib_b", "anim_ib_a", "anim_mv_x",
            "anim_mv_y", "anim_mv_l", "anim_mv_r", "anim_mv_g", "anim_mv_b",
            "anim_mv_a", "anim_smooth_zoom", "anim_smooth_rot", "anim_smooth_cx",
            "anim_smooth_cy", "anim_smooth_warp", "anim_smooth_wave",
        ]
        return {
            "field_type": self.field_type,
            "warp": self.warp,
            "palette": self.palette,
            "feedback_mode": self.feedback_mode,
            "lighting": self.lighting,
            "camera": self.camera,
            "particles": self.particles,
            "post": list(self.post),
            "field_params": dict(self.field_params),
            "warp_params": dict(self.warp_params),
            "palette_params": dict(self.palette_params),
            "feedback_params": dict(self.feedback_params),
            "lighting_params": dict(self.lighting_params),
            "camera_params": dict(self.camera_params),
            "particle_params": dict(self.particle_params),
            "post_params": dict(self.post_params),
            "animation": {k: getattr(self, k, 0) for k in anim_keys},
        }

    def to_dict(self) -> dict:
        anim_keys = [
            "anim_zoom", "anim_rot", "anim_cx", "anim_cy", "anim_dx", "anim_dy",
            "anim_sx", "anim_sy", "anim_warp", "anim_wave_r", "anim_wave_g",
            "anim_wave_b", "anim_wave_x", "anim_wave_y", "anim_ob_size",
            "anim_ob_r", "anim_ob_g", "anim_ob_b", "anim_ob_a", "anim_ib_size",
            "anim_ib_r", "anim_ib_g", "anim_ib_b", "anim_ib_a", "anim_mv_x",
            "anim_mv_y", "anim_mv_l", "anim_mv_r", "anim_mv_g", "anim_mv_b",
            "anim_mv_a", "anim_zoom_target", "anim_rot_target", "anim_cx_target",
            "anim_cy_target", "anim_warp_target", "anim_smooth_zoom", "anim_smooth_rot",
            "anim_smooth_cx", "anim_smooth_cy", "anim_smooth_warp", "anim_smooth_wave",
        ]
        return {
            "field": self.field_type,
            "warp": self.warp,
            "palette": self.palette,
            "feedback_mode": self.feedback_mode,
            "lighting": self.lighting,
            "camera": self.camera,
            "particles": self.particles,
            "post": list(self.post),
            "field_params": copy.deepcopy(self.field_params),
            "warp_params": copy.deepcopy(self.warp_params),
            "palette_params": copy.deepcopy(self.palette_params),
            "feedback_params": copy.deepcopy(self.feedback_params),
            "lighting_params": copy.deepcopy(self.lighting_params),
            "camera_params": copy.deepcopy(self.camera_params),
            "particle_params": copy.deepcopy(self.particle_params),
            "post_params": copy.deepcopy(self.post_params),
            "animation": {k: getattr(self, k) for k in anim_keys},
            "_lineage": self.lineage,
            "_generation": self.generation,
            "_fitness": self.fitness,
            "_parent_id": self.parent_id,
            "_mutations": list(self.mutations),
        }

    def from_dict(self, data: dict) -> None:
        # Backward compatible: accept both "field" and "field_type"
        self.field_type = data.get("field_type", data.get("field", "fbm"))
        self.warp = data.get("warp", "curl")
        self.palette = data.get("palette", "cosine")
        self.feedback_mode = data.get("feedback_mode", "spiral")
        self.lighting = data.get("lighting", "none")
        self.camera = data.get("camera", "none")
        self.particles = data.get("particles", "none")
        self.post = list(data.get("post", ["glow"]))
        self.field_params = copy.deepcopy(data.get("field_params", self._default_field_params()))
        self.warp_params = copy.deepcopy(
            data.get("warp_params", self._default_warp_params())
        )
        self.palette_params = copy.deepcopy(
            data.get("palette_params", self._default_palette_params())
        )
        self.feedback_params = copy.deepcopy(
            data.get("feedback_params", self._default_feedback_params())
        )
        self.lighting_params = copy.deepcopy(
            data.get("lighting_params", self._default_lighting_params())
        )
        self.camera_params = copy.deepcopy(
            data.get("camera_params", self._default_camera_params())
        )
        self.particle_params = copy.deepcopy(
            data.get("particle_params", self._default_particle_params())
        )
        self.post_params = copy.deepcopy(data.get("post_params", self._default_post_params()))

        # Animation parameters
        anim_data = data.get("animation", {})
        for k, v in anim_data.items():
            setattr(self, k, v)

        self.lineage = data.get("_lineage", self.lineage or str(uuid.uuid4())[:8])
        self.generation = int(data.get("_generation", 0))
        self.fitness = float(data.get("_fitness", 0.0))
        self.parent_id = data.get("_parent_id")
        self.mutations = list(data.get("_mutations", []))[-MAX_MUTATION_HISTORY:]

    def _all_params(self) -> dict[str, float | int]:
        """Merge every module parameter dict for distance comparisons."""
        merged: dict[str, float | int] = {}
        for d in (
            self.field_params,
            self.warp_params,
            self.palette_params,
            self.feedback_params,
            self.lighting_params,
            self.camera_params,
            self.particle_params,
            self.post_params,
        ):
            merged.update(d)
        return merged

    def distance(self, other: ModularGenome) -> float:
        """Normalized 0..1 phenotypic distance to another genome.

        Module differences count as a full step; post-module set differences
        count per module; numeric parameters are compared relative to their
        magnitude.
        """
        d = 0.0
        n = 0

        for attr in (
            "field_type",
            "warp",
            "palette",
            "feedback_mode",
            "lighting",
            "camera",
            "particles",
        ):
            if getattr(self, attr) != getattr(other, attr):
                d += 1.0
            n += 1

        d += len(set(self.post) ^ set(other.post))
        n += 1

        mine = self._all_params()
        theirs = other._all_params()
        for k in set(mine) | set(theirs):
            a = mine.get(k)
            b = theirs.get(k)
            if a is None or b is None:
                d += 1.0
            else:
                denom = max(abs(float(a)), abs(float(b)), 1e-6)
                d += min(abs(float(a) - float(b)) / denom, 1.0)
            n += 1

        return d / max(n, 1)

    def environmental_affinity(self, audio: dict[str, float]) -> float:
        """Bonus for how well this genome suits the current audio.

        Bass-heavy music favours structural fields (mandelbrot/sdf/voronoi)
        and vortex/twist warps; treble favours glow; calm/ambient favours slow
        reaction-diffusion.
        """
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        score = 0.0

        if bass > 0.7 and self.field_type in ("mandelbrot", "sdf", "voronoi"):
            score += 0.2
        if treb > 0.7 and "glow" in self.post:
            score += 0.1
        if self.warp in ("vortex", "twist"):
            score += 0.1 * bass
        if self.field_type == "reaction_diffusion" and bass < 0.3:
            score += 0.1
        return score

    def update_animation(self, audio: dict[str, float], t: float, beat: float) -> None:
        """Update per-frame animation parameters (Milkdrop-style).

        Computes target values from audio/time and smoothly interpolates
        current values toward targets. This mimics Milkdrop's per-frame
        equations that animate zoom, rot, cx, cy, warp, waves, etc.
        """
        bass = float(audio.get("bass", 0.0))
        mid = float(audio.get("mid", 0.0))
        treb = float(audio.get("treb", 0.0))

        # Zoom: pulse with bass, slower drift with mid
        self.anim_zoom_target = 1.0 + 0.15 * bass + 0.05 * mid * math.sin(t * 0.5)
        # Rotation: continuous slow rotation + beat kicks
        self.anim_rot_target += 0.01 * (1.0 + 0.5 * treb) + 0.1 * beat
        # Center: slow organic drift
        self.anim_cx_target = 0.5 + 0.1 * math.sin(t * 0.13 + bass * 2.0)
        self.anim_cy_target = 0.5 + 0.1 * math.cos(t * 0.11 + mid * 1.5)
        # Warp: audio-reactive distortion
        self.anim_warp_target = 0.1 * bass + 0.05 * treb * math.sin(t * 0.7)

        # Smooth interpolation toward targets
        self.anim_zoom += (self.anim_zoom_target - self.anim_zoom) * self.anim_smooth_zoom
        self.anim_rot += (self.anim_rot_target - self.anim_rot) * self.anim_smooth_rot
        self.anim_cx += (self.anim_cx_target - self.anim_cx) * self.anim_smooth_cx
        self.anim_cy += (self.anim_cy_target - self.anim_cy) * self.anim_smooth_cy
        self.anim_warp += (self.anim_warp_target - self.anim_warp) * self.anim_smooth_warp

        # Wave parameters (oscillate with audio)
        wave_speed = 0.5 + bass
        self.anim_wave_r = 0.5 + 0.3 * math.sin(t * wave_speed + 0.0)
        self.anim_wave_g = 0.5 + 0.3 * math.sin(t * wave_speed + 2.094)
        self.anim_wave_b = 0.5 + 0.3 * math.sin(t * wave_speed + 4.189)
        self.anim_wave_x = 0.5 + 0.2 * math.sin(t * 0.7 + bass)
        self.anim_wave_y = 0.5 + 0.2 * math.cos(t * 0.7 + mid)

        # Beat-reactive kicks
        if beat > 0.5:
            self.anim_zoom = min(1.3, self.anim_zoom + 0.08 * beat)
            self.anim_rot += 0.15 * beat
            self.anim_warp = min(0.5, self.anim_warp + 0.1 * beat)

    def get_animation_uniforms(self) -> dict[str, float]:
        """Return current animation values as uniform dict for shader."""
        return {
            "anim_zoom": self.anim_zoom,
            "anim_rot": self.anim_rot,
            "anim_cx": self.anim_cx,
            "anim_cy": self.anim_cy,
            "anim_dx": self.anim_dx,
            "anim_dy": self.anim_dy,
            "anim_sx": self.anim_sx,
            "anim_sy": self.anim_sy,
            "anim_warp": self.anim_warp,
            "anim_wave_r": self.anim_wave_r,
            "anim_wave_g": self.anim_wave_g,
            "anim_wave_b": self.anim_wave_b,
            "anim_wave_x": self.anim_wave_x,
            "anim_wave_y": self.anim_wave_y,
            "anim_ob_size": self.anim_ob_size,
            "anim_ob_r": self.anim_ob_r,
            "anim_ob_g": self.anim_ob_g,
            "anim_ob_b": self.anim_ob_b,
            "anim_ob_a": self.anim_ob_a,
            "anim_ib_size": self.anim_ib_size,
            "anim_ib_r": self.anim_ib_r,
            "anim_ib_g": self.anim_ib_g,
            "anim_ib_b": self.anim_ib_b,
            "anim_ib_a": self.anim_ib_a,
            "anim_mv_x": self.anim_mv_x,
            "anim_mv_y": self.anim_mv_y,
            "anim_mv_l": self.anim_mv_l,
            "anim_mv_r": self.anim_mv_r,
            "anim_mv_g": self.anim_mv_g,
            "anim_mv_b": self.anim_mv_b,
            "anim_mv_a": self.anim_mv_a,
        }
