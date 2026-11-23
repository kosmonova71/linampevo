"""Configuration dataclasses for the evolutionary visualizer.

Contains MultiResTier, EvoConfig, and helper utilities.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical constants live in evo_visualizer.constants; re-export here for
# backward compatibility so existing `from evo_visualizer.config import X`
# imports keep working.  New code should import from constants directly.
from evo_visualizer.constants import (  # noqa: F401  (re-export)
    AUDIO_CLAMP_MAX,
    AUDIO_CLAMP_MIN,
    AUDIO_KEYS,
    FULLSCREEN_QUAD_VERTICES,
    GLSL_FLOAT_PRECISION,
    MAX_MUTATION_HISTORY,
    QUAD_STRIDE_BYTES,
    QUAD_VERTEX_COUNT,
    TWO_PI,
)


def _lerp_cfg(lo: float, hi: float, t: float) -> float:
    """Linear interpolation clamped to [lo, hi] by t in [0, 1].

    Used by the adaptive renderer-complexity controller to map the single
    quality scalar onto each concrete complexity knob.
    """
    t = max(0.0, min(1.0, float(t)))
    return float(lo) + (float(hi) - float(lo)) * t


# ---------------------------------------------------------------------------
# Feature profiles.
#
# EvoConfig enables dozens of experimental systems by default, and many
# combinations are untested and fight for the same resources (FBOs, readback,
# population).  A small set of named profiles picks a known-good subset and
# is applied via :meth:`EvoConfig.apply_profile`.  Individual flags can still
# be toggled after the profile is applied.
# ---------------------------------------------------------------------------
class EvoProfile:
    """A named subset of :class:`EvoConfig` flags."""

    MINIMAL = "minimal"
    SPATIAL = "spatial"
    GPU = "gpu"
    FULL = "full"


# Module-level lookup table (kept out of the class body so ruff's RUF012
# mutable-class-attribute warning does not fire; the dicts below are
# immutable in practice -- callers only read them).
_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    EvoProfile.MINIMAL: {
        "description": "Single genome, CPU fitness, no spatial ecosystem.",
        "on": {
            "enable_visual_fitness": True,
            "enable_lineage": True,
            "enable_feedback_operators": True,
            "enable_steady_state_evolution": True,
            "enable_cultural_evolution": True,
            "enable_multi_resolution_evolution": False,
            "enable_spatial_ecosystem": False,
            "enable_multi_species": False,
            "enable_gpu_compute": False,
            "enable_gpu_fitness_kernels": False,
            "enable_dynamic_shader_graph": False,
            "enable_emergent_camera": False,
            "enable_gene_regulatory_network": False,
            "enable_continuous_world": False,
            "enable_region_rendering": False,
            "enable_organism_maps": False,
            "enable_niches": False,
        },
        "off": {},
    },
    EvoProfile.SPATIAL: {
        "description": "Adds the spatial ecosystem, niches, senses/organs/brain.",
        "on": {
            "enable_spatial_ecosystem": True,
            "enable_senses": True,
            "enable_organs": True,
            "enable_brain": True,
            "enable_signals": True,
            "enable_flocking": True,
            "enable_predator_prey": True,
            "enable_infection": True,
            "enable_climate": True,
            "enable_complexity": True,
            "enable_migration": True,
            "enable_extinction": True,
            "enable_energy_economy": True,
            "enable_niches": True,
            "enable_organism_maps": True,
        },
        "off": {
            "enable_multi_resolution_evolution": False,
            "enable_gpu_compute": False,
            "enable_gpu_fitness_kernels": False,
            "enable_dynamic_shader_graph": False,
            "enable_emergent_camera": False,
            "enable_gene_regulatory_network": False,
            "enable_continuous_world": False,
            "enable_region_rendering": False,
        },
    },
    EvoProfile.GPU: {
        "description": "Adds GPU compute fitness, multi-species, async compile.",
        "on": {
            "enable_multi_species": True,
            "enable_gpu_compute": True,
            "enable_gpu_fitness_kernels": True,
            "enable_async_shader_compile": True,
            "enable_multi_resolution_evolution": True,
            "enable_dynamic_shader_graph": True,
        },
        "off": {
            "enable_spatial_ecosystem": False,
            "enable_region_rendering": False,
            "enable_organism_maps": False,
            "enable_niches": False,
        },
    },
    EvoProfile.FULL: {
        "description": "Every experimental system enabled.",
        "on": {},
        "off": {},
    },
}


@dataclass
class MultiResTier:
    """Definition of one resolution tier in the multi-resolution evolution pyramid.

    Attributes:
        name: Tier identifier (e.g. ``"exploration"``, ``"normal"``, ``"champions"``).
        res: Fitness-evaluation resolution (square FBO).
        population_size: Number of organisms kept at this tier.
        mutation_mult: Multiplier applied to mutation strength at this tier.
            Use ``>1.0`` for exploratory tiers and ``<1.0`` for refinement tiers.
        promote_interval: Number of evolution frames between promotion attempts.
    """

    name: str
    res: int
    population_size: int
    mutation_mult: float = 1.0
    promote_interval: int = 120


@dataclass
class EvoConfig:
    """Configuration for the evolutionary renderer.

    Attributes:
        feedback_alpha: Blending factor for feedback buffer (0.0-1.0)
        feedback_res: Resolution of feedback buffer textures (square)
        mutation_strength: Initial mutation step size
        mutation_strength_min: Minimum mutation strength bound
        mutation_strength_max: Maximum mutation strength bound
        debug: Enable debug logging and visualization
        score_weights: Weighted scoring for genome parameters
        enable_logging: Enable detailed logging output
        preset_dir: Directory for saving/loading presets
        brightness_base: Base brightness multiplier
        brightness_audio_scale: Audio influence on brightness
        population_size: Number of organisms in the population
        fitness_image_res: Resolution for fitness readback (square)
        enable_visual_fitness: Use image-based fitness instead of parameter score
        enable_lineage: Enable memory DNA lineage tracking
        enable_feedback_operators: Enable advanced feedback modes
        crossover_rate: Probability of crossover vs mutation in population
    """

    feedback_alpha: float = 0.85
    feedback_res: int = 512
    mutation_strength: float = 0.1
    mutation_strength_min: float = 0.01
    mutation_strength_max: float = 0.5
    debug: bool = False
    score_weights: dict[str, float] = field(default_factory=dict)
    enable_logging: bool = True
    preset_dir: Path | None = None
    brightness_base: float = 0.85
    brightness_audio_scale: float = 0.35
    population_size: int = 4
    fitness_image_res: int = 64
    enable_visual_fitness: bool = True
    enable_lineage: bool = True
    enable_feedback_operators: bool = True
    crossover_rate: float = 0.3
    enable_ecosystem_memory: bool = False
    enable_dream_mode: bool = False
    dream_threshold: float = 0.15
    enable_world_evolution: bool = False
    enable_music_pressure: bool = True
    music_pressure_weight: float = 0.3
    music_match_threshold: float = 0.75
    memory_db_path: str | None = None
    memory_watchdog_mb: float = 4096.0

    # ------------------------------------------------------------------
    # Genome entropy (spec: entropy per genome)
    #   - entropy_high: threshold above which a genome "invents"
    #   - entropy_low:  threshold below which a genome "refines"
    #   - stagnation_raise: entropy increment applied when fitness
    #       stagnation is detected, preventing local minima.
    #   - stagnation_window: number of frames without fitness
    #       improvement before stagnation is declared.
    # ------------------------------------------------------------------
    entropy_high: float = 0.7
    entropy_low: float = 0.2
    stagnation_raise: float = 0.15
    stagnation_window: int = 120

    # ------------------------------------------------------------------
    # GPU memory management (OOM post-mortem)
    #   - LRU shader cache       -> shader_cache_max_size
    #   - texture / FBO pooling  -> fbo_pool enabled by default
    #   - recycle replay buffers -> replay_max_frames (bounded ring)
    #   - adaptive history length -> *_history_min/max ramps under pressure
    #   - proactive watchdog      -> memory_budget_mb + trim triggers
    # ------------------------------------------------------------------
    shader_cache_max_size: int = 128
    enable_fbo_pool: bool = True
    replay_max_frames: int = 256
    replay_max_frames_min: int = 32
    feedback_history_min: int = 0
    lineage_max_records: int = 2000
    lineage_max_records_min: int = 250
    novelty_history_max: int = 200
    novelty_history_min: int = 32
    memory_budget_mb: float = 4096.0
    memory_trim_enabled: bool = True

    # ------------------------------------------------------------------
    # Preset transitions (specs 70)
    #   When a new genome / preset is promoted, the renderer can dissolve
    #   from the previous frame into the new one using a shader transition
    #   instead of an abrupt swap.  Each transition is a standalone .frag
    #   in the ``transitions/`` directory that samples two textures
    #   (``u_texture_from`` = the pre-transition frame, ``u_texture_to`` =
    #   the new genome's first frame) and blends them over a progress
    #   ramp.
    # ------------------------------------------------------------------
    enable_transitions: bool = True
    transition_dir: Path | None = None
    transition_kind: str = "crossfade"
    transition_duration_secs: float = 0.9
    transition_random: bool = True
    transition_audio_reactive: bool = True

    # ------------------------------------------------------------------
    # Evolving post-effects (``effects/`` shader folder)
    #   Every ``.frag`` in ``effects_dir`` is a candidate full-screen
    #   effect.  One effect is active at a time; its numeric uniforms act
    #   as genes that drift continuously, and on a timer the active effect
    #   crossfades into a newly selected one (transition shaders from the
    #   same pool drive the blend).  ``effects_dir=None`` defaults to the
    #   repo-level ``effects/`` directory.
    # ------------------------------------------------------------------
    enable_effects: bool = True
    effects_dir: Path | None = None
    effects_evolve_secs: float = 14.0
    effects_transition_secs: float = 1.4
    effects_param_mutation: float = 0.15
    effects_audio_reactive: bool = True
    effects_seed: int | None = None

    # ------------------------------------------------------------------
    # Folder textures (``textures/`` image pool)
    #   Effect shaders can sample still images from a folder through the
    #   ``u_image`` / ``u_image_next`` sampler uniforms.  The active image
    #   rotates on a timer so the imagery evolves over time; shaders can
    #   blend current -> next with ``u_image_blend`` and read pool state
    #   via ``u_image_index`` / ``u_image_count`` / ``u_image_aspect`` /
    #   ``u_image_seed``.  ``effects_texture_dir=None`` defaults to the
    #   repo-level ``textures/`` directory.  Requires Pillow + numpy.
    # ------------------------------------------------------------------
    enable_effect_textures: bool = True
    effects_texture_dir: Path | None = None
    effects_texture_evolve_secs: float = 20.0
    effects_texture_max_size: int = 512
    effects_texture_cache_size: int = 8

    # ------------------------------------------------------------------
    # Folder visualiser (``shaders/`` + ``effects/`` + ``textures/``)
    #   Opt-in mode that replaces the genetic genome on screen with a
    #   visualiser built only from the repo asset folders.  A full-screen
    #   shader from ``folder_shaders_dir`` (default: repo ``shaders/``) is
    #   the base layer; its numeric uniforms drift as genes and the active
    #   shader crossfades into a newly picked one every
    #   ``folder_evolve_secs``.  When ``folder_use_effects`` is set the
    #   renderer's EffectManager composites the evolving ``effects/``
    #   post chain (and the rotating ``textures/`` image pool) on top.
    # ------------------------------------------------------------------
    enable_folder_visualizer: bool = False
    folder_shaders_dir: Path | None = None
    folder_evolve_secs: float = 18.0
    folder_transition_secs: float = 1.6
    folder_param_mutation: float = 0.12
    folder_seed: int | None = None
    folder_use_effects: bool = True

    # ------------------------------------------------------------------
    # Spatial Ecosystem (specs 46-60)
    # ------------------------------------------------------------------
    enable_spatial_ecosystem: bool = False
    spatial_mode: str = "territories"
    spatial_blend: str = "soft"
    spatial_population_size: int = 18
    spatial_world_w: int = 64  # resource field grid resolution
    spatial_world_h: int = 64
    ecosystem_step_hz: float = 30.0
    enable_senses: bool = True  # 53 evolution of senses
    enable_organs: bool = True  # 54 internal organs
    enable_brain: bool = True  # 55 brain network
    enable_signals: bool = True  # 50 communication
    enable_flocking: bool = True  # 51 collective behaviour
    enable_predator_prey: bool = True  # 49 food chains
    enable_infection: bool = True  # infection spread between organisms
    enable_climate: bool = True  # 52 seasonal music climate
    enable_complexity: bool = True  # 57 evolution of complexity
    enable_migration: bool = True  # 58 migration
    enable_extinction: bool = True  # 59 extinction & recovery
    enable_energy_economy: bool = True  # 56 energy economy
    show_territories: bool = False  # debug overlay
    # Seconds between champion -> visual gene syncs. The copy is throttled
    # because it can invalidate the active shader: without a throttle a
    # spatial champion that mutates every frame would queue a shader
    # recompile every frame.
    spatial_champion_sync_interval: float = 0.5

    # ------------------------------------------------------------------
    # Cultural Evolution (spec 48)
    #   Organisms can copy, imitate, steal, modify, and teach
    #   successful techniques, achieving much faster convergence
    #   than genetic mutation alone.
    # ------------------------------------------------------------------
    enable_cultural_evolution: bool = True
    cultural_copy_rate: float = 0.3
    cultural_imitate_rate: float = 0.2
    cultural_steal_rate: float = 0.05
    cultural_modify_rate: float = 0.15
    cultural_teach_rate: float = 0.1
    cultural_convergence_window: int = 10

    # ------------------------------------------------------------------
    # Persistent world textures (heat / food / life / species / damage /
    # memory / history) that accumulate across frames instead of being
    # cleared each tick.  Visuals become landscapes rather than isolated
    # effects.
    # ------------------------------------------------------------------
    world_texture_size: int = 256
    world_texture_decay: float = 0.995

    # ------------------------------------------------------------------
    # Advanced developmental evolution (specs 16-30)
    # ------------------------------------------------------------------
    enable_gene_regulatory_network: bool = True  # 16 GRN
    enable_development: bool = True  # 17 life stages
    enable_continuous_mutation: bool = True  # 18 per-frame drift
    enable_multi_objective: bool = True  # 19 Pareto fitness
    enable_novelty_search: bool = True  # 20 reward difference
    enable_visual_novelty: bool = True  # 9 visual uniqueness vs history
    novelty_window_size: int = 200
    novelty_weight: float = 0.15
    enable_memory_evolution: bool = True  # 21 anticipation
    enable_emotional_state: bool = True  # 22 feelings
    enable_genome_repair: bool = True  # 23 self-repair
    enable_sexual_selection: bool = True  # 24 mate choice
    enable_renderer_evolution: bool = True  # 25 invent renderers
    enable_living_materials: bool = True  # 26 tissues
    enable_niches: bool = True  # 27 auto habitats
    enable_evolution_replay: bool = True  # 28 replay events
    enable_genetic_archaeology: bool = True  # 29 family tree
    enable_self_modifying: bool = True  # 30 evolution rewrites itself
    replay_window_secs: float = 300.0

    # ------------------------------------------------------------------
    # 31 Per-organism spatial maps (the "spatial GPU ecosystem" tier)
    # Every organism owns four evolving maps -- territory / food / pheromone /
    # danger -- packed into one GPU atlas and composited into its territory so
    # evolution becomes visible.
    # ------------------------------------------------------------------
    enable_organism_maps: bool = True  # 31 per-organism maps
    organism_map_res: int = 12  # tile resolution (res*res)
    organism_map_max_tiles: int = 256  # atlas capacity
    organism_map_strength: float = 0.6  # visual compositing weight
    organism_map_update_interval: int = 3  # ticks between map updates

    # ------------------------------------------------------------------
    # Features 6-10: multi-species, GPU compute, continuous world,
    # dynamic shader invention, emergent camera
    # ------------------------------------------------------------------
    enable_multi_species: bool = True  # 6 multi-species renderer
    max_species: int = 20  # 6 max species count
    enable_gpu_compute: bool = True  # 7 GPU compute fitness
    enable_gpu_fitness_kernels: bool = True  # 7b GPU fitness kernels
    gpu_fitness_res: int = 64  # Resolution for GPU fitness reduction
    gpu_fitness_weights: dict[str, float] = field(
        default_factory=lambda: {
            "beauty": 0.30,
            "motion": 0.20,
            "contrast": 0.10,
            "novelty": 0.15,
            "entropy": 0.10,
            "coverage": 0.05,
            "symmetry": 0.05,
            "energy": 0.05,
        }
    )
    enable_continuous_world: bool = True  # 8 persistent world
    enable_dynamic_shader_graph: bool = True  # 9 shader graph invention
    enable_emergent_camera: bool = True  # 10 camera organism
    camera_population_size: int = 3  # 10 camera organisms

    # ------------------------------------------------------------------
    # Adaptive LOD (spec 14)
    # ------------------------------------------------------------------
    enable_adaptive_lod: bool = True
    lod_near_px: int = 80
    lod_far_px: int = 25
    feedback_history: int = 0

    # ------------------------------------------------------------------
    # Specs 31-50: gene / module / renderer / ecosystem evolution
    # ------------------------------------------------------------------
    enable_gene_duplication: bool = True  # 31 copy & diverge genes
    enable_gene_deletion: bool = True  # 32 prune unused genes
    enable_gene_invention: bool = True  # 33 invent new genes
    enable_genome_modules: bool = True  # 34 modular genome
    enable_module_duplication: bool = True  # 35 duplicate modules
    enable_module_fusion: bool = True  # 36 fuse modules
    enable_module_death: bool = True  # 37 remove weak modules
    enable_dynamic_shader_assembly: bool = True  # 38 swappable modules
    enable_shader_graph_evolution: bool = True  # 39 evolve graph topology
    enable_new_rendering_passes: bool = True  # 40 invent passes
    enable_adaptive_resolution: bool = True  # 41 adaptive LOD
    enable_self_profiling: bool = True  # 42 organism knows its cost
    enable_energy_budget: bool = True  # 43 rendering costs energy
    enable_gpu_ecology: bool = True  # 44 territorial GPU
    enable_hybrid_organisms: bool = True  # 45 combine parents' modules
    enable_coevolution: bool = True  # 46 predator/prey pressure
    enable_visual_parasites: bool = True  # 47 small visual modifiers
    enable_symbiosis: bool = True  # 48 cooperative links
    enable_seasonal_evolution: bool = True  # 49 cycling environments
    enable_evolutionary_memory_forest: bool = True  # 50 tree-based restart

    # ------------------------------------------------------------------
    # Region-based rendering (ecosystem paints the screen together)
    # ------------------------------------------------------------------
    enable_region_rendering: bool = False
    region_layout: str = "split_left_right"
    region_count: int = 5
    region_blend: float = 0.02

    # ------------------------------------------------------------------
    # Environmental niches (spec: niche-based evolution)
    #   - niche_layout: how niches are assigned to regions
    #       (stripes/blocks/quadrants/radial/random)
    #   - niche_fitness_scale: how strongly niche mismatch penalises
    #       organisms (0 = off, 1 = strong)
    #   - niche_migration_rate: probability per step that an organism
    #       moves to an adjacent region
    #   - niche_overlay_opacity: visual tint strength per niche
    # ------------------------------------------------------------------
    niche_layout: str = "stripes"
    niche_fitness_scale: float = 0.3
    niche_migration_rate: float = 0.01
    niche_overlay_opacity: float = 0.08

    # ------------------------------------------------------------------
    # Steady-state evolution (continuous, per-frame micro-evolution)
    # ------------------------------------------------------------------
    enable_steady_state_evolution: bool = True
    steady_state_mutations_per_frame: int = 2
    # Dynamic mutation scheduling (spec 13): instead of a fixed
    # mutations/frame count, adapt it every frame from GPU load, FPS,audio
    # energy, population diversity, fitness stagnation and species count.
    enable_dynamic_mutation_schedule: bool = True
    steady_state_mutations_min: int = 0
    steady_state_mutations_max: int = 12
    # Weighting of each driver on the requested mutation rate (0..1).  Set a
    # weight to 0 to disable that channel entirely without code changes.
    schedule_weight_gpu: float = 0.9  # high GPU load -> fewer mutations
    schedule_weight_fps: float = 0.9  # low FPS -> fewer mutations
    schedule_weight_audio: float = 0.6  # loud music -> more mutations
    schedule_weight_diversity: float = 0.5  # diverse pop -> more exploration
    schedule_weight_stagnation: float = 1.0  # stalled fitness -> more mutations
    schedule_weight_species: float = 0.4  # more species -> more mutations
    schedule_fps_target: float = 60.0
    schedule_gpu_target: float = 0.7  # ~70% GPU is the sweet spot
    schedule_smoothing: float = 0.25  # EMA factor for the chosen rate
    # How often (in render frames) the population diversity (O(n²)) and species
    # count (O(n)) signals are recomputed by the dynamic mutation scheduler.
    # Both feed the EMA-smoothed mutation rate, so recomputing them every frame
    # is pure overhead — 4 frames strikes a balance between responsiveness and
    # cost.  Set to 1 to restore the original every-frame behaviour.
    diversity_recompute_interval: int = 4
    elite_fraction: float = 0.15
    tournament_size: int = 4
    enable_gene_fitness_tracking: bool = True
    champion_migration_interval_secs: float = 30.0

    # ------------------------------------------------------------------
    # Multi-resolution evolution (simultaneous tiered populations)
    # ------------------------------------------------------------------
    enable_multi_resolution_evolution: bool = False
    res_tiers: list[MultiResTier] = field(
        default_factory=lambda: [
            MultiResTier(
                name="exploration",
                res=32,
                population_size=24,
                mutation_mult=2.0,
                promote_interval=120,
            ),
            MultiResTier(
                name="normal", res=128, population_size=8, mutation_mult=1.0, promote_interval=180
            ),
            MultiResTier(
                name="champions",
                res=512,
                population_size=2,
                mutation_mult=0.5,
                promote_interval=300,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Asynchronous shader compilation (spec 6)
    # Compile shader GLSL source on a worker thread and swap the program in
    # on a later frame, so live mutations never stall the render loop.
    # ------------------------------------------------------------------
    enable_async_shader_compile: bool = True
    async_shader_max_pending: int = 64
    async_shader_uploads_per_frame: int = 4

    # ------------------------------------------------------------------
    # Adaptive renderer complexity (spec 41/6/15).
    #
    # Instead of only scaling resolution, a single quality scalar in [0,1]
    # drives EVERY expensive knob together: ray-march steps, FBM/fractal
    # layers, particle density, bloom taps, DoF taps, feedback echo history
    # and the internal render scale.  When frames run under budget the
    # controller ramps quality back up to 1.0 (full fidelity); when frames
    # blow the budget it ramps everything down in lockstep so the frame rate
    # is preserved.  At quality == 1.0 output is identical to the un-adaptive
    # renderer -- this is purely a budget guard.
    # ------------------------------------------------------------------
    enable_adaptive_complexity: bool = True
    perf_target_fps: float = 60.0
    # Frame-time hysteresis thresholds (ms).  Ramp quality up only when we are
    # comfortably under budget, ramp down when we exceed the frame budget.
    perf_up_threshold_ms: float = 13.5
    perf_down_threshold_ms: float = 17.5
    perf_adjustment_interval: float = 0.4  # seconds between quality steps
    perf_enable_adaptive_population: bool = True
    # Complexity floors/ceilings the quality scalar interpolates between.
    perf_min_raymarch_steps: int = 16
    perf_max_raymarch_steps: int = 32
    perf_min_fractal_layers: int = 4
    perf_max_fractal_layers: int = 8
    perf_min_feedback_scale: float = 0.5
    perf_max_feedback_scale: float = 1.0
    perf_min_internal_scale: float = 0.5
    perf_max_internal_scale: float = 1.0
    # Temporal feedback echo history collapses under pressure too.
    perf_min_feedback_history: int = 0
    perf_max_feedback_history: int = 4
    perf_min_particle_count: int = 2
    perf_max_particle_count: int = 16
    perf_min_blur_taps: int = 1
    perf_max_blur_taps: int = 8
    perf_min_lighting_samples: int = 2
    perf_max_lighting_samples: int = 16
    perf_min_shadow_samples: int = 1
    perf_max_shadow_samples: int = 8
    perf_min_bloom_passes: int = 1
    perf_max_bloom_passes: int = 4
    perf_min_dof_samples: int = 2
    perf_max_dof_samples: int = 16

    def __post_init__(self):
        """Initialize default score weights if not provided."""
        if not self.score_weights:
            self.score_weights = {
                "warp": 0.45,
                "noise_scale": 0.20,
                "feedback": 0.20,
                "symmetry": 0.15,
            }

        # Ensure preset_dir exists
        if self.preset_dir:
            self.preset_dir = Path(self.preset_dir)
            self.preset_dir.mkdir(parents=True, exist_ok=True)

        # Validate extended config
        self.population_size = max(self.population_size, 1)
        if self.crossover_rate < 0.0:
            self.crossover_rate = 0.0
        elif self.crossover_rate > 1.0:
            self.crossover_rate = 1.0
        self.dream_threshold = max(0.0, min(1.0, float(self.dream_threshold)))
        self.music_match_threshold = max(0.0, min(1.0, float(self.music_match_threshold)))

        # Validate spatial ecosystem config
        self.spatial_population_size = max(2, int(self.spatial_population_size))
        self.spatial_world_w = max(8, int(self.spatial_world_w))
        self.spatial_world_h = max(8, int(self.spatial_world_h))
        self.lod_near_px = max(1, int(self.lod_near_px))
        self.lod_far_px = max(1, int(self.lod_far_px))
        if self.lod_far_px > self.lod_near_px:
            self.lod_far_px, self.lod_near_px = self.lod_near_px, self.lod_far_px
        self.ecosystem_step_hz = max(1.0, float(self.ecosystem_step_hz))
        self.spatial_mode = str(getattr(self, "spatial_mode", "territories")).lower()
        valid_modes = {"territories", "colonies", "clouds", "flocks", "cells", "islands", "veins"}
        if self.spatial_mode not in valid_modes:
            self.spatial_mode = "territories"
        self.spatial_blend = str(getattr(self, "spatial_blend", "soft")).lower()
        valid_blends = {"hard", "soft", "glow", "fluid"}
        if self.spatial_blend not in valid_blends:
            self.spatial_blend = "soft"
        self.spatial_champion_sync_interval = max(
            0.0, float(getattr(self, "spatial_champion_sync_interval", 0.5))
        )

        # Validate persistent world texture config
        self.world_texture_size = max(32, int(self.world_texture_size))
        self.world_texture_decay = max(0.9, min(1.0, float(self.world_texture_decay)))

        # Validate steady-state evolution config
        self.enable_steady_state_evolution = bool(self.enable_steady_state_evolution)
        self.steady_state_mutations_per_frame = max(1, int(self.steady_state_mutations_per_frame))
        self.steady_state_mutations_min = max(0, int(self.steady_state_mutations_min))
        self.steady_state_mutations_max = max(
            self.steady_state_mutations_min + 1, int(self.steady_state_mutations_max)
        )
        self.elite_fraction = max(0.0, min(0.5, float(self.elite_fraction)))
        self.tournament_size = max(2, int(self.tournament_size))
        self.enable_gene_fitness_tracking = bool(self.enable_gene_fitness_tracking)
        self.champion_migration_interval_secs = max(
            5.0, float(self.champion_migration_interval_secs)
        )
        self.enable_dynamic_mutation_schedule = bool(self.enable_dynamic_mutation_schedule)
        self.schedule_fps_target = max(1.0, float(self.schedule_fps_target))
        self.schedule_gpu_target = max(0.05, min(1.0, float(self.schedule_gpu_target)))
        self.schedule_smoothing = max(0.0, min(1.0, float(self.schedule_smoothing)))
        self.diversity_recompute_interval = max(1, int(self.diversity_recompute_interval))

        # Validate adaptive renderer complexity config
        self.enable_adaptive_complexity = bool(self.enable_adaptive_complexity)
        self.perf_target_fps = max(1.0, float(self.perf_target_fps))
        self.perf_up_threshold_ms = max(1.0, float(self.perf_up_threshold_ms))
        self.perf_down_threshold_ms = max(
            self.perf_up_threshold_ms + 0.5, float(self.perf_down_threshold_ms)
        )
        self.perf_adjustment_interval = max(0.05, float(self.perf_adjustment_interval))
        self.perf_min_raymarch_steps = max(1, int(self.perf_min_raymarch_steps))
        self.perf_max_raymarch_steps = max(
            self.perf_min_raymarch_steps, int(self.perf_max_raymarch_steps)
        )
        self.perf_min_fractal_layers = max(1, int(self.perf_min_fractal_layers))
        self.perf_max_fractal_layers = max(
            self.perf_min_fractal_layers, int(self.perf_max_fractal_layers)
        )
        self.perf_min_feedback_scale = max(0.1, min(1.0, float(self.perf_min_feedback_scale)))
        self.perf_max_feedback_scale = max(
            self.perf_min_feedback_scale, min(1.0, float(self.perf_max_feedback_scale))
        )
        self.perf_min_internal_scale = max(0.1, min(1.0, float(self.perf_min_internal_scale)))
        self.perf_max_internal_scale = max(
            self.perf_min_internal_scale, min(1.0, float(self.perf_max_internal_scale))
        )
        self.perf_min_feedback_history = max(0, int(self.perf_min_feedback_history))
        self.perf_max_feedback_history = max(
            self.perf_min_feedback_history, int(self.perf_max_feedback_history)
        )
        self.perf_min_particle_count = max(0, int(self.perf_min_particle_count))
        self.perf_max_particle_count = max(
            self.perf_min_particle_count, int(self.perf_max_particle_count)
        )
        self.perf_min_blur_taps = max(1, int(self.perf_min_blur_taps))
        self.perf_max_blur_taps = max(self.perf_min_blur_taps, int(self.perf_max_blur_taps))
        self.perf_min_lighting_samples = max(1, int(self.perf_min_lighting_samples))
        self.perf_max_lighting_samples = max(
            self.perf_min_lighting_samples, int(self.perf_max_lighting_samples)
        )
        self.perf_min_shadow_samples = max(1, int(self.perf_min_shadow_samples))
        self.perf_max_shadow_samples = max(
            self.perf_min_shadow_samples, int(self.perf_max_shadow_samples)
        )
        self.perf_min_bloom_passes = max(1, int(self.perf_min_bloom_passes))
        self.perf_max_bloom_passes = max(
            self.perf_min_bloom_passes, int(self.perf_max_bloom_passes)
        )
        self.perf_min_dof_samples = max(1, int(self.perf_min_dof_samples))
        self.perf_max_dof_samples = max(self.perf_min_dof_samples, int(self.perf_max_dof_samples))

        self.enable_multi_resolution_evolution = bool(self.enable_multi_resolution_evolution)
        if not self.res_tiers:
            self.res_tiers = [
                MultiResTier(
                    name="exploration",
                    res=32,
                    population_size=24,
                    mutation_mult=2.0,
                    promote_interval=120,
                ),
                MultiResTier(
                    name="normal",
                    res=128,
                    population_size=8,
                    mutation_mult=1.0,
                    promote_interval=180,
                ),
                MultiResTier(
                    name="champions",
                    res=512,
                    population_size=2,
                    mutation_mult=0.5,
                    promote_interval=300,
                ),
            ]
        validated = []
        seen = set()
        for tier in self.res_tiers:
            tier = copy.deepcopy(tier)
            tier.name = str(tier.name)
            tier.res = max(4, int(tier.res))
            tier.population_size = max(1, int(tier.population_size))
            tier.mutation_mult = max(0.01, float(tier.mutation_mult))
            tier.promote_interval = max(1, int(tier.promote_interval))
            if tier.name in seen:
                continue
            seen.add(tier.name)
            validated.append(tier)
        self.res_tiers = validated

        # Validate cultural evolution config
        self.enable_cultural_evolution = bool(self.enable_cultural_evolution)

        # Validate preset transition config
        self.enable_transitions = bool(self.enable_transitions)
        if self.transition_dir:
            self.transition_dir = Path(self.transition_dir)
            self.transition_dir.mkdir(parents=True, exist_ok=True)
        self.transition_kind = str(getattr(self, "transition_kind", "crossfade")).lower()
        self.transition_duration_secs = max(
            0.0, float(getattr(self, "transition_duration_secs", 0.9))
        )
        self.transition_random = bool(getattr(self, "transition_random", True))
        self.transition_audio_reactive = bool(
            getattr(self, "transition_audio_reactive", True)
        )

        # Validate evolving post-effects config
        self.enable_effects = bool(getattr(self, "enable_effects", True))
        if self.effects_dir is not None:
            self.effects_dir = Path(self.effects_dir)
        elif self.enable_effects:
            # Default: the repo-level ``effects/`` folder next to this package.
            self.effects_dir = Path(__file__).resolve().parent.parent / "effects"
        self.effects_evolve_secs = max(
            1.0, float(getattr(self, "effects_evolve_secs", 14.0))
        )
        self.effects_transition_secs = max(
            0.05, float(getattr(self, "effects_transition_secs", 1.4))
        )
        self.effects_param_mutation = max(
            0.0, float(getattr(self, "effects_param_mutation", 0.15))
        )
        self.effects_audio_reactive = bool(
            getattr(self, "effects_audio_reactive", True)
        )
        seed = getattr(self, "effects_seed", None)
        self.effects_seed = int(seed) if seed is not None else None

        # Validate folder visualiser config
        self.enable_folder_visualizer = bool(
            getattr(self, "enable_folder_visualizer", False)
        )
        if self.folder_shaders_dir is not None:
            self.folder_shaders_dir = Path(self.folder_shaders_dir)
        elif self.enable_folder_visualizer:
            # Default: the repo-level ``shaders/`` folder next to this package.
            self.folder_shaders_dir = (
                Path(__file__).resolve().parent.parent / "shaders"
            )
        self.folder_evolve_secs = max(
            1.0, float(getattr(self, "folder_evolve_secs", 18.0))
        )
        self.folder_transition_secs = max(
            0.05, float(getattr(self, "folder_transition_secs", 1.6))
        )
        self.folder_param_mutation = max(
            0.0, float(getattr(self, "folder_param_mutation", 0.12))
        )
        fseed = getattr(self, "folder_seed", None)
        self.folder_seed = int(fseed) if fseed is not None else None
        self.folder_use_effects = bool(getattr(self, "folder_use_effects", True))
        self.cultural_copy_rate = max(0.0, min(1.0, float(self.cultural_copy_rate)))
        self.cultural_imitate_rate = max(0.0, min(1.0, float(self.cultural_imitate_rate)))
        self.cultural_steal_rate = max(0.0, min(1.0, float(self.cultural_steal_rate)))
        self.cultural_modify_rate = max(0.0, min(1.0, float(self.cultural_modify_rate)))
        self.cultural_teach_rate = max(0.0, min(1.0, float(self.cultural_teach_rate)))
        self.cultural_convergence_window = max(1, int(self.cultural_convergence_window))

    # ------------------------------------------------------------------
    # Profile / serialisation helpers
    # ------------------------------------------------------------------
    def apply_profile(self, profile: str) -> None:
        """Apply a named feature profile (minimal/spatial/gpu/full).

        Profiles pick a known-good subset of the dozens of experimental
        flags so cold-start cost and failure surface stay small.  Individual
        flags can still be toggled after the profile is applied.
        """
        spec = _PROFILE_SPECS.get(profile)
        if spec is None:
            return
        for flag, value in spec["on"].items():
            if hasattr(self, flag):
                setattr(self, flag, bool(value))
        for flag, value in spec["off"].items():
            if hasattr(self, flag):
                setattr(self, flag, bool(value))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the config to a plain dict (for presets / persistence)."""
        out: dict[str, Any] = {}
        for f in self.__dataclass_fields__:
            out[f] = getattr(self, f)
        # Path / MultiResTier objects need coercion for JSON.
        if isinstance(out.get("preset_dir"), Path):
            out["preset_dir"] = str(out["preset_dir"])
        if isinstance(out.get("effects_dir"), Path):
            out["effects_dir"] = str(out["effects_dir"])
        if isinstance(out.get("res_tiers"), list):
            out["res_tiers"] = [
                {
                    k: getattr(t, k)
                    for k in (
                        "name",
                        "res",
                        "population_size",
                        "mutation_mult",
                        "promote_interval",
                    )
                }
                for t in out["res_tiers"]
            ]
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvoConfig:
        """Reconstruct from a serialised dict (tolerant of unknown keys)."""
        valid = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in valid}
        # Coerce res_tiers dicts back into MultiResTier objects.
        if isinstance(kwargs.get("res_tiers"), list):
            tiers = []
            for t in kwargs["res_tiers"]:
                if isinstance(t, MultiResTier):
                    tiers.append(t)
                elif isinstance(t, dict):
                    tiers.append(
                        MultiResTier(
                            name=str(t.get("name", "tier")),
                            res=int(t.get("res", 64)),
                            population_size=int(t.get("population_size", 8)),
                            mutation_mult=float(t.get("mutation_mult", 1.0)),
                            promote_interval=int(t.get("promote_interval", 120)),
                        )
                    )
            kwargs["res_tiers"] = tiers
        # Coerce preset_dir back to a Path.
        if isinstance(kwargs.get("preset_dir"), str):
            kwargs["preset_dir"] = Path(kwargs["preset_dir"])
        if isinstance(kwargs.get("effects_dir"), str):
            kwargs["effects_dir"] = Path(kwargs["effects_dir"])
        return cls(**kwargs)
