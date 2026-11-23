#!/usr/bin/env python3
from __future__ import annotations

import array
import copy
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gi
import numpy as np

gi.require_version("Gtk", "4.0")

from gi.repository import Gtk  # noqa: E402
from OpenGL import GL  # noqa: E402

# Allow `python evo_visualizer/renderer.py` to resolve the top-level sibling
# packages (engine, evo, evo_bio, ...), which live in the repo root rather than
# inside this package.
if __package__ in (None, ""):
    _ROOT = str(Path(__file__).resolve().parent.parent)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from engine.systems.audio_system import AudioSystem  # noqa: E402
from evo.genome_base import ShaderCache  # noqa: E402
from evo.gl.shader_disk_cache import ShaderDiskCache, _hash  # noqa: E402
from evo.milkdrop.compiler import MilkdropCompiler  # noqa: E402
from evo.milkdrop.preset import MilkdropPreset, parse_milkdrop_file  # noqa: E402
from evo.milkdrop.renderer import MilkdropShaderRenderer  # noqa: E402
from evo.modular_genome import ModularGenome  # noqa: E402
from evo.modular_shader_factory import ModularShaderFactory  # noqa: E402
from evo.modules.module_source_cache import ModuleSourceCache  # noqa: E402
from evo.pipeline_genome import PipelineGenome  # noqa: E402
from evo_bio.gene_usage_tracking import GeneUsageTracker, MutationOperatorTracker  # noqa: E402
from evo_bio.music_pressure import MusicPressure, MusicSignature  # noqa: E402
from evo_bio.pareto import Objectives  # noqa: E402
from evo_gpu_pool import (  # noqa: E402
    AdaptiveHistoryController,
    FboPool,
    GpuMemoryBudget,
    ReplayRingBuffer,
    apply_adaptive_trims,
)
from evo_multi_pass import (  # noqa: E402
    CompetitionRenderer,
    MultiPassPipeline,
    RegionBasedRenderer,
    RegionLayout,
)
from evo_perf import AsyncReadback, AsyncShaderCompiler  # noqa: E402
from evo_performance import (  # noqa: E402
    FrustumCuller,
    GpuTimerQuery,
    PersistentSSBO,
    ShaderBinaryCache,
    TextureAtlasPacker,
)
from evo_renderer.evolution_memory import EvolutionMemory  # noqa: E402
from evo_shader_graph import ShaderGraphGenome  # noqa: E402
from evo_visualizer.config import EvoConfig  # noqa: E402
from evo_visualizer.constants import (  # noqa: E402
    FULLSCREEN_QUAD_VERTICES,
    QUAD_STRIDE_BYTES,
    QUAD_VERTEX_COUNT,
)
from evo_visualizer.ecosystem import (  # noqa: E402
    EnhancedSpatialEcosystem,
    NicheDefinition,
    NicheManager,
    SpatialOrganism,
)
from evo_visualizer.effects import EffectManager  # noqa: E402
from evo_visualizer.folder_visualizer import FolderVisualizer  # noqa: E402
from evo_visualizer.multi_res import MultiResolutionEvolution  # noqa: E402
from evo_visualizer.population import MemoryDNA, Population  # noqa: E402
from evo_visualizer.renderer_support import (  # noqa: E402
    AudioState,
    FeedbackBuffer,
    FitnessEvaluator,
    GeneFitnessTracker,
    MutationHeatmap,
)

logger = logging.getLogger(__name__)


class EvoRenderer(Gtk.GLArea):
    """Real-time evolutionary visual renderer using OpenGL and GTK.

    Features:
      - Real-time shader generation from evolving genome parameters
      - Feedback-based rendering with ping-pong buffers
      - Audio-reactive effects
      - Human feedback steering (like/dislike)
      - Automatic view toggling driven by system state
    """

    VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;
out vec2 v_uv;
void main(){
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

    MILKDROP_FRAGMENT_SHADER_TEMPLATE = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D previous_frame;
uniform float u_feedback;
uniform float bass;
uniform float mid;
uniform float treble;
uniform float beat;

// Milkdrop equation bodies are injected here by MilkdropShaderRenderer.
// The per-frame block runs once per frame; the per-pixel block runs
// once per fragment.  Both assign to Milkdrop variables (warp, zoom,
// wave_r, etc.) which the per-pixel body then consumes.
void main(){
    vec2 uv = v_uv;
    vec2 p = uv - 0.5;

    __MILKDROP_PER_FRAME_BODY__

    __MILKDROP_PER_PIXEL_BODY__
}
"""

    CAMERA_ORGANISM_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec3 u_tint;
uniform float u_energy;
uniform vec2 u_territory_min;
uniform vec2 u_territory_size;

void main(){
    vec2 uv = v_uv;
    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);

    float b = 0.0;
    float cl = 0.12;
    float th = 0.0025;

    vec2 corners[4];
    corners[0] = vec2(-0.35,  0.28);
    corners[1] = vec2( 0.35,  0.28);
    corners[2] = vec2(-0.35, -0.28);
    corners[3] = vec2( 0.35, -0.28);

    vec2 dirs[4];
    dirs[0] = vec2( 1.0, -1.0);
    dirs[1] = vec2(-1.0, -1.0);
    dirs[2] = vec2( 1.0,  1.0);
    dirs[3] = vec2(-1.0,  1.0);

    for (int i = 0; i < 4; i++) {
        vec2 d = (p - corners[i]) * dirs[i];
        float h = step(abs(d.x), th) * step(0.0, d.y) * step(d.y, cl);
        float v = step(abs(d.y), th) * step(0.0, d.x) * step(d.x, cl);
        b = max(b, max(h, v));
    }

    float focus = 1.0 - smoothstep(0.0, 0.015, length(p));
    float track = step(abs(p.y), 0.0015) * step(abs(p.x), 0.25);

    float pulse = 0.7 + 0.3 * sin(u_time * 3.0);
    vec3 col = u_tint * (b * 0.9 + focus * 0.6 * pulse + track * 0.35);
    float alpha = max(b * 0.85, max(focus * 0.6 * pulse, track * 0.45));
    alpha *= 0.4 + 0.6 * u_energy;

    fragColor = vec4(col, alpha);
}
"""

    CAMERA_UNIFORM_NAMES = (
        "k_camera_speed",
        "k_camera_orbit",
        "k_camera_depth",
        "k_camera_roll",
        "k_camera_fov",
        "k_depth_scale",
        "k_ray_steps",
        "k_fog_density",
        "k_dimension",
        "k_camera_focus",
        "k_camera_aperture",
        "k_camera_exposure",
        "k_camera_autoexp",
        "k_camera_shake",
        "k_camera_shake_freq",
        "k_camera_tracking",
        "k_camera_framing",
        "k_camera_headroom",
        "k_camera_school",
        "k_camera_shot_length",
        "k_camera_transition",
        "k_camera_crane",
        "k_camera_parallax",
        "k_camera_drift",
        "k_camera_focus_pull",
        "k_camera_handheld",
        "k_camera_dolly",
        "k_camera_pan",
        "k_camera_tilt",
    )

    UNIFORM_NAMES = (
        "u_time",
        "u_resolution",
        "u_quality",
        "u_particle_count",
        "u_blur_taps",
        "u_lighting_samples",
        "u_shadow_samples",
        "u_bloom_passes",
        "u_dof_samples",
        "u_feedback_history",
        "bass",
        "mid",
        "treble",
        "previous_frame",
        "u_audio",
        "u_prev_frame",
        "u_feedback_alpha",
        "u_brightness",
        "md_bpm",
        "md_energy",
        "md_danceability",
        "md_beat_strength",
        "md_spectral_centroid",
        "md_harmonic_complexity",
        "md_dynamic_range",
        "md_key",
        "md_mood",
        *CAMERA_UNIFORM_NAMES,
    )

    # Ceiling for the per-program uniform-location memo. Programs are already
    # bounded by the shader cache, so this only guards against a cache that is
    # repeatedly cleared and rebuilt.
    UNIFORM_LOC_CACHE_MAX = 512

    def __init__(self, config: EvoConfig):
        """Initialize renderer.

        Args:
            config: EvoConfig instance with renderer parameters
        """
        super().__init__()

        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Cache for frequently accessed config values to avoid repeated getattr calls
        self._config_cache: dict[str, Any] = {}
        # Performance timing for debugging
        # Values are ring buffers (deque(maxlen=100)) so the per-frame
        # _perf_timer finally-block never reallocates or slices a list.
        self._perf_timers: dict[str, deque[float]] = {}

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_hexpand_set(True)
        self.set_vexpand_set(True)
        self.set_halign(Gtk.Align.FILL)
        self.set_valign(Gtk.Align.FILL)
        self.set_required_version(3, 3)
        self.set_has_depth_buffer(False)
        self.set_has_stencil_buffer(False)
        self.set_auto_render(True)
        self.set_visible(True)

        self.genome = ShaderGraphGenome()
        self.module_source_cache = ModuleSourceCache()
        self.shader_factory = ModularShaderFactory()
        self.shader_factory.source_cache = self.module_source_cache
        # on_evict releases the GL program so evicted entries do not leak on the
        # GPU (the root cause of the OOM). The cache never evicts the program
        # currently bound because _set_program calls shader_cache.protect().
        cfg = self._config
        self.shader_cache = ShaderCache(
            max_size=int(getattr(cfg, "shader_cache_max_size", 128)),
            on_evict=self._delete_program,
        )

        try:
            # Constructed only; init_gl() is deferred to on_realize() after the
            # GL context is current. Calling it here would run without a
            # context and silently disable the cache.
            self.disk_cache = ShaderDiskCache(Path(".shader_cache"), max_size=256)
            self._disk_cache_needs_init = True
        except (OSError, PermissionError, RuntimeError) as exc:
            self._logger.debug("Shader disk cache unavailable: %s", exc)
            self.disk_cache = None
            self._disk_cache_needs_init = False
        except Exception as exc:
            self._logger.warning("Unexpected error initializing shader disk cache: %s", exc)
            self.disk_cache = None
            self._disk_cache_needs_init = False

        # Spec 6: asynchronous shader compilation. The CPU-heavy GLSL source
        # generation runs on a worker thread; the fast GL upload + swap happens
        # on the render thread once the source is ready (see _get_program /
        # _drain_async_shaders). While a program is still compiling the caller
        # keeps rendering the organism's previous program, so live mutations
        # never freeze the visuals.
        self.async_shader_compiler: AsyncShaderCompiler | None = None
        if self._config_get("enable_async_shader_compile", True, bool):
            self.async_shader_compiler = AsyncShaderCompiler(
                max_pending=self._config_get("async_shader_max_pending", 64, int),
            )

        self.audio = AudioState()
        self._audio_system = AudioSystem()
        self._preference_bias = 0.0

        self._population_view = False
        self._heatmap_view = False
        # Set by build_heatmap_overlay(): the Cairo drawing area stacked on top
        # of this GLArea. Gtk.GLArea cannot host Cairo children, so the mutation
        # heatmap is only drawn when the host wraps the renderer.
        self._heatmap_drawing_area: Gtk.DrawingArea | None = None
        self._heatmap_ctx_warned = False
        self._competition_view = False
        self._multi_res_pyramid_view = False

        self._vao: int | None = None
        self._vbo: int | None = None
        self._program: int | None = None
        self._last_module_fingerprint: str | None = None
        # deque, not list: on_render() drains this every frame with popleft()
        # (O(1)); a list + pop(0) would make the drain O(n^2) when shader
        # evolution queues bursts of program deletions.
        self.pending_delete: deque = deque()
        # Memoized uniform locations per GL program id; see _uniform_locations.
        self._uniform_loc_cache: dict[int, dict[str, int]] = {}
        self.gl_ready = False
        self._tick_id: int | None = None
        self._last_glarea_w: int = 0
        self._last_glarea_h: int = 0

        self._vein_vbo: int | None = None
        self._vein_vao: int | None = None
        self._vein_topo_version: int = -1
        self._vein_draw_groups: list[tuple[int, int, tuple[float, float, float]]] = []

        self._loc_time = -1
        self._loc_res = -1
        self._loc_audio = -1  # legacy (unused in modular)
        self._loc_prev = -1  # legacy (unused in modular)
        self._loc_fb_alpha = -1  # legacy (unused in modular)
        self._loc_brightness = -1  # legacy (unused in modular)

        self._loc_bass = -1
        self._loc_mid = -1
        self._loc_treble = -1
        self._loc_previous_frame = -1

        # Music DNA uniforms
        self._loc_md_bpm = -1
        self._loc_md_energy = -1
        self._loc_md_danceability = -1
        self._loc_md_beat_strength = -1
        self._loc_md_spectral_centroid = -1
        self._loc_md_harmonic_complexity = -1
        self._loc_md_dynamic_range = -1
        self._loc_md_key = -1
        self._loc_md_mood = -1

        # Shared GPU object pool (texture + framebuffer pooling). A single
        # bounded pool backs the feedback ping-pong pair, the persistent-world
        # textures and any replay passes, so resize storms and new organisms
        # reuse GL objects instead of leaking a fresh pair each time.
        self._fbo_pool: FboPool | None = None
        if self._config_get("enable_fbo_pool", True, bool):
            try:
                self._fbo_pool = FboPool()
            except (RuntimeError, MemoryError, OSError) as e:
                self._logger.warning("FBO pool unavailable, using direct alloc: %s", e)
                self._fbo_pool = None
            except Exception as e:
                self._logger.warning("Unexpected error initializing FBO pool: %s", e)
                self._fbo_pool = None

        self.feedback = FeedbackBuffer(
            self._config.feedback_res,
            history=self._config.feedback_history,
            pool=self._fbo_pool,
        )
        self._start_time = time.time()
        self._generation = 0  # renderer-level evolution step counter (logging only)

        # Multi-pass render pass pipeline (adaptive render passes).
        self._mp_pipeline: MultiPassPipeline | None = None
        # Competition rendering: render top N organisms, blend, compete, replace.
        self._competition_renderer: CompetitionRenderer | None = None
        self._competition_n: int = 8
        # Region-based rendering: ecosystem paints the screen together.
        self._region_renderer: RegionBasedRenderer | None = None
        self._region_view: bool = self._config_get("enable_region_rendering", False, bool)

        # Environmental niche manager
        self._niche_manager: NicheManager | None = None
        if self._config_get("enable_niches", True, bool) and self._config_get(
            "enable_region_rendering", False, bool
        ):
            self._niche_manager = NicheManager(
                count=self._config_get("region_count", 5, int),
                layout=self._config_get("niche_layout", "stripes", str),
            )

        # Population system
        self.population = Population(
            size=self._config.population_size,
            crossover_rate=self._config.crossover_rate,
            genome_factory=lambda: type(self.genome)(),
            elite_fraction=self._config.elite_fraction,
            tournament_size=self._config.tournament_size,
        )
        # Keep Population[0] in sync with the renderer's current genome.
        self.population.organisms[0] = self.genome

        # Multi-resolution evolution (optional faster pyramid).
        self._multi_res: MultiResolutionEvolution | None = None
        if self._config_get("enable_multi_resolution_evolution", False, bool):
            try:
                self._multi_res = MultiResolutionEvolution(self, init_gl=False)
                self._logger.info("Multi-resolution evolution enabled")
            except (ImportError, AttributeError, RuntimeError) as exc:
                self._logger.warning("Multi-res init failed: %s", exc)
            except Exception as exc:
                self._logger.warning(
                    "Unexpected error initializing multi-resolution evolution: %s", exc
                )

        # Fitness and lineage
        self.fitness_evaluator = FitnessEvaluator(res=self._config.fitness_image_res)
        self.gene_fitness_tracker = GeneFitnessTracker()
        self.mutation_heatmap = MutationHeatmap()
        self.gene_usage_tracker = GeneUsageTracker()
        self.mutation_op_tracker = MutationOperatorTracker()
        self.memory_dna = MemoryDNA(
            preset_dir=self._config.preset_dir / "lineage" if self._config.preset_dir else None
        )
        self._previous_image: bytes | None = None
        self._previous_frames: deque = deque(maxlen=8)
        self._previous_fitness_tex: int | None = None
        # Spec 28: replay events stored in a *bounded* ring so a long session
        # cannot accumulate an unbounded list of encoded replay frames. The
        # adaptive controller shrinks this ring under memory pressure.
        self._replay = ReplayRingBuffer(maxlen=self._config_get("replay_max_frames", 256, int))
        self._fitness_readback: AsyncReadback | None = None
        self._fitness_fbo: int | None = None
        self._fitness_tex: int | None = None

        # Ecosystem memory
        self.ecosystem_memory: EvolutionMemory | None = None
        self._music_pattern_interval_secs = 15.0
        self._last_music_pattern_ts = time.time()
        self._dream_active = False
        self._current_music_sig: MusicSignature | None = None
        self._current_music_dna: Any | None = None
        self._music_pressure: MusicPressure | None = None
        self._last_beat_mutation_ts: float = 0.0
        self._last_beat_transition_ts: float = 0.0
        self._last_beat_evolution_ts: float = 0.0
        self._beat_evolution_cooldown: float = 2.0
        self._last_micro_evolution_ts: float = 0.0
        self._micro_evolution_interval: float = 0.05
        self._last_migration_ts: float = 0.0
        self._last_full_fitness_ts: float = 0.0
        # Automatic toggle state (replaces keyboard controls)
        self._auto_toggle_step: int = 0
        self._last_auto_mode_ts: float = 0.0
        self._last_auto_blend_ts: float = 0.0
        self._last_auto_toggle_ts: float = 0.0
        # Throttle for the spatial champion -> visual genome gene sync (the
        # copy can invalidate the active shader, so it must not run per frame).
        self._last_spatial_sync_ts: float = 0.0
        self._auto_toggle_interval: float = 15.0
        self._full_fitness_interval: float = 5.0
        self._last_scene_entropy: float = 0.5
        self._last_scene_motion: float = 0.0
        self._last_novelty: float = 0.0
        self._recent_births: list[tuple[float, float, float]] = []

        # Beat-driven smooth transitions
        self._beat_transition_target: Any = None
        self._beat_transition_progress: float = 0.0
        self._beat_transition_duration: float = 0.5  # seconds
        self._beat_transition_start_ts: float = 0.0

        # Dynamic mutation scheduling (spec 13) runtime state.
        self._ema_fps: float = self._config_get("schedule_fps_target", 60.0, float)
        self._ema_gpu_load: float = 0.4
        self._ema_audio_energy: float = 0.0
        self._best_fitness_ema: float = 0.0
        self._fitness_stagnation: float = 0.0  # 0..1, 1 == fully stalled
        self._last_best_fitness: float = 0.0
        self._scheduled_mutations_per_frame: float = self._config_get(
            "steady_state_mutations_per_frame", 2.0, float
        )

        # Throttled diversity / species signals for the dynamic mutation
        # scheduler.  _population_diversity() is O(n²) and _species_count() is
        # O(n), but their output feeds the EMA-smoothed mutation rate, so
        # recomputing every frame is pure overhead.  The cache is invalidated
        # whenever the population is structurally modified (see
        # _invalidate_evolution_cache).
        self._diversity_cache: float | None = None
        self._species_cache: int | None = None
        self._diversity_frame_counter: int = 0

        # Adaptive renderer complexity: one controller drives every expensive
        # knob (ray steps, FBM layers, particles, bloom, DoF, feedback,
        # internal scale) together from a single measured-frame-time budget.
        self.quality_controller: Any | None = None
        if self._config_get("enable_adaptive_complexity", True, bool):
            try:
                from evo_perf import AdaptiveQualityController

                self.quality_controller = AdaptiveQualityController(self._config)
                self._logger.info("Adaptive renderer complexity enabled")
            except (ImportError, AttributeError, RuntimeError) as e:
                self._logger.warning("Adaptive complexity init failed: %s", e)
                self.quality_controller = None
            except Exception as e:
                self._logger.warning("Unexpected error initializing adaptive complexity: %s", e)
                self.quality_controller = None
        # Uniform location cache for the shared quality uniform (per program).
        self._loc_quality: int = -1
        # Adaptive-complexity loop-bound uniforms; refreshed by _set_program.
        self._loc_particle_count: int = -1
        self._loc_blur_taps: int = -1
        self._loc_lighting_samples: int = -1
        self._loc_shadow_samples: int = -1
        self._loc_bloom_passes: int = -1
        self._loc_dof_samples: int = -1
        self._loc_feedback_history: int = -1
        # When True, _current_lod / _get_program force the maximum LOD tier so
        # fitness scoring is not biased by the live frame-time quality budget.
        self._force_full_lod: bool = False

        if self._config_get("enable_ecosystem_memory", False, bool):
            self.ecosystem_memory = EvolutionMemory(db_path=self._config.memory_db_path)
            self.ecosystem_memory.load_world_dna()
            self._logger.info("Ecosystem memory enabled")

        # Connect GTK signals
        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)
        self.connect("resize", self._on_glarea_resize)
        self._glarea_w = 0
        self._glarea_h = 0

        # Spatial ecosystem (specs 46-60)
        self.ecosystem: EnhancedSpatialEcosystem | None = None
        self._spatial_active = self._config_get("enable_spatial_ecosystem", False, bool)
        if self._spatial_active:
            try:
                self.ecosystem = EnhancedSpatialEcosystem(self._config)
                self._logger.info(
                    "Spatial ecosystem enabled: %d organisms, %d species",
                    len(self.ecosystem.organisms),
                    len(self.ecosystem.species),
                )
            except (ImportError, AttributeError, RuntimeError, MemoryError) as e:
                self._logger.error("Spatial ecosystem init failed: %s", e)
                self.ecosystem = None
                self._spatial_active = False
            except Exception as e:
                self._logger.error("Unexpected error initializing spatial ecosystem: %s", e)
                self.ecosystem = None
                self._spatial_active = False
        self._eco_programs: dict[str, int] = {}
        self._eco_loc: dict[int, dict[str, int]] = {}
        self._eco_used: dict[str, int] = {}
        self._resource_tex: int | None = None
        self._resource_tex_size: tuple[int, int] = (0, 0)
        # 31 per-organism map atlas (food/pheromone/danger/territory).
        self._org_map_tex: int | None = None
        self._org_map_size: tuple[int, int] = (0, 0)
        self._last_frame_ts = time.time()
        self._cur_eco_prog: int | None = None

        # Frustum culler for spatial culling.  Uses an explicit identity MVP
        # since the 2D evolutionary renderer has no camera projection matrix.
        # Avoids deprecated GL_MODELVIEW_PROJECTION_MATRIX queries.
        self.frustum = FrustumCuller()
        self.frustum.set_mvp([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # fmt: skip

        # Voronoi cache for spatial layout computation
        self._voronoi_cache: dict[tuple, list] = {}

        # ------------------------------------------------------------------
        # Milkdrop preset mode (classic preset-driven rendering).  When
        # ``_milkdrop_mode`` is True the renderer draws a compiled .milk
        # preset instead of the evolutionary genome.  This gives linamp
        # access to the full Milkdrop preset library (waves, plasma,
        # per-frame / per-pixel equations, etc.) alongside the evolving
        # visual organism.
        # ------------------------------------------------------------------
        self._milkdrop_mode: bool = False
        self._milkdrop_preset: MilkdropPreset | None = None
        self._milkdrop_compiler = MilkdropCompiler()
        self._milkdrop_shader_renderer = MilkdropShaderRenderer()
        self._milkdrop_program: int | None = None
        self._milkdrop_loc: dict[str, int] = {}
        self._milkdrop_prev_prog: int | None = None
        self.available_milk_files: list[Path] = []
        self._disable_auto_switching: bool = False
        self._last_milkdrop_auto_switch_ts: float = 0.0
        self._milkdrop_auto_switch_interval: float = 12.0
        self._milkdrop_preset_index: int = -1

        # ------------------------------------------------------------------
        # Evolving post-effects (``effects/*.frag`` shaders).  The manager
        # owns its own scene + ping-pong targets; the final screen pass is
        # routed through it so effects composite over whatever the genome
        # (or Milkdrop feedback) produced, without touching the genome's
        # own feedback buffer.
        # ------------------------------------------------------------------
        self._effects: EffectManager | None = None

        # ------------------------------------------------------------------
        # Folder visualiser (``shaders/`` + ``effects/`` + ``textures/``).
        # Opt-in mode: a full-screen shader pool from ``shaders/`` forms the
        # base layer and the EffectManager composites the evolving
        # ``effects/`` post chain on top.
        # ------------------------------------------------------------------
        self._folder_viz: FolderVisualizer | None = None

        # ------------------------------------------------------------------
        # Performance optimizations (spec items)
        # ------------------------------------------------------------------
        # Shader binary cache: fast path for programs already extracted by a
        # previous session / earlier in this process.  The per-organism
        # `_get_eco_program` consults it before falling back to source compile.
        self._perf_shader_bin_cache = ShaderBinaryCache(max_entries=512)
        # Persistent SSBO for per-organism parameters (probe only; the direct
        # render loop in _render_spatial binds it when available).
        self._perf_org_ssbo = PersistentSSBO(max_organisms=64, floats_per_org=16)
        # GPU timer queries for per-organism cost measurement.
        self._perf_gpu_timer = GpuTimerQuery(max_queries=128)
        # Texture atlas packer for the per-organism map atlas (food/pheromone/
        # danger/territory).  Initialized in on_realize() and reused for the
        # life of the GL context.
        self._perf_atlas_packer = TextureAtlasPacker(max_tiles=256, tile_res=16)

        # Feature 6: Multi-species extensions
        self.multi_species: Any | None = None
        self._parasite_system: Any | None = None
        if self._config_get("enable_multi_species", False, bool) and self.ecosystem is not None:
            try:
                from evo_multi_species import MultiSpeciesEcosystem, ParasiteSystem

                self.multi_species = MultiSpeciesEcosystem(self.ecosystem, self._config)
                self._parasite_system = ParasiteSystem(self.ecosystem)
                self._logger.info("Multi-species renderer enabled")
            except (ImportError, AttributeError, RuntimeError) as e:
                self._logger.warning("Multi-species init failed: %s", e)
            except Exception as e:
                self._logger.warning("Unexpected error initializing multi-species: %s", e)

        # Feature 7: GPU compute fitness
        self.gpu_compute: Any | None = None
        if self._config_get("enable_gpu_compute", False, bool):
            try:
                from evo_gpu_compute import (
                    OUT_ENTROPY,
                    OUT_FITNESS,
                    OUT_MOTION,
                    OUT_NOVELTY,
                    ComputeShaderManager,
                )

                self.gpu_compute = ComputeShaderManager(res=self._config.fitness_image_res)
                self.OUT_FITNESS = OUT_FITNESS
                self.OUT_ENTROPY = OUT_ENTROPY
                self.OUT_MOTION = OUT_MOTION
                self.OUT_NOVELTY = OUT_NOVELTY
                self._logger.info("GPU compute fitness enabled")
            except (ImportError, AttributeError, RuntimeError, MemoryError) as e:
                self._logger.warning("GPU compute init failed: %s", e)
            except Exception as e:
                self._logger.warning("Unexpected error initializing GPU compute: %s", e)

        # Feature 9: Dynamic shader graph (always enabled as primary renderer)
        self.shader_graph_genome: Any | None = None
        self.shader_graph_compiler: Any | None = None
        try:
            from evo_shader_graph import ShaderGraphCompiler

            self.shader_graph_compiler = ShaderGraphCompiler()
            self._logger.info("Shader graph renderer enabled")
        except (ImportError, AttributeError, RuntimeError) as e:
            self._logger.warning("Shader graph init failed: %s", e)
        except Exception as e:
            self._logger.warning("Unexpected error initializing shader graph: %s", e)

        # Feature 10: Emergent camera
        self.camera_population: Any | None = None
        self._camera_uniform_locs: dict[str, int] = {}
        if self._config_get("enable_emergent_camera", False, bool):
            try:
                from evo_camera import CameraPopulation

                self.camera_population = CameraPopulation(
                    size=self._config_get("camera_population_size", 16, int)
                )
                if self.ecosystem is not None:
                    for cam in self.camera_population.cameras:
                        cam.pos = [random.random(), random.random(), 0.0]
                        self.ecosystem.organisms.append(cam)
                    self._logger.info(
                        "Camera organisms added to ecosystem: %d cameras",
                        len(self.camera_population.cameras),
                    )
                self._logger.info("Emergent camera enabled")
            except (ImportError, AttributeError, RuntimeError) as e:
                self._logger.warning("Emergent camera init failed: %s", e)
            except Exception as e:
                self._logger.warning("Unexpected error initializing emergent camera: %s", e)

        # Adaptive history-length controller + proactive memory budget. These
        # drive the OOM-mitigation trims: when RSS climbs toward the budget the
        # history rings and lineage log shrink toward their floors, and the
        # shader cache is trimmed proportionally.
        self._adaptive_history = AdaptiveHistoryController()
        feedback_min = self._config_get("feedback_history_min", 0, int)
        feedback_max = self._config_get("feedback_history", 0, int)
        self._adaptive_history.add(
            "feedback_history",
            feedback_min,
            max(feedback_min, feedback_max),
        )
        self._adaptive_history.add(
            "replay",
            self._config_get("replay_max_frames_min", 32, int),
            self._config_get("replay_max_frames", 256, int),
        )
        self._adaptive_history.add(
            "lineage",
            self._config_get("lineage_max_records_min", 250, int),
            self._config_get("lineage_max_records", 2000, int),
        )
        self._adaptive_history.add(
            "novelty",
            self._config_get("novelty_history_min", 32, int),
            self._config_get("novelty_history_max", 200, int),
        )
        self._memory_budget = GpuMemoryBudget(
            threshold_mb=self._config_get("memory_budget_mb", 4096.0, float),
        )
        self._watchdog_ts = 0.0

        self._logger.info("EvoRenderer initialized with config: %s", asdict(config))

    def _config_get(
        self,
        key: str,
        default: Any = None,
        expected_type: type[Any] | None = None,
        use_cache: bool = True,
    ) -> Any:
        """Get a config value with optional type conversion and caching.

        This is a performance optimization to avoid repeated getattr calls
        and provides a single point for config access with type safety.

        Args:
            key: Configuration key name
            default: Default value if key not found
            expected_type: Optional type to convert the value to
            use_cache: Whether to cache the result (default True for frequently accessed values)

        Returns:
            The configuration value, converted to expected_type if provided
        """
        if use_cache and key in self._config_cache:
            return self._config_cache[key]

        value = getattr(self._config, key, default)
        if expected_type is not None and value is not None:
            try:
                if expected_type is bool:
                    value = bool(value)
                elif expected_type is int:
                    value = int(value)
                elif expected_type is float:
                    value = float(value)
                elif expected_type is str:
                    value = str(value)
            except (ValueError, TypeError):
                value = default

        if use_cache:
            self._config_cache[key] = value
        return value

    @contextmanager
    def _perf_timer(self, name: str, log_threshold_ms: float = 5.0):
        """Context manager for timing critical sections for performance debugging.

        Args:
            name: Name of the timed section
            log_threshold_ms: Only log if execution time exceeds this threshold (ms)

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # deque(maxlen=100) evicts the oldest sample automatically on every
            # append, so this is a single O(1) operation with no per-frame list
            # reallocation or [-100:] slicing -- critical because _perf_timer
            # wraps nearly every block in the per-frame on_render() hot path.
            self._perf_timers.setdefault(name, deque(maxlen=100)).append(elapsed_ms)
            if elapsed_ms > log_threshold_ms:
                self._logger.debug("[PERF] %s took %.2f ms", name, elapsed_ms)

    def _get_perf_stats(self) -> dict[str, dict[str, float]]:
        """Get performance statistics for all tracked timers.

        Returns:
            Dictionary mapping timer names to stats (min, max, avg, count)
        """
        stats = {}
        for name, times in self._perf_timers.items():
            if not times:
                continue
            stats[name] = {
                "min_ms": min(times),
                "max_ms": max(times),
                "avg_ms": sum(times) / len(times),
                "count": len(times),
            }
        return stats

    def _log_perf_stats(self) -> None:
        """Log current performance statistics for debugging."""
        stats = self._get_perf_stats()
        if not stats:
            return
        self._logger.info("Performance statistics:")
        for name, data in sorted(stats.items()):
            self._logger.info(
                "  %s: avg=%.2fms min=%.2fms max=%.2fms count=%d",
                name,
                data["avg_ms"],
                data["min_ms"],
                data["max_ms"],
                data["count"],
            )

    def _clear_config_cache(self) -> None:
        """Clear the config cache. Call this when config values change dynamically."""
        self._config_cache.clear()
        self._logger.debug("Config cache cleared")

    def _get_field_genome(self, genome: Any = None) -> Any:
        """Return the primary field-pass ModularGenome from a PipelineGenome,
        or the genome itself if it's already a ModularGenome."""
        if genome is None:
            genome = self.genome
        try:
            from evo.pipeline_genome import PipelineGenome

            if isinstance(genome, PipelineGenome):
                for p in genome.passes:
                    if p.pass_type == "field":
                        return p.modules
                return genome.passes[0].modules if genome.passes else ModularGenome()
        except ImportError:
            logging.getLogger(__name__).debug(
                "PipelineGenome import unavailable, using fallback genome"
            )
        return genome

    def _gen_frag_source(self, genome: Any, lod: dict[str, int] | None = None) -> str:
        from evo.pipeline_genome import PipelineGenome

        if isinstance(genome, PipelineGenome):
            return self.shader_factory.generate_pipeline(genome)[0]
        try:
            if isinstance(genome, ShaderGraphGenome):
                return self._graph_compiler_for_thread().compile(genome.graph)
        except (ImportError, AttributeError):
            logging.getLogger(__name__).debug(
                "Shader graph compilation unavailable, falling back to LOD path"
            )
        if lod is not None:
            # Adaptive renderer complexity: bake the active LOD tier's loop
            # bounds (ray steps / fractal layers) into this variant. Each
            # discrete tier compiles once and is cached, so switching quality
            # is a cheap program swap, not a per-frame recompile.
            return self.shader_factory.generate(
                genome,
                raymarch_steps=lod.get("raymarch_steps"),
                fractal_layers=lod.get("fractal_layers"),
            )[0]
        return self.shader_factory.generate(genome)[0]

    def _graph_compiler_for_thread(self):
        """Return a ShaderGraphCompiler private to the calling thread.

        ``ShaderGraphCompiler.compile`` stores per-call state on the instance
        (``self.graph`` / ``self._func_cache``), so the worker thread that does
        async source generation must not share the render thread's compiler.
        Each thread lazily gets its own instance.
        """
        import threading as _threading

        tls = getattr(self, "_graph_compiler_tls", None)
        if tls is None:
            tls = _threading.local()
            self._graph_compiler_tls = tls
        compiler = getattr(tls, "compiler", None)
        if compiler is None:
            # Reuse the eagerly-created compiler on the render/main thread; give
            # every other thread (the async worker) a fresh one.
            main_compiler = self.shader_graph_compiler
            if (
                main_compiler is not None
                and _threading.current_thread() is _threading.main_thread()
            ):
                compiler = main_compiler
            else:
                from evo_shader_graph import ShaderGraphCompiler

                compiler = ShaderGraphCompiler()
            tls.compiler = compiler
        return compiler

    def update_audio(
        self,
        audio_levels: dict[str, float],
        fft_samples: Any | None = None,
        sample_rate: float = 44100.0,
    ) -> None:
        """Update audio levels for reactive effects.

        Args:
            audio_levels: Dict with 'bass', 'mid', 'treb', 'beat' keys
            fft_samples: Optional numpy array of raw FFT / waveform data
                for deeper audio DNA analysis.
            sample_rate: Sampling rate of the raw audio (Hz).
        """
        self.audio.update_with_audio(audio_levels, fft_samples, sample_rate)
        if self._audio_system is not None:
            self._audio_system.update(
                {
                    "bass": float(audio_levels.get("bass", 0.0)),
                    "mid": float(audio_levels.get("mid", 0.0)),
                    "treble": float(audio_levels.get("treb", 0.0)),
                    "beat": float(audio_levels.get("beat", 0.0)),
                }
            )

    def _init_fitness_fbo(self) -> None:
        """Initialize FBOs and textures for fitness evaluation.

        Allocates the rendering FBO/texture (``_fitness_fbo`` / ``_fitness_tex``)
        and a second texture (``_previous_fitness_tex``) used as the
        motion-comparison reference for GPU compute analysis.  The previous
        texture is *not* attached to its own FBO; it is filled each frame via
        :meth:`_copy_fitness_to_previous` so the compute shader can compare
        consecutive frames.
        """
        res = self._config.fitness_image_res
        try:
            self._fitness_tex = GL.glGenTextures(1)
            self._fitness_fbo = GL.glGenFramebuffers(1)

            GL.glBindTexture(GL.GL_TEXTURE_2D, self._fitness_tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA16F,
                res,
                res,
                0,
                GL.GL_RGBA,
                GL.GL_HALF_FLOAT,
                None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fitness_fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D,
                self._fitness_tex,
                0,
            )

            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                # Clear the FBO ID so downstream guards (e.g. the fitness
                # capture path) see ``None`` and skip it instead of blitting
                # into an incomplete FBO, which raises
                # GL_INVALID_FRAME_BUFFER_OPERATION (0x0506).
                try:
                    GL.glDeleteFramebuffers(1, [self._fitness_fbo])
                except Exception:
                    pass
                self._fitness_fbo = None
                raise RuntimeError(f"Fitness FBO incomplete: {hex(status)}")

            # Previous-frame texture for GPU motion/novelty comparison.
            self._previous_fitness_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._previous_fitness_tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA16F,
                res,
                res,
                0,
                GL.GL_RGBA,
                GL.GL_HALF_FLOAT,
                None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

            self._fitness_readback = AsyncReadback(res)
            self._fitness_readback.init_gl()

            self._logger.debug(f"Fitness FBO initialized: {res}x{res}")
        except Exception as e:
            self._logger.warning(f"Fitness FBO init failed: {e}")

    def _apply_camera_to_genome(self) -> None:
        """Copy emergent camera state into the active genome's camera genes."""
        if not self._config.enable_emergent_camera or self.camera_population is None:
            return
        try:
            best = self.camera_population.best()
            g = self.genome
            cam_genome = getattr(best, "genome", None)
            if hasattr(g, "camera"):
                g.camera.camera_orbit = best.orbit * 6.28
                g.camera.camera_fov = best.fov
                g.camera.camera_roll = best.roll
                g.camera.camera_depth = 3.0 - best.zoom * 2.0
                g.camera.camera_focus = best.focus
                g.camera.camera_aperture = best.aperture
                g.camera.camera_exposure = best.exposure
                g.camera.camera_autoexp = getattr(cam_genome, "auto_exposure", 0.4)
                g.camera.camera_shake = best.shake
                g.camera.camera_shake_freq = getattr(cam_genome, "shake_frequency", 0.5)
                g.camera.camera_tracking = best.tracking
                g.camera.camera_framing = best.framing
                g.camera.camera_headroom = best.headroom
                g.camera.camera_school = getattr(cam_genome, "cinemat_school", 0)
                g.camera.camera_crane = getattr(cam_genome, "crane", 0.0)
                g.camera.camera_drift = getattr(cam_genome, "drift", 0.0)
                g.camera.camera_parallax = getattr(cam_genome, "parallax", 0.3)
                g.camera.camera_dolly = getattr(cam_genome, "dolly", 0.5)
                g.camera.camera_pan = best.pan
                g.camera.camera_tilt = best.tilt
                g.camera.camera_shot_length = getattr(cam_genome, "shot_length", 1.0)
                g.camera.camera_transition = getattr(cam_genome, "transition", 0.3)
                g.camera.camera_focus_pull = getattr(cam_genome, "focus_pull", 0.3)
                g.camera.camera_handheld = getattr(cam_genome, "handheld", 0.2)
            elif hasattr(g, "global_params"):
                g.global_params["camera_orbit"] = best.orbit * 6.28
                g.global_params["camera_fov"] = best.fov
                g.global_params["camera_roll"] = best.roll
                g.global_params["camera_depth"] = 3.0 - best.zoom * 2.0
                g.global_params["camera_focus"] = best.focus
                g.global_params["camera_aperture"] = best.aperture
                g.global_params["camera_exposure"] = best.exposure
                g.global_params["camera_autoexp"] = getattr(cam_genome, "auto_exposure", 0.4)
                g.global_params["camera_shake"] = best.shake
                g.global_params["camera_shake_freq"] = getattr(cam_genome, "shake_frequency", 0.5)
                g.global_params["camera_tracking"] = best.tracking
                g.global_params["camera_framing"] = best.framing
                g.global_params["camera_headroom"] = best.headroom
                g.global_params["camera_school"] = getattr(cam_genome, "cinemat_school", 0)
                g.global_params["camera_crane"] = getattr(cam_genome, "crane", 0.0)
                g.global_params["camera_drift"] = getattr(cam_genome, "drift", 0.0)
                g.global_params["camera_parallax"] = getattr(cam_genome, "parallax", 0.3)
                g.global_params["camera_dolly"] = getattr(cam_genome, "dolly", 0.5)
                g.global_params["camera_pan"] = best.pan
                g.global_params["camera_tilt"] = best.tilt
                g.global_params["camera_shot_length"] = getattr(cam_genome, "shot_length", 1.0)
                g.global_params["camera_transition"] = getattr(cam_genome, "transition", 0.3)
                g.global_params["camera_focus_pull"] = getattr(cam_genome, "focus_pull", 0.3)
                g.global_params["camera_handheld"] = getattr(cam_genome, "handheld", 0.2)
            cam_species = getattr(best, "species_id", 0)
            cam_fit = getattr(best, "fitness", 0.0)
            if hasattr(self, "_camera_species_log_ts"):
                if time.time() - self._camera_species_log_ts > 12.0:
                    self._camera_species_log_ts = time.time()
                    sp_names = getattr(self.camera_population, "species", {})
                    sp_info = sp_names.get(cam_species, None)
                    sp_name = getattr(sp_info, "name", f"species_{cam_species}")
                    self._logger.info(
                        f"Camera species: {sp_name} | "
                        f"fitness={cam_fit:.3f} | "
                        f"cuts={len(getattr(best, 'cut_log', []))}"
                    )
            else:
                self._camera_species_log_ts = time.time()
        except Exception as e:
            self._logger.debug(f"Camera-to-genome sync failed: {e}")

    def _render_fitness_fbo(self, genome=None, source_fbo=None, src_w=0, src_h=0) -> bool:
        """Render or blit the current phenotype into the fitness FBO.

        In the normal (non-spatial) path the genome's shader is drawn into the
        small fitness FBO so the compute/capture pass has fresh pixel data.

        In spatial mode (*source_fbo* is set) the already-composited screen
        framebuffer is *blitted* into the fitness FBO instead, so fitness measures
        the emergent ecosystem image (all organisms, territories, species blending)
        rather than re-rendering the active genome in isolation.

        Returns ``True`` on success, ``False`` on failure.  Does not manage GL
        state; callers must snapshot/restore via :meth:`_snapshot_gl_state` and
        :meth:`_restore_gl_state`.
        """
        if self._fitness_fbo is None or self._fitness_tex is None:
            return False
        if genome is None:
            genome = self.genome
        res = self._config.fitness_image_res
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fitness_fbo)
            GL.glViewport(0, 0, res, res)

            if source_fbo is not None:
                # Spatial mode: copy the composited screen into the fitness FBO.
                sw = max(1, src_w or getattr(self, "_last_glarea_w", res))
                sh = max(1, src_h or getattr(self, "_last_glarea_h", res))
                # FBO 0 is the default framebuffer; blitting *from* it into an
                # offscreen FBO is undefined on some drivers and raises
                # GL_INVALID_FRAME_BUFFER_OPERATION (0x0506). Skip the blit
                # rather than risking a context error mid-frame.
                if int(source_fbo) != 0:
                    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(source_fbo))
                    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._fitness_fbo)
                    GL.glBlitFramebuffer(
                        0,
                        0,
                        sw,
                        sh,
                        0,
                        0,
                        res,
                        res,
                        GL.GL_COLOR_BUFFER_BIT,
                        GL.GL_LINEAR,
                    )
                    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
            else:
                # Normal path: render the genome via its shader.
                score_program = self._program
                score_loc_time = self._loc_time
                score_loc_res = self._loc_res
                score_loc_bass = self._loc_bass
                score_loc_mid = self._loc_mid
                score_loc_treble = self._loc_treble
                score_loc_prev = self._loc_previous_frame
                score_loc_quality = getattr(self, "_loc_quality", -1)
                if self.quality_controller is not None:
                    prev_force = self._force_full_lod
                    self._force_full_lod = True
                    try:
                        full_prog = self._get_program(genome, allow_async=False)
                    finally:
                        self._force_full_lod = prev_force
                    if full_prog:
                        score_program = int(full_prog)
                        self.shader_cache.protect(score_program)
                        score_locs = self._uniform_locations(score_program)
                        score_loc_time = score_locs["u_time"]
                        score_loc_res = score_locs["u_resolution"]
                        score_loc_bass = score_locs["bass"]
                        score_loc_mid = score_locs["mid"]
                        score_loc_treble = score_locs["treble"]
                        score_loc_prev = score_locs["previous_frame"]
                        score_loc_quality = score_locs["u_quality"]

                GL.glUseProgram(score_program)
                GL.glBindVertexArray(self._vao)

                bass, mid, treb, _beat = self.audio.uvec()
                if score_loc_time >= 0:
                    GL.glUniform1f(score_loc_time, float(time.time() - self._start_time))
                if score_loc_res >= 0:
                    GL.glUniform2f(score_loc_res, float(res), float(res))

                if score_loc_bass >= 0:
                    GL.glUniform1f(score_loc_bass, bass)
                if score_loc_mid >= 0:
                    GL.glUniform1f(score_loc_mid, mid)
                if score_loc_treble >= 0:
                    GL.glUniform1f(score_loc_treble, treb)

                # Force full fidelity for scoring so the measured image is
                # tier-independent (modular shaders bake the tier and ignore
                # this uniform).
                if score_loc_quality >= 0:
                    GL.glUniform1f(score_loc_quality, 1.0)

                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.feedback.src_tex())
                if score_loc_prev >= 0:
                    GL.glUniform1i(score_loc_prev, 0)

                GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

            return True
        except Exception as e:
            self._logger.debug(f"Fitness FBO render failed: {e}")
            return False

    def _copy_fitness_to_previous(self) -> None:
        """Copy the current fitness texture into the previous-frame texture.

        Called after a successful GPU dispatch so the next frame has a valid
        previous-frame reference for motion / novelty analysis.
        """
        if self._previous_fitness_tex is None:
            return
        res = self._config.fitness_image_res
        try:
            gl_state = self._snapshot_gl_state()
            try:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._fitness_fbo)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._previous_fitness_tex)
                GL.glCopyTexSubImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    0,
                    0,
                    res,
                    res,
                )
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            finally:
                self._restore_gl_state(gl_state)
        except Exception as e:
            self._logger.debug(f"Fitness-to-previous copy failed: {e}")

    def _gpu_compute_fitness(self, source_fbo=None, src_w=0, src_h=0) -> None:
        """Dispatch GPU compute shader for fitness evaluation instead of CPU readback.

        Renders (or blits, in spatial mode) the current phenotype into the
        fitness FBO first, then dispatches the compute analysis and copies the
        result into the previous-frame texture for the next iteration.
        """
        if self.gpu_compute is None or not self.gpu_compute.available:
            self._capture_and_score(source_fbo=source_fbo, src_w=src_w, src_h=src_h)
            return
        if self._fitness_fbo is None or self._fitness_tex is None:
            self._capture_and_score()
            return
        old_fbo = None
        gl_state = None
        rendered = False
        try:
            gl_state = self._snapshot_gl_state()
            old_fbo = gl_state.get("framebuffer") if gl_state else None
            novelty = getattr(self, "_last_novelty", 0.0)
            # Render/copy the current phenotype into the fitness texture so the
            # compute shader analyses fresh pixel data instead of a stale or
            # never-written texture.
            rendered = self._render_fitness_fbo(
                genome=self.genome,
                source_fbo=source_fbo,
                src_w=src_w,
                src_h=src_h,
            )

            if rendered:
                # Dispatch compute analysis on the freshly rendered fitness texture.
                previous = self._previous_fitness_tex or 0
                if previous == 0:
                    # First frame: no previous, fall back to self so motion reads 0.
                    previous = self._fitness_tex
                success = self.gpu_compute.dispatch_analysis(self._fitness_tex, previous, novelty)
                if success:
                    metrics = self.gpu_compute.read_output(block=False)
                    if metrics:
                        beauty = float(metrics[self.OUT_FITNESS]) * 0.1
                        self.genome.objectives.beauty = beauty
                        if not hasattr(self, "_last_scene_entropy"):
                            self._last_scene_entropy = 0.0
                        self._last_scene_entropy = float(metrics[self.OUT_ENTROPY]) / 4.0
                        self._last_scene_motion = float(metrics[self.OUT_MOTION])
                        self._last_novelty = float(metrics[self.OUT_NOVELTY])
                        self.genome.objectives.novelty = self._last_novelty
                        # Persist current frame as previous for motion detection.
                        self._copy_fitness_to_previous()
                        return
        except Exception as e:
            self._logger.debug(f"GPU compute fitness failed: {e}")
        finally:
            if old_fbo is not None:
                self._restore_gl_state(gl_state)
        # CPU fallback: render failed, dispatch failed, or no metrics.
        self._capture_and_score(source_fbo=source_fbo, src_w=src_w, src_h=src_h)

    def _run_visual_fitness_capture(self) -> None:
        """Score the current phenotype into the fitness FBO.

        Shared by every presentation mode (feedback, population, competition,
        region, multi-resolution *and* spatial) so visual fitness never depends
        on which view happens to be on screen.
        """
        if not self._config.enable_visual_fitness or self._fitness_fbo is None:
            return
        # In spatial mode the screen already contains the composited ecosystem
        # scene (all organisms, territories, species blending).  Blit it into
        # the fitness FBO so the GPU compute shader and CPU readback analyse
        # what the viewer actually sees, rather than re-rendering the active
        # genome in isolation.
        source_fbo = None
        src_w = 0
        src_h = 0
        if getattr(self, "_spatial_active", False):
            source_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
            src_w = max(1, getattr(self, "_last_glarea_w", 0))
            src_h = max(1, getattr(self, "_last_glarea_h", 0))
        try:
            if (
                self._config.enable_gpu_compute
                and self.gpu_compute is not None
                and self.gpu_compute.available
            ):
                self._gpu_compute_fitness(
                    source_fbo=source_fbo,
                    src_w=src_w,
                    src_h=src_h,
                )
            else:
                self._capture_and_score(
                    source_fbo=source_fbo,
                    src_w=src_w,
                    src_h=src_h,
                )
        except Exception as e:
            self._logger.debug(f"Fitness capture failed: {e}")

    @staticmethod
    def _gl_get_int(enum: int) -> int | list[int] | None:
        """Safely get an integer GL state value.

        Returns a scalar ``int`` for single-value queries (e.g.
        ``GL_DEPTH_FUNC``) and a ``list[int]`` for multi-value queries
        (e.g. ``GL_VIEWPORT`` which yields four values).  Returns ``None``
        on any GL or conversion error so callers can degrade gracefully.

        PyOpenGL's ``glGetIntegerv`` frequently returns a one-element numpy
        array even for scalar queries, so we unwrap those here.
        """
        try:
            value = GL.glGetIntegerv(enum)
        except (GL.error.GLError, TypeError, ValueError):
            return None
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                # 0-d array, e.g. array(5)
                return int(value.item())
            flat = value.ravel()
            if flat.size == 0:
                return None
            if flat.size == 1:
                # 1-element array, e.g. array([5]) — unwrap to scalar
                return int(flat[0])
            # multi-element array, e.g. array([0, 0, 800, 600])
            try:
                return [int(v) for v in flat.tolist()]
            except (TypeError, ValueError):
                return None
        if isinstance(value, (list, tuple)):
            try:
                ints = [int(v) for v in value]
                return ints[0] if len(ints) == 1 else ints
            except (TypeError, ValueError):
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _gl_get_bools(enum: int) -> list[bool] | None:
        """Safely get a boolean GL state value."""
        try:
            value = GL.glGetBooleanv(enum)
        except (GL.error.GLError, TypeError, ValueError):
            return None
        if isinstance(value, np.ndarray):
            flat = value.ravel()
            if flat.size == 0:
                return None
            return [bool(v) for v in flat.tolist()]
        if isinstance(value, (list, tuple)):
            return [bool(v) for v in value]
        return [bool(value)]

    @staticmethod
    def _gl_get_flag(enum: int) -> bool | None:
        """Safely check if a GL capability is enabled."""
        try:
            return bool(GL.glIsEnabled(enum))
        except (GL.error.GLError, TypeError, ValueError):
            return None

    def _snapshot_gl_state(self) -> dict[str, Any]:
        """Capture every GL state item an offscreen scoring pass mutates.

        Uses explicit attribute queries (never ``glPushAttrib``) so the
        behaviour is identical on a core-profile GL 3.3 context. Restore with
        :meth:`_restore_gl_state`.
        """
        state: dict[str, Any] = {}

        for key, enum in (
            ("framebuffer", GL.GL_FRAMEBUFFER_BINDING),
            ("draw_fbo", GL.GL_DRAW_FRAMEBUFFER_BINDING),
            ("read_fbo", GL.GL_READ_FRAMEBUFFER_BINDING),
            ("viewport", GL.GL_VIEWPORT),
            ("scissor_box", GL.GL_SCISSOR_BOX),
            ("depth_func", GL.GL_DEPTH_FUNC),
            ("blend_src_rgb", GL.GL_BLEND_SRC_RGB),
            ("blend_dst_rgb", GL.GL_BLEND_DST_RGB),
            ("blend_src_alpha", GL.GL_BLEND_SRC_ALPHA),
            ("blend_dst_alpha", GL.GL_BLEND_DST_ALPHA),
            ("blend_equation_rgb", GL.GL_BLEND_EQUATION_RGB),
            ("blend_equation_alpha", GL.GL_BLEND_EQUATION_ALPHA),
            ("program", GL.GL_CURRENT_PROGRAM),
            ("vao", GL.GL_VERTEX_ARRAY_BINDING),
            ("array_buffer", GL.GL_ARRAY_BUFFER_BINDING),
            ("element_array_buffer", GL.GL_ELEMENT_ARRAY_BUFFER_BINDING),
            ("pixel_pack_buffer", GL.GL_PIXEL_PACK_BUFFER_BINDING),
            ("pixel_unpack_buffer", GL.GL_PIXEL_UNPACK_BUFFER_BINDING),
            ("read_buffer", GL.GL_READ_BUFFER),
            ("draw_buffer", GL.GL_DRAW_BUFFER),
        ):
            state[key] = self._gl_get_int(enum)
        state["color_writemask"] = self._gl_get_bools(GL.GL_COLOR_WRITEMASK)
        for key, enum in (
            ("scissor_test", GL.GL_SCISSOR_TEST),
            ("blend", GL.GL_BLEND),
            ("depth_test", GL.GL_DEPTH_TEST),
            ("cull_face", GL.GL_CULL_FACE),
            ("polygon_offset_fill", GL.GL_POLYGON_OFFSET_FILL),
            ("sample_alpha_to_coverage", GL.GL_SAMPLE_ALPHA_TO_COVERAGE),
            ("framebuffer_srgb", GL.GL_FRAMEBUFFER_SRGB),
            ("dither", GL.GL_DITHER),
        ):
            state[key] = self._gl_get_flag(enum)
        # glIsEnabled only accepts capability enums, so the depth write mask
        # must be queried with glGetBooleanv instead.
        try:
            mask = GL.glGetBooleanv(GL.GL_DEPTH_WRITEMASK)
            if isinstance(mask, np.ndarray):
                flat = mask.ravel()
                mask = flat[0] if flat.size > 0 else False
            elif isinstance(mask, (list, tuple)):
                mask = mask[0] if len(mask) > 0 else False
            state["depth_write"] = bool(mask)
        except (GL.error.GLError, TypeError, ValueError, IndexError) as e:
            self._logger.debug("Failed to query depth write mask: %s", e)
            state["depth_write"] = None

        active = self._gl_get_int(GL.GL_ACTIVE_TEXTURE)
        state["active_texture"] = active
        # Only snapshot the texture units actually touched by the fitness pass
        # (binding 0 = current frame tex, binding 1 = previous frame tex) instead
        # of iterating over every GL_MAX_TEXTURE_IMAGE_UNITS slot, which is
        # expensive on Mesa (often 32+) and is wasted work every frame.
        fitness_tex_units = 2
        bindings: dict[int, Any] = {}
        for unit_index in range(fitness_tex_units):
            unit = GL.GL_TEXTURE0 + unit_index
            try:
                GL.glActiveTexture(unit)
                bindings[unit] = self._gl_get_int(GL.GL_TEXTURE_BINDING_2D)
            except (GL.error.GLError, TypeError, ValueError):
                bindings[unit] = None
        try:
            if active:
                GL.glActiveTexture(self._gl_safe_int(active) or 0)
        except (GL.error.GLError, TypeError, ValueError) as e:
            self._logger.debug("Failed to restore active texture: %s", e)
        state["texture_bindings"] = bindings
        return state

    @staticmethod
    def _gl_safe_int(value: Any) -> int | None:
        """Coerce a GL state value to a scalar ``int``.

        Handles numpy arrays, lists, and tuples that may slip through by
        taking the first element when the container holds exactly one value.
        Returns ``None`` when the value is missing or cannot be coerced.
        """
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            flat = value.ravel()
            if flat.size == 0:
                return None
            if flat.size == 1:
                value = flat[0]
            else:
                return None
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            if len(value) == 1:
                value = value[0]
            else:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _restore_gl_state(self, state: dict[str, Any] | None) -> None:
        """Restore the state captured by :meth:`_snapshot_gl_state`."""
        if not state:
            return

        framebuffer = self._gl_safe_int(state.get("framebuffer"))
        if framebuffer is not None:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        draw_fbo = self._gl_safe_int(state.get("draw_fbo"))
        if draw_fbo is not None:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
        read_fbo = self._gl_safe_int(state.get("read_fbo"))
        if read_fbo is not None:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)

        viewport = state.get("viewport")
        if isinstance(viewport, list) and len(viewport) == 4:
            GL.glViewport(int(viewport[0]), int(viewport[1]), int(viewport[2]), int(viewport[3]))
        scissor_box = state.get("scissor_box")
        if isinstance(scissor_box, list) and len(scissor_box) == 4:
            GL.glScissor(
                int(scissor_box[0]), int(scissor_box[1]), int(scissor_box[2]), int(scissor_box[3])
            )

        color_mask = state.get("color_writemask")
        if isinstance(color_mask, list) and len(color_mask) >= 4:
            GL.glColorMask(*(bool(value) for value in color_mask[:4]))

        scissor_test = state.get("scissor_test")
        if scissor_test is not None:
            (GL.glEnable if scissor_test else GL.glDisable)(GL.GL_SCISSOR_TEST)
        blend = state.get("blend")
        if blend is not None:
            (GL.glEnable if blend else GL.glDisable)(GL.GL_BLEND)
            if blend:
                try:
                    GL.glBlendFuncSeparate(
                        self._gl_safe_int(state["blend_src_rgb"]) or 0,
                        self._gl_safe_int(state["blend_dst_rgb"]) or 0,
                        self._gl_safe_int(state["blend_src_alpha"]) or 0,
                        self._gl_safe_int(state["blend_dst_alpha"]) or 0,
                    )
                    GL.glBlendEquationSeparate(
                        self._gl_safe_int(state["blend_equation_rgb"]) or 0,
                        self._gl_safe_int(state["blend_equation_alpha"]) or 0,
                    )
                except (TypeError, ValueError, KeyError, GL.error.GLError) as e:
                    self._logger.debug("Failed to restore blend state: %s", e)

        depth_test = state.get("depth_test")
        if depth_test is not None:
            (GL.glEnable if depth_test else GL.glDisable)(GL.GL_DEPTH_TEST)
        depth_write = state.get("depth_write")
        if depth_write is not None:
            GL.glDepthMask(bool(depth_write))
        depth_func = self._gl_safe_int(state.get("depth_func"))
        if depth_func is not None:
            GL.glDepthFunc(depth_func)

        for key, capability in (
            ("cull_face", GL.GL_CULL_FACE),
            ("polygon_offset_fill", GL.GL_POLYGON_OFFSET_FILL),
            ("sample_alpha_to_coverage", GL.GL_SAMPLE_ALPHA_TO_COVERAGE),
            ("framebuffer_srgb", GL.GL_FRAMEBUFFER_SRGB),
            ("dither", GL.GL_DITHER),
        ):
            enabled = state.get(key)
            if enabled is not None:
                (GL.glEnable if enabled else GL.glDisable)(capability)

        for unit, tex in (state.get("texture_bindings") or {}).items():
            safe_tex = self._gl_safe_int(tex)
            if safe_tex is None:
                continue
            GL.glActiveTexture(int(unit))
            GL.glBindTexture(GL.GL_TEXTURE_2D, safe_tex)
        active = self._gl_safe_int(state.get("active_texture"))
        if active is not None:
            GL.glActiveTexture(active)

        array_buffer = self._gl_safe_int(state.get("array_buffer"))
        if array_buffer is not None:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, array_buffer)
        element_array_buffer = self._gl_safe_int(state.get("element_array_buffer"))
        if element_array_buffer is not None:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, element_array_buffer)
        pixel_pack_buffer = self._gl_safe_int(state.get("pixel_pack_buffer"))
        if pixel_pack_buffer is not None:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pixel_pack_buffer)
        pixel_unpack_buffer = self._gl_safe_int(state.get("pixel_unpack_buffer"))
        if pixel_unpack_buffer is not None:
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pixel_unpack_buffer)

        read_buffer = self._gl_safe_int(state.get("read_buffer"))
        if read_buffer is not None:
            try:
                GL.glReadBuffer(read_buffer)
            except (GL.error.GLError, TypeError, ValueError) as e:
                self._logger.debug("Failed to restore read buffer: %s", e)
        draw_buffer = self._gl_safe_int(state.get("draw_buffer"))
        if draw_buffer is not None:
            try:
                GL.glDrawBuffer(draw_buffer)
            except (GL.error.GLError, TypeError, ValueError) as e:
                self._logger.debug("Failed to restore draw buffer: %s", e)

        vao = self._gl_safe_int(state.get("vao"))
        if vao is not None:
            GL.glBindVertexArray(vao)
        program = self._gl_safe_int(state.get("program"))
        if program is not None:
            GL.glUseProgram(program)

    def _capture_and_score(self, genome=None, source_fbo=None, src_w=0, src_h=0) -> Objectives:
        """Capture current frame and score objectives.

        Args:
            genome: Genome to score. Defaults to self.genome.
            source_fbo: When set (spatial mode) the composited screen framebuffer
                is blitted into the fitness FBO instead of re-rendering the
                genome, so fitness measures the full ecosystem image.
            src_w, src_h: Pixel dimensions of *source_fbo* for the blit.

        Returns:
            Objectives vector
        """

        if genome is None:
            genome = self.genome
        if self._fitness_fbo is None or self._fitness_tex is None:
            objectives = self.fitness_evaluator.score_from_genome(genome)
            genome.objectives = objectives
            return objectives

        res = self._config.fitness_image_res
        readback = self._fitness_readback
        async_path = genome is self.genome and readback is not None and readback.gl_ready
        old_fbo = None
        gl_state = None
        try:
            gl_state = self._snapshot_gl_state()
            old_fbo = gl_state.get("framebuffer") if gl_state else None
            if not self._render_fitness_fbo(
                genome=genome,
                source_fbo=source_fbo,
                src_w=src_w,
                src_h=src_h,
            ):
                objectives = self.fitness_evaluator.score_from_genome(genome)
                genome.objectives = objectives
                return objectives

            if async_path:
                readback.read_into_pbo()
                image_data = readback.collect()
            else:
                image_data = GL.glReadPixels(0, 0, res, res, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
                image_data = (
                    bytes(bytearray(image_data))
                    if not isinstance(image_data, bytes)
                    else image_data
                )

            if image_data is None:
                objectives = self.fitness_evaluator.score_from_genome(genome)
                genome.objectives = objectives
                return objectives

            current_fitness = self.fitness_evaluator.evaluate(
                genome,
                image_data,
                self._previous_image,
                audio_data=self.audio.audio,
                previous_frames=list(self._previous_frames),
                ecosystem=getattr(self, "ecosystem", None)
                if getattr(self, "_spatial_active", False)
                else None,
            )

            self._previous_frames.append(self._previous_image)
            self._previous_image = image_data
            genome.objectives = current_fitness

            return current_fitness
        except Exception as e:
            self._logger.debug(f"Fitness capture failed: {e}")
            objectives = self.fitness_evaluator.score_from_genome(genome)
            genome.objectives = objectives
            return objectives
        finally:
            if old_fbo is not None:
                self._restore_gl_state(gl_state)

    def score(self, candidate: Any) -> float:
        """Score a genome. Returns scalar for backward compatibility."""
        if hasattr(candidate, "objectives"):
            return candidate.objectives.scalar()
        return 0.5

    def _score_genome(self, genome) -> Objectives:
        """Render a genome to the fitness FBO and return its objectives."""
        current_genome = self.genome
        program = self._get_program(genome, allow_async=False)
        self._set_program(program)
        score = self._capture_and_score(genome)
        self._set_program(self._get_program(current_genome, allow_async=False))
        self.genome = current_genome
        return score

    def _copy_phenotype(self, target, source) -> None:
        # Shader-graph genomes carry their phenotype in ``graph``; without
        # this branch the copy below is a no-op for them and accepted mutants
        # never actually enter the population.
        src_graph = getattr(source, "graph", None)
        if (
            src_graph is not None
            and hasattr(src_graph, "clone")
            and getattr(target, "graph", None) is not None
        ):
            try:
                target.graph = src_graph.clone()
            except (AttributeError, TypeError, ValueError) as e:
                self._logger.debug("Graph phenotype copy failed: %s", e)
            for attr in (
                "mutations",
                "_strength",
                "_recent_accepts",
                "_usage_tracker",
                "_prune_every",
                "_generations_since_prune",
                "last_event",
                "generation",
                "parent_id",
            ):
                if hasattr(source, attr) and hasattr(target, attr):
                    try:
                        setattr(target, attr, copy.deepcopy(getattr(source, attr)))
                    except (AttributeError, TypeError, ValueError) as e:
                        self._logger.debug("Genome attr copy %s failed: %s", attr, e)
        if hasattr(target, "params") and isinstance(target.params, dict):
            target.params = copy.deepcopy(source.params)
        else:
            for attr in (
                "field_type",
                "warp",
                "palette",
                "feedback_mode",
                "post",
                "field_params",
                "warp_params",
                "palette_params",
                "post_params",
            ):
                if hasattr(source, attr):
                    setattr(target, attr, copy.deepcopy(getattr(source, attr)))
            try:
                from evo.pipeline_genome import PipelineGenome

                if isinstance(source, PipelineGenome) and not isinstance(target, PipelineGenome):
                    field_mod = self._get_field_genome(source)
                    for attr in (
                        "field_type",
                        "warp",
                        "palette",
                        "feedback_mode",
                        "post",
                        "field_params",
                        "warp_params",
                        "palette_params",
                        "post_params",
                    ):
                        if hasattr(field_mod, attr):
                            setattr(target, attr, copy.deepcopy(getattr(field_mod, attr)))
            except ImportError:
                logging.getLogger(__name__).debug(
                    "PipelineGenome import unavailable, skipping field module copy"
                )
        if hasattr(source, "objectives") and hasattr(target, "objectives"):
            target.objectives = (
                source.objectives if isinstance(source.objectives, Objectives) else Objectives()
            )

    def _population_diversity(self) -> float:
        """Mean pairwise 0..1 phenotypic distance across the population.

        Used as the diversity driver for the dynamic mutation scheduler:
        a converged (low-diversity) population gets fewer mutations because
        small tweaks are unlikely to escape the local basin, while a diverse
        population can absorb more exploratory mutations.

        Throttled: the O(n²) pairwise ``distance()`` sweep is only recomputed
        every ``diversity_recompute_interval`` frames.  The result feeds the
        EMA-smoothed mutation rate in ``_compute_mutation_rate()``, so the
        intermediate frames reuse the cached value without perceptible lag.
        """
        if self._diversity_cache is not None:
            self._diversity_frame_counter += 1
            if self._diversity_frame_counter < self._config.diversity_recompute_interval:
                return self._diversity_cache
        self._diversity_frame_counter = 0
        self._species_cache = None  # stay in sync with diversity recompute
        organisms = getattr(self.population, "organisms", None)
        if not organisms or len(organisms) <= 1:
            self._diversity_cache = 0.0
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(organisms)):
            for j in range(i + 1, len(organisms)):
                try:
                    total += organisms[i].distance(organisms[j])
                except (AttributeError, TypeError, ValueError):
                    continue
                count += 1
        if count == 0:
            self._diversity_cache = 0.0
            return 0.0
        result = min(total / count, 1.0)
        self._diversity_cache = result
        return result

    def _invalidate_population_cache(self) -> None:
        """Drop cached diversity / species values so the next read recomputes."""
        self._diversity_cache = None
        self._species_cache = None
        self._diversity_frame_counter = 0

    def _species_count(self) -> int:
        """Number of distinct species currently alive (spec 13 driver).

        Falls back to the multi-species ecosystem, the spatial ecosystem, or
        the population lineage count when those are unavailable.

        Throttled: shares the same frame gate as
        :meth:`_population_diversity` so the heavy O(n²) diversity sweep is
        not paired with an O(n) species recompute on intermediate frames.  When
        a cached value exists it is returned directly; otherwise both signals
        are recomputed together and cached for the next *interval* frames.
        """
        if self._species_cache is not None:
            return self._species_cache
        # Cache miss — recompute (this only happens when diversity also
        # recomputed, so we share the cost).
        if self.multi_species is not None:
            try:
                result = max(1, len(self.multi_species.species))
                self._species_cache = result
                return result
            except (AttributeError, TypeError):
                logging.getLogger(__name__).debug(
                    "multi_species species lookup failed, falling back"
                )
        eco = getattr(self, "ecosystem", None)
        if eco is not None:
            sp = getattr(eco, "species", None)
            if sp:
                result = max(1, len(sp))
                self._species_cache = result
                return result
        organisms = getattr(self.population, "organisms", None)
        if organisms:
            lineages = {getattr(o, "lineage", None) for o in organisms}
            result = max(1, len(lineages))
            self._species_cache = result
            return result
        self._species_cache = 1
        return 1

    def _best_fitness(self) -> float:
        """Scalar fitness of the current best organism (for stagnation)."""
        best = None
        organisms = getattr(self.population, "organisms", None)
        if organisms:
            for o in organisms:
                try:
                    f = float(o.fitness)
                except (AttributeError, TypeError, ValueError):
                    continue
                if best is None or f > best:
                    best = f
        return best if best is not None else float(getattr(self.genome, "fitness", 0.0))

    def _update_scheduler_inputs(self, dt: float) -> None:
        """Refresh the EMA-tracked signals the scheduler reads each frame.

        Args:
            dt: Seconds since the previous frame (for FPS estimation).
        """
        fps = 1.0 / max(dt, 1e-3)
        a = 0.1
        self._ema_fps += (fps - self._ema_fps) * a

        # GPU load: prefer an explicit signal (adaptive quality controller or
        # GL timer query) and otherwise estimate it from frame-time slack.
        gpu_load = 0.4
        if hasattr(self, "_perf_monitor") and self._perf_monitor is not None:
            try:
                gpu_load = float(self._perf_monitor.gpu_load)
            except (AttributeError, TypeError, ValueError) as e:
                self._logger.debug("Failed to read perf monitor GPU load: %s", e)
        elif hasattr(self, "_gpu_timer_query") and self._gpu_timer_query is not None:
            try:
                gpu_load = float(self._gpu_timer_query.elapsed_ms / 16.0)
            except (AttributeError, TypeError, ValueError) as e:
                self._logger.debug("Failed to read GPU timer query: %s", e)
        self._ema_gpu_load += (gpu_load - self._ema_gpu_load) * a

        # Audio energy: smoothed from current frame.
        try:
            self._ema_audio_energy += (float(self.audio.energy) - self._ema_audio_energy) * 0.2
        except (AttributeError, TypeError, ValueError) as e:
            self._logger.debug("Failed to update audio energy EMA: %s", e)

    def _compute_mutation_rate(self) -> int:
        """Return the number of mutations to apply this frame."""
        cfg = self._config
        if not cfg.enable_dynamic_mutation_schedule:
            return cfg.steady_state_mutations_per_frame

        base = float(cfg.steady_state_mutations_per_frame)

        gpu = self._ema_gpu_load
        gpu_factor = 1.0 + cfg.schedule_weight_gpu * (cfg.schedule_gpu_target - gpu)

        fps_ratio = min(1.0, self._ema_fps / max(1.0, cfg.schedule_fps_target))
        fps_factor = 1.0 - cfg.schedule_weight_fps * (1.0 - fps_ratio)

        audio_factor = 1.0 + cfg.schedule_weight_audio * self._ema_audio_energy

        diversity = self._population_diversity()
        diversity_factor = 1.0 + cfg.schedule_weight_diversity * (diversity - 0.5) * 2.0

        stagnation = getattr(self, "_fitness_stagnation", 0.0)
        stagnation_factor = 1.0 + cfg.schedule_weight_stagnation * stagnation

        species = self._species_count()
        species_norm = min(1.0, species / max(1, cfg.max_species))
        species_factor = 1.0 + cfg.schedule_weight_species * (species_norm - 0.5) * 2.0

        rate = (
            base
            * gpu_factor
            * fps_factor
            * audio_factor
            * diversity_factor
            * stagnation_factor
            * species_factor
        )

        rate = max(0.0, rate)
        self._scheduled_mutations_per_frame += (
            rate - self._scheduled_mutations_per_frame
        ) * cfg.schedule_smoothing
        return round(self._scheduled_mutations_per_frame)

    def _audio_mutation_strength(self, base: float) -> float:
        """Adjust mutation strength from audio pressure.

        Quiet audio explores more, loud audio is conservative.
        """
        if self._audio_system is None:
            return base
        try:
            pressure = self._audio_system.get_audio_pressure()
            if pressure.get("is_silent", False):
                return base * 1.3
            energy = float(pressure.get("energy", 0.0))
            if energy > 0.7:
                return base * 0.8
            return base
        except (AttributeError, TypeError, ValueError) as e:
            self._logger.debug("Audio mutation strength failed: %s", e)
            return base

    def _pick_evolution_parent(self, organisms: list) -> Any:
        """Tournament-select a parent from the living population.

        Fitness-proportional-ish selection (tournament of 3) that still gives
        weaker organisms a chance to contribute genes, so crossover can
        recombine distant lineages instead of only ever mutating the champion.
        """
        if not organisms:
            return self.genome
        pool = random.sample(organisms, min(3, len(organisms)))
        best_parent = None
        best_score = float("-inf")
        for org in pool:
            try:
                score = float(getattr(org, "objectives", Objectives()).scalar())
            except (AttributeError, TypeError, ValueError):
                score = 0.0
            if score > best_score:
                best_score = score
                best_parent = org
        return best_parent if best_parent is not None else self.genome

    def _make_evolution_candidate(self, organisms: list) -> tuple[Any, Any]:
        """Produce one candidate genome plus its parent.

        Candidates are bred from the whole population: mostly clone+mutate a
        tournament winner, sometimes crossover two distinct parents, and rarely
        restart from a fresh genome (immigrant) to reopen extinct lineages.
        Returns ``(candidate, parent)``.
        """
        cfg = self._config
        crossover_rate = float(getattr(cfg, "crossover_rate", 0.3) or 0.0)
        parent = self._pick_evolution_parent(organisms)
        candidate = None
        if (
            crossover_rate > 0.0
            and len(organisms) >= 2
            and random.random() < crossover_rate
            and hasattr(parent, "crossover")
        ):
            other = self._pick_evolution_parent(organisms)
            if other is not parent:
                try:
                    candidate = parent.crossover(other)
                except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                    self._logger.debug("Crossover failed, falling back to clone: %s", e)
                    candidate = None
        if candidate is None:
            try:
                candidate = parent.clone()
            except Exception as e:
                self._logger.debug("Parent clone failed, using champion: %s", e)
                parent = self.genome
                candidate = self.genome.clone()
        is_immigrant = random.random() < 0.02
        if is_immigrant:
            try:
                candidate = type(self.genome)()
                # Keep an unmutated snapshot as the "parent" so the caller's
                # pre/post-mutation signature comparison stays meaningful.
                parent = candidate.clone()
            except Exception as e:
                self._logger.debug("Immigrant genome creation failed: %s", e)
                is_immigrant = False
        try:
            candidate.mutate(
                strength=self._audio_mutation_strength(0.02),
                large=is_immigrant,
            )
        except TypeError:
            candidate.mutate(strength=self._audio_mutation_strength(0.02))
        # Crossover products bypass mutate()'s compile-safety rollback, so
        # verify the graph can still compile; otherwise fall back to a plain
        # mutant of the parent.
        graph = getattr(candidate, "graph", None)
        validate = getattr(graph, "validate_for_compile", None)
        if callable(validate):
            try:
                from evo_shader_graph import MAX_GRAPH_DEPTH, MAX_GRAPH_NODES

                limits = {"max_nodes": MAX_GRAPH_NODES, "max_depth": MAX_GRAPH_DEPTH}
            except ImportError:
                limits = {}
            try:
                if not validate(**limits):
                    self._logger.debug("Crossover produced invalid graph, using parent mutant")
                    candidate = parent.clone()
                    candidate.mutate(strength=self._audio_mutation_strength(0.02))
            except Exception as e:
                self._logger.debug("Graph validation failed: %s", e)
        return candidate, parent

    def _steady_state_evolve(self) -> None:
        """Continuous per-frame evolution: mutate, evaluate, replace weakest.

        No generations, no timers.  Every frame we apply a small number of
        mutations, score the candidates, and promote the best ones into the
        population by replacing the weakest organisms.

        Entropy drives the mutation regime for each genome:
        * high entropy (invent) -> amplified mutation strength
        * low entropy (refine) -> dampened mutation strength
        * stagnation is detected per-genome; when fitness stops improving
          entropy is raised automatically, forcing exploratory mutations
          that escape local minima.

        Gene usage tracking records the full lifecycle:
        mutation -> visual change -> fitness change -> survival.
        """

        if getattr(self, "_multi_res", None) is not None:
            self._multi_res.step()
            champ = self._multi_res.champion()
            if champ is not None:
                try:
                    if not hasattr(champ, "objectives") or champ.objectives.scalar() <= 0.0:
                        pass
                    elif not hasattr(self.genome, "objectives") or champ.objectives.dominates(
                        self.genome.objectives
                    ):
                        self.genome = champ
                        self.population.organisms[0] = self.genome
                        if self._program is None:
                            self._set_program(self._get_program(self.genome))
                except (AttributeError, TypeError, ValueError) as e:
                    self._logger.debug("Multi-res champion handling failed: %s", e)
            return

        cfg = self._config
        stagnation_raise = float(getattr(cfg, "stagnation_raise", 0.15))

        best_fitness = self._best_fitness()

        # Feed the dynamic mutation scheduler's stagnation signal.  It was
        # initialised to 0 and never updated, so schedule_weight_stagnation
        # had no effect — mutations never ramped up when evolution stalled.
        # Stagnation rises while the best fitness fails to improve and decays
        # on any improvement.
        if best_fitness > self._last_best_fitness + 1e-6:
            self._last_best_fitness = best_fitness
            self._fitness_stagnation = max(0.0, self._fitness_stagnation - 0.05)
        else:
            self._fitness_stagnation = min(1.0, self._fitness_stagnation + 0.002)
        self._best_fitness_ema += (best_fitness - self._best_fitness_ema) * 0.05

        # Cache the genome's pre-mutation signature once per evolution step.
        # In the single-organism path ``old`` *is* ``self.genome``; in the
        # multi-organism path every candidate is a clone of ``self.genome``,
        # so the pre-mutation signature is identical for all of them.
        _genome_has_sig = hasattr(self.genome, "shader_signature")
        _cached_genome_sig = self.genome.shader_signature() if _genome_has_sig else {}

        if cfg.population_size <= 1:
            old = self.genome
            old_sig = _cached_genome_sig

            candidate = copy.deepcopy(old)
            candidate.mutate(strength=self._audio_mutation_strength(0.005))

            cand_sig = (
                candidate.shader_signature() if hasattr(candidate, "shader_signature") else {}
            )
            visual_changed = old_sig != cand_sig
            mutated_genes = list(getattr(candidate, "_mutated_genes_last", []))
            if not mutated_genes and candidate.mutations:
                last_m = candidate.mutations[-1]
                mutated_genes = [last_m.split("=")[0].split("+")[0].split("-")[0]]

            old_objectives = self.fitness_evaluator.score_from_genome(old)
            candidate_objectives = self.fitness_evaluator.score_from_genome(candidate)

            # Music pressure tie-breaker: when visual fitness is equal,
            # prefer candidates whose traits match the current music signature.
            accepted = candidate_objectives.dominates(old_objectives)
            if not accepted and self._music_pressure is not None:
                old_mp = self._music_pressure_score(old)
                cand_mp = self._music_pressure_score(candidate)
                if cand_mp > old_mp:
                    accepted = True
            if accepted:
                self._set_program(self._get_program(candidate))
                self.genome = candidate
                self.population.organisms[0] = self.genome
                if self._config.enable_gene_fitness_tracking:
                    delta = candidate_objectives.scalar() - old_objectives.scalar()
                    for m in candidate.mutations:
                        param = m.split("=")[0].split("+")[0].split("-")[0] if m else "unknown"
                        self.gene_fitness_tracker.record(param, delta)
                        self.mutation_heatmap.record(param, delta, True)
            else:
                for m in candidate.mutations:
                    param = m.split("=")[0].split("+")[0].split("-")[0] if m else "unknown"
                    self.mutation_heatmap.record(param, 0.0, False)

            self._record_gene_usage(
                mutated_genes=mutated_genes,
                visual_changed=visual_changed,
                fitness_delta=candidate_objectives.scalar() - old_objectives.scalar(),
                survived=accepted,
            )
            self._record_op_usage(
                mutated_genes, accepted, candidate_objectives.scalar() - old_objectives.scalar()
            )

            old.record_accept(accepted)
            if hasattr(old, "compute_entropy"):
                old.compute_entropy()
            if (
                hasattr(old, "_detect_stagnation")
                and old._detect_stagnation(best_fitness)
                and hasattr(old, "raise_entropy")
            ):
                old.raise_entropy(stagnation_raise)
            old.adapt_strength(
                self._config.mutation_strength_min,
                self._config.mutation_strength_max,
            )
            return

        organisms = self.population.organisms
        if not organisms:
            return

        n_mutations = self._compute_mutation_rate()
        mp_weight = self._config_get("music_pressure_weight", 0.3, float)
        candidates = []
        for _ in range(n_mutations):
            candidate, parent = self._make_evolution_candidate(organisms)
            if parent is self.genome:
                # Champion-parented candidates share the cached pre-mutation
                # signature; other parents need their own.
                old_sig = _cached_genome_sig
            else:
                old_sig = (
                    parent.shader_signature()
                    if hasattr(parent, "shader_signature")
                    else _cached_genome_sig
                )
            cand_sig = (
                candidate.shader_signature() if hasattr(candidate, "shader_signature") else {}
            )
            candidate_objectives = self.fitness_evaluator.score_from_genome(candidate)
            mp_score = self._music_pressure_score(candidate)
            candidates.append(
                (
                    candidate_objectives.scalar(),
                    candidate_objectives,
                    candidate,
                    old_sig != cand_sig,
                    mp_score,
                    parent,
                )
            )

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0] + mp_weight * x[4], reverse=True)

        # Sort organisms by fitness once (ascending = weakest first) instead
        # of calling min() per candidate.  The ranking is updated in-place
        # only when an organism's fitness actually changes.
        _org_fitness = [
            (float(getattr(o, "objectives", Objectives()).scalar()), o) for o in organisms
        ]
        _org_fitness.sort(key=lambda t: t[0])

        for score, obj, cand, visual_changed, mp_score, parent in candidates:
            weakest = _org_fitness[0][1]
            weakest_obj = getattr(weakest, "objectives", None)
            if weakest_obj is None or not isinstance(weakest_obj, Objectives):
                weakest_obj = Objectives()
            # Accept on Pareto dominance, or when the weighted scalar improves
            # meaningfully — strict dominance alone rejects every trade-off
            # (e.g. better beauty at a small novelty cost) and stalls progress.
            accepted = obj.dominates(weakest_obj)
            if not accepted:
                try:
                    accepted = obj.scalar() > weakest_obj.scalar() + 1e-4
                except (AttributeError, TypeError, ValueError):
                    accepted = False
            if accepted:
                self._copy_phenotype(weakest, cand)
                weakest.objectives = obj
                # Update the weakest organism's fitness in the ranking.
                _org_fitness[0] = (float(obj.scalar()), weakest)
                _org_fitness.sort(key=lambda t: t[0])
                if self._config.enable_gene_fitness_tracking:
                    delta = obj.scalar() - weakest_obj.scalar()
                    for m in cand.mutations:
                        param = m.split("=")[0].split("+")[0].split("-")[0] if m else "unknown"
                        self.gene_fitness_tracker.record(param, delta)
                        self.mutation_heatmap.record(param, delta, True)
                else:
                    for m in cand.mutations:
                        param = m.split("=")[0].split("+")[0].split("-")[0] if m else "unknown"
                        self.mutation_heatmap.record(param, 0.0, True)
                survived = True
                # Feed the accept/reject signal back into the surviving
                # organism: ShaderGraphGenome uses it to fill its subgraph
                # library (macros) and to adapt its own mutation strength.
                if hasattr(weakest, "record_accept"):
                    try:
                        weakest.record_accept(True)
                    except (AttributeError, TypeError, ValueError) as e:
                        self._logger.debug("record_accept failed: %s", e)
                if hasattr(weakest, "adapt_strength"):
                    try:
                        weakest.adapt_strength(
                            self._config.mutation_strength_min,
                            self._config.mutation_strength_max,
                        )
                    except (AttributeError, TypeError, ValueError) as e:
                        self._logger.debug("adapt_strength failed: %s", e)
            else:
                for m in cand.mutations:
                    param = m.split("=")[0].split("+")[0].split("-")[0] if m else "unknown"
                    self.mutation_heatmap.record(param, 0.0, False)
                survived = False
                # Attribute the rejected mutation to the parent so its
                # acceptance-rate statistics (and thus adapt_strength) see
                # both outcomes instead of only accepts.
                if parent is not weakest and hasattr(parent, "record_accept"):
                    try:
                        parent.record_accept(False)
                    except (AttributeError, TypeError, ValueError):
                        pass

            mutated_genes = list(getattr(cand, "_mutated_genes_last", []))
            if not mutated_genes and cand.mutations:
                last_m = cand.mutations[-1]
                mutated_genes = [last_m.split("=")[0].split("+")[0].split("-")[0]]

            self._record_gene_usage(
                mutated_genes=mutated_genes,
                visual_changed=visual_changed,
                fitness_delta=obj.scalar() - weakest_obj.scalar(),
                survived=survived,
            )
            self._record_op_usage(mutated_genes, survived, obj.scalar() - weakest_obj.scalar())

        now = time.time()
        if (
            now - self._last_full_fitness_ts >= self._full_fitness_interval
            and self._config.enable_visual_fitness
        ):
            self._last_full_fitness_ts = now
            best = self.population.best()
            if best is not None:
                self._score_genome(best)
                best_obj = getattr(best, "objectives", None)
                gen_obj = getattr(self.genome, "objectives", None)
                if best_obj is not None and gen_obj is not None:
                    if best_obj.dominates(gen_obj):
                        self._set_program(self._get_program(best))
                        self.genome = best
                        self.population.organisms[0] = self.genome
                    if self._config.enable_lineage:
                        self.memory_dna.record(self.genome)

        for org in organisms:
            if hasattr(org, "compute_entropy"):
                org.compute_entropy()
                try:
                    org_fitness = float(getattr(org, "objectives", Objectives()).scalar())
                except (AttributeError, TypeError, ValueError):
                    org_fitness = best_fitness
                # Stagnation must be measured against the organism's own
                # fitness history, not the population best — otherwise every
                # non-champion looks permanently stalled and is forced into
                # the high-entropy regime every frame.
                if org._detect_stagnation(org_fitness):
                    org.raise_entropy(stagnation_raise)

        # Population structure may have changed (phenotypes copied in);
        # invalidate the throttled diversity / species cache so the next
        # scheduler tick recomputes from fresh data.
        self._invalidate_population_cache()

    def _record_gene_usage(self, mutated_genes, visual_changed, fitness_delta, survived):
        """Record gene usage tracking data if enabled."""
        if not hasattr(self, "gene_usage_tracker") or self.gene_usage_tracker is None:
            return
        if not mutated_genes:
            return
        try:
            self.gene_usage_tracker.record_mutation(
                mutated_genes,
                visual_changed=visual_changed,
                visual_magnitude=abs(fitness_delta),
            )
            self.gene_usage_tracker.record_fitness_change(mutated_genes, fitness_delta)
            self.gene_usage_tracker.record_survival(mutated_genes, survived)
        except (AttributeError, TypeError, ValueError) as e:
            self._logger.debug("Gene usage tracking failed: %s", e)

    def _record_op_usage(self, mutated_genes, accepted, fitness_delta):
        """Record mutation operator usage for operator evolution."""
        if not hasattr(self, "mutation_op_tracker") or self.mutation_op_tracker is None:
            return
        if not mutated_genes:
            return
        try:
            gene = mutated_genes[0] if mutated_genes else "unknown"
            module = gene.split(".")[0] if "." in gene else gene
            op_name = f"modular:{module}" if "." not in gene else gene
            self.mutation_op_tracker.record(op_name, accepted, fitness_delta)
        except (AttributeError, TypeError, ValueError) as e:
            self._logger.debug("Mutation operator tracking failed: %s", e)

    def human_feedback(self, sentiment: float) -> None:
        """Record human preference to steer evolution.

        Args:
            sentiment: Positive (like) or negative (dislike) feedback
        """
        self._preference_bias += sentiment * 0.1
        self._preference_bias = max(-1.0, min(1.0, self._preference_bias))

        if self._config.enable_lineage:
            self.memory_dna.record(self.genome)

        self._logger.info(f"Human feedback: {sentiment:+.2f}, bias: {self._preference_bias:+.2f}")

    # ------------------------------------------------------------------
    # Ecosystem memory integration
    # ------------------------------------------------------------------

    def _update_music_signature(self) -> None:
        """Update the current music signature and pressure bias.

        Runs independently of ``enable_ecosystem_memory`` so music pressure
        can steer parent selection without the SQLite memory layer.
        """
        if not self._config_get("enable_music_pressure", True, bool):
            self._music_pressure = None
            return
        try:
            self._current_music_sig = MusicSignature.from_audio(self.audio.audio)
            self._music_pressure = MusicPressure.from_signature(self._current_music_sig)
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            self._logger.debug("Music signature update failed: %s", e)
            self._music_pressure = None
            return
        # Audio DNA analysis is still gated on the ecosystem memory layer.
        if not self._config.enable_ecosystem_memory or not self.ecosystem_memory:
            return
        try:
            from evo_renderer.audio_analyzer import AudioAnalyzer

            analyzer = AudioAnalyzer()
            if hasattr(self.audio, "raw") and self.audio.raw is not None:
                self._current_music_dna = analyzer.analyze(self.audio, fft_frame=self.audio.raw)
        except ImportError as e:
            logger.debug("Music DNA analysis failed: %s", e)

    def _music_pressure_score(self, genome: Any) -> float:
        """Return a 0..1 music-matching score for a genome's flat params.

        Uses ``_flat_genome_params`` to extract numeric traits and scores
        them against the current ``MusicPressure`` signature.
        """
        if self._music_pressure is None:
            return 0.0
        flat = self._flat_genome_params(genome)
        if not flat:
            return 0.0
        return self._music_pressure.pressure_for(flat)

    def _music_pressure_bonus(self, genome: Any) -> float:
        """Return a multiplicative bonus factor in [1, 1+weight] for a genome.

        Used to bias parent selection in the steady-state evolution loop so
        that music-matching traits (e.g. ``vortex`` for high energy) are
        preferentially promoted.
        """
        if self._music_pressure is None:
            return 1.0
        weight = self._config_get("music_pressure_weight", 0.3, float)
        score = self._music_pressure_score(genome)
        return 1.0 + weight * score

    def _check_dream_mode(self) -> bool:
        if not self._config.enable_dream_mode or not self.ecosystem_memory:
            return False
        energy = float(getattr(self.audio, "energy", 0.0))
        if energy < self._config.dream_threshold and not self._dream_active:
            self._dream_active = True
            self._logger.info(f"Dream mode activated (energy={energy:.3f})")
            return True
        elif energy >= self._config.dream_threshold and self._dream_active:
            self._dream_active = False
            self._logger.info("Dream mode deactivated")
        return self._dream_active

    def _maybe_store_music_pattern(self) -> None:
        if (
            not self._config.enable_ecosystem_memory
            or not self.ecosystem_memory
            or not self._current_music_sig
        ):
            return
        now = time.time()
        if now - self._last_music_pattern_ts >= self._music_pattern_interval_secs:
            self._last_music_pattern_ts = now
            self.ecosystem_memory.store_music_pattern(
                self._current_music_sig, self.genome, self.genome.fitness
            )

    def _maybe_evolve_world(self) -> None:
        if not self._config.enable_world_evolution or not self.ecosystem_memory:
            return
        self.ecosystem_memory.world_dna.mutate(strength=0.02)
        if not hasattr(self, "_world_save_counter"):
            self._world_save_counter = 0
        self._world_save_counter += 1
        if self._world_save_counter >= 60:
            self._world_save_counter = 0
            self.ecosystem_memory.save_world_dna()
        self._logger.debug(f"World DNA evolved: {self.ecosystem_memory.world_dna.to_dict()}")

    def _maybe_revive_species(self) -> None:
        if (
            not self._config.enable_ecosystem_memory
            or not self.ecosystem_memory
            or not self._current_music_sig
        ):
            return
        match = self.ecosystem_memory.revive_species(
            self._current_music_sig, threshold=self._config.music_match_threshold
        )
        if match:
            self._logger.info(
                f"Species revived from music match (similarity={match['similarity']:.3f}, section={match['section']})"
            )

    def _compute_long_term_fitness(self, genome) -> float:
        raw_fitness = float(genome.fitness)
        if not self._config.enable_ecosystem_memory or not self.ecosystem_memory:
            return raw_fitness * self._music_pressure_bonus(genome)
        lineage_id = getattr(genome, "lineage", "")
        adaptability = float(getattr(genome, "_strength", 0.1))
        ltf = self.ecosystem_memory.long_term_fitness(lineage_id, raw_fitness, adaptability)
        return ltf * self._music_pressure_bonus(genome)

    def dream_once(self) -> None:
        """Trigger a single dream-mode mutation."""
        if not self._config.enable_ecosystem_memory or not self.ecosystem_memory:
            return
        dreams = self.ecosystem_memory.dream(lambda: PipelineGenome.random_initial(), count=1)
        if dreams:
            candidate = dreams[0]
            frag = self._gen_frag_source(candidate)
            self._recompile_fragment_shader(frag, candidate)
            self.genome = candidate
            self.population.organisms[0] = self.genome

    def save_ecosystem(self, path: str | Path | None = None) -> bool:
        if not self.ecosystem_memory:
            return False
        if path is None and self._config.preset_dir:
            path = self._config.preset_dir / "ecosystem_memory.json"
        if path is None:
            return False
        return self.ecosystem_memory.save(path)

    def load_ecosystem(self, path: str | Path) -> bool:
        if not self.ecosystem_memory:
            return False
        return self.ecosystem_memory.load(path)

    def trigger_transition(self) -> None:
        """Trigger a large mutation.

        Compiles the new shader before swapping the genome so the rendered
        program and the genome stay in sync. A compile failure propagates
        rather than silently reverting to the previous genome.
        """
        candidate = copy.deepcopy(self.genome)
        candidate.mutate(strength=self._config.mutation_strength_max)
        frag = self._gen_frag_source(candidate)
        self._recompile_fragment_shader(frag, candidate)
        self.genome = candidate
        self.population.organisms[0] = self.genome
        if self._config.enable_lineage:
            self.memory_dna.record(self.genome)

    def save_preset(self, path: str | Path | None = None) -> bool:
        """Save current genome as a preset file.

        Args:
            path: Save path. If None, uses config preset_dir with auto-generated name.

        Returns:
            True if successful
        """
        try:
            if path is None:
                if not self._config.preset_dir:
                    self._logger.error("No preset directory configured")
                    return False
                path = self._config.preset_dir / f"preset_{int(time.time())}.json"
            else:
                path = Path(path)

            path.parent.mkdir(parents=True, exist_ok=True)

            preset_data = {
                "timestamp": time.time(),
                "genome": self.genome.to_dict(),
                "preference_bias": self._preference_bias,
                "lineage": self.genome.lineage,
                "generation": self.genome.generation,
                "parent_id": self.genome.parent_id,
                "fitness": self.genome.fitness,
                "mutations": list(self.genome.mutations),
                "population_stats": self.population.stats()
                if self._config.population_size > 1
                else None,
            }

            with open(path, "w") as f:
                json.dump(preset_data, f, indent=2, default=str)

            self._logger.info(f"Preset saved to {path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save preset: {e}")
            return False

    def load_preset(self, path: str | Path) -> bool:
        """Load genome from a preset file.

        Args:
            path: Preset file path

        Returns:
            True if successful
        """
        try:
            path = Path(path)
            with open(path) as f:
                preset_data = json.load(f)

            self.genome.from_dict(preset_data.get("genome", {}))
            self._preference_bias = float(preset_data.get("preference_bias", 0.0))
            self.genome.lineage = preset_data.get("lineage", self.genome.lineage)
            self.genome.generation = int(preset_data.get("generation", self.genome.generation))
            self.genome.parent_id = preset_data.get("parent_id")
            self.genome.fitness = float(preset_data.get("fitness", 0.0))
            self.genome.mutations = list(preset_data.get("mutations", []))

            if self.gl_ready:
                frag = self._gen_frag_source(self.genome)
                self._recompile_fragment_shader(frag, self.genome)

            self._logger.info(f"Preset loaded from {path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to load preset: {e}")
            return False

    # ------------------------------------------------------------------
    # Milkdrop preset mode (classic .milk rendering alongside evolution)
    # ------------------------------------------------------------------
    def set_available_milk_files(self, files) -> None:
        """Set the list of Milkdrop preset paths available for switching."""
        self.available_milk_files = [Path(f) for f in (files or []) if f]

    def load_milkdrop_preset(self, path: str | Path) -> bool:
        """Load and compile a ``.milk`` preset, switching to Milkdrop mode.

        Args:
            path: Path to the ``.milk`` file.

        Returns:
            True on success.  On failure the renderer stays in its current
            mode and the caller can inspect ``self._milkdrop_preset`` /
            ``self._milkdrop_program`` (both None after a failure).
        """
        try:
            preset = parse_milkdrop_file(path)
            if preset.is_empty():
                self._logger.warning(f"Milkdrop preset {path} is empty")
                return False

            compilation = self._milkdrop_compiler.compile_preset(preset)
            shader = self._milkdrop_shader_renderer.build(compilation, preset.name)
            if not shader.ok:
                self._logger.error(f"Milkdrop preset {path} failed to build: {shader.errors}")
                return False

            # Save the currently-bound program so we can restore it when
            # Milkdrop mode exits.  Never delete it — it may be cached.
            self._milkdrop_prev_prog = self._program

            if self.gl_ready:
                prog = self._compile_program(shader.vertex_source, shader.fragment_source)
            else:
                prog = 0
            self._milkdrop_program = prog
            self._milkdrop_preset = preset
            self._milkdrop_mode = True
            self._milkdrop_loc = self._collect_milkdrop_uniforms(prog)
            self._logger.info(
                f"Milkdrop preset loaded: {preset.name} "
                f"({len(compilation.per_frame)} per-frame, "
                f"{len(compilation.per_pixel)} per-pixel eqs)"
            )
            return True
        except Exception as e:
            self._logger.error(f"Failed to load Milkdrop preset {path}: {e}")
            self._milkdrop_mode = False
            self._milkdrop_preset = None
            self._milkdrop_program = None
            return False

    def exit_milkdrop_mode(self) -> None:
        """Leave Milkdrop mode and restore the evolutionary genome program."""
        if not self._milkdrop_mode:
            return
        self._milkdrop_mode = False
        self._milkdrop_preset = None
        if self._milkdrop_program is not None:
            try:
                GL.glDeleteProgram(self._milkdrop_program)
            except (GL.error.GLError, TypeError, ValueError) as e:
                self._logger.debug("Failed to delete Milkdrop program: %s", e)
            self._milkdrop_program = None
        self._milkdrop_loc = {}
        # Restore the evolutionary program if one was bound before.
        if self._milkdrop_prev_prog is not None:
            self._set_program(self._milkdrop_prev_prog)
            self._milkdrop_prev_prog = None

    def next_milkdrop_preset(self, direction: int = 1) -> bool:
        """Switch to the next/previous Milkdrop preset in the library.

        Args:
            direction: +1 for next, -1 for previous.

        Returns:
            True if a preset was loaded.
        """
        if not self.available_milk_files:
            return False
        idx = self._milkdrop_preset_index
        idx = max(idx, 0)
        idx = (idx + direction) % len(self.available_milk_files)
        self._milkdrop_preset_index = idx
        return self.load_milkdrop_preset(self.available_milk_files[idx])

    def maybe_auto_switch_milkdrop(self, dt: float) -> None:
        """Periodically auto-switch Milkdrop presets when enabled.

        Respects ``_disable_auto_switching`` (set by linamp when the user
        toggles auto-switch off) and the ``_milkdrop_auto_switch_interval``
        cooldown.
        """
        if self._disable_auto_switching or not self._milkdrop_mode:
            return
        if not self.available_milk_files:
            return
        now = time.time()
        if now - self._last_milkdrop_auto_switch_ts < self._milkdrop_auto_switch_interval:
            return
        self._last_milkdrop_auto_switch_ts = now
        self.next_milkdrop_preset(direction=1)

    def _collect_milkdrop_uniforms(self, program: int) -> dict[str, int]:
        """Query uniform locations for a compiled Milkdrop program."""
        if not program:
            return {}
        locs: dict[str, int] = {}
        for name in (
            "u_time",
            "u_resolution",
            "previous_frame",
            "u_feedback",
            "bass",
            "mid",
            "treble",
            "beat",
        ):
            locs[name] = GL.glGetUniformLocation(program, name)
        return locs

    def _apply_milkdrop_uniforms(
        self,
        bass: float,
        mid: float,
        treb: float,
        beat: float,
        t: float,
        device_w: int,
        device_h: int,
        previous_frame_tex: int,
    ) -> None:
        """Upload the shared Milkdrop uniforms before drawing."""
        locs = self._milkdrop_loc
        if not locs or self._milkdrop_program is None:
            return
        if locs.get("u_time", -1) >= 0:
            GL.glUniform1f(locs["u_time"], float(t))
        if locs.get("u_resolution", -1) >= 0:
            GL.glUniform2f(locs["u_resolution"], float(device_w), float(device_h))
        if locs.get("bass", -1) >= 0:
            GL.glUniform1f(locs["bass"], self._audio_response(bass))
        if locs.get("mid", -1) >= 0:
            GL.glUniform1f(locs["mid"], self._audio_response(mid))
        if locs.get("treble", -1) >= 0:
            GL.glUniform1f(locs["treble"], self._audio_response(treb))
        if locs.get("beat", -1) >= 0:
            GL.glUniform1f(locs["beat"], self._audio_response(beat))
        if locs.get("u_feedback", -1) >= 0:
            GL.glUniform1f(locs["u_feedback"], float(self._config.feedback_alpha))
        if locs.get("previous_frame", -1) >= 0:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, int(previous_frame_tex))
            GL.glUniform1i(locs["previous_frame"], 0)

    def on_realize(self, area: Gtk.GLArea) -> None:
        """Handle widget realization and GL initialization.

        Args:
            area: The GLArea widget
        """
        try:
            self.make_current()

            self._logger.info("OpenGL context made current")

            # The GL context is current here, so this is the safe place to
            # initialise the disk cache (constructor only built the object).
            if getattr(self, "disk_cache", None) is not None and getattr(
                self, "_disk_cache_needs_init", False
            ):
                try:
                    self.disk_cache.init_gl()
                except (RuntimeError, OSError, GL.error.GLError) as exc:
                    self._logger.debug("Shader disk cache unavailable: %s", exc)
                    self.disk_cache = None
                except Exception as exc:
                    self._logger.warning(
                        "Unexpected error initializing shader disk cache: %s", exc
                    )
                    self.disk_cache = None
                finally:
                    self._disk_cache_needs_init = False
            try:
                self._create_fullscreen_quad()
                self._logger.debug("Fullscreen quad created")
            except (GL.error.GLError, RuntimeError, MemoryError) as e:
                self._logger.error("Failed to build fullscreen quad: %s", e)
                raise
            except Exception as e:
                self._logger.error("Unexpected error building fullscreen quad: %s", e)
                raise

            try:
                self.feedback.init_gl()
                self._logger.debug("Feedback buffer initialized")
            except (GL.error.GLError, RuntimeError, MemoryError) as e:
                self._logger.error("Failed to init feedback buffer: %s", e)
                raise
            except Exception as e:
                self._logger.error("Unexpected error initializing feedback buffer: %s", e)
                raise

            try:
                if self._config_get("enable_effects", True, bool):
                    mgr = EffectManager(self._config, self._logger)
                    mgr.bind_compiler(
                        lambda vs, fs, key: self._compile_program(
                            vs, fs, cache_key=key
                        )
                    )
                    if mgr.enabled:
                        self._effects = mgr
                        self._logger.info("Evolving effects enabled: %s", mgr.status())
            except Exception as e:
                self._logger.warning("Effects init failed: %s", e)
                self._effects = None

            try:
                if self._config_get("enable_folder_visualizer", False, bool):
                    fv = FolderVisualizer(self._config, self._effects, self._logger)
                    fv.bind_compiler(
                        lambda vs, fs, key: self._compile_program(
                            vs, fs, cache_key=key
                        )
                    )
                    if fv.enabled:
                        self._folder_viz = fv
                        self._logger.info(
                            "Folder visualizer enabled: %s", fv.status()
                        )
            except Exception as e:
                self._logger.warning("Folder visualizer init failed: %s", e)
                self._folder_viz = None

            try:
                if self._config_get("enable_visual_fitness", False, bool):
                    self._init_fitness_fbo()
            except (GL.error.GLError, RuntimeError, MemoryError) as e:
                self._logger.warning("Fitness FBO init skipped: %s", e)
            except Exception as e:
                self._logger.warning("Unexpected error initializing fitness FBO: %s", e)

            try:
                if self.gpu_compute is not None:
                    self.gpu_compute.init_gl()
                    self._logger.info("GPU compute fitness initialized")
            except (GL.error.GLError, RuntimeError, MemoryError) as e:
                self._logger.warning("GPU compute init skipped: %s", e)
            except Exception as e:
                self._logger.warning("Unexpected error initializing GPU compute: %s", e)

            try:
                if getattr(self, "_multi_res", None) is not None:
                    self._multi_res.init_gl()
                    self._logger.info("Multi-resolution evolution GL initialized")
            except (GL.error.GLError, RuntimeError, MemoryError) as e:
                self._logger.warning("Multi-res GL init skipped: %s", e)
            except Exception as e:
                self._logger.warning("Unexpected error initializing multi-res GL: %s", e)

            try:
                frag = self._gen_frag_source(self.genome)
                self._recompile_fragment_shader(frag, self.genome)
                self._logger.info("Initial shader compiled successfully")
            except (GL.error.GLError, RuntimeError, ValueError) as e:
                self._logger.error("Failed to compile initial shader: %s", e)
                raise
            except Exception as e:
                self._logger.error("Unexpected error compiling initial shader: %s", e)
                raise

            try:
                self._ensure_mp_pipeline()
            except (RuntimeError, AttributeError) as e:
                self._logger.warning("Multi-pass init skipped: %s", e)
            except Exception as e:
                self._logger.warning("Unexpected error initializing multi-pass: %s", e)

            # Spec 6: start the async shader-compilation worker now that the GL
            # context exists. Subsequent mutations queue their GLSL source build
            # here instead of stalling the render thread.
            if self.async_shader_compiler is not None:
                try:
                    self.async_shader_compiler.start()
                    self._logger.info("Async shader compiler started")
                except (RuntimeError, OSError) as e:
                    self._logger.warning("Async shader compiler start failed: %s", e)
                    self.async_shader_compiler = None
                except Exception as e:
                    self._logger.warning("Unexpected error starting async shader compiler: %s", e)
                    self.async_shader_compiler = None

            try:
                if self.ecosystem is not None:
                    self._init_resource_texture()
                    # Initialise the per-organism map atlas so _upload_org_map_atlas
                    # can pack tiles instead of allocating a fresh texture each
                    # frame.  The packer is a persistent GL object; init it
                    # once here and reuse it for the life of the context.
                    if getattr(self, "_perf_atlas_packer", None) is not None:
                        try:
                            self._perf_atlas_packer.init_gl()
                            self._atlas_packer = self._perf_atlas_packer
                            self._logger.debug("Organism map atlas packer initialized")
                        except (GL.error.GLError, RuntimeError, MemoryError) as e:
                            self._logger.debug("Atlas packer init failed: %s", e)
                        except Exception as e:
                            self._logger.debug("Unexpected error initializing atlas packer: %s", e)
            except Exception as e:
                self._logger.warning(f"Resource texture init skipped: {e}")

            self.gl_ready = True

            if self._tick_id is None:
                self._tick_id = self.add_tick_callback(self._on_tick)
                self._logger.debug("Tick callback registered")

            self.queue_render()

        except Exception as e:
            self._logger.critical(f"Realization failed: {e}")
            self.gl_ready = False

    def _on_tick(self, widget: Gtk.GLArea, frame_clock) -> bool:
        """Tick callback to request continuous redraws.

        Without this GTK only repaints on resize/realize, freezing the output.

        Args:
            widget: The GLArea widget
            frame_clock: Frame clock (unused)

        Returns:
            True to continue receiving ticks
        """
        self.queue_render()
        return True

    # ------------------------------------------------------------------
    # Spatial ecosystem rendering (specs 46-60)
    # ------------------------------------------------------------------

    def _init_resource_texture(self) -> None:
        """Create the GPU resource-field texture (specs 48, 60)."""
        if self.ecosystem is None:
            return
        w = self.ecosystem.field.w
        h = self.ecosystem.field.h
        self._resource_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._resource_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._resource_tex_size = (w, h)

    def _upload_resource_texture(self) -> None:
        if self._resource_tex is None or self.ecosystem is None:
            return
        data = array.array("f", self.ecosystem.field.rgba_array())
        w, h = self._resource_tex_size
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._resource_tex)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGBA, GL.GL_FLOAT, data.tobytes()
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _upload_org_map_atlas(self) -> None:
        """Pack every organism's four maps into one GPU texture (spec 31).

        Uses :class:`TextureAtlasPacker` for efficient shelf-based packing
        and :class:`BindlessTextureArray` for bindless sampler access,
        eliminating per-organism texture binds.

        Channels per texel: R=food, G=pheromone, B=danger, A=territory. One
        tile per organism; each organism's shader samples only its own tile.
        """
        eco = self.ecosystem
        adv = getattr(eco, "advanced", None) if eco is not None else None
        if adv is None or not hasattr(adv, "map_atlas_frame"):
            return
        frame = adv.map_atlas_frame()
        if frame is None:
            return
        buf, atlas = frame
        w, h = int(atlas.width), int(atlas.height)
        if w <= 0 or h <= 0:
            return
        if hasattr(buf, "tobytes"):
            payload = buf.tobytes()
        else:
            payload = array.array("f", buf).tobytes()

        # Use the atlas packer for efficient tile packing.
        packer = getattr(self, "_atlas_packer", None)
        if packer is not None and packer.available:
            org_id = id(self)
            # pack_tile uploads the pixel data immediately by default,
            # so the explicit update_tile + upload() below are redundant.
            tile = packer.pack_tile(org_id, payload, tile_w=w, tile_h=h)
            if tile is not None:
                self._org_map_tex = packer.texture_id()
                self._org_map_size = (w, h)
                self._org_map_atlas_packer = packer
                return

        # Fallback: direct upload.
        if self._org_map_tex is None:
            self._org_map_tex = GL.glGenTextures(1)
            self._org_map_size = (0, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._org_map_tex)
        if self._org_map_size != (w, h):
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_FLOAT, payload
            )
            # NEAREST so tiles never bleed into each other across the atlas.
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            self._org_map_size = (w, h)
        else:
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGBA, GL.GL_FLOAT, payload)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _spatialize_shader(self, src: str) -> str:
        """Inject territory + species + spatial-mode uniforms into a generated shader.

        ``sample_uv`` is remapped to global world coordinates so feedback
        trails persist across the whole world, while the organism's own pattern
        still uses its local ``v_uv``.
        """
        if "u_territory_min" in src:
            return src
        decl = (
            "\nuniform vec2 u_territory_min;\n"
            "uniform vec2 u_territory_size;\n"
            "uniform float u_org_energy;\n"
            "uniform vec3 u_org_tint;\n"
            "uniform sampler2D u_resource_tex;\n"
            "uniform sampler2D u_world_tex;\n"
            "uniform sampler2D u_world_tex2;\n"
            "uniform sampler2D u_org_maps;\n"
            "uniform vec2 u_org_map_uv_min;\n"
            "uniform vec2 u_org_map_uv_size;\n"
            "uniform float u_org_map_strength;\n"
            "uniform float u_spatial_mode;\n"
            "uniform float u_spatial_blend;\n"
            "uniform float u_spatial_group;\n"
            "uniform vec2 u_spatial_group_cx;\n"
        )
        src = src.replace("void main() {", decl + "void main() {", 1)
        src = src.replace(
            "    vec2 sample_uv = v_uv;",
            "    vec2 sample_uv = u_territory_min + v_uv * u_territory_size;",
            1,
        )
        src = src.replace(
            "    fragColor = vec4(col, 1.0);",
            "    vec4 rfield = texture(u_resource_tex, u_territory_min + v_uv * u_territory_size);\n"
            "    col += rfield.rgb * 0.15;\n"
            "    vec4 world = texture(u_world_tex, u_territory_min + v_uv * u_territory_size);\n"
            "    col += vec3(1.0, 0.3, 0.0) * world.r * 0.15;\n"
            "    col += vec3(0.3, 0.9, 0.2) * world.g * 0.12;\n"
            "    col += vec3(0.2, 0.4, 1.0) * world.b * 0.10;\n"
            "    col = mix(col, col * (0.35 + 0.65 * u_org_tint), world.a * 0.15);\n"
            "    vec4 world2 = texture(u_world_tex2, u_territory_min + v_uv * u_territory_size);\n"
            "    col += vec3(0.9, 0.1, 0.05) * world2.r * 0.12;\n"
            "    col += vec3(0.5, 0.2, 0.9) * world2.g * 0.10;\n"
            "    col += vec3(0.5, 0.45, 0.4) * world2.b * 0.08;\n"
            "    col *= mix(0.2, 1.0, clamp(u_org_energy, 0.0, 1.0));\n"
            "    col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.55);\n"
            "    // Spec 31: composite this organism's own four maps so evolution\n"
            "    // (food seeking, pheromone trails, danger memory, territory\n"
            "    // claiming) becomes visible right inside its territory.\n"
            "    vec2 map_uv = u_org_map_uv_min + clamp(v_uv, 0.0, 1.0) * u_org_map_uv_size;\n"
            "    vec4 omap = texture(u_org_maps, map_uv);\n"
            "    float m_food = omap.r;\n"
            "    float m_phero = omap.g;\n"
            "    float m_danger = omap.b;\n"
            "    float m_terr = omap.a;\n"
            "    float ms = u_org_map_strength;\n"
            "    // Spatial mode blending\n"
            "    float sp_mode = u_spatial_mode + 0.5;\n"
            "    // uv already exists in scope (vec2 uv = v_uv; from assembler), no re-declaration needed\n"
            "    // Colonies: shared group color bleeds from colony center\n"
            "    if (sp_mode > 1.5 && sp_mode < 2.5) {\n"
            "        float gc_d = length(uv - (u_spatial_group_cx - u_territory_min) / u_territory_size);\n"
            "        float gc = smoothstep(0.7, 0.0, gc_d) * 0.25;\n"
            "        col = mix(col, u_org_tint * 1.4, gc);\n"
            "    }\n"
            "    // Cells: bright Voronoi boundary\n"
            "    if (sp_mode > 4.5 && sp_mode < 5.5) {\n"
            "        float edge = smoothstep(0.02, 0.05, abs(uv.x - 0.5)) + smoothstep(0.02, 0.05, abs(uv.y - 0.5));\n"
            "        col += vec3(0.3, 0.6, 1.0) * edge * 0.4;\n"
            "    }\n"
            "    // Islands: water-edge tint\n"
            "    if (sp_mode > 5.5 && sp_mode < 6.5) {\n"
            "        float edge = smoothstep(0.01, 0.06, uv.x) * smoothstep(0.01, 0.06, 1.0 - uv.x)\n"
            "                   * smoothstep(0.01, 0.06, uv.y) * smoothstep(0.01, 0.06, 1.0 - uv.y);\n"
            "        col = mix(vec3(0.0, 0.15, 0.4), col, edge);\n"
            "    }\n"
            "    // Veins: pulse along connections\n"
            "    if (sp_mode > 6.5) {\n"
            "        float pulse = 0.5 + 0.5 * sin(u_time * 4.0 + uv.x * 20.0);\n"
            "        col = mix(col, u_org_tint * 1.3, pulse * 0.3 * (1.0 - abs(uv.y - 0.5) * 2.0));\n"
            "    }\n"
            "    // Soft blend falloff for clouds and soft territory\n"
            "    if (u_spatial_blend > 0.5 && u_spatial_blend < 1.5) {\n"
            "        float d = length(uv - 0.5) * 2.0;\n"
            "        col *= 1.0 - smoothstep(0.6, 1.0, d) * 0.4;\n"
            "    }\n"
            "    col += vec3(0.9, 0.85, 0.35) * m_food * 0.5 * ms;\n"
            "    col = mix(col, vec3(0.2, 0.9, 0.7), clamp(m_phero * 0.6 * ms, 0.0, 0.7));\n"
            "    col = mix(col, vec3(1.0, 0.12, 0.1), clamp(m_danger * 0.8 * ms, 0.0, 0.85));\n"
            "    float border = smoothstep(0.35, 0.5, abs(uv.x - 0.5)) + smoothstep(0.35, 0.5, abs(uv.y - 0.5));\n"
            "    col += u_org_tint * m_terr * border * 0.4 * ms;\n"
            "    col *= (0.65 + 0.5 * m_terr * ms);\n"
            "    fragColor = vec4(col, 1.0);",
            1,
        )
        return src

    def _get_eco_program(self, org: SpatialOrganism, lod_tier: int = 2) -> int | None:
        is_camera = hasattr(org, "genome") and hasattr(org.genome, "FLOAT_GENES")
        if is_camera:
            key = f"camera_organism:{id(org)}"
            prog = self._eco_programs.get(key)
            if prog is None:
                try:
                    prog = self._compile_program(self.VERTEX_SHADER, self.CAMERA_ORGANISM_FRAG)
                    self._eco_programs[key] = prog
                except Exception as e:
                    self._logger.debug(f"Camera organism shader compile failed: {e}")
                    return None
            return prog

        lod_tag = f"lod{lod_tier}:"
        key = lod_tag + str(self.shader_cache.key(org.genome))
        prog = self._eco_programs.get(key)
        if prog is None:
            genome = org.genome
            raymarch = None
            fractal = None
            post = list(genome.post)
            if lod_tier <= 0:
                raymarch = 12
                fractal = 2
                post = [p for p in post if p in ("glow", "feedback")]
            elif lod_tier == 1:
                raymarch = 24
                fractal = 3
                post = [p for p in post if p in ("glow", "feedback", "distortion")]
            lod_genome = genome.clone()
            lod_genome.post = post
            if fractal is not None:
                lod_genome.field_params["fractal_layers"] = min(
                    int(lod_genome.field_params.get("fractal_layers", 4)), fractal
                )
            if raymarch is not None and "march_steps" in lod_genome.field_params:
                lod_genome.field_params["march_steps"] = min(
                    int(lod_genome.field_params.get("march_steps", 48)), raymarch
                )
            try:
                frag = self.shader_factory.generate(
                    lod_genome, brightness=1.0, feedback_alpha=0.85
                )[0]
                frag = self._spatialize_shader(frag)
                if self.multi_species is not None:
                    try:
                        frag = self.multi_species.species_shader_injections(org, frag)
                    except (AttributeError, TypeError, ValueError) as e:
                        self._logger.debug("Multi-species shader injection failed: %s", e)
                # Try loading a cached binary before compiling from source.
                bin_cache = getattr(self, "_perf_shader_bin_cache", None)
                source_hash = hash(frag)
                if bin_cache is not None and bin_cache.available:
                    try:
                        new_prog = self._compile_program(self.VERTEX_SHADER, frag)
                        if bin_cache.try_load(new_prog, str(source_hash)):
                            self._eco_programs[key] = new_prog
                            if self.ecosystem is not None:
                                self._eco_used[key] = self.ecosystem.step_count
                            return new_prog
                    except (GL.error.GLError, TypeError, ValueError) as e:
                        self._logger.debug("Shader binary cache load failed: %s", e)
                prog = self._compile_program(self.VERTEX_SHADER, frag)
                self._eco_programs[key] = prog
                # Cache the binary for future sessions.
                if bin_cache is not None and bin_cache.available:
                    try:
                        bin_cache.store(prog, str(source_hash))
                    except (AttributeError, TypeError, ValueError) as e:
                        self._logger.debug("Shader binary cache store failed: %s", e)
            except Exception as e:
                self._logger.error(f"Eco shader compile failed for {getattr(org, 'id', '?')}: {e}")
                prog = None
        if prog is not None and self.ecosystem is not None:
            self._eco_used[key] = self.ecosystem.step_count
        return prog

    def _bind_eco_program(self, program: int) -> dict[str, int]:
        if program == self._cur_eco_prog and program in self._eco_loc:
            return self._eco_loc[program]
        GL.glUseProgram(program)
        self._cur_eco_prog = program
        loc = {
            "time": GL.glGetUniformLocation(program, "u_time"),
            "res": GL.glGetUniformLocation(program, "u_resolution"),
            "bass": GL.glGetUniformLocation(program, "bass"),
            "mid": GL.glGetUniformLocation(program, "mid"),
            "treb": GL.glGetUniformLocation(program, "treble"),
            "prev": GL.glGetUniformLocation(program, "previous_frame"),
            "tmin": GL.glGetUniformLocation(program, "u_territory_min"),
            "tsize": GL.glGetUniformLocation(program, "u_territory_size"),
            "energy": GL.glGetUniformLocation(program, "u_org_energy"),
            "tint": GL.glGetUniformLocation(program, "u_org_tint"),
            "restex": GL.glGetUniformLocation(program, "u_resource_tex"),
            "worldtex": GL.glGetUniformLocation(program, "u_world_tex"),
            "worldtex2": GL.glGetUniformLocation(program, "u_world_tex2"),
            "orgmaps": GL.glGetUniformLocation(program, "u_org_maps"),
            "mapmin": GL.glGetUniformLocation(program, "u_org_map_uv_min"),
            "mapsize": GL.glGetUniformLocation(program, "u_org_map_uv_size"),
            "mapstr": GL.glGetUniformLocation(program, "u_org_map_strength"),
            "spatial_mode": GL.glGetUniformLocation(program, "u_spatial_mode"),
            "spatial_blend": GL.glGetUniformLocation(program, "u_spatial_blend"),
            "spatial_group": GL.glGetUniformLocation(program, "u_spatial_group"),
            "spatial_group_cx": GL.glGetUniformLocation(program, "u_spatial_group_cx"),
        }
        # Animation uniforms for spatial organisms
        anim_names = [
            "anim_zoom",
            "anim_rot",
            "anim_cx",
            "anim_cy",
            "anim_dx",
            "anim_dy",
            "anim_sx",
            "anim_sy",
            "anim_warp",
            "anim_wave_r",
            "anim_wave_g",
            "anim_wave_b",
            "anim_wave_x",
            "anim_wave_y",
            "anim_ob_size",
            "anim_ob_r",
            "anim_ob_g",
            "anim_ob_b",
            "anim_ob_a",
            "anim_ib_size",
            "anim_ib_r",
            "anim_ib_g",
            "anim_ib_b",
            "anim_ib_a",
            "anim_mv_x",
            "anim_mv_y",
            "anim_mv_l",
            "anim_mv_r",
            "anim_mv_g",
            "anim_mv_b",
            "anim_mv_a",
        ]
        for name in anim_names:
            loc[name] = GL.glGetUniformLocation(program, name)
        self._eco_loc[program] = loc
        return loc

    def _draw_organism(
        self,
        org: SpatialOrganism,
        program: int,
        prev_tex: int,
        W: int,
        H: int,
        rect,
        t: float,
        bass: float,
        mid: float,
        treb: float,
        world_tex: int = 0,
        world_tex2: int = 0,
        lod_tier: int = 2,
        spatial_mode: str = "territories",
        spatial_blend: str = "soft",
    ) -> None:
        loc = self._bind_eco_program(program)
        x0, y0, w, h = rect
        iw = max(1, int(w))
        ih = max(1, int(h))
        GL.glViewport(int(x0), int(y0), iw, ih)
        GL.glScissor(int(x0), int(y0), iw, ih)

        if loc["time"] >= 0:
            GL.glUniform1f(loc["time"], float(t))
        if loc["res"] >= 0:
            GL.glUniform2f(loc["res"], float(w), float(h))
        if loc["bass"] >= 0:
            GL.glUniform1f(loc["bass"], float(bass))
        if loc["mid"] >= 0:
            GL.glUniform1f(loc["mid"], float(mid))
        if loc["treb"] >= 0:
            GL.glUniform1f(loc["treb"], float(treb))

        gmin_x = x0 / W
        gmin_y = y0 / H
        gsize_x = w / W
        gsize_y = h / H
        if loc["tmin"] >= 0:
            GL.glUniform2f(loc["tmin"], gmin_x, gmin_y)
        if loc["tsize"] >= 0:
            GL.glUniform2f(loc["tsize"], gsize_x, gsize_y)

        en = max(
            0.0, min(1.0, org.energy / (self.ecosystem.max_energy if self.ecosystem else 3.0))
        )
        if loc["energy"] >= 0:
            GL.glUniform1f(loc["energy"], en)

        tint = (1.0, 1.0, 1.0)
        if self.ecosystem is not None and org.species in self.ecosystem.species:
            tint = self.ecosystem.species[org.species].color
        if loc["tint"] >= 0:
            GL.glUniform3f(loc["tint"], float(tint[0]), float(tint[1]), float(tint[2]))

        if loc["restex"] >= 0 and self._resource_tex is not None:
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._resource_tex)
            GL.glUniform1i(loc["restex"], 1)
        if loc.get("worldtex", -1) >= 0 and world_tex:
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, world_tex)
            GL.glUniform1i(loc["worldtex"], 2)
        if loc.get("worldtex2", -1) >= 0:
            GL.glActiveTexture(GL.GL_TEXTURE4)
            GL.glBindTexture(GL.GL_TEXTURE_2D, world_tex2)
            GL.glUniform1i(loc["worldtex2"], 4)
        # Spec 31: bind the per-organism map atlas and this organism's tile.
        # The atlas is bound once on texture unit 3 and each organism only
        # uploads its own tile UVs -- no per-org texture bind, no bindless
        # handle (which would require a uint64_t uniform the generated
        # shaders never declare).
        if loc.get("orgmaps", -1) >= 0 and self._org_map_tex:
            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._org_map_tex)
            GL.glUniform1i(loc["orgmaps"], 3)
            umin = (0.0, 0.0)
            usize = (0.0, 0.0)
            strength = float(getattr(self._config, "organism_map_strength", 0.6))
            atlas_packer = getattr(self, "_org_map_atlas_packer", None)
            if atlas_packer is not None and atlas_packer.available:
                tile = atlas_packer.tile_for(id(org))
                if tile is not None:
                    umin = (tile.texcoord_u, tile.texcoord_v)
                    usize = (tile.texcoord_w, tile.texcoord_h)
            if loc.get("mapmin", -1) >= 0:
                GL.glUniform2f(loc["mapmin"], umin[0], umin[1])
            if loc.get("mapsize", -1) >= 0:
                GL.glUniform2f(loc["mapsize"], usize[0], usize[1])
            if loc.get("mapstr", -1) >= 0:
                GL.glUniform1f(loc["mapstr"], strength)

        if loc["prev"] >= 0:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, prev_tex)
            GL.glUniform1i(loc["prev"], 0)

        # Spatial mode uniforms
        mode_id = 0.0
        mode_map = {
            "territories": 0.0,
            "colonies": 1.0,
            "clouds": 2.0,
            "flocks": 3.0,
            "cells": 4.0,
            "islands": 5.0,
            "veins": 6.0,
        }
        mode_id = mode_map.get(spatial_mode, 0.0)
        if loc.get("spatial_mode", -1) >= 0:
            GL.glUniform1f(loc["spatial_mode"], mode_id)
        if loc.get("spatial_blend", -1) >= 0:
            GL.glUniform1f(
                loc["spatial_blend"],
                {"hard": 0.0, "soft": 1.0, "glow": 2.0, "fluid": 3.0}.get(spatial_blend, 1.0),
            )
        if loc.get("spatial_group", -1) >= 0:
            GL.glUniform1f(loc["spatial_group"], float(getattr(org, "_group_id", 0)))
        if loc.get("spatial_group_cx", -1) >= 0:
            GL.glUniform2f(
                loc["spatial_group_cx"], float(org._group_center[0]), float(org._group_center[1])
            )

        # Animation uniforms (Milkdrop-style per-frame animation)
        if hasattr(org, "genome") and hasattr(org.genome, "get_animation_uniforms"):
            anim_values = org.genome.get_animation_uniforms()
            for name, value in anim_values.items():
                u_loc = loc.get(name, -1)
                if u_loc >= 0:
                    GL.glUniform1f(u_loc, float(value))

        # Rasterise this organism's territory. The caller binds the shared
        # fullscreen-quad VAO before the loop and this method sets the
        # per-organism scissor box + viewport, so this draw is what actually
        # executes the organism's shader for its rect. Without it every
        # organism pass is pure state setup and the spatial view stays black.
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

    def _compute_spatial_layout(self, orgs: list, W: int, H: int) -> list:
        """Compute pixel territory rectangles using the configured spatial mode."""
        mode = getattr(self._config, "spatial_mode", "territories")

        if mode == "territories":
            return self._territories_layout(orgs, W, H)
        elif mode == "colonies":
            return self._colonies_layout(orgs, W, H)
        elif mode == "clouds":
            return self._clouds_layout(orgs, W, H)
        elif mode == "flocks":
            return self._flocks_layout(orgs, W, H)
        elif mode == "cells":
            return self._cells_layout(orgs, W, H)
        elif mode == "islands":
            return self._islands_layout(orgs, W, H)
        elif mode == "veins":
            return self._veins_layout(orgs, W, H)
        else:
            return self._territories_layout(orgs, W, H)

    def _territories_layout(self, orgs: list, W: int, H: int) -> list:
        """Compute pixel territory rectangles with competition (spec 47)."""
        min_dim = float(min(W, H))
        raw = []
        for o in orgs:
            en = max(0.0, min(1.0, o.energy / 3.0))
            cr = o.radius * (0.4 + 0.6 * en)
            cx = o.pos[0] * W
            cy = o.pos[1] * H
            half = cr * min_dim
            raw.append([cx, cy, cr, half, en])
        n = len(raw)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(raw[i][0] - raw[j][0], raw[i][1] - raw[j][1])
                min_d = raw[i][2] + raw[j][2]
                if 1e-4 < d < min_d:
                    weak, strong = (i, j) if raw[i][4] <= raw[j][4] else (j, i)
                    new_r = max(0.005, min_d - raw[strong][2] - 0.002)
                    raw[weak][2] = new_r
                    raw[weak][3] = new_r * min_dim
        rects = []
        for cx, cy, _cr, half, _en in raw:
            if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(half)):
                cx = random.uniform(0.1, 0.9) * W
                cy = random.uniform(0.1, 0.9) * H
                half = 0.05 * min_dim
            x0 = max(0.0, min(float(W) - 1.0, cx - half))
            y0 = max(0.0, min(float(H) - 1.0, cy - half))
            w = max(1.0, min(float(W) - x0, 2.0 * half))
            h = max(1.0, min(float(H) - y0, 2.0 * half))
            rects.append((x0, y0, w, h))
        return rects

    def _voronoi_cells_for_points(self, points: list[list[float]], W: int, H: int) -> list:
        """Compute Approximate Voronoi diagram for a set of seed points.

        Returns a list of (x0, y0, w, h, cell_poly) tuples where cell_poly is
        a list of (x, y) vertices (in pixel coords). For GPU rendering we
        return a bounding rect for each seed.

        Caches results based on point positions to avoid recomputing every frame.
        """
        n = len(points)
        if n == 0:
            return []

        # Create a cache key from point positions (rounded to reduce cache misses)
        cache_key = tuple((round(p[0], 4), round(p[1], 4)) for p in points)
        cached = self._voronoi_cache.get(cache_key)
        if cached is not None:
            return cached

        # Use a grid-based approximation for the GPU
        step = max(4, min(16, W // 40))
        grid_w = math.ceil(W / step)
        grid_h = math.ceil(H / step)

        # Vectorized computation using numpy for speed
        try:
            pts = np.array(points, dtype=np.float32)
            pts[:, 0] *= W
            pts[:, 1] *= H

            gx_coords = (np.arange(grid_w) + 0.5) * step
            gy_coords = (np.arange(grid_h) + 0.5) * step

            # Compute all distances at once using broadcasting
            # Shape: (grid_w, grid_h, n)
            dx = gx_coords[:, None, None] - pts[None, None, :, 0]
            dy = gy_coords[None, :, None] - pts[None, None, :, 1]
            dists = dx * dx + dy * dy

            owner = np.argmin(dists, axis=2)  # Shape: (grid_w, grid_h)

            rects = []
            for k in range(n):
                mask = owner == k
                if np.any(mask):
                    gx_indices, gy_indices = np.where(mask)
                    x0 = float(gx_indices.min()) * step
                    y0 = float(gy_indices.min()) * step
                    x1 = float(gx_indices.max() + 1) * step
                    y1 = float(gy_indices.max() + 1) * step
                    rects.append(
                        (
                            max(0.0, x0),
                            max(0.0, y0),
                            max(1.0, min(W - x0, x1 - x0)),
                            max(1.0, min(H - y0, y1 - y0)),
                        )
                    )
                else:
                    cx = points[k][0] * W
                    cy = points[k][1] * H
                    r = 20.0
                    rects.append((max(0.0, cx - r), max(0.0, cy - r), r * 2, r * 2))
        except (AttributeError, TypeError, ValueError, ImportError, np.linalg.LinAlgError) as e:
            # Fallback to original implementation if numpy fails
            self._logger.debug("NumPy Voronoi failed, falling back: %s", e)
            owner = [[-1] * grid_h for _ in range(grid_w)]

            for gx_idx in range(grid_w):
                gx = (gx_idx + 0.5) * step
                for gy_idx in range(grid_h):
                    gy = (gy_idx + 0.5) * step
                    best_d = 1e9
                    best_k = -1
                    for k, p in enumerate(points):
                        dx = gx - p[0] * W
                        dy = gy - p[1] * H
                        d = dx * dx + dy * dy
                        if d < best_d:
                            best_d = d
                            best_k = k
                    owner[gx_idx][gy_idx] = best_k

            rects = []
            for k in range(n):
                x0 = W
                y0 = H
                x1 = 0.0
                y1 = 0.0
                found = False
                for gx_idx in range(grid_w):
                    for gy_idx in range(grid_h):
                        if owner[gx_idx][gy_idx] == k:
                            found = True
                            x0 = min(x0, gx_idx * step)
                            y0 = min(y0, gy_idx * step)
                            x1 = max(x1, (gx_idx + 1) * step)
                            y1 = max(y1, (gy_idx + 1) * step)
                if found:
                    rects.append(
                        (
                            max(0.0, x0),
                            max(0.0, y0),
                            max(1.0, min(W - x0, x1 - x0)),
                            max(1.0, min(H - y0, y1 - y0)),
                        )
                    )
                else:
                    cx = points[k][0] * W
                    cy = points[k][1] * H
                    r = 20.0
                    rects.append((max(0.0, cx - r), max(0.0, cy - r), r * 2, r * 2))

        # Cache the result (limit cache size)
        if len(self._voronoi_cache) >= 32:
            self._voronoi_cache.clear()
        self._voronoi_cache[cache_key] = rects
        return rects

    def _colonies_layout(self, orgs: list, W: int, H: int) -> list:
        """Organisms are grouped into colonies by species; colonies get Voronoi
        territory; all organisms within a colony share the colony's visual."""
        if not orgs:
            return []

        # Assign group ids by species cluster
        species_grid: dict[int, list[SpatialOrganism]] = {}
        for o in orgs:
            species_grid.setdefault(o.species, []).append(o)

        colony_centers = []
        colony_map = {}
        for cid, members in enumerate(species_grid.values()):
            cx = sum(m.pos[0] for m in members) / len(members)
            cy = sum(m.pos[1] for m in members) / len(members)
            colony_centers.append([cx, cy])
            for m in members:
                colony_map[id(m)] = cid
                m._group_id = cid
                m._group_center = (cx, cy)

        # Add some noise to colony centers to make boundaries interesting
        for c in colony_centers:
            c[0] += random.uniform(-0.05, 0.05)
            c[1] += random.uniform(-0.05, 0.05)
            c[0] = max(0.05, min(0.95, c[0]))
            c[1] = max(0.05, min(0.95, c[1]))

        colony_rects = self._voronoi_cells_for_points(colony_centers, W, H)

        # Each organism gets the colony rect + a sub-region
        rects = []
        for o in orgs:
            cid = colony_map.get(id(o), 0)
            if cid < len(colony_rects):
                cx0, cy0, cw, ch = colony_rects[cid]
            else:
                cx0, cy0 = o.pos[0] * W, o.pos[1] * H
                cw = ch = 40.0

            # Sub-divide colony rect among its members
            en = max(0.0, min(1.0, o.energy / 3.0))
            base_r = 12.0 + 18.0 * en
            px = o.pos[0] * W
            py = o.pos[1] * H
            half = base_r

            x0 = max(cx0, max(0.0, px - half))
            y0 = max(cy0, max(0.0, py - half))
            x1 = min(cx0 + cw, min(float(W), px + half))
            y1 = min(cy0 + ch, min(float(H), py + half))
            rects.append((x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)))
        return rects

    def _clouds_layout(self, orgs: list, W: int, H: int) -> list:
        """Soft overlapping gaussian blobs; territories expand with energy.
        No hard borders between organisms."""
        rects = []
        for o in orgs:
            en = max(0.0, min(1.0, o.energy / 3.0))
            cx = o.pos[0] * W
            cy = o.pos[1] * H
            radius = o.radius * (0.3 + 0.9 * en) * float(min(W, H))
            # Cloud mode uses larger, overlapping territories
            radius *= 1.4
            x0 = max(0.0, cx - radius)
            y0 = max(0.0, cy - radius)
            x1 = min(float(W), cx + radius)
            y1 = min(float(H), cy + radius)
            rects.append((x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)))
        return rects

    def _flocks_layout(self, orgs: list, W: int, H: int) -> list:
        """Each organism rendered as a compact swarm tile; flock neighbors
        share a larger collective territory."""
        if not orgs:
            return []

        # Build flock groups from proximity
        flock_radius = 0.12
        parent = list(range(len(orgs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(len(orgs)):
            for j in range(i + 1, len(orgs)):
                d = math.hypot(orgs[i].pos[0] - orgs[j].pos[0], orgs[i].pos[1] - orgs[j].pos[1])
                if d < flock_radius:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i, o in enumerate(orgs):
            groups.setdefault(find(i), []).append(i)

        rects = []
        for members in groups.values():
            # Collective center
            cx = sum(orgs[i].pos[0] for i in members) / len(members)
            cy = sum(orgs[i].pos[1] for i in members) / len(members)
            r = 20.0 + 8.0 * len(members)
            px = cx * W
            py = cy * H
            x0 = max(0.0, px - r)
            y0 = max(0.0, py - r)
            x1 = min(float(W), px + r)
            y1 = min(float(H), py + r)
            for i in members:
                rects.append((x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)))
        return rects

    def _cells_layout(self, orgs: list, W: int, H: int) -> list:
        """True Voronoi cell for each organism. Sharp boundaries."""
        points = [[o.pos[0], o.pos[1]] for o in orgs]
        raw = self._voronoi_cells_for_points(points, W, H)
        rects = []
        for o, (x0, y0, w, h) in zip(orgs, raw):
            # Shrink slightly to show cell wall
            shrink = 2.0
            sx0 = max(0.0, x0 + shrink)
            sy0 = max(0.0, y0 + shrink)
            sx1 = min(float(W), x0 + w - shrink)
            sy1 = min(float(H), y0 + h - shrink)
            rects.append((sx0, sy0, max(1.0, sx1 - sx0), max(1.0, sy1 - sy0)))
        return rects

    def _islands_layout(self, orgs: list, W: int, H: int) -> list:
        """Cluster organisms into islands with water gaps between them."""
        if not orgs:
            return []

        n_islands = max(3, min(6, len(orgs) // 3))
        # Simple k-means style clustering
        centers = []
        for i in range(n_islands):
            centers.append([random.random(), random.random()])

        for _ in range(20):
            sums = [[0.0, 0.0, 0] for _ in range(n_islands)]
            for o in orgs:
                best = 0
                best_d = 1e9
                for k, c in enumerate(centers):
                    d = math.hypot(o.pos[0] - c[0], o.pos[1] - c[1])
                    if d < best_d:
                        best_d = d
                        best = k
                sums[best][0] += o.pos[0]
                sums[best][1] += o.pos[1]
                sums[best][2] += 1
            for k in range(n_islands):
                if sums[k][2] > 0:
                    centers[k][0] = sums[k][0] / sums[k][2]
                    centers[k][1] = sums[k][1] / sums[k][2]

        # Assign groups
        for o in orgs:
            best = 0
            best_d = 1e9
            for k, c in enumerate(centers):
                d = math.hypot(o.pos[0] - c[0], o.pos[1] - c[1])
                if d < best_d:
                    best_d = d
                    best = k
            o._group_id = best
            o._group_center = tuple(centers[best])

        # Compute island territories
        island_rects = self._voronoi_cells_for_points(centers, W, H)

        rects = []
        for o in orgs:
            cid = o._group_id
            if cid < len(island_rects):
                ix0, iy0, iw, ih = island_rects[cid]
            else:
                ix0 = o.pos[0] * W - 30
                iy0 = o.pos[1] * H - 30
                iw = ih = 60.0

            en = max(0.0, min(1.0, o.energy / 3.0))
            r = 10.0 + 12.0 * en
            px = o.pos[0] * W
            py = o.pos[1] * H
            x0 = max(ix0, max(0.0, px - r))
            y0 = max(iy0, max(0.0, py - r))
            x1 = min(ix0 + iw, min(float(W), px + r))
            y1 = min(iy0 + ih, min(float(H), py + r))
            rects.append((x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)))
        return rects

    def _veins_layout(self, orgs: list, W: int, H: int) -> list:
        """Build connection network (MST) between organisms; each organism
        owns a territory at the end of a vein."""
        if not orgs:
            return []

        # Compute MST
        edges = []
        n = len(orgs)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(orgs[i].pos[0] - orgs[j].pos[0], orgs[i].pos[1] - orgs[j].pos[1])
                edges.append((d, i, j))
        edges.sort()

        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                return True
            return False

        vein_edges = []
        for d, i, j in edges:
            if union(i, j):
                vein_edges.append((i, j))
                if len(vein_edges) >= n - 1:
                    break

        # Store vein connections for shader use
        conn = [set() for _ in range(n)]
        for i, j in vein_edges:
            conn[i].add(j)
            conn[j].add(i)
        for i, o in enumerate(orgs):
            o._vein_connections = sorted(conn[i])

        # Each organism gets its own territory rect
        rects = []
        for o in orgs:
            en = max(0.0, min(1.0, o.energy / 3.0))
            cr = o.radius * (0.4 + 0.6 * en)
            cx = o.pos[0] * W
            cy = o.pos[1] * H
            half = cr * float(min(W, H)) * 0.6
            x0 = max(0.0, cx - half)
            y0 = max(0.0, cy - half)
            x1 = min(float(W), cx + half)
            y1 = min(float(H), cy + half)
            rects.append((x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)))
        return rects

    def _build_indirect_commands(
        self, orgs: list, rects: list, lod_tiers: list, W: int, H: int
    ) -> int:
        """Build GL_DRAW_INDIRECT_BUFFER commands for batch organism rendering.

        .. deprecated:: disabled
           Unconditionally returns 0 and never fills the command buffer.
           Each organism requires its own program, viewport/scissor, uniform
           set and texture bindings (see :meth:`_draw_organism`), but an
           indirect batch binds a *single* program for every command, so the
           commands built here would render every organism with whichever
           program happened to be current. Callers must use the direct
           per-organism loop in :meth:`_render_spatial` until the shaders and
           the organism SSBO support explicit per-organism indexing.
        """
        return 0

    def _write_org_ssbo(self, orgs: list, W: int, H: int) -> int:
        """Write organism parameters into the persistent SSBO.

        Returns the number of organisms written.
        """
        org_ssbo = getattr(self, "_perf_org_ssbo", None)
        if org_ssbo is None or not org_ssbo.available:
            return 0
        eco = self.ecosystem
        max_energy = eco.max_energy if eco and hasattr(eco, "max_energy") else 3.0
        n = org_ssbo.write_organisms(
            orgs,
            get_params=lambda org: [
                float(org.pos[0]),
                float(org.pos[1]),
                float(org.energy) / max(max_energy, 0.01),
                float(getattr(org, "radius", 0.1)),
                float(getattr(org, "_lod_tier", 2)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
        return n

    def _prune_eco_programs(self) -> None:
        if self.ecosystem is None:
            return
        cur = self.ecosystem.step_count
        for key in list(self._eco_programs.keys()):
            if cur - self._eco_used.get(key, -9999) > 180:
                prog = self._eco_programs.pop(key)
                if prog not in self._eco_programs.values():
                    try:
                        GL.glDeleteProgram(prog)
                    except Exception as e:
                        logger.debug("Failed to delete GL program %s: %s", prog, e)
                self._eco_loc.pop(prog, None)
                self._eco_used.pop(key, None)

    def _compute_lod_tiers_sim(self, organisms: list, W: int, H: int) -> None:
        """Compute LOD tiers for simulation step (before render).

        This is called before the simulation step so the CPU update path can
        skip expensive subsystems for far organisms.
        """
        if not self._config_get("enable_adaptive_lod", False, bool):
            for org in organisms:
                org._lod_tier = 2
            return

        lod_near = self._config_get("lod_near_px", 32.0, float)
        lod_far = self._config_get("lod_far_px", 16.0, float)
        min_dim = float(min(W, H))

        for org in organisms:
            cr = org.radius * (0.4 + 0.6 * max(0.0, min(1.0, org.energy / 3.0)))
            if cr >= lod_near / min_dim:
                org._lod_tier = 2
            elif cr >= lod_far / min_dim:
                org._lod_tier = 1
            else:
                org._lod_tier = 0

    def _compute_lod_tiers_render(
        self, rects: list[tuple[float, float, float, float]]
    ) -> list[int]:
        """Compute LOD tiers from rendered territory size for shader selection.

        Returns:
            List of LOD tiers (0=far, 1=mid, 2=near) for each rectangle
        """
        if not self._config_get("enable_adaptive_lod", False, bool):
            return [2] * len(rects)

        lod_near = self._config_get("lod_near_px", 32.0, float)
        lod_far = self._config_get("lod_far_px", 16.0, float)

        lod_tiers = []
        for x0, y0, w, h in rects:
            half = min(w, h) / 2.0
            if half >= lod_near:
                lod_tiers.append(2)
            elif half >= lod_far:
                lod_tiers.append(1)
            else:
                lod_tiers.append(0)
        return lod_tiers

    def _apply_frustum_culling(self, orgs: list, rects: list) -> tuple[list, list]:
        """Apply frustum/visibility culling to organism list.

        Returns:
            Tuple of (visible_orgs, visible_rects)
        """
        frustum = getattr(self, "frustum", None)
        if frustum is None:
            return orgs, rects

        try:
            frustum.set_mvp(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )
        except (AttributeError, TypeError, ValueError, GL.error.GLError) as e:
            self._logger.debug("Frustum MVP setup failed: %s", e)
            return orgs, rects

        visible_orgs = []
        visible_rects = []
        for org, rect in zip(orgs, rects):
            if frustum.rect_visible(rect[0], rect[1], rect[2], rect[3]):
                visible_orgs.append(org)
                visible_rects.append(rect)

        return visible_orgs, visible_rects

    def _render_spatial(self, dt: float, W: int, H: int, default_fbo: int | None = None) -> bool:
        """Render the spatial ecosystem as territorial per-organism passes.

        Assumes the caller has already stepped the simulation (``eco.step``).

        Cost model: this mirrors the feedback (non-spatial) path, which also
        draws the active genome twice per frame -- once into the ping-pong
        feedback buffer for trail accumulation and once, fresh, to the screen.
        Spatial mode therefore runs each organism's shader twice per frame
        (feedback pass + display pass). The two passes are not redundant: the
        feedback pass accumulates into ``feedback.dst_fbo()`` while the display
        pass draws over the cleared screen, so their pixels legitimately differ.
        Per-organism programs are resolved once per frame and reused by both
        passes so only the GPU work is duplicated.
        """
        with self._perf_timer("render_spatial_total", log_threshold_ms=10.0):
            eco = self.ecosystem
            if eco is None:
                return False

            # The GTK-managed on-screen FBO must be captured by the caller *before*
            # any GL setup here rebinds the framebuffer to 0 (world_tex.init_gl /
            # feedback.resize both leave the binding at 0). Fall back to reading it
            # at entry if not supplied, but never infer it mid-render.
            if default_fbo is None:
                default_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)

            if hasattr(eco, "world_tex") and eco.world_tex.tex is None and self.gl_ready:
                try:
                    eco.world_tex.init_gl(self._fbo_pool)
                except (RuntimeError, MemoryError, GL.error.GLError) as e:
                    self._logger.warning("World texture init failed: %s", e)
                except Exception as e:
                    self._logger.warning("Unexpected error initializing world texture: %s", e)

            # Spec 14 (Adaptive LOD): assign LOD tiers before the simulation step
            self._compute_lod_tiers_sim(eco.organisms, W, H)

            # Feature 6: Multi-species extensions
            if self.multi_species is not None:
                try:
                    self.multi_species.step(dt, self.audio.audio)
                except (RuntimeError, AttributeError) as e:
                    self._logger.debug("Multi-species step failed: %s", e)
                except Exception as e:
                    self._logger.warning("Unexpected error in multi-species step: %s", e)
            if self._parasite_system is not None:
                try:
                    self._parasite_system.step(dt)
                except (RuntimeError, AttributeError) as e:
                    self._logger.debug("Parasite step failed: %s", e)
                except Exception as e:
                    self._logger.warning("Unexpected error in parasite step: %s", e)

            self._upload_resource_texture()
            self._upload_org_map_atlas()

            bass, mid, treb, _beat = self.audio.uvec()
            t = time.time() - self._start_time

            self.feedback.resize(W, H)

            orgs = eco.alive_organisms()
            mode = self._config_get("spatial_mode", "territories", str)
            blend = self._config_get("spatial_blend", "soft", str)
            rects = self._compute_spatial_layout(orgs, W, H)

            # Apply frustum culling
            orgs, rects = self._apply_frustum_culling(orgs, rects)

            # Recompute exact LOD from rendered territory size for shader selection
            lod_tiers = self._compute_lod_tiers_render(rects)

            GL.glBindVertexArray(self._vao)

            world_tex_id = getattr(getattr(eco, "world_tex", None), "tex", None) or 0
            world_tex2_id = getattr(getattr(eco, "world_tex", None), "tex2", None) or 0

            # Per-organism map atlas is bound on texture unit 3 inside
            # _draw_organism() itself, so no handle is needed here.

            # Resolve each organism's program once per frame. Both the feedback pass
            # and the display pass run the same per-organism shaders, and the lookup
            # (shader-cache key + LOD variant) is not free.
            programs = [self._get_eco_program(org, lod) for org, lod in zip(orgs, lod_tiers)]

            # Select blend mode based on spatial mode
            is_overlapping = mode in ("clouds", "flocks")

            # Feedback pass (accumulate world trails).
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.feedback.dst_fbo())
            GL.glEnable(GL.GL_SCISSOR_TEST)
            prev_tex = self.feedback.src_tex()

            # Write organism parameters into the persistent SSBO.
            org_ssbo = getattr(self, "_perf_org_ssbo", None)
            if org_ssbo is not None and org_ssbo.available:
                try:
                    self._write_org_ssbo(orgs, W, H)
                    org_ssbo.bind(binding=0)
                except (AttributeError, TypeError, ValueError, GL.error.GLError) as e:
                    self._logger.debug("SSBO write/bind failed: %s", e)

            # GPU timer queries for per-organism cost measurement.
            gpu_timer = getattr(self, "_perf_gpu_timer", None)
            timer_available = gpu_timer is not None and gpu_timer.available

            # Multi-draw indirect is intentionally disabled: each organism needs
            # its own program, viewport/scissor, uniforms and textures
            # (see _draw_organism), but an indirect batch binds a single program
            # and has no per-organism program selection. Re-enable only once the
            # shaders/SSBO support explicit per-organism indexing. The capability
            # probe (_perf_multi_draw) is still created so its stats stay available.
            use_indirect = False

            if not use_indirect:
                with self._perf_timer("spatial_feedback_pass"):
                    for org_idx, (org, rect, lod) in enumerate(zip(orgs, rects, lod_tiers)):
                        prog = programs[org_idx]
                        if prog is None:
                            continue
                        if timer_available:
                            gpu_timer.begin(org_idx)
                        self._draw_organism(
                            org,
                            prog,
                            prev_tex,
                            W,
                            H,
                            rect,
                            t,
                            bass,
                            mid,
                            treb,
                            world_tex=world_tex_id,
                            world_tex2=world_tex2_id,
                            lod_tier=lod,
                            spatial_mode=mode,
                            spatial_blend=blend,
                        )
                        if timer_available:
                            gpu_timer.end()
                    if timer_available:
                        gpu_timer.collect()

            GL.glDisable(GL.GL_SCISSOR_TEST)
            self.feedback.swap(default_fbo)

            # Screen pass - restore GTK's default framebuffer, not 0.
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, default_fbo)
            GL.glViewport(0, 0, W, H)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            if is_overlapping:
                GL.glEnable(GL.GL_BLEND)
                if blend == "glow":
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
                elif blend == "fluid":
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                else:  # soft
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
                # Overlapping modes render all organisms without scissor so they blend together
                prev_tex = self.feedback.src_tex()
                with self._perf_timer("spatial_screen_pass"):
                    for org, rect, lod, prog in zip(orgs, rects, lod_tiers, programs):
                        if prog is None:
                            continue
                        self._draw_organism(
                            org,
                            prog,
                            prev_tex,
                            W,
                            H,
                            rect,
                            t,
                            bass,
                            mid,
                            treb,
                            world_tex=world_tex_id,
                            world_tex2=world_tex2_id,
                            lod_tier=lod,
                            spatial_mode=mode,
                            spatial_blend=blend,
                        )
                GL.glDisable(GL.GL_BLEND)
            else:
                GL.glEnable(GL.GL_SCISSOR_TEST)
                prev_tex = self.feedback.src_tex()
                with self._perf_timer("spatial_screen_pass"):
                    for org, rect, lod, prog in zip(orgs, rects, lod_tiers, programs):
                        if prog is None:
                            continue
                        self._draw_organism(
                            org,
                            prog,
                            prev_tex,
                            W,
                            H,
                            rect,
                            t,
                            bass,
                            mid,
                            treb,
                            world_tex=world_tex_id,
                            world_tex2=world_tex2_id,
                            lod_tier=lod,
                            spatial_mode=mode,
                            spatial_blend=blend,
                        )
                GL.glDisable(GL.GL_SCISSOR_TEST)

            # Draw vein connections for veins mode
            if mode == "veins":
                self._draw_vein_connections(
                    orgs, W, H, world_tex_id, world_tex2_id, t, bass, mid, treb
                )

            # Draw island water gaps
            if mode == "islands":
                self._draw_island_gaps(orgs, W, H, rects, t)

            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            self._cur_eco_prog = None
            self._prune_eco_programs()
            return True

    def _draw_vein_connections(self, orgs, W, H, world_tex, world_tex2, t, bass, mid, treb):
        """Draw vein connection lines between linked organisms."""
        if not orgs:
            return

        vein_prog = getattr(self, "_vein_prog", None)
        if vein_prog is None:
            vs = """
#version 330 core
layout(location = 0) in vec2 position;
void main(){ gl_Position = vec4(position, 0.0, 1.0); }
"""
            fs = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec3 u_tint;
void main(){
    vec2 uv = v_uv;
    float pulse = 0.5 + 0.5 * sin(u_time * 3.0 + uv.x * 10.0);
    vec3 col = u_tint * (0.4 + 0.6 * pulse);
    col += vec3(0.1, 0.05, 0.2) * (1.0 - abs(uv.y - 0.5) * 2.0);
    fragColor = vec4(col, 0.6);
}
"""
            try:
                vein_prog = self._compile_program(vs, fs)
                self._vein_prog = vein_prog
                self._vein_loc = {
                    "time": GL.glGetUniformLocation(vein_prog, "u_time"),
                    "res": GL.glGetUniformLocation(vein_prog, "u_resolution"),
                    "tint": GL.glGetUniformLocation(vein_prog, "u_tint"),
                }
            except Exception as e:
                self._logger.debug(f"Vein program compile failed: {e}")
                return

        GL.glUseProgram(vein_prog)
        GL.glUniform1f(self._vein_loc["time"], t)
        GL.glUniform2f(self._vein_loc["res"], W, H)

        conn_pairs = []
        for i, org in enumerate(orgs):
            for j in org._vein_connections:
                if j < len(orgs) and i < j:
                    conn_pairs.append((i, j))
        topo_key = tuple(sorted(conn_pairs))

        vein_list = []
        for i, org in enumerate(orgs):
            if not org._vein_connections:
                continue
            x1, y1 = org.pos[0] * W, org.pos[1] * H
            tint = (1.0, 1.0, 1.0)
            if self.ecosystem and org.species in self.ecosystem.species:
                tint = self.ecosystem.species[org.species].color

            for conn_idx in org._vein_connections:
                if conn_idx >= len(orgs):
                    continue
                other = orgs[conn_idx]
                x2, y2 = other.pos[0] * W, other.pos[1] * H
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy) or 1e-3
                nx = -dy / length * 1.5
                ny = dx / length * 1.5
                vein_list.append(
                    (
                        tint,
                        array.array(
                            "f",
                            [
                                x1 + nx,
                                y1 + ny,
                                x1 - nx,
                                y1 - ny,
                                x2 + nx,
                                y2 + ny,
                                x1 - nx,
                                y1 - ny,
                                x2 + nx,
                                y2 + ny,
                                x2 - nx,
                                y2 - ny,
                            ],
                        ),
                    )
                )

        topo_changed = getattr(self, "_vein_topo_version", -1) != topo_key

        draw_groups = []
        vert_offset = 0
        for tint, verts in vein_list:
            count = len(verts) // 2
            if draw_groups and draw_groups[-1][2] == tint:
                prev_offset, prev_count, prev_tint = draw_groups[-1]
                draw_groups[-1] = (prev_offset, prev_count + count, prev_tint)
            else:
                draw_groups.append((vert_offset, count, tint))
            vert_offset += count

        if topo_changed:
            self._vein_topo_version = topo_key
            if not vein_list:
                if getattr(self, "_vein_vbo", None) is not None:
                    GL.glDeleteBuffers(1, [self._vein_vbo])
                    self._vein_vbo = None
                if getattr(self, "_vein_vao", None) is not None:
                    GL.glDeleteVertexArrays(1, [self._vein_vao])
                    self._vein_vao = None
                self._vein_draw_groups = []
                return

            flat = array.array("f")
            for _, verts in vein_list:
                flat.extend(verts)

            if self._vein_vbo is None:
                self._vein_vbo = GL.glGenBuffers(1)
            if self._vein_vao is None:
                self._vein_vao = GL.glGenVertexArrays(1)

            GL.glBindVertexArray(self._vein_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vein_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, len(flat) * 4, flat.tobytes(), GL.GL_DYNAMIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
            GL.glBindVertexArray(0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

            self._vein_draw_groups = draw_groups
        else:
            if vein_list and getattr(self, "_vein_vbo", None) is not None:
                flat = array.array("f")
                for _, verts in vein_list:
                    flat.extend(verts)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vein_vbo)
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, len(flat) * 4, flat.tobytes())
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            elif not vein_list:
                return
            self._vein_draw_groups = draw_groups

        if not self._vein_draw_groups:
            return

        GL.glBindVertexArray(self._vein_vao)
        tint_loc = self._vein_loc.get("tint", -1)
        for vert_offset, count, tint in self._vein_draw_groups:
            if tint_loc >= 0:
                GL.glUniform3f(tint_loc, tint[0], tint[1], tint[2])
            GL.glDrawArrays(GL.GL_TRIANGLES, vert_offset, count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _draw_island_gaps(self, orgs, W, H, rects, t):
        """Draw water gaps between islands."""
        if not orgs or len(orgs) < 2:
            return

        # Use a simple full-screen quad with shader for water gaps
        gap_prog = getattr(self, "_gap_prog", None)
        if gap_prog is None:
            gap_prog = self._compile_program(
                self.VERTEX_SHADER,
                """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_island_centers[12];
uniform float u_island_radii[12];
uniform int u_island_count;
void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float dmin = 1.0;
    for (int i = 0; i < 12; i++) {
        if (i >= u_island_count) break;
        float d = length(uv - u_island_centers[i]);
        dmin = min(dmin, d);
    }
    float water = smoothstep(0.0, 0.02, dmin);
    float wave = 0.02 * sin(uv.x * 30.0 + u_time * 2.0) * sin(uv.y * 20.0 + u_time * 1.5);
    water = max(water, wave);
    fragColor = vec4(0.0, 0.1, 0.3, (1.0 - water) * 0.5);
}
""",
            )
            self._gap_prog = gap_prog
            self._gap_loc = {
                "previous_frame": GL.glGetUniformLocation(gap_prog, "previous_frame"),
                "time": GL.glGetUniformLocation(gap_prog, "u_time"),
                "res": GL.glGetUniformLocation(gap_prog, "u_resolution"),
                "centers": GL.glGetUniformLocation(gap_prog, "u_island_centers"),
                "radii": GL.glGetUniformLocation(gap_prog, "u_island_radii"),
                "count": GL.glGetUniformLocation(gap_prog, "u_island_count"),
            }

        if gap_prog is None:
            return

        # Collect island centers from organisms
        island_centers = {}
        for o in orgs:
            cid = getattr(o, "_group_id", 0)
            if cid not in island_centers:
                island_centers[cid] = (o._group_center[0], o._group_center[1])

        centers = list(island_centers.values())[:12]
        c_arr = []
        r_arr = []
        for cx, cy in centers:
            c_arr.extend([cx, cy])
            r_arr.append(0.08)

        GL.glUseProgram(gap_prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.feedback.src_tex())
        gl = self._gap_loc
        if gl["previous_frame"] >= 0:
            GL.glUniform1i(gl["previous_frame"], 0)
        if gl["time"] >= 0:
            GL.glUniform1f(gl["time"], t)
        if gl["res"] >= 0:
            GL.glUniform2f(gl["res"], W, H)

        loc_c = gl["centers"]
        if loc_c >= 0 and c_arr:
            GL.glUniform2fv(loc_c, len(centers), array.array("f", c_arr))
        loc_r = gl["radii"]
        if loc_r >= 0 and r_arr:
            GL.glUniform1fv(loc_r, len(r_arr), array.array("f", r_arr))
        loc_n = gl["count"]
        if loc_n >= 0:
            GL.glUniform1i(loc_n, len(centers))

        GL.glBindVertexArray(self._vao)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def toggle_spatial_ecosystem(self) -> None:
        """Toggle the spatial ecosystem renderer on/off."""
        if self.ecosystem is None and not self._config.enable_spatial_ecosystem:
            try:
                self.ecosystem = EnhancedSpatialEcosystem(self._config)
                self._config.enable_spatial_ecosystem = True
            except Exception as e:
                self._logger.error(f"Could not start spatial ecosystem: {e}")
                return
        if self._spatial_active is False and self.gl_ready:
            try:
                self._init_resource_texture()
                if hasattr(self.ecosystem, "world_tex") and self.ecosystem.world_tex.tex is None:
                    self.ecosystem.world_tex.init_gl(self._fbo_pool)
            except Exception as e:
                self._logger.warning(f"Resource texture lazy-init skipped: {e}")
        self._spatial_active = not self._spatial_active
        self._logger.info(f"Spatial ecosystem {'on' if self._spatial_active else 'off'}")
        self.queue_render()

    def toggle_emergent_camera(self) -> None:
        if self.camera_population is None and not self._config.enable_emergent_camera:
            try:
                from evo_camera import CameraPopulation

                self.camera_population = CameraPopulation(size=self._config.camera_population_size)
                self._config.enable_emergent_camera = True
            except Exception as e:
                self._logger.error(f"Could not start emergent camera: {e}")
                return
        self._config.enable_emergent_camera = not self._config.enable_emergent_camera
        self._logger.info(
            f"Emergent camera {'on' if self._config.enable_emergent_camera else 'off'}"
        )
        self.queue_render()

    def toggle_gpu_compute(self) -> None:
        self._config.enable_gpu_compute = not self._config.enable_gpu_compute
        self._logger.info(f"GPU compute {'on' if self._config.enable_gpu_compute else 'off'}")
        self.queue_render()

    def toggle_dynamic_shader(self) -> None:
        self._config.enable_dynamic_shader_graph = not self._config.enable_dynamic_shader_graph
        self._logger.info(
            f"Dynamic shader graph {'on' if self._config.enable_dynamic_shader_graph else 'off'}"
        )
        self.queue_render()

    def toggle_multi_species(self) -> None:
        self._config.enable_multi_species = not self._config.enable_multi_species
        self._logger.info(
            f"Multi-species renderer {'on' if self._config.enable_multi_species else 'off'}"
        )
        self.queue_render()

    def toggle_effects(self) -> bool:
        """Toggle the evolving post-effects layer on/off.

        Returns the new enabled state (False when no manager exists).
        """
        fx = self._effects
        if fx is None:
            return False
        fx.enabled = not fx.enabled
        self._config.enable_effects = fx.enabled
        self._logger.info(f"Evolving effects {'on' if fx.enabled else 'off'}")
        self.queue_render()
        return fx.enabled

    def next_effect(self) -> bool:
        """Force the effect pool to evolve to a new shader now."""
        fx = self._effects
        if fx is None or not fx.enabled:
            return False
        ok = fx.evolve_now()
        self.queue_render()
        return ok

    def effects_status(self) -> dict:
        """Snapshot of the evolving-effects state for UI/debug display."""
        fx = self._effects
        if fx is None:
            return {"enabled": False, "active": None}
        return fx.status()

    def folder_viz_status(self) -> dict:
        """Snapshot of the folder visualiser (shaders/+effects/+textures/)."""
        fv = self._folder_viz
        if fv is None:
            return {"enabled": False, "active": None}
        return fv.status()

    def toggle_folder_visualizer(self) -> bool:
        """Toggle the folder visualiser on/off at runtime.

        Returns the new enabled state.  Disabling falls back to the genome
        renderer; enabling requires the manager to have been initialised.
        """
        fv = self._folder_viz
        if fv is None:
            return False
        fv.enabled = not fv.enabled
        self._config.enable_folder_visualizer = fv.enabled
        self._logger.info(
            "Folder visualizer %s", "on" if fv.enabled else "off"
        )
        return fv.enabled

    def on_unrealize(self, area: Gtk.GLArea) -> None:
        """Handle widget unrealization and GL cleanup.

        Args:
            area: The GLArea widget
        """
        try:
            if self._tick_id is not None:
                self.remove_tick_callback(self._tick_id)
                self._tick_id = None
        except Exception as e:
            self._logger.warning(f"Error removing tick callback: {e}")

        # Spec 6: stop the async shader worker before tearing down GL so no
        # source generation is queued while the context is being destroyed.
        try:
            if getattr(self, "async_shader_compiler", None) is not None:
                self.async_shader_compiler.stop()
        except Exception as e:
            self._logger.warning(f"Error stopping async shader compiler: {e}")

        # Clean up GL resources
        try:
            if getattr(self, "_fitness_readback", None) is not None:
                self._fitness_readback.cleanup()
                self._fitness_readback = None
        except Exception as e:
            self._logger.warning(f"Error deleting fitness readback: {e}")

        try:
            for attr in (
                "_program",
                "_vbo",
                "_vao",
                "_fitness_fbo",
                "_fitness_tex",
                "_previous_fitness_tex",
            ):
                obj = getattr(self, attr, None)
                if obj:
                    try:
                        if attr == "_program":
                            GL.glDeleteProgram(obj)
                        elif attr == "_vbo":
                            GL.glDeleteBuffers(1, [obj])
                        elif attr == "_vao":
                            GL.glDeleteVertexArrays(1, [obj])
                        elif attr in ("_fitness_fbo",):
                            GL.glDeleteFramebuffers(1, [obj])
                        elif attr in ("_fitness_tex", "_previous_fitness_tex"):
                            GL.glDeleteTextures(1, [obj])
                    except Exception as e:
                        self._logger.warning(f"Error deleting {attr}: {e}")
                    finally:
                        setattr(self, attr, None)

            for attr in ("_vein_vbo", "_vein_vao"):
                obj = getattr(self, attr, None)
                if obj:
                    try:
                        if attr == "_vein_vbo":
                            GL.glDeleteBuffers(1, [obj])
                        elif attr == "_vein_vao":
                            GL.glDeleteVertexArrays(1, [obj])
                    except Exception as e:
                        self._logger.warning(f"Error deleting {attr}: {e}")
                    finally:
                        setattr(self, attr, None)

            # Clean up feedback buffer
            if hasattr(self, "feedback") and self.feedback:
                self.feedback.cleanup()

            # Clean up evolving post-effects (programs + ping-pong targets)
            if getattr(self, "_effects", None) is not None:
                try:
                    self._effects.cleanup()
                except Exception as e:
                    self._logger.debug(f"Error cleaning effects manager: {e}")
                self._effects = None

            # Clean up the folder visualiser (its own targets/programs/pool;
            # the shared EffectManager was already cleaned above).
            if getattr(self, "_folder_viz", None) is not None:
                try:
                    self._folder_viz.cleanup()
                except Exception as e:
                    self._logger.debug(f"Error cleaning folder visualizer: {e}")
                self._folder_viz = None

            # Clean up multi-pass pipeline
            if getattr(self, "_mp_pipeline", None) is not None:
                try:
                    self._mp_pipeline.cleanup()
                except Exception as e:
                    self._logger.debug(f"Error cleaning multi-pass pipeline: {e}")
                self._mp_pipeline = None
                self._mp_last_topology = None

            # Clean up competition renderer
            if getattr(self, "_competition_renderer", None) is not None:
                try:
                    self._competition_renderer.cleanup()
                except Exception as e:
                    self._logger.debug(f"Error cleaning competition renderer: {e}")
                self._competition_renderer = None

            # Clean up multi-resolution evolution resources.
            if getattr(self, "_multi_res", None) is not None:
                try:
                    self._multi_res.cleanup()
                except Exception as e:
                    self._logger.debug(f"Error cleaning multi-res: {e}")
                self._multi_res = None

            # Clean up spatial ecosystem GPU resources (specs 46-60).
            for prog in getattr(self, "_eco_programs", {}).values():
                try:
                    GL.glDeleteProgram(prog)
                except Exception as e:
                    logger.debug("Failed to delete eco GL program: %s", e)
            self._eco_programs = {}
            self._eco_loc = {}
            self._eco_used = {}
            self._cur_eco_prog = None
            if getattr(self, "_resource_tex", None):
                try:
                    GL.glDeleteTextures(1, [self._resource_tex])
                except Exception as e:
                    logger.debug("Failed to delete resource texture: %s", e)
                self._resource_tex = None
            if getattr(self, "_org_map_tex", None):
                try:
                    GL.glDeleteTextures(1, [self._org_map_tex])
                except Exception as e:
                    logger.debug("Failed to delete organism map atlas texture: %s", e)
                self._org_map_tex = None
                self._org_map_size = (0, 0)
            if hasattr(self, "ecosystem") and self.ecosystem is not None:
                wt = getattr(self.ecosystem, "world_tex", None)
                if wt is not None:
                    wt.cleanup()

            # Clean up cached compiled programs (clear() queues via on_evict).
            if hasattr(self, "shader_cache") and self.shader_cache:
                self.shader_cache.clear()
                self._program = None
            # The cached programs are gone, so the fingerprint that says "the
            # active program already matches this genome" no longer describes
            # anything. Without this reset _get_program would skip recompiling
            # and leave _program None until the genome mutated again.
            self._last_module_fingerprint = None
            # Program objects are going away, so their memoized locations are
            # invalid and the ids will be reused by the next context.
            self._uniform_loc_cache.clear()

            # Drain the shared FBO/texture pool (releases every pooled GL object).
            if getattr(self, "_fbo_pool", None) is not None:
                try:
                    self._fbo_pool.drain()
                except Exception as e:
                    self._logger.debug(f"FBO pool drain failed: {e}")
                self._fbo_pool = None

            # Replay ring is pure CPU-side bytes; drop it.
            if getattr(self, "_replay", None) is not None:
                try:
                    self._replay.clear()
                except (AttributeError, TypeError, ValueError) as e:
                    self._logger.debug("Replay ring clear failed: %s", e)
                self._replay = None

            # Flush any queued deletions inside the current GL context.
            while self.pending_delete:
                prog = self.pending_delete.popleft()
                try:
                    if GL.glIsProgram(prog):
                        GL.glDeleteProgram(prog)
                except Exception as e:
                    self._logger.debug(f"Failed to delete queued GL program {prog}: {e}")

            self._logger.info("GL cleanup complete")

        except Exception as e:
            self._logger.error(f"Unrealize error: {e}")

        self.gl_ready = False

    def _create_fullscreen_quad(self) -> None:
        """Create a fullscreen quad for rendering.

        Raises:
            RuntimeError: If numpy is not available or GL operations fail
        """
        if np is None:
            raise RuntimeError("EvoRenderer requires numpy for the fullscreen quad")

        data = np.asarray(FULLSCREEN_QUAD_VERTICES, dtype=np.float32)

        try:
            self._vao = GL.glGenVertexArrays(1)
            self._vbo = GL.glGenBuffers(1)

            GL.glBindVertexArray(self._vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, len(data) * 4, data, GL.GL_STATIC_DRAW)

            # Vertex layout: position (2 floats) + texcoord (2 floats)
            stride = QUAD_STRIDE_BYTES
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(8))

            GL.glBindVertexArray(0)
            self._logger.debug(f"Fullscreen quad created: VAO={self._vao}, VBO={self._vbo}")

        except Exception as e:
            self._logger.error(f"Failed to create fullscreen quad: {e}")
            raise

    def _compile_program(self, vs_source: str, fs_source: str, cache_key: str = "") -> int:
        """Compile and link a complete GL program.

        Args:
            vs_source: Vertex shader source code
            fs_source: Fragment shader source code
            cache_key: If supplied and the disk cache is available, the compiled
                binary is stored and future restores skip the compile step.

        Returns:
            Program object ID

        Raises:
            RuntimeError: If compilation or linking fails
        """
        # Shader binary cache: try restoring a previously extracted binary
        # before falling back to full recompilation.  This is a fast path on
        # drivers that support ARB_get_program_binary.
        if cache_key and self.disk_cache is not None:
            try:
                prog = GL.glCreateProgram()
                if self.disk_cache.try_load_binary(prog, cache_key):
                    return int(prog)
                # A failed glProgramBinary likely left the program in a dead
                # state; discard it and fall through to the normal path.
                GL.glDeleteProgram(prog)
            except (GL.error.GLError, TypeError, ValueError, AttributeError) as e:
                self._logger.debug("Shader binary cache load failed: %s", e)

        try:
            vs = self._compile_shader(GL.GL_VERTEX_SHADER, vs_source)
            fs = self._compile_shader(GL.GL_FRAGMENT_SHADER, fs_source)

            prog = GL.glCreateProgram()
            GL.glAttachShader(prog, vs)
            GL.glAttachShader(prog, fs)
            GL.glLinkProgram(prog)

            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)

            if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
                err = GL.glGetProgramInfoLog(prog)
                if isinstance(err, bytes):
                    err = err.decode("utf-8", errors="ignore")
                GL.glDeleteProgram(prog)
                raise RuntimeError(f"Program link failed: {err}")

            # Persist the compiled binary for next session / same process.
            if cache_key and self.disk_cache is not None:
                try:
                    binary = self.disk_cache.extract_binary(prog)
                    if binary:
                        self.disk_cache.put_binary(cache_key, binary[1], binary[0])
                except (GL.error.GLError, TypeError, ValueError, AttributeError) as e:
                    self._logger.debug("Shader binary cache store failed: %s", e)

            return int(prog)

        except Exception as e:
            try:
                GL.glDeleteProgram(prog)
            except (GL.error.GLError, TypeError, ValueError):
                pass
            self._logger.error(f"Program compilation failed: {e}")
            raise

    def _compile_cached(
        self, vs_source: str, fs_source: str, genome: Any = None, lod: dict[str, int] | None = None
    ) -> int:
        """Compile with both source- and binary-level disk caching.

        Derives a stable cache key from ``genome.shader_signature()`` (or the
        genome's ``to_dict()`` if no signature method exists) layers it with
        the LOD tier so each quality variant is independently cached.
        """
        if self.disk_cache is not None and genome is not None:
            try:
                sig_factory = getattr(genome, "shader_signature", None)
                sig = sig_factory() if sig_factory else getattr(genome, "to_dict", dict)()
                driver_id = self.disk_cache.driver._id if self.disk_cache.driver else ""
                fs_hash = _hash(sig)
                src_key = self.disk_cache._source_key(
                    fs_hash,
                    GL.GL_FRAGMENT_SHADER,
                    specialization=str(lod.get("tier", "")) if lod else "",
                    driver_id=driver_id,
                )
                # Try binary first (fastest path)
                prog = GL.glCreateProgram()
                if self.disk_cache.try_load_binary(prog, src_key):
                    return int(prog)
                GL.glDeleteProgram(prog)
                # Fall back to normal compilation, keyed for binary storage.
                return self._compile_program(vs_source, fs_source, cache_key=src_key)
            except (GL.error.GLError, TypeError, ValueError, AttributeError) as e:
                self._logger.debug("Shader binary cache operation failed: %s", e)
        return self._compile_program(vs_source, fs_source)

    def _compile_shader(self, shader_type: int, source: str) -> int:
        """Compile a single shader.

        Args:
            shader_type: GL.GL_VERTEX_SHADER or GL.GL_FRAGMENT_SHADER
            source: Shader source code

        Returns:
            Shader object ID

        Raises:
            RuntimeError: If compilation fails
        """
        try:
            sh = GL.glCreateShader(shader_type)
            GL.glShaderSource(sh, source)
            GL.glCompileShader(sh)

            if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
                err = GL.glGetShaderInfoLog(sh)
                if isinstance(err, bytes):
                    err = err.decode("utf-8", errors="ignore")
                raise RuntimeError(f"Shader compile failed: {err}")

            return int(sh)

        except Exception as e:
            self._logger.error(f"Shader compilation error: {e}")
            raise

    def _recompile_fragment_shader(self, new_frag_source: str, genome: Any = None) -> None:
        """Recompile fragment shader and refresh uniform locations.

        Args:
            new_frag_source: New fragment shader source code.
            genome: Optional genome to associate with the compiled program in
                the shader cache. When provided the program is cached so future
                scoring of an identical genome reuses it.

        Raises:
            RuntimeError: If compilation fails
        """
        try:
            new_prog = self._compile_program(self.VERTEX_SHADER, new_frag_source)

            # Never delete a program still referenced by the cache (it may be
            # bound by another organism's scored program).
            if self._program is not None and self._program not in self.shader_cache.programs():
                GL.glDeleteProgram(self._program)

            self._set_program(new_prog)

            if genome is not None:
                self.shader_cache.store(self.shader_cache.key(genome), new_prog)

            self._logger.debug(f"Fragment shader recompiled: program={self._program}")

        except Exception as e:
            self._logger.error(f"Fragment shader recompilation failed: {e}")
            raise

    def _get_program(self, genome: Any, allow_async: bool = True) -> int | None:
        """Return a compiled program for ``genome``, compiling + caching on
        first use. Falls back to the current program if compilation fails.

        When async compilation is enabled and ``allow_async`` is True the
        expensive GLSL source generation is queued on a worker thread instead of
        run inline: this call returns immediately with the best program already
        available (a previous program or the current one) so the render loop
        never stalls waiting on a mutation. The freshly compiled program is
        swapped in a later frame by :meth:`_drain_async_shaders`.

        Set ``allow_async=False`` when the caller needs this genome's *own*
        program right now (e.g. fitness scoring), so it compiles synchronously
        instead of returning some other genome's currently-bound program.
        """
        # Module fingerprint: if the currently-bound program's modules match
        # the requested genome, reuse it without any cache lookup or GL work.
        new_fp = self._compute_module_fingerprint(genome)
        if (
            self._program is not None
            and self._last_module_fingerprint == new_fp
            and new_fp is not None
        ):
            return self._program
        self._last_module_fingerprint = new_fp

        # Adaptive renderer complexity: the active LOD tier bakes loop bounds
        # (ray steps / fractal layers / feedback history) into the shader, so
        # the cache key must fold in the tier -- otherwise a low-quality variant
        # would masquerade as the full-fidelity one (or vice-versa).
        lod = self._current_lod()
        base_key = self.shader_cache.key(genome)
        if lod is not None:
            # Combine base_key with LOD parameters into a stable integer key.
            # Use a deterministic hash that works across Python runs (no PYTHONHASHSEED).
            lod_tuple = tuple(sorted(lod.items()))
            # Use a simple but stable combination: hash the tuple of (base_key, lod_tuple)
            # with a fixed algorithm that doesn't depend on PYTHONHASHSEED
            combined = (base_key, lod_tuple)
            key = int(hashlib.md5(str(combined).encode()).hexdigest(), 16) & 0x7FFFFFFF
        else:
            key = base_key
        program = self.shader_cache.get(key)
        if program is not None:
            return program

        # Shader disk cache: check if the generated source for this genome +
        # LOD tier is already persisted on disk.  A hit skips both the CPU-
        # heavy source generation AND the async round-trip.
        if self.disk_cache is not None and genome is not None:
            try:
                sig_factory = getattr(genome, "shader_signature", None)
                sig = sig_factory() if sig_factory else getattr(genome, "to_dict", dict)()
                driver_id = self.disk_cache.driver._id if self.disk_cache.driver else ""
                sig_hash = _hash(sig)
                src_key = self.disk_cache._source_key(
                    sig_hash,
                    GL.GL_FRAGMENT_SHADER,
                    specialization=str(lod.get("tier", "")) if lod else "",
                    driver_id=driver_id,
                )
                cached_src = self.disk_cache.get_source(src_key)
                if cached_src is not None:
                    try:
                        prog = self._compile_cached(
                            self.VERTEX_SHADER, cached_src, genome=genome, lod=lod
                        )
                        self.shader_cache.store(key, prog)
                        return prog
                    except (
                        GL.error.GLError,
                        TypeError,
                        ValueError,
                        AttributeError,
                        RuntimeError,
                    ) as e:
                        self._logger.debug("Disk cache shader compile failed: %s", e)
            except (GL.error.GLError, TypeError, ValueError, AttributeError) as e:
                self._logger.debug("Disk cache source lookup failed: %s", e)

        compiler = self.async_shader_compiler
        if compiler is not None and allow_async:
            # Queue the source build off-thread (deduplicated by key) and return
            # a program to keep drawing with in the meantime. Only snapshot the
            # genome when the key is not already queued -- request() would drop a
            # duplicate anyway, so cloning it every frame is wasted work. Snapshot
            # so the worker reads an immutable copy: the live genome may be
            # mutated in-place by the evolution loop while the worker runs.
            if not compiler.is_pending(key):
                snapshot = self._snapshot_genome(genome)
                compiler.request(
                    key, lambda g=snapshot, lod_level=lod: self._gen_frag_source(g, lod_level)
                )
            # Keep drawing the currently-bound program until the fresh one is
            # uploaded (next frame at the latest, by _drain_async_shaders).
            # If no program is bound yet (first frame / gl_ready just flipped),
            # fall through to synchronous compilation so we never return None
            # and leave the screen black.
            if self._program is not None:
                return self._program

        # Synchronous compile (async disabled, caller requires this genome's
        # own program immediately, or the async path had no existing program).
        try:
            frag = self._gen_frag_source(genome, lod)
            program = self._compile_cached(self.VERTEX_SHADER, frag, genome=genome, lod=lod)
        except Exception as e:
            self._logger.error(
                f"Shader compile failed for genome {getattr(genome, 'lineage', '?')}: {e}"
            )
            # Last resort: a simple passthrough shader so the screen is never
            # black even when the evolutionary genome fails to compile.
            fallback_prog = self._program
            if fallback_prog is None:
                try:
                    fallback_frag = (
                        "#version 330 core\nin vec2 v_uv;\nout vec4 fragColor;\n"
                        "uniform float u_time;\nvoid main(){\n"
                        "    vec3 c = 0.5 + 0.5*cos(u_time + v_uv.xyx + vec3(0,2,4));\n"
                        "    fragColor = vec4(c, 1.0);\n"
                        "}"
                    )
                    fallback_prog = self._compile_program(self.VERTEX_SHADER, fallback_frag)
                except (GL.error.GLError, TypeError, ValueError, RuntimeError) as e:
                    fallback_prog = None
                    self._logger.critical("Even fallback shader failed to compile: %s", e)
            return fallback_prog
        self.shader_cache.store(key, program)
        return program

    def _current_lod(self) -> dict[str, int] | None:
        """Snapshot the active adaptive-complexity LOD tier as loop bounds.

        Returns ``None`` when adaptive complexity is disabled so the renderer
        falls back to full-fidelity behaviour (identical to the pre-adaptive
        code path). Otherwise returns the ray-march / fractal-layer counts
        derived from the quality controller's current tier.

        When ``self._force_full_lod`` is set (fitness scoring), the maximum-tier
        bounds are returned instead of the live tier so evolution is always
        scored on a consistent, full-fidelity image regardless of the current
        frame-time budget.
        """
        ctrl = getattr(self, "quality_controller", None)
        if ctrl is None:
            return None
        cfg = self._config
        if getattr(self, "_force_full_lod", False):
            return {
                "tier": int(ctrl.LOD_TIERS - 1),
                "raymarch_steps": int(cfg.perf_max_raymarch_steps),
                "fractal_layers": int(cfg.perf_max_fractal_layers),
                "particle_count": int(cfg.perf_max_particle_count),
                "blur_taps": int(cfg.perf_max_blur_taps),
                "lighting_samples": int(cfg.perf_max_lighting_samples),
                "shadow_samples": int(cfg.perf_max_shadow_samples),
                "bloom_passes": int(cfg.perf_max_bloom_passes),
                "dof_samples": int(cfg.perf_max_dof_samples),
                "feedback_history": int(cfg.perf_max_feedback_history),
            }
        return {
            "tier": int(ctrl.lod_tier),
            "raymarch_steps": int(ctrl.raymarch_steps),
            "fractal_layers": int(ctrl.fractal_layers),
            "particle_count": int(ctrl.particle_count),
            "blur_taps": int(ctrl.blur_taps),
            "lighting_samples": int(ctrl.lighting_samples),
            "shadow_samples": int(ctrl.shadow_samples),
            "bloom_passes": int(ctrl.bloom_passes),
            "dof_samples": int(ctrl.dof_samples),
            "feedback_history": int(ctrl.feedback_history),
        }

    def _compute_module_fingerprint(self, genome: Any) -> str | None:
        """Compute a compact fingerprint from the genome's module selections.

        This is a no-GL fast path: it hashes the genome's module identity
        (field type, warp, palette, feedback, post list, and all params) to
        decide whether the currently-bound program already matches.  If so,
        the renderer skips source generation, compilation and linking.
        """
        try:
            sig_factory = getattr(genome, "shader_signature", None)
            if sig_factory is not None:
                sig = sig_factory()
            else:
                sig = {}
            material = json.dumps(sig, sort_keys=True, default=str).encode()
            fw = hashlib.sha256(material).hexdigest()[:16]
            return fw
        except (TypeError, ValueError, AttributeError, json.JSONDecodeError) as e:
            self._logger.debug("Module fingerprint failed: %s", e)
            return None

    def _snapshot_genome(self, genome: Any) -> Any:
        """Return a copy of ``genome`` for the async worker to read.

        Source generation only touches the phenotype (params / graph / modules),
        so a clone decouples the worker from any in-place mutation the evolution
        loop performs on the live genome while the worker runs. Falls back to
        the original genome if it cannot be copied.

        Returns:
            A copy of the genome, or the original if copying fails
        """
        try:
            clone = getattr(genome, "clone", None)
            if callable(clone):
                return clone()
        except (AttributeError, TypeError, ValueError) as e:
            self._logger.debug("Genome clone failed: %s", e)
        try:
            return copy.deepcopy(genome)
        except (TypeError, ValueError):
            return genome

    def _drain_async_shaders(self) -> None:
        """Upload any shaders whose source finished generating off-thread.

        Runs once per frame on the render thread (which owns the GL context).
        The GL compile/link is fast; the slow source generation already happened
        on the worker thread. Newly built programs are stored in the shader
        cache so the next lookup for that genome swaps to the fresh program.
        """
        compiler = self.async_shader_compiler
        if compiler is None:
            return

        def upload_fn(src: str) -> int:
            try:
                prog = self._compile_program(self.VERTEX_SHADER, src)
                if self.disk_cache is not None and prog:
                    try:
                        src_hash = _hash(src)
                        driver_id = self.disk_cache.driver._id if self.disk_cache.driver else ""
                        src_key = self.disk_cache._source_key(
                            src_hash, GL.GL_FRAGMENT_SHADER, driver_id=driver_id
                        )
                        self.disk_cache.put_source(
                            src_key,
                            src,
                            shader_type=GL.GL_FRAGMENT_SHADER,
                        )
                        binary = self.disk_cache.extract_binary(prog)
                        if binary:
                            bin_key = self.disk_cache._binary_key(src_hash, driver_id)
                            self.disk_cache.put_binary(bin_key, binary[1], binary[0])
                    except (GL.error.GLError, TypeError, ValueError, AttributeError) as e:
                        self._logger.debug("Disk cache store in async upload failed: %s", e)
                return prog
            except (GL.error.GLError, TypeError, ValueError, AttributeError, RuntimeError):
                return 0

        try:
            uploads = compiler.poll_ready(
                upload_fn,
                max_uploads=int(getattr(self._config, "async_shader_uploads_per_frame", 4)),
            )
        except Exception as e:
            self._logger.debug(f"Async shader drain failed: {e}")
            return
        for key, program in uploads:
            # If a program for this key already landed (e.g. a synchronous
            # score-time compile happened while the async build was in flight),
            # keep the existing one and drop this freshly-built duplicate rather
            # than churning the cache (allocate + immediate evict). Deletion is
            # deferred to the pending-delete queue so it runs in a valid context.
            existing = self.shader_cache.get(key)
            if existing is not None and existing != program:
                self.pending_delete.append(int(program))
                continue
            self.shader_cache.store(key, program)

    def _uniform_locations(self, program: int) -> dict[str, int]:
        """Return every uniform location this renderer uses for ``program``.

        ``glGetUniformLocation`` is a driver round-trip, and this renderer
        asks for the same ~60 names on every program switch (mutation, LOD
        change, fitness scoring). Locations are stable for the lifetime of a
        program object, so they are memoized per program id. Entries are
        dropped in :meth:`_delete_program` because GL recycles program names:
        a stale entry would otherwise hand back locations belonging to the
        program that inherits the id next.
        """
        cached = self._uniform_loc_cache.get(program)
        if cached is None:
            cached = {name: GL.glGetUniformLocation(program, name) for name in self.UNIFORM_NAMES}
            # Programs are bounded by the shader cache, but the cache can be
            # cleared and rebuilt repeatedly; keep a hard ceiling so a long
            # session cannot grow this dict without bound.
            if len(self._uniform_loc_cache) >= self.UNIFORM_LOC_CACHE_MAX:
                self._uniform_loc_cache.clear()
            self._uniform_loc_cache[program] = cached
        return cached

    def _set_program(self, program: int | None) -> None:
        """Bind ``program`` as the active program and refresh uniform locations.

        Programs are cached so the GL program object is reused; only the
        uniform-location lookups need refreshing when switching programs. The
        active program is "protected" so the cache never evicts/deletes it
        while it is bound.
        """
        if program is None or program == self._program:
            return
        self._program = int(program)
        self.shader_cache.protect(self._program)
        locs = self._uniform_locations(self._program)

        self._loc_time = locs["u_time"]
        self._loc_res = locs["u_resolution"]
        # Adaptive renderer complexity: runtime quality scalar (0..1). Present
        # in the fallback ShaderFactory shaders that scale their loops live.
        self._loc_quality = locs["u_quality"]
        # Adaptive renderer complexity: explicit loop bounds for all quality knobs.
        self._loc_particle_count = locs["u_particle_count"]
        self._loc_blur_taps = locs["u_blur_taps"]
        self._loc_lighting_samples = locs["u_lighting_samples"]
        self._loc_shadow_samples = locs["u_shadow_samples"]
        self._loc_bloom_passes = locs["u_bloom_passes"]
        self._loc_dof_samples = locs["u_dof_samples"]
        self._loc_feedback_history = locs["u_feedback_history"]

        # ModularShaderFactory uniforms
        self._loc_bass = locs["bass"]
        self._loc_mid = locs["mid"]
        self._loc_treble = locs["treble"]
        self._loc_previous_frame = locs["previous_frame"]

        # Legacy uniforms (may not exist in modular shaders)
        self._loc_audio = locs["u_audio"]
        self._loc_prev = locs["u_prev_frame"]
        self._loc_fb_alpha = locs["u_feedback_alpha"]
        self._loc_brightness = locs["u_brightness"]

        # Camera organism uniforms (dynamic per-frame camera state)
        self._camera_uniform_locs.clear()
        for name in self.CAMERA_UNIFORM_NAMES:
            loc = locs[name]
            if loc >= 0:
                self._camera_uniform_locs[name] = loc

        # Music DNA uniforms (optional)
        self._loc_md_bpm = locs["md_bpm"]
        self._loc_md_energy = locs["md_energy"]
        self._loc_md_danceability = locs["md_danceability"]
        self._loc_md_beat_strength = locs["md_beat_strength"]
        self._loc_md_spectral_centroid = locs["md_spectral_centroid"]
        self._loc_md_harmonic_complexity = locs["md_harmonic_complexity"]
        self._loc_md_dynamic_range = locs["md_dynamic_range"]
        self._loc_md_key = locs["md_key"]
        self._loc_md_mood = locs["md_mood"]

        # Animation uniforms (Milkdrop-style per-frame animation)
        self._anim_uniform_locs = {}
        anim_names = [
            "anim_zoom",
            "anim_rot",
            "anim_cx",
            "anim_cy",
            "anim_dx",
            "anim_dy",
            "anim_sx",
            "anim_sy",
            "anim_warp",
            "anim_wave_r",
            "anim_wave_g",
            "anim_wave_b",
            "anim_wave_x",
            "anim_wave_y",
            "anim_ob_size",
            "anim_ob_r",
            "anim_ob_g",
            "anim_ob_b",
            "anim_ob_a",
            "anim_ib_size",
            "anim_ib_r",
            "anim_ib_g",
            "anim_ib_b",
            "anim_ib_a",
            "anim_mv_x",
            "anim_mv_y",
            "anim_mv_l",
            "anim_mv_r",
            "anim_mv_g",
            "anim_mv_b",
            "anim_mv_a",
        ]
        for name in anim_names:
            loc = locs.get(name, -1)
            if loc >= 0:
                self._anim_uniform_locs[name] = loc

    def _delete_program(self, program: int | None) -> None:
        """Queue a compiled GL program for deletion during the next render.

        Used as the :class:`ShaderCache` ``on_evict`` callback. Deferring
        deletion avoids ``GLError: invalid operation`` when the cache evicts
        outside an active GTK GLArea context.
        """
        if not program:
            return
        self.pending_delete.append(int(program))
        # Drop memoized locations: GL recycles program names, so keeping them
        # would return locations that belong to whatever program is created
        # with this id next.
        self._uniform_loc_cache.pop(int(program), None)

    @staticmethod
    def _audio_response(v: float) -> float:
        """Perceptual lift for shader uniforms: expands quiet levels so
        visuals stay alive between transients, gentle roll-off up top."""
        v = max(0.0, min(1.5, float(v)))
        return v**0.85 if v > 0.0 else 0.0

    def _apply_render_uniforms(
        self,
        *,
        bass: float,
        mid: float,
        treb: float,
        t: float,
        device_w: int,
        device_h: int,
        feedback_res: int,
        previous_frame_tex: int,
        beat: float = 0.0,
        resolution_only: bool = False,
    ) -> None:
        """Apply the uniforms shared by the feedback and screen passes.

        ``resolution_only`` re-uploads just ``u_resolution`` for the second
        pass of a frame. The feedback pass and the screen pass draw the same
        program with the same audio/time/quality state and differ only in
        target size, so the screen pass does not need the other ~20 uploads.
        """
        if self._program is None:
            return

        if self._loc_res >= 0:
            GL.glUniform2f(self._loc_res, float(device_w), float(device_h))

        if resolution_only:
            return

        if self._loc_time >= 0:
            GL.glUniform1f(self._loc_time, float(t))

        ctrl = getattr(self, "quality_controller", None)

        # Adaptive renderer complexity: upload runtime quality scalar so the
        # fallback ShaderFactory shaders scale their loops live. Modular shaders
        # bake the tier at compile time instead and simply ignore these uniforms.
        if self._loc_quality >= 0:
            q = float(getattr(ctrl, "quality", 1.0)) if ctrl is not None else 1.0
            GL.glUniform1f(self._loc_quality, max(0.0, min(1.0, q)))

        if self._loc_particle_count >= 0:
            GL.glUniform1i(self._loc_particle_count, int(getattr(ctrl, "particle_count", 8)))
        if self._loc_blur_taps >= 0:
            GL.glUniform1i(self._loc_blur_taps, int(getattr(ctrl, "blur_taps", 4)))
        if self._loc_lighting_samples >= 0:
            GL.glUniform1i(self._loc_lighting_samples, int(getattr(ctrl, "lighting_samples", 8)))
        if self._loc_shadow_samples >= 0:
            GL.glUniform1i(self._loc_shadow_samples, int(getattr(ctrl, "shadow_samples", 4)))
        if self._loc_bloom_passes >= 0:
            GL.glUniform1i(self._loc_bloom_passes, int(getattr(ctrl, "bloom_passes", 1)))
        if self._loc_dof_samples >= 0:
            GL.glUniform1i(self._loc_dof_samples, int(getattr(ctrl, "dof_samples", 8)))
        if self._loc_feedback_history >= 0:
            GL.glUniform1i(self._loc_feedback_history, int(getattr(ctrl, "feedback_history", 0)))

        if self._loc_bass >= 0:
            GL.glUniform1f(self._loc_bass, self._audio_response(bass))
        if self._loc_mid >= 0:
            GL.glUniform1f(self._loc_mid, self._audio_response(mid))
        if self._loc_treble >= 0:
            GL.glUniform1f(self._loc_treble, self._audio_response(treb))
        if self._loc_audio >= 0:
            GL.glUniform4f(
                self._loc_audio,
                self._audio_response(bass),
                self._audio_response(mid),
                self._audio_response(treb),
                self._audio_response(beat),
            )

        if self._loc_fb_alpha >= 0:
            GL.glUniform1f(self._loc_fb_alpha, float(self._config.feedback_alpha))
        if self._loc_brightness >= 0:
            GL.glUniform1f(self._loc_brightness, float(self._config.brightness_base))

        if self._loc_previous_frame >= 0 or self._loc_prev >= 0:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, previous_frame_tex)
            if self._loc_previous_frame >= 0:
                GL.glUniform1i(self._loc_previous_frame, 0)
            if self._loc_prev >= 0:
                GL.glUniform1i(self._loc_prev, 0)

        # Music DNA uniforms
        self._apply_music_dna_uniforms()

        # Camera organism uniforms (dynamic per-frame camera state)
        self._apply_camera_uniforms()

        # Animation uniforms (Milkdrop-style per-frame animation)
        self._apply_animation_uniforms()

    def _apply_animation_uniforms(self) -> None:
        """Apply Milkdrop-style animation uniforms from genome."""
        if not self._anim_uniform_locs or not hasattr(self.genome, "get_animation_uniforms"):
            return
        anim_values = self.genome.get_animation_uniforms()
        for name, loc in self._anim_uniform_locs.items():
            if name in anim_values:
                GL.glUniform1f(loc, float(anim_values[name]))

    def _apply_camera_uniforms(self) -> None:
        """Upload the dominant camera organism's state as shader uniforms.

        When an emergent camera population is active, the best camera's
        genome and current behavioural state drive the viewport.  When the
        camera system is off, the genome's own camera genes are used as
        static defaults.
        """
        if not self._camera_uniform_locs:
            return
        best = None
        cam_genome = None
        if self.camera_population is not None:
            try:
                best = self.camera_population.best()
                cam_genome = getattr(best, "genome", None)
            except (AttributeError, TypeError, ValueError) as e:
                self._logger.debug("Camera population best() failed: %s", e)
        g = self.genome
        if hasattr(g, "camera") and g.camera is not None:
            cg = g.camera

            def gv(key, default):
                try:
                    return float(getattr(cg, key, default))
                except (AttributeError, TypeError, ValueError):
                    return float(default)
        elif hasattr(g, "params") and isinstance(g.params, dict):
            cparams = g.params

            def gv(key, default):
                try:
                    return float(cparams.get(key, default))
                except (TypeError, ValueError):
                    return float(default)
        else:

            def gv(key, default):
                return float(default)

        if best is not None:
            vals = {
                "k_camera_speed": getattr(cam_genome, "orbit_speed", gv("camera_speed", 1.0)),
                "k_camera_orbit": best.orbit * 6.28318,
                "k_camera_depth": 3.0 - best.zoom * 2.0,
                "k_camera_roll": best.roll,
                "k_camera_fov": best.fov,
                "k_depth_scale": gv("depth_scale", 1.0),
                "k_ray_steps": int(gv("ray_steps", 32)),
                "k_fog_density": gv("fog_density", 1.0),
                "k_dimension": gv("dimension", 0.0),
                "k_camera_focus": best.focus,
                "k_camera_aperture": best.aperture,
                "k_camera_exposure": best.exposure,
                "k_camera_autoexp": getattr(
                    cam_genome, "auto_exposure", gv("camera_autoexp", 0.4)
                ),
                "k_camera_shake": best.shake,
                "k_camera_shake_freq": getattr(
                    cam_genome, "shake_frequency", gv("camera_shake_freq", 0.5)
                ),
                "k_camera_tracking": best.tracking,
                "k_camera_framing": best.framing,
                "k_camera_headroom": best.headroom,
                "k_camera_school": float(
                    getattr(cam_genome, "cinemat_school", gv("camera_school", 0)) % 10
                ),
                "k_camera_shot_length": getattr(
                    cam_genome, "shot_length", gv("camera_shot_length", 1.0)
                ),
                "k_camera_transition": getattr(
                    cam_genome, "transition", gv("camera_transition", 0.3)
                ),
                "k_camera_crane": getattr(cam_genome, "crane", gv("camera_crane", 0.0)),
                "k_camera_parallax": getattr(cam_genome, "parallax", gv("camera_parallax", 0.3)),
                "k_camera_drift": getattr(cam_genome, "drift", gv("camera_drift", 0.0)),
                "k_camera_focus_pull": getattr(
                    cam_genome, "focus_pull", gv("camera_focus_pull", 0.3)
                ),
                "k_camera_handheld": getattr(cam_genome, "handheld", gv("camera_handheld", 0.2)),
                "k_camera_dolly": getattr(cam_genome, "dolly", gv("camera_dolly", 0.5)),
                "k_camera_pan": best.pan,
                "k_camera_tilt": best.tilt,
            }
        else:
            vals = {
                "k_camera_speed": gv("camera_speed", 1.0),
                "k_camera_orbit": gv("camera_orbit", 1.0),
                "k_camera_depth": gv("camera_depth", 3.0),
                "k_camera_roll": gv("camera_roll", 0.0),
                "k_camera_fov": gv("camera_fov", 1.5),
                "k_depth_scale": gv("depth_scale", 1.0),
                "k_ray_steps": int(gv("ray_steps", 32)),
                "k_fog_density": gv("fog_density", 1.0),
                "k_dimension": gv("dimension", 0.0),
                "k_camera_focus": gv("camera_focus", 0.5),
                "k_camera_aperture": gv("camera_aperture", 0.2),
                "k_camera_exposure": gv("camera_exposure", 0.5),
                "k_camera_autoexp": gv("camera_autoexp", 0.4),
                "k_camera_shake": gv("camera_shake", 0.15),
                "k_camera_shake_freq": gv("camera_shake_freq", 0.5),
                "k_camera_tracking": gv("camera_tracking", 0.5),
                "k_camera_framing": gv("camera_framing", 0.5),
                "k_camera_headroom": gv("camera_headroom", 0.5),
                "k_camera_school": float(int(gv("camera_school", 0)) % 10),
                "k_camera_shot_length": gv("camera_shot_length", 1.0),
                "k_camera_transition": gv("camera_transition", 0.3),
                "k_camera_crane": gv("camera_crane", 0.0),
                "k_camera_parallax": gv("camera_parallax", 0.3),
                "k_camera_drift": gv("camera_drift", 0.0),
                "k_camera_focus_pull": gv("camera_focus_pull", 0.3),
                "k_camera_handheld": gv("camera_handheld", 0.2),
                "k_camera_dolly": gv("camera_dolly", 0.5),
                "k_camera_pan": gv("camera_pan", 0.3),
                "k_camera_tilt": gv("camera_tilt", 0.3),
            }

        for name, val in vals.items():
            loc = self._camera_uniform_locs.get(name)
            if loc is None:
                continue
            if name == "k_ray_steps" or name == "k_camera_school":
                GL.glUniform1i(loc, int(val))
            else:
                GL.glUniform1f(loc, float(val))

    def _apply_music_dna_uniforms(self) -> None:
        """Apply optional MusicDNA uniforms to the current shader program."""
        if (
            self._program is None
            or not hasattr(self, "_current_music_dna")
            or self._current_music_dna is None
        ):
            return
        dna = self._current_music_dna
        if self._loc_md_bpm >= 0:
            GL.glUniform1f(self._loc_md_bpm, float(dna.bpm))
        if self._loc_md_energy >= 0:
            GL.glUniform1f(self._loc_md_energy, float(dna.energy))
        if self._loc_md_danceability >= 0:
            GL.glUniform1f(self._loc_md_danceability, float(dna.danceability))
        if self._loc_md_beat_strength >= 0:
            GL.glUniform1f(self._loc_md_beat_strength, float(dna.beat_strength))
        if self._loc_md_spectral_centroid >= 0:
            GL.glUniform1f(self._loc_md_spectral_centroid, float(dna.spectral_centroid))
        if self._loc_md_harmonic_complexity >= 0:
            GL.glUniform1f(self._loc_md_harmonic_complexity, float(dna.harmonic_complexity))

    def _mp_topology_key(self) -> str:
        """Return a stable topology key for the active PipelineGenome so the
        multi-pass pipeline is only recompiled when the pass graph changes."""
        try:
            parts = []
            for p in self.genome.passes:
                parts.append(
                    (
                        p.pass_type,
                        p.blend_mode,
                        p.feedback_route,
                        p.modules.field_type,
                        tuple(p.modules.post),
                    )
                )
            return str(tuple(parts))
        except (AttributeError, TypeError, ValueError) as e:
            self._logger.debug("MP topology key failed: %s", e)
            return ""

    def _ensure_mp_pipeline(self) -> None:
        """Rebuild the multi-pass pipeline if the genome topology changed."""
        from evo.pipeline_genome import PipelineGenome

        if not isinstance(self.genome, PipelineGenome):
            self._mp_pipeline = None
            return
        key = self._mp_topology_key()
        if getattr(self, "_mp_last_topology", None) == key:
            return
        self._mp_last_topology = key
        try:
            if self._mp_pipeline is None:
                self._mp_pipeline = MultiPassPipeline(
                    factory=self.shader_factory,
                    pool=self._fbo_pool,
                    disk_cache=self.disk_cache,
                )
            self._mp_pipeline.rebuild(self.genome)
        except Exception as exc:
            self._logger.debug(f"Multi-pass pipeline ensure failed: {exc}")
            self._mp_pipeline = None

    def _on_glarea_resize(self, area: Gtk.GLArea, width: int, height: int) -> None:
        scale = self.get_scale_factor()
        self._glarea_w = int(width)
        self._glarea_h = int(height)
        self._logger.debug(
            "ALLOC-CHAIN: glarea_resize surface=%dx%d alloc=%dx%d scale=%d",
            self._glarea_w,
            self._glarea_h,
            area.get_width(),
            area.get_height(),
            scale,
        )

    def _update_ecosystem_memory(self) -> None:
        """Update ecosystem memory and world evolution if enabled."""
        if not self._config_get("enable_ecosystem_memory", False, bool):
            return
        try:
            self._maybe_store_music_pattern()
            self._maybe_evolve_world()
            if self._check_dream_mode():
                self.dream_once()
            self._maybe_revive_species()
        except (RuntimeError, AttributeError, ValueError) as e:
            self._logger.debug("Ecosystem memory update failed: %s", e)
        except Exception as e:
            self._logger.warning("Unexpected error in ecosystem memory update: %s", e)

    def _update_auto_toggles_safe(self, t: float, beat: float) -> None:
        """Update automatic view toggles."""
        try:
            self._update_auto_toggles(t, beat)
        except (RuntimeError, AttributeError, ValueError) as e:
            self._logger.debug("Auto toggles failed: %s", e)
        except Exception as e:
            self._logger.warning("Unexpected error in auto toggles: %s", e)

    def _update_memory_watchdog_safe(self) -> None:
        """Run memory watchdog for OOM prevention."""
        try:
            self._memory_watchdog()
        except (RuntimeError, OSError, MemoryError) as e:
            self._logger.debug("Memory watchdog failed: %s", e)
        except Exception as e:
            self._logger.warning("Unexpected error in memory watchdog: %s", e)

    def _pre_render_update(self, dt: float) -> tuple[float, float, float, float, float]:
        """Run simulation/evolution updates shared by every presentation mode.

        This must run before choosing the rendering path (spatial, main,
        competition, ...) so the spatial view cannot bypass steady-state
        evolution, beat evolution, auto toggles, fitness cadence, or the
        memory watchdog. Returns ``(bass, mid, treb, beat, t)`` for the
        render passes.

        The core evolution calls (``ecosystem.step``,
        ``_sync_camera_organisms``, ``_evolve_shader_graph_genome``,
        ``_steady_state_evolve``, ``_maybe_beat_evolution``) are invoked
        directly here rather than through thin wrappers so the full
        evolution loop is visible at the call site and cannot be silently
        bypassed by a branch that skips ``_pre_render_update``.
        """
        with self._perf_timer("pre_render_update", log_threshold_ms=10.0):
            # Spatial ecosystem step — always runs, even when the spatial
            # renderer is off, so the world simulation stays current.
            if self.ecosystem is not None:
                try:
                    self.ecosystem.step(dt, self.audio.audio, self._current_music_dna)
                except (RuntimeError, AttributeError, ValueError) as e:
                    self._logger.debug("Ecosystem step failed: %s", e)
                except Exception as e:
                    self._logger.warning("Unexpected error in ecosystem step: %s", e)
                # Champion sync: pull the best spatial organism into the
                # visual genome so the shader keeps evolving even when the
                # spatial view is off.
                if self._config_get("enable_spatial_ecosystem", False, bool):
                    try:
                        self._sync_spatial_champion()
                    except (AttributeError, RuntimeError) as e:
                        self._logger.debug("Champion sync failed: %s", e)
                    except Exception as e:
                        self._logger.warning("Unexpected error in champion sync: %s", e)

            # Update music signature + pressure bias for trait-aware selection
            self._update_music_signature()

            # Update ecosystem memory and world evolution
            self._update_ecosystem_memory()

            # Emergent camera population — evolves camera organisms
            # independently of the main shader genome.
            if (
                self._config_get("enable_emergent_camera", False, bool)
                and self.camera_population is not None
            ):
                try:
                    scene_entropy = float(getattr(self, "_last_scene_entropy", 0.5))
                    scene_motion = float(getattr(self, "_last_scene_motion", 0.0))
                    births = getattr(self, "_recent_births", [])
                    self.camera_population.step(
                        dt,
                        self.audio.audio,
                        scene_entropy,
                        scene_motion,
                        self.ecosystem.organisms if self.ecosystem else [],
                        births,
                    )
                    self._sync_camera_organisms()
                except (AttributeError, RuntimeError, ValueError) as e:
                    self._logger.debug("Camera step failed: %s", e)
                except Exception as e:
                    self._logger.warning("Unexpected error in camera population update: %s", e)

            # Shader-graph genome — prune and mutate the active graph.
            if (
                isinstance(self.genome, ShaderGraphGenome)
                and self.shader_graph_compiler is not None
            ):
                try:
                    self._evolve_shader_graph_genome()
                except (AttributeError, RuntimeError, ValueError) as e:
                    self._logger.debug("Shader graph mutation failed: %s", e)
                except Exception as e:
                    self._logger.warning("Unexpected error in shader graph update: %s", e)

            # Continuous per-frame evolution — scheduled mutations,
            # fitness evaluation, and weakest-organism replacement.
            if self._config_get("enable_steady_state_evolution", False, bool):
                try:
                    self._update_scheduler_inputs(dt)
                    self._steady_state_evolve()
                except (RuntimeError, AttributeError, ValueError) as e:
                    self._logger.debug("Steady-state evolution failed: %s", e)
                except Exception as e:
                    self._logger.warning("Unexpected error in steady-state evolution: %s", e)

            # Get audio levels for beat evolution
            bass, mid, treb, beat = self.audio.uvec()

            # Beat-driven evolution: a burst of mutations on strong beats.
            try:
                self._maybe_beat_evolution(bass, mid, treb, beat)
            except (RuntimeError, AttributeError, ValueError) as e:
                self._logger.debug("Beat evolution failed: %s", e)
            except Exception as e:
                self._logger.warning("Unexpected error in beat evolution: %s", e)

            # Calculate elapsed time
            t = time.time() - self._start_time

            # Update automatic view toggles
            self._update_auto_toggles_safe(t, beat)

            # Memory watchdog for OOM prevention
            self._update_memory_watchdog_safe()

            return bass, mid, treb, beat, t

    def _evolve_shader_graph_genome(self) -> None:
        """Prune/mutate the active shader-graph genome (extracted helper)."""
        g = self.genome
        g.track_usage()
        if random.random() < 0.05:
            removed = g.prune_unused_nodes()
            if removed:
                self._logger.debug(f"Self-pruned {removed} unused node(s) from shader graph")
                start = time.perf_counter()
                src = self.shader_graph_compiler.compile(
                    g.graph,
                    brightness=self._config.brightness_base,
                    feedback_alpha=self._config.feedback_alpha,
                )
                compile_ms = (time.perf_counter() - start) * 1000.0
                self._recompile_fragment_shader(src, g)
        if random.random() < 0.02:
            g.generation += 1
            event = g.mutate(strength=0.15)
            if not event["valid"]:
                self._logger.debug(
                    f"Shader graph mutation rejected (invalid): "
                    f"{event['mutation']} -> {event['detail']}"
                )
            else:
                if not hasattr(self, "_shader_graph_basis"):
                    self._shader_graph_basis = []
                nov = g.novelty(self._shader_graph_basis)
                sig = None
                try:
                    from evo_shader_graph import _graph_signature

                    sig = _graph_signature(g.graph)
                except (ImportError, AttributeError, TypeError, ValueError) as e:
                    self._logger.debug("Graph signature failed: %s", e)
                if sig is not None:
                    self._shader_graph_basis.append(sig)
                    if len(self._shader_graph_basis) > 64:
                        self._shader_graph_basis.pop(0)
                start = time.perf_counter()
                src = self.shader_graph_compiler.compile(
                    g.graph,
                    brightness=self._config.brightness_base,
                    feedback_alpha=self._config.feedback_alpha,
                )
                compile_ms = (time.perf_counter() - start) * 1000.0
                self._recompile_fragment_shader(src, g)
                n0, n1 = event["nodes"]
                e0, e1 = event["edges"]
                metrics = event.get("metrics", {})
                self._logger.debug(
                    f"Generation {event['generation']} | "
                    f"Mutation: {event['mutation']} | {event['detail']} | "
                    f"Nodes: {n0}->{n1} | Edges: {e0}->{e1} | "
                    f"Instructions: {metrics.get('instructions', '?')} | "
                    f"ALU: {metrics.get('alu', '?')} | "
                    f"TexReads: {metrics.get('texture_reads', '?')} | "
                    f"Branches: {metrics.get('branches', '?')} | "
                    f"Compile: OK ({compile_ms:.2f} ms) | "
                    f"Fitness: {event['fitness']:+.2f}% | "
                    f"Novelty: {nov:.2f}"
                )

    def on_render(self, area: Gtk.GLArea, ctx) -> bool:
        """Main render callback. MUST always return True (GTK GLArea contract)."""
        with self._perf_timer("on_render_total", log_threshold_ms=16.0):
            if not self._program or not self.gl_ready:
                now = time.time()
                since = now - getattr(self, "_last_black_screen_warn", 0.0)
                if since >= 1.0:
                    self._last_black_screen_warn = now
                    self._logger.warning(
                        "Render suppressed: program=%s gl_ready=%s",
                        self._program is not None,
                        self.gl_ready,
                    )
                return True

            gtk_default_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)

            area_alloc_w = area.get_width()
            area_alloc_h = area.get_height()
            if self._logger.isEnabledFor(logging.DEBUG):
                # glGetIntegerv(GL_VIEWPORT) is a synchronous round-trip to the
                # driver; only pay for it when the debug line will actually print.
                self._logger.debug(
                    "ALLOC-CHAIN: Notebook→EvoVisualizerTab→EvoRenderer→glViewport | "
                    "alloc=%dx%d scale=%d default_fbo=%s viewport=%s glarea_w=%d glarea_h=%d",
                    area_alloc_w,
                    area_alloc_h,
                    area.get_scale_factor(),
                    gtk_default_fbo,
                    GL.glGetIntegerv(GL.GL_VIEWPORT),
                    self._glarea_w,
                    self._glarea_h,
                )

            with self._perf_timer("gl_program_cleanup"):
                while self.pending_delete:
                    prog = self.pending_delete.popleft()
                    try:
                        GL.glDeleteProgram(prog)
                    except (GL.error.GLError, TypeError, ValueError) as e:
                        self._logger.debug("Failed to delete queued GL program %s: %s", prog, e)
                    except Exception as e:
                        self._logger.warning(
                            "Unexpected error deleting GL program %s: %s", prog, e
                        )

            with self._perf_timer("drain_async_shaders"):
                self._drain_async_shaders()

            now = time.time()
            dt = min(0.1, max(1e-3, now - self._last_frame_ts))
            self._last_frame_ts = now

            ctrl = getattr(self, "quality_controller", None)
            if ctrl is not None:
                try:
                    ctrl.update(dt * 1000.0)
                except (RuntimeError, AttributeError, ValueError) as e:
                    self._logger.debug("Quality controller update failed: %s", e)
                except Exception as e:
                    self._logger.warning("Unexpected error in quality controller: %s", e)

            # Simulation/evolution updates run BEFORE choosing the presentation
            # mode, so spatial rendering cannot bypass steady-state evolution,
            # beat evolution, auto toggles, fitness cadence, or the watchdog.
            # ``beat`` is consumed inside _pre_render_update() (beat evolution and
            # auto toggles), so the render passes below only need the bands + t.
            bass, mid, treb, _beat, t = self._pre_render_update(dt)

            # Update per-frame animation (Milkdrop-style)
            if hasattr(self.genome, "update_animation"):
                self.genome.update_animation(self.audio.audio, t, _beat)

            # Beat-driven smooth transitions
            self._update_beat_transition(_beat, dt, bass, mid, treb)

            # Evolving post-effects: gene drift, effect selection and
            # crossfade state advance on the same cadence as the genome.
            if self._effects is not None:
                try:
                    self._effects.update(dt, t, bass, mid, treb, _beat)
                except Exception as e:
                    self._logger.debug("Effects update failed: %s", e)

            # Folder visualiser: shader-pool gene drift and selection timers.
            if self._folder_viz is not None:
                try:
                    self._folder_viz.update(dt, t, bass, mid, treb, _beat)
                except Exception as e:
                    self._logger.debug("Folder visualizer update failed: %s", e)

            if self._spatial_active and self.ecosystem is not None and self._vao is not None:
                scale = float(area.get_scale_factor())
                aw = max(1, int(area.get_width() * scale))
                ah = max(1, int(area.get_height() * scale))
                self._last_glarea_w = aw
                self._last_glarea_h = ah
                try:
                    with self._perf_timer("render_spatial"):
                        if self._render_spatial(dt, aw, ah, gtk_default_fbo):
                            # Spatial rendering must not starve visual fitness: score
                            # the rendered phenotype through the same capture path the
                            # feedback mode uses, so spatial organisms evolve from what
                            # is actually on screen.
                            self._run_visual_fitness_capture()
                            self._queue_heatmap_overlay_draw()
                            return True
                except (GL.error.GLError, RuntimeError, MemoryError) as e:
                    import traceback

                    self._logger.error("Spatial render error: %s", e)
                    self._logger.error(traceback.format_exc(limit=8))
                    self._spatial_active = False

        if (
            self._milkdrop_mode
            and self._milkdrop_program is not None
            and self._milkdrop_program != 0
        ):
            # Classic Milkdrop preset render: draw the compiled .milk
            # equations through the same feedback ping-pong path the
            # evolutionary genome uses, so presets get the same
            # audio-reactive feedback and previous-frame sampling.
            #
            # This branch shares the _run_visual_fitness_capture() call
            # with the main (evolutionary) path below — both non-spatial
            # render paths funnel through the same capture site so the
            # fitness cadence is identical regardless of which visual
            # source is on screen.
            try:
                with self._perf_timer("milkdrop_render"):
                    self._maybe_auto_switch_milkdrop(dt)
                default_fbo = gtk_default_fbo
                scale = float(area.get_scale_factor())
                aw = area.get_width()
                ah = area.get_height()
                allocated_w = max(1, int(aw * scale))
                allocated_h = max(1, int(ah * scale))
                self._last_glarea_w = allocated_w
                self._last_glarea_h = allocated_h
                fb_scale = 1.0
                if ctrl is not None:
                    fb_scale = float(getattr(ctrl, "feedback_scale", 1.0))
                fb_w = max(64, round(allocated_w * fb_scale))
                fb_h = max(64, round(allocated_h * fb_scale))
                self.feedback.resize(fb_w, fb_h)

                GL.glUseProgram(self._milkdrop_program)
                GL.glBindVertexArray(self._vao)

                # Feedback pass: render into the feedback FBO.
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.feedback.dst_fbo())
                GL.glViewport(0, 0, self.feedback.width, self.feedback.height)
                self._apply_milkdrop_uniforms(
                    bass=bass,
                    mid=mid,
                    treb=treb,
                    beat=_beat,
                    t=t,
                    device_w=self.feedback.width,
                    device_h=self.feedback.height,
                    previous_frame_tex=self.feedback.src_tex(),
                )
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

                self.feedback.swap(default_fbo)

                # Screen pass: blit feedback to the GTK default FBO.
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, default_fbo)
                GL.glViewport(0, 0, allocated_w, allocated_h)
                self._apply_milkdrop_uniforms(
                    bass=bass,
                    mid=mid,
                    treb=treb,
                    beat=_beat,
                    t=t,
                    device_w=allocated_w,
                    device_h=allocated_h,
                    previous_frame_tex=self.feedback.src_tex(),
                )
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

                GL.glBindVertexArray(0)
                GL.glUseProgram(0)

                # Shared with the main path: fall through to the capture
                # + heatmap block at the bottom of the non-spatial branch.
            except Exception as e:
                import traceback

                self._logger.error(f"Milkdrop render error: {e}")
                self._logger.error(traceback.format_exc(limit=8))
                self.exit_milkdrop_mode()

        if not self._spatial_active:
            try:
                default_fbo = gtk_default_fbo

                # Presentation-only sizing: simulation/evolution already ran in
                # _pre_render_update() before the spatial/main branch.
                scale = float(area.get_scale_factor())
                aw = area.get_width()
                ah = area.get_height()
                allocated_w = max(1, int(aw * scale))
                allocated_h = max(1, int(ah * scale))
                self._last_glarea_w = allocated_w
                self._last_glarea_h = allocated_h
                fb_scale = 1.0
                if ctrl is not None:
                    fb_scale = float(getattr(ctrl, "feedback_scale", 1.0))
                fb_w = max(64, round(allocated_w * fb_scale))
                fb_h = max(64, round(allocated_h * fb_scale))
                self.feedback.resize(fb_w, fb_h)
                if ctrl is not None:
                    self.feedback.ensure_history(int(getattr(ctrl, "feedback_history", 0)))

                # Folder visualiser mode: the shaders/ + effects/ + textures/
                # pools replace the genome on screen.  The manager renders
                # its own base layer and composites the effects chain, then
                # shares the fitness-capture + heatmap tail below.
                fv = self._folder_viz
                if fv is not None and fv.enabled:
                    try:
                        fv.ensure_size(allocated_w, allocated_h)
                        if fv.render(
                            self,
                            gtk_default_fbo,
                            allocated_w,
                            allocated_h,
                            t=t,
                            dt=dt,
                            bass=bass,
                            mid=mid,
                            treb=treb,
                            beat=_beat,
                        ):
                            self._run_visual_fitness_capture()
                            self._queue_heatmap_overlay_draw()
                            return True
                    except Exception as e:
                        self._logger.warning("Folder visualizer render failed: %s", e)
                        fv.disable()

                if (
                    self._multi_res_pyramid_view
                    and getattr(self, "_multi_res", None) is not None
                    and self._render_multi_res_pyramid(
                        t, bass, mid, treb, allocated_w, allocated_h
                    )
                ):
                    return True

                if (
                    self._competition_view
                    and self._config.population_size > 1
                    and self._render_competition(
                        t, bass, mid, treb, allocated_w, allocated_h, gtk_default_fbo
                    )
                ):
                    return True

                if self._population_view and self._config.population_size > 1:
                    self._render_population_grid(t, bass, mid, treb, allocated_w, allocated_h)

                if self._region_view and self._config.population_size > 1:
                    self._render_region_view(
                        t, bass, mid, treb, allocated_w, allocated_h, gtk_default_fbo
                    )

                GL.glViewport(0, 0, allocated_w, allocated_h)
                self._logger.debug(
                    "glViewport: restored to full GLArea %dx%d after sub-renders",
                    allocated_w,
                    allocated_h,
                )

                self._apply_camera_to_genome()

                self._ensure_mp_pipeline()
                if (
                    isinstance(self.genome, PipelineGenome)
                    and self._mp_pipeline is not None
                    and self._mp_pipeline.programs
                ):
                    try:
                        res = self.feedback.res
                        self._mp_pipeline.resize(res, res)
                        self._mp_pipeline.render(
                            vao=self._vao,
                            quad_count=QUAD_VERTEX_COUNT,
                            time=t,
                            bass=bass,
                            mid=mid,
                            treb=treb,
                            input_tex=self.feedback.src_tex(),
                            feedback_tex=self.feedback.src_tex(),
                            # GTK's GLArea renders into its own framebuffer,
                            # which is not necessarily 0; target the binding
                            # captured at the top of on_render().
                            final_fbo=gtk_default_fbo,
                            anim_values=(
                                self.genome.get_animation_uniforms()
                                if hasattr(self.genome, "get_animation_uniforms")
                                else None
                            ),
                        )
                        self.feedback.swap(default_fbo)
                        return True
                    except Exception as exc:
                        self._logger.debug(f"Multi-pass render failed: {exc}")
                        self._mp_pipeline = None

                self._set_program(self._get_program(self.genome))
                GL.glUseProgram(self._program)
                GL.glBindVertexArray(self._vao)

                self._apply_render_uniforms(
                    bass=bass,
                    mid=mid,
                    treb=treb,
                    t=t,
                    device_w=self.feedback.width,
                    device_h=self.feedback.height,
                    feedback_res=self.feedback.res,
                    previous_frame_tex=self.feedback.src_tex(),
                )

                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.feedback.dst_fbo())
                self._logger.debug(
                    "glViewport: feedback FBO size=%dx%d",
                    self.feedback.width,
                    self.feedback.height,
                )
                GL.glViewport(0, 0, self.feedback.width, self.feedback.height)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

                self.feedback.swap(default_fbo)

                # When evolving effects are active the "screen pass" renders
                # into the effects manager's scene target instead of the GTK
                # default framebuffer; the manager then composites the active
                # effect chain on top and blits the result to the screen.
                fx = self._effects
                scene_fbo = 0
                if fx is not None and fx.enabled:
                    try:
                        fx.ensure_size(allocated_w, allocated_h)
                        scene_fbo = fx.scene_fbo()
                    except Exception as e:
                        self._logger.debug("Effects sizing failed: %s", e)
                        scene_fbo = 0
                GL.glBindFramebuffer(
                    GL.GL_FRAMEBUFFER, scene_fbo if scene_fbo else default_fbo
                )
                self._logger.debug(
                    "glViewport: screen size=%dx%d (allocated=%dx%d)",
                    allocated_w,
                    allocated_h,
                    area_alloc_w,
                    area_alloc_h,
                )
                GL.glViewport(0, 0, allocated_w, allocated_h)
                w = allocated_w
                h = allocated_h

                self._apply_render_uniforms(
                    bass=bass,
                    mid=mid,
                    treb=treb,
                    t=t,
                    device_w=w,
                    device_h=h,
                    feedback_res=self.feedback.res,
                    previous_frame_tex=self.feedback.src_tex(),
                    resolution_only=True,
                )

                GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

                if scene_fbo and fx is not None:
                    try:
                        fx.present(
                            self,
                            default_fbo,
                            allocated_w,
                            allocated_h,
                            t=t,
                            dt=dt,
                            bass=bass,
                            mid=mid,
                            treb=treb,
                            beat=_beat,
                        )
                    except Exception as e:
                        self._logger.warning("Effects present failed: %s", e)
                        fx.disable()

                GL.glBindVertexArray(0)
                GL.glUseProgram(0)

                # Fitness capture is shared with the spatial path so both modes
                # score the same way and at the same cadence.
                self._run_visual_fitness_capture()

                # Already run in _pre_render_update(); kept out of the spatial
                # branch deliberately so both modes share one cadence.
                self._queue_heatmap_overlay_draw()

                if self._logger.isEnabledFor(logging.DEBUG):
                    vp_end = GL.glGetIntegerv(GL.GL_VIEWPORT)
                    self._logger.debug(
                        "Render end: feedback=%dx%d screen_viewport=%s end_fbo=%s",
                        self.feedback.width,
                        self.feedback.height,
                        vp_end,
                        GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING),
                    )

                return True

            except (GL.error.GLError, RuntimeError, MemoryError) as e:
                import traceback

                self._logger.error("Render error: %s", e)
                self._logger.error(traceback.format_exc(limit=8))
                # GTK's GLArea contract: always return True. Returning False
                # makes GTK treat the render signal as failed, which turns one
                # bad preset/program into a persistent black widget.
                return True
            except Exception as e:
                import traceback

                self._logger.error("Unexpected render error: %s", e)
                self._logger.error(traceback.format_exc(limit=8))
                return True

    def _render_competition(
        self, t: float, bass: float, mid: float, treb: float, w: int, h: int, default_fbo: int = 0
    ) -> bool:
        """Render the competition: Population -> Top N -> Blend -> Compete -> Replace.

        1. Select top N organisms by fitness.
        2. Render each organism into its own tile on a shared grid FBO.
        3. Blend all tiles fitness-weighted into a composite view.
        4. Compete: evaluate each organism's tile fitness.
        5. Replace: promote the winner to champion if better.
        """
        organisms = self.population.organisms
        n = len(organisms)
        if n < 2:
            return True
        n_comp = min(self._competition_n, n)
        scored = []
        for org in organisms:
            try:
                s = float(getattr(org, "fitness", 0.0))
            except (AttributeError, TypeError, ValueError):
                s = 0.0
            scored.append((s, org))
        scored.sort(key=lambda x: -x[0])
        top = [org for _, org in scored[:n_comp]]

        if self._competition_renderer is None:
            self._competition_renderer = CompetitionRenderer(pool=self._fbo_pool)
        cr = self._competition_renderer
        cr.n = n_comp
        cr.resize(w, h)

        programs = []
        uniforms_list = []
        for org in top:
            try:
                prog = self._get_program(org, allow_async=False)
                programs.append(int(prog) if prog else 0)
            except (GL.error.GLError, TypeError, ValueError, AttributeError, RuntimeError) as e:
                self._logger.debug("Competition program compile failed: %s", e)
                programs.append(0)
            uniforms_list.append({})

        cr.render_tiles(
            vao=self._vao,
            organisms=top,
            programs=programs,
            uniforms_list=uniforms_list,
            time=t,
            bass=bass,
            mid=mid,
            treb=treb,
            previous_frame_tex=self.feedback.src_tex() if self.feedback else 0,
        )

        fits = []
        for org in top:
            try:
                f = max(0.01, float(getattr(org, "fitness", 0.0)))
            except (AttributeError, TypeError, ValueError):
                f = 0.01
            fits.append(f)
        total_f = sum(fits) or 1.0
        weights = [f / total_f for f in fits]

        winner_idx = int(max(range(len(fits)), key=lambda i: fits[i]))
        blend_tex = cr.render_blend(
            vao=self._vao,
            weights=weights,
        )

        try:
            GL.glBindFramebuffer(
                GL.GL_FRAMEBUFFER,
                default_fbo if default_fbo else GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING),
            )
            self._logger.debug("glViewport: competition restore %dx%d", w, h)
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            if blend_tex:
                GL.glUseProgram(cr._blend_prog)
                GL.glBindVertexArray(self._vao)
                for i in range(len(top)):
                    if i >= 8:
                        break
                    loc = cr._blend_locs["u_tiles"][i]
                    if loc >= 0:
                        GL.glActiveTexture(GL.GL_TEXTURE0 + i)
                        GL.glBindTexture(GL.GL_TEXTURE_2D, cr.tile_tex(i))
                        GL.glUniform1i(loc, i)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)
        except GL.error.GLError as e:
            self._logger.debug(f"Screen blit skipped: {e}")

        winner = top[winner_idx]
        current = self.genome
        try:
            wf = float(getattr(winner, "fitness", 0.0))
            cf = float(getattr(current, "fitness", 0.0))
        except (AttributeError, TypeError, ValueError):
            wf = cf = 0.0
        if wf > cf:
            self._set_program(self._get_program(winner))
            self.genome = winner
            self.population.organisms[0] = self.genome
            if self._config.enable_lineage:
                try:
                    self.memory_dna.record(self.genome)
                except (AttributeError, TypeError, ValueError) as e:
                    self._logger.debug("Memory DNA record failed: %s", e)

        return True

    def _render_population_grid(
        self, t: float, bass: float, mid: float, treb: float, w: int, h: int
    ) -> bool:
        """Draw every organism in the population into a tiled grid.
        Used by the population visualization mode so the whole ecosystem is
        visible at once. The champion is promoted automatically based on
        fitness.
        """
        organisms = self.population.organisms
        n = max(1, len(organisms))
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        tile_w = w / cols
        tile_h = h / rows
        GL.glUseProgram(self._program)
        GL.glBindVertexArray(self._vao)
        for idx, org in enumerate(organisms):
            r = idx // cols
            c = idx % cols
            x0 = c * tile_w
            y0 = h - (r + 1) * tile_h  # GL viewport origin is bottom-left
            GL.glViewport(int(x0), int(y0), int(tile_w), int(tile_h))
            self._set_program(self._get_program(org))
            self._apply_render_uniforms(
                bass=bass,
                mid=mid,
                treb=treb,
                t=t,
                device_w=int(tile_w),
                device_h=int(tile_h),
                feedback_res=self.feedback.res,
                previous_frame_tex=self.feedback.src_tex(),
            )
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)
        self._logger.debug("glViewport: population grid restore %dx%d", w, h)
        GL.glViewport(0, 0, w, h)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        return True

    def _ensure_region_renderer(self, w: int, h: int) -> None:
        if self._region_renderer is None:
            self._region_renderer = RegionBasedRenderer(
                pool=self._fbo_pool,
                n=getattr(self._config, "region_count", 5),
            )
            self._region_renderer.resize(w, h)
            layout_name = getattr(self._config, "region_layout", "split_left_right")
            if layout_name == "split_top_bottom":
                regions = RegionLayout.split_top_bottom(self._region_renderer.n)
            elif layout_name == "split_quadrants":
                regions = RegionLayout.split_quadrants(self._region_renderer.n)
            elif layout_name == "split_grid":
                regions = RegionLayout.split_grid(self._region_renderer.n)
            else:
                regions = RegionLayout.split_left_right(self._region_renderer.n)
            if self._niche_manager is not None:
                self._niche_manager.update_regions(
                    [{"x": r.x, "y": r.y, "w": r.w, "h": r.h} for r in regions]
                )
            self._region_renderer.set_regions(regions)

    def _render_region_view(
        self, t: float, bass: float, mid: float, treb: float, w: int, h: int, default_fbo: int = 0
    ) -> bool:
        """Render the population into named screen regions with
        environmental niche overlays. Each region is tinted by its
        niche colour and organisms receive niche-based gene bias
        so that genes favoured by the local environment thrive.

        Each organism paints its own region of the screen; a composite
        pass blends them together with soft edges so the ecosystem literally
        paints the screen as one collaborative piece.
        """
        organisms = self.population.organisms
        n = len(organisms)
        if n < 2:
            return True
        self._ensure_region_renderer(w, h)
        rr = self._region_renderer
        rr.resize(w, h)

        n_use = min(rr.n, n)
        top = organisms[:n_use]

        # Build region bounds and niche colours for the composite pass.
        region_bounds: list[dict[str, float]] = []
        niche_colours: list[tuple[float, float, float]] = []
        niche_opacity = float(getattr(self._config, "niche_overlay_opacity", 0.08))
        for i, region in enumerate(rr._regions):
            region_bounds.append(
                {
                    "x": region.x,
                    "y": region.y,
                    "w": region.w,
                    "h": region.h,
                }
            )
            if self._niche_manager is not None:
                niche_colours.append(self._niche_manager.region_colour(i))
            else:
                niche_colours.append((0.5, 0.5, 0.5))

        # Apply niche-based gene bias to each organism in its region.
        for i, org in enumerate(top):
            if self._niche_manager is not None and i < len(region_bounds):
                niche = self._niche_manager.niches[i % len(self._niche_manager.niches)]
                self._apply_niche_gene_bias(org, niche)

        programs = []
        uniforms_list = []
        for idx, org in enumerate(top):
            try:
                prog = self._get_program(org, allow_async=False)
                programs.append(int(prog) if prog else 0)
            except (GL.error.GLError, TypeError, ValueError, AttributeError, RuntimeError) as e:
                self._logger.debug("Region program compile failed: %s", e)
                programs.append(0)
            unis = {}
            if self._niche_manager is not None and idx < len(niche_colours):
                c = niche_colours[idx]
                unis["u_niche_colour"] = {
                    "kind": "float",
                    "value": c[0],
                    "value_g": c[1],
                    "value_b": c[2],
                }
                unis["u_niche_opacity"] = {
                    "kind": "float",
                    "value": niche_opacity,
                }
            uniforms_list.append(unis)

        rr.render_regions(
            vao=self._vao,
            organisms=top,
            programs=programs,
            uniforms_list=uniforms_list,
            time=t,
            bass=bass,
            mid=mid,
            treb=treb,
            previous_frame_tex=self.feedback.src_tex() if self.feedback else 0,
            feedback_tex=self.feedback.src_tex() if self.feedback else 0,
        )

        try:
            GL.glBindFramebuffer(
                GL.GL_FRAMEBUFFER,
                default_fbo if default_fbo else GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING),
            )
            self._logger.debug("glViewport: region composite restore %dx%d", w, h)
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            rr.composite(
                vao=self._vao,
                default_fbo=default_fbo
                if default_fbo
                else GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING),
            )
        except Exception as e:
            self._logger.debug(f"Region composite blit failed: {e}")

        return True

    def _apply_niche_gene_bias(self, organism: Any, niche: NicheDefinition) -> None:
        """Bias organism gene expression toward niche-favoured parameters.

        For genes that the niche favours, we nudge the genome parameter
        toward the niche-preferred range, making organisms in matching
        regions express their niche-adapted traits more strongly.

        Accepts either an organism wrapper exposing ``.genome`` (e.g.
        :class:`SpatialOrganism`) or a bare genome (``ShaderGraphGenome`` /
        ``VisualGenome``) as produced by ``self.population.organisms``.
        """
        g = getattr(organism, "genome", None) or organism
        if g is None:
            return
        niche_scale = float(getattr(self._config, "niche_fitness_scale", 0.3))
        step = 0.02
        biased: dict[str, float] = {}
        current = self._flat_genome_params(g)
        for gene_key, weight in niche.favours.items():
            cur = current.get(gene_key, 0.0)
            target = weight * niche_scale
            biased[gene_key] = cur + (target - cur) * step
        for gene_key, weight in niche.penalises.items():
            cur = current.get(gene_key, 0.0)
            target = -weight * niche_scale
            biased[gene_key] = cur + (target - cur) * step
        if biased:
            self._write_genome_params(g, biased)

    def _render_multi_res_pyramid(
        self, t: float, bass: float, mid: float, treb: float, w: int, h: int
    ) -> bool:
        """Draw the best organism from each multi-resolution tier in a rowed grid."""
        multi = getattr(self, "_multi_res", None)
        if multi is None:
            return True
        tiers = multi.tiers
        n = len(tiers)
        row_h = h // n
        GL.glUseProgram(self._program)
        GL.glBindVertexArray(self._vao)
        for idx, tier in enumerate(tiers):
            pop = multi.populations.get(tier.name)
            if not pop:
                continue
            org = pop.best()
            if org is None:
                continue
            prog = self._get_program(org, allow_async=False)
            if not prog:
                continue
            y0 = h - (idx + 1) * row_h
            GL.glViewport(0, y0, w, row_h)
            self._set_program(int(prog))
            self._apply_render_uniforms(
                bass=bass,
                mid=mid,
                treb=treb,
                t=t,
                device_w=w,
                device_h=row_h,
                feedback_res=self.feedback.res,
                previous_frame_tex=self.feedback.src_tex(),
            )
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)
        self._logger.debug("glViewport: multi-res pyramid restore %dx%d", w, h)
        GL.glViewport(0, 0, w, h)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        return True

    def toggle_multi_res_view(self) -> None:
        """Toggle the multi-resolution pyramid overlay."""
        self._multi_res_pyramid_view = not self._multi_res_pyramid_view
        self._logger.info(
            f"Multi-res pyramid view {'on' if self._multi_res_pyramid_view else 'off'}"
        )
        self.queue_render()

    def _memory_watchdog(self) -> None:
        """Proactively trim GPU/process memory when RSS approaches the budget.

        This is the OOM-mitigation tripwire from the post-mortem, upgraded from
        *log only* to *act*:

          * samples RSS at a low cadence (psutil -> /proc/self -> resource);
          * computes ``pressure`` = RSS / budget (0..1);
          * feeds the pressure into :class:`AdaptiveHistoryController` to shrink
            the feedback history, replay ring, lineage log and novelty history
            toward their floors;
          * proportionally trims the LRU shader cache (the biggest single GPU
            allocation) once pressure passes 50%.

        All of this is throttled via the budget's sampling interval and only
        *acts* when pressure is meaningful, so it costs almost nothing in the
        render loop and fires *before* the kernel OOM killer does.
        """
        if not getattr(self._config, "memory_trim_enabled", True):
            return
        budget = getattr(self, "_memory_budget", None)
        if budget is None:
            return
        # The budget self-throttles its RSS sampling; call it every frame cheaply.
        pressure = budget.pressure()
        if pressure < 0.5:
            # Below half budget there is nothing to reclaim; skip the trim work.
            return

        try:
            action = apply_adaptive_trims(self, budget, self._adaptive_history)
        except Exception as e:
            self._logger.debug("Memory watchdog trim failed: %s", e)
            return

        if action:
            stats = budget.stats()
            self._logger.warning(
                "MEMORY WATCHDOG: RSS %.0f MB / %.0f MB (pressure %.2f) -> trimmed %s "
                "(gen=%d, cached_programs=%d, lineage_records=%d)",
                stats["rss_mb"],
                stats["threshold_mb"],
                pressure,
                ", ".join(f"{k}={v}" for k, v in action.items()),
                self._generation,
                len(self.shader_cache.cache),
                len(self.memory_dna._records),
            )

    def toggle_population_view(self) -> None:
        """Toggle the population visualization grid on/off."""
        self._population_view = not self._population_view
        self._logger.info(f"Population view {'on' if self._population_view else 'off'}")
        self.queue_render()

    def toggle_competition_view(self) -> None:
        """Toggle the competition renderer (Top 8 -> Blend -> Compete -> Replace)."""
        self._competition_view = not self._competition_view
        if not self._competition_view and self._competition_renderer is not None:
            try:
                self._competition_renderer.cleanup()
            except (AttributeError, TypeError, ValueError, GL.error.GLError) as e:
                self._logger.debug("Competition renderer cleanup failed: %s", e)
            self._competition_renderer = None
        self._logger.info(f"Competition view {'on' if self._competition_view else 'off'}")
        self.queue_render()

    def toggle_heatmap_view(self) -> None:
        """Toggle the on-screen mutation heatmap overlay on/off."""
        self._heatmap_view = not self._heatmap_view
        self._logger.info(f"Mutation heatmap {'on' if self._heatmap_view else 'off'}")
        # Repaint the overlay immediately; on_render() only keeps it in sync
        # while the heatmap is enabled.
        self._queue_heatmap_overlay_draw()
        self.queue_render()

    def toggle_region_view(self) -> None:
        """Toggle the region-based renderer (ecosystem paints the screen together)."""
        self._region_view = not self._region_view
        if not self._region_view and self._region_renderer is not None:
            try:
                self._region_renderer.cleanup()
            except (AttributeError, TypeError, ValueError, GL.error.GLError) as e:
                self._logger.debug("Region renderer cleanup failed: %s", e)
            self._region_renderer = None
        self._logger.info(f"Region view {'on' if self._region_view else 'off'}")
        self.queue_render()

    def build_heatmap_overlay(self) -> Gtk.Widget:
        """Return this renderer wrapped in an overlay that paints the heatmap.

        ``Gtk.GLArea`` cannot host Cairo children and its ``render`` signal
        hands the handler a ``GdkGLContext`` (which has no Cairo drawing API),
        so the mutation heatmap is drawn by a transparent
        :class:`Gtk.DrawingArea` stacked on top of the GL surface inside a
        :class:`Gtk.Overlay`.

        The drawing area is input-transparent (``set_can_target(False)``) so it
        never swallows pointer/keyboard events aimed at the renderer. Hosts that
        simply append the renderer directly still work; they just get no
        heatmap overlay.

        Returns:
            A ``Gtk.Overlay`` whose child is this renderer.
        """
        overlay = Gtk.Overlay()
        overlay.set_hexpand(self.get_hexpand())
        overlay.set_vexpand(self.get_vexpand())
        overlay.set_halign(self.get_halign())
        overlay.set_valign(self.get_valign())
        overlay.set_hexpand_set(self.get_hexpand_set())
        overlay.set_vexpand_set(self.get_vexpand_set())
        overlay.set_child(self)

        drawing = Gtk.DrawingArea()
        drawing.set_can_target(False)
        drawing.set_hexpand(True)
        drawing.set_vexpand(True)
        drawing.set_halign(Gtk.Align.FILL)
        drawing.set_valign(Gtk.Align.FILL)
        drawing.set_draw_func(self._on_heatmap_draw)
        overlay.add_overlay(drawing)

        self._heatmap_drawing_area = drawing
        self._logger.debug("Heatmap overlay attached (Gtk.Overlay + Gtk.DrawingArea)")
        return overlay

    def _queue_heatmap_overlay_draw(self) -> None:
        """Request a repaint of the heatmap overlay, if one is attached."""
        if not self._heatmap_view:
            return
        drawing = self._heatmap_drawing_area
        if drawing is None:
            if not self._heatmap_ctx_warned:
                self._heatmap_ctx_warned = True
                self._logger.warning(
                    "Mutation heatmap view is on but no overlay is attached; the "
                    "host must use EvoRenderer.build_heatmap_overlay(). "
                    "Gtk.GLArea cannot draw Cairo content by itself."
                )
            return
        # queue_draw() is a no-op if a frame is already pending, so this does
        # not add render pressure beyond the renderer's own auto_render cadence.
        drawing.queue_draw()

    def _on_heatmap_draw(self, _drawing_area: Gtk.DrawingArea, ctx, _w: int, _h: int) -> None:
        """``Gtk.DrawingArea`` draw callback: paint the heatmap onto ``ctx``."""
        if not self._heatmap_view:
            return
        try:
            self._draw_heatmap(ctx)
        except Exception as e:
            self._logger.debug(f"Heatmap overlay failed: {e}")

    def _draw_heatmap(self, ctx) -> None:
        """Render the mutation heatmap as horizontal bars onto a Cairo context.

        Layout mirrors the spec:

            warp     ███████    +0.42   41x  82%
            glow     ██         +0.07    9x  55%
            feedback  ███████████ ...

        Each row shows the parameter name, a bar scaled by ``times_mutated``,
        the accumulated fitness gain (signed), how many times the gene was
        mutated, and its acceptance rate. Evolution becomes explainable.
        """
        try:
            import gi

            gi.require_version("Gtk", "4.0")
            from gi.repository import Pango, PangoCairo
        except Exception as e:
            self._logger.debug(f"Heatmap text backend unavailable: {e}")
            return

        rows = self.mutation_heatmap.snapshot()
        if not rows:
            return

        total_mut, total_acc = self.mutation_heatmap.totals()
        max_count = max(r.times_mutated for r in rows) or 1

        bands = getattr(getattr(self, "audio", None), "audio", {}) or {}
        pulse = min(
            1.0,
            0.65 * float(bands.get("beat", 0.0)) + 0.35 * float(bands.get("bass", 0.0)),
        )

        pad = 14.0
        line_h = 18.0
        label_w = 96.0
        bar_w = 150.0
        stats_w = 200.0
        width = pad * 2 + label_w + bar_w + stats_w
        height = pad * 2 + line_h * (len(rows) + 1)

        # Panel background.
        ctx.set_source_rgba(0.05, 0.06, 0.10, 0.82)
        self._round_rect(ctx, pad * 0.5, pad * 0.5, width, height, 8.0)
        ctx.fill()

        layout = PangoCairo.create_layout(ctx)
        layout.set_font_description(Pango.font_description_from_string("Monospace 11"))
        header = (
            f"MUTATION HEATMAP   "
            f"{total_mut} mutations  "
            f"{100.0 * total_acc / max(total_mut, 1):.0f}% accepted"
        )
        layout.set_text(header)
        ctx.set_source_rgba(0.7, 0.85, 1.0, 0.75 + 0.25 * pulse)
        ctx.move_to(pad, pad)
        PangoCairo.show_layout(ctx, layout)

        y = pad + line_h
        bar_h = (line_h - 6.0) * (0.85 + 0.3 * pulse)
        bar_alpha = 0.7 + 0.3 * pulse
        for r in rows:
            acc = self.mutation_heatmap.acceptance(r.param)
            frac = r.times_mutated / max_count
            # Bar colour: green (high acceptance) -> red (low acceptance).
            ctx.set_source_rgba(1.0 - acc, 0.4 + acc * 0.6, 0.35 * (1.0 - acc), bar_alpha)
            self._round_rect(
                ctx,
                pad + label_w,
                y + 2.0 + (line_h - 6.0 - bar_h) / 2.0,
                max(2.0, bar_w * frac),
                bar_h,
                3.0,
            )
            ctx.fill()

            layout.set_text(f"{r.param[:14]:<14}")
            ctx.set_source_rgba(0.9, 0.95, 1.0, 0.95)
            ctx.move_to(pad, y)
            PangoCairo.show_layout(ctx, layout)

            stats = f"{r.fitness_gain:+.2f}   {r.times_mutated}x   {100.0 * acc:.0f}%"
            layout.set_text(stats)
            ctx.set_source_rgba(0.8, 0.85, 0.9, 0.9)
            ctx.move_to(pad + label_w + bar_w + 8.0, y)
            PangoCairo.show_layout(ctx, layout)

            y += line_h

    @staticmethod
    def _round_rect(ctx, x: float, y: float, w: float, h: float, r: float) -> None:
        """Add a rounded-rectangle path to the Cairo context (does not fill)."""
        r = min(r, w / 2.0, h / 2.0)
        ctx.new_path()
        ctx.move_to(x + r, y)
        ctx.arc(x + w - r, y + r, r, -1.5708, 0.0)
        ctx.arc(x + w - r, y + h - r, r, 0.0, 1.5708)
        ctx.arc(x + r, y + h - r, r, 1.5708, 3.14159)
        ctx.arc(x + r, y + r, r, 3.14159, -1.5708)
        ctx.close_path()

    def _maybe_beat_evolution(self, bass: float, mid: float, treb: float, beat: float) -> None:
        """Trigger a burst of mutations on strong beats, scaled by intensity."""
        if beat < 0.7:
            return
        now = time.time()
        if now - self._last_beat_mutation_ts < self._beat_evolution_cooldown:
            return
        self._last_beat_mutation_ts = now
        # Keep the legacy shared timestamp in sync for external readers.
        self._last_beat_evolution_ts = now
        try:
            genome = self.genome
            if getattr(self, "_multi_res", None) is not None:
                champ = self._multi_res.champion()
                if champ is not None:
                    genome = champ
            burst_size = max(1, int(beat * 6))
            for _ in range(burst_size):
                genome.mutate(strength=self._audio_mutation_strength(0.01 + beat * 0.08))
            frag = self._gen_frag_source(genome)
            self._recompile_fragment_shader(frag, genome)
            if genome is not self.genome:
                self.genome = genome
                self.population.organisms[0] = self.genome
            self._logger.debug(f"Beat-driven burst: {burst_size} mutations")
        except Exception as e:
            self._logger.debug(f"Beat-driven mutation failed: {e}")

    def _update_beat_transition(
        self, beat: float, dt: float, bass: float, mid: float, treb: float
    ) -> None:
        """Update beat-driven smooth genome transitions.

        On strong beats, create a target genome with mutated parameters
        and smoothly interpolate toward it over several frames.
        """
        if beat < 0.6:
            # Update existing transition progress
            if self._beat_transition_target is not None:
                self._beat_transition_progress += dt / self._beat_transition_duration
                if self._beat_transition_progress >= 1.0:
                    self._beat_transition_progress = 1.0
                    self._beat_transition_target = None
                self._interpolate_genome(
                    self._beat_transition_target, self._beat_transition_progress
                )
            return

        now = time.time()
        if now - self._beat_transition_start_ts < 0.3:  # Debounce
            return

        self._beat_transition_start_ts = now

        # Create a mutated copy as transition target
        try:
            target = self.genome.clone()
            # Mutate with beat-reactive strength
            target.mutate(strength=self._audio_mutation_strength(0.02 + beat * 0.15))
            # Add extra animation kicks
            if hasattr(target, "anim_zoom"):
                target.anim_zoom_target = min(1.5, target.anim_zoom_target + 0.2 * beat)
                target.anim_rot_target += 0.3 * beat
                target.anim_warp_target = min(0.8, target.anim_warp_target + 0.2 * beat)

            self._beat_transition_target = target
            self._beat_transition_progress = 0.0
            self._logger.debug("Beat-driven transition started")
        except Exception as e:
            self._logger.debug(f"Beat transition setup failed: {e}")

    def _interpolate_genome(self, target: Any, factor: float) -> None:
        """Smoothly interpolate current genome toward target.

        Args:
            target: Target genome to interpolate toward
            factor: Interpolation factor (0.0 = current, 1.0 = target)
        """
        if not hasattr(target, "shader_signature"):
            return
        current = self.genome
        try:
            # Interpolate numeric parameters
            for attr in (
                "field_params",
                "warp_params",
                "palette_params",
                "feedback_params",
                "lighting_params",
                "camera_params",
                "particle_params",
                "post_params",
            ):
                if hasattr(current, attr) and hasattr(target, attr):
                    cur_params = getattr(current, attr, {})
                    tgt_params = getattr(target, attr, {})
                    for k in cur_params:
                        if k in tgt_params:
                            cur_params[k] = cur_params[k] * (1.0 - factor) + tgt_params[k] * factor

            # Interpolate animation parameters
            if hasattr(current, "anim_zoom") and hasattr(target, "anim_zoom"):
                anim_attrs = [k for k in current.__dict__ if k.startswith("anim_")]
                for attr in anim_attrs:
                    if hasattr(target, attr):
                        cur_val = getattr(current, attr, 0.0)
                        tgt_val = getattr(target, attr, 0.0)
                        setattr(current, attr, cur_val * (1.0 - factor) + tgt_val * factor)

            # Recompile shader with interpolated genome
            frag = self._gen_frag_source(current)
            self._recompile_fragment_shader(frag, current)
        except Exception as e:
            self._logger.debug(f"Genome interpolation failed: {e}")

    def _update_auto_toggles(self, t: float, beat: float) -> None:
        """Automatically toggle views and modes based on system state.

        Replaces keyboard controls so that all interaction is automatic:
        transitions trigger on strong beats, views cycle on timers, and
        spatial modes blend based on audio energy.
        """
        now = time.time()
        self._auto_toggle_step += 1

        # Auto-trigger transition on strong beats (replaces Space key).
        # Uses its own cooldown timestamp so a beat mutation does not suppress
        # the transition (or vice versa).
        if beat >= 0.7 and now - self._last_beat_transition_ts >= self._beat_evolution_cooldown:
            self._last_beat_transition_ts = now
            self._last_beat_evolution_ts = now
            try:
                self.trigger_transition()
                self._logger.debug("Auto beat-driven transition triggered")
            except Exception as e:
                self._logger.debug(f"Auto beat-driven transition failed: {e}")

        # Auto-cycle spatial mode every interval (replaces M key).
        # Uses its own timestamp so blend cycling cannot starve it.
        if now - self._last_auto_mode_ts >= self._auto_toggle_interval:
            self._last_auto_mode_ts = now
            self._last_auto_toggle_ts = now
            modes = ["territories", "colonies", "clouds", "flocks", "cells", "islands", "veins"]
            cur = getattr(self._config, "spatial_mode", "territories")
            try:
                idx = modes.index(cur)
                idx = (idx + 1) % len(modes)
            except ValueError:
                idx = 0
            self._config.spatial_mode = modes[idx]
            self._logger.info(f"Auto spatial mode: {modes[idx]}")
            self.queue_render()

        # Auto-cycle spatial blend every second interval (replaces b key).
        # Independent timestamp: mode cycling above must not reset/starve this.
        if now - self._last_auto_blend_ts >= self._auto_toggle_interval * 2:
            self._last_auto_blend_ts = now
            self._last_auto_toggle_ts = now
            blends = ["hard", "soft", "glow", "fluid"]
            cur = getattr(self._config, "spatial_blend", "soft")
            try:
                idx = blends.index(cur)
                idx = (idx + 1) % len(blends)
            except ValueError:
                idx = 0
            self._config.spatial_blend = blends[idx]
            self._logger.info(f"Auto spatial blend: {blends[idx]}")
            self.queue_render()

        # Auto-toggle views based on population size and config (disabled by default).
        if self._multi_res_pyramid_view != bool(
            getattr(self._config, "enable_multi_resolution_evolution", False)
        ):
            self._multi_res_pyramid_view = bool(
                getattr(self._config, "enable_multi_resolution_evolution", False)
            )
            self._logger.info(
                f"Auto multi-res view: {'on' if self._multi_res_pyramid_view else 'off'}"
            )
            self.queue_render()

        if self._region_view != bool(getattr(self._config, "enable_region_rendering", False)):
            self._region_view = bool(getattr(self._config, "enable_region_rendering", False))
            self._logger.info(f"Auto region view: {'on' if self._region_view else 'off'}")
            self.queue_render()

        # Auto-toggle territory overlay based on spatial ecosystem state.
        if (
            self._config.enable_spatial_ecosystem
            and self._spatial_active
            and not self._config.show_territories
        ):
            self._config.show_territories = True
            self._logger.info("Auto territory overlay on")
            self.queue_render()

    #: Legacy flat gene names tried first when copying the spatial champion's
    #: genome into the visual genome. Anything else the two genomes share is
    #: copied as a fallback.
    _SPATIAL_CHAMPION_GENES = (
        "warp",
        "noise_scale",
        "feedback",
        "glow",
        "symmetry",
        "contrast",
        "flow_speed",
        "vortex",
        "camera_speed",
        "camera_orbit",
        "domain_warp",
    )

    #: Per-module parameter dicts of a ModularGenome (the pipeline's field
    #: pass). The ``extra_*_params`` lists are skipped on purpose: they are
    #: positional module lists, not flat genes.
    _NAMED_PARAM_STORES = (
        "field_params",
        "warp_params",
        "palette_params",
        "feedback_params",
        "lighting_params",
        "camera_params",
        "particle_params",
        "post_params",
    )

    def _sync_spatial_champion(self) -> None:
        """Copy a slice of the best spatial organism's genome into the visual
        genome so the shader keeps evolving even when the spatial view is
        hidden or the user is looking at the feedback mode.

        The copy goes through the target genome's supported parameter API --
        the old ``g.params.update(...)`` was dead code twice over:
        ``VisualGenome.params`` is a property that rebuilds a fresh dict on
        every read (so ``.update()`` only mutated a throwaway copy) and
        ``ShaderGraphGenome`` (the default genome) has no ``params`` at all.

        The sync is throttled by ``spatial_champion_sync_interval`` so a
        champion that improves every frame cannot force a shader compile every
        frame. When a write changes ``shader_signature()``, the active program
        is invalidated and refreshed immediately; unchanged signatures leave
        the compiled shader and cache untouched.
        """
        eco = self.ecosystem
        if eco is None:
            return
        now = time.time()
        interval = float(getattr(self._config, "spatial_champion_sync_interval", 0.5))
        if now - float(getattr(self, "_last_spatial_sync_ts", 0.0)) < interval:
            return
        self._last_spatial_sync_ts = now
        alive = [o for o in eco.organisms if o.alive]
        if not alive:
            return
        genome = self.genome
        if genome is None:
            return
        best = max(alive, key=lambda o: o.fitness)
        source = getattr(best, "genome", None)
        if source is None or source is genome:
            return
        before_signature = self._genome_shader_signature(genome)
        try:
            params = self._flat_genome_params(source)
            preferred = {k: v for k, v in params.items() if k in self._SPATIAL_CHAMPION_GENES}
            # Legacy flat names first; otherwise copy gene-for-gene between
            # same-named stores so "scale" from the champion's field module
            # cannot land in the target's lighting module.
            written = self._write_genome_params(genome, preferred)
            if not written:
                written = self._copy_genome_params(genome, source)
            if written:
                self._logger.debug(
                    "Spatial champion synced into visual genome (fitness %.3f)",
                    float(best.fitness),
                )
                if self._genome_shader_signature(genome) != before_signature:
                    self._refresh_shader_for_genome(genome)
        except Exception as e:
            self._logger.debug(f"Champion sync failed: {e}")

    def _refresh_shader_for_genome(self, genome: Any) -> None:
        """Compile and bind a genome's program after an in-place gene sync."""
        try:
            self._last_module_fingerprint = None
            program = self._get_program(genome, allow_async=False)
            if program is not None:
                self._set_program(program)
        except Exception as e:
            self._logger.debug(f"Champion shader refresh failed: {e}")

    @staticmethod
    def _flat_genome_params(genome: Any) -> dict[str, float]:
        """Flat numeric gene view of ``genome`` using its own API.

        Graph genomes are handled before serialized signatures because their
        numeric genes live below each node's ``params`` mapping. Other genomes
        use ``shader_signature()``, ``params`` or ``to_dict()``. Nested dicts
        are flattened two levels deep; bools, strings (module selections) and
        lists of strings are skipped.
        """
        nodes = EvoRenderer._graph_nodes(genome)
        if nodes is not None:
            flat: dict[str, float] = {}
            for params in nodes.values():
                for key, value in params.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    flat[str(key)] = float(value)
            if flat:
                return flat

        for name in ("shader_signature", "params", "to_dict"):
            member = getattr(genome, name, None)
            if member is None:
                continue
            try:
                data = member() if callable(member) else member
            except Exception as e:
                logger.debug("Failed to extract genome data from %s: %s", name, e)
                continue
            if not isinstance(data, dict):
                continue
            flat = {}

            def _collect(mapping: dict[str, Any], depth: int, flat: dict[str, float]) -> None:
                for key, value in mapping.items():
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)):
                        flat[str(key)] = float(value)
                    elif isinstance(value, dict) and depth > 0:
                        _collect(value, depth - 1, flat)
                    elif isinstance(value, list) and depth > 0:
                        for item in value:
                            if isinstance(item, dict):
                                _collect(item, depth - 1, flat)

            _collect(data, 2, flat)
            if flat:
                return flat
        return {}

    @staticmethod
    def _genome_shader_signature(genome: Any) -> dict[str, Any]:
        """Return a stable, shader-relevant signature for change detection.

        Returns:
            A dictionary representing the genome's shader signature, or empty dict if unavailable
        """
        for name in ("shader_signature", "to_dict"):
            member = getattr(genome, name, None)
            if member is None:
                continue
            try:
                signature = member() if callable(member) else member
            except Exception as e:
                logger.debug("Failed to read genome signature from %s: %s", name, e)
                continue
            return copy.deepcopy(signature)
        return {}

    @staticmethod
    def _graph_nodes(genome: Any) -> dict[str, dict[str, Any]] | None:
        """Map of ``node_id -> params dict`` for a graph genome, else None.

        Returns ``None`` when the genome has no graph layout so callers can
        tell "no such structure" apart from "structure present but empty".

        Returns:
            Dictionary mapping node IDs to parameter dictionaries, or None if no graph structure
        """
        nodes = getattr(getattr(genome, "graph", None), "nodes", None)
        if isinstance(nodes, dict):
            return {
                str(nid): node.params
                for nid, node in nodes.items()
                if isinstance(getattr(node, "params", None), dict)
            }
        if isinstance(nodes, list):
            result: dict[str, dict] = {}
            for node in nodes:
                store = getattr(node, "params", None)
                nid = getattr(node, "node_id", None)
                if isinstance(store, dict) and nid is not None:
                    result[str(nid)] = store
            return result
        return None

    @staticmethod
    def _graph_node_types(genome: Any) -> dict[str, str | None] | None:
        """Map graph node IDs to node types without exposing node objects.

        Returns:
            Dictionary mapping node IDs to node types, or None if no graph structure
        """
        nodes = getattr(getattr(genome, "graph", None), "nodes", None)
        if isinstance(nodes, dict):
            return {str(nid): getattr(node, "ntype", None) for nid, node in nodes.items()}
        if isinstance(nodes, list):
            return {
                str(getattr(node, "node_id", index)): getattr(node, "ntype", None)
                for index, node in enumerate(nodes)
            }
        return None

    def _write_genome_params(self, genome: Any, params: dict[str, float]) -> bool:
        """Write flat ``params`` into ``genome`` through its supported API.

        Returns True when at least one gene actually changed. Dispatches on
        the genome's layout instead of assuming a single ``params`` dict:

        * per-module dicts (``ModularGenome``, the pipeline's field pass):
          keys are written into every module dict that owns them.
        * graph nodes (``ShaderGraphGenome``): same, per node parameters.
        * a *settable* ``params`` property (``VisualGenome``): assigns the
          whole merged dict -- never ``params.update(...)``, which would
          mutate the temporary dict the getter just built.
        * otherwise: existing numeric attributes only, so no bogus attributes
          are invented on a genome that has no such gene.

        Returns:
            True if any parameters were changed, False otherwise
        """
        target = self._get_field_genome(genome)

        stores = [getattr(target, name, None) for name in self._NAMED_PARAM_STORES]
        if any(isinstance(store, dict) for store in stores):
            changed = False
            for store in stores:
                if isinstance(store, dict):
                    changed |= self._write_param_dict(store, params)
            return changed

        nodes = self._graph_nodes(target)
        if nodes is not None:
            changed = False
            for store in nodes.values():
                changed |= self._write_param_dict(store, params)
            return changed

        params_prop = getattr(type(target), "params", None)
        if isinstance(params_prop, property) and params_prop.fset is not None:
            try:
                current = getattr(target, "params", None)
            except (AttributeError, TypeError, ValueError):
                return False
            if not isinstance(current, dict):
                return False
            merged = dict(current)
            changed = False
            for key, value in params.items():
                if key not in merged:
                    continue
                old = merged[key]
                if isinstance(old, bool) or not isinstance(old, (int, float)):
                    continue
                new_value = self._coerce_like(old, value)
                if new_value != old:
                    merged[key] = new_value
                    changed = True
            if changed:
                target.params = merged  # property setter, not in-place update
            return changed

        changed = False
        for key, value in params.items():
            if not hasattr(target, key):
                continue
            old = getattr(target, key)
            if isinstance(old, bool) or not isinstance(old, (int, float)):
                continue
            new_value = self._coerce_like(old, value)
            if new_value != old:
                setattr(target, key, new_value)
                changed = True
        return changed

    def _copy_genome_params(self, target: Any, source: Any) -> bool:
        """Copy numeric genes from ``source`` into ``target`` structurally.

        Same-named parameter stores are paired first (ModularGenome module
        dicts, then graph nodes by id) so a gene such as ``scale`` from the
        champion's field module only lands in the target's *field* module.
        Falls back to a flat key-wise copy when the layouts differ (e.g. a
        ModularGenome champion feeding a ShaderGraphGenome or VisualGenome
        target).

        Returns:
            True if any parameters were changed, False otherwise
        """
        target = self._get_field_genome(target)
        changed = False
        for name in self._NAMED_PARAM_STORES:
            dst = getattr(target, name, None)
            src = getattr(source, name, None)
            if isinstance(dst, dict) and isinstance(src, dict):
                changed |= self._write_param_dict(dst, src)
        if changed:
            return True

        dst_nodes = self._graph_nodes(target)
        src_nodes = self._graph_nodes(source)
        if dst_nodes is not None and src_nodes is not None:
            changed = False
            matched_src_ids: set[str] = set()
            dst_types = self._graph_node_types(target) or {}
            src_types = self._graph_node_types(source) or {}
            for dst_id in sorted(dst_nodes):
                dst = dst_nodes[dst_id]
                src = src_nodes.get(dst_id)
                if not isinstance(src, dict):
                    dst_type = dst_types.get(dst_id)
                    src_id = next(
                        (
                            candidate_id
                            for candidate_id in sorted(src_types)
                            if candidate_id not in matched_src_ids
                            and src_types[candidate_id] == dst_type
                        ),
                        None,
                    )
                    src = src_nodes.get(src_id) if src_id is not None else None
                if isinstance(src, dict):
                    matched_src_ids.add(src_id) if src_id is not None else None
                    changed |= self._write_param_dict(dst, src)
            return changed

        return self._write_genome_params(target, self._flat_genome_params(source))

    def _write_param_dict(self, dst: dict[str, Any], src: dict[str, Any]) -> bool:
        """Copy matching numeric genes from ``src`` into ``dst`` in place.

        Returns:
            True if any parameters were changed, False otherwise
        """
        changed = False
        for key, value in src.items():
            if key not in dst or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue
            old = dst[key]
            if isinstance(old, bool) or not isinstance(old, (int, float)):
                continue
            new_value = self._coerce_like(old, value)
            if new_value != old:
                dst[key] = new_value
                changed = True
        return changed

    @staticmethod
    def _coerce_like(current: Any, value: Any) -> int | float:
        """Coerce ``value`` to the same int/float kind as ``current``.

        Returns:
            The coerced value as int or float
        """
        if isinstance(current, int) and not isinstance(current, bool):
            return round(float(value))
        return float(value)

    def _sync_camera_organisms(self) -> None:
        """Sync camera population births/deaths with the spatial ecosystem."""
        if self.ecosystem is None or self.camera_population is None:
            return
        try:
            existing_ids = {
                id(o)
                for o in self.ecosystem.organisms
                if hasattr(o, "genome") and hasattr(o.genome, "FLOAT_GENES")
            }
            for cam in self.camera_population.cameras:
                if cam.alive and id(cam) not in existing_ids:
                    cam.pos = [random.random(), random.random(), 0.0]
                    self.ecosystem.organisms.append(cam)
            dead = [
                o
                for o in self.ecosystem.organisms
                if hasattr(o, "genome") and hasattr(o.genome, "FLOAT_GENES") and not o.alive
            ]
            for o in dead:
                self.ecosystem.organisms.remove(o)
        except Exception as e:
            self._logger.debug(f"Camera organism sync failed: {e}")


if __name__ == "__main__":
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk

    class App(Gtk.Application):
        """Main application class."""

        def __init__(self):
            super().__init__(application_id="org.blackboxai.evo")
            self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        def do_activate(self):
            """Activate the application."""
            try:
                preset_dir = Path.home() / ".evo_renderer" / "presets"
                config = EvoConfig(
                    feedback_alpha=0.85,
                    feedback_res=512,
                    mutation_strength=0.1,
                    enable_logging=True,
                    preset_dir=preset_dir,
                    debug=False,
                    enable_spatial_ecosystem=True,
                    spatial_mode="territories",
                    spatial_blend="soft",
                    spatial_population_size=18,
                )
                win = Gtk.ApplicationWindow(application=self)
                win.set_default_size(1200, 800)
                win.set_title("EvoRenderer - Real-time Evolutionary Visualization")
                area = EvoRenderer(config)
                area.set_hexpand(True)
                area.set_vexpand(True)
                area.set_halign(Gtk.Align.FILL)
                area.set_valign(Gtk.Align.FILL)
                box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
                box.set_hexpand(True)
                box.set_vexpand(True)
                box.append(area.build_heatmap_overlay())
                win.set_child(box)
                win.present()
                self._logger.info("Application started successfully")
            except Exception as e:
                self._logger.error(f"Failed to activate application: {e}")
                raise

    app = App()
    exit_code = app.run(None)
    logger.info(f"Application exited with code: {exit_code}")
    sys.exit(exit_code)
