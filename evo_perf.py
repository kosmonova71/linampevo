#!/usr/bin/env python3

"""evo_perf.py

Adaptive performance subsystem for :class:`EvoRenderer`.

This module implements the "Long-Term Optimization Pipeline" from the renderer
performance spec, focusing on the techniques that fit a single-fullscreen-quad
feedback renderer:

  1.  Adaptive Quality System   - frame-budget controller (the centerpiece)
  2.  Multi-Resolution Rendering - reduced-resolution internal/feedback buffer
  4.  Dynamic Raymarch Budget   - runtime-capped raymarch / fractal iterations
  5.  Variable Feedback Resolution - feedback buffer decoupled from viewport
  6.  Shader LOD               - genome-derived quality tiers -> cheaper shaders
  8.  Persistent GPU Buffers   - owned elsewhere (FeedbackBuffer), reused here
  9.  Shader Permutation Cache - owned elsewhere (ShaderCache), reused here
  10. Dirty State Tracking      - UniformTracker only uploads changed uniforms
  11. Asynchronous Readback     - PBO ring for fitness readback (no glFinish)
  13. Adaptive Population Size  - population cap scales with available GPU time
  14. Separate Simulation/Rendering Rates - ecosystem steps at sim_hz, not FPS
  15. GPU Profiling             - per-stage CPU timers + frame report

Items 3 (tile-based) and 7 (GPU instancing) do not apply: there is a single
fullscreen quad (already one draw call) and the feedback effect cannot be tiled
without breaking the recursive trail accumulation.  Item 12 (texture atlas)
does not apply because all fields are procedural (no per-organism textures).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from OpenGL import GL

logger = logging.getLogger(__name__)


@dataclass
class PerfConfig:
    """Consolidated performance configuration with sensible defaults."""

    target_fps: float = 60.0
    up_threshold_ms: float | None = None  # auto = 0.7 * budget
    down_threshold_ms: float | None = None  # auto = 1.0 * budget
    adjustment_interval: float = 0.5
    consecutive_frames: int = 3

    enable_adaptive_population: bool = True

    min_feedback_scale: float = 0.4
    max_feedback_scale: float = 1.0
    min_internal_scale: float = 0.4
    max_internal_scale: float = 1.0
    min_raymarch_steps: int = 8
    max_raymarch_steps: int = 64
    min_fractal_layers: int = 2
    max_fractal_layers: int = 8
    min_particle_count: int = 2
    max_particle_count: int = 16
    min_blur_taps: int = 1
    max_blur_taps: int = 8
    min_lighting_samples: int = 2
    max_lighting_samples: int = 16
    min_shadow_samples: int = 1
    max_shadow_samples: int = 8
    min_bloom_passes: int = 1
    max_bloom_passes: int = 4
    min_dof_samples: int = 2
    max_dof_samples: int = 16
    min_feedback_history: int = 0
    max_feedback_history: int = 4


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _gl_check(where: str) -> None:
    """Check for GL errors and log them."""
    err = GL.glGetError()
    if err != GL.GL_NO_ERROR:
        logger.warning("GL error in %s: 0x%04X", where, err)


# ============================================================================
# 1. Adaptive Quality System (frame-budget controller)
# ============================================================================


def _resolve_config(cfg) -> PerfConfig:
    """Normalize arbitrary config object to PerfConfig, preserving any provided attrs."""
    if isinstance(cfg, PerfConfig):
        return cfg
    # Build from arbitrary object's attributes with fallback to defaults.
    defaults = PerfConfig()
    return PerfConfig(
        target_fps=float(getattr(cfg, "perf_target_fps", defaults.target_fps)),
        up_threshold_ms=getattr(cfg, "perf_up_threshold_ms", None),
        down_threshold_ms=getattr(cfg, "perf_down_threshold_ms", None),
        adjustment_interval=float(
            getattr(cfg, "perf_adjustment_interval", defaults.adjustment_interval)
        ),
        consecutive_frames=int(
            getattr(cfg, "perf_consecutive_frames", defaults.consecutive_frames)
        ),
        enable_adaptive_population=getattr(
            cfg, "perf_enable_adaptive_population", defaults.enable_adaptive_population
        ),
        min_feedback_scale=float(
            getattr(cfg, "perf_min_feedback_scale", defaults.min_feedback_scale)
        ),
        max_feedback_scale=float(
            getattr(cfg, "perf_max_feedback_scale", defaults.max_feedback_scale)
        ),
        min_internal_scale=float(
            getattr(cfg, "perf_min_internal_scale", defaults.min_internal_scale)
        ),
        max_internal_scale=float(
            getattr(cfg, "perf_max_internal_scale", defaults.max_internal_scale)
        ),
        min_raymarch_steps=int(
            getattr(cfg, "perf_min_raymarch_steps", defaults.min_raymarch_steps)
        ),
        max_raymarch_steps=int(
            getattr(cfg, "perf_max_raymarch_steps", defaults.max_raymarch_steps)
        ),
        min_fractal_layers=int(
            getattr(cfg, "perf_min_fractal_layers", defaults.min_fractal_layers)
        ),
        max_fractal_layers=int(
            getattr(cfg, "perf_max_fractal_layers", defaults.max_fractal_layers)
        ),
        min_particle_count=int(
            getattr(cfg, "perf_min_particle_count", defaults.min_particle_count)
        ),
        max_particle_count=int(
            getattr(cfg, "perf_max_particle_count", defaults.max_particle_count)
        ),
        min_blur_taps=int(getattr(cfg, "perf_min_blur_taps", defaults.min_blur_taps)),
        max_blur_taps=int(getattr(cfg, "perf_max_blur_taps", defaults.max_blur_taps)),
        min_lighting_samples=int(
            getattr(cfg, "perf_min_lighting_samples", defaults.min_lighting_samples)
        ),
        max_lighting_samples=int(
            getattr(cfg, "perf_max_lighting_samples", defaults.max_lighting_samples)
        ),
        min_shadow_samples=int(
            getattr(cfg, "perf_min_shadow_samples", defaults.min_shadow_samples)
        ),
        max_shadow_samples=int(
            getattr(cfg, "perf_max_shadow_samples", defaults.max_shadow_samples)
        ),
        min_bloom_passes=int(getattr(cfg, "perf_min_bloom_passes", defaults.min_bloom_passes)),
        max_bloom_passes=int(getattr(cfg, "perf_max_bloom_passes", defaults.max_bloom_passes)),
        min_dof_samples=int(getattr(cfg, "perf_min_dof_samples", defaults.min_dof_samples)),
        max_dof_samples=int(getattr(cfg, "perf_max_dof_samples", defaults.max_dof_samples)),
        min_feedback_history=int(
            getattr(cfg, "perf_min_feedback_history", defaults.min_feedback_history)
        ),
        max_feedback_history=int(
            getattr(cfg, "perf_max_feedback_history", defaults.max_feedback_history)
        ),
    )


class AdaptiveQualityController:
    """Drives every quality knob from a measured frame time.

    The controller maintains a single ``quality`` scalar in [0, 1] and derives
    all concrete settings from it.  It reacts to a smoothed (EMA) frame time
    with hysteresis (a cooldown between adjustments) so quality does not
    oscillate frame-to-frame.

    When ``quality == 1.0`` every derived setting is at its maximum, so the
    renderer produces exactly the same output it did before this subsystem
    existed -- the optimization is purely a *budget guard* until a frame exceeds
    the frame budget.
    """

    # Discrete Shader-LOD tiers (spec 6).  Higher tier == more shader work.
    LOD_TIERS = 5  # tiers 0..4

    def __init__(self, cfg):
        self.cfg = _resolve_config(cfg)
        c = self.cfg
        self.target_fps = c.target_fps
        self.frame_budget_ms = 1000.0 / max(1.0, self.target_fps)
        # Scale thresholds from frame budget if not explicitly configured.
        # up = 0.7x budget (quality rises when well under budget)
        # down = 1.0x budget (quality drops when at/over budget)
        self.up_threshold_ms = (
            c.up_threshold_ms if c.up_threshold_ms is not None else self.frame_budget_ms * 0.7
        )
        self.down_threshold_ms = (
            c.down_threshold_ms if c.down_threshold_ms is not None else self.frame_budget_ms * 1.0
        )
        self.adjustment_interval = c.adjustment_interval
        # Require N consecutive samples on one side before adjusting (hysteresis).
        self._consecutive_required = c.consecutive_frames
        self._consecutive_under = 0
        self._consecutive_over = 0

        self.quality = 1.0
        # Smoothed frame / gpu time.
        self._ema_frame_ms = self.frame_budget_ms
        self._ema_gpu_ms = self.frame_budget_ms
        self._last_adjust = 0.0

        # Adaptive population factor (spec 13).
        self.population_scale = 1.0

        # Derived settings, recomputed from quality.
        self.feedback_scale = 1.0
        self.internal_scale = 1.0
        self.raymarch_steps = c.max_raymarch_steps
        self.fractal_layers = c.max_fractal_layers
        self.lod_tier = self.LOD_TIERS - 1
        self.bloom_quality = 1.0
        self.glow_quality = 1.0
        self.particle_scale = 1.0
        self.particle_count = c.max_particle_count
        self.blur_taps = c.max_blur_taps
        self.lighting_samples = c.max_lighting_samples
        self.shadow_samples = c.max_shadow_samples
        self.bloom_passes = c.max_bloom_passes
        self.dof_samples = c.max_dof_samples
        self.feedback_history = c.max_feedback_history

        self._recompute()

    # -- public API ---------------------------------------------------------

    def update(self, frame_ms: float, gpu_ms: float | None = None) -> None:
        """Feed one frame's measured time; adjust quality if due."""
        a = 0.2
        self._ema_frame_ms = self._ema_frame_ms * (1.0 - a) + frame_ms * a
        if gpu_ms is not None and gpu_ms > 0.0:
            self._ema_gpu_ms = self._ema_gpu_ms * (1.0 - a) + gpu_ms * a

        now = time.time()
        if now - self._last_adjust < self.adjustment_interval:
            return

        ms = self._ema_frame_ms
        if ms > self.down_threshold_ms:
            self._consecutive_over += 1
            self._consecutive_under = 0
            if self._consecutive_over >= self._consecutive_required:
                self.quality = max(0.0, self.quality - 0.1)
                self._recompute()
                self._last_adjust = now
                if self.cfg.enable_adaptive_population:
                    self.population_scale = max(0.25, self.population_scale - 0.1)
        elif ms < self.up_threshold_ms:
            self._consecutive_under += 1
            self._consecutive_over = 0
            if self._consecutive_under >= self._consecutive_required:
                self.quality = min(1.0, self.quality + 0.05)
                self._recompute()
                self._last_adjust = now
                if self.cfg.enable_adaptive_population:
                    self.population_scale = min(1.0, self.population_scale + 0.05)
        else:
            self._consecutive_under = 0
            self._consecutive_over = 0

    def effective_population_cap(self, base_cap: int) -> int:
        """Return the population cap scaled by available GPU headroom (spec 13)."""
        if not self.cfg.enable_adaptive_population:
            return base_cap
        return max(2, round(base_cap * self.population_scale))

    def apply(self, renderer) -> None:
        """Push derived quality settings to a renderer instance.

        Expected renderer attributes (create if missing):
        - feedback_scale, internal_scale (float)
        - max_raymarch_steps, max_fractal_layers (int)
        - lod_tier (int)
        - bloom_quality, glow_quality, particle_scale (float)
        - particle_count, blur_taps, lighting_samples, shadow_samples (int)
        - bloom_passes, dof_samples, feedback_history (int)
        """
        renderer.feedback_scale = self.feedback_scale
        renderer.internal_scale = self.internal_scale
        renderer.max_raymarch_steps = self.raymarch_steps
        renderer.max_fractal_layers = self.fractal_layers
        renderer.lod_tier = self.lod_tier
        renderer.bloom_quality = self.bloom_quality
        renderer.glow_quality = self.glow_quality
        renderer.particle_scale = self.particle_scale
        renderer.particle_count = self.particle_count
        renderer.blur_taps = self.blur_taps
        renderer.lighting_samples = self.lighting_samples
        renderer.shadow_samples = self.shadow_samples
        renderer.bloom_passes = self.bloom_passes
        renderer.dof_samples = self.dof_samples
        renderer.feedback_history = self.feedback_history

    def report(self) -> dict[str, float]:
        """Return the human-readable "Frame report" values (spec 15)."""
        return {
            "quality": round(self.quality, 3),
            "feedback_scale": round(self.feedback_scale, 3),
            "internal_scale": round(self.internal_scale, 3),
            "raymarch_steps": self.raymarch_steps,
            "fractal_layers": self.fractal_layers,
            "lod_tier": self.lod_tier,
            "pop_scale": round(self.population_scale, 3),
            "particle_count": self.particle_count,
            "blur_taps": self.blur_taps,
            "lighting_samples": self.lighting_samples,
            "shadow_samples": self.shadow_samples,
            "bloom_passes": self.bloom_passes,
            "dof_samples": self.dof_samples,
            "feedback_history": self.feedback_history,
            "frame_ms": round(self._ema_frame_ms, 2),
            "gpu_ms": round(self._ema_gpu_ms, 2),
            "budget_ms": round(self.frame_budget_ms, 2),
        }

    # -- internals ----------------------------------------------------------

    def _recompute(self) -> None:
        q = self.quality
        c = self.cfg
        self.feedback_scale = _lerp(c.min_feedback_scale, c.max_feedback_scale, q)
        self.internal_scale = _lerp(c.min_internal_scale, c.max_internal_scale, q)
        self.raymarch_steps = round(
            _lerp(float(c.min_raymarch_steps), float(c.max_raymarch_steps), q)
        )
        self.fractal_layers = round(
            _lerp(float(c.min_fractal_layers), float(c.max_fractal_layers), q)
        )
        self.lod_tier = max(0, min(self.LOD_TIERS - 1, round(q * (self.LOD_TIERS - 1))))
        self.bloom_quality = _lerp(0.25, 1.0, q)
        self.glow_quality = _lerp(0.25, 1.0, q)
        self.particle_scale = _lerp(0.3, 1.0, q)
        self.particle_count = round(
            _lerp(float(c.min_particle_count), float(c.max_particle_count), q)
        )
        self.blur_taps = round(_lerp(float(c.min_blur_taps), float(c.max_blur_taps), q))
        self.lighting_samples = round(
            _lerp(float(c.min_lighting_samples), float(c.max_lighting_samples), q)
        )
        self.shadow_samples = round(
            _lerp(float(c.min_shadow_samples), float(c.max_shadow_samples), q)
        )
        self.bloom_passes = round(_lerp(float(c.min_bloom_passes), float(c.max_bloom_passes), q))
        self.dof_samples = round(_lerp(float(c.min_dof_samples), float(c.max_dof_samples), q))
        self.feedback_history = round(
            _lerp(float(c.min_feedback_history), float(c.max_feedback_history), q)
        )


# ============================================================================
# 15. GPU Profiling (per-stage CPU timers)
# ============================================================================


class _StageTimer:
    def __init__(self, name: str):
        self.name = name
        self._ema = 0.0
        self._t0: float | None = None

    def begin(self) -> None:
        self._t0 = time.perf_counter()

    def end(self) -> None:
        if self._t0 is None:
            return
        dt = (time.perf_counter() - self._t0) * 1000.0
        self._t0 = None
        self._ema += (dt - self._ema) * 0.2

    def value(self) -> float:
        return self._ema


class GpuProfiler:
    """Measures where time is spent each frame (spec 15).

    Uses lightweight CPU-side timers (perf_counter) around CPU-side stage
    wrappers.  No GL timer-query extension dependency, so it works on every
    driver.  Note: these are CPU-side measurements of the time spent
    *submitting* work to the GPU, not GPU execution time.  For actual GPU
    elapsed time, timer queries (GL_EXT_disjoint_timer_query) would be needed.
    Each stage is wrapped with :meth:`stage`, which returns a context manager.
    """

    STAGES = ("geometry", "organisms", "feedback", "reaction", "bloom", "composite", "readback")

    def __init__(self):
        self._stages = {k: _StageTimer(k) for k in self.STAGES}
        self._idle_ema = 0.0

    def begin(self, name: str) -> None:
        t = self._stages.setdefault(name, _StageTimer(name))
        t.begin()

    def end(self, name: str) -> None:
        t = self._stages.setdefault(name, _StageTimer(name))
        t.end()

    class _StageCtx:
        def __init__(self, profiler: GpuProfiler, name: str):
            self._p = profiler
            self._name = name

        def __enter__(self):
            self._p.begin(self._name)
            return self

        def __exit__(self, *exc):
            self._p.end(self._name)
            return False

    def stage(self, name: str) -> _StageCtx:
        return self._StageCtx(self, name)

    def frame_report(self, total_ms: float) -> dict[str, float]:
        """Produce the "Frame report" showing ms spent per stage (spec 15)."""
        d: dict[str, float] = {k: self._stages[k].value() for k in self.STAGES}
        used = sum(d.values())
        self._idle_ema += (max(0.0, total_ms - used) - self._idle_ema) * 0.2
        d["idle"] = self._idle_ema
        d["total"] = total_ms
        return d

    def format_report(self, total_ms: float) -> str:
        d = self.frame_report(total_ms)
        parts = [f"Frame: {d['total']:.1f} ms"]
        for k in self.STAGES:
            parts.append(f"{k.capitalize():>10}: {d[k]:.1f} ms")
        parts.append(f"{'Idle':>10}: {d['idle']:.1f} ms")
        return "\n".join(parts)


# ============================================================================
# 10. Dirty State Tracking (uniform tracker)
# ============================================================================


class UniformTracker:
    """Only uploads a uniform when its value actually changed (spec 10).

    Uniform locations are program-specific, so the cache must be cleared
    whenever a different program is bound (call :meth:`reset`).
    """

    _FLOAT_EPS = 1e-6

    def __init__(self):
        self._cache: dict[int, object] = {}

    def reset(self) -> None:
        self._cache.clear()

    def set1f(self, loc: int, v: float) -> None:
        if loc < 0:
            return
        prev = self._cache.get(loc)
        if prev is None or abs(prev - v) > self._FLOAT_EPS:
            GL.glUniform1f(loc, float(v))
            self._cache[loc] = v

    def set2f(self, loc: int, x: float, y: float) -> None:
        if loc < 0:
            return
        key = (x, y)
        prev = self._cache.get(loc)
        if (
            prev is None
            or abs(prev[0] - x) > self._FLOAT_EPS
            or abs(prev[1] - y) > self._FLOAT_EPS
        ):
            GL.glUniform2f(loc, float(x), float(y))
            self._cache[loc] = key

    def set3f(self, loc: int, x: float, y: float, z: float) -> None:
        if loc < 0:
            return
        key = (x, y, z)
        prev = self._cache.get(loc)
        if (
            prev is None
            or abs(prev[0] - x) > self._FLOAT_EPS
            or abs(prev[1] - y) > self._FLOAT_EPS
            or abs(prev[2] - z) > self._FLOAT_EPS
        ):
            GL.glUniform3f(loc, float(x), float(y), float(z))
            self._cache[loc] = key

    def set1i(self, loc: int, v: int) -> None:
        if loc < 0:
            return
        prev = self._cache.get(loc)
        if prev != v:
            GL.glUniform1i(loc, int(v))
            self._cache[loc] = v


# ============================================================================
# 11. Asynchronous Readback (PBO ring)
# ============================================================================


class AsyncReadback:
    """Non-blocking fitness readback using a triple-PBO ring (spec 11).

    Triple-buffered PBO pattern: each frame the current framebuffer is
    copied into one PBO (``glReadPixels`` into a bound PBO returns
    immediately, the DMA happens on the GPU), while the CPU maps the
    PBO from two frames ago, which the GPU has long since finished
    filling.  This removes the blocking ``glFinish`` / synchronous read
    the previous code used, so the GPU keeps rendering while the CPU
    scores fitness on stale-by-two data.

    ``collect()`` must be called *before* ``read_into_pbo()`` so the
    two operations always touch opposite buffers (the map reads
    last-previous-frame's data, the copy schedules this frame's), which
    is what keeps the CPU from stalling on an in-flight GPU copy.
    """

    def __init__(self, res: int):
        self.res = max(1, int(res))
        self._pbos: list[int] = []
        self._write_idx = 0
        self._has_data = False
        self._last_pixels: bytes | None = b""
        self.gl_ready = False

    def init_gl(self) -> None:
        try:
            self._pbos = list(GL.glGenBuffers(3))
            nbytes = self.res * self.res * 4
            for p in self._pbos:
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, p)
                GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, nbytes, None, GL.GL_STREAM_READ)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            _gl_check("AsyncReadback.init_gl")
            self.gl_ready = True
        except Exception as _e:
            logger.warning("AsyncReadback init failed: %s", _e)
            self.gl_ready = False

    def cleanup(self) -> None:
        if self._pbos:
            try:
                GL.glDeleteBuffers(len(self._pbos), self._pbos)
            except Exception as _e:
                logger.warning("PBO cleanup failed: %s", _e)
        self._pbos = []
        self.gl_ready = False

    def read_into_pbo(self) -> None:
        """Schedule an async copy of the currently-bound FBO into a PBO.

        Call *after* ``collect()`` and after the fitness FBO / render
        target is bound and drawn.  Safe to call when GL is
        uninitialized (no-op).
        """
        if not self.gl_ready or len(self._pbos) != 3:
            return
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbos[self._write_idx % 3])
        GL.glReadPixels(0, 0, self.res, self.res, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, 0)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        _gl_check("AsyncReadback.read_into_pbo")
        self._write_idx += 1
        self._has_data = True

    def collect(self) -> bytes | None:
        """Return the pixels written two frames ago (or None on the first frames).

        This is the CPU side of the triple-PBO pattern: map the buffer
        holding last-previous-frame's data, which the GPU has long since
        finished filling.
        """
        if not self.gl_ready or len(self._pbos) != 3 or not self._has_data:
            return self._last_pixels
        if self._write_idx < 2:
            return self._last_pixels
        nbytes = self.res * self.res * 4
        buf = (GL.GLubyte * nbytes)()
        read_idx = (self._write_idx - 2) % 3
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbos[read_idx])
        try:
            GL.glGetBufferSubData(GL.GL_PIXEL_PACK_BUFFER, 0, nbytes, buf)
            data = bytes(buf)
        except Exception as _e:
            data = self._last_pixels
            logger.warning("PBO read failed: %s", _e)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        _gl_check("AsyncReadback.collect")
        self._last_pixels = data
        return data


# ============================================================================
# 6. Shader LOD selector
# ============================================================================


class ShaderLOD:
    """Maps a quality tier to the shader-cost caps used by the factories.

    Lower tiers bake fewer loop iterations (cheaper programs, cached per tier
    by :class:`ShaderCache` so identical tiers reuse a compiled program).

    Note: :class:`AdaptiveQualityController` already derives raymarch_steps
    and fractal_layers from its quality scalar.  This class exists for cases
    where a tier is chosen directly (e.g. by genome complexity) rather than
    through the controller.
    """

    TIER_FRACTIONS = (0.25, 0.45, 0.65, 0.85, 1.0)

    @staticmethod
    def caps_for_tier(tier: int, max_ray: int = 64, max_fractal: int = 8) -> dict[str, int]:
        t = max(0, min(4, int(tier)))
        frac = ShaderLOD.TIER_FRACTIONS[t]
        return {
            "raymarch_steps": max(8, round(max_ray * frac)),
            "fractal_layers": max(2, round(max_fractal * frac)),
        }


# ============================================================================
# Asynchronous shader compilation (spec 6: never stall rendering on a mutation)
# ============================================================================
#
#   render thread                worker thread
#   -------------                -------------
#   request(key, gen_source)  ->  gen_source()   (CPU-heavy GLSL text build)
#        (returns instantly)          |
#   ...keep rendering current...      v
#   poll_ready()             <-  (key, source)
#   upload_program(source)   (fast GL calls, on the GL context thread)
#   swap shader next frame
#
# The expensive part of "compiling a shader" here is *generating the GLSL
# source*: the modular factory / shader-graph compiler run Python loops that
# build large strings (FBM unrolling, graph walks, palette assembly).  That is
# pure CPU work with no GL calls, so it is safe to run on a worker thread while
# the render thread keeps drawing the organism's previous program.
#
# Creating the actual GL program object (glCompileShader / glLinkProgram) MUST
# happen on the thread that owns the GL context, so that step stays on the
# render thread -- but it is comparatively cheap and, crucially, only runs once
# the source is already built off-thread.  If a genome's program is not ready
# yet, the caller simply renders whatever it rendered last frame, so visuals
# never freeze.


class AsyncShaderCompiler:
    """Off-thread GLSL source generation with next-frame GL swap.

    Usage from the render thread:

        # Ask for a program (returns None until it is ready):
        prog = compiler.request(key, lambda: gen_source(genome))

        # Once per frame, turn finished source into GL programs:
        for key, program in compiler.poll_ready(upload_fn):
            shader_cache.store(key, program)

    ``request`` never blocks: if the program for ``key`` is already built it is
    returned immediately; otherwise the source generation is queued on the
    worker thread and ``None`` is returned so the caller keeps using its current
    program.  The same ``key`` is never queued twice while a build is in flight.
    """

    def __init__(self, max_pending: int = 64, max_ready: int = 128):
        # Source-generation jobs waiting for / running on the worker thread.
        self._requests: queue.Queue[tuple[int, Callable[[], str]] | None] = queue.Queue()
        # Finished GLSL source ready for GL upload on the render thread.
        self._ready: queue.Queue[tuple[int, str | None, str | None]] = queue.Queue(
            maxsize=max_ready
        )
        # Keys with a build currently queued or running (dedup guard).
        self._in_flight: dict[int, bool] = {}
        self._lock = threading.Lock()
        self._max_pending = max(1, int(max_pending))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stats_submitted = 0
        self._stats_completed = 0
        self._stats_failed = 0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="shader-compiler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to exit and wait briefly for it to drain."""
        self._stop.set()
        # Unblock the worker if it is parked on an empty queue.
        try:
            self._requests.put_nowait(None)
        except Exception as _e:
            logger.warning("Queue put failed: %s", _e)
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)
        self._thread = None
        with self._lock:
            self._in_flight.clear()
        # Drain remaining queues for deterministic teardown.
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._ready.get_nowait()
            except queue.Empty:
                break

    # -- render-thread API --------------------------------------------------

    def request(self, key: int, gen_source: Callable[[], str]) -> None:
        """Queue ``gen_source`` for ``key`` unless it is already in flight.

        This is fire-and-forget: the render thread should have already checked
        its shader cache for ``key`` and only call this when a compile is
        actually needed.  Returns immediately; the result surfaces later via
        :meth:`poll_ready`.
        """
        key = int(key)
        with self._lock:
            if key in self._in_flight:
                return
            if len(self._in_flight) >= self._max_pending:
                # Back-pressure: drop the request. The render thread keeps its
                # current program; the genome will be re-requested next frame
                # once the queue drains, so nothing is lost permanently.
                return
            self._in_flight[key] = True
            self._stats_submitted += 1
        # Ensure the worker exists even if start() was forgotten.
        self.start()
        self._requests.put((key, gen_source))

    def poll_ready(
        self,
        upload_fn: Callable[[str], int],
        max_uploads: int = 4,
    ) -> list[tuple[int, int]]:
        """Drain finished source and upload it to GL on the calling thread.

        ``upload_fn(source) -> program_id`` performs the GL-context-bound
        compile/link and must be called from the render thread. Returns a list
        of ``(key, program_id)`` that were successfully uploaded this frame so
        the caller can store them in its shader cache and swap.

        ``max_uploads`` bounds how many programs are uploaded per frame so a
        burst of finished shaders cannot itself cause a frame spike.
        """
        uploaded: list[tuple[int, int]] = []
        for _ in range(max(1, int(max_uploads))):
            if self._ready.empty():
                break
            key = None
            try:
                key, source, err = self._ready.get_nowait()
            except queue.Empty:
                break
            except Exception as _e:
                logger.warning("Queue get failed: %s", _e)
            try:
                if err is not None or source is None:
                    self._stats_failed += 1
                    continue
                program = upload_fn(source)
                if program:
                    uploaded.append((key, int(program)))
                    self._stats_completed += 1
                else:
                    self._stats_failed += 1
            finally:
                if key is not None:
                    with self._lock:
                        self._in_flight.pop(key, None)
        return uploaded

    def pending(self) -> int:
        """Number of keys with a build queued or running."""
        with self._lock:
            return len(self._in_flight)

    def is_pending(self, key: int) -> bool:
        """True if ``key`` already has a build queued or running.

        Lets the caller skip the (potentially expensive) work of snapshotting a
        genome for a request that would be dropped as a duplicate anyway.
        """
        with self._lock:
            return int(key) in self._in_flight

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "submitted": self._stats_submitted,
                "completed": self._stats_completed,
                "failed": self._stats_failed,
                "in_flight": len(self._in_flight),
            }

    # -- worker -------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            item = self._requests.get()
            if item is None:
                break
            key, gen_source = item
            source: str | None = None
            err: str | None = None
            try:
                source = gen_source()
            except Exception as exc:  # keep the worker alive on bad genomes
                err = str(exc)
                logger.debug("async shader source generation failed: %s", exc)
            if err is None and not source:
                err = "empty shader source"
            # If _ready is full, drop oldest finished job to make room.
            try:
                self._ready.put_nowait((key, source, err))
            except queue.Full:
                try:
                    self._ready.get_nowait()  # drop oldest
                except queue.Empty:
                    pass
                try:
                    self._ready.put_nowait((key, source, err))
                except queue.Full:
                    pass  # give up under extreme pressure
