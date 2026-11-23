#!/usr/bin/env python3

"""evo_gpu_pool.py
GPU resource pooling and memory-budget helpers for the evolutionary renderer.

This module addresses the six GPU-memory improvements from the OOM post-mortem:

  * Texture pooling      -- :class:`TexturePool` recycles RGBA16F textures so an
                            organism's per-frame resize does not leak a texture
                            pair every time the viewport changes.
  * Framebuffer pooling  -- :class:`FboPool` pairs pooled textures with pooled
                            framebuffers behind ``acquire_fbo_with_texture`` so
                            the persistent-world, feedback and replay passes all
                            draw from one shared, bounded pool of GL objects.
  * Recycle replay buffers -- :class:`ReplayRingBuffer` keeps a fixed window of
                            encoded replay frames in a ring (no unbounded list
                            growth) and recycles the backing storage when the
                            window shrinks.
  * Adaptive history length -- :class:`AdaptiveHistoryController` shrinks the
                            feedback history, replay window, lineage log and
                            novelty history as GPU/process memory pressure rises,
                            so a long session cannot accumulate forever.
  * Memory watchdog proactive trim -- :class:`GpuMemoryBudget` centralises the
                            per-frame budget check and *acts* (trims the shader
                            cache, trims lineage, shrinks history) instead of
                            only logging.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from typing import Any

from OpenGL import GL

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Texture pooling
# ----------------------------------------------------------------------------


class TexturePool:
    """Recycles GL textures of a fixed internal format to avoid churn.

    Every allocation/deletion of an RGBA16F texture is a GPU bookkeeping cost
    and, under resize storms (window drag, GTK scale-factor changes), the old
    code allocated a fresh pair each time and deleted the previous one -- which
    is exactly the kind of churn that leaves orphaned GL objects around. The
    pool hands back a previously-liberated texture when one is available, and
    only calls ``glGenTextures`` when the free list is empty.

    **Contract — GL-thread only.** All methods issue GL calls and must be
    called from the thread that owns the GL context.  Callers should release
    every texture via :meth:`release` before calling :meth:`drain`; if textures
    are still checked-out at drain time they are deleted and a warning is
    logged.

    ``max_free`` caps the number of recycled textures the free list will hold.
    When the cap is exceeded the oldest excess textures are deleted with
    ``glDeleteTextures`` rather than retained forever, bounding GPU memory under
    resize storms that produce many size variants.
    """

    def __init__(
        self,
        internal_format: int = GL.GL_RGBA16F,
        src_format: int = GL.GL_RGBA,
        src_type: int = GL.GL_HALF_FLOAT,
        min_filter: int = GL.GL_LINEAR,
        mag_filter: int = GL.GL_LINEAR,
        wrap: int = GL.GL_CLAMP_TO_EDGE,
        max_free: int = 32,
    ):
        self._internal = internal_format
        self._src_format = src_format
        self._src_type = src_type
        self._min_filter = min_filter
        self._mag_filter = mag_filter
        self._wrap = wrap
        self._free: list[int] = []
        self._max_free: int = max(1, int(max_free))
        # All textures ever allocated by this pool (both free and checked-out).
        # The live count is derived from this set minus the free list so that
        # double-release cannot drive _live negative.
        self._allocated: set[int] = set()
        self._max_live: int = 0

    @property
    def _live(self) -> int:
        """Textures currently checked out (not in the free list)."""
        return len(self._allocated) - len(self._free)

    def acquire(self, width: int, height: int) -> int:
        """Return a texture sized ``width x height`` (recycled if possible)."""
        if self._free:
            tex = self._free.pop()
        else:
            tex = int(GL.glGenTextures(1))
            self._allocated.add(tex)
        self._max_live = max(self._max_live, self._live)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            self._internal,
            int(width),
            int(height),
            0,
            self._src_format,
            self._src_type,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, self._min_filter)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, self._mag_filter)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, self._wrap)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, self._wrap)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return int(tex)

    def release(self, tex: int | None) -> None:
        """Return ``tex`` to the pool for reuse (does not delete the GL object).

        Safe against double-release: if ``tex`` is already in the free list it
        is ignored (a debug message is logged) so the live count stays
        consistent.
        """
        if tex is None:
            return
        tex = int(tex)
        if tex not in self._allocated:
            logger.debug("TexturePool.release: texture %d not tracked by this pool", tex)
            return
        if tex in self._free:
            logger.debug(
                "TexturePool.release: texture %d already in free list (double-release guard)", tex
            )
            return
        self._free.append(tex)
        self._trim_free()

    def _trim_free(self) -> None:
        """Delete oldest free textures so ``len(self._free) <= self._max_free``."""
        while len(self._free) > self._max_free:
            excess = self._free.pop(0)
            self._allocated.discard(excess)
            try:
                GL.glDeleteTextures(1, [excess])
            except Exception:
                logger.debug("GL texture deletion failed", exc_info=True)

    def drain(self) -> None:
        """Delete every pooled GL texture (call on context teardown).

        Deletes both free and still-checked-out textures.  If any textures are
        still live (not yet released) a warning is logged — the recommended
        contract is to release all textures before calling ``drain``.
        """
        live = self._live
        if live > 0:
            logger.warning(
                "TexturePool.drain: %d texture(s) still live (not released) "
                "— caller should release before drain",
                live,
            )
        for tex in self._allocated:
            try:
                GL.glDeleteTextures(1, [tex])
            except Exception:
                logger.debug("GL texture teardown failed", exc_info=True)
        self._free = []
        self._allocated = set()

    def stats(self) -> dict[str, int]:
        return {
            "free": len(self._free),
            "live": self._live,
            "max_live": self._max_live,
            "allocated": len(self._allocated),
        }


class FboPool:
    """Pools framebuffers + paired textures so passes share one bounded pool.

    ``acquire_fbo_with_texture`` hands back an ``(fbo, tex)`` pair sized to the
    request; ``release`` returns the pair to the free lists so the next acquirer
    reuses the GL objects. This is what lets :class:`FeedbackBuffer`,
    :class:`PersistentWorldTextures` and replay passes all stop allocating
    fresh GL objects on every resize / new organism.

    The pool is keyed by framebuffer object (the texture travels with it) so a
    released pair is re-issued intact, avoiding format/attachment mismatch.

    **Contract — GL-thread only.** All methods issue GL calls and must be
    called from the thread that owns the GL context.  Callers must release
    every ``(fbo, tex)`` pair via :meth:`release` before calling
    :meth:`drain`; any FBO still checked-out at drain time is deleted and a
    warning is logged.  Once a pair is handed out, never call
    ``TexturePool.release`` on its texture directly — always return the whole
    pair via :meth:`FboPool.release` so the fbo<->tex association stays coherent.

    ``max_free`` caps both the free-FBO list and the underlying
    :class:`TexturePool` free list, bounding GPU memory under resize storms.
    """

    def __init__(
        self,
        internal_format: int = GL.GL_RGBA16F,
        src_format: int = GL.GL_RGBA,
        src_type: int = GL.GL_HALF_FLOAT,
        max_free: int = 32,
    ):
        self._tex_pool = TexturePool(
            internal_format=internal_format,
            src_format=src_format,
            src_type=src_type,
            max_free=max_free,
        )
        self._free_fbos: list[int] = []
        self._max_free: int = max(1, int(max_free))
        # All FBOs ever allocated by this pool (free + checked-out).
        self._allocated_fbos: set[int] = set()
        self._max_live_fbos: int = 0
        # Keep the tex<->fbo association so a released pair stays coherent.
        self._pairs: dict[int, int] = {}

    @property
    def _live_fbos(self) -> int:
        """FBOs currently checked out (not in the free list)."""
        return len(self._allocated_fbos) - len(self._free_fbos)

    def acquire_fbo_with_texture(self, width: int, height: int) -> tuple[int, int]:
        """Return ``(fbo, tex)`` sized to ``width x height``.

        Reuses a released pair when available, otherwise allocates a new
        framebuffer and draws its texture from the shared :class:`TexturePool`.
        """
        if self._free_fbos:
            fbo = self._free_fbos.pop()
            tex = self._pairs.get(fbo)
            if tex is None:
                # Pair bookkeeping was lost — acquire a fresh texture.
                tex = self._tex_pool.acquire(width, height)
            else:
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    self._tex_pool._internal,
                    int(width),
                    int(height),
                    0,
                    self._tex_pool._src_format,
                    self._tex_pool._src_type,
                    None,
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, self._tex_pool._min_filter
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, self._tex_pool._mag_filter
                )
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, self._tex_pool._wrap)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, self._tex_pool._wrap)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
                GL.glFramebufferTexture2D(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_COLOR_ATTACHMENT0,
                    GL.GL_TEXTURE_2D,
                    tex,
                    0,
                )
                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                if status != GL.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError(f"Pooled FBO incomplete after resize: {hex(status)}")
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            # Always restore the pairing, even when tex was None (the branch
            # above acquired a fresh texture that was never paired with this
            # fbo before).
            self._pairs[int(fbo)] = int(tex)
        else:
            fbo = int(GL.glGenFramebuffers(1))
            self._allocated_fbos.add(fbo)
            self._max_live_fbos = max(self._max_live_fbos, self._live_fbos + 1)
            tex = self._tex_pool.acquire(width, height)
            self._pairs[fbo] = int(tex)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D,
                tex,
                0,
            )
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                # Clean up the newly allocated FBO and texture before the
                # exception propagates to avoid leaking on rare driver failures.
                self._allocated_fbos.discard(fbo)
                del self._pairs[fbo]
                self._tex_pool.release(tex)
                try:
                    GL.glDeleteFramebuffers(1, [fbo])
                except Exception:
                    logger.debug("GL framebuffer deletion on error failed", exc_info=True)
                raise RuntimeError(f"Pooled FBO incomplete: {hex(status)}")
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return int(fbo), int(tex)

    def release(self, fbo: int | None, tex: int | None) -> None:
        """Return an ``(fbo, tex)`` pair to the pool.

        The paired texture stays attached to its FBO and travels with it on
        the next :meth:`acquire_fbo_with_texture` call.  Only if ``tex``
        differs from the paired texture (orphaned / foreign) is it returned
        to the :class:`TexturePool` separately.

        Double-release is guarded: if ``fbo`` is already in the free list it
        is ignored (a debug message is logged).
        """
        if fbo is None:
            if tex is not None:
                self._tex_pool.release(tex)
            return
        fbo = int(fbo)
        if fbo not in self._allocated_fbos:
            logger.debug("FboPool.release: FBO %d not tracked by this pool", fbo)
            return
        if fbo in self._free_fbos:
            logger.debug(
                "FboPool.release: FBO %d already in free list (double-release guard)", fbo
            )
            return
        self._free_fbos.append(fbo)
        self._trim_free_fbos()
        # The texture stays attached to its fbo; release it only if it is not
        # tracked with the pair (orphaned), otherwise it is reused as a unit.
        paired_tex = self._pairs.get(fbo)
        if tex is not None and paired_tex != int(tex):
            self._tex_pool.release(tex)

    def _trim_free_fbos(self) -> None:
        """Delete oldest free FBOs so ``len(self._free_fbos) <= self._max_free``."""
        while len(self._free_fbos) > self._max_free:
            excess = self._free_fbos.pop(0)
            self._allocated_fbos.discard(excess)
            paired_tex = self._pairs.pop(excess, None)
            if paired_tex is not None:
                self._tex_pool.release(paired_tex)
            try:
                GL.glDeleteFramebuffers(1, [excess])
            except Exception:
                logger.debug("GL framebuffer deletion failed", exc_info=True)

    def drain(self) -> None:
        """Delete every pooled framebuffer and texture (context teardown).

        Deletes both free and still-checked-out FBOs (and their paired
        textures).  If any FBOs are still live (not yet released) a warning is
        logged — the recommended contract is to release all pairs before
        calling ``drain``.
        """
        live = self._live_fbos
        if live > 0:
            logger.warning(
                "FboPool.drain: %d FBO(s) still live (not released) "
                "— caller should release before drain",
                live,
            )
        # Delete every allocated FBO (free + live).
        for fbo in self._allocated_fbos:
            try:
                GL.glDeleteFramebuffers(1, [fbo])
            except Exception:
                logger.debug("GL framebuffer teardown failed", exc_info=True)
        # Release all paired textures to the TexturePool, then drain it.
        for tex in self._pairs.values():
            self._tex_pool.release(tex)
        self._free_fbos = []
        self._allocated_fbos = set()
        self._pairs = {}
        self._tex_pool.drain()

    def stats(self) -> dict[str, int]:
        return {
            "free_fbos": len(self._free_fbos),
            "live_fbos": self._live_fbos,
            "max_live_fbos": self._max_live_fbos,
            **self._tex_pool.stats(),
        }


# ----------------------------------------------------------------------------
# Recycle replay buffers
# ----------------------------------------------------------------------------


class ReplayRingBuffer:
    """Bounded ring of encoded replay frames (spec 28 replay events).

    The previous history approach appended to an unbounded deque / list, so a
    long session kept accumulating replay frames in memory. This ring keeps at
    most ``maxlen`` frames and recycles a compact ``bytes`` payload per slot:
    when the window shrinks, the surplus slots are dropped immediately and
    their bytes are allowed to be garbage-collected (no retained buffer list).
    """

    def __init__(self, maxlen: int = 256):
        self._maxlen = max(1, int(maxlen))
        self._frames: deque = deque(maxlen=self._maxlen)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def set_maxlen(self, maxlen: int) -> None:
        """Shrink / grow the ring. Shrinking drops the oldest surplus frames."""
        maxlen = max(1, int(maxlen))
        self._maxlen = maxlen
        while len(self._frames) > self._maxlen:
            self._frames.popleft()

    def push(self, payload: bytes, meta: dict | None = None) -> None:
        """Append one replay frame (``payload`` is the encoded snapshot)."""
        self._frames.append((meta or {}, bytes(payload)))

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)

    def latest(self) -> tuple[dict, bytes] | None:
        if not self._frames:
            return None
        return self._frames[-1]

    def clear(self) -> None:
        self._frames.clear()


# ----------------------------------------------------------------------------
# Adaptive history length + proactive memory budget
# ----------------------------------------------------------------------------


class AdaptiveHistoryController:
    """Shrinks history-bearing buffers as memory pressure rises.

    Several evolving structures grow with session length: the feedback history
    ring, the replay window, the lineage log and the novelty history. Each has
    a ``(min, max)`` span; as the process RSS climbs toward the configured
    watchdog budget the controller scales every span down toward its floor so
    the renderer gracefully degrades memory use instead of OOMing. When memory
    is comfortable it restores the configured maxima.

    ``pressure`` is 0.0 (no pressure, full history) .. 1.0 (max pressure, all
    history at its floor).
    """

    def __init__(self, targets: dict[str, tuple[int, int]] | None = None):
        # name -> (min, max)
        self._targets: dict[str, tuple[int, int]] = dict(targets or {})

    def add(self, name: str, minimum: int, maximum: int) -> None:
        self._targets[name] = (max(0, int(minimum)), max(1, int(maximum)))

    def value(self, name: str, pressure: float) -> int:
        """Return the current (pressure-scaled) length for ``name``."""
        if name not in self._targets:
            return 0
        lo, hi = self._targets[name]
        p = max(0.0, min(1.0, float(pressure)))
        # Linear ramp from hi (no pressure) down to lo (full pressure).
        return round(lo + (hi - lo) * (1.0 - p))


class GpuMemoryBudget:
    """Per-frame process-memory check that *acts*, not just logs.

    Wraps the existing RSS probe (psutil -> /proc/self -> resource) and exposes
    a single ``pressure()`` call returning 0..1 (fraction of the watchdog budget
    already consumed). The renderer feeds that pressure into
    :class:`AdaptiveHistoryController` and into a proportional shader-cache trim,
    so caches are reclaimed *before* the kernel OOM killer fires.

    RSS sampling uses ``psutil`` first, falls back to ``/proc/self/status``
    (Linux only), and finally to ``resource.getrusage`` — which reports
    ``ru_maxrss`` in KB on Linux and in bytes on macOS (Darwin).  The fallback
    branches on ``sys.platform`` so the value is correct on both.
    """

    def __init__(self, threshold_mb: float = 4096.0, sample_interval: float = 5.0):
        self.threshold_mb = float(max(1.0, threshold_mb))
        self.sample_interval = float(max(0.5, sample_interval))
        self._last_sample = 0.0
        self._cached_mb = 0.0
        self._peak_mb = 0.0

    def _sample_rss_mb(self) -> float:
        import importlib.util

        if importlib.util.find_spec("psutil") is not None:
            try:
                import psutil  # optional  # type: ignore[reportMissingModuleSource]

                return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                logger.debug("psutil unavailable", exc_info=True)
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except OSError:
            pass
        try:
            import resource

            # ru_maxrss is in KB on Linux, bytes on macOS (Darwin).
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return float(maxrss) / (1024.0 * 1024.0)
            return float(maxrss) / 1024.0
        except Exception:
            logger.debug("resource module unavailable", exc_info=True)
            return 0.0

    def rss_mb(self) -> float:
        """Return the most recent RSS sample (refreshed at ``sample_interval``)."""
        now = time.time()
        if now - self._last_sample >= self.sample_interval:
            self._last_sample = now
            self._cached_mb = self._sample_rss_mb()
            self._peak_mb = max(self._peak_mb, self._cached_mb)
        return self._cached_mb

    def pressure(self) -> float:
        """Return 0..1 fraction of the configured budget currently in use."""
        rss = self.rss_mb()
        if self.threshold_mb <= 0.0:
            return 0.0
        return max(0.0, min(1.0, rss / self.threshold_mb))

    def stats(self) -> dict[str, float]:
        return {
            "rss_mb": self.rss_mb(),
            "threshold_mb": self.threshold_mb,
            "pressure": self.pressure(),
            "peak_mb": self._peak_mb,
        }


def apply_adaptive_trims(
    renderer: Any,
    budget: GpuMemoryBudget,
    history: AdaptiveHistoryController,
) -> dict[str, Any]:
    """Run one proactive trim pass; mutate ``renderer`` caches in place.

    Returns a small dict of what was trimmed so the watchdog can log it. Safe to
    call every frame (it self-throttles via the budget's sampling interval and
    only *acts* when pressure crosses meaningful thresholds).

    **Required renderer surface.**  This function introspects the following
    attributes on ``renderer`` (all optional — missing attributes are silently
    skipped).  If the renderer renames these fields the corresponding trim is
    skipped (logged at debug level), so the surface is documented but not
    strictly enforced:

    * ``feedback``              — object with ``ensure_history(int)`` method.
    * ``_replay``               — object with ``set_maxlen(int)`` / ``maxlen`` attr.
    * ``memory_dna``            — object with ``trim_to(int)`` method.
    * ``fitness_evaluator``     — object with ``_novelty_history`` deque.
    * ``shader_cache``          — object with ``max_size`` attr and ``trim_to(int)``
                                  method returning the number of freed entries.
    """
    pressure = budget.pressure()
    action = {}

    # 1. Adaptive history lengths: feedback ring, replay window, novelty, lineage.
    if hasattr(renderer, "feedback") and renderer.feedback is not None:
        target = history.value("feedback_history", pressure)
        if target != getattr(renderer.feedback, "history", 0):
            try:
                renderer.feedback.ensure_history(target)
                action["feedback_history"] = target
            except Exception as e:  # pragma: no cover
                logger.debug("Feedback history resize failed: %s", e)

    if hasattr(renderer, "_replay") and renderer._replay is not None:
        target = history.value("replay", pressure)
        if target != renderer._replay.maxlen:
            renderer._replay.set_maxlen(target)
            action["replay"] = target

    if hasattr(renderer, "memory_dna") and renderer.memory_dna is not None:
        target = history.value("lineage", pressure)
        trimmed = renderer.memory_dna.trim_to(target)
        if trimmed:
            action["lineage"] = trimmed

    if hasattr(renderer, "fitness_evaluator") and renderer.fitness_evaluator is not None:
        fe = renderer.fitness_evaluator
        target = history.value("novelty", pressure)
        if target < len(getattr(fe, "_novelty_history", ())):
            try:
                while len(fe._novelty_history) > target:
                    fe._novelty_history.popleft()
                action["novelty"] = target
            except Exception:
                logger.debug("Novelty history trim failed", exc_info=True)

    # 2. Proportional shader-cache trim: the biggest single GPU allocation.
    cache = getattr(renderer, "shader_cache", None)
    if cache is not None and pressure > 0.5:
        # Keep at least 8 programs; trim down toward a fraction of capacity that
        # tightens as pressure rises (0.75x at 50% -> 0.25x at 100%).
        keep_frac = max(0.1, 1.0 - (pressure - 0.5) * 1.4)
        target = max(8, int(cache.max_size * keep_frac))
        freed = cache.trim_to(target)
        if freed:
            action["shader_cache_freed"] = freed

    return action
