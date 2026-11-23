#!/usr/bin/env python3

"""evo_gpu_compute.py

GPU compute-shader based fitness evaluation for the EvoRenderer project.

This module moves the per-pixel fitness analysis that the CPU
``FitnessEvaluator`` (in ``evo_visualizer.py``) used to do with a full
``glReadPixels`` readback of an ``res x res`` RGBA image into a small set of
OpenGL compute shaders.  The shaders reduce the image down to a handful of
scalar metrics (entropy, motion, novelty, colour variance) and write them into
a tiny SSBO / PBO (4-16 floats) that the CPU maps.  This removes the bandwidth
bottleneck of shipping the entire frame back over the PCIe bus every evaluation.

Two code paths are provided:

* **Compute path (OpenGL 4.3+, ``GL_ARB_compute_shader``):** the real
  implementation.  A single dispatch tiles the fitness texture into a grid of
  work-groups, each one accumulating partial histograms / sums into a shared
  work-group buffer, then a final reduction shader sums those partial results
  into the small output buffer.

* **Fragment-shader fallback (OpenGL 3.3):** if compute shaders are not
  available we emulate the same reduction with a mip-style ping-pong blur so
  the API surface (``entropy`` / ``motion`` / ``novelty`` / ``color_variance``
  / ``fitness``) stays identical for callers.  The readback is still a small
  buffer, not the full image.

Every GL call in this module goes through ``_gl_check`` / guarded try blocks so
that a driver without the required extensions degrades gracefully to ``None``
and the caller can fall back to the CPU evaluator.

Integration
-----------
The :class:`GpuFitnessEvaluator` is designed to be a drop-in helper for the
:class:`EvoRenderer` fitness loop in ``evo_visualizer.py``.  It exposes
:meth:`GpuFitnessEvaluator.evaluate` which mirrors ``FitnessEvaluator.evaluate``
but takes a *texture name* and a *previous texture name* instead of raw bytes,
and returns the same weighted fitness.  When the GPU path is unavailable
``available`` is ``False`` and the renderer keeps using its CPU path.

Thread / context note
---------------------
All GL objects created here must live in the same OpenGL context that the
``Gtk.GLArea`` uses (the *realize* callback).  ``init_gl`` is therefore a no-op
unless it is called from inside that context.
"""

from __future__ import annotations

import logging
from typing import Any

from OpenGL import GL

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy optional hard dep here
    np = None

logger = logging.getLogger(__name__)

# Number of scalar metrics carried in the output buffer.  Kept small (<=16) so
# the CPU readback is a few dozen bytes, never the full frame.
OUTPUT_FLOATS = 16
PARTIAL_HIST_BINS = 16
PARTIAL_SCALARS = 6
PARTIAL_BLOCK_WORDS = PARTIAL_HIST_BINS + PARTIAL_SCALARS
PARTIAL_BLOCK_BYTES = PARTIAL_BLOCK_WORDS * 4
FALLBACK_PACKED_FLOATS = 16
# Layout of the output buffer (indices into the float array).
OUT_ENTROPY = 0
OUT_MOTION = 1
OUT_NOVELTY = 2
OUT_COLOR_VAR = 3
OUT_MEAN_BRIGHT = 4
OUT_EDGE = 5
OUT_FITNESS = 6
OUT_PIXELS = 7

# Default fitness weights (sum roughly to 1.0).  These mirror the weighting
# used by the CPU evaluator so behaviour is comparable.
DEFAULT_WEIGHTS: dict[str, float] = {
    "entropy": 0.30,
    "motion": 0.20,
    "novelty": 0.20,
    "color_variance": 0.25,
    "edge": 0.05,
}


# ============================================================================
# GL helpers
# ============================================================================


def _gl_error() -> str | None:
    """Return a human readable GL error string, or None if no error is pending."""
    try:
        err = GL.glGetError()
    except RuntimeError:  # pragma: no cover - context not current
        return "glGetError-failed"
    if err == GL.GL_NO_ERROR:
        return None
    return f"GL_ERROR(0x{err:04X})"


def _gl_check(label: str) -> None:
    """Raise RuntimeError if a GL error is pending after ``label``."""
    err = _gl_error()
    if err is not None:
        raise RuntimeError(f"{label}: {err}")


def _have_compute_shaders() -> bool:
    """Best-effort detection of OpenGL 4.3+ compute shader support."""
    try:
        if getattr(GL, "glDispatchCompute", None) is None:
            return False
        if getattr(GL, "glBindBufferBase", None) is None:
            return False

        major = minor = None
        if hasattr(GL, "GL_MAJOR_VERSION") and hasattr(GL, "GL_MINOR_VERSION"):
            try:
                major = int(GL.glGetIntegerv(GL.GL_MAJOR_VERSION))
                minor = int(GL.glGetIntegerv(GL.GL_MINOR_VERSION))
            except (RuntimeError, TypeError, ValueError):
                major = minor = None
            finally:
                GL.glGetError()
        if major is not None and minor is not None and (major, minor) >= (4, 3):
            return True

        version = GL.glGetString(GL.GL_VERSION)
        if version is not None:
            text = version.decode(errors="ignore") if isinstance(version, bytes) else str(version)
            try:
                version_parts = text.split()[0].split(".")
                major, minor = int(version_parts[0]), int(version_parts[1])
            except (ValueError, IndexError):
                return False
            if (major, minor) >= (4, 3):
                return True

        extensions = GL.glGetString(GL.GL_EXTENSIONS)
        if extensions is not None:
            if isinstance(extensions, bytes):
                extensions = extensions.decode(errors="ignore")
            return "GL_ARB_compute_shader" in extensions
        return False
    except (RuntimeError, TypeError, ValueError):  # pragma: no cover - called outside a context
        return False


# ============================================================================
# Shader sources
# ============================================================================

# Shared GLSL helper: 16-bin luminance histogram + per-pixel accumulators.
_COMPUTE_COMMON = """
#version 430 core
#extension GL_ARB_compute_shader : enable

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D uCurrent;
layout(binding = 1) uniform sampler2D uPrevious;

shared uint  s_count;

uniform int uRes;

float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}
"""

# Pass 1: each 8x8 work group tiles the image. Every invocation accumulates
# its own pixel metrics into a per-invocation slot of shared memory; the
# single thread with local id 0 then reduces those slots and writes the
# work-group's partial block into the global SSBO. Only the histogram bin uses
# an atomic, and that one is on a uint (GLSL atomics are int/uint only), so no
# floating-point atomic is required.
_COMPUTE_REDUCE = (
    _COMPUTE_COMMON
    + """
struct PartialBlock {
    uint hist[16];
    float lumSum;
    float lumSumSq;
    float motionSum;
    float edgeSum;
    float brightSum;
    uint count;
};

layout(std430, binding = 2) buffer PartialBuf {
    PartialBlock partial[];
};

shared uint  s_histShared[16];
shared float s_lum[64];
shared float s_lumSq[64];
shared float s_motion[64];
shared float s_edge[64];
shared float s_bright[64];

void main() {
    uint lid = gl_LocalInvocationIndex;
    if (lid < 16u) s_histShared[lid] = 0u;
    s_lum[lid] = 0.0; s_lumSq[lid] = 0.0; s_motion[lid] = 0.0;
    s_edge[lid] = 0.0; s_bright[lid] = 0.0;
    if (lid == 0u) s_count = 0u;
    barrier();

    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    int res = uRes;
    if (gid.x < res && gid.y < res) {
        vec3 cur = texelFetch(uCurrent, gid, 0).rgb;
        float lum = luminance(cur);
        int bin = int(clamp(lum * 15.0, 0.0, 15.0));
        atomicAdd(s_histShared[bin], 1u);
        s_lum[lid] = lum;
        s_lumSq[lid] = lum * lum;
        s_bright[lid] = dot(cur, vec3(1.0)) / 3.0;
        atomicAdd(s_count, 1u);

        vec3 prev = texelFetch(uPrevious, gid, 0).rgb;
        s_motion[lid] = abs(lum - luminance(prev));

        float l = luminance(texelFetch(uCurrent, clamp(gid + ivec2(-1, 0), ivec2(0), ivec2(res - 1)), 0).rgb);
        float r = luminance(texelFetch(uCurrent, clamp(gid + ivec2( 1, 0), ivec2(0), ivec2(res - 1)), 0).rgb);
        float u = luminance(texelFetch(uCurrent, clamp(gid + ivec2(0, -1), ivec2(0), ivec2(res - 1)), 0).rgb);
        float d = luminance(texelFetch(uCurrent, clamp(gid + ivec2(0,  1), ivec2(0), ivec2(res - 1)), 0).rgb);
        s_edge[lid] = abs(l - r) + abs(u - d);
    }
    barrier();

    if (lid == 0u) {
        uint wg = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
        float lumSum = 0.0, lumSumSq = 0.0, motionSum = 0.0;
        float edgeSum = 0.0, brightSum = 0.0;
        for (uint i = 0u; i < 64u; i++) {
            lumSum   += s_lum[i];
            lumSumSq += s_lumSq[i];
            motionSum += s_motion[i];
            edgeSum  += s_edge[i];
            brightSum += s_bright[i];
        }

        for (uint i = 0u; i < 16u; i++) {
            partial[wg].hist[i] = s_histShared[i];
        }
        partial[wg].lumSum    = lumSum;
        partial[wg].lumSumSq  = lumSumSq;
        partial[wg].motionSum = motionSum;
        partial[wg].edgeSum   = edgeSum;
        partial[wg].brightSum = brightSum;
        partial[wg].count     = s_count;
    }
}
"""
)

# Pass 2: a single work group sums all partials and writes the final metrics
# into the small output buffer (binding 3).
_COMPUTE_FINALIZE = """
#version 430 core
#extension GL_ARB_compute_shader : enable

layout(local_size_x = 1) in;

struct PartialBlock {
    uint hist[16];
    float lumSum;
    float lumSumSq;
    float motionSum;
    float edgeSum;
    float brightSum;
    uint count;
};

layout(std430, binding = 2) readonly buffer PartialBuf {
    PartialBlock partial[];
};

layout(std430, binding = 3) buffer OutBuf {
    float outData[16];
};

uniform float uNovelty;
uniform uint uPartialCount;

void main() {
    uint hist[16];
    for (uint i = 0u; i < 16u; i++) {
        hist[i] = 0u;
    }

    float lumSum = 0.0;
    float lumSumSq = 0.0;
    float motionSum = 0.0;
    float edgeSum = 0.0;
    float brightSum = 0.0;
    uint n = 0u;
    for (uint wg = 0u; wg < uPartialCount; wg++) {
        for (uint i = 0u; i < 16u; i++) {
            hist[i] += partial[wg].hist[i];
        }
        lumSum    += partial[wg].lumSum;
        lumSumSq  += partial[wg].lumSumSq;
        motionSum += partial[wg].motionSum;
        edgeSum   += partial[wg].edgeSum;
        brightSum += partial[wg].brightSum;
        n         += partial[wg].count;
    }

    float inv = (n > 0u) ? (1.0 / float(n)) : 0.0;
    float entropy = 0.0;
    for (uint i = 0u; i < 16u; i++) {
        float p = float(hist[i]) * inv;
        if (p > 1e-8) entropy -= p * log2(p);
    }

    float mean = lumSum * inv;
    float var  = max(lumSumSq * inv - mean * mean, 0.0);
    float motion = motionSum * inv;
    float edge = edgeSum * inv;
    float bright = brightSum * inv;

    outData[0] = entropy;
    outData[1] = motion;
    outData[2] = uNovelty;
    outData[3] = var;
    outData[4] = bright;
    outData[5] = edge;
    outData[6] = 0.0;
    outData[7] = float(n);
}
"""


# OpenGL 3.3 fallback: a fragment-shader reduction that averages a 2x2 block
# down into a smaller texture repeatedly until we reach a 1x1 "average" texel
# carrying the mean colour; a final 1x1 pass composites the metrics into the
# small output buffer.  Novelty stays CPU-side (SSIM-like).
_FALLBACK_VERT = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;
out vec2 v_uv;
void main() {
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

_FALLBACK_REDUCE_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D uSrc;
uniform float uTexel;
void main() {
    vec3 a = texture(uSrc, v_uv + vec2(-uTexel, -uTexel)).rgb;
    vec3 b = texture(uSrc, v_uv + vec2( uTexel, -uTexel)).rgb;
    vec3 c = texture(uSrc, v_uv + vec2(-uTexel,  uTexel)).rgb;
    vec3 d = texture(uSrc, v_uv + vec2( uTexel,  uTexel)).rgb;
    fragColor = vec4((a + b + c + d) * 0.25, 1.0);
}
"""

_FALLBACK_FINALIZE_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D uHi;
uniform sampler2D uPrev;
uniform float uResF;
uniform float uNovelty;
void main() {
    float edgeSum = 0.0;
    float brightSum = 0.0;
    float motionSum = 0.0;
    float lumSum = 0.0;
    float lumSumSq = 0.0;
    uint hist[16];
    for (int i = 0; i < 16; i++) hist[i] = 0u;
    int res = max(1, int(uResF));
    for (int y = 0; y < res; y++) {
        for (int x = 0; x < res; x++) {
            ivec2 texel = ivec2(x, y);
            vec3 c = texelFetch(uHi, texel, 0).rgb;
            float l = dot(c, vec3(0.299, 0.587, 0.114));
            int bin = int(clamp(l * 15.0, 0.0, 15.0));
            hist[bin] += 1u;
            lumSum += l;
            lumSumSq += l * l;
            brightSum += dot(c, vec3(1.0)) / 3.0;
            vec3 p = texelFetch(uPrev, texel, 0).rgb;
            motionSum += abs(l - dot(p, vec3(0.299, 0.587, 0.114)));
            float l = dot(c, vec3(0.299, 0.587, 0.114));
            float leftLum = dot(texelFetch(uHi, clamp(texel + ivec2(-1, 0), ivec2(0), ivec2(res - 1)), 0).rgb, vec3(0.299, 0.587, 0.114));
            float rightLum = dot(texelFetch(uHi, clamp(texel + ivec2(1, 0), ivec2(0), ivec2(res - 1)), 0).rgb, vec3(0.299, 0.587, 0.114));
            float upLum = dot(texelFetch(uHi, clamp(texel + ivec2(0, 1), ivec2(0), ivec2(res - 1)), 0).rgb, vec3(0.299, 0.587, 0.114));
            float downLum = dot(texelFetch(uHi, clamp(texel + ivec2(0, -1), ivec2(0), ivec2(res - 1)), 0).rgb, vec3(0.299, 0.587, 0.114));
            edgeSum += abs(leftLum - rightLum) + abs(upLum - downLum);
        }
    }
    float count = float(res * res);
    float inv = 1.0 / count;
    float mean = lumSum * inv;
    float varSum = max(lumSumSq * inv - mean * mean, 0.0);
    float entropy = 0.0;
    for (int i = 0; i < 16; i++) {
        float p = float(hist[i]) * inv;
        if (p > 1e-8) entropy -= p * log2(p);
    }
    int slot = int(gl_FragCoord.x);
    if (slot == 0) {
        fragColor = vec4(entropy, motionSum * inv, uNovelty, varSum);
    } else if (slot == 1) {
        fragColor = vec4(brightSum * inv, edgeSum * inv, 0.0, count);
    } else {
        fragColor = vec4(0.0);
    }
}
"""


# ============================================================================
# AsyncComputeFence
# ============================================================================


class AsyncComputeFence:
    """Non-blocking GPU->CPU synchronisation object.

    Wraps ``glFenceSync`` (GL 3.2+) so the caller can submit a compute dispatch
    / draw and then *poll* for completion instead of calling ``glFinish`` (which
    stalls the whole pipeline).  Used by :class:`GpuFitnessEvaluator` to know
    when it is safe to map the small output buffer without forcing a sync point.

    On contexts without ``glFenceSync`` (GL < 3.2) it degrades to an immediate
    "always ready" fence so the rest of the code path still works.
    """

    def __init__(self) -> None:
        self._sync = None
        self._have_fence = getattr(GL, "glFenceSync", None) is not None
        self._signaled = False

    def signal(self) -> None:
        """Insert a fence after the current GL commands."""
        self._signaled = False
        if self._sync is not None and getattr(GL, "glDeleteSync", None) is not None:
            try:
                GL.glDeleteSync(self._sync)
            except RuntimeError:  # pragma: no cover - driver quirk
                pass
            self._sync = None
        if not self._have_fence:
            self._signaled = True
            return
        try:
            self._sync = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
            _gl_check("AsyncComputeFence.signal")
        except RuntimeError as e:  # pragma: no cover - driver quirk
            logger.debug("AsyncComputeFence.signal failed: %s", e)
            self._sync = None
            self._signaled = True

    def ready(self) -> bool:
        """Return True if the fence has been signalled and work is complete."""
        if self._signaled:
            return True
        if self._sync is None:
            return True
        try:
            res = GL.glClientWaitSync(self._sync, 0, 0)
            if res in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED):
                self._signaled = True
                return True
            return False
        except RuntimeError:  # pragma: no cover
            self._signaled = True
            return True

    def wait(self, timeout_ns: int = 1_000_000_000) -> bool:
        """Block (with timeout) until the fence is ready. Returns readiness."""
        if self._signaled:
            return True
        if self._sync is None:
            return True
        try:
            res = GL.glClientWaitSync(self._sync, GL.GL_SYNC_FLUSH_COMMANDS_BIT, timeout_ns)
            self._signaled = res in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED)
        except RuntimeError:  # pragma: no cover
            self._signaled = True
        return self._signaled

    def cleanup(self) -> None:
        if self._sync is not None and getattr(GL, "glDeleteSync", None) is not None:
            try:
                GL.glDeleteSync(self._sync)
            except RuntimeError:  # pragma: no cover
                pass
        self._sync = None
        self._signaled = False


# ============================================================================
# ComputeShaderManager
# ============================================================================


class ComputeShaderManager:
    """Compiles and manages compute (or fallback) shader programs for fitness.

    Responsibilities:

    * Detect compute-shader availability and pick the GL 4.3+ compute path or
      the GL 3.3 fragment-shader fallback.
    * Compile / link the reduction + finalize programs and the ping-pong
      reduction chain.
    * Provide the four analysis primitives required by the spec:

      * :meth:`compute_histogram` -> entropy
      * :meth:`detect_motion`     -> frame difference
      * :meth:`estimate_novelty`  -> SSIM-like comparison (CPU helper on a tiny
        downsample; GPU supply is optional)
      * :meth:`aggregate_fitness` -> weighted sum

    * Own the small output SSBO / PBO and the :class:`AsyncComputeFence`.
    """

    def __init__(self, res: int = 64, weights: dict[str, float] | None = None):
        self.res = max(1, int(res))
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        self.available = False
        self.using_compute = False

        self._reduce_prog: int | None = None
        self._finalize_prog: int | None = None
        self._fallback_reduce_prog: int | None = None
        self._fallback_finalize_prog: int | None = None
        self._ssbo: int | None = None
        self._out_ssbo: int | None = None
        self._out_pbo: int | None = None
        self._partial_wg = 0
        self._reduce_chain: list[tuple[int, int]] = []
        self._finalize_fbo: int | None = None
        self._finalize_tex: int | None = None
        self._vao: int | None = None
        self._vbo: int | None = None

        self.fence = AsyncComputeFence()
        self._last_metrics: tuple[float, ...] = (0.0,) * OUTPUT_FLOATS
        self.last_read_ready = True

    # -- GL object creation -------------------------------------------------

    @staticmethod
    def _compile_shader(stype: int, src: str) -> int:
        sh = GL.glCreateShader(stype)
        try:
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            if GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
                log = GL.glGetShaderInfoLog(sh)
                raise RuntimeError(f"Shader compile failed: {log}")
            return sh
        except Exception:
            try:
                GL.glDeleteShader(sh)
            except RuntimeError:  # pragma: no cover
                pass
            raise

    @staticmethod
    def _link_program(*shaders: int) -> int:
        prog = GL.glCreateProgram()
        for s in shaders:
            GL.glAttachShader(prog, s)
        GL.glLinkProgram(prog)
        if GL.glGetProgramiv(prog, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            log = GL.glGetProgramInfoLog(prog)
            for s in shaders:
                try:
                    GL.glDetachShader(prog, s)
                    GL.glDeleteShader(s)
                except RuntimeError:  # pragma: no cover
                    pass
            try:
                GL.glDeleteProgram(prog)
            except RuntimeError:  # pragma: no cover
                pass
            raise RuntimeError(f"Program link failed: {log}")
        for s in shaders:
            GL.glDetachShader(prog, s)
            GL.glDeleteShader(s)
        return prog

    def _compile_program(self, *args) -> int:
        """Compile a program from a flat sequence of (type, src) pairs."""
        shaders = []
        try:
            for i in range(0, len(args), 2):
                shaders.append(self._compile_shader(args[i], args[i + 1]))
            return self._link_program(*shaders)
        except Exception:
            for s in shaders:
                try:
                    GL.glDeleteShader(s)
                except RuntimeError:  # pragma: no cover
                    pass
            raise

    def init_gl(self) -> bool:
        """Initialise GL resources. Returns ``available``.

        Safe to call outside a live context (returns False and never raises).
        """
        try:
            self.using_compute = _have_compute_shaders()
            nbytes = (OUTPUT_FLOATS if self.using_compute else FALLBACK_PACKED_FLOATS) * 4

            if self.using_compute:
                wg_x = (self.res + 7) // 8
                wg_y = (self.res + 7) // 8
                self._partial_wg = wg_x * wg_y

                self._reduce_prog = self._compile_program(GL.GL_COMPUTE_SHADER, _COMPUTE_REDUCE)
                self._finalize_prog = self._compile_program(
                    GL.GL_COMPUTE_SHADER, _COMPUTE_FINALIZE
                )

                self._ssbo = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._ssbo)
                GL.glBufferData(
                    GL.GL_SHADER_STORAGE_BUFFER,
                    self._partial_wg * PARTIAL_BLOCK_BYTES,
                    None,
                    GL.GL_STREAM_READ,
                )
                _gl_check("partial SSBO alloc")

                self._out_ssbo = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._out_ssbo)
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, nbytes, None, GL.GL_STREAM_READ)
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
                _gl_check("output SSBO alloc")

                self._out_pbo = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._out_pbo)
                GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, nbytes, None, GL.GL_STREAM_READ)
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            else:
                self._fallback_finalize_prog = self._compile_program(
                    GL.GL_VERTEX_SHADER,
                    _FALLBACK_VERT,
                    GL.GL_FRAGMENT_SHADER,
                    _FALLBACK_FINALIZE_FRAG,
                )
                self._build_fallback_chain()
                self._out_pbo = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._out_pbo)
                GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, nbytes, None, GL.GL_STREAM_READ)
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

            self.available = True
            logger.info(
                "ComputeShaderManager ready (compute=%s, res=%d)", self.using_compute, self.res
            )
        except RuntimeError as e:
            logger.warning("ComputeShaderManager init failed, GPU fitness disabled: %s", e)
            self.available = False
            self._cleanup_gl_objects()
        return self.available

    def _build_fallback_chain(self) -> None:
        """Create fallback geometry and the 4x1 metric output target."""
        verts = [
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            1.0,
        ]
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        arr = (GL.GLfloat * len(verts))(*verts)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(verts) * 4, arr, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, GL.GLvoidp(8))
        GL.glBindVertexArray(0)

        self._reduce_chain = []

        self._finalize_fbo = GL.glGenFramebuffers(1)
        self._finalize_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._finalize_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, 4, 1, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._finalize_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._finalize_tex, 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        _gl_check("fallback finalize FBO")

    # -- Analysis primitives ------------------------------------------------

    def dispatch_analysis(self, current_tex: int, previous_tex: int, novelty: float = 0.0) -> bool:
        """Run the full reduction producing the small output buffer.

        Binds ``current_tex`` / ``previous_tex``, dispatches the reduction, then
        inserts a fence so :meth:`read_output` can poll before mapping.  Returns
        True on success.
        """
        if not self.available:
            return False
        if not self.fence.ready():
            return False
        try:
            if previous_tex == 0:
                previous_tex = current_tex
            if self.using_compute:
                self._dispatch_compute(current_tex, previous_tex, novelty)
            else:
                self._dispatch_fallback(current_tex, previous_tex, novelty)
            self.fence.signal()
            return True
        except RuntimeError as e:  # pragma: no cover - driver/runtime error
            logger.debug("dispatch_analysis failed: %s", e)
            return False

    def _dispatch_compute(self, current_tex: int, previous_tex: int, novelty: float) -> None:
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._ssbo)
        GL.glBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER,
            0,
            self._partial_wg * PARTIAL_BLOCK_BYTES,
            (GL.GLuint * (self._partial_wg * PARTIAL_BLOCK_WORDS))(),
        )

        GL.glUseProgram(self._reduce_prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, current_tex)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, previous_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._reduce_prog, "uRes"), self.res)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, self._ssbo)
        wg_x = (self.res + 7) // 8
        wg_y = (self.res + 7) // 8
        GL.glDispatchCompute(wg_x, wg_y, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        GL.glUseProgram(self._finalize_prog)
        GL.glUniform1f(GL.glGetUniformLocation(self._finalize_prog, "uNovelty"), float(novelty))
        GL.glUniform1ui(
            GL.glGetUniformLocation(self._finalize_prog, "uPartialCount"), self._partial_wg
        )
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, self._ssbo)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, self._out_ssbo)
        GL.glDispatchCompute(1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, self._out_ssbo)
        GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, self._out_pbo)
        GL.glCopyBufferSubData(
            GL.GL_COPY_READ_BUFFER, GL.GL_COPY_WRITE_BUFFER, 0, 0, OUTPUT_FLOATS * 4
        )
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, 0)
        GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, 0)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        GL.glUseProgram(0)
        _gl_check("compute dispatch")

    def _dispatch_fallback(self, current_tex: int, previous_tex: int, novelty: float) -> None:
        GL.glBindVertexArray(self._vao)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._finalize_fbo)
        GL.glViewport(0, 0, 4, 1)
        GL.glUseProgram(self._fallback_finalize_prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, current_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._fallback_finalize_prog, "uHi"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, previous_tex or current_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._fallback_finalize_prog, "uPrev"), 1)
        GL.glUniform1f(
            GL.glGetUniformLocation(self._fallback_finalize_prog, "uResF"), float(self.res)
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self._fallback_finalize_prog, "uNovelty"), float(novelty)
        )
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._out_pbo)
        GL.glReadPixels(0, 0, 4, 1, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        _gl_check("fallback dispatch")

    def read_output(self, block: bool = False) -> tuple[float, ...] | None:
        """Map the small output PBO and return the metrics tuple.

        If ``block`` is False and the fence is not yet ready, returns the last
        cached metrics (or None on the very first call) so the CPU never stalls.
        """
        if not self.available or self._out_pbo is None:
            return None
        if not block and not self.fence.ready():
            self.last_read_ready = False
            return self._last_metrics if self._last_metrics != (0.0,) * OUTPUT_FLOATS else None
        if block and not self.fence.wait():
            self.last_read_ready = False
            return self._last_metrics
        self.last_read_ready = True

        try:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._out_pbo)
            ptr = GL.glMapBuffer(GL.GL_PIXEL_PACK_BUFFER, GL.GL_READ_ONLY)
            if ptr is None:
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
                return self._last_metrics
            packed_floats = OUTPUT_FLOATS if self.using_compute else FALLBACK_PACKED_FLOATS
            floats = (GL.GLfloat * packed_floats).from_address(ptr)
            result = tuple(float(floats[i]) for i in range(packed_floats))
            if packed_floats < OUTPUT_FLOATS:
                result += (0.0,) * (OUTPUT_FLOATS - packed_floats)
            values = list(result)
            values[OUT_FITNESS] = self.aggregate_fitness(result)
            result = tuple(values)
            GL.glUnmapBuffer(GL.GL_PIXEL_PACK_BUFFER)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            self._last_metrics = result
            return result
        except RuntimeError as e:  # pragma: no cover
            logger.debug("read_output failed: %s", e)
            try:
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            except RuntimeError:  # pragma: no cover
                pass
            return self._last_metrics

    # -- Metric accessors ---------------------------------------------------

    def compute_histogram(self, metrics: tuple[float, ...]) -> float:
        """Return entropy [0..4] from a metrics tuple (index 0)."""
        return float(metrics[OUT_ENTROPY]) if metrics else 0.0

    def detect_motion(self, metrics: tuple[float, ...]) -> float:
        """Return motion (mean abs frame diff) from a metrics tuple."""
        return float(metrics[OUT_MOTION]) if metrics else 0.0

    def capture_signature(self, tex: int, n: int = 8) -> bytes | None:
        if tex == 0:
            return None
        return self._sample_tex(tex, n)

    def signature_novelty(self, current: bytes | None, reference: bytes | None) -> float:
        if np is None or current is None or reference is None:
            return 0.0
        try:
            c = (
                np.frombuffer(current, dtype=np.uint8).reshape(-1, 4)[:, :3].astype(np.float32)
                / 255.0
            )
            r = (
                np.frombuffer(reference, dtype=np.uint8).reshape(-1, 4)[:, :3].astype(np.float32)
                / 255.0
            )
            mu_c = c.mean()
            mu_r = r.mean()
            sig_c = c.std()
            sig_r = r.std()
            c_sim = (2.0 * sig_c * sig_r + 1e-3) / (sig_c * sig_c + sig_r * sig_r + 1e-3)
            l_sim = (2.0 * mu_c * mu_r + 1e-3) / (mu_c * mu_c + mu_r * mu_r + 1e-3)
            return 1.0 - max(0.0, min(1.0, c_sim * l_sim))
        except (RuntimeError, ValueError):
            return 0.0

    def estimate_novelty(self, current_tex: int, reference_tex: int) -> float:
        """SSIM-like novelty estimate between two textures.

        When reference is missing (first frame / no history) novelty is 0.0.
        This is the "SSIM-like" proxy the spec asks for: a structural
        (luminance/contrast) distance rather than a raw pixel diff.
        """
        if current_tex == 0 or reference_tex == 0:
            return 0.0
        try:
            return self.signature_novelty(
                self.capture_signature(current_tex, 8), self.capture_signature(reference_tex, 8)
            )
        except RuntimeError:  # pragma: no cover
            return 0.0

    def _sample_tex(self, tex: int, n: int) -> bytes | None:
        fbo = GL.glGenFramebuffers(1)
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
            )
            GL.glViewport(0, 0, n, n)
            data = GL.glReadPixels(0, 0, n, n, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            if isinstance(data, bytes):
                return data
            return bytes(bytearray(data))
        finally:
            try:
                GL.glDeleteFramebuffers(1, [fbo])
            except RuntimeError:  # pragma: no cover
                pass

    def aggregate_fitness(
        self,
        metrics: tuple[float, ...],
        brightness_penalty: float = 0.3,
        max_brightness: float = 0.8,
    ) -> float:
        """Weighted-sum aggregation of the metrics into a single fitness.

        Mirrors the CPU ``FitnessEvaluator`` weighting: entropy, motion,
        novelty and colour variance contribute positively; excessive brightness
        is penalised so white-flash noise does not evolve.  Returns [0..1].
        """
        if not metrics:
            return 0.0
        entropy = self.compute_histogram(metrics)
        motion = self.detect_motion(metrics)
        novelty = metrics[OUT_NOVELTY]
        color_var = metrics[OUT_COLOR_VAR]
        edge = metrics[OUT_EDGE]
        brightness = metrics[OUT_MEAN_BRIGHT]

        w = self.weights
        fitness = 0.0
        fitness += min(entropy / 4.0, 1.0) * w.get("entropy", 0.30)
        fitness += min(motion * 15.0, 1.0) * w.get("motion", 0.20)
        fitness += min(novelty, 1.0) * w.get("novelty", 0.20)
        fitness += min(color_var * 20.0, 1.0) * w.get("color_variance", 0.25)
        fitness += min(edge * 10.0, 1.0) * w.get("edge", 0.05)
        fitness -= max(brightness - max_brightness, 0.0) * brightness_penalty
        return float(max(0.0, min(fitness, 1.0)))

    # -- Cleanup ------------------------------------------------------------

    def _cleanup_gl_objects(self) -> None:
        for attr in (
            "_reduce_prog",
            "_finalize_prog",
            "_fallback_reduce_prog",
            "_fallback_finalize_prog",
        ):
            prog = getattr(self, attr, None)
            if prog is not None:
                try:
                    GL.glDeleteProgram(prog)
                except RuntimeError:  # pragma: no cover
                    pass
                setattr(self, attr, None)
        if self._ssbo is not None:
            try:
                GL.glDeleteBuffers(1, [self._ssbo])
            except RuntimeError:  # pragma: no cover
                pass
            self._ssbo = None
        if self._out_ssbo is not None:
            try:
                GL.glDeleteBuffers(1, [self._out_ssbo])
            except RuntimeError:  # pragma: no cover
                pass
            self._out_ssbo = None
        if self._out_pbo is not None:
            try:
                GL.glDeleteBuffers(1, [self._out_pbo])
            except RuntimeError:  # pragma: no cover
                pass
            self._out_pbo = None
        for fbo, tex in self._reduce_chain:
            try:
                GL.glDeleteFramebuffers(1, [fbo])
                GL.glDeleteTextures(1, [tex])
            except RuntimeError:  # pragma: no cover
                pass
        self._reduce_chain = []
        for attr in ("_finalize_fbo", "_finalize_tex", "_vao", "_vbo"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    if attr == "_finalize_fbo":
                        GL.glDeleteFramebuffers(1, [obj])
                    elif attr == "_finalize_tex":
                        GL.glDeleteTextures(1, [obj])
                    elif attr == "_vao":
                        GL.glDeleteVertexArrays(1, [obj])
                    elif attr == "_vbo":
                        GL.glDeleteBuffers(1, [obj])
                except RuntimeError:  # pragma: no cover
                    pass
                setattr(self, attr, None)
        self.fence.cleanup()

    def cleanup(self) -> None:
        self._cleanup_gl_objects()
        self.available = False
        self.using_compute = False


# ============================================================================
# GpuFitnessEvaluator
# ============================================================================


class GpuFitnessEvaluator:
    """Replaces CPU readback with a GPU compute / fallback fitness path.

    Instead of shipping the full ``res x res`` RGBA frame back to the CPU via
    ``glReadPixels``, this evaluates fitness entirely on the GPU: the renderer
    renders the candidate into a fitness FBO (texture), then a compute shader
    (or fallback fragment chain) collapses it into a tiny 16-float buffer.  The
    CPU only maps those 4-16 floats.

    The public surface mirrors :class:`FitnessEvaluator` in ``evo_visualizer``:

    * :meth:`evaluate` -> fitness score (entropy, motion, novelty,
      color_variance, fitness)
    * :meth:`score_from_genome` -> genome-only fallback (delegates to the
      existing CPU evaluator so behaviour is unchanged when GPU is off).

    Integration
    -----------
    In :class:`EvoRenderer`::

        self.gpu_fitness = GpuFitnessEvaluator(self._config.fitness_image_res)
        # call inside the realize callback, after the GL context is live:
        self.gpu_fitness.init_gl()

        # in the per-frame / candidate scoring path:
        if self.gpu_fitness.available:
            fit = self.gpu_fitness.evaluate(
                genome, self._fitness_tex,
                previous_tex=self._previous_fitness_tex)
        else:
            fit = self.fitness_evaluator.evaluate(genome, image_bytes, prev)

    If the GPU path is unavailable (``available == False``) every method
    degrades to the CPU evaluator, so the renderer never breaks.
    """

    def __init__(
        self,
        res: int = 64,
        novelty_weight: float = 0.20,
        weights: dict[str, float] | None = None,
        cpu_evaluator: Any | None = None,
    ):
        self.res = max(1, int(res))
        self.novelty_weight = max(0.0, min(1.0, float(novelty_weight)))
        self.available = False
        self.using_compute = False
        self.manager = ComputeShaderManager(res=self.res, weights=weights)
        if weights is None or "novelty" not in weights:
            self.manager.weights["novelty"] = self.novelty_weight

        # CPU fallback used for genome-only scoring and when GPU is off.
        self._cpu = cpu_evaluator
        if self._cpu is None:
            self._cpu = self._make_cpu_evaluator()

        # Tiny novelty history (SSIM-like, kept CPU-side like the legacy path).
        self._novelty_history: list[bytes] = []
        self._last_metrics: tuple[float, ...] | None = None

    def _make_cpu_evaluator(self) -> Any:
        """Lazily import the existing CPU ``FitnessEvaluator`` if available."""
        try:
            from evo_visualizer import FitnessEvaluator

            return FitnessEvaluator(res=self.res)
        except ImportError:  # pragma: no cover - import path differs
            return None

    def init_gl(self) -> bool:
        """Initialise GL resources inside a live context. Returns availability."""
        self.available = self.manager.init_gl()
        self.using_compute = self.manager.using_compute
        return self.available

    # -- Novelty (SSIM-like, CPU side, bounded history) ----------------------

    def _novelty_for(self, current_tex: int) -> float:
        """Compute SSIM-like novelty vs the most-similar recent frame."""
        current_signature = self.manager.capture_signature(current_tex, 8)
        if current_signature is None:
            return 0.0
        if not self._novelty_history:
            self._novelty_history.append(current_signature)
            return 0.0
        best = max(
            self.manager.signature_novelty(current_signature, reference)
            for reference in self._novelty_history
        )
        self._novelty_history.append(current_signature)
        if len(self._novelty_history) > 200:
            self._novelty_history.pop(0)
        return float(min(1.0, best))

    # -- Public evaluation API ----------------------------------------------

    def evaluate(self, genome: Any, current_tex: int, previous_tex: int = 0) -> float:
        """Evaluate fitness on the GPU from two texture names.

        Args:
            genome: Genome (used only for symmetry penalty compatibility).
            current_tex: GL texture name of the current fitness frame.
            previous_tex: GL texture name of the previous frame (0 if none).

        Returns:
            Fitness score [0..1].  Falls back to the CPU evaluator on any
            GPU failure so the renderer keeps working.
        """
        if not self.available or current_tex == 0:
            return self.score_from_genome(genome)

        try:
            novelty = self._novelty_for(current_tex)
            if not self.manager.dispatch_analysis(current_tex, previous_tex, novelty):
                return self.score_from_genome(genome)
            # Non-blocking read: returns cached metrics if the GPU is not done.
            metrics = self.manager.read_output(block=False)
            if metrics is None or not self.manager.last_read_ready:
                metrics = self.manager.read_output(block=True)
            if metrics is None or not self.manager.last_read_ready:
                return self.score_from_genome(genome)

            self._last_metrics = metrics
            fitness = self.manager.aggregate_fitness(metrics)

            genome.fitness = fitness
            return fitness
        except RuntimeError as e:  # pragma: no cover - GPU path error
            logger.debug("GPU fitness evaluate failed: %s", e)
            return self.score_from_genome(genome)

    def metrics(self) -> dict[str, float]:
        """Return the last computed metrics dict for logging / debugging."""
        m = self._last_metrics or (0.0,) * OUTPUT_FLOATS
        return {
            "entropy": float(m[OUT_ENTROPY]),
            "motion": float(m[OUT_MOTION]),
            "novelty": float(m[OUT_NOVELTY]),
            "color_variance": float(m[OUT_COLOR_VAR]),
            "mean_brightness": float(m[OUT_MEAN_BRIGHT]),
            "edge": float(m[OUT_EDGE]),
            "fitness": float(m[OUT_FITNESS]),
        }

    def score_from_genome(self, genome: Any, weights: dict[str, float] | None = None) -> float:
        """Genome-only fallback score (delegates to the CPU evaluator)."""
        if self._cpu is not None and hasattr(self._cpu, "score_from_genome"):
            return float(self._cpu.score_from_genome(genome, weights))
        return 0.5

    def cleanup(self) -> None:
        """Release all GL resources."""
        try:
            self.manager.cleanup()
        except RuntimeError:  # pragma: no cover
            pass
        self.available = False
        self.using_compute = False


# ============================================================================
# Convenience: patch an EvoRenderer instance to use GPU fitness
# ============================================================================


def attach_to_renderer(renderer: Any, res: int | None = None) -> GpuFitnessEvaluator:
    """Attach a :class:`GpuFitnessEvaluator` to an ``EvoRenderer`` instance.

    Creates the evaluator, wires it to the renderer's cpu ``fitness_evaluator``
    for fallback, and (if a GL context is live) initialises GL.  The renderer
    should call ``gpu.evaluate(...)`` in its fitness path, falling back to the
    existing CPU ``fitness_evaluator`` when ``gpu.available`` is False.

    Returns the created evaluator so the caller can keep a reference.
    """
    if res is None:
        res = int(getattr(getattr(renderer, "_config", None), "fitness_image_res", 64) or 64)
    gpu = GpuFitnessEvaluator(
        res=res,
        cpu_evaluator=getattr(renderer, "fitness_evaluator", None),
    )
    renderer.gpu_fitness_evaluator = gpu
    try:
        gpu.init_gl()
    except RuntimeError as e:  # pragma: no cover - context not live yet
        logger.debug("attach_to_renderer: GPU init deferred (%s)", e)
    return gpu
