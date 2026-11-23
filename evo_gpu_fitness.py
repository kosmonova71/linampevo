#!/usr/bin/env python3

"""evo_gpu_fitness.py

GPU fitness kernels for the evolutionary renderer.

Replaces the CPU readback path:

    GPU render → CPU reads 512x512 RGBA → NumPy evaluates

with a fully GPU-resident pipeline:

    GPU render → GPU reduction → GPU histogram → GPU motion
    → GPU entropy → Read back only 8 floats

The CPU receives a compact Objectives vector with 8 scalar fields:

    beauty, motion, contrast, novelty, entropy, coverage,
    symmetry, energy

Instead of a 512x512 RGBA image (1 MB per frame), only 32 bytes
(8 x float32) are transferred over PCIe per evaluation.

Notes:
- novelty: Expected to be provided by the CPU as a uniform (archive distance
  to nearest neighbor in behavior space). The GPU does not compute novelty;
  it simply passes through the provided value.
- entropy: Computed via a 16-bin luminance histogram on the GPU.
- energy: Defined as mean_luminance * motion (distinct from edge magnitude).
"""

from __future__ import annotations

import logging

try:
    import numpy as np
except ImportError:
    np = None

from OpenGL import GL, error

logger = logging.getLogger(__name__)

OUTPUT_FLOATS = 8

OUT_BEAUTY = 0
OUT_MOTION = 1
OUT_CONTRAST = 2
OUT_NOVELTY = 3
OUT_ENTROPY = 4
OUT_COVERAGE = 5
OUT_SYMMETRY = 6
OUT_ENERGY = 7

DEFAULT_WEIGHTS: dict[str, float] = {
    "beauty": 0.30,
    "motion": 0.20,
    "contrast": 0.10,
    "novelty": 0.15,
    "entropy": 0.10,
    "coverage": 0.05,
    "symmetry": 0.05,
    "energy": 0.05,
}

_REDUCE_COMMON = """
#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D uCurrent;
layout(binding = 1) uniform sampler2D uPrevious;

shared float s_lum[64];
shared float s_lumSq[64];
shared float s_motion[64];
shared float s_edge[64];
shared float s_bright[64];
shared float s_symLeft[64];
shared float s_symRight[64];
shared float s_cover[64];
shared uint s_hist[64][16];

uniform int uRes;
uniform float uNovelty;

float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void hist_add(float lum) {
    uint bin = uint(clamp(lum * 16.0, 0.0, 15.999));
    uint lid = gl_LocalInvocationIndex;
    atomicAdd(s_hist[lid][bin], 1u);
}
"""

_REDUCE_PASS = (
    _REDUCE_COMMON
    + """
layout(std430, binding = 2) buffer PartialBuf {
    float data[];
} partial;

void main() {
    uint lid = gl_LocalInvocationIndex;
    s_lum[lid] = 0.0;
    s_lumSq[lid] = 0.0;
    s_motion[lid] = 0.0;
    s_edge[lid] = 0.0;
    s_bright[lid] = 0.0;
    s_symLeft[lid] = 0.0;
    s_symRight[lid] = 0.0;
    s_cover[lid] = 0.0;
    for (uint b = 0u; b < 16u; b++) {
        s_hist[lid][b] = 0u;
    }
    barrier();

    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    int res = uRes;
    if (gid.x < res && gid.y < res) {
        vec3 cur = texelFetch(uCurrent, gid, 0).rgb;
        float lum = luminance(cur);
        s_lum[lid] = lum;
        s_lumSq[lid] = lum * lum;
        s_bright[lid] = dot(cur, vec3(1.0)) / 3.0;
        s_cover[lid] = lum > 0.02 ? 1.0 : 0.0;
        hist_add(lum);

        float l = luminance(texelFetch(uCurrent,
            clamp(gid + ivec2(-1, 0), ivec2(0), ivec2(res - 1)), 0).rgb);
        float r = luminance(texelFetch(uCurrent,
            clamp(gid + ivec2( 1, 0), ivec2(0), ivec2(res - 1)), 0).rgb);
        float u = luminance(texelFetch(uCurrent,
            clamp(gid + ivec2(0, -1), ivec2(0), ivec2(res - 1)), 0).rgb);
        float d = luminance(texelFetch(uCurrent,
            clamp(gid + ivec2(0,  1), ivec2(0), ivec2(res - 1)), 0).rgb);
        float hEdge = abs(l - r);
        float vEdge = abs(u - d);
        s_edge[lid] = hEdge + vEdge;

        vec3 prev = texelFetch(uPrevious, gid, 0).rgb;
        s_motion[lid] = abs(lum - luminance(prev));

        int halfW = res / 2;
        if (gid.x < halfW) {
            vec3 left = texelFetch(uCurrent, gid, 0).rgb;
            vec3 right = texelFetch(uCurrent,
                ivec2(res - 1 - gid.x, gid.y), 0).rgb;
            s_symLeft[lid] = luminance(left);
            s_symRight[lid] = luminance(right);
        } else {
            s_symLeft[lid] = 0.0;
            s_symRight[lid] = 0.0;
        }
    } else {
        s_symLeft[lid] = 0.0;
        s_symRight[lid] = 0.0;
    }
    barrier();

    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            s_lum[lid] += s_lum[lid + stride];
            s_lumSq[lid] += s_lumSq[lid + stride];
            s_motion[lid] += s_motion[lid + stride];
            s_edge[lid] += s_edge[lid + stride];
            s_bright[lid] += s_bright[lid + stride];
            s_cover[lid] += s_cover[lid + stride];
            s_symLeft[lid] += s_symLeft[lid + stride];
            s_symRight[lid] += s_symRight[lid + stride];
            for (uint b = 0u; b < 16u; b++) {
                s_hist[lid][b] += s_hist[lid + stride][b];
            }
        }
        barrier();
    }

    if (lid == 0u) {
        uint wgID = gl_WorkGroupID.y * gl_NumWorkGroups.x
                  + gl_WorkGroupID.x;
        uint base = wgID * 11u;

        float lumSum = s_lum[0];
        float lumSumSq = s_lumSq[0];
        float motionSum = s_motion[0];
        float edgeSum = s_edge[0];
        float brightSum = s_bright[0];
        float coverSum = s_cover[0];
        float symLeftSum = s_symLeft[0];
        float symRightSum = s_symRight[0];
        uint pixCount = 0u;
        for (uint b = 0u; b < 16u; b++) {
            pixCount += s_hist[0][b];
        }

        partial.data[base + 0] = lumSum;
        partial.data[base + 1] = lumSumSq;
        partial.data[base + 2] = motionSum;
        partial.data[base + 3] = edgeSum;
        partial.data[base + 4] = brightSum;
        partial.data[base + 5] = coverSum;
        partial.data[base + 6] = symLeftSum;
        partial.data[base + 7] = symRightSum;
        partial.data[base + 8] = float(pixCount);
        partial.data[base + 9] = uNovelty;
        for (uint b = 0u; b < 16u; b++) {
            partial.data[base + 10 + b] = float(s_hist[0][b]);
        }
    }
}
"""
)

_FINALIZE_COMPUTE = """
#version 430 core

layout(local_size_x = 1, local_size_y = 1) in;

layout(std430, binding = 2) readonly buffer PartialBuf {
    float data[];
} partial;

layout(std430, binding = 3) buffer OutBuf {
    float outData[8];
} outputBuf;

uniform uint uWgCount;

float entropy_from_hist(float hist[16], uint total) {
    if (total == 0u) return 0.0;
    float e = 0.0;
    for (uint i = 0u; i < 16u; i++) {
        float p = hist[i] / float(total);
        if (p > 0.0) {
            e -= p * log2(p);
        }
    }
    return e / 4.0;
}

void main() {
    float lumSum = 0.0, lumSumSq = 0.0, motionSum = 0.0;
    float edgeSum = 0.0, brightSum = 0.0, coverSum = 0.0;
    float symLeftSum = 0.0, symRightSum = 0.0;
    uint pixCount = 0u;
    float novelty = 0.0;
    float hist[16];
    for (uint b = 0u; b < 16u; b++) hist[b] = 0.0;

    for (uint i = 0u; i < uWgCount; i++) {
        uint base = i * 27u;
        lumSum     += partial.data[base + 0];
        lumSumSq   += partial.data[base + 1];
        motionSum  += partial.data[base + 2];
        edgeSum    += partial.data[base + 3];
        brightSum  += partial.data[base + 4];
        coverSum   += partial.data[base + 5];
        symLeftSum += partial.data[base + 6];
        symRightSum+= partial.data[base + 7];
        pixCount   += uint(partial.data[base + 8]);
        novelty     = partial.data[base + 9];
        for (uint b = 0u; b < 16u; b++) {
            hist[b] += partial.data[base + 10 + b];
        }
    }

    float inv = (pixCount > 0u) ? (1.0 / float(pixCount)) : 0.0;

    float mean = lumSum * inv;
    float var  = max(lumSumSq * inv - mean * mean, 0.0);

    float contrast = var;

    float symDiff = abs(symLeftSum - symRightSum);
    float symScore = 1.0 - min(symDiff / max(max(lumSum, 1e-6), 1e-6), 1.0);

    float coverage = coverSum * inv;

    float entropy = entropy_from_hist(hist, pixCount);

    float motion = motionSum * inv;

    float beautyVal = 0.0;
    beautyVal += min(contrast * 20.0, 1.0) * 0.30;
    beautyVal += symScore * 0.25;
    beautyVal += min(coverage, 1.0) * 0.20;
    beautyVal += min(edgeSum * inv * 10.0, 1.0) * 0.25;

    float energy = mean * motion;

    outputBuf.outData[0] = clamp(beautyVal, 0.0, 1.0);
    outputBuf.outData[1] = clamp(motion, 0.0, 1.0);
    outputBuf.outData[2] = clamp(contrast, 0.0, 1.0);
    outputBuf.outData[3] = clamp(novelty, 0.0, 1.0);
    outputBuf.outData[4] = clamp(entropy, 0.0, 1.0);
    outputBuf.outData[5] = clamp(coverage, 0.0, 1.0);
    outputBuf.outData[6] = clamp(symScore, 0.0, 1.0);
    outputBuf.outData[7] = clamp(energy, 0.0, 1.0);
}
"""

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
uniform sampler2D uCurrent;
uniform sampler2D uPrevious;
uniform float uNovelty;
uniform int uRes;
float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}
void main() {
    ivec2 res = ivec2(uRes);
    float lumSum = 0.0, lumSumSq = 0.0, motionSum = 0.0;
    float edgeSum = 0.0, brightSum = 0.0, coverSum = 0.0;
    float symLeftSum = 0.0, symRightSum = 0.0;
    uint hist[16];
    for (int i = 0; i < 16; i++) hist[i] = 0u;
    uint pixCount = 0u;

    for (int y = 0; y < res.y; y++) {
        for (int x = 0; x < res.x; x++) {
            ivec2 gid = ivec2(x, y);
            vec3 cur = texelFetch(uCurrent, gid, 0).rgb;
            float lum = luminance(cur);
            lumSum += lum;
            lumSumSq += lum * lum;
            brightSum += dot(cur, vec3(1.0)) / 3.0;
            coverSum += lum > 0.02 ? 1.0 : 0.0;

            int bin = int(clamp(lum * 16.0, 0.0, 15.999));
            hist[bin]++;

            float l = luminance(texelFetch(uCurrent,
                clamp(gid + ivec2(-1, 0), ivec2(0), ivec2(res - 1)), 0).rgb);
            float r = luminance(texelFetch(uCurrent,
                clamp(gid + ivec2( 1, 0), ivec2(0), ivec2(res - 1)), 0).rgb);
            float u = luminance(texelFetch(uCurrent,
                clamp(gid + ivec2(0, -1), ivec2(0), ivec2(res - 1)), 0).rgb);
            float d = luminance(texelFetch(uCurrent,
                clamp(gid + ivec2(0,  1), ivec2(0), ivec2(res - 1)), 0).rgb);
            edgeSum += abs(l - r) + abs(u - d);

            vec3 prev = texelFetch(uPrevious, gid, 0).rgb;
            motionSum += abs(lum - luminance(prev));

            int halfW = res.x / 2;
            if (gid.x < halfW) {
                vec3 left = texelFetch(uCurrent, gid, 0).rgb;
                vec3 right = texelFetch(uCurrent,
                    ivec2(res.x - 1 - gid.x, gid.y), 0).rgb;
                symLeftSum += luminance(left);
                symRightSum += luminance(right);
            }
            pixCount++;
        }
    }

    float inv = (pixCount > 0u) ? (1.0 / float(pixCount)) : 0.0;
    float mean = lumSum * inv;
    float var = max(lumSumSq * inv - mean * mean, 0.0);

    float symDiff = abs(symLeftSum - symRightSum);
    float symScore = 1.0 - min(symDiff / max(max(lumSum, 1e-6), 1e-6), 1.0);

    float coverage = coverSum * inv;
    float contrast = var;

    float e = 0.0;
    for (int i = 0; i < 16; i++) {
        float p = float(hist[i]) * inv;
        if (p > 0.0) {
            e -= p * log2(p);
        }
    }
    float entropy = e / 4.0;

    float motion = motionSum * inv;

    float beautyVal = 0.0;
    beautyVal += min(contrast * 20.0, 1.0) * 0.30;
    beautyVal += symScore * 0.25;
    beautyVal += min(coverage, 1.0) * 0.20;
    beautyVal += min(edgeSum * inv * 10.0, 1.0) * 0.25;

    float energy = mean * motion;

    fragColor = vec4(
        clamp(beautyVal, 0.0, 1.0),
        clamp(motion, 0.0, 1.0),
        clamp(contrast, 0.0, 1.0),
        clamp(uNovelty, 0.0, 1.0)
    );
}
"""

_FALLBACK_FINALIZE_FRAG2 = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D uCurrent;
uniform sampler2D uPrevious;
uniform float uNovelty;
uniform int uRes;
float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}
void main() {
    ivec2 res = ivec2(uRes);
    float lumSum = 0.0, lumSumSq = 0.0, motionSum = 0.0;
    float edgeSum = 0.0, brightSum = 0.0, coverSum = 0.0;
    float symLeftSum = 0.0, symRightSum = 0.0;
    uint hist[16];
    for (int i = 0; i < 16; i++) hist[i] = 0u;
    uint pixCount = 0u;

    for (int y = 0; y < res.y; y++) {
        for (int x = 0; x < res.x; x++) {
            ivec2 gid = ivec2(x, y);
            vec3 cur = texelFetch(uCurrent, gid, 0).rgb;
            float lum = luminance(cur);
            lumSum += lum;
            lumSumSq += lum * lum;
            brightSum += dot(cur, vec3(1.0)) / 3.0;
            coverSum += lum > 0.02 ? 1.0 : 0.0;

            int bin = int(clamp(lum * 16.0, 0.0, 15.999));
            hist[bin]++;

            float l = luminance(texelFetch(uCurrent,
                clamp(gid + ivec2(-1, 0), ivec2(0), ivec2(res - 1)), 0).rgb);
            float r = luminance(texelFetch(uCurrent,
                clamp(gid + ivec2( 1, 0), ivec2(0), ivec2(res - 1)), 0).rgb);
            float u = luminance(texelFetch(uCurrent,
                clamp(gid + ivec2(0, -1), ivec2(0), ivec2(res - 1)), 0).rgb);
            float d = luminance(texelFetch(uCurrent,
                clamp(gid + ivec2(0,  1), ivec2(0), ivec2(res - 1)), 0).rgb);
            edgeSum += abs(l - r) + abs(u - d);

            vec3 prev = texelFetch(uPrevious, gid, 0).rgb;
            motionSum += abs(lum - luminance(prev));

            int halfW = res.x / 2;
            if (gid.x < halfW) {
                vec3 left = texelFetch(uCurrent, gid, 0).rgb;
                vec3 right = texelFetch(uCurrent,
                    ivec2(res.x - 1 - gid.x, gid.y), 0).rgb;
                symLeftSum += luminance(left);
                symRightSum += luminance(right);
            }
            pixCount++;
        }
    }

    float inv = (pixCount > 0u) ? (1.0 / float(pixCount)) : 0.0;
    float mean = lumSum * inv;
    float var = max(lumSumSq * inv - mean * mean, 0.0);

    float symDiff = abs(symLeftSum - symRightSum);
    float symScore = 1.0 - min(symDiff / max(max(lumSum, 1e-6), 1e-6), 1.0);

    float coverage = coverSum * inv;
    float contrast = var;

    float e = 0.0;
    for (int i = 0; i < 16; i++) {
        float p = float(hist[i]) * inv;
        if (p > 0.0) {
            e -= p * log2(p);
        }
    }
    float entropy = e / 4.0;

    float motion = motionSum * inv;

    float beautyVal = 0.0;
    beautyVal += min(contrast * 20.0, 1.0) * 0.30;
    beautyVal += symScore * 0.25;
    beautyVal += min(coverage, 1.0) * 0.20;
    beautyVal += min(edgeSum * inv * 10.0, 1.0) * 0.25;

    float energy = mean * motion;

    fragColor = vec4(
        clamp(entropy, 0.0, 1.0),
        clamp(coverage, 0.0, 1.0),
        clamp(symScore, 0.0, 1.0),
        clamp(energy, 0.0, 1.0)
    );
}
"""


def _gl_check(label: str) -> None:
    err = GL.glGetError()
    if err != GL.GL_NO_ERROR:
        raise RuntimeError(f"{label}: GL error 0x{err:04X}")


class AsyncFloatReadback:
    def __init__(self) -> None:
        self._pbos: list[int] = []
        self._write_idx = 0
        self._has_data = False
        self._last_values: tuple[float, ...] = (0.0,) * OUTPUT_FLOATS
        self.gl_ready = False

    def init_gl(self) -> bool:
        try:
            self._pbos = list(GL.glGenBuffers(3))
            nbytes = OUTPUT_FLOATS * 4
            for p in self._pbos:
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, p)
                GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, nbytes, None, GL.GL_STREAM_READ)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            self.gl_ready = True
            return True
        except error.Error as e:
            logger.warning("AsyncFloatReadback init failed: %s", e)
            self.gl_ready = False
            return False

    def cleanup(self) -> None:
        if self._pbos:
            try:
                GL.glDeleteBuffers(len(self._pbos), self._pbos)
            except Exception:
                logger.debug("AsyncFloatReadback cleanup failed", exc_info=True)
        self._pbos = []
        self.gl_ready = False

    def read_into_pbo(self) -> None:
        pass

    def collect(self, fence: int | None = None) -> tuple[float, ...] | None:
        if not self.gl_ready or len(self._pbos) != 3 or not self._has_data:
            return self._last_values

        if fence is not None:
            wait_result = GL.glClientWaitSync(fence, GL.GL_SYNC_FLUSH_COMMANDS_BIT, 100000000)
            if wait_result == GL.GL_WAIT_FAILED:
                logger.debug("Fence wait failed")
                return self._last_values
            elif wait_result == GL.GL_TIMEOUT_EXPIRED:
                return None

        nbytes = OUTPUT_FLOATS * 4
        buf = (GL.GLfloat * OUTPUT_FLOATS)()
        read_idx = (self._write_idx + 1) % 3
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbos[read_idx])
        try:
            GL.glGetBufferSubData(GL.GL_PIXEL_PACK_BUFFER, 0, nbytes, buf)
            data = tuple(float(buf[i]) for i in range(OUTPUT_FLOATS))
        except error.Error:
            data = self._last_values
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self._last_values = data
        self._has_data = True
        return data


class GpuFitnessPipeline:
    def __init__(self, res: int = 64, weights: dict[str, float] | None = None) -> None:
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
        self._fallback_finalize_prog2: int | None = None
        self._out_pbo: int | None = None
        self._out_ssbo: int | None = None
        self._partial_ssbo: int | None = None
        self._partial_wg = 0
        self._partial_per_wg = 0
        self._reduce_chain: list[tuple[int, int]] = []
        self._finalize_fbo: int | None = None
        self._finalize_tex: int | None = None
        self._vao: int | None = None
        self._vbo: int | None = None

        self._readback = AsyncFloatReadback()
        self._last_metrics: tuple[float, ...] = (0.0,) * OUTPUT_FLOATS
        self._dispatch_fence: int | None = None

    def init_gl(self) -> bool:
        try:
            self.using_compute = self._have_compute_shaders()
            if self.using_compute:
                self._init_compute()
            else:
                self._init_fallback()
            self._readback.init_gl()
            self.available = True
            logger.info(
                "GpuFitnessPipeline ready (compute=%s, res=%d)", self.using_compute, self.res
            )
            return True
        except Exception as e:
            logger.warning("GpuFitnessPipeline init failed: %s", e)
            self.available = False
            self._cleanup_gl_objects()
            return False

    @staticmethod
    def _have_compute_shaders() -> bool:
        try:
            if getattr(GL, "glDispatchCompute", None) is None:
                return False
            if getattr(GL, "glBindBufferBase", None) is None:
                return False
            if getattr(GL, "glGetIntegerv", None) is None:
                return False
            major = GL.glGetIntegerv(GL.GL_MAJOR_VERSION)
            minor = GL.glGetIntegerv(GL.GL_MINOR_VERSION)
            return (major, minor) >= (4, 3)
        except Exception:
            return False

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
            except Exception:
                logger.debug("Failed to delete shader on compile error", exc_info=True)
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
                except Exception:
                    logger.debug("Failed to detach/delete shader", exc_info=True)
            try:
                GL.glDeleteProgram(prog)
            except Exception:
                logger.debug("Failed to delete program on link error", exc_info=True)
            raise RuntimeError(f"Program link failed: {log}")
        for s in shaders:
            GL.glDetachShader(prog, s)
            GL.glDeleteShader(s)
        return prog

    def _compile_program(self, *args) -> int:
        shaders = []
        try:
            for i in range(0, len(args), 2):
                shaders.append(self._compile_shader(args[i], args[i + 1]))
            return self._link_program(*shaders)
        except Exception:
            for s in shaders:
                try:
                    GL.glDeleteShader(s)
                except Exception:
                    logger.debug("Failed to delete shader during cleanup", exc_info=True)
            raise

    def _init_compute(self) -> None:
        wg_x = (self.res + 7) // 8
        wg_y = (self.res + 7) // 8
        self._partial_wg = wg_x * wg_y
        self._partial_per_wg = 27

        self._reduce_prog = self._compile_program(GL.GL_COMPUTE_SHADER, _REDUCE_PASS)
        self._finalize_prog = self._compile_program(GL.GL_COMPUTE_SHADER, _FINALIZE_COMPUTE)

        self._partial_ssbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._partial_ssbo)
        GL.glBufferData(
            GL.GL_SHADER_STORAGE_BUFFER,
            self._partial_wg * self._partial_per_wg * 4,
            None,
            GL.GL_STREAM_READ,
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self._out_ssbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._out_ssbo)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, OUTPUT_FLOATS * 4, None, GL.GL_STREAM_READ)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        _gl_check("compute SSBO alloc")

    def _dispatch_compute(self, current_tex: int, previous_tex: int, novelty: float) -> bool:
        try:
            GL.glUseProgram(self._reduce_prog)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, current_tex)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, previous_tex)

            loc_res = GL.glGetUniformLocation(self._reduce_prog, "uRes")
            loc_nov = GL.glGetUniformLocation(self._reduce_prog, "uNovelty")
            if loc_res >= 0:
                GL.glUniform1i(loc_res, self.res)
            if loc_nov >= 0:
                GL.glUniform1f(loc_nov, float(novelty))

            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, self._partial_ssbo)

            wg_x = (self.res + 7) // 8
            wg_y = (self.res + 7) // 8
            GL.glDispatchCompute(wg_x, wg_y, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            # Finalize pass: sum all partials into the output SSBO.
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._out_ssbo)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, self._partial_ssbo)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, self._out_ssbo)
            GL.glUseProgram(self._finalize_prog)
            GL.glUniform1ui(
                GL.glGetUniformLocation(self._finalize_prog, "uWgCount"), self._partial_wg
            )
            GL.glDispatchCompute(1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            # Copy out SSBO to dedicated readback PBO (index 0).
            if self._readback.gl_ready and len(self._readback._pbos) >= 3:
                GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, self._out_ssbo)
                GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, self._readback._pbos[0])
                GL.glCopyBufferSubData(
                    GL.GL_COPY_READ_BUFFER, GL.GL_COPY_WRITE_BUFFER, 0, 0, OUTPUT_FLOATS * 4
                )
                GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, 0)
                GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, 0)

            # Insert fence for CPU-GPU sync
            self._dispatch_fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            GL.glUseProgram(0)
            return True
        except Exception as e:
            logger.debug("Compute dispatch failed: %s", e)
            return False

    def _init_fallback(self) -> None:
        self._fallback_reduce_prog = self._compile_program(
            GL.GL_VERTEX_SHADER, _FALLBACK_VERT, GL.GL_FRAGMENT_SHADER, _FALLBACK_REDUCE_FRAG
        )
        self._fallback_finalize_prog = self._compile_program(
            GL.GL_VERTEX_SHADER, _FALLBACK_VERT, GL.GL_FRAGMENT_SHADER, _FALLBACK_FINALIZE_FRAG
        )
        self._fallback_finalize_prog2 = self._compile_program(
            GL.GL_VERTEX_SHADER, _FALLBACK_VERT, GL.GL_FRAGMENT_SHADER, _FALLBACK_FINALIZE_FRAG2
        )

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

        size = self.res
        self._reduce_chain = []
        while size > 1:
            size = max(1, size // 2)
            fbo = GL.glGenFramebuffers(1)
            tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA16F,
                size,
                size,
                0,
                GL.GL_RGBA,
                GL.GL_HALF_FLOAT,
                None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
            )
            self._reduce_chain.append((fbo, tex))

        self._finalize_fbo = GL.glGenFramebuffers(1)
        self._finalize_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._finalize_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, 2, 1, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._finalize_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._finalize_tex, 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        nbytes = OUTPUT_FLOATS * 4
        self._out_pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._out_pbo)
        GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, nbytes, None, GL.GL_STREAM_READ)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

    def _dispatch_fallback(self, current_tex: int, previous_tex: int, novelty: float) -> bool:
        try:
            GL.glBindVertexArray(self._vao)
            src = current_tex
            size = self.res
            for fbo, tex in self._reduce_chain:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
                GL.glViewport(0, 0, max(1, size // 2), max(1, size // 2))
                GL.glUseProgram(self._fallback_reduce_prog)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, src)
                GL.glUniform1i(GL.glGetUniformLocation(self._fallback_reduce_prog, "uSrc"), 0)
                GL.glUniform1f(
                    GL.glGetUniformLocation(self._fallback_reduce_prog, "uTexel"),
                    1.0 / float(size),
                )
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
                src = tex
                size = size // 2 or 1

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._finalize_fbo)
            GL.glViewport(0, 0, 2, 1)

            # First pass: beauty, motion, contrast, novelty
            GL.glUseProgram(self._fallback_finalize_prog)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, current_tex)
            GL.glUniform1i(GL.glGetUniformLocation(self._fallback_finalize_prog, "uCurrent"), 0)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, previous_tex)
            GL.glUniform1i(GL.glGetUniformLocation(self._fallback_finalize_prog, "uPrevious"), 1)
            GL.glUniform1f(
                GL.glGetUniformLocation(self._fallback_finalize_prog, "uNovelty"), float(novelty)
            )
            GL.glUniform1i(GL.glGetUniformLocation(self._fallback_finalize_prog, "uRes"), self.res)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

            # Second pass: entropy, coverage, symmetry, energy
            GL.glUseProgram(self._fallback_finalize_prog2)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, current_tex)
            GL.glUniform1i(GL.glGetUniformLocation(self._fallback_finalize_prog2, "uCurrent"), 0)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, previous_tex)
            GL.glUniform1i(GL.glGetUniformLocation(self._fallback_finalize_prog2, "uPrevious"), 1)
            GL.glUniform1f(
                GL.glGetUniformLocation(self._fallback_finalize_prog2, "uNovelty"), float(novelty)
            )
            GL.glUniform1i(
                GL.glGetUniformLocation(self._fallback_finalize_prog2, "uRes"), self.res
            )
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._out_pbo)
            GL.glReadPixels(0, 0, 2, 1, GL.GL_RGBA, GL.GL_FLOAT, None)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            return True
        except Exception as e:
            logger.debug("Fallback dispatch failed: %s", e)
            return False

    def dispatch(self, current_tex: int, previous_tex: int = 0, novelty: float = 0.0) -> bool:
        if not self.available or current_tex == 0:
            return False
        try:
            if self.using_compute:
                ok = self._dispatch_compute(current_tex, previous_tex, novelty)
                if ok:
                    self._readback._has_data = True
            else:
                ok = self._dispatch_fallback(current_tex, previous_tex, novelty)
            return ok
        except Exception as e:
            logger.debug("GpuFitnessPipeline dispatch failed: %s", e)
            return False

    def read_metrics(self, block: bool = False) -> tuple[float, ...] | None:
        if not self.available:
            return None
        if not block:
            if self._last_metrics != (0.0,) * OUTPUT_FLOATS:
                return self._last_metrics
            return None
        if self.using_compute:
            metrics = self._readback.collect(self._dispatch_fence)
            if self._dispatch_fence is not None:
                GL.glDeleteSync(self._dispatch_fence)
                self._dispatch_fence = None
            if metrics is not None:
                self._last_metrics = metrics
                self._readback._write_idx = (self._readback._write_idx + 1) % 3
        else:
            # Fallback path: read directly from PBO
            nbytes = OUTPUT_FLOATS * 4
            buf = (GL.GLfloat * OUTPUT_FLOATS)()
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._out_pbo)
            try:
                GL.glGetBufferSubData(GL.GL_PIXEL_PACK_BUFFER, 0, nbytes, buf)
                data = tuple(float(buf[i]) for i in range(OUTPUT_FLOATS))
            except error.Error:
                data = self._last_metrics
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            self._last_metrics = data
            metrics = data
        return metrics

    def compute_objectives(self, metrics: tuple[float, ...]) -> dict[str, float]:
        return {
            "beauty": float(metrics[OUT_BEAUTY]) if metrics else 0.0,
            "motion": float(metrics[OUT_MOTION]) if metrics else 0.0,
            "contrast": float(metrics[OUT_CONTRAST]) if metrics else 0.0,
            "novelty": float(metrics[OUT_NOVELTY]) if metrics else 0.0,
            "entropy": float(metrics[OUT_ENTROPY]) if metrics else 0.0,
            "coverage": float(metrics[OUT_COVERAGE]) if metrics else 0.0,
            "symmetry": float(metrics[OUT_SYMMETRY]) if metrics else 0.0,
            "energy": float(metrics[OUT_ENERGY]) if metrics else 0.0,
        }

    def aggregate_fitness(self, metrics: tuple[float, ...]) -> float:
        if not metrics:
            return 0.0
        obj = self.compute_objectives(metrics)
        total = 0.0
        for key, val in obj.items():
            total += self.weights.get(key, 0.0) * max(0.0, min(1.0, val))
        return float(min(total, 1.0))

    def _cleanup_gl_objects(self) -> None:
        for attr in (
            "_reduce_prog",
            "_finalize_prog",
            "_fallback_reduce_prog",
            "_fallback_finalize_prog",
            "_fallback_finalize_prog2",
        ):
            prog = getattr(self, attr, None)
            if prog is not None:
                try:
                    GL.glDeleteProgram(prog)
                except Exception:
                    logger.debug("Failed to delete program", exc_info=True)
                setattr(self, attr, None)
        for buf_attr in ("_out_ssbo", "_partial_ssbo", "_out_pbo", "_vbo"):
            buf = getattr(self, buf_attr, None)
            if buf is not None:
                try:
                    GL.glDeleteBuffers(1, [buf])
                except Exception:
                    logger.debug("Failed to delete buffer", exc_info=True)
                setattr(self, attr, None)
        for fbo, tex in self._reduce_chain:
            try:
                GL.glDeleteFramebuffers(1, [fbo])
                GL.glDeleteTextures(1, [tex])
            except Exception:
                logger.debug("Failed to delete framebuffer or texture", exc_info=True)
        self._reduce_chain = []
        for attr in ("_finalize_fbo", "_finalize_tex", "_vao"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    if attr == "_finalize_fbo":
                        GL.glDeleteFramebuffers(1, [obj])
                    elif attr == "_finalize_tex":
                        GL.glDeleteTextures(1, [obj])
                    elif attr == "_vao":
                        GL.glDeleteVertexArrays(1, [obj])
                except Exception:
                    logger.debug("Failed to delete GL object", exc_info=True)
                setattr(self, attr, None)
        self._readback.cleanup()

    def cleanup(self) -> None:
        self._cleanup_gl_objects()
        self.available = False
        self.using_compute = False
