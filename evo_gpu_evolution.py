#!/usr/bin/env python3

"""evo_gpu_evolution.py

GPU-resident evolution pipeline for the EvoRenderer project.

The architecture moves the full evolution cycle onto the GPU compute
shader so that populations can scale into the hundreds or thousands
without CPU bottlenecks:

    Population texture          (genome params stored on GPU)
        ↓
    Fitness compute shader      (parallel fitness evaluation)
        ↓
    Fitness texture / SSBO      (fitness scores stay on GPU)
        ↓
    Non-dominated sorting       (NSGA-II rank assignment on GPU)
        ↓
    Crowding distance           (diversity preservation)
        ↓
    Selection compute shader    (binary tournament on rank + crowding)
        ↓
    SBX Crossover + Polynomial Mutation (offspring on GPU)
        ↓
    Next generation texture     (ready on GPU)

The CPU only reads back the single winner genome when it needs to
present the champion organism.  All intermediate population data
never leaves the GPU.
"""

from __future__ import annotations

import logging
import random
import struct
from typing import Any

from OpenGL import GL

try:
    import numpy as np
except ImportError:
    np = None

from evo_gpu_pool import FboPool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POP_SIZE = 128
DEFAULT_GENOME_WIDTH = 32  # float params per organism
LOCAL_SIZE_X = 64  # compute work-group width
OUTPUT_STRIDE = 4  # floats per output slot (fitness, rank, crowding, parent)

# NSGA-II constants
MAX_OBJECTIVES = 6  # beauty, novelty, complexity, motion, audio_sync, stability

# ---------------------------------------------------------------------------
# GLSL compute shaders
# ---------------------------------------------------------------------------

# Shared preamble - defines the genome layout (32 float params per organism
# stored row-major in a R32F texture).  All compute shaders include this.
_EVO_PREAMBLE = (
    """
#version 430 core

layout(local_size_x = """
    + str(LOCAL_SIZE_X)
    + """, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D uPopulation;
uniform int uPopSize;
uniform int uGenomeWidth;

layout(std430, binding = 1) buffer FitnessBuf {
    float fitness[];
};

layout(std430, binding = 2) buffer IndexBuf {
    uint indices[];
};

layout(std430, binding = 3) buffer ObjectiveBuf {
    float objectives[];  // [pop_size * MAX_OBJECTIVES]
};

layout(std430, binding = 4) buffer RankBuf {
    int rank[];  // non-domination rank per individual
};

layout(std430, binding = 5) buffer CrowdingBuf {
    float crowding[];  // crowding distance per individual
};

layout(binding = 6) uniform image2D uOffspring;

const int GW = """
    + str(DEFAULT_GENOME_WIDTH)
    + """;
const int MAX_OBJ = """ + str(MAX_OBJECTIVES) + """;

float param(int gid, int p) {
    ivec2 uv = ivec2(p, gid);
    return texelFetch(uPopulation, uv, 0).x;
}

void set_param(int gid, int p, float v) {
    imageStore(uOffspring, ivec2(p, gid), vec4(clamp(v, 0.0, 1.0), 0.0, 0.0, 1.0));
}

// PCG RNG - better statistical properties than xorshift
// Using 32-bit constants for GLSL compatibility
uint pcg_state;
uint pcg_inc;

uint pcg_next() {
    uint old_state = pcg_state;
    pcg_state = old_state * 1664525u + (pcg_inc | 1u);
    uint xorshifted = ((old_state >> 18u) ^ old_state) >> 27u;
    uint rot = old_state >> 27u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

float pcg_float() {
    return float(pcg_next() & 0x7FFFFFFF) / float(0x7FFFFFFF);
}

// Initialize RNG per-thread
void rng_init(uint gid) {
    pcg_state = gid * 2654435761u + 1u;
    pcg_inc = gid * 1442695040u + 1u;
}

// Gaussian via Box-Muller (two calls produce two values)
float rng_gauss() {
    float u1 = max(pcg_float(), 1e-6);
    float u2 = pcg_float();
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

// SBX crossover helper
float sbx_crossover(float x1, float x2, float eta) {
    float u = pcg_float();
    float beta;
    if (u <= 0.5) {
        beta = pow(2.0 * u, 1.0 / (eta + 1.0));
    } else {
        beta = pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0));
    }
    return 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2);
}

// Polynomial mutation helper
float polynomial_mutation(float x, float eta, float prob) {
    if (pcg_float() >= prob) return x;
    float u = pcg_float();
    float delta;
    if (u < 0.5) {
        delta = pow(2.0 * u, 1.0 / (eta + 1.0)) - 1.0;
    } else {
        delta = 1.0 - pow(2.0 * (1.0 - u), 1.0 / (eta + 1.0));
    }
    return clamp(x + delta, 0.0, 1.0);
}
"""
)

# ---------------------------------------------------------------------------
# Fitness compute shader
# ---------------------------------------------------------------------------
# Evaluates multiple objectives for each organism based on its genome
# parameter vector. Each work-group handles one organism (its row in the
# population texture). Outputs 6 objectives to ObjectiveBuf SSBO.

_FITNESS_COMPUTE = (
    _EVO_PREAMBLE
    + """

layout(std430, binding = 1) buffer FitnessBuf {
    float fitness[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(uPopSize)) return;

    rng_init(gid);

    float warp         = param(int(gid), 0);
    float noise_scale  = param(int(gid), 1);
    float feedback     = param(int(gid), 2);
    float rotation     = param(int(gid), 3);
    float symmetry     = param(int(gid), 4);
    float colour_shift = param(int(gid), 5);
    float flow_speed   = param(int(gid), 6);
    float glow         = param(int(gid), 7);
    float contrast     = param(int(gid), 8);
    float vortex       = param(int(gid), 9);
    float particle_density = param(int(gid), 10);
    float domain_warp  = param(int(gid), 11);
    float turbulence   = param(int(gid), 12);
    float fractal_layers = param(int(gid), 13);
    float curl_strength = param(int(gid), 14);
    float field_type   = param(int(gid), 15);
    float feedback_mode = param(int(gid), 16);
    float mirror       = param(int(gid), 17);
    float dimension    = param(int(gid), 18);
    float camera_speed = param(int(gid), 19);
    float camera_orbit = param(int(gid), 20);
    float camera_depth = param(int(gid), 21);
    float camera_fov   = param(int(gid), 22);
    float depth_scale  = param(int(gid), 23);
    float ray_steps    = param(int(gid), 24);
    float fog_density  = param(int(gid), 25);
    float colour_palette = param(int(gid), 26);
    float colour_preset = param(int(gid), 27);
    float brightness   = param(int(gid), 28);
    float exposure     = param(int(gid), 29);
    float complexity   = param(int(gid), 30);
    float novelty_weight = param(int(gid), 31);

    // Objective 0: Beauty (visual appeal)
    float beauty = 0.0;
    beauty += warp * 0.12;
    beauty += noise_scale * 0.08;
    beauty += feedback * 0.10;
    beauty += glow * 0.10;
    beauty += contrast * 0.08;
    beauty += colour_shift * 0.06;
    beauty += symmetry * 0.08;
    beauty += flow_speed * 0.06;
    beauty += domain_warp * 0.06;
    beauty += turbulence * 0.06;
    beauty += fractal_layers * 0.05;
    beauty += curl_strength * 0.04;
    beauty += vortex * 0.06;
    beauty += camera_speed * 0.03;
    beauty += camera_orbit * 0.03;
    beauty += camera_fov * 0.03;
    beauty += ray_steps * 0.03;
    beauty += colour_palette * 0.04;
    beauty += brightness * 0.02;
    beauty += exposure * 0.02;
    beauty += complexity * 0.03;

    // Objective 1: Novelty (diversity)
    float novelty = novelty_weight * 0.5 + (sin(colour_shift * 6.2831) * 0.5 + 0.5) * 0.5;

    // Objective 2: Complexity (structural complexity)
    float complexity_score = (fractal_layers / 32.0) * 0.15;
    complexity_score += min(turbulence, 2.0) * 0.05;
    complexity_score += min(domain_warp, 1.0) * 0.05;

    // Objective 3: Motion (dynamic movement)
    float motion = abs(vortex) * 0.15 + flow_speed * 0.10 + abs(camera_speed) * 0.05 + abs(camera_orbit) * 0.05;

    // Objective 4: Audio sync (reactivity)
    float audio_sync = feedback * 0.15 + abs(vortex) * 0.10 + particle_density * 0.05;

    // Objective 5: Stability (parameter coherence)
    float stability = 1.0 - (abs(warp - 0.5) + abs(noise_scale - 0.5) + abs(feedback - 0.5)) * 0.3;
    stability = max(0.0, min(1.0, stability));

    // Scalar fitness (weighted sum) for backward compatibility
    float total = (beauty * 0.40
                 + novelty * 0.15
                 + complexity_score * 0.10
                 + motion * 0.15
                 + audio_sync * 0.10
                 + stability * 0.10);
    total = clamp(total, 0.0, 1.0);
    fitness[gid] = total;

    // Store individual objectives for NSGA-II
    objectives[gid * MAX_OBJ + 0] = beauty;
    objectives[gid * MAX_OBJ + 1] = novelty;
    objectives[gid * MAX_OBJ + 2] = complexity_score;
    objectives[gid * MAX_OBJ + 3] = motion;
    objectives[gid * MAX_OBJ + 4] = audio_sync;
    objectives[gid * MAX_OBJ + 5] = stability;
}
"""
)

# ---------------------------------------------------------------------------
# NSGA-II Non-dominated sorting compute shader
# ---------------------------------------------------------------------------
# Computes non-domination ranks for all individuals using the fast
# non-dominated sorting algorithm. Runs in parallel with one thread per
# individual, but requires multiple passes (one per front).
# We use a simplified version: each thread compares against all others.

_NSGA2_SORT_COMPUTE = (
    _EVO_PREAMBLE
    + """

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(uPopSize)) return;

    rng_init(gid);

    // Read my objectives
    float my_obj[MAX_OBJ];
    for (int i = 0; i < MAX_OBJ; i++) {
        my_obj[i] = objectives[gid * MAX_OBJ + i];
    }

    // Count how many individuals dominate me, and who I dominate
    int n_dominate_me = 0;
    int dominated_count[128];  // max pop size
    for (int i = 0; i < 128; i++) dominated_count[i] = 0;
    uint dominated_list[128];
    int dominated_len = 0;

    for (uint j = 0; j < uint(uPopSize); j++) {
        if (j == gid) continue;

        float other_obj[MAX_OBJ];
        for (int i = 0; i < MAX_OBJ; i++) {
            other_obj[i] = objectives[j * MAX_OBJ + i];
        }

        bool i_dominates_j = true;
        bool j_dominates_i = true;
        for (int i = 0; i < MAX_OBJ; i++) {
            if (my_obj[i] < other_obj[i]) i_dominates_j = false;
            if (my_obj[i] > other_obj[i]) j_dominates_i = false;
        }

        // Strict dominance: all objectives better or equal, at least one strictly better
        bool strict_i_dom_j = i_dominates_j;
        bool strict_j_dom_i = j_dominates_i;
        for (int i = 0; i < MAX_OBJ; i++) {
            if (my_obj[i] == other_obj[i]) {
                strict_i_dom_j = false;
                strict_j_dom_i = false;
            }
        }

        if (strict_j_dom_i) {
            n_dominate_me++;
        }
        if (strict_i_dom_j) {
            dominated_list[dominated_len++] = j;
        }
    }

    rank[gid] = n_dominate_me;

    // Store dominated count and list for crowding distance (simplified)
    // In a full implementation, we'd use atomic operations or a second pass
}
"""
)

# ---------------------------------------------------------------------------
# Crowding distance compute shader
# ---------------------------------------------------------------------------
# Calculates crowding distance for each front to maintain diversity.
# Each thread processes one objective for one front.

_CROWDING_COMPUTE = (
    _EVO_PREAMBLE
    + """

uniform int uFrontStart;   // starting index of this front in sorted population
uniform int uFrontSize;    // number of individuals in this front

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(uFrontSize)) return;

    uint idx = uint(uFrontStart) + gid;
    if (idx >= uint(uPopSize)) return;

    rng_init(gid);

    // Initialize crowding distance
    float cd = 0.0;

    // For each objective, sort and compute distance
    for (int obj = 0; obj < MAX_OBJ; obj++) {
        // Find min and max for this objective in the front
        float fmin = 1e9;
        float fmax = -1e9;
        for (int i = 0; i < uFrontSize; i++) {
            uint ii = uint(uFrontStart) + i;
            float val = objectives[ii * MAX_OBJ + obj];
            fmin = min(fmin, val);
            fmax = max(fmax, val);
        }
        if (fmax - fmin < 1e-6) continue;

        // Get my objective value
        float my_val = objectives[idx * MAX_OBJ + obj];

        // Find neighbors (simplified: assume sorted order in indices buffer)
        float prev_val = my_val, next_val = my_val;
        if (gid > 0) {
            uint prev_idx = uint(uFrontStart) + gid - 1;
            prev_val = objectives[prev_idx * MAX_OBJ + obj];
        }
        if (gid < uFrontSize - 1) {
            uint next_idx = uint(uFrontStart) + gid + 1;
            next_val = objectives[next_idx * MAX_OBJ + obj];
        }
        cd += (next_val - prev_val) / (fmax - fmin);
    }

    crowding[idx] = cd;
}
"""
)

# ---------------------------------------------------------------------------
# SBX Crossover compute shader
# ---------------------------------------------------------------------------

_CROSSOVER_COMPUTE = (
    _EVO_PREAMBLE
    + """

uniform float uCrossoverRate;
uniform float uSBXEta;  // distribution index for SBX (typically 15-20)

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(uPopSize)) return;

    rng_init(gid);

    // Parents are stored in indices buffer (two per offspring)
    uint pa = indices[gid * 2];
    uint pb = indices[gid * 2 + 1];

    for (int p = 0; p < GW; p++) {
        float va = param(int(pa), p);
        float vb = param(int(pb), p);
        float child_val;
        if (pcg_float() < uCrossoverRate) {
            child_val = sbx_crossover(va, vb, uSBXEta);
        } else {
            child_val = (pcg_float() < 0.5) ? va : vb;
        }
        set_param(int(gid), p, child_val);
    }
}
"""
)

# ---------------------------------------------------------------------------
# Selection compute shader (NSGA-II binary tournament on rank + crowding)
# ---------------------------------------------------------------------------

_SELECTION_COMPUTE = (
    _EVO_PREAMBLE
    + """

uniform int uTournamentSize;
uniform float uEliteFraction;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(uPopSize)) return;

    rng_init(gid);

    uint elite_count = uint(ceil(float(uPopSize) * uEliteFraction));
    bool is_elite = gid < elite_count;

    if (is_elite) {
        indices[gid * 2] = gid;
        indices[gid * 2 + 1] = gid;
        return;
    }

    // Binary tournament: lower rank is better, if equal higher crowding is better
    int best_rank = 1000000;
    float best_crowding = -1.0;
    uint best_idx = 0;

    for (int i = 0; i < uTournamentSize; i++) {
        uint contender = uint(pcg_float() * float(uPopSize));
        if (contender >= uint(uPopSize)) contender = uint(uPopSize) - 1u;

        int cr = rank[contender];
        float cc = crowding[contender];

        if (cr < best_rank || (cr == best_rank && cc > best_crowding)) {
            best_rank = cr;
            best_crowding = cc;
            best_idx = contender;
        }
    }
    indices[gid * 2] = best_idx;
    indices[gid * 2 + 1] = best_idx;
}
"""
)

# ---------------------------------------------------------------------------
# Polynomial Mutation compute shader
# ---------------------------------------------------------------------------

_MUTATION_COMPUTE = (
    _EVO_PREAMBLE
    + """

uniform float uMutationRate;
uniform float uPolyEta;  // distribution index for polynomial mutation (typically 20)

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(uPopSize)) return;

    rng_init(gid);

    for (int p = 0; p < GW; p++) {
        float v = param(int(gid), p);
        v = polynomial_mutation(v, uPolyEta, uMutationRate);
        set_param(int(gid), p, v);
    }
}
"""
)

# ---------------------------------------------------------------------------
# C++ side: GpuPopulation
# ---------------------------------------------------------------------------


class GpuPopulation:
    """Manages the population data entirely on the GPU.

    Genome parameters are stored as a single ``R32F`` texture (row =
    organism, column = parameter).  The texture is backed by a texture
    pool so that resize / regeneration does not leak GL objects.

    Supporting buffers:

    * ``_fitness_ssbo``   - per-organism float fitness scores (scalar)
    * ``_objectives_ssbo`` - per-organism multi-objective scores [pop_size * MAX_OBJECTIVES]
    * ``_rank_ssbo``      - non-domination rank per individual
    * ``_crowding_ssbo``  - crowding distance per individual
    * ``_index_ssbo``     - selected parent indices for the next generation (2 per offspring)
    * ``_offspring_fbo`` / ``_offspring_tex`` — render target for
      crossover/mutation pass (same dimensions as the population texture)
    """

    def __init__(
        self,
        pop_size: int = DEFAULT_POP_SIZE,
        genome_width: int = DEFAULT_GENOME_WIDTH,
        pool: FboPool | None = None,
    ):
        self._pop_size = max(1, int(pop_size))
        self._genome_width = max(1, int(genome_width))
        self._pool = pool
        self._tex: int | None = None
        self._fitness_ssbo: int | None = None
        self._objectives_ssbo: int | None = None
        self._rank_ssbo: int | None = None
        self._crowding_ssbo: int | None = None
        self._index_ssbo: int | None = None
        self._offspring_tex: int | None = None
        self._offspring_fbo: int | None = None
        self._available = False

    # ------------------------------------------------------------------
    # GL resource management
    # ------------------------------------------------------------------

    def init_gl(self) -> bool:
        """Create all GPU objects.  Must be called from a live GL context."""
        try:
            self._tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_R32F,
                self._genome_width,
                self._pop_size,
                0,
                GL.GL_RED,
                GL.GL_FLOAT,
                None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            n_floats = self._pop_size
            self._fitness_ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._fitness_ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n_floats * 4, None, GL.GL_DYNAMIC_DRAW)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

            # Objectives buffer: pop_size * MAX_OBJECTIVES floats
            n_obj_floats = self._pop_size * MAX_OBJECTIVES
            self._objectives_ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._objectives_ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n_obj_floats * 4, None, GL.GL_DYNAMIC_DRAW)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

            # Rank buffer: pop_size ints
            self._rank_ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._rank_ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n_floats * 4, None, GL.GL_DYNAMIC_DRAW)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

            # Crowding distance buffer: pop_size floats
            self._crowding_ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._crowding_ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n_floats * 4, None, GL.GL_DYNAMIC_DRAW)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

            # Index buffer: pop_size * 2 uints (two parents per offspring)
            n_indices = self._pop_size * 2
            self._index_ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._index_ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n_indices * 4, None, GL.GL_DYNAMIC_DRAW)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

            # Offspring texture + FBO for the crossover/mutation passes.
            if self._pool is not None:
                self._offspring_tex, self._offspring_fbo = self._pool.acquire_fbo_with_texture(
                    self._genome_width, self._pop_size
                )
            else:
                self._offspring_tex = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._offspring_tex)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_R32F,
                    self._genome_width,
                    self._pop_size,
                    0,
                    GL.GL_RED,
                    GL.GL_FLOAT,
                    None,
                )
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                self._offspring_fbo = GL.glGenFramebuffers(1)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._offspring_fbo)
                GL.glFramebufferTexture2D(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_COLOR_ATTACHMENT0,
                    GL.GL_TEXTURE_2D,
                    self._offspring_tex,
                    0,
                )
                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                if status != GL.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError(f"Offspring FBO incomplete: {hex(status)}")
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

            self._available = True
            return True
        except RuntimeError as e:
            logger.warning("GpuPopulation init_gl failed: %s", e)
            self._cleanup()
            return False

    def _cleanup(self) -> None:
        for name in ("_tex", "_fitness_ssbo", "_objectives_ssbo", "_rank_ssbo", "_crowding_ssbo", "_index_ssbo", "_offspring_tex", "_offspring_fbo"):
            val = getattr(self, name, None)
            if val is not None:
                try:
                    if name == "_tex":
                        GL.glDeleteTextures(1, [val])
                    elif name in ("_fitness_ssbo", "_objectives_ssbo", "_rank_ssbo", "_crowding_ssbo", "_index_ssbo"):
                        GL.glDeleteBuffers(1, [val])
                    elif name == "_offspring_tex":
                        GL.glDeleteTextures(1, [val])
                    elif name == "_offspring_fbo":
                        GL.glDeleteFramebuffers(1, [val])
                except RuntimeError as exc:
                    logger.debug("cleanup failed: %s", exc)
                setattr(self, name, None)
        self._available = False

    def cleanup(self) -> None:
        self._cleanup()

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Population texture I/O
    # ------------------------------------------------------------------

    def populate_from_cpu(
        self,
        genomes: list[Any],
        param_extractor,
    ) -> None:
        """Upload genome parameters into the GPU population texture.

        Args:
            genomes: list of genome objects (CPU side)
            param_extractor: callable(genome) -> list[float] of length
                ``genome_width``.  Values must be in [0, 1]; values outside
                are clamped by the GPU shader anyway.
        """
        if not self._available or self._tex is None:
            return
        rows: list[float] = []
        for g in genomes:
            params = param_extractor(g)
            # Pad or truncate to genome_width.
            padded = (params + [0.0] * self._genome_width)[: self._genome_width]
            rows.extend(padded)
        if np is not None:
            data = np.array(rows, dtype=np.float32)
        else:
            data = struct.pack(f"{len(rows)}f", *rows)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._genome_width,
            self._pop_size,
            GL.GL_RED,
            GL.GL_FLOAT,
            data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def upload_random(self, num: int) -> None:
        """Fill the first ``num`` rows with random genome parameters."""
        if not self._available or self._tex is None:
            return
        rows: list[float] = []
        for _ in range(num):
            rows.extend(random.random() for _ in range(self._genome_width))
        rows.extend(0.0 for _ in range((self._pop_size - num) * self._genome_width))
        if np is not None:
            data = np.array(rows, dtype=np.float32)
        else:
            data = struct.pack(f"{len(rows)}f", *rows)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._genome_width,
            self._pop_size,
            GL.GL_RED,
            GL.GL_FLOAT,
            data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    # ------------------------------------------------------------------
    # Fitness readback
    # ------------------------------------------------------------------

    def read_fitness(self, count: int | None = None) -> list[float] | None:
        """Map the fitness SSBO and return a Python list of scores."""
        if not self._available or self._fitness_ssbo is None:
            return None
        n = count or self._pop_size
        try:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._fitness_ssbo)
            ptr = GL.glMapBufferRange(GL.GL_SHADER_STORAGE_BUFFER, 0, n * 4, GL.GL_MAP_READ_BIT)
            if ptr is None:
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
                return None
            arr = (GL.GLfloat * n).from_address(ptr)
            result = [float(arr[i]) for i in range(n)]
            GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            return result
        except RuntimeError as e:
            logger.debug("read_fitness failed: %s", e)
            try:
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            except RuntimeError as exc:
                logger.debug("cleanup failed: %s", exc)
            return None

    def read_best_genome(
        self,
        index: int,
        param_extractor_rev,
    ) -> list[float] | None:
        """Read back a single genome row from the population texture."""
        if not self._available or self._tex is None:
            return None
        n = self._genome_width
        try:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
            data = GL.glGetTexImage(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RED,
                GL.GL_FLOAT,
            )
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            # glGetTexImage returns the full texture; we need row `index`.
            row_start = index * self._genome_width
            return [float(data[row_start + i]) for i in range(n)]
        except RuntimeError as e:
            logger.debug("read_best_genome failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Compute dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch(
        self, prog: int, wg_count: int, extra_binds: dict[int, int] | None = None
    ) -> None:
        GL.glUseProgram(prog)
        GL.glBindTextureUnit(0, self._tex)
        GL.glUniform1i(GL.glGetUniformLocation(prog, "uPopulation"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(prog, "uPopSize"), self._pop_size)
        GL.glUniform1i(GL.glGetUniformLocation(prog, "uGenomeWidth"), self._genome_width)
        if extra_binds:
            for unit, tex_name in extra_binds.items():
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex_name)
                loc = GL.glGetUniformLocation(prog, tex_name)
                if loc >= 0:
                    GL.glUniform1i(loc, unit)
        GL.glDispatchCompute(wg_count, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        GL.glUseProgram(0)


# ---------------------------------------------------------------------------
# GpuEvolutionEngine
# ---------------------------------------------------------------------------


class GpuEvolutionEngine:
    """Orchestrates the full GPU-resident evolution pipeline with NSGA-II.

    The CPU only ever touches:
    * uploading initial genome data (the current champion genome params)
    * reading back the single winner
    * rebuilding a genome object from the float array

    Everything else — fitness evaluation, non-dominated sorting, crowding
    distance, selection, SBX crossover, polynomial mutation — runs on the
    GPU via compute shaders.

    Attributes:
        pop_size: number of organisms in the population
        genome_width: number of float parameters per organism (default 32)
        programs: compiled compute shader programs for each pipeline stage
    """

    def __init__(
        self,
        pop_size: int = DEFAULT_POP_SIZE,
        genome_width: int = DEFAULT_GENOME_WIDTH,
        tournament_size: int = 4,
        elite_fraction: float = 0.15,
        crossover_rate: float = 0.9,
        sbx_eta: float = 20.0,
        mutation_rate: float = 0.1,
        poly_eta: float = 20.0,
        pool: FboPool | None = None,
    ):
        self._pop_size = max(1, int(pop_size))
        self._genome_width = max(1, int(genome_width))
        self._tournament_size = max(2, int(tournament_size))
        self._elite_fraction = max(0.0, min(0.5, float(elite_fraction)))
        self._crossover_rate = max(0.0, min(1.0, float(crossover_rate)))
        self._sbx_eta = max(1.0, float(sbx_eta))
        self._mutation_rate = max(0.0, min(1.0, float(mutation_rate)))
        self._poly_eta = max(1.0, float(poly_eta))
        self._pool = pool

        self._available = False
        self._using_compute = False

        # Compiled programs
        self._fitness_prog: int | None = None
        self._nsga2_sort_prog: int | None = None
        self._crowding_prog: int | None = None
        self._selection_prog: int | None = None
        self._crossover_prog: int | None = None
        self._mutation_prog: int | None = None

        # Population buffer
        self._population = GpuPopulation(
            pop_size=self._pop_size,
            genome_width=self._genome_width,
            pool=pool,
        )

        # Work-group count
        self._wg_count = (self._pop_size + LOCAL_SIZE_X - 1) // LOCAL_SIZE_X

    # ------------------------------------------------------------------
    # GL initialisation
    # ------------------------------------------------------------------

    def init_gl(self) -> bool:
        """Compile compute shaders and create GPU resources.

        Returns ``True`` if GPU compute is available and all shaders
        compiled successfully.
        """
        if not _have_compute_shaders():
            logger.info("GPU compute shaders unavailable; evolution stays CPU")
            self._available = False
            return False

        try:
            self._fitness_prog = self._compile(_FITNESS_COMPUTE)
            self._nsga2_sort_prog = self._compile(_NSGA2_SORT_COMPUTE)
            self._crowding_prog = self._compile(_CROWDING_COMPUTE)
            self._selection_prog = self._compile(_SELECTION_COMPUTE)
            self._crossover_prog = self._compile(_CROSSOVER_COMPUTE)
            self._mutation_prog = self._compile(_MUTATION_COMPUTE)

            self._population.init_gl()

            self._available = True
            self._using_compute = True
            logger.info(
                "GpuEvolutionEngine ready (pop=%d, params=%d, wg=%d, NSGA-II)",
                self._pop_size,
                self._genome_width,
                self._wg_count,
            )
            return True
        except RuntimeError as e:
            logger.warning("GpuEvolutionEngine init failed, falling back to CPU: %s", e)
            self._cleanup()
            return False

    def _compile(self, source: str) -> int:
        prog = GL.glCreateProgram()
        cs = GL.glCreateShader(GL.GL_COMPUTE_SHADER)
        GL.glShaderSource(cs, source)
        GL.glCompileShader(cs)
        if GL.glGetShaderiv(cs, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
            log = GL.glGetShaderInfoLog(cs)
            GL.glDeleteShader(cs)
            GL.glDeleteProgram(prog)
            raise RuntimeError(f"Compute shader compile failed: {log}")
        GL.glAttachShader(prog, cs)
        GL.glLinkProgram(prog)
        if GL.glGetProgramiv(prog, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            log = GL.glGetProgramInfoLog(prog)
            GL.glDeleteShader(cs)
            GL.glDeleteProgram(prog)
            raise RuntimeError(f"Compute program link failed: {log}")
        GL.glDeleteShader(cs)
        return prog

    def _cleanup(self) -> None:
        for name in ("_fitness_prog", "_nsga2_sort_prog", "_crowding_prog", "_selection_prog", "_crossover_prog", "_mutation_prog"):
            prog = getattr(self, name, None)
            if prog is not None:
                try:
                    GL.glDeleteProgram(prog)
                except RuntimeError as exc:
                    logger.debug("cleanup failed: %s", exc)
                setattr(self, name, None)
        self._population.cleanup()
        self._available = False
        self._using_compute = False

    def cleanup(self) -> None:
        self._cleanup()

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Evolution step (NSGA-II pipeline)
    # ------------------------------------------------------------------

    def step(self, genomes: list[Any], param_extractor) -> list[Any] | None:
        """Execute one full GPU evolution step using NSGA-II.

        1. Upload population texture from CPU genome data.
        2. Dispatch fitness compute shader (multi-objective).
        3. Dispatch NSGA-II non-dominated sorting.
        4. Dispatch crowding distance calculation (per front).
        5. Dispatch selection (binary tournament on rank + crowding).
        6. Dispatch SBX crossover.
        7. Dispatch polynomial mutation.
        8. Read back the best genome (rank 0, highest crowding).

        Args:
            genomes: current population (CPU objects)
            param_extractor: callable(genome) -> list[float] (length genome_width)

        Returns:
            New population as CPU genome objects, or ``None`` if the GPU
            path failed and the caller should use its CPU fallback.
        """
        if not self._available:
            return None

        try:
            n = len(genomes)
            if n == 0:
                return None

            # ---- 1. Upload population ----
            self._population.populate_from_cpu(genomes, param_extractor)

            # ---- 2. Fitness evaluation (multi-objective) ----
            self._dispatch_fitness()

            # ---- 3. NSGA-II Non-dominated sorting ----
            self._dispatch_nsga2_sort()

            # ---- 4. Crowding distance (per front) ----
            self._dispatch_crowding()

            # ---- 5. Selection (binary tournament on rank + crowding) ----
            self._dispatch_selection()

            # ---- 6. SBX Crossover ----
            self._dispatch_crossover()

            # ---- 7. Polynomial Mutation ----
            self._dispatch_mutation()

            # ---- 8. Read back winner (best rank 0 individual) ----
            fitness_scores = self._population.read_fitness()
            if fitness_scores is None:
                return None

            # Find best individual: rank 0 with highest crowding distance
            best_idx = self._find_best_individual()
            if best_idx is None:
                best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])

            winner_params = self._population.read_best_genome(best_idx, param_extractor)
            if winner_params is None:
                return None

            # Reconstruct genomes for the new population from the offspring texture.
            return self._reconstruct_population(winner_params, fitness_scores)

        except RuntimeError as e:
            logger.debug("GpuEvolutionEngine step failed: %s", e)
            return None

    def _find_best_individual(self) -> int | None:
        """Find the best individual: rank 0 with highest crowding distance."""
        if not self._available:
            return None
        try:
            # Read rank buffer
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._population._rank_ssbo)
            rank_ptr = GL.glMapBufferRange(GL.GL_SHADER_STORAGE_BUFFER, 0, self._pop_size * 4, GL.GL_MAP_READ_BIT)
            if rank_ptr is None:
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
                return None
            rank_arr = (GL.GLint * self._pop_size).from_address(rank_ptr)
            ranks = [int(rank_arr[i]) for i in range(self._pop_size)]
            GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

            # Read crowding buffer
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._population._crowding_ssbo)
            crowding_ptr = GL.glMapBufferRange(GL.GL_SHADER_STORAGE_BUFFER, 0, self._pop_size * 4, GL.GL_MAP_READ_BIT)
            if crowding_ptr is None:
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
                return None
            crowding_arr = (GL.GLfloat * self._pop_size).from_address(crowding_ptr)
            crowding = [float(crowding_arr[i]) for i in range(self._pop_size)]
            GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

            # Find rank 0 individuals, pick one with highest crowding
            rank0_indices = [i for i, r in enumerate(ranks) if r == 0]
            if not rank0_indices:
                return 0
            best_idx = max(rank0_indices, key=lambda i: crowding[i])
            return best_idx
        except RuntimeError as e:
            logger.debug("_find_best_individual failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Individual dispatch passes
    # ------------------------------------------------------------------

    def _dispatch_fitness(self) -> None:
        if self._fitness_prog is None:
            return
        GL.glUseProgram(self._fitness_prog)
        GL.glBindTextureUnit(0, self._population._tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._fitness_prog, "uPopulation"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self._fitness_prog, "uPopSize"), self._pop_size)
        GL.glUniform1i(GL.glGetUniformLocation(self._fitness_prog, "uGenomeWidth"), self._genome_width)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, self._population._fitness_ssbo)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, self._population._objectives_ssbo)
        GL.glDispatchCompute(self._wg_count, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        GL.glUseProgram(0)

    def _dispatch_nsga2_sort(self) -> None:
        if self._nsga2_sort_prog is None:
            return
        GL.glUseProgram(self._nsga2_sort_prog)
        GL.glBindTextureUnit(0, self._population._tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._nsga2_sort_prog, "uPopulation"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self._nsga2_sort_prog, "uPopSize"), self._pop_size)
        GL.glUniform1i(GL.glGetUniformLocation(self._nsga2_sort_prog, "uGenomeWidth"), self._genome_width)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, self._population._objectives_ssbo)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 4, self._population._rank_ssbo)
        GL.glDispatchCompute(self._wg_count, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        GL.glUseProgram(0)

    def _dispatch_crowding(self) -> None:
        # For simplicity, compute crowding for the whole population as one front
        # A full implementation would iterate over fronts
        if self._crowding_prog is None:
            return
        GL.glUseProgram(self._crowding_prog)
        GL.glBindTextureUnit(0, self._population._tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._crowding_prog, "uPopulation"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self._crowding_prog, "uPopSize"), self._pop_size)
        GL.glUniform1i(GL.glGetUniformLocation(self._crowding_prog, "uGenomeWidth"), self._genome_width)
        GL.glUniform1i(GL.glGetUniformLocation(self._crowding_prog, "uFrontStart"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self._crowding_prog, "uFrontSize"), self._pop_size)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, self._population._objectives_ssbo)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 5, self._population._crowding_ssbo)
        GL.glDispatchCompute(self._wg_count, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        GL.glUseProgram(0)

    def _dispatch_selection(self) -> None:
        if self._selection_prog is None:
            return
        GL.glUseProgram(self._selection_prog)
        GL.glBindTextureUnit(0, self._population._tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._selection_prog, "uPopulation"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self._selection_prog, "uPopSize"), self._pop_size)
        GL.glUniform1i(GL.glGetUniformLocation(self._selection_prog, "uGenomeWidth"), self._genome_width)
        GL.glUniform1i(
            GL.glGetUniformLocation(self._selection_prog, "uTournamentSize"),
            self._tournament_size,
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self._selection_prog, "uEliteFraction"),
            self._elite_fraction,
        )
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, self._population._index_ssbo)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 4, self._population._rank_ssbo)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 5, self._population._crowding_ssbo)
        GL.glDispatchCompute(self._wg_count, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        GL.glUseProgram(0)

    def _dispatch_crossover(self) -> None:
        if self._crossover_prog is None:
            return
        GL.glUseProgram(self._crossover_prog)
        GL.glBindTextureUnit(0, self._population._tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._crossover_prog, "uPopulation"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self._crossover_prog, "uPopSize"), self._pop_size)
        GL.glUniform1i(GL.glGetUniformLocation(self._crossover_prog, "uGenomeWidth"), self._genome_width)
        GL.glUniform1f(
            GL.glGetUniformLocation(self._crossover_prog, "uCrossoverRate"),
            self._crossover_rate,
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self._crossover_prog, "uSBXEta"),
            self._sbx_eta,
        )
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, self._population._index_ssbo)
        GL.glBindImageTexture(
            0,
            self._population._offspring_tex,
            0,
            GL.GL_FALSE,
            0,
            GL.GL_WRITE_ONLY,
            GL.GL_R32F,
        )
        GL.glDispatchCompute(self._wg_count, 1, 1)
        GL.glMemoryBarrier(
            GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
        )
        GL.glUseProgram(0)

    def _dispatch_mutation(self) -> None:
        if self._mutation_prog is None:
            return
        GL.glUseProgram(self._mutation_prog)
        GL.glBindTextureUnit(0, self._population._offspring_tex)  # Read from offspring
        GL.glUniform1i(GL.glGetUniformLocation(self._mutation_prog, "uPopulation"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self._mutation_prog, "uPopSize"), self._pop_size)
        GL.glUniform1i(GL.glGetUniformLocation(self._mutation_prog, "uGenomeWidth"), self._genome_width)
        GL.glUniform1f(
            GL.glGetUniformLocation(self._mutation_prog, "uMutationRate"),
            self._mutation_rate,
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self._mutation_prog, "uPolyEta"),
            self._poly_eta,
        )
        GL.glBindImageTexture(
            0,
            self._population._offspring_tex,
            0,
            GL.GL_FALSE,
            0,
            GL.GL_READ_WRITE,
            GL.GL_R32F,
        )
        GL.glDispatchCompute(self._wg_count, 1, 1)
        GL.glMemoryBarrier(
            GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
        )
        GL.glUseProgram(0)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_population(
        self,
        winner_params: list[float],
        fitness_scores: list[float],
    ) -> list[Any]:
        """Read back the offspring texture and return new genome objects.

        Only the winner's parameters are read from the CPU side; the rest
        of the population is reconstructed on demand when needed.
        """
        # For now, return a minimal list containing just the winner.
        # The full population reconstruction can be done by reading the
        # offspring texture row-by-row, but that is expensive; callers
        # that need the full population should do a bulk readback.
        return [winner_params]

    def read_offspring_texture(self) -> bytes | None:
        """Read the entire offspring texture for bulk population reconstruction."""
        if not self._available or self._population._offspring_tex is None:
            return None
        try:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._population._offspring_tex)
            data = GL.glGetTexImage(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RED,
                GL.GL_FLOAT,
            )
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            if isinstance(data, bytes):
                return data
            return bytes(bytearray(data))
        except RuntimeError as e:
            logger.debug("read_offspring_texture failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Utility: helper to produce a GpuEvolutionEngine, patched onto a renderer
# ---------------------------------------------------------------------------


def patch_evolution(
    renderer: Any,
    pop_size: int = DEFAULT_POP_SIZE,
    genome_width: int = DEFAULT_GENOME_WIDTH,
    tournament_size: int = 4,
    elite_fraction: float = 0.15,
    crossover_rate: float = 0.9,
    sbx_eta: float = 20.0,
    mutation_rate: float = 0.1,
    poly_eta: float = 20.0,
) -> GpuEvolutionEngine:
    """Attach a :class:`GpuEvolutionEngine` to an ``EvoRenderer`` instance.

    Creates the engine, wires up the FBO pool from the renderer (if available),
    initialises GL resources, and stores a reference for the renderer's
    evolution loop.

    Returns the created engine so the caller can keep a reference.
    """
    pool = getattr(renderer, "_fbo_pool", None) or getattr(renderer, "pool", None)
    engine = GpuEvolutionEngine(
        pop_size=pop_size,
        genome_width=genome_width,
        tournament_size=tournament_size,
        elite_fraction=elite_fraction,
        crossover_rate=crossover_rate,
        sbx_eta=sbx_eta,
        mutation_rate=mutation_rate,
        poly_eta=poly_eta,
        pool=pool,
    )
    renderer.gpu_evolution = engine
    try:
        ok = engine.init_gl()
        if not ok:
            logger.info("GpuEvolutionEngine deferred init")
    except RuntimeError as e:
        logger.debug("patch_evolution: GPU init deferred (%s)", e)
    return engine


# ---------------------------------------------------------------------------
# Detect compute-shader availability (mirrors the check in evo_gpu_compute)
# ---------------------------------------------------------------------------


def _have_compute_shaders() -> bool:
    try:
        if getattr(GL, "glDispatchCompute", None) is None:
            return False
        if getattr(GL, "glBindBufferBase", None) is None:
            return False
        version = GL.glGetString(GL.GL_VERSION)
        if version is None:
            return False
        digits = "".join(chr(c) for c in version).split()[0].split(".")
        if len(digits) < 2:
            return False
        major, minor = int(digits[0]), int(digits[1])
        return (major, minor) >= (4, 3)
    except (ValueError, IndexError, TypeError):
        return False
