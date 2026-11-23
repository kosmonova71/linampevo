"""engine/systems/gpu_particle_system.py

GPU-accelerated particle organism ecosystem.

This module implements a GPU-backed particle ecosystem in the same spirit as
:mod:`evo.particle_genome` (the CPU ``ParticleSpeciesGenome``) but scaled to
hundreds of thousands of individual particles living in SSBOs.  Per-particle
physics, reproduction and fitness are advanced by OpenGL 4.3 compute shaders so
the CPU only uploads the seed population and reads back statistics.

Layout
------
Each particle is split across four SSBOs:

* ``ssbo_position`` : ``vec2`` world-space position in [0,1].
* ``ssbo_velocity`` : ``vec2`` velocity.
* ``ssbo_dna``      : ``vec4`` gene values (speed, size, color_hue, behaviour).
* ``ssbo_meta``     : a struct { energy, age, species_id, fitness,
  mutation_rate, alive, parent_id, generation, pad }.

Because the GPU needs a flat, std430-friendly layout, the CPU-side
:class:`GPUParticleOrganism` dataclass carries the same data in a convenient
form and the buffer round-trips it through numpy ``float32`` arrays.

Lazy initialisation
-------------------
Mirroring :mod:`evo_gpu_compute`, every GL object is created only inside a live
OpenGL context (the ``init_gl`` / ``init_buffers`` methods).  The module is
importable without a context and the :class:`GPUParticleEcosystem` degrades
gracefully: if compute shaders are unavailable it falls back to a CPU update of
its local organism list so the public API (seed, simulate, stats) still works.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from OpenGL import GL

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy expected to be present
    np = None

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_PARTICLE_COUNT = 131072  # 128k
SEED_COUNT = 4096
REPRODUCTION_THRESHOLD = 0.5
DEATH_ENERGY = 0.0
ENERGY_DECAY = 0.001
META_FLOATS = 9  # energy, age, species_id, fitness, mutation_rate,
# alive, parent_id, generation, pad

# SSBO binding points.
BIND_POSITION = 0
BIND_VELOCITY = 1
BIND_DNA = 2
BIND_META = 3

# Local work-group size used by every compute shader.
WORK_GROUP_SIZE = 256

# Default species palette (hue-derived colours assigned by species id).
DEFAULT_SPECIES_COLORS: dict[int, tuple[float, float, float]] = {
    0: (0.20, 0.60, 1.00),
    1: (1.00, 0.30, 0.30),
    2: (0.30, 1.00, 0.40),
    3: (1.00, 0.85, 0.20),
    4: (0.80, 0.30, 1.00),
    5: (0.20, 1.00, 0.90),
    6: (1.00, 0.50, 0.80),
    7: (0.60, 0.60, 0.70),
}


# ============================================================================
# GL helpers (mirrors evo_gpu_compute.py)
# ============================================================================


def _have_compute_shaders() -> bool:
    """Best-effort detection of OpenGL 4.3+ compute shader support."""
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
    except Exception:  # pragma: no cover - called outside a context
        return False


def _gl_error() -> str | None:
    try:
        err = GL.glGetError()
    except Exception:  # pragma: no cover
        return "glGetError-failed"
    if err == GL.GL_NO_ERROR:
        return None
    return f"GL_ERROR(0x{err:04X})"


def _gl_check(label: str) -> None:
    err = _gl_error()
    if err is not None:
        raise RuntimeError(f"{label}: {err}")


# ============================================================================
# CPU-side particle organism metadata
# ============================================================================


@dataclass
class GPUParticleOrganism:
    """CPU-side mirror of one GPU particle organism.

    The GPU keeps the data split across SSBOs; this dataclass is the convenient
    Python representation used for uploads and read-backs.
    """

    dna: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.5)
    energy: float = 1.0
    age: float = 0.0
    species_id: int = 0
    fitness: float = 0.0
    mutation_rate: float = 0.01
    x: float = 0.5
    y: float = 0.5
    vx: float = 0.0
    vy: float = 0.0
    alive: bool = True
    parent_id: int = -1
    generation: int = 0

    @property
    def speed(self) -> float:
        return self.dna[0]

    @property
    def size(self) -> float:
        return self.dna[1]

    @property
    def color_hue(self) -> float:
        return self.dna[2]

    @property
    def behaviour_weight(self) -> float:
        return self.dna[3]


# ============================================================================
# Embedded GLSL compute shaders (GLSL 430 core)
# ============================================================================

# Shared struct + buffer declarations reused by the three compute shaders.
_COMMON = f"""
#version 430 core
#extension GL_ARB_compute_shader : enable

layout(local_size_x = {WORK_GROUP_SIZE}) in;

struct ParticleMeta {{
    float energy;
    float age;
    int   species_id;
    float fitness;
    float mutation_rate;
    int   alive;
    int   parent_id;
    int   generation;
    float pad;
}};

layout(std430, binding = {BIND_POSITION}) buffer PosBuf {{
    vec2 positions[];
}};
layout(std430, binding = {BIND_VELOCITY}) buffer VelBuf {{
    vec2 velocities[];
}};
layout(std430, binding = {BIND_DNA}) buffer DnaBuf {{
    vec4 dnas[];
}};
layout(std430, binding = {BIND_META}) buffer MetaBuf {{
    ParticleMeta metas[];
}};

uniform float uDt;
uniform int   uCount;
uniform float uTime;

// Hash-based pseudo-random generator (per-invocation, stateless).
uint hash(uint x) {{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}}
float rnd(inout uint seed) {{
    seed = hash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}}
"""

# Pass 1: physics, energy decay, aging, death.
PARTICLE_UPDATE_CS = (
    _COMMON
    + """
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uCount)) return;

    ParticleMeta m = metas[i];
    if (m.alive == 0) return;

    vec4 dna = dnas[i];
    uint seed = i * 747796405u + uint(uTime * 60.0) * 2891336453u + 1u;

    // Random walk scaled by the speed gene.
    float rx = rnd(seed) * 2.0 - 1.0;
    float ry = rnd(seed) * 2.0 - 1.0;
    vec2 walk = vec2(rx, ry);

    velocities[i] += walk * dna.x * uDt;
    velocities[i] = clamp(velocities[i], vec2(-2.0), vec2(2.0));
    positions[i]  += velocities[i] * uDt;

    // Wrap around the [0,1] world (toroidal topology).
    positions[i] = fract(positions[i]);

    m.energy -= ENERGY_DECAY_PLACEHOLDER * uDt;
    m.age    += uDt;

    // Death by starvation or old age (size gene ~ lifespan scale).
    if (m.energy <= 0.0 || m.age > dna.y * 100.0) {
        m.alive = 0;
    }

    metas[i] = m;
}
"""
)

# Pass 2: sexual / asexual reproduction into free slots.
PARTICLE_REPRODUCE_CS = (
    _COMMON
    + """
uniform int uMaxCount;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uCount)) return;

    ParticleMeta m = metas[i];
    if (m.alive == 0) return;
    if (m.energy <= REPRO_THRESHOLD_PLACEHOLDER) return;

    uint seed = i * 2654435761u + uint(uTime * 97.0) + 7u;
    uint partner = uint(rnd(seed) * float(uCount));
    ParticleMeta pm = metas[partner];
    if (partner == i || pm.alive == 0) return;

    // Find a free slot near the end of the buffer.
    int slot = -1;
    for (int s = uCount - 1; s >= 0; s--) {
        if (metas[s].alive == 0) { slot = s; break; }
    }
    if (slot < 0) return;  // population full

    vec4 childDna = mix(dnas[i], dnas[partner], 0.5);
    float mr = max(m.mutation_rate, pm.mutation_rate);
    // Mutate each gene within the mutation rate.
    for (int g = 0; g < 4; g++) {
        childDna[g] += (rnd(seed) * 2.0 - 1.0) * mr;
        childDna[g] = clamp(childDna[g], 0.0, 1.0);
    }

    positions[slot]  = positions[i];
    velocities[slot] = vec2(rnd(seed) * 2.0 - 1.0, rnd(seed) * 2.0 - 1.0) * 0.1;
    dnas[slot] = childDna;
    ParticleMeta cm;
    cm.energy         = m.energy * 0.5;
    cm.age            = 0.0;
    cm.species_id     = m.species_id;
    cm.fitness        = 0.0;
    cm.mutation_rate  = mr * 0.99 + 0.001;
    cm.alive          = 1;
    cm.parent_id      = int(i);
    cm.generation     = m.generation + 1;
    cm.pad            = 0.0;
    metas[slot] = cm;

    // Parent spends energy on reproduction.
    m.energy *= 0.5;
    metas[i] = m;
}
"""
)

# Pass 3: fitness from survival time, energy, species diversity.
PARTICLE_FITNESS_CS = (
    _COMMON
    + """
uniform int   uSpeciesTotal;   // number of distinct species currently alive

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uCount)) return;

    ParticleMeta m = metas[i];
    if (m.alive == 0) { m.fitness = 0.0; metas[i] = m; return; }

    vec4 dna = dnas[i];

    // Survival component: how long relative to the gene's nominal lifespan.
    float lifespan = max(dna.y * 100.0, 1.0);
    float survival = clamp(m.age / lifespan, 0.0, 1.0);

    // Energy component.
    float energy = clamp(m.energy, 0.0, 1.0);

    // Diversity component: rarer species are rewarded (more species -> smaller
    // per-species share, so a small uSpeciesTotal boosts the score).
    float diversity = (uSpeciesTotal > 1)
                      ? (1.0 - float(m.species_id) / float(uSpeciesTotal)) * 0.5 + 0.25
                      : 0.5;

    m.fitness = clamp(survival * 0.45 + energy * 0.45 + diversity * 0.10, 0.0, 1.0);
    metas[i] = m;
}
"""
)

# Fill in the compile-time placeholders with Python constants.
PARTICLE_UPDATE_CS = PARTICLE_UPDATE_CS.replace(
    "ENERGY_DECAY_PLACEHOLDER", repr(float(ENERGY_DECAY))
)
PARTICLE_REPRODUCE_CS = PARTICLE_REPRODUCE_CS.replace(
    "REPRO_THRESHOLD_PLACEHOLDER", repr(float(REPRODUCTION_THRESHOLD))
)


# ============================================================================
# Vertex shader for point-sprite rendering
# ============================================================================


def create_particle_vertex_shader() -> str:
    """Return GLSL for rendering particles as attenuated point sprites.

    The position SSBO is bound as a vertex attribute via the VAO.  Per-particle
    colour comes from the DNA hue (converted to RGB in the vertex shader) and
    the meta buffer supplies alive/size.  Size attenuates with distance from
    the camera (here a fixed scale on the unit world).
    """
    return """
#version 430 core

layout(location = 0) in vec2  aPosition;
layout(location = 1) in vec4  aDna;       // speed, size, color_hue, behaviour
layout(location = 2) in float aAlive;     // 1.0 alive, 0.0 dead
layout(location = 3) in float aSize;      // alive flag not needed; size scale

out vec3 vColor;
out float vAlive;

// HSV -> RGB conversion.
vec3 hsv2rgb(vec3 c) {{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}}

uniform float uPointScale = 200.0;

void main() {{
    vAlive = aAlive;
    // colour from hue gene, full saturation, value from speed gene.
    vColor = hsv2rgb(vec3(aDna.z, 0.85, 0.5 + 0.5 * aDna.x));
    // Map [0,1] world to clip space.
    gl_Position = vec4(aPosition * 2.0 - 1.0, 0.0, 1.0);
    // Size attenuation: genes in [0,1], scaled to pixels.
    float size = mix(1.0, 8.0, aDna.y) * (aAlive > 0.5 ? 1.0 : 0.0);
    gl_PointSize = max(size, 0.0) * (uPointScale / 200.0);
}}
"""


def create_particle_fragment_shader() -> str:
    """Return a minimal fragment shader that softens point sprites."""
    return """
#version 430 core
in vec3 vColor;
in float vAlive;
out vec4 fragColor;
void main() {
    if (vAlive < 0.5) discard;
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = dot(d, d);
    if (r > 0.25) discard;
    float a = smoothstep(0.25, 0.0, r);
    fragColor = vec4(vColor, a);
}
"""


# ============================================================================
# GPUParticleBuffer
# ============================================================================


class GPUParticleBuffer:
    """Manages the GPU SSBOs and VAO for a population of particles.

    All GL objects are created lazily by :meth:`init_buffers`, which must be
    called inside a live OpenGL context.  Outside a context every handle is
    ``None`` and the buffer acts as a CPU-only store so the rest of the
    ecosystem still works.
    """

    def __init__(self, particle_count: int = DEFAULT_PARTICLE_COUNT):
        self.particle_count = int(particle_count)
        self.ssbo_position: int | None = None
        self.ssbo_velocity: int | None = None
        self.ssbo_dna: int | None = None
        self.ssbo_meta: int | None = None
        self.vao: int | None = None
        self._pbo: int | None = None
        self._available = False
        self._organisms: list[GPUParticleOrganism] = []

    # -- GL object creation -------------------------------------------------

    def init_buffers(self) -> bool:
        """Create the empty SSBOs and VAO. Returns True on success."""
        if self.ssbo_position is not None:
            return self._available
        try:
            n = self.particle_count

            self.ssbo_position = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo_position)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n * 2 * 4, None, GL.GL_DYNAMIC_DRAW)

            self.ssbo_velocity = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo_velocity)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n * 2 * 4, None, GL.GL_DYNAMIC_DRAW)

            self.ssbo_dna = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo_dna)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n * 4 * 4, None, GL.GL_DYNAMIC_DRAW)

            self.ssbo_meta = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo_meta)
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER, n * META_FLOATS * 4, None, GL.GL_DYNAMIC_DRAW
            )

            self._pbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbo)
            GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, n * META_FLOATS * 4, None, GL.GL_STREAM_READ)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

            self._build_vao()
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            self._available = True
            _gl_check("GPUParticleBuffer.init_buffers")
            logger.info("GPU particle buffers initialised for %d particles", n)
        except Exception as e:  # pragma: no cover - driver / context error
            logger.warning("GPUParticleBuffer init failed (CPU fallback): %s", e)
            self._cleanup_gl_objects()
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        return self._available

    def _build_vao(self) -> None:
        """Build a VAO that sources position from ssbo_position for drawing."""
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        # Treat the position SSBO as a vertex buffer for attribute 0.
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.ssbo_position)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    # -- Packing helpers ----------------------------------------------------

    def _pack(self, organisms: list[GPUParticleOrganism]):
        """Return (pos, vel, dna, meta) numpy float32 arrays for upload."""
        n = self.particle_count
        m = len(organisms)
        pos = np.zeros((n, 2), dtype=np.float32)
        vel = np.zeros((n, 2), dtype=np.float32)
        dna = np.zeros((n, 4), dtype=np.float32)
        meta = np.zeros((n, META_FLOATS), dtype=np.float32)
        for i in range(min(m, n)):
            o = organisms[i]
            pos[i, 0] = o.x
            pos[i, 1] = o.y
            vel[i, 0] = o.vx
            vel[i, 1] = o.vy
            dna[i, 0] = o.dna[0]
            dna[i, 1] = o.dna[1]
            dna[i, 2] = o.dna[2]
            dna[i, 3] = o.dna[3]
            meta[i, 0] = o.energy
            meta[i, 1] = o.age
            meta[i, 2] = float(o.species_id)
            meta[i, 3] = o.fitness
            meta[i, 4] = o.mutation_rate
            meta[i, 5] = 1.0 if o.alive else 0.0
            meta[i, 6] = float(o.parent_id)
            meta[i, 7] = float(o.generation)
            meta[i, 8] = 0.0
        return pos, vel, dna, meta

    @staticmethod
    def _unpack(pos, vel, dna, meta) -> list[GPUParticleOrganism]:
        m = len(pos)
        out: list[GPUParticleOrganism] = []
        for i in range(m):
            alive = bool(round(meta[i, 5]) != 0)
            out.append(
                GPUParticleOrganism(
                    dna=(float(dna[i, 0]), float(dna[i, 1]), float(dna[i, 2]), float(dna[i, 3])),
                    energy=float(meta[i, 0]),
                    age=float(meta[i, 1]),
                    species_id=round(meta[i, 2]),
                    fitness=float(meta[i, 3]),
                    mutation_rate=float(meta[i, 4]),
                    x=float(pos[i, 0]),
                    y=float(pos[i, 1]),
                    vx=float(vel[i, 0]),
                    vy=float(vel[i, 1]),
                    alive=alive,
                    parent_id=round(meta[i, 6]),
                    generation=round(meta[i, 7]),
                )
            )
        return out

    # -- Upload / download --------------------------------------------------

    def upload_particles(self, organisms: list[GPUParticleOrganism]) -> None:
        """Upload a list of organisms into the GPU SSBOs (or cache on CPU)."""
        self._organisms = list(organisms)
        if not self._available or np is None:
            return
        try:
            pos, vel, dna, meta = self._pack(organisms)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo_position)
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, pos.nbytes, pos.tobytes())
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo_velocity)
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, vel.nbytes, vel.tobytes())
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo_dna)
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, dna.nbytes, dna.tobytes())
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo_meta)
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, meta.nbytes, meta.tobytes())
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            _gl_check("GPUParticleBuffer.upload")
        except Exception as e:  # pragma: no cover
            logger.debug("upload_particles failed: %s", e)

    def download_particles(self, block: bool = True) -> list[GPUParticleOrganism]:
        """Read particles back from the GPU (PBO when available, else direct).

        When GPU is unavailable the last uploaded CPU list is returned so the
        round-trip test still works without a context.
        """
        if not self._available or np is None:
            return list(self._organisms)

        n = self.particle_count
        try:
            pos = np.zeros((n, 2), dtype=np.float32)
            vel = np.zeros((n, 2), dtype=np.float32)
            dna = np.zeros((n, 4), dtype=np.float32)
            meta = np.zeros((n, META_FLOATS), dtype=np.float32)

            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbo)
            # We cannot copy interleaved meta directly into a PBO of the same
            # layout; read each SSBO with a transient PBO for correctness.
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

            for buf, arr in (
                (self.ssbo_position, pos),
                (self.ssbo_velocity, vel),
                (self.ssbo_dna, dna),
                (self.ssbo_meta, meta),
            ):
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buf)
                ptr = GL.glMapBuffer(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_READ_ONLY)
                if ptr is not None:
                    # Copy raw bytes straight out of the mapped pointer into
                    # the numpy array's memory.
                    raw = (GL.GLubyte * arr.nbytes).from_address(ptr)
                    arr_view = (
                        np.frombuffer(raw, dtype=np.uint8).view(arr.dtype).reshape(arr.shape)
                    )
                    np.copyto(arr, arr_view)
                    GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER)
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

            return self._unpack(pos, vel, dna, meta)
        except Exception as e:  # pragma: no cover - readback failure
            logger.debug("download_particles fell back to CPU cache: %s", e)
            return list(self._organisms)

    # -- Cleanup ------------------------------------------------------------

    def _cleanup_gl_objects(self) -> None:
        for attr in ("ssbo_position", "ssbo_velocity", "ssbo_dna", "ssbo_meta", "_pbo"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    GL.glDeleteBuffers(1, [obj])
                except Exception:  # pragma: no cover
                    pass
                setattr(self, attr, None)
        if self.vao is not None:
            try:
                GL.glDeleteVertexArrays(1, [self.vao])
            except Exception:  # pragma: no cover
                pass
            self.vao = None

    def cleanup(self) -> None:
        self._cleanup_gl_objects()
        self._available = False


# ============================================================================
# GPUParticleEcosystem
# ============================================================================


class GPUParticleEcosystem:
    """High-level orchestrator for a GPU particle organism ecosystem.

    Owns the :class:`GPUParticleBuffer`, a CPU-side organism mirror (used for
    the seed population, evolution bookkeeping and as a fallback when compute
    shaders are unavailable), and per-species colouring.
    """

    def __init__(self, particle_count: int = DEFAULT_PARTICLE_COUNT):
        self.particle_count = int(particle_count)
        self.buffer = GPUParticleBuffer(self.particle_count)
        self.species_colors: dict[int, tuple[float, float, float]] = dict(DEFAULT_SPECIES_COLORS)
        self.generation: int = 0
        self.total_births: int = 0
        self.total_deaths: int = 0

        self._organisms: list[GPUParticleOrganism] = []
        self._compute_available = False
        self._shaders: dict[str, int] = {}
        self._next_id = 0

    # -- GL initialisation --------------------------------------------------

    def init_gl(self) -> bool:
        """Initialise GPU resources. Returns True if compute path is live."""
        self._compute_available = _have_compute_shaders()
        ok = self.buffer.init_buffers()
        if ok and self._compute_available:
            self._compile_compute_programs()
        else:
            logger.info("GPU compute unavailable; using CPU evolution fallback")
        return ok

    @property
    def using_gpu(self) -> bool:
        return self._compute_available and self.buffer.available

    def _compile_compute_programs(self) -> None:
        try:
            self._shaders["update"] = self._compile_cs(PARTICLE_UPDATE_CS)
            self._shaders["reproduce"] = self._compile_cs(PARTICLE_REPRODUCE_CS)
            self._shaders["fitness"] = self._compile_cs(PARTICLE_FITNESS_CS)
            self._shaders["vert"] = self._compile_program(
                GL.GL_VERTEX_SHADER, create_particle_vertex_shader()
            )
            self._shaders["frag"] = self._compile_program(
                GL.GL_FRAGMENT_SHADER, create_particle_fragment_shader()
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Compute program compile failed: %s", e)
            self._compute_available = False
            self._shaders = {}

    @staticmethod
    def _compile_cs(src: str) -> int:
        sh = GL.glCreateShader(GL.GL_COMPUTE_SHADER)
        try:
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            if GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
                raise RuntimeError(GL.glGetShaderInfoLog(sh))
            prog = GL.glCreateProgram()
            GL.glAttachShader(prog, sh)
            GL.glLinkProgram(prog)
            if GL.glGetProgramiv(prog, GL.GL_LINK_STATUS) != GL.GL_TRUE:
                raise RuntimeError(GL.glGetProgramInfoLog(prog))
            return prog
        finally:
            GL.glDeleteShader(sh)

    @staticmethod
    def _compile_program(stype: int, src: str) -> int:
        sh = GL.glCreateShader(stype)
        try:
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            if GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
                raise RuntimeError(GL.glGetShaderInfoLog(sh))
            prog = GL.glCreateProgram()
            GL.glAttachShader(prog, sh)
            GL.glLinkProgram(prog)
            if GL.glGetProgramiv(prog, GL.GL_LINK_STATUS) != GL.GL_TRUE:
                raise RuntimeError(GL.glGetProgramInfoLog(prog))
            return prog
        finally:
            GL.glDeleteShader(sh)

    # -- Population seeding -------------------------------------------------

    def seed_population(self, count: int = SEED_COUNT) -> None:
        """Create an initial random population of ``count`` organisms."""
        count = min(int(count), self.particle_count)
        self._organisms = []
        for _ in range(count):
            sid = random.randint(0, len(self.species_colors) - 1)
            org = GPUParticleOrganism(
                dna=(
                    random.uniform(0.1, 1.0),  # speed
                    random.uniform(0.1, 1.0),  # size
                    random.uniform(0.0, 1.0),  # color_hue
                    random.uniform(0.0, 1.0),
                ),  # behaviour_weight
                energy=random.uniform(0.5, 1.0),
                age=0.0,
                species_id=sid,
                fitness=0.0,
                mutation_rate=random.uniform(0.005, 0.05),
                x=random.random(),
                y=random.random(),
                vx=random.uniform(-0.1, 0.1),
                vy=random.uniform(-0.1, 0.1),
                alive=True,
                parent_id=-1,
                generation=0,
            )
            self._organisms.append(org)
        # Fill the rest of the buffer with dead placeholders so reproduction
        # and the GPU shaders have valid free slots to write into.
        for _ in range(self.particle_count - len(self._organisms)):
            self._organisms.append(GPUParticleOrganism(alive=False, energy=0.0))
        self.total_births += count
        self.buffer.upload_particles(self._organisms)

    # -- Simulation ---------------------------------------------------------

    def simulate(self, dt: float = 1.0 / 60.0) -> None:
        """Advance the population one step: update + reproduce (+ fitness)."""
        if self.using_gpu:
            self._simulate_gpu(dt)
        else:
            self._simulate_cpu(dt)

    def _simulate_gpu(self, dt: float) -> None:
        try:
            count = self.buffer.particle_count
            wg = (count + WORK_GROUP_SIZE - 1) // WORK_GROUP_SIZE

            def set_prog(prog):
                GL.glUseProgram(prog)
                GL.glUniform1f(GL.glGetUniformLocation(prog, "uDt"), dt)
                GL.glUniform1i(GL.glGetUniformLocation(prog, "uCount"), count)
                GL.glUniform1f(GL.glGetUniformLocation(prog, "uTime"), float(self.generation))
                GL.glBindBufferBase(
                    GL.GL_SHADER_STORAGE_BUFFER, BIND_POSITION, self.buffer.ssbo_position
                )
                GL.glBindBufferBase(
                    GL.GL_SHADER_STORAGE_BUFFER, BIND_VELOCITY, self.buffer.ssbo_velocity
                )
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, BIND_DNA, self.buffer.ssbo_dna)
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, BIND_META, self.buffer.ssbo_meta)

            set_prog(self._shaders["update"])
            GL.glDispatchCompute(wg, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            set_prog(self._shaders["reproduce"])
            GL.glUniform1i(GL.glGetUniformLocation(self._shaders["reproduce"], "uMaxCount"), count)
            GL.glDispatchCompute(wg, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            set_prog(self._shaders["fitness"])
            GL.glUniform1i(
                GL.glGetUniformLocation(self._shaders["fitness"], "uSpeciesTotal"),
                max(1, len(self.species_colors)),
            )
            GL.glDispatchCompute(wg, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            GL.glUseProgram(0)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            _gl_check("GPUParticleEcosystem.simulate_gpu")

            # Refresh CPU mirror from GPU so stats/evolution stay consistent.
            self._organisms = self.buffer.download_particles(block=True)
        except Exception as e:  # pragma: no cover
            logger.debug("GPU simulate failed, CPU fallback: %s", e)
            self._simulate_cpu(dt)

    def _simulate_cpu(self, dt: float) -> None:
        """CPU reference implementation mirroring the compute shaders."""
        new_orgs: list[GPUParticleOrganism] = []
        for o in self._organisms:
            if not o.alive:
                new_orgs.append(o)
                continue
            o.vx += (random.random() * 2.0 - 1.0) * o.dna[0] * dt
            o.vy += (random.random() * 2.0 - 1.0) * o.dna[0] * dt
            o.vx = max(-2.0, min(2.0, o.vx))
            o.vy = max(-2.0, min(2.0, o.vy))
            o.x = (o.x + o.vx * dt) % 1.0
            o.y = (o.y + o.vy * dt) % 1.0
            o.energy -= ENERGY_DECAY * dt
            o.age += dt
            if o.energy <= DEATH_ENERGY or o.age > o.dna[1] * 100.0:
                o.alive = False
                self.total_deaths += 1
            new_orgs.append(o)

        # Reproduction (sexual, into free slots).  A free slot is any entry
        # that is not alive or has no energy (dead placeholders / starved).
        alive_now = [o for o in new_orgs if o.alive and o.energy > 0.0]
        slots = [i for i, o in enumerate(new_orgs) if not (o.alive and o.energy > 0.0)]
        for o in alive_now:
            if o.energy <= REPRODUCTION_THRESHOLD:
                continue
            if not slots:
                break
            partner = random.choice(alive_now)
            if partner is o:
                continue
            child_dna = tuple(
                max(
                    0.0,
                    min(
                        1.0,
                        (o.dna[g] + partner.dna[g]) * 0.5
                        + (random.random() * 2.0 - 1.0)
                        * max(o.mutation_rate, partner.mutation_rate),
                    ),
                )
                for g in range(4)
            )
            mr = max(o.mutation_rate, partner.mutation_rate) * 0.99 + 0.001
            slot = slots.pop()
            child = GPUParticleOrganism(
                dna=child_dna,
                energy=o.energy * 0.5,
                age=0.0,
                species_id=o.species_id,
                fitness=0.0,
                mutation_rate=mr,
                x=o.x,
                y=o.y,
                vx=(random.random() * 2.0 - 1.0) * 0.1,
                vy=(random.random() * 2.0 - 1.0) * 0.1,
                alive=True,
                parent_id=self._organisms.index(o),
                generation=o.generation + 1,
            )
            new_orgs[slot] = child
            self.total_births += 1
            o.energy *= 0.5

        # Fitness.
        for o in new_orgs:
            if not o.alive:
                o.fitness = 0.0
                continue
            lifespan = max(o.dna[1] * 100.0, 1.0)
            survival = max(0.0, min(o.age / lifespan, 1.0))
            energy = max(0.0, min(o.energy, 1.0))
            st = max(1, len(self.species_colors))
            diversity = (1.0 - o.species_id / float(st)) * 0.5 + 0.25
            o.fitness = max(0.0, min(survival * 0.45 + energy * 0.45 + diversity * 0.10, 1.0))

        self._organisms = new_orgs

    # -- Evolution ----------------------------------------------------------

    def step_evolution(self) -> None:
        """Select survivors and trigger mutations on the CPU mirror."""
        survivors = [o for o in self._organisms if o.alive]
        # Tournament-style mutation pressure: weaker survivors mutate harder.
        for o in survivors:
            if random.random() < o.mutation_rate:
                new_dna = list(o.dna)
                for g in range(4):
                    new_dna[g] = max(
                        0.0, min(1.0, new_dna[g] + (random.random() * 2.0 - 1.0) * o.mutation_rate)
                    )
                o.dna = tuple(new_dna)
                o.generation += 1
        self.generation += 1
        if self.using_gpu:
            self.buffer.upload_particles(self._organisms)

    # -- Rendering ----------------------------------------------------------

    def render_points(self, program: int | None = None) -> None:
        """Render live particles as GL_POINTS via the VAO.

        ``program`` is an already-linked program (defaults to the embedded
        one). No-op (safe) when the GPU is unavailable.
        """
        if not self.buffer.available or self.buffer.vao is None:
            return
        try:
            prog = program if program is not None else self._shaders.get("vert")
            if prog is None:
                return
            GL.glUseProgram(prog)
            GL.glBindVertexArray(self.buffer.vao)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.buffer.particle_count)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            _gl_check("render_points")
        except Exception as e:  # pragma: no cover
            logger.debug("render_points failed: %s", e)

    # -- Statistics ---------------------------------------------------------

    def get_species_stats(self) -> dict[int, dict[str, float]]:
        """Return per-species counts, avg fitness and avg age for live ones."""
        stats: dict[int, dict[str, float]] = {}
        for o in self._organisms:
            if not o.alive:
                continue
            s = stats.setdefault(o.species_id, {"count": 0, "avg_fitness": 0.0, "avg_age": 0.0})
            s["count"] += 1
            s["avg_fitness"] += o.fitness
            s["avg_age"] += o.age
        for s in stats.values():
            c = max(1, int(s["count"]))
            s["avg_fitness"] /= c
            s["avg_age"] /= c
        return stats

    def alive_count(self) -> int:
        return sum(1 for o in self._organisms if o.alive)

    # -- Resize -------------------------------------------------------------

    def resize(self, new_count: int) -> None:
        """Reallocate buffers for a new particle count."""
        new_count = max(1, int(new_count))
        was_available = self.buffer.available
        self._cleanup_gl_objects()
        self.particle_count = new_count
        self.buffer = GPUParticleBuffer(new_count)
        if was_available:
            self.init_gl()

    def _cleanup_gl_objects(self) -> None:
        for prog in self._shaders.values():
            try:
                GL.glDeleteProgram(prog)
            except Exception:  # pragma: no cover
                pass
        self._shaders = {}
        self.buffer.cleanup()

    def cleanup(self) -> None:
        self._cleanup_gl_objects()
