#!/usr/bin/env python3

"""evo_multi_pass.py

Two components for adaptive render passes:

MultiPassBuffer
    Low-level FBO chain manager.  Renders into a chain of FBO+texture pairs
    so pass[i] reads from pass[i-1].

MultiPassPipeline
    High-level executor that takes a :class:`PipelineGenome`, compiles one
    GL program per pass, and drives the multi-pass render loop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from OpenGL import GL

from evo.gl.shader_disk_cache import ShaderDiskCache

if TYPE_CHECKING:
    from evo.modular_shader_factory import ModularShaderFactory
    from evo.pipeline_genome import PipelineGenome
    from evo_gpu_pool import FboPool

logger = logging.getLogger(__name__)

_INPUT_TEXTURE_UNIT = 0
_FEEDBACK_TEXTURE_UNIT = 1
_COMPOSITE_TEXTURE_UNIT = 0


def _array_uniform_locations(prog: int, name: str, count: int) -> list[int]:
    try:
        base = int(GL.glGetUniformLocation(prog, name))
        if base < 0:
            base = int(GL.glGetUniformLocation(prog, f"{name}[0]"))
    except Exception:
        return [-1] * count
    if base < 0:
        return [-1] * count
    return [base + i for i in range(count)]


def _set_uniform(loc: int, meta: dict[str, Any]) -> None:
    if loc < 0:
        return
    kind = meta.get("kind", "float")
    value = float(meta.get("value", 0.0))
    if kind == "int":
        if "value_g" in meta or "value_b" in meta or "value_a" in meta:
            GL.glUniform4i(
                loc,
                round(value),
                round(float(meta.get("value_g", 0.0))),
                round(float(meta.get("value_b", 0.0))),
                round(float(meta.get("value_a", 0.0))),
            )
        else:
            GL.glUniform1i(loc, round(value))
        return
    if "value_g" in meta or "value_b" in meta or "value_a" in meta:
        GL.glUniform4f(
            loc,
            value,
            float(meta.get("value_g", 0.0)),
            float(meta.get("value_b", 0.0)),
            float(meta.get("value_a", 0.0)),
        )
    else:
        GL.glUniform1f(loc, value)


class MultiPassBuffer:
    """Chain of N+1 FBOs for multi-pass rendering.

    Manages:
      - ``base`` FBO: the initial input (feedback buffer or cleared black)
      - ``pass_fbos[i]``: FBO for pass i (i=0..N-1)
      - ``pass_textures[i]``: colour attachment for pass i

    Usage::

        mpb = MultiPassBuffer(pool=fbo_pool, n_passes=4)
        mpb.resize(width, height)

        for i in range(n_passes):
            mpb.bind(i)                       # bind pass[i] FBO
            ... render ...
            mpb.finish()                      # unbind

        final_tex = mpb.output_tex()           # texture to composite to screen
        mpb.cleanup()
    """

    def __init__(self, pool: FboPool | None = None, n_passes: int = 1):
        self._pool = pool
        self._n_passes: int = max(1, int(n_passes))

        # Base FBO (scratch input - initially black)
        self.base_fbo: int | None = None
        self.base_tex: int | None = None

        # Per-pass FBOs
        self.pass_fbos: list[int | None] = [None] * self._n_passes
        self.pass_textures: list[int | None] = [None] * self._n_passes

        # Dimensions
        self._width: int = 256
        self._height: int = 256

        # Currently bound pass index
        self._bound_idx: int = -1

    @property
    def n_passes(self) -> int:
        return self._n_passes

    @n_passes.setter
    def n_passes(self, value: int) -> None:
        old = self._n_passes
        self._n_passes = max(1, int(value))
        if self._n_passes > old:
            self.pass_fbos.extend([None] * (self._n_passes - old))
            self.pass_textures.extend([None] * (self._n_passes - old))
        elif self._n_passes < old:
            for i in range(self._n_passes, old):
                self._release_pass(i)
            self.pass_fbos = self.pass_fbos[: self._n_passes]
            self.pass_textures = self.pass_textures[: self._n_passes]
        self._ensure_allocated()

    def resize(self, width: int, height: int) -> None:
        width = max(32, int(width))
        height = max(32, int(height))
        if width == self._width and height == self._height:
            return
        self._bound_idx = -1
        self._release_all()
        self._width = width
        self._height = height
        self._ensure_allocated()

    def _ensure_allocated(self) -> None:
        if self.base_tex is None:
            self.base_fbo, self.base_tex = self._alloc_pair()
            self._clear_fbo(self.base_fbo, self._width, self._height)
        self._alloc_range(0, self._n_passes)

    def _alloc_range(self, start: int, end: int) -> None:
        for i in range(start, end):
            if self.pass_textures[i] is None:
                fbo, tex = self._alloc_pair()
                self.pass_fbos[i] = fbo
                self.pass_textures[i] = tex
                self._clear_fbo(fbo, self._width, self._height)

    def _alloc_pair(self) -> tuple[int, int]:
        if self._pool is not None:
            return self._pool.acquire_fbo_with_texture(self._width, self._height)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA16F,
            self._width,
            self._height,
            0,
            GL.GL_RGBA,
            GL.GL_HALF_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
        )
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"MultiPass FBO incomplete: {hex(status)}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return int(fbo), int(tex)

    def _clear_fbo(self, fbo: int, w: int, h: int) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, w, h)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self._bound_idx = -1

    def bind(self, pass_idx: int, clear: bool = True) -> None:
        """Bind a pass target; each pass starts cleared by default."""
        self._ensure_allocated()
        if pass_idx < 0 or pass_idx >= self._n_passes:
            logger.warning("MultiPass: pass %d out of range [0, %d)", pass_idx, self._n_passes)
            return
        fbo = self.pass_fbos[pass_idx]
        if fbo is None:
            return
        self._bound_idx = pass_idx
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, self._width, self._height)
        if clear:
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

    def finish(self) -> None:
        self._bound_idx = -1
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def input_tex(self, pass_idx: int, start_from_tex: int | None = None) -> int | None:
        if pass_idx == 0:
            return start_from_tex if start_from_tex is not None else self.base_tex
        prev = pass_idx - 1
        if prev < len(self.pass_textures):
            return self.pass_textures[prev]
        return None

    def output_tex(self) -> int | None:
        if self._n_passes > 0 and self.pass_textures:
            return self.pass_textures[-1]
        return self.base_tex

    def pass_tex(self, pass_idx: int) -> int | None:
        if 0 <= pass_idx < len(self.pass_textures):
            return self.pass_textures[pass_idx]
        return None

    def _release_pass(self, idx: int) -> None:
        fbo = self.pass_fbos[idx]
        tex = self.pass_textures[idx]
        if self._pool is not None:
            self._pool.release(fbo, tex)
        else:
            if fbo is not None:
                try:
                    GL.glDeleteFramebuffers(1, [fbo])
                except Exception as e:
                    logger.debug(f"MultiPass: FBO delete error: {e}")
            if tex is not None:
                try:
                    GL.glDeleteTextures(1, [tex])
                except Exception as e:
                    logger.debug(f"MultiPass: tex delete error: {e}")
        self.pass_fbos[idx] = None
        self.pass_textures[idx] = None

    def _release_base(self) -> None:
        if self._pool is not None:
            self._pool.release(self.base_fbo, self.base_tex)
        else:
            if self.base_fbo is not None:
                try:
                    GL.glDeleteFramebuffers(1, [self.base_fbo])
                except Exception as e:
                    logger.debug(f"MultiPass: base FBO delete error: {e}")
            if self.base_tex is not None:
                try:
                    GL.glDeleteTextures(1, [self.base_tex])
                except Exception as e:
                    logger.debug(f"MultiPass: base tex delete error: {e}")
        self.base_fbo = None
        self.base_tex = None

    def _release_all(self) -> None:
        self._release_base()
        for i in range(len(self.pass_fbos)):
            self._release_pass(i)

    def cleanup(self) -> None:
        self._bound_idx = -1
        self._release_all()
        self.pass_fbos = [None] * self._n_passes
        self.pass_textures = [None] * self._n_passes

    def stats(self) -> dict[str, Any]:
        return {
            "n_passes": self._n_passes,
            "width": self._width,
            "height": self._height,
            "pool_used": self._pool is not None,
        }


# ============================================================================
# Multi-pass pipeline executor
# ============================================================================

_SHADER_TYPE_FRAG = 0x8B30
_SHADER_TYPE_VERT = 0x8B31


def _compile_shader(source: str, shader_type: int) -> int:
    sh = GL.glCreateShader(shader_type)
    GL.glShaderSource(sh, source)
    GL.glCompileShader(sh)
    if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
        err = GL.glGetShaderInfoLog(sh)
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="ignore")
        GL.glDeleteShader(sh)
        raise RuntimeError(f"MultiPass shader compile failed: {err}")
    return int(sh)


def _link_program(
    vs_source: str, fs_source: str, disk_cache: ShaderDiskCache | None = None, cache_key: str = ""
) -> int:
    prog = GL.glCreateProgram()
    if disk_cache is not None and cache_key:
        try:
            if disk_cache.try_load_binary(prog, cache_key) and GL.glGetProgramiv(
                prog, GL.GL_LINK_STATUS
            ):
                return int(prog)
        except Exception:
            logger.debug("Shader binary restore failed", exc_info=True)
        GL.glDeleteProgram(prog)
        prog = GL.glCreateProgram()

    vs: int | None = None
    fs: int | None = None
    try:
        vs = _compile_shader(vs_source, GL.GL_VERTEX_SHADER)
        fs = _compile_shader(fs_source, GL.GL_FRAGMENT_SHADER)
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
    except Exception:
        GL.glDeleteProgram(prog)
        raise
    finally:
        if vs is not None:
            GL.glDeleteShader(vs)
        if fs is not None:
            GL.glDeleteShader(fs)

    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        err = GL.glGetProgramInfoLog(prog)
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="ignore")
        GL.glDeleteProgram(prog)
        raise RuntimeError(f"MultiPass shader link failed: {err}")

    if disk_cache is not None and cache_key:
        try:
            binary = disk_cache.extract_binary(prog)
            if binary:
                disk_cache.put_binary(cache_key, binary[1], binary[0])
        except Exception:
            logger.debug("Shader binary extract failed", exc_info=True)

    return int(prog)


_QUAD_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;
out vec2 v_uv;
void main() {
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

_NOOP_FRAGMENT_SHADER = """#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_input_tex;
void main() {
    fragColor = texture(u_input_tex, v_uv);
}
"""


class MultiPassPipeline:
    """Compiles and executes a :class:`PipelineGenome` over an FBO chain.

    Usage::

        mpp = MultiPassPipeline(factory=factory, pool=fbo_pool, genome=genome)
        mpp.resize(w, h)
        mpp.render(
            vao=vao,
            quad_count=6,
            time=t,
            bass=bass, mid=mid, treb=treb,
            input_tex=input_tex,      # previous pass output / feedback
            feedback_tex=fb_tex,       # global feedback buffer
        )
        output = mpp.output_tex()
        mpp.cleanup()
    """

    def __init__(
        self,
        factory: ModularShaderFactory | None = None,
        pool: FboPool | None = None,
        genome: PipelineGenome | None = None,
        disk_cache: ShaderDiskCache | None = None,
    ):
        self._factory = factory
        self._pool = pool
        self._vao: int | None = None
        self.disk_cache = disk_cache

        # Per-pass compiled programs (one GL program per pass)
        self.programs: list[int] = []
        self.uniforms: list[dict[str, dict[str, Any]]] = []
        self._program_locs: dict[int, dict[str, int]] = {}
        self._last_output_tex: int | None = None
        self.genome: PipelineGenome | None = None

        self._pass_sources: list[str] = []
        self._pass_width: int = 256
        self._pass_height: int = 256

        if genome is not None and factory is not None:
            self.rebuild(genome)

    # ---- Rebuild -----------------------------------------------------------

    def rebuild(self, genome: PipelineGenome) -> None:
        """Recompile all passes for a new genome."""
        self._free_programs()
        self.genome = genome

        if self._factory is None:
            return

        unified: list[dict[str, Any]] | None = None
        if self.disk_cache is not None:
            try:
                driver_id = self.disk_cache.driver._id if self.disk_cache.driver else ""
                pipe_key = self.disk_cache._pipeline_key(
                    self.disk_cache._hash(genome.to_dict()), driver_id=driver_id
                )
                cached = self.disk_cache.get_pipeline(pipe_key)
                if cached:
                    cached_passes = cached.get("passes", [])
                    if cached_passes:
                        unified = [
                            {
                                "source": str(pass_cfg.get("source", "")),
                                "uniforms": dict(pass_cfg.get("uniforms") or {}),
                            }
                            for pass_cfg in cached_passes
                        ]
                        self._pass_sources = [cfg["source"] for cfg in unified]
            except Exception:
                logger.debug("Pipeline cache load failed", exc_info=True)

        if unified is None:
            try:
                sources_uniforms = self._factory.assembler.build_pipeline_per_pass(
                    pipeline=genome,
                )
            except Exception as exc:
                logger.warning("MultiPassPipeline rebuild failed: %s", exc)
                return

            self._pass_sources = [src for src, _unis in sources_uniforms]
            unified = [{"source": src, "uniforms": unis} for src, unis in sources_uniforms]

        self.programs = []
        self.uniforms = []

        for idx, cfg in enumerate(unified):
            src = cfg["source"]
            unis = cfg["uniforms"]
            cache_key = ""
            if self.disk_cache is not None:
                try:
                    driver_id = self.disk_cache.driver._id if self.disk_cache.driver else ""
                    src_hash = self.disk_cache._hash(src)
                    cache_key = self.disk_cache._source_key(
                        src_hash, GL.GL_FRAGMENT_SHADER, driver_id=driver_id
                    )
                except Exception:
                    logger.debug("Source key generation failed", exc_info=True)
                    cache_key = ""
            try:
                prog = _link_program(
                    _QUAD_VERTEX_SHADER,
                    src,
                    disk_cache=self.disk_cache,
                    cache_key=cache_key,
                )
            except RuntimeError as exc:
                logger.warning("MultiPass program compile failed (pass %d): %s", idx, exc)
                try:
                    prog = _link_program(_QUAD_VERTEX_SHADER, _NOOP_FRAGMENT_SHADER)
                    unis = {"u_input_tex": {"kind": "int", "value": 0}}
                except RuntimeError as fallback_exc:
                    logger.warning("MultiPass pass-through compile failed: %s", fallback_exc)
                    prog = 0
                    unis = {}
            self.programs.append(prog)
            self.uniforms.append(unis)

        # Persist the pipeline configuration so we can skip generation next run.
        if self.disk_cache is not None and self._pass_sources:
            try:
                driver_id = self.disk_cache.driver._id if self.disk_cache.driver else ""
                pipe_key = self.disk_cache._pipeline_key(
                    self.disk_cache._hash(genome.to_dict()), driver_id=driver_id
                )
                self.disk_cache.put_pipeline(
                    pipe_key,
                    {
                        "passes": unified,
                        "shader_keys": [],
                        "global_params": getattr(genome, "global_params", {}),
                        "driver_id": driver_id,
                    },
                )
            except Exception:
                logger.debug("Pipeline cache persist failed", exc_info=True)

    # ---- FBO management ----------------------------------------------------

    def _ensure_fbos(self, w: int, h: int) -> None:
        if not hasattr(self, "_buffer"):
            self._buffer = MultiPassBuffer(pool=self._pool, n_passes=len(self.programs))
        if self._buffer.n_passes != len(self.programs):
            self._buffer.n_passes = len(self.programs)
        self._buffer.resize(w, h)
        self._pass_width = w
        self._pass_height = h

    def resize(self, w: int, h: int) -> None:
        self._pass_width = max(32, int(w))
        self._pass_height = max(32, int(h))
        if hasattr(self, "_buffer"):
            self._buffer.resize(self._pass_width, self._pass_height)

    # ---- Rendering ---------------------------------------------------------

    def render(
        self,
        vao: int,
        quad_count: int,
        time: float,
        bass: float = 0.0,
        mid: float = 0.0,
        treb: float = 0.0,
        input_tex: int | None = None,
        feedback_tex: int | None = None,
        final_fbo: int = 0,
        anim_values: dict[str, float] | None = None,
    ) -> int | None:
        """Render the pipeline.

        Args:
            vao: Vertex array object for the full-screen quad.
            quad_count: Number of vertices in the quad (usually 6).
            time: Elapsed time in seconds.
            bass, mid, treb: Audio levels.
            input_tex: Texture ID for the base input (pass 0 reads this).
            feedback_tex: Texture ID for global feedback buffer.
            final_fbo: OpenGL framebuffer ID for the last pass.  Pass 0 to
                render directly to the default (screen) framebuffer.

        Returns:
            The last texture rendered into the pass chain. Screen and external
            final-framebuffer passes return the previous intermediate texture,
            or ``None`` when the first pass renders directly to a final target.
        """
        if not self.programs:
            return None

        self._ensure_fbos(self._pass_width, self._pass_height)
        n = len(self.programs)
        last_output_tex: int | None = None

        for i in range(n):
            is_last_pass = i == n - 1
            if not self.programs[i]:
                if not is_last_pass:
                    self._buffer.bind(i)
                    last_output_tex = self._buffer.pass_tex(i)
                    self._buffer.finish()
                continue
            renders_to_final = is_last_pass and final_fbo != 0
            renders_to_screen = is_last_pass and final_fbo == 0
            if renders_to_final or renders_to_screen:
                self._buffer.finish()
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, final_fbo)
                GL.glViewport(0, 0, self._pass_width, self._pass_height)
            else:
                self._buffer.bind(i)

            prog = self.programs[i]
            unis = self.uniforms[i]
            locs = self._get_locs(prog, unis)

            GL.glUseProgram(prog)
            GL.glBindVertexArray(vao)

            GL.glUniform1f(locs.get("u_time", -1), float(time))
            GL.glUniform2f(
                locs.get("u_resolution", -1), float(self._pass_width), float(self._pass_height)
            )
            GL.glUniform1f(locs.get("bass", -1), float(bass))
            GL.glUniform1f(locs.get("mid", -1), float(mid))
            GL.glUniform1f(locs.get("treble", -1), float(treb))

            in_tex = self._buffer.input_tex(i, start_from_tex=input_tex)
            if in_tex is not None and "u_input_tex" in locs:
                GL.glActiveTexture(GL.GL_TEXTURE0 + _INPUT_TEXTURE_UNIT)
                GL.glBindTexture(GL.GL_TEXTURE_2D, in_tex)
                GL.glUniform1i(locs["u_input_tex"], _INPUT_TEXTURE_UNIT)

            if feedback_tex is not None and "u_feedback_tex" in locs:
                GL.glActiveTexture(GL.GL_TEXTURE0 + _FEEDBACK_TEXTURE_UNIT)
                GL.glBindTexture(GL.GL_TEXTURE_2D, feedback_tex)
                GL.glUniform1i(locs["u_feedback_tex"], _FEEDBACK_TEXTURE_UNIT)

            for name, meta in unis.items():
                if name in (
                    "u_time",
                    "u_resolution",
                    "bass",
                    "mid",
                    "treble",
                    "u_input_tex",
                    "u_feedback_tex",
                ) or name.startswith("anim_"):
                    continue
                try:
                    _set_uniform(locs.get(name, -1), meta)
                except Exception:
                    logger.debug("Uniform set failed", exc_info=True)

            # Milkdrop-style per-frame animation uniforms (zoom/rot/warp).
            # Field passes declare anim_* uniforms, which the static loop
            # above skips (via the anim_* guard), so we set them live here.
            # The renderer drives update_animation() each frame and passes
            # live values; if omitted, fall back to the pipeline genome.
            _anim_values = anim_values
            if _anim_values is None and getattr(self, "genome", None) is not None:
                _get_anim = getattr(self.genome, "get_animation_uniforms", None)
                if _get_anim is not None:
                    _anim_values = _get_anim() or {}
            if _anim_values:
                for _aname, _aval in _anim_values.items():
                    _loc = locs.get(_aname, -1)
                    if _loc != -1:
                        try:
                            _set_uniform(_loc, {"value": float(_aval), "kind": "float"})
                        except Exception:
                            logger.debug("Anim uniform set failed", exc_info=True)

            GL.glDrawArrays(GL.GL_TRIANGLES, 0, quad_count)

            if not is_last_pass:
                last_output_tex = self._buffer.pass_tex(i)
                self._buffer.finish()

            GL.glBindVertexArray(0)
            GL.glUseProgram(0)

        self._last_output_tex = last_output_tex
        return last_output_tex

    def output_tex(self) -> int | None:
        return self._last_output_tex

    def cleanup(self) -> None:
        self._free_programs()
        if hasattr(self, "_buffer"):
            self._buffer.cleanup()
            delattr(self, "_buffer")

    # ---- Internals ---------------------------------------------------------

    def _get_locs(self, prog: int, uniforms: dict[str, dict[str, Any]]) -> dict[str, int]:
        cached = self._program_locs.get(prog)
        if cached is not None:
            return cached
        names = {
            "u_time",
            "u_resolution",
            "bass",
            "mid",
            "treble",
            "u_input_tex",
            "u_feedback_tex",
        }
        names.update(uniforms)
        locs = {name: int(GL.glGetUniformLocation(prog, name)) for name in names}
        locs = {name: loc for name, loc in locs.items() if loc >= 0}
        self._program_locs[prog] = locs
        return locs

    def _free_programs(self) -> None:
        for prog in self.programs:
            try:
                GL.glDeleteProgram(int(prog))
            except Exception:
                logger.debug("Program delete failed", exc_info=True)
        self.programs = []
        self.uniforms = []
        self._program_locs = {}
        self._last_output_tex = None
        self._pass_sources = []


# ============================================================================
# Competition Renderer — Population → Top N → Blend → Compete → Replace
# ============================================================================

_COMPETITION_BLEND_FS = """#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tiles[8];
uniform float u_weights[8];
uniform int u_cols;
uniform int u_rows;

void main() {
    vec3 blended = vec3(0.0);
    float total_w = 0.0;
    vec3 c0 = texture(u_tiles[0], v_uv).rgb; float w0 = u_weights[0]; blended += c0 * w0; total_w += w0;
    vec3 c1 = texture(u_tiles[1], v_uv).rgb; float w1 = u_weights[1]; blended += c1 * w1; total_w += w1;
    vec3 c2 = texture(u_tiles[2], v_uv).rgb; float w2 = u_weights[2]; blended += c2 * w2; total_w += w2;
    vec3 c3 = texture(u_tiles[3], v_uv).rgb; float w3 = u_weights[3]; blended += c3 * w3; total_w += w3;
    vec3 c4 = texture(u_tiles[4], v_uv).rgb; float w4 = u_weights[4]; blended += c4 * w4; total_w += w4;
    vec3 c5 = texture(u_tiles[5], v_uv).rgb; float w5 = u_weights[5]; blended += c5 * w5; total_w += w5;
    vec3 c6 = texture(u_tiles[6], v_uv).rgb; float w6 = u_weights[6]; blended += c6 * w6; total_w += w6;
    vec3 c7 = texture(u_tiles[7], v_uv).rgb; float w7 = u_weights[7]; blended += c7 * w7; total_w += w7;
    if (total_w > 0.0) blended /= total_w;

    vec2 grid = fract(v_uv * vec2(float(u_cols), float(u_rows)));
    float b = 0.012;
    float border = 0.0;
    if (grid.x < b || grid.x > 1.0-b || grid.y < b || grid.y > 1.0-b) {
        border = 1.0;
    }
    vec3 border_col = vec3(0.12, 0.55, 1.0);
    fragColor = vec4(mix(blended, border_col, border * 0.85), 1.0);
}
"""


class CompetitionRenderer:
    """Render the top N organisms into a tiled grid, blend their outputs,
    compete for fitness, and return the winner for replacement.

    Pipeline::

        Population
            -> select top N by fitness
            -> render each into its own tile FBO
            -> blend pass (fitness-weighted composite)
            -> compete (evaluate each tile)
            -> winner returned for replacement

    The renderer owns a grid of N per-organism FBOs, one blend FBO and the
    shader programs needed for each stage.
    """

    def __init__(
        self,
        pool: FboPool | None = None,
        n_organisms: int = 8,
        tile_pad: float = 0.008,
    ):
        self._pool = pool
        self.n = max(1, min(8, int(n_organisms)))
        self._tile_pad = max(0.0, min(0.1, float(tile_pad)))

        self._width: int = 256
        self._height: int = 256
        self._cols: int = 4
        self._rows: int = 2

        self._tile_fbos: list[int | None] = [None] * self.n
        self._tile_textures: list[int | None] = [None] * self.n
        self._blend_fbo: int | None = None
        self._blend_tex: int | None = None
        self._blend_prog: int | None = None
        self._blend_locs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        width = max(64, int(width))
        height = max(64, int(height))
        if width == self._width and height == self._height and self._blend_tex is not None:
            return
        self._width = width
        self._height = height
        self._realloc()

    def _realloc(self) -> None:
        self._release_all()
        self._cols = max(1, math.ceil(math.sqrt(self.n)))
        self._rows = max(1, math.ceil(self.n / max(1, self._cols)))

        for i in range(self.n):
            fbo, tex = self._alloc_tex(self._width, self._height)
            self._tile_fbos[i] = fbo
            self._tile_textures[i] = tex

        self._blend_fbo, self._blend_tex = self._alloc_tex(self._width, self._height)
        if self._blend_prog is None:
            self._compile_blend_program()

    def _alloc_tex(self, w: int, h: int) -> tuple[int, int]:
        if self._pool is not None:
            return self._pool.acquire_fbo_with_texture(w, h)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_HALF_FLOAT, None
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
        )
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Competition FBO incomplete: {hex(status)}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return int(fbo), int(tex)

    def _compile_blend_program(self) -> None:
        vs = _QUAD_VERTEX_SHADER
        try:
            prog = _link_program(vs, _COMPETITION_BLEND_FS)
            self._blend_prog = prog
            GL.glUseProgram(prog)
            self._blend_locs = {
                "u_weights": _array_uniform_locations(prog, "u_weights", 8),
                "u_tiles": _array_uniform_locations(prog, "u_tiles", 8),
                "u_count": GL.glGetUniformLocation(prog, "u_count"),
                "u_cols": GL.glGetUniformLocation(prog, "u_cols"),
                "u_rows": GL.glGetUniformLocation(prog, "u_rows"),
            }
            GL.glUseProgram(0)
        except Exception as exc:
            logger.warning("Competition blend program compile failed: %s", exc)
            self._blend_prog = None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_tiles(
        self,
        vao: int,
        organisms: list[Any],
        programs: list[int],
        uniforms_list: list[dict[str, dict[str, Any]]],
        time: float,
        bass: float,
        mid: float,
        treb: float,
        previous_frame_tex: int = 0,
        feedback_tex: int = 0,
    ) -> None:
        """Render each organism into its own tile on the grid FBO."""
        if not organisms:
            return

        self.resize(self._width, self._height)
        if len(self._tile_fbos) < self.n:
            for i in range(len(self._tile_fbos), self.n):
                fbo, tex = self._alloc_tex(self._width, self._height)
                self._tile_fbos.append(fbo)
                self._tile_textures.append(tex)

        tile_w = self._width / self._cols
        tile_h = self._height / self._rows

        for idx, org in enumerate(organisms):
            if idx >= self.n:
                break
            prog = programs[idx] if idx < len(programs) else 0
            if not prog:
                continue

            w = max(1, int(tile_w))
            h = max(1, int(tile_h))

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._tile_fbos[idx])
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            GL.glUseProgram(prog)
            GL.glBindVertexArray(vao)

            GL.glUniform1f(GL.glGetUniformLocation(prog, "u_time"), float(time))
            GL.glUniform2f(GL.glGetUniformLocation(prog, "u_resolution"), float(w), float(h))
            GL.glUniform1f(GL.glGetUniformLocation(prog, "bass"), float(bass))
            GL.glUniform1f(GL.glGetUniformLocation(prog, "mid"), float(mid))
            GL.glUniform1f(GL.glGetUniformLocation(prog, "treble"), float(treb))

            in_tex = previous_frame_tex
            loc_prev = GL.glGetUniformLocation(prog, "previous_frame")
            if loc_prev >= 0 and in_tex:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, in_tex)
                GL.glUniform1i(loc_prev, 0)

            unis = uniforms_list[idx] if idx < len(uniforms_list) else {}
            for name, meta in unis.items():
                if name in (
                    "u_time",
                    "u_resolution",
                    "bass",
                    "mid",
                    "treble",
                    "previous_frame",
                    "u_feedback_tex",
                ):
                    continue
                loc = GL.glGetUniformLocation(prog, name)
                if loc < 0:
                    continue
                try:
                    if meta.get("kind") == "int":
                        GL.glUniform1i(loc, round(float(meta["value"])))
                    else:
                        GL.glUniform1f(loc, float(meta["value"]))
                except Exception:
                    logger.debug("Uniform set failed", exc_info=True)

            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def render_blend(
        self,
        vao: int,
        weights: list[float],
    ) -> int:
        """Blend all tiles fitness-weighted into the blend FBO.

        Renders a single full-screen quad that averages each organism's tile
        using the supplied fitness weights.

        Returns the blend FBO texture id.
        """
        if self._blend_prog is None:
            return 0

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._blend_fbo)
        GL.glViewport(0, 0, self._width, self._height)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self._blend_prog)
        GL.glBindVertexArray(vao)

        count = max(1, min(self.n, len(weights)))
        for i in range(count):
            loc_w = self._blend_locs["u_weights"][i]
            loc_t = self._blend_locs["u_tiles"][i]
            if loc_w >= 0:
                GL.glUniform1f(loc_w, float(weights[i]))
            if loc_t >= 0 and self._tile_textures[i] is not None:
                GL.glActiveTexture(GL.GL_TEXTURE0 + _COMPOSITE_TEXTURE_UNIT + i)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._tile_textures[i]))
                GL.glUniform1i(loc_t, _COMPOSITE_TEXTURE_UNIT + i)

        if self._blend_locs.get("u_count", -1) >= 0:
            GL.glUniform1i(self._blend_locs["u_count"], count)
        if self._blend_locs.get("u_cols", -1) >= 0:
            GL.glUniform1i(self._blend_locs["u_cols"], self._cols)
        if self._blend_locs.get("u_rows", -1) >= 0:
            GL.glUniform1i(self._blend_locs["u_rows"], self._rows)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        return int(self._blend_tex or 0)

    def output_tex(self) -> int:
        return int(self._blend_tex or 0)

    def tile_tex(self, idx: int) -> int:
        if 0 <= idx < len(self._tile_textures):
            return int(self._tile_textures[idx] or 0)
        return 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _release_all(self) -> None:
        for i in range(len(self._tile_fbos)):
            self._release_pair(self._tile_fbos[i], self._tile_textures[i])
            self._tile_fbos[i] = None
            self._tile_textures[i] = None
        self._release_pair(self._blend_fbo, self._blend_tex)
        self._blend_fbo = None
        self._blend_tex = None

    def _release_pair(self, fbo: int | None, tex: int | None) -> None:
        if fbo is not None or tex is not None:
            if self._pool is not None:
                if fbo is not None or tex is not None:
                    self._pool.release(fbo, tex)
            else:
                if fbo is not None:
                    try:
                        GL.glDeleteFramebuffers(1, [fbo])
                    except Exception:
                        logger.debug("FBO delete failed", exc_info=True)
                if tex is not None:
                    try:
                        GL.glDeleteTextures(1, [tex])
                    except Exception:
                        logger.debug("Texture delete failed", exc_info=True)

    def cleanup(self) -> None:
        if self._blend_prog is not None:
            try:
                GL.glDeleteProgram(self._blend_prog)
            except Exception:
                logger.debug("Program delete failed", exc_info=True)
            self._blend_prog = None
        self._release_all()


# ---------------------------------------------------------------------------
# Region-based rendering (spec: ecosystem paints the screen together)
# ---------------------------------------------------------------------------


@dataclass
class ScreenRegion:
    """One named region of the screen rendered by a single organism.

    Attributes:
        name:        Human-readable label (left, right, spiral, particles, bg).
        x, y, w, h:  Normalised bounds in [0, 1].
        shape:       Hint for the composite shader: rect, ring, spiral, particles, full.
        organism_idx: Index into the population list.
        blend:       Soft-edge blend width as a fraction of the smaller dimension.
    """

    name: str = "region"
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    shape: str = "rect"
    organism_idx: int = 0
    blend: float = 0.0


class RegionLayout:
    """Factory for predefined screen layouts.

    Each layout returns a list of :class:`ScreenRegion` objects whose
    ``organism_idx`` values are filled from the caller's population order.
    """

    @staticmethod
    def split_left_right(n: int = 5) -> list[ScreenRegion]:
        """organism A -> left, organism B -> right, rest -> smaller panels."""
        regions: list[ScreenRegion] = []
        if n >= 1:
            regions.append(
                ScreenRegion("left", 0.0, 0.0, 0.5, 1.0, shape="rect", organism_idx=0, blend=0.005)
            )
        if n >= 2:
            regions.append(
                ScreenRegion(
                    "right", 0.5, 0.0, 0.5, 1.0, shape="rect", organism_idx=1, blend=0.005
                )
            )
        if n >= 3:
            regions.append(
                ScreenRegion(
                    "spiral", 0.25, 0.25, 0.5, 0.5, shape="spiral", organism_idx=2, blend=0.02
                )
            )
        if n >= 4:
            regions.append(
                ScreenRegion(
                    "particles", 0.0, 0.0, 1.0, 1.0, shape="particles", organism_idx=3, blend=0.0
                )
            )
        if n >= 5:
            regions.append(
                ScreenRegion(
                    "background", 0.0, 0.0, 1.0, 1.0, shape="full", organism_idx=4, blend=0.0
                )
            )
        return regions[:n]

    @staticmethod
    def split_top_bottom(n: int = 5) -> list[ScreenRegion]:
        """Organism A -> top, organism B -> bottom, rest -> smaller panels."""
        regions: list[ScreenRegion] = []
        if n >= 1:
            regions.append(
                ScreenRegion("top", 0.0, 0.0, 1.0, 0.5, shape="rect", organism_idx=0, blend=0.005)
            )
        if n >= 2:
            regions.append(
                ScreenRegion(
                    "bottom", 0.0, 0.5, 1.0, 0.5, shape="rect", organism_idx=1, blend=0.005
                )
            )
        if n >= 3:
            regions.append(
                ScreenRegion(
                    "spiral", 0.25, 0.25, 0.5, 0.5, shape="spiral", organism_idx=2, blend=0.02
                )
            )
        if n >= 4:
            regions.append(
                ScreenRegion(
                    "particles", 0.0, 0.0, 1.0, 1.0, shape="particles", organism_idx=3, blend=0.0
                )
            )
        if n >= 5:
            regions.append(
                ScreenRegion(
                    "background", 0.0, 0.0, 1.0, 1.0, shape="full", organism_idx=4, blend=0.0
                )
            )
        return regions[:n]

    @staticmethod
    def split_quadrants(n: int = 5) -> list[ScreenRegion]:
        """Four corner quadrants, then spiral / particles / background."""
        regions: list[ScreenRegion] = []
        if n >= 1:
            regions.append(
                ScreenRegion(
                    "top-left", 0.0, 0.0, 0.5, 0.5, shape="rect", organism_idx=0, blend=0.005
                )
            )
        if n >= 2:
            regions.append(
                ScreenRegion(
                    "top-right", 0.5, 0.0, 0.5, 0.5, shape="rect", organism_idx=1, blend=0.005
                )
            )
        if n >= 3:
            regions.append(
                ScreenRegion(
                    "bottom-left", 0.0, 0.5, 0.5, 0.5, shape="rect", organism_idx=2, blend=0.005
                )
            )
        if n >= 4:
            regions.append(
                ScreenRegion(
                    "bottom-right", 0.5, 0.5, 0.5, 0.5, shape="rect", organism_idx=3, blend=0.005
                )
            )
        if n >= 5:
            regions.append(
                ScreenRegion(
                    "spiral", 0.25, 0.25, 0.5, 0.5, shape="spiral", organism_idx=4, blend=0.02
                )
            )
        if n >= 6:
            regions.append(
                ScreenRegion(
                    "particles", 0.0, 0.0, 1.0, 1.0, shape="particles", organism_idx=5, blend=0.0
                )
            )
        if n >= 7:
            regions.append(
                ScreenRegion(
                    "background", 0.0, 0.0, 1.0, 1.0, shape="full", organism_idx=6, blend=0.0
                )
            )
        return regions[:n]

    @staticmethod
    def split_grid(n: int = 5) -> list[ScreenRegion]:
        """Tile the screen into a regular grid of equal rectangles."""
        regions: list[ScreenRegion] = []
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, math.ceil(n / cols))
        for idx in range(n):
            c = idx % cols
            r = idx // cols
            x = c / cols
            y = r / rows
            w = 1.0 / cols
            h = 1.0 / rows
            regions.append(
                ScreenRegion(
                    f"grid_{idx}",
                    x,
                    y,
                    w,
                    h,
                    shape="rect",
                    organism_idx=idx,
                    blend=0.004,
                )
            )
        return regions

    @staticmethod
    def named(roles: dict[str, dict[str, Any]], n: int = 5) -> list[ScreenRegion]:
        """Build layout from a role dictionary::

        roles = {
            "left":       {"x": 0, "y": 0, "w": 0.5, "h": 1, "shape": "rect"},
            "right":      {"x": 0.5, "y": 0, "w": 0.5, "h": 1, "shape": "rect"},
            "spiral":     {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "shape": "spiral"},
            "particles":  {"x": 0, "y": 0, "w": 1, "h": 1, "shape": "particles"},
            "background": {"x": 0, "y": 0, "w": 1, "h": 1, "shape": "full"},
        }
        """
        regions: list[ScreenRegion] = []
        for idx, (name, cfg) in enumerate(roles.items()):
            if idx >= n:
                break
            regions.append(
                ScreenRegion(
                    name=name,
                    x=float(cfg.get("x", 0.0)),
                    y=float(cfg.get("y", 0.0)),
                    w=float(cfg.get("w", 1.0)),
                    h=float(cfg.get("h", 1.0)),
                    shape=str(cfg.get("shape", "rect")),
                    organism_idx=idx,
                    blend=float(cfg.get("blend", 0.0)),
                )
            )
        return regions


class RegionBasedRenderer:
    """Render N organisms into named screen regions and composite them.

    Pipeline::

        population
            -> select top N by fitness
            -> for each region:
                 render organism into region FBO (viewport-clipped)
            -> composite pass: blend all region FBOs onto the default FBO

    Soft-edge blending uses a normalized weighted average with a simple
    distance-based alpha ramp along each region's border.
    """

    _MAX_REGIONS = 8
    _QUAD_VS = _QUAD_VERTEX_SHADER

    _COMPOSITE_FS = """
    #version 330 core
    in vec2 v_uv;
    out vec4 fragColor;

    uniform sampler2D u_regions[8];
    uniform int u_count;
    uniform vec4 u_bounds[8];   // x,y,w,h in pixels
    uniform float u_blend[8];
    uniform vec2 u_resolution;

    void main(){
        vec2 uv = v_uv;
        vec4 col = vec4(0.0);
        float total_w = 0.0;

        for(int i=0; i<8; i++){
            if(i >= u_count) break;
            vec4 b = u_bounds[i];
            vec2 ruv = (uv * u_resolution - b.xy) / b.zw;
            float mask = 1.0;
            float bw = u_blend[i];
            if(bw > 0.0){
                vec2 d = min(ruv, 1.0 - ruv);
                float edge = min(d.x, d.y);
                mask = smoothstep(0.0, bw, edge);
            }
            if(ruv.x >= -0.05 && ruv.x <= 1.05 &&
               ruv.y >= -0.05 && ruv.y <= 1.05){
                vec4 s = texture(u_regions[i], ruv);
                col += s * mask;
                total_w += mask;
            }
        }
        if(total_w > 0.0) col /= total_w;
        fragColor = col;
    }
    """

    def __init__(self, pool: FboPool | None = None, n: int = 5):
        self._pool = pool
        self.n = max(1, min(self._MAX_REGIONS, int(n)))
        self._width: int = 256
        self._height: int = 256

        self._regions: list[ScreenRegion] = []
        self._region_fbos: list[int | None] = [None] * self.n
        self._region_textures: list[int | None] = [None] * self.n

        self._composite_fbo: int | None = None
        self._composite_tex: int | None = None
        self._composite_prog: int | None = None
        self._composite_locs: dict[str, Any] = {}
        self._organism_locs: dict[int, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def set_regions(self, regions: list[ScreenRegion]) -> None:
        self._regions = regions[: self.n]
        for i in range(len(self._regions), self.n):
            self._regions.append(ScreenRegion(f"region_{i}", organism_idx=i))
        self._realloc()

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        width = max(64, int(width))
        height = max(64, int(height))
        if width == self._width and height == self._height and self._composite_tex is not None:
            return
        self._width = width
        self._height = height
        if self._regions:
            self._realloc()

    def _realloc(self) -> None:
        self._release_all()
        for i in range(self.n):
            region = self._regions[i]
            rw = max(1, min(self._width, int(region.w * self._width)))
            rh = max(1, min(self._height, int(region.h * self._height)))
            fbo, tex = self._alloc_tex(rw, rh)
            self._region_fbos[i] = fbo
            self._region_textures[i] = tex
        self._composite_fbo, self._composite_tex = self._alloc_tex(self._width, self._height)
        if self._composite_prog is None:
            self._compile_composite_program()

    def _alloc_tex(self, w: int, h: int) -> tuple[int, int]:
        if self._pool is not None:
            return self._pool.acquire_fbo_with_texture(w, h)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_HALF_FLOAT, None
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
        )
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Region FBO incomplete: {hex(status)}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return int(fbo), int(tex)

    def _compile_composite_program(self) -> None:
        try:
            prog = _link_program(self._QUAD_VS, self._COMPOSITE_FS)
            self._composite_prog = prog
            GL.glUseProgram(prog)
            self._composite_locs = {
                "u_regions": _array_uniform_locations(prog, "u_regions", 8),
                "u_count": GL.glGetUniformLocation(prog, "u_count"),
                "u_bounds": _array_uniform_locations(prog, "u_bounds", 8),
                "u_blend": _array_uniform_locations(prog, "u_blend", 8),
                "u_resolution": GL.glGetUniformLocation(prog, "u_resolution"),
            }
            GL.glUseProgram(0)
        except Exception as exc:
            logger.warning("Region composite program compile failed: %s", exc)
            self._composite_prog = None

    def _get_organism_locs(
        self,
        prog: int,
        uniforms: dict[str, dict[str, Any]],
    ) -> dict[str, int]:
        cached = self._organism_locs.get(prog)
        if cached is not None:
            return cached
        names = {
            "u_time",
            "u_resolution",
            "bass",
            "mid",
            "treble",
            "previous_frame",
            "u_feedback_tex",
        }
        names.update(uniforms)
        locs = dict(cached or {})
        for name in names:
            if name not in locs:
                loc = int(GL.glGetUniformLocation(prog, name))
                if loc >= 0:
                    locs[name] = loc
        self._organism_locs[prog] = locs
        return locs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_regions(
        self,
        vao: int,
        organisms: list[Any],
        programs: list[int],
        uniforms_list: list[dict[str, dict[str, Any]]],
        time: float,
        bass: float,
        mid: float,
        treb: float,
        previous_frame_tex: int = 0,
        feedback_tex: int = 0,
    ) -> None:
        """Render each organism into its assigned region FBO."""
        if not self._regions or not programs or any(tex is None for tex in self._region_textures):
            return

        for idx, region in enumerate(self._regions):
            organism_idx = int(region.organism_idx)
            if not 0 <= organism_idx < min(len(organisms), len(programs)):
                logger.warning(
                    "Region %s has invalid organism index %d", region.name, organism_idx
                )
                continue
            prog = programs[organism_idx]
            if not prog:
                continue

            unis = uniforms_list[organism_idx] if organism_idx < len(uniforms_list) else {}
            locs = self._get_organism_locs(prog, unis)

            rw = max(1, min(self._width, int(region.w * self._width)))
            rh = max(1, min(self._height, int(region.h * self._height)))

            fbo = self._region_fbos[idx]
            if fbo is None:
                continue

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glViewport(0, 0, rw, rh)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            GL.glUseProgram(prog)
            GL.glBindVertexArray(vao)

            GL.glUniform1f(locs.get("u_time", -1), float(time))
            GL.glUniform2f(locs.get("u_resolution", -1), float(rw), float(rh))
            GL.glUniform1f(locs.get("bass", -1), float(bass))
            GL.glUniform1f(locs.get("mid", -1), float(mid))
            GL.glUniform1f(locs.get("treble", -1), float(treb))

            if previous_frame_tex:
                loc = locs.get("previous_frame", -1)
                if loc >= 0:
                    GL.glActiveTexture(GL.GL_TEXTURE0 + _INPUT_TEXTURE_UNIT)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, previous_frame_tex)
                    GL.glUniform1i(loc, _INPUT_TEXTURE_UNIT)

            if feedback_tex:
                loc = locs.get("u_feedback_tex", -1)
                if loc >= 0:
                    GL.glActiveTexture(GL.GL_TEXTURE0 + _FEEDBACK_TEXTURE_UNIT)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, feedback_tex)
                    GL.glUniform1i(loc, _FEEDBACK_TEXTURE_UNIT)

            for name, meta in unis.items():
                if name in (
                    "u_time",
                    "u_resolution",
                    "bass",
                    "mid",
                    "treble",
                    "previous_frame",
                    "u_feedback_tex",
                ):
                    continue
                try:
                    _set_uniform(locs.get(name, -1), meta)
                except Exception:
                    logger.debug("Uniform set failed", exc_info=True)

            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def composite(self, vao: int, default_fbo: int = 0) -> None:
        """Blend all region FBOs onto the default framebuffer."""
        if self._composite_prog is None:
            return

        GL.glBindFramebuffer(
            GL.GL_FRAMEBUFFER,
            default_fbo if default_fbo else GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING),
        )
        GL.glViewport(0, 0, self._width, self._height)

        GL.glUseProgram(self._composite_prog)
        GL.glBindVertexArray(vao)

        locs = self._composite_locs
        count = 0
        for i, region in enumerate(self._regions):
            if i >= self.n:
                break
            tex = self._region_textures[i]
            if tex is None:
                continue

            loc_r = locs["u_regions"][i]
            if loc_r >= 0:
                GL.glActiveTexture(GL.GL_TEXTURE0 + _COMPOSITE_TEXTURE_UNIT + i)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
                GL.glUniform1i(loc_r, _COMPOSITE_TEXTURE_UNIT + i)

            loc_b = locs["u_bounds"][i]
            if loc_b >= 0:
                px = region.x * self._width
                py = region.y * self._height
                pw = max(1.0, region.w * self._width)
                ph = max(1.0, region.h * self._height)
                GL.glUniform4f(loc_b, px, py, pw, ph)

            loc_bl = locs["u_blend"][i]
            if loc_bl >= 0:
                GL.glUniform1f(loc_bl, float(region.blend))

            count += 1

        if locs.get("u_count", -1) >= 0:
            GL.glUniform1i(locs["u_count"], count)
        if locs.get("u_resolution", -1) >= 0:
            GL.glUniform2f(locs["u_resolution"], float(self._width), float(self._height))

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def region_tex(self, idx: int) -> int:
        if 0 <= idx < len(self._region_textures):
            return int(self._region_textures[idx] or 0)
        return 0

    def output_tex(self) -> int:
        return int(self._composite_tex or 0)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _release_all(self) -> None:
        for i in range(len(self._region_fbos)):
            self._release_pair(self._region_fbos[i], self._region_textures[i])
            self._region_fbos[i] = None
            self._region_textures[i] = None
        self._release_pair(self._composite_fbo, self._composite_tex)
        self._composite_fbo = None
        self._composite_tex = None

    def _release_pair(self, fbo: int | None, tex: int | None) -> None:
        if fbo is not None or tex is not None:
            if self._pool is not None:
                if fbo is not None or tex is not None:
                    self._pool.release(fbo, tex)
            else:
                if fbo is not None:
                    try:
                        GL.glDeleteFramebuffers(1, [fbo])
                    except Exception:
                        logger.debug("FBO delete failed", exc_info=True)
                if tex is not None:
                    try:
                        GL.glDeleteTextures(1, [tex])
                    except Exception:
                        logger.debug("Texture delete failed", exc_info=True)

    def cleanup(self) -> None:
        if self._composite_prog is not None:
            try:
                GL.glDeleteProgram(self._composite_prog)
            except Exception:
                logger.debug("Program delete failed", exc_info=True)
            self._composite_prog = None
            self._composite_locs = {}
            self._organism_locs = {}
        self._release_all()
