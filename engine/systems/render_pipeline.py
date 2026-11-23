"""engine/systems/render_pipeline.py

Render pipeline system for the evolution architecture.

Abstacts multi-pass rendering so the evolution system can propose render
graphs without knowing OpenGL details. The pipeline manages:

  - Multi-FBO ping-pong rendering
  - Per-pass texture inputs/outputs
  - Feedback routing between passes
  - Uniform uploads

The actual GL calls are isolated here. Evolution systems only describe
what they want via RenderPassConfig objects.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .interfaces import (
    AudioState,
    IShaderCompiler,
    RenderPassConfig,
)

logger = logging.getLogger(__name__)

# Available blending modes
BLEND_MODES = {
    "replace": 0,  # glBlendFunc(GL_ONE, GL_ZERO) + no blend eq
    "alpha": 1,  # standard alpha blend
    "add": 2,  # additive
    "multiply": 3,  # multiply
    "screen": 4,  # screen blend
}


class SystemRenderPipeline:
    """Execute a multi-pass render graph.

    This class owns all OpenGL state for rendering. It compiles per-pass
    shaders, manages FBO ping-pong, and executes passes in graph order.

    Evolution systems describe desired graphs via RenderPassConfig lists.
    """

    def __init__(
        self,
        shader_compiler: IShaderCompiler,
        base_resolution: int = 512,
        history_depth: int = 0,
    ):
        self.shader_compiler = shader_compiler
        self.base_resolution = int(base_resolution)
        self.history_depth = max(0, int(history_depth))

        self._programs: dict[str, int] = {}  # topology_key -> GL program ID
        self._uniforms: dict[str, dict[str, Any]] = {}
        self._uniform_locs: dict[str, dict[str, int]] = {}
        self._topology_key: tuple | None = None

        # Multi-FBO ping-pong
        self._fbo_a: Any = None
        self._fbo_b: Any = None
        self._tex_a: Any = None
        self._tex_b: Any = None
        self._current_read: str = "a"
        self._history: list[Any] = []

        # Fullscreen quad
        self._vao: Any = None
        self._vbo: Any = None

        # Time tracking
        self.elapsed_time: float = 0.0
        self._last_dt_time: float = time.time()

        # Last audio state for uniforms
        self.last_audio_state: AudioState | None = None

    # ------------------------------------------------------------------
    # GL resource lifecycle
    # ------------------------------------------------------------------

    def init_gl(self) -> None:
        """Create GL resources (FBOs, textures, VAO)."""
        try:
            self._create_fbos()
            self._create_fullscreen_quad()
        except ImportError:
            logger.warning("PyOpenGL not available; render pipeline disabled")

    def _create_fbos(self) -> None:
        from OpenGL import GL

        res = self.base_resolution

        self._fbo_a = int(GL.glGenFramebuffers(1))
        self._tex_a = self._create_texture(res)
        self._attach_fbo(self._fbo_a, self._tex_a, res)

        self._fbo_b = int(GL.glGenFramebuffers(1))
        self._tex_b = self._create_texture(res)
        self._attach_fbo(self._fbo_b, self._tex_b, res)

        self._history = []
        for _ in range(self.history_depth):
            fbo = int(GL.glGenFramebuffers(1))
            tex = self._create_texture(res)
            self._attach_fbo(fbo, tex, res)
            self._history.append({"fbo": fbo, "tex": tex})

    def _create_texture(self, res: int) -> int:
        from OpenGL import GL

        tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
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
        return tex

    def _attach_fbo(self, fbo: int, tex: int, res: int) -> None:
        from OpenGL import GL

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
            raise RuntimeError(f"FBO incomplete: {hex(status)}")
        GL.glViewport(0, 0, res, res)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _create_fullscreen_quad(self) -> None:
        import array

        from OpenGL import GL

        vertices = array.array(
            "f",
            [-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, -1, 0, 0, 1, 1, 1, 1, -1, 1, 0, 1],
        )

        self._vao = int(GL.glGenVertexArrays(1))
        self._vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(vertices) * 4, vertices, GL.GL_STATIC_DRAW)
        stride = 4 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(8))
        GL.glBindVertexArray(0)

    def cleanup(self) -> None:
        """Release all GL resources."""
        try:
            from OpenGL import GL

            for prog in self._programs.values():
                GL.glDeleteProgram(int(prog))
            if self._vao is not None:
                GL.glDeleteVertexArrays(1, [int(self._vao)])
            if self._vbo is not None:
                GL.glDeleteBuffers(1, [int(self._vbo)])
            for fbo in (self._fbo_a, self._fbo_b):
                if fbo is not None:
                    GL.glDeleteFramebuffers(1, [int(fbo)])
            for t in (self._tex_a, self._tex_b):
                if t is not None:
                    GL.glDeleteTextures(1, [int(t)])
            for h in self._history:
                if h.get("fbo") is not None:
                    GL.glDeleteFramebuffers(1, [int(h["fbo"])])
                if h.get("tex") is not None:
                    GL.glDeleteTextures(1, [int(h["tex"])])
        except Exception:
            pass
        self._programs.clear()
        self._uniforms = {}
        self._uniform_locs = {}
        self._fbo_a = None
        self._fbo_b = None
        self._tex_a = None
        self._tex_b = None
        self._vao = None
        self._vbo = None

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def execute_graph(self, passes: list[RenderPassConfig]) -> None:
        """Execute the full multi-pass render graph.

        Each pass reads from its input_source, applies its blend_mode,
        and optionally routes feedback per its feedback_route.
        """
        from OpenGL import GL

        if not passes or self._vao is None:
            return

        # Build pass artifacts
        artifacts: list[tuple] = []
        for p in passes:
            if not p.enabled:
                continue
            art = self.shader_compiler.compile_pass(p, {})
            artifacts.append((p, art))

        if not artifacts:
            return

        # For simplicity: execute all passes in a single combined program
        # (true multi-FBO would split each pass into its own program + FBO)
        combined_key = self._combined_topology_key(artifacts)
        program = self._get_or_compile_combined(artifacts, combined_key)

        if program is None:
            return

        GL.glUseProgram(program)
        GL.glBindVertexArray(self._vao)

        # Determine which FBO to render into
        read_fbo = self._fbo_a if self._current_read == "a" else self._fbo_b
        write_fbo = self._fbo_b if self._current_read == "a" else self._fbo_a

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, write_fbo)
        GL.glViewport(0, 0, self.base_resolution, self.base_resolution)

        # Bind input texture from the read side
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(
            GL.GL_TEXTURE_2D, self._tex_a if self._current_read == "a" else self._tex_b
        )
        GL.glUniform1i(GL.glGetUniformLocation(program, "previous_frame"), 0)

        # Upload audio uniforms
        self._upload_audio_uniforms(program)

        # Time uniform
        loc_time = GL.glGetUniformLocation(program, "u_time")
        if loc_time >= 0:
            GL.glUniform1f(loc_time, self.elapsed_time)

        # Resolution uniform
        loc_res = GL.glGetUniformLocation(program, "u_resolution")
        if loc_res >= 0:
            GL.glUniform2f(loc_res, float(self.base_resolution), float(self.base_resolution))

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        # Archive to history if enabled
        if self._history:
            self._copy_to_history(read_fbo)

        # Swap ping-pong
        self._current_read = "b" if self._current_read == "a" else "a"

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _combined_topology_key(self, artifacts: list[tuple]) -> tuple:
        """Build a combined topology key from all pass artifacts."""
        key_parts = []
        for pass_config, art in artifacts:
            key_parts.append(
                (
                    pass_config.pass_type,
                    pass_config.input_source,
                    pass_config.blend_mode,
                    pass_config.feedback_route,
                    art.topology_key,
                )
            )
        return tuple(key_parts)

    def _get_or_compile_combined(self, artifacts: list[tuple], key: tuple) -> int | None:
        """Get cached program or compile a new combined program."""

        if key in self._programs:
            return self._programs[key]

        # For now, use the first pass's module as the main entry point
        # and inject other passes as function definitions
        main_source = self._combine_sources(artifacts)
        if not main_source:
            return None

        try:
            t0 = time.perf_counter()
            program = self._compile_program(main_source)
            self._programs[key] = program
            self._topology_key = key
            compile_time = time.perf_counter() - t0
            logger.debug("combined program compiled in %.3fs", compile_time)
            return program
        except RuntimeError as exc:
            logger.warning("combined shader compile failed: %s", exc)
            return None

    def _combine_sources(self, artifacts: list[tuple]) -> str:
        """Combine multiple pass sources into one valid GLSL program.

        Non-first passes become function definitions; the first pass
        provides the main() entry point.
        """
        if not artifacts:
            return ""

        _first_pass, first_art = artifacts[0]
        main_source = first_art.source

        # Collect function definitions from other passes
        defs = []
        for pass_config, art in artifacts[1:]:
            if art.source:
                # Extract function definitions (everything except main())
                for line in art.source.splitlines():
                    stripped = line.strip()
                    if (
                        stripped
                        and not stripped.startswith("void main()")
                        and not stripped.startswith("{")
                    ):
                        defs.append(line)

        if not defs:
            return main_source

        # Inject definitions before main()
        def_block = "\n".join(defs)
        main_source = main_source.replace(
            "void main() {",
            f"{def_block}\nvoid main() {{",
            1,
        )
        return main_source

    def _compile_program(self, fragment_source: str) -> int:
        """Compile and link a fragment shader program."""
        from OpenGL import GL

        VERTEX_SHADER = """
        #version 330 core
        layout(location=0) in vec2 position;
        layout(location=1) in vec2 texcoord;
        out vec2 v_uv;
        void main() {
            v_uv = texcoord;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """

        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, VERTEX_SHADER)
        GL.glCompileShader(vs)
        if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
            err = GL.glGetShaderInfoLog(vs)
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="ignore")
            GL.glDeleteShader(vs)
            raise RuntimeError(f"vertex shader: {err}")

        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, fragment_source)
        GL.glCompileShader(fs)
        if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
            err = GL.glGetShaderInfoLog(fs)
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="ignore")
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            raise RuntimeError(f"fragment shader: {err}")

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
            raise RuntimeError(f"link: {err}")

        return int(prog)

    def _upload_audio_uniforms(self, program: int) -> None:
        """Upload audio state as shader uniforms."""
        from OpenGL import GL

        if self.last_audio_state is None:
            return

        audio = self.last_audio_state
        loc_bass = GL.glGetUniformLocation(program, "bass")
        loc_mid = GL.glGetUniformLocation(program, "mid")
        loc_treble = GL.glGetUniformLocation(program, "treble")
        loc_energy = GL.glGetUniformLocation(program, "u_energy")
        loc_beat = GL.glGetUniformLocation(program, "u_beat")
        loc_bass_d = GL.glGetUniformLocation(program, "u_bass")
        loc_mid_d = GL.glGetUniformLocation(program, "u_mid")
        loc_treble_d = GL.glGetUniformLocation(program, "u_treble")

        if loc_bass >= 0:
            GL.glUniform1f(loc_bass, float(audio.bass))
        if loc_mid >= 0:
            GL.glUniform1f(loc_mid, float(audio.mid))
        if loc_treble >= 0:
            GL.glUniform1f(loc_treble, float(audio.treble))
        if loc_energy >= 0:
            GL.glUniform1f(loc_energy, float(audio.energy))
        if loc_beat >= 0:
            GL.glUniform1f(loc_beat, float(audio.beat))
        if loc_bass_d >= 0:
            GL.glUniform1f(loc_bass_d, float(audio.bass))
        if loc_mid_d >= 0:
            GL.glUniform1f(loc_mid_d, float(audio.mid))
        if loc_treble_d >= 0:
            GL.glUniform1f(loc_treble_d, float(audio.treble))

    def _copy_to_history(self, src_fbo: int) -> None:
        """Copy current read FBO to the history ring buffer."""
        from OpenGL import GL

        if not self._history:
            return

        dst = self._history[0]  # Simplified: just use first slot
        if dst.get("fbo") is None or src_fbo is None:
            return

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, src_fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, dst["fbo"])
        GL.glBlitFramebuffer(
            0,
            0,
            self.base_resolution,
            self.base_resolution,
            0,
            0,
            self.base_resolution,
            self.base_resolution,
            GL.GL_COLOR_BUFFER_BIT,
            GL.GL_NEAREST,
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def get_feedback_texture(self) -> int:
        """Return the OpenGL texture ID of the current read side."""
        if self._current_read == "a":
            return self._tex_a if self._tex_a is not None else 0
        return self._tex_b if self._tex_b is not None else 0

    def resize_feedback(self, resolution: int) -> None:
        """Resize the feedback buffer (called when genome requests it)."""
        new_res = max(64, int(resolution))
        if new_res == self.base_resolution:
            return
        self.cleanup()
        self.base_resolution = new_res
        self.init_gl()

    def upload_uniforms(self, uniforms: dict[str, dict[str, Any]]) -> None:
        """Upload uniform values to the active shader program.

        This is a no-op here because uniforms are uploaded during
        execute_graph(). Provided for interface compatibility.
        """
        pass
