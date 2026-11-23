"""Generate a complete Milkdrop-style fragment shader from a compiled preset.

Wraps the V9-compiled per-frame / per-pixel GLSL bodies in the standard
Milkdrop uniform set so the renderer can draw them with the same feedback
ping-pong path the evolutionary genome uses.

The fragment shader follows the Milkdrop contract:

    * ``u_time``        — elapsed seconds
    * ``u_resolution``  — vec2(device_w, device_h)
    * ``bass``/``mid``/``treb``/``beat`` — audio bands
    * ``previous_frame`` — texture sampler of the previous feedback buffer
    * ``u_feedback``    — feedback alpha (0..1)

Per-frame equations run once per frame *before* the per-pixel body; the
per-pixel body samples ``previous_frame`` and writes ``fragColor``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .compiler import MilkdropCompilation

logger = logging.getLogger(__name__)


MILKDROP_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;
out vec2 v_uv;
void main(){
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


# ---------------------------------------------------------------------------
# MilkDrop helper functions
# ---------------------------------------------------------------------------
# The V9 ``GLSLGenerator`` (_generate_functions) emits these helper functions
# before ``void main()``.  They implement MilkDrop-specific operations such
# as if/above/below/rand.  We hard-code them here rather than extracting them
# from the IR output so that the renderer is self-contained and can inject
# them even when *no* equation actually calls a helper (e.g. an empty preset
# that still needs the passthrough fallback).
#
# ``md_rand`` uses ``v_uv`` (our varying name) rather than the V9 generator's
# ``v_texcoord``.
MILKDROP_HELPER_FUNCTIONS = """\
// MilkDrop helper functions
float md_if(float cond, float a, float b) {
    return cond > 0.0 ? a : b;
}
float md_above(float a, float b) {
    return a > b ? 1.0 : 0.0;
}
float md_below(float a, float b) {
    return a < b ? 1.0 : 0.0;
}
float md_equal(float a, float b) {
    return abs(a - b) < 0.001 ? 1.0 : 0.0;
}
float sqr(float x) {
    return x * x;
}
float sign(float x) {
    return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
}
float band(float a, float b) {
    return float(int(a) & int(b));
}
float bor(float a, float b) {
    return float(int(a) | int(b));
}
float bnot(float x) {
    return float(~int(x));
}
float md_rand() {
    return fract(sin(dot(v_uv, vec2(12.9898, 78.233))) * 43758.5453);
}"""


@dataclass
class MilkdropShader:
    """A complete, ready-to-compile Milkdrop shader pair."""

    vertex_source: str
    fragment_source: str
    preset_name: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.fragment_source)


class MilkdropShaderRenderer:
    """Build a complete Milkdrop fragment shader from a compiled preset.

    The renderer assembles the compiled per-frame and per-pixel GLSL bodies
    into a single fragment shader with:

    * Standard Milkdrop uniforms (``u_time``, ``u_resolution``,
      ``bass``/``mid``/``treb``/``beat``).
    * MilkDrop helper functions (``md_if``, ``md_above``, ``md_below``,
      ``md_rand``, etc.) so equations using ``if``, ``above``, ``below``,
      ``rand`` compile successfully.
    * MilkDrop built-in variables (``rad``, ``theta``, ``x``, ``y``,
      ``time``) computed from the UV coordinates and uniforms.  Each is
      declared with the ``_0`` SSA suffix that the V9 compiler emits when
      the MilkDrop variable is first read.
    * Feedback sampling from the previous frame.
    * A fallback colour-output block when no equation sets ``fragColor``.
    """

    def build(self, compilation: MilkdropCompilation, preset_name: str = "") -> MilkdropShader:
        """Assemble a complete shader from a :class:`MilkdropCompilation`.

        Args:
            compilation: The result of compiling a preset via
                :class:`MilkdropCompiler`.
            preset_name: Optional friendly name (typically the file stem).

        Returns:
            A :class:`MilkdropShader` with vertex + fragment sources.
        """
        errors = list(compilation.errors)
        body, fragcolor_defined = self._build_body(compilation)
        frag = self._wrap_fragment(body, fragcolor_defined)
        return MilkdropShader(
            vertex_source=MILKDROP_VERTEX_SHADER,
            fragment_source=frag,
            preset_name=preset_name,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_body(self, compilation: MilkdropCompilation) -> tuple[str, bool]:
        """Emit the per-frame then per-pixel GLSL body.

        Returns a ``(body, fragcolor_defined)`` tuple where *body* is the
        inner ``main()`` content and *fragcolor_defined* indicates whether
        any per-pixel equation writes to ``fragColor`` / ``frag_color``
        (which suppresses the fallback colour block).
        """
        lines: list[str] = []

        # --- Detect fragColor assignment ---------------------------------
        # If any per-pixel equation targets fragColor/frag_color the
        # fallback colour block is suppressed via the
        # MILKDROP_FRAGCOLOR_DEFINED guard.
        fragcolor_defined = any(
            ce.target.strip().lower() in ("fragcolor", "frag_color")
            for ce in compilation.per_pixel
            if not ce.error and ce.glsl
        )

        # --- Per-frame equations -----------------------------------------
        # Each equation gets its own block scope so V9 temp names (c0, t3,
        # ...) don't collide across equations.
        if compilation.per_frame:
            lines.append("    // ---- per-frame equations ----")
            for ce in compilation.per_frame:
                if ce.error or not ce.glsl:
                    lines.append(
                        f"    // skipped: {ce.target or ce.source} — {ce.error or 'no GLSL'}"
                    )
                    continue
                body = self._normalize_glsl(ce.glsl)
                if body:
                    lines.append("    {")
                    for bline in body.splitlines():
                        lines.append("    " + bline if bline.strip() else bline)
                    lines.append("    }")
                else:
                    lines.append("    // (empty body after normalisation)")

        # --- Per-pixel equations -----------------------------------------
        if compilation.per_pixel:
            lines.append("    // ---- per-pixel equations ----")
            for ce in compilation.per_pixel:
                if ce.error or not ce.glsl:
                    lines.append(
                        f"    // skipped: {ce.target or ce.source} — {ce.error or 'no GLSL'}"
                    )
                    continue
                body = self._normalize_glsl(ce.glsl)
                if body:
                    lines.append("    {")
                    for bline in body.splitlines():
                        lines.append("    " + bline if bline.strip() else bline)
                    lines.append("    }")
                else:
                    lines.append("    // (empty body after normalisation)")

        # --- Colour output -----------------------------------------------
        if not compilation.per_pixel:
            # No per-pixel equations — emit a simple passthrough so the
            # screen is never black.  ``uv`` is already declared in the
            # shader prologue, so we only use it here.
            lines.append("    // ---- default passthrough ----")
            lines.append("    uv = clamp(uv, 0.0, 1.0);")
            lines.append("    float v = 0.5 + 0.5 * sin(u_time + length(p) * 8.0);")
            lines.append("    vec3 col = vec3(v);")
            lines.append("    col.r *= 1.0 + bass * 0.5;")
            lines.append("    col.g *= 1.0 + mid * 0.5;")
            lines.append("    col.b *= 1.0 + treb * 0.5;")
            lines.append("    col *= 0.6;")
            lines.append("    col = clamp(col, 0.0, 8.0);")
            lines.append("    if (any(isnan(col)) || any(isinf(col))) col = vec3(0.0);")
            lines.append("    col = col / (vec3(1.0) + col);")
            lines.append("    fragColor = vec4(col, 1.0);")

        else:
            # The per-pixel equations assign to MilkDrop variables (zoom,
            # rad, r, g, b, etc.).  Produce a visible colour output that
            # blends the previous feedback frame with a hue derived from the
            # audio bands and time.  Presets that explicitly assign to
            # ``fragColor`` (via the MILKDROP_FRAGCOLOR_DEFINED guard)
            # override this.
            lines.append("    // ---- colour output ----")
            lines.append("    #ifndef MILKDROP_FRAGCOLOR_DEFINED")
            lines.append("    uv = clamp(uv, 0.0, 1.0);")
            lines.append("    vec4 prev = texture(previous_frame, uv);")
            lines.append("    if (any(isnan(prev)) || any(isinf(prev))) prev = vec4(0.0);")
            lines.append("    float hue = u_time * 0.1 + bass * 2.0 + mid * 1.5 + treb * 3.0;")
            lines.append("    vec3 cyc = vec3(")
            lines.append("        0.5 + 0.5 * sin(hue),")
            lines.append("        0.5 + 0.5 * sin(hue + 2.094),")
            lines.append("        0.5 + 0.5 * sin(hue + 4.189)")
            lines.append("    );")
            lines.append("    vec3 col = mix(cyc, prev.rgb, clamp(u_feedback, 0.0, 0.999));")
            lines.append("    col *= 1.0 + beat * 0.3;")
            lines.append("    col = clamp(col, vec3(-16.0), vec3(16.0));")
            lines.append("    if (any(isnan(col)) || any(isinf(col))) col = vec3(0.0);")
            lines.append("    col = col / (vec3(1.0) + max(col, vec3(0.0)));")
            lines.append("    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);")
            lines.append("    #endif")


        return "\n".join(lines), fragcolor_defined

    @staticmethod
    def _normalize_glsl(glsl: str) -> str:
        """Normalise the V9-generated GLSL to our fragment interface.

        The V9 :class:`~engine.gl.glsl_generator.GLSLGenerator` emits
        ``v_texcoord`` (varying) and ``frag_color`` (output), and appends
        a default ``frag_color = vec4(1.0)`` to every IR block.  This
        method:

        1. Strips the function definitions and ``void main()`` wrapper,
           keeping only the body that assigns to MilkDrop variables.
        2. Replaces ``v_texcoord`` → ``v_uv`` and ``frag_color`` →
           ``fragColor`` to match our fragment interface.
        3. Strips the default colour write so per-equation bodies do not
           clobber the final output.
        """
        lines = glsl.splitlines()
        body_lines: list[str] = []
        in_main = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("void main()"):
                in_main = True
                continue
            if in_main:
                if stripped == "}":
                    break
                body_lines.append(line)

        if not body_lines:
            # Fallback: use the raw GLSL verbatim.
            body_lines = lines

        normalized = "\n".join(body_lines)
        normalized = normalized.replace("v_texcoord", "v_uv")
        normalized = normalized.replace("frag_color", "fragColor")
        # Strip the V9 generator's default colour write so per-equation
        # bodies do not clobber the final output.  Use MULTILINE so the
        # regex matches regardless of whether this line is the last line
        # in the string or is followed by a trailing newline.
        normalized = re.sub(
            r"^\s*fragColor\s*=\s*vec4\(\s*1\.0\s*\)\s*;\s*\n?",
            "",
            normalized,
            flags=re.MULTILINE,
        )
        return normalized.strip()

    @staticmethod
    def _wrap_fragment(body: str, fragcolor_defined: bool) -> str:
        """Wrap the compiled body in the standard MilkDrop fragment prologue.

        The prologue declares:
        * Vertex/fragment I/O (``v_uv`` → ``fragColor``).
        * Standard uniforms (``u_time``, ``u_resolution``, audio bands).
        * MilkDrop helper functions (``md_if``, ``md_above``, etc.).
        * MilkDrop built-in variables (``rad``, ``theta``, ``x``, ``y``,
          ``time``) declared with the ``_0`` SSA suffix that the V9
          compiler emits when the MilkDrop variable is first read, so
          equations referencing them find a matching declaration.

        Args:
            body: The normalised per-frame/per-pixel body (inside main).
            fragcolor_defined: If True, emit
                ``#define MILKDROP_FRAGCOLOR_DEFINED`` to suppress the
                fallback colour block.
        """
        lines: list[str] = [
            "#version 330 core",
            "in vec2 v_uv;",
            "out vec4 fragColor;",
            "",
            "uniform float u_time;",
            "uniform vec2 u_resolution;",
            "uniform sampler2D previous_frame;",
            "uniform float u_feedback;",
            "uniform float bass;",
            "uniform float mid;",
            "uniform float treb;",
            "uniform float beat;",
            "",
        ]

        # Include helper functions needed by MilkDrop equations
        # (md_if, md_above, md_below, md_rand, etc.).
        lines.append(MILKDROP_HELPER_FUNCTIONS)
        lines.append("")

        if fragcolor_defined:
            # Suppress the fallback colour-output block so the equation's
            # own fragColor assignment takes precedence.
            lines.append("#define MILKDROP_FRAGCOLOR_DEFINED")
            lines.append("")

        lines.extend(
            [
                "void main(){",
                "    vec2 uv = v_uv;",
                "    vec2 p = uv - 0.5;",
                "",
                "    // MilkDrop built-in variables.",
                "    // Each is declared with the _0 SSA suffix that the V9",
                "    // compiler emits when the MilkDrop variable is first",
                "    // read, so equations referencing rad, time, bass, etc.",
                "    // find a matching declaration.",
                "    float rad_0 = length(p);",
                "    float theta_0 = atan(p.y, p.x);",
                "    float x_0 = uv.x;",
                "    float y_0 = uv.y;",
                "    float time_0 = u_time;",
                "    float dt_0 = 1.0 / 60.0;",
                "    float bass_0 = bass;",
                "    float mid_0 = mid;",
                "    float treb_0 = treb;",
                "    float beat_0 = beat;",
                "",
            ]
        )

        if body:
            lines.append(body)

        lines.append("}")
        return "\n".join(lines)
