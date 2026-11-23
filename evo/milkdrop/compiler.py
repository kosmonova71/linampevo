"""Compile Milkdrop equations to GLSL via the existing V9 engine.

The V9 compiler pipeline (``engine.parser`` → ``engine.ir`` →
``engine.gl``) already understands the Milkdrop expression syntax
(``if``/``above``/``below``/``rand``/implicit multiplication, etc.).
This module is a thin adapter that:

1. Splits each ``per_frame_*`` / ``per_pixel_*`` equation into a
   ``target = expr`` pair.
2. Runs the V9 compiler on the expression side.
3. Emits the resulting SSA IR as a GLSL fragment shader body.

The output is *not* a complete shader - it is a fragment shader body that
the :mod:`evo.milkdrop.renderer` wraps in the standard Milkdrop uniform
set (``u_time``, ``u_resolution``, ``bass``/``mid``/``treb``/``beat``,
previous-frame texture, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from engine.compiler.compiler_v9 import V9Compiler
from engine.ir.ssa import IRBlock
from engine.parser.parser import ASTTransformer
from engine.parser.tokenizer import V9Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class CompiledEquation:
    """A single compiled Milkdrop equation."""

    target: str = ""  # LHS variable name (e.g. ``warp``, ``zoom``)
    source: str = ""  # Original equation text
    ir: IRBlock | None = None
    glsl: str = ""
    error: str | None = None


@dataclass
class MilkdropCompilation:
    """Result of compiling a whole preset."""

    per_frame: list[CompiledEquation] = field(default_factory=list)
    per_pixel: list[CompiledEquation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class MilkdropCompiler:
    """Compile Milkdrop preset equations to GLSL.

    Reuses the V9 compiler so the Milkdrop expression grammar is parsed
    deterministically.  Each equation is compiled independently so a single
    bad equation does not invalidate the whole preset - the renderer can
    still draw the remaining equations.
    """

    def __init__(self):
        self._compiler = V9Compiler()
        self._tokenizer = V9Tokenizer()
        self._transformer = ASTTransformer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compile_preset(self, preset: Any) -> MilkdropCompilation:
        """Compile every equation in ``preset``.

        Args:
            preset: A :class:`MilkdropPreset` (or any object exposing
                ``per_frame_eqs`` / ``per_pixel_eqs`` lists of ``"key=value"``
                strings).

        Returns:
            A :class:`MilkdropCompilation` with one entry per equation.
        """
        result = MilkdropCompilation()
        for eq in getattr(preset, "per_frame_eqs", []) or []:
            result.per_frame.append(self._compile_equation(eq, "per_frame"))
        for eq in getattr(preset, "per_pixel_eqs", []) or []:
            result.per_pixel.append(self._compile_equation(eq, "per_pixel"))
        return result

    def compile_equations(
        self, per_frame_eqs: list[str], per_pixel_eqs: list[str]
    ) -> MilkdropCompilation:
        """Compile raw equation lists directly."""
        result = MilkdropCompilation()
        for eq in per_frame_eqs or []:
            result.per_frame.append(self._compile_equation(eq, "per_frame"))
        for eq in per_pixel_eqs or []:
            result.per_pixel.append(self._compile_equation(eq, "per_pixel"))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compile_equation(self, equation: str, kind: str) -> CompiledEquation:
        """Compile a single ``target = expr`` equation.

        Milkdrop wraps equations in a key of the form ``per_frame_N`` or
        ``per_pixel_N``, so the raw string is::

            per_frame_1=ib_r=0.7+0.4*sin(3*time);

        We strip the outer ``per_frame_N=`` prefix and the trailing ``;``
        before handing the inner ``var = expr`` to the V9 compiler.
        """
        ce = CompiledEquation(source=equation)
        if "=" not in equation:
            ce.error = f"no '=' in equation: {equation!r}"
            return ce

        # Strip the outer ``per_frame_N=`` / ``per_pixel_N=`` wrapper.
        _outer_key, _, inner = equation.partition("=")
        inner = inner.strip()
        # Trailing semicolons are legal in Milkdrop expressions.
        if inner.endswith(";"):
            inner = inner[:-1].strip()

        if "=" not in inner:
            # The whole thing is just an expression statement - assign to
            # a throwaway so the V9 compiler still emits a value.
            inner = f"_md_result = {inner}"

        target, _, expr = inner.partition("=")
        ce.target = target.strip()
        expr = expr.strip()
        if not ce.target or not expr:
            ce.error = f"empty target/expression: {equation!r}"
            return ce

        # Compile the FULL ``target = expr`` form so the V9 lowering
        # emits an assignment to the Milkdrop variable (e.g.
        # ``zoom_0 = ...``) rather than a throwaway ``_result_0``.
        try:
            ir = self._compiler.compile(inner)
        except Exception as exc:
            ce.error = f"compile failed: {exc}"
            return ce

        if ir is None:
            ce.error = f"compiler returned None for: {expr!r}"
            return ce

        ce.ir = ir
        try:
            from engine.gl.glsl_generator import ir_to_glsl

            ce.glsl = ir_to_glsl(ir)
        except Exception as exc:
            ce.error = f"GLSL generation failed: {exc}"
            return ce

        return ce


def compile_preset_to_glsl(preset: Any) -> MilkdropCompilation:
    """Module-level convenience wrapper around :class:`MilkdropCompiler`."""
    return MilkdropCompiler().compile_preset(preset)
