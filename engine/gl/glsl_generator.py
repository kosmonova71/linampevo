#!/usr/bin/env python3
"""
ASCEND V9 - GLSL Generator (Direct Mapping from SSA IR)

Converts SSA IR to GLSL shader code.
Direct mapping, no string hacks, deterministic output.
"""

from __future__ import annotations

from typing import ClassVar

from engine.ir.ssa import IRBlock, IRInstruction


class GLSLGenerator:
    """Generate GLSL code from SSA IR.

    Direct mapping from IR instructions to GLSL statements.
    Deterministic output: same IR always produces same GLSL.
    """

    # Operation to GLSL mapping
    OP_MAP: ClassVar[dict[str, str]] = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "div_floor": "/",
        "mod": "mod",
        "pow": "pow",
        "neg": "-",
        "not": "!",
        "eq": "==",
        "neq": "!=",
        "lt": "<",
        "lte": "<=",
        "gt": ">",
        "gte": ">=",
        "and": "&&",
        "or": "||",
        "shl": "<<",
        "shr": ">>",
        "xor": "^",
    }

    # Function name mapping
    FUNC_MAP: ClassVar[dict[str, str]] = {
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "acos": "acos",
        "atan": "atan",
        "atan2": "atan",
        "sqrt": "sqrt",
        "abs": "abs",
        "exp": "exp",
        "log": "log",
        "log10": "log10",
        "floor": "floor",
        "ceil": "ceil",
        "round": "round",
        "min": "min",
        "max": "max",
        "clamp": "clamp",
        "mix": "mix",
        "step": "step",
        "smoothstep": "smoothstep",
        "md_if": "md_if",
        "md_above": "md_above",
        "md_below": "md_below",
        "md_equal": "md_equal",
        "md_sqr": "sqr",
        "md_sign": "sign",
        "md_min": "min",
        "md_max": "max",
        "md_band": "band",
        "md_bor": "bor",
        "md_bnot": "bnot",
        "md_pow": "pow",
        "if": "md_if",
        "above": "md_above",
        "below": "md_below",
        "equal": "md_equal",
        "sqr": "sqr",
        "sign": "sign",
        "rand": "md_rand",
    }

    def __init__(self):
        self.uniforms: set[str] = set()
        self.inputs: set[str] = set()

    def generate(self, block: IRBlock, version: str = "330 core") -> str:
        """Generate complete GLSL shader from IR block.

        Args:
            block: SSA IR block
            version: GLSL version string

        Returns:
            Complete GLSL shader source code
        """
        lines = [f"#version {version}\n"]

        # Add common uniforms
        lines.append("uniform float u_time;")
        lines.append("uniform vec2 u_resolution;")
        lines.append("uniform sampler2D u_texture;")
        lines.append("in vec2 v_texcoord;")
        lines.append("out vec4 frag_color;\n")

        # Add custom uniforms
        for uniform in sorted(self.uniforms):
            lines.append(f"uniform float {uniform};")

        if self.uniforms:
            lines.append("")

        # Generate function definitions
        lines.extend(self._generate_functions())

        # Generate main function
        lines.append("void main() {")
        lines.extend(self._generate_body(block))
        lines.append("}")

        return "\n".join(lines)

    def _generate_functions(self) -> list[str]:
        """Generate MilkDrop-specific GLSL functions."""
        return [
            "// MilkDrop functions",
            "float md_if(float cond, float a, float b) {",
            "    return cond > 0.0 ? a : b;",
            "}",
            "float md_above(float a, float b) {",
            "    return a > b ? 1.0 : 0.0;",
            "}",
            "float md_below(float a, float b) {",
            "    return a < b ? 1.0 : 0.0;",
            "}",
            "float md_equal(float a, float b) {",
            "    return abs(a - b) < 0.001 ? 1.0 : 0.0;",
            "}",
            "float sqr(float x) {",
            "    return x * x;",
            "}",
            "float sign(float x) {",
            "    return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);",
            "}",
            "float band(float a, float b) {",
            "    return float(int(a) & int(b));",
            "}",
            "float bor(float a, float b) {",
            "    return float(int(a) | int(b));",
            "}",
            "float bnot(float x) {",
            "    return float(~int(x));",
            "}",
            "float md_rand() {",
            "    return fract(sin(dot(v_texcoord, vec2(12.9898, 78.233))) * 43758.5453);",
            "}",
            "",
        ]

    def _generate_body(self, block: IRBlock) -> list[str]:
        """Generate main function body from IR block."""
        lines = []

        for instr in block.instructions:
            line = self._emit_instruction(instr)
            if line:
                lines.append(f"    {line}")

        # Generate final color output
        lines.append("    frag_color = vec4(1.0);")

        return lines

    def _emit_instruction(self, instr: IRInstruction) -> str:
        """Emit single instruction as GLSL statement."""
        op = instr.op
        out = instr.out
        args = instr.args

        # Constant
        if op == "const":
            return f"float {out} = {args[0]};"

        # Assignment
        if op == "assign":
            # Track output variables
            if out.startswith(("zoom", "rot", "cx", "cy")):
                self.uniforms.add(out)
            return f"float {out} = {args[0]};"

        # Binary operations
        if op in self.OP_MAP:
            glsl_op = self.OP_MAP[op]
            if op == "div":
                # Allow inf/nan like MilkDrop/GLSL
                return f"float {out} = {args[0]} / {args[1]};"
            return f"float {out} = {args[0]} {glsl_op} {args[1]};"

        # Unary operations
        if op == "neg":
            return f"float {out} = -{args[0]};"
        if op == "not":
            return f"float {out} = {args[0]} > 0.0 ? 0.0 : 1.0;"

        # Function calls
        if op.startswith("call:"):
            func = op.split(":")[1]
            glsl_func = self.FUNC_MAP.get(func, func)
            args_str = ", ".join(args)
            return f"float {out} = {glsl_func}({args_str});"

        # Unknown operation
        return f"float {out} = 0.0;"

    def generate_fragment_only(self, block: IRBlock) -> str:
        """Generate only fragment shader body (for hot-reload)."""
        lines = self._generate_body(block)
        return "\n".join(lines)


def ir_to_glsl(block: IRBlock, version: str = "330 core") -> str:
    """Convert SSA IR to GLSL shader.

    Main entry point for IR → GLSL conversion.
    """
    generator = GLSLGenerator()
    return generator.generate(block, version)
