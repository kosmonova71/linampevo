#!/usr/bin/env python3
"""
ASCEND V9 - Compiler (Clean Pipeline)

Single deterministic compilation pipeline:
expr → tokenize → parse → transform → lower to SSA → optimize → cache
"""

from __future__ import annotations

from typing import Any

from engine.gl.glsl_generator import GLSLGenerator, ir_to_glsl
from engine.gl.shader_cache import get_shader_cache
from engine.ir.lowering import lower_to_ssa
from engine.ir.ssa import IRBlock, IROptimizer
from engine.parser.parser import ASTTransformer, V9Parser
from engine.parser.tokenizer import V9Tokenizer


class V9Compiler:
    """V9 Compiler: Single IR pipeline, deterministic, no exec().

    Pipeline:
        expr
        → tokenize()
        → parse() → AST
        → transform() → normalized AST
        → lower_to_ssa() → IR
        → optimize(IR)
        → cache(hash(expr))
    """

    def __init__(self, cache_size: int = 1000):
        """Initialize V9 compiler.

        Args:
            cache_size: Maximum cache size for IR and shaders
        """
        self.tokenizer = V9Tokenizer()
        self.parser = V9Parser()
        self.transformer = ASTTransformer()
        self.optimizer = IROptimizer()
        self.glsl_generator = GLSLGenerator()
        self.shader_cache = get_shader_cache(cache_size)

        # IR cache (expression → IR block)
        self.ir_cache: dict[str, IRBlock] = {}

    def compile(self, expr: str) -> IRBlock | None:
        """Compile expression to SSA IR.

        Args:
            expr: Input expression string

        Returns:
            Optimized SSA IR block, or None if compilation fails
        """
        if not expr or not expr.strip():
            return None

        # Check IR cache
        if expr in self.ir_cache:
            return self.ir_cache[expr]

        try:
            # Step 1: Tokenize
            tokens = self.tokenizer.tokenize_with_implicit_mult(expr)
            code = self.tokenizer.tokens_to_string(tokens)

            # Step 2: Parse
            tree = self.parser.parse(code)

            # Step 3: Transform
            tree = self.transformer.visit(tree)

            # Step 4: Lower to SSA
            ir = lower_to_ssa(tree)

            # Step 5: Optimize
            ir = self.optimizer.optimize(ir)

            # Cache result
            self.ir_cache[expr] = ir

            return ir

        except Exception:
            # Compilation failed
            return None

    def compile_to_glsl(self, expr: str, version: str = "330 core") -> str | None:
        """Compile expression directly to GLSL shader.

        Args:
            expr: Input expression string
            version: GLSL version string

        Returns:
            GLSL shader source code, or None if compilation fails
        """
        # Compile to IR
        ir = self.compile(expr)
        if ir is None:
            return None

        # Check shader cache
        cached = self.shader_cache.get(ir)
        if cached is not None:
            return cached

        # Generate GLSL
        glsl = ir_to_glsl(ir, version)

        # Cache shader
        self.shader_cache.put(ir, glsl)

        return glsl

    def invalidate(self, expr: str) -> None:
        """Invalidate cache entry for expression.

        Args:
            expr: Expression string
        """
        if expr in self.ir_cache:
            ir = self.ir_cache[expr]
            self.shader_cache.invalidate(ir)
            del self.ir_cache[expr]

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.ir_cache.clear()
        self.shader_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get compilation statistics.

        Returns:
            Dictionary with compilation stats
        """
        return {
            "ir_cache_size": len(self.ir_cache),
            "shader_cache_stats": self.shader_cache.get_stats(),
        }
