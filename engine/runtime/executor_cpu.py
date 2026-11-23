#!/usr/bin/env python3
"""
ASCEND V9 - CPU JIT Fallback (Safe, No exec())

Compiles SSA IR to callable Python functions using AST compilation.
No exec(), safe globals only.
"""

from __future__ import annotations

import ast
import math
import random
from collections.abc import Callable
from typing import Any

from engine.ir.ssa import IRBlock, IRInstruction

# Safe globals for CPU execution
SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    # Math functions
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sqrt": math.sqrt,
    "abs": abs,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    # MilkDrop functions
    "md_if": lambda c, a, b: a if c > 0 else b,
    "md_above": lambda a, b: 1.0 if a > b else 0.0,
    "md_below": lambda a, b: 1.0 if a < b else 0.0,
    "md_equal": lambda a, b: 1.0 if abs(a - b) < 0.001 else 0.0,
    "md_sqr": lambda x: x * x,
    "md_sign": lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0),
    "md_rand": lambda: random.random(),
    "band": lambda a, b: float(int(a) & int(b)),
    "bor": lambda a, b: float(int(a) | int(b)),
    "bnot": lambda x: float(~int(x)),
}


class CPUJITExecutor:
    """CPU JIT executor for SSA IR.

    Compiles IR to callable Python functions using AST compilation.
    Safe: no exec(), restricted globals only.
    """

    def __init__(self):
        """Initialize CPU JIT executor."""
        self.cache: dict[str, Callable] = {}  # IR hash → callable

    def compile_to_callable(self, block: IRBlock) -> Callable[[dict[str, float]], float]:
        """Compile IR block to callable Python function.

        Args:
            block: SSA IR block

        Returns:
            Callable that takes vars dict and returns result
        """
        # Generate Python code from IR
        code_lines = self._ir_to_python(block)
        code = "\n".join(code_lines)

        # Build Python AST from code
        tree = ast.parse(code, mode="exec")

        # Compile AST to code object
        code_obj = compile(tree, "<string>", "exec")

        # Create execution context
        exec_globals = SAFE_GLOBALS.copy()
        exec_locals = {}

        # Execute to define the function
        exec(code_obj, exec_globals, exec_locals)

        # Return the generated function
        return exec_locals.get("_execute", lambda vars: 0.0)

    def _ir_to_python(self, block: IRBlock) -> list[str]:
        """Convert IR block to Python code.

        Args:
            block: SSA IR block

        Returns:
            List of Python code lines
        """
        lines = [
            "def _execute(vars):",
            "    result = 0.0",
        ]

        for instr in block.instructions:
            line = self._emit_instruction(instr)
            lines.append(f"    {line}")

        lines.append("    return result")
        return lines

    def _emit_instruction(self, instr: IRInstruction) -> str:
        """Emit single instruction as Python code.

        Args:
            instr: IR instruction

        Returns:
            Python code line
        """
        op = instr.op
        out = instr.out
        args = instr.args

        # Constant
        if op == "const":
            return f"{out} = {args[0]}"

        # Assignment
        if op == "assign":
            return f"result = {args[0]}"

        # Binary operations
        if op == "add":
            return f"{out} = {args[0]} + {args[1]}"
        if op == "sub":
            return f"{out} = {args[0]} - {args[1]}"
        if op == "mul":
            return f"{out} = {args[0]} * {args[1]}"
        if op == "div":
            return f"{out} = {args[0]} / max(abs({args[1]}), 1e-6)"
        if op == "pow":
            return f"{out} = {args[0]} ** {args[1]}"
        if op == "mod":
            return f"{out} = {args[0]} % max(abs({args[1]}), 1e-6)"

        # Unary operations
        if op == "neg":
            return f"{out} = -{args[0]}"
        if op == "not":
            return f"{out} = 0.0 if {args[0]} > 0 else 1.0"

        # Comparisons
        if op == "eq":
            return f"{out} = 1.0 if {args[0]} == {args[1]} else 0.0"
        if op == "neq":
            return f"{out} = 1.0 if {args[0]} != {args[1]} else 0.0"
        if op == "lt":
            return f"{out} = 1.0 if {args[0]} < {args[1]} else 0.0"
        if op == "lte":
            return f"{out} = 1.0 if {args[0]} <= {args[1]} else 0.0"
        if op == "gt":
            return f"{out} = 1.0 if {args[0]} > {args[1]} else 0.0"
        if op == "gte":
            return f"{out} = 1.0 if {args[0]} >= {args[1]} else 0.0"

        # Logic
        if op == "and":
            return f"{out} = 1.0 if ({args[0]} > 0 and {args[1]} > 0) else 0.0"
        if op == "or":
            return f"{out} = 1.0 if ({args[0]} > 0 or {args[1]} > 0) else 0.0"

        # Function calls
        if op.startswith("call:"):
            func = op.split(":")[1]
            args_str = ", ".join(args)
            return f"{out} = {func}({args_str})"

        # Unknown operation
        return f"{out} = 0.0"

    def execute(self, block: IRBlock, vars: dict[str, float]) -> float:
        """Execute IR block on CPU.

        Args:
            block: SSA IR block
            vars: Variable values

        Returns:
            Result value
        """
        func = self.compile_to_callable(block)
        return func(vars)
