#!/usr/bin/env python3
"""
ASCEND V9 - SSA IR (Static Single Assignment)

The heart of V9: deterministic, linear instruction sequence.
No recursion, perfect for GPU mapping, enables optimization passes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class IRInstruction:
    """Single SSA instruction.

    Structure: out = op(args...)

    Example:
        IRInstruction("add", "t1", ["x", "y"])  # t1 = x + y
        IRInstruction("sin", "t2", ["t1"])      # t2 = sin(t1)
    """

    op: str  # Operation: "add", "mul", "sin", "assign", etc.
    out: str  # Output variable name (SSA form: "t1", "t2", etc.)
    args: list[str]  # Input operands (variable names or constants)

    def __repr__(self):
        if self.op == "assign":
            return f"{self.out} = {self.args[0]}"
        if self.op == "const":
            return f"{self.out} = {self.args[0]}"
        return f"{self.out} = {self.op}({', '.join(self.args)})"

    def is_pure(self) -> bool:
        """Check if instruction has no side effects."""
        return self.op not in {"assign", "store"}

    def is_const(self) -> bool:
        """Check if instruction produces a constant."""
        return self.op == "const"

    def uses(self, var: str) -> bool:
        """Check if instruction uses variable."""
        return var in self.args


@dataclass
class IRBlock:
    """Basic block of SSA instructions.

    Linear sequence of instructions with deterministic order.
    """

    instructions: list[IRInstruction] = field(default_factory=list)

    def add(self, instr: IRInstruction) -> None:
        """Add instruction to block."""
        self.instructions.append(instr)

    def __len__(self):
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)

    def __getitem__(self, idx):
        return self.instructions[idx]

    def __repr__(self):
        lines = [f"IRBlock ({len(self)} instructions):"]
        for i, instr in enumerate(self.instructions):
            lines.append(f"  {i}: {instr}")
        return "\n".join(lines)

    def copy(self) -> IRBlock:
        """Create a deep copy of this block."""
        return IRBlock([IRInstruction(i.op, i.out, i.args.copy()) for i in self.instructions])


class IRBuilder:
    """Builder for constructing SSA IR from AST or other representations.

    Maintains variable mapping for SSA form: assigns unique names to each definition.
    """

    def __init__(self):
        self.block = IRBlock()
        self.var_counter: dict[str, int] = {}  # Original var -> SSA counter
        self.ssa_map: dict[str, str] = {}  # Original var -> current SSA name
        self.temp_counter = 0  # For temporary values

    def fresh(self, prefix: str = "t") -> str:
        """Generate fresh SSA variable name."""
        name = f"{prefix}{self.temp_counter}"
        self.temp_counter += 1
        return name

    def ssa_name(self, var: str) -> str:
        """Get or create SSA name for variable."""
        if var not in self.ssa_map:
            self.var_counter[var] = 0
            self.ssa_map[var] = f"{var}_0"
        return self.ssa_map[var]

    def define(self, var: str) -> str:
        """Create new SSA definition for variable."""
        if var not in self.var_counter:
            self.var_counter[var] = 0
        else:
            self.var_counter[var] += 1

        ssa_name = f"{var}_{self.var_counter[var]}"
        self.ssa_map[var] = ssa_name
        return ssa_name

    def emit(self, op: str, out: str | None = None, args: list[str] | None = None) -> str:
        """Emit instruction and return output name."""
        if args is None:
            args = []
        if out is None:
            out = self.fresh()

        instr = IRInstruction(op, out, args)
        self.block.add(instr)
        return out

    def emit_const(self, value: float) -> str:
        """Emit constant instruction."""
        out = self.fresh("c")
        self.emit("const", out, [str(value)])
        return out

    def emit_binop(self, op: str, left: str, right: str) -> str:
        """Emit binary operation."""
        out = self.fresh()
        self.emit(op, out, [left, right])
        return out

    def emit_unop(self, op: str, arg: str) -> str:
        """Emit unary operation."""
        out = self.fresh()
        self.emit(op, out, [arg])
        return out

    def emit_call(self, func: str, args: list[str]) -> str:
        """Emit function call."""
        out = self.fresh()
        self.emit(f"call:{func}", out, args)
        return out

    def emit_assign(self, target: str, value: str) -> None:
        """Emit assignment to output variable."""
        ssa_target = self.define(target)
        self.emit("assign", ssa_target, [value])

    def build(self) -> IRBlock:
        """Get the built IR block."""
        return self.block

    def reset(self) -> None:
        """Reset builder state."""
        self.block = IRBlock()
        self.var_counter.clear()
        self.ssa_map.clear()
        self.temp_counter = 0


class IROptimizer:
    """Optimization passes for SSA IR.

    Passes run in order:
    1. Constant folding
    2. Dead code elimination
    3. Common subexpression elimination
    4. Strength reduction
    """

    @staticmethod
    def constant_folding(block: IRBlock) -> IRBlock:
        """Fold constant expressions at compile time.

        Example: 2 + 3 → 5
        """
        new_block = IRBlock()
        const_values: dict[str, float] = {}

        for instr in block.instructions:
            # Check if all args are constants.  An arg is a compile-time
            # constant when it is a known const SSA value OR a numeric
            # literal (int/float, including a leading minus sign).
            def _is_const(arg: str) -> bool:
                if arg in const_values:
                    return True
                try:
                    float(arg)
                    return True
                except (TypeError, ValueError):
                    return False

            all_const = all(_is_const(arg) for arg in instr.args)

            if all_const and instr.op not in {"assign", "const"}:
                # Evaluate constant expression
                try:
                    arg_values = [
                        const_values[arg] if arg in const_values else float(arg)
                        for arg in instr.args
                    ]
                    result = IROptimizer._eval_const(instr.op, arg_values)
                    out = instr.out
                    const_values[out] = result
                    new_block.emit("const", out, [str(result)])
                    continue
                except Exception:
                    # Evaluation failed, keep original
                    pass

            # Keep original instruction
            new_block.add(IRInstruction(instr.op, instr.out, instr.args.copy()))

            # Track constant definitions
            if instr.op == "const":
                try:
                    const_values[instr.out] = float(instr.args[0])
                except (TypeError, ValueError):
                    pass

        return new_block

    @staticmethod
    def _eval_const(op: str, args: list[float]) -> float:
        """Evaluate constant operation."""
        if op == "add":
            return args[0] + args[1]
        elif op == "sub":
            return args[0] - args[1]
        elif op == "mul":
            return args[0] * args[1]
        elif op == "div":
            return args[0] / max(abs(args[1]), 1e-6)
        elif op == "pow":
            return args[0] ** args[1]
        elif op == "mod":
            return args[0] % max(abs(args[1]), 1e-6)
        elif op == "sin":
            return __import__("math").sin(args[0])
        elif op == "cos":
            return __import__("math").cos(args[0])
        elif op == "sqrt":
            return __import__("math").sqrt(max(args[0], 0.0))
        elif op == "abs":
            return abs(args[0])
        raise ValueError(f"Unknown constant op: {op}")

    @staticmethod
    def dead_code_elimination(block: IRBlock) -> IRBlock:
        """Remove unused instructions.

        Instructions whose output is never used are removed.
        """
        # Collect all uses
        uses: set[str] = set()
        for instr in block.instructions:
            for arg in instr.args:
                if arg.startswith(("t", "c")):  # SSA temps
                    uses.add(arg)

        # Keep instructions whose output is used
        new_block = IRBlock()
        for instr in reversed(block.instructions):
            if instr.out in uses or instr.op in {"assign"}:
                new_block.add(instr)
                uses.discard(instr.out)

        # Reverse back to original order
        new_block.instructions = list(reversed(new_block.instructions))
        return new_block

    @staticmethod
    def common_subexpression_elimination(block: IRBlock) -> IRBlock:
        """Eliminate duplicate subexpressions.

        Example: sin(x) + sin(x) → t1=sin(x), t1+t1
        """
        new_block = IRBlock()
        expr_map: dict[tuple[str, tuple[str, ...]], str] = {}

        for instr in block.instructions:
            if instr.op in {"const", "assign"}:
                new_block.add(instr)
                continue

            # Create expression key
            key = (instr.op, tuple(instr.args))

            if key in expr_map:
                # Reuse previous result
                out = expr_map[key]
                # Rename this instruction's output to previous
                new_block.add(IRInstruction("const", instr.out, [out]))
            else:
                new_block.add(instr)
                expr_map[key] = instr.out

        return new_block

    @staticmethod
    def strength_reduction(block: IRBlock) -> IRBlock:
        """Replace expensive operations with cheaper equivalents.

        Example: x^2 → x*x
        """
        new_block = IRBlock()

        for instr in block.instructions:
            if instr.op == "pow" and instr.args[1] == "2":
                # x^2 → x*x
                out = instr.out
                new_block.emit("mul", out, [instr.args[0], instr.args[0]])
            elif instr.op == "pow" and instr.args[1] == "0.5":
                # x^0.5 → sqrt(x)
                out = instr.out
                new_block.emit("call:sqrt", out, [instr.args[0]])
            else:
                new_block.add(instr)

        return new_block

    @staticmethod
    def optimize(block: IRBlock) -> IRBlock:
        """Run all optimization passes in order."""
        block = IROptimizer.constant_folding(block)
        block = IROptimizer.dead_code_elimination(block)
        block = IROptimizer.common_subexpression_elimination(block)
        block = IROptimizer.strength_reduction(block)
        return block


def hash_ir(block: IRBlock) -> str:
    """Generate SHA256 hash of IR block for caching.

    Hash is deterministic: same IR always produces same hash.
    """
    # Serialize instructions
    lines = []
    for instr in block.instructions:
        lines.append(f"{instr.op}|{instr.out}|{'|'.join(instr.args)}")

    data = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
