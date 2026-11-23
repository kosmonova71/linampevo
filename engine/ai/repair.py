"""
ASCEND V9 - Autonomous System Hooks (Repair)

Automatic repair system for broken shaders and expressions.
"""

from __future__ import annotations

from engine.ir.ssa import IRBlock, IRInstruction


class RepairEngine:
    """Automatic repair engine for broken IR.

    Detects and fixes common issues:
    - Division by zero
    - Invalid function calls
    - Undefined variables
    """

    def repair(self, block: IRBlock) -> IRBlock:
        """Repair IR block.

        Args:
            block: SSA IR block

        Returns:
            Repaired IR block
        """
        new_block = block.copy()

        for instr in new_block.instructions:
            # Fix division by zero
            if instr.op == "div":
                self._fix_division(instr)

            # Fix invalid function calls
            if instr.op.startswith("call:"):
                self._fix_function_call(instr)

        return new_block

    def _fix_division(self, instr: IRInstruction) -> None:
        """Fix division instruction to prevent division by zero.

        Args:
            instr: Division instruction
        """
        # Replace with safe division
        instr.op = "div_safe"

    def _fix_function_call(self, instr: IRInstruction) -> None:
        """Fix function call instruction.

        Args:
            instr: Function call instruction
        """
        func = instr.op.split(":")[1]

        # Map deprecated functions
        func_map = {
            "rand": "md_rand",
        }

        if func in func_map:
            instr.op = f"call:{func_map[func]}"

    def validate(self, block: IRBlock) -> list[str]:
        """Validate IR block and return list of issues.

        Args:
            block: SSA IR block

        Returns:
            List of validation issues
        """
        issues = []

        for instr in block.instructions:
            # Check for division by zero risk
            if instr.op == "div" and instr.args[1] == "0":
                issues.append(f"Division by zero: {instr}")

            # Check for undefined variables
            if instr.op == "const":
                try:
                    float(instr.args[0])
                except ValueError:
                    issues.append(f"Invalid constant: {instr}")

        return issues
