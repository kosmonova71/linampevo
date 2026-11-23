#!/usr/bin/env python3
"""
ASCEND V9 - AST to SSA Lowering

Converts Python AST to SSA IR (Static Single Assignment).
Deterministic, linear instruction sequence.
"""

from __future__ import annotations

import ast

from engine.ir.ssa import IRBlock, IRBuilder


class ASTToSSALowerer(ast.NodeVisitor):
    """Lower Python AST to SSA IR.

    Converts AST nodes to linear SSA instructions.
    Maintains SSA form: each variable assigned exactly once.
    """

    def __init__(self):
        self.builder = IRBuilder()

    def lower(self, tree: ast.AST) -> IRBlock:
        """Lower AST to SSA IR block.

        Args:
            tree: Python AST node

        Returns:
            IRBlock with SSA instructions
        """
        if isinstance(tree, ast.Expression):
            result = self.visit(tree.body)
            # Emit final assignment to output
            self.builder.emit_assign("_result", result)
        elif isinstance(tree, ast.Module):
            for stmt in tree.body:
                self.visit(stmt)
        else:
            self.visit(tree)

        return self.builder.build()

    def visit_Expression(self, node: ast.Expression) -> str:
        return self.visit(node.body)

    def visit_Expr(self, node: ast.Expr) -> str:
        return self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> str:
        if isinstance(node.value, (int, float)):
            return self.builder.emit_const(float(node.value))
        return self.builder.emit_const(0.0)

    def visit_Name(self, node: ast.Name) -> str:
        return self.builder.ssa_name(node.id)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        operand = self.visit(node.operand)

        if isinstance(node.op, ast.UAdd):
            return operand  # No-op
        elif isinstance(node.op, ast.USub):
            return self.builder.emit_unop("neg", operand)
        elif isinstance(node.op, ast.Not):
            return self.builder.emit_unop("not", operand)

        return operand

    def visit_BinOp(self, node: ast.BinOp) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)

        op_map = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.FloorDiv: "div_floor",
            ast.Mod: "mod",
            ast.Pow: "pow",
            ast.LShift: "shl",
            ast.RShift: "shr",
            ast.BitOr: "or",
            ast.BitXor: "xor",
            ast.BitAnd: "and",
        }

        op = op_map.get(type(node.op), "unknown")
        return self.builder.emit_binop(op, left, right)

    def visit_Compare(self, node: ast.Compare) -> str:
        left = self.visit(node.left)

        # Handle chained comparisons by converting to AND
        results = []
        prev = left
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)

            op_map = {
                ast.Eq: "eq",
                ast.NotEq: "neq",
                ast.Lt: "lt",
                ast.LtE: "lte",
                ast.Gt: "gt",
                ast.GtE: "gte",
                ast.Is: "eq",
                ast.IsNot: "neq",
                ast.In: "in",
                ast.NotIn: "notin",
            }

            op = op_map.get(type(op), "unknown")
            result = self.builder.emit_binop(op, prev, right)
            results.append(result)
            prev = right

        # Chain with AND if multiple comparisons
        if len(results) == 1:
            return results[0]
        elif len(results) > 1:
            return self.builder.emit_binop("and", results[0], results[1])

        return left

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        values = [self.visit(v) for v in node.values]

        op = "and" if isinstance(node.op, ast.And) else "or"

        result = values[0]
        for val in values[1:]:
            result = self.builder.emit_binop(op, result, val)

        return result

    def visit_Call(self, node: ast.Call) -> str:
        func_name = self._get_func_name(node.func)
        args = [self.visit(arg) for arg in node.args]

        return self.builder.emit_call(func_name, args)

    def _get_func_name(self, node: ast.expr) -> str:
        """Extract function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return "unknown"

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise ValueError("Only single assignment supported")

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise TypeError("Only name assignment supported")

        value = self.visit(node.value)
        self.builder.emit_assign(target.id, value)

    def visit_If(self, node: ast.If) -> str:
        test = self.visit(node.test)
        then_val = self.visit(node.body[0]) if node.body else "0"
        else_val = self.visit(node.orelse[0]) if node.orelse else "0"

        return self.builder.emit_call("if", [test, then_val, else_val])

    def visit_Subscript(self, node: ast.Subscript) -> str:
        value = self.visit(node.value)
        slice_val = self.visit(node.slice)

        return self.builder.emit_call("index", [value, slice_val])


def lower_to_ssa(tree: ast.AST) -> IRBlock:
    """Lower AST to SSA IR.

    Main entry point for AST → SSA conversion.
    """
    lowerer = ASTToSSALowerer()
    return lowerer.lower(tree)
