#!/usr/bin/env python3
"""
ASCEND V9 - Parser (Deterministic AST Generation)

Converts tokens to AST using Python's ast module.
Deterministic, no string hacks after parse.
"""

from __future__ import annotations

import ast
from typing import ClassVar

from engine.parser.tokenizer import V9Tokenizer


class V9Parser:
    """Deterministic parser for MilkDrop expressions.

    Uses Python's ast module for parsing, ensuring deterministic output.
    No string manipulation after AST generation.
    """

    def __init__(self):
        self.tokenizer = V9Tokenizer()

    def parse(self, expr: str) -> ast.AST:
        """Parse expression into AST.

        Args:
            expr: Input expression string

        Returns:
            Python AST node

        Raises:
            SyntaxError: If expression is invalid
        """
        # Tokenize with implicit multiplication
        tokens = self.tokenizer.tokenize_with_implicit_mult(expr)

        # Reconstruct string from tokens
        code = self.tokenizer.tokens_to_string(tokens)

        # Parse using Python's ast module
        try:
            tree = ast.parse(code, mode="eval")
            return tree
        except SyntaxError as e:
            # Try exec mode for statements
            try:
                tree = ast.parse(code, mode="exec")
                return tree
            except SyntaxError:
                raise SyntaxError(f"Failed to parse: {expr}") from e

    def parse_expression(self, expr: str) -> ast.Expression:
        """Parse as expression (single value)."""
        tree = self.parse(expr)
        if isinstance(tree, ast.Expression):
            return tree
        if isinstance(tree, ast.Module) and len(tree.body) == 1:
            return ast.Expression(tree.body[0].value)
        raise SyntaxError(f"Expected expression, got statement: {expr}")


class ASTTransformer(ast.NodeTransformer):
    """Transform Python AST to MilkDrop-compatible AST.

    Handles:
    - Ternary if() conversion
    - Function name mapping
    - Operator normalization
    """

    # MilkDrop function mappings
    FUNC_MAP: ClassVar[dict[str, str]] = {
        "if": "md_if",
        "above": "md_above",
        "below": "md_below",
        "equal": "md_equal",
        "sqr": "md_sqr",
        "sign": "md_sign",
        "min": "md_min",
        "max": "md_max",
        "band": "md_band",
        "bor": "md_bor",
        "bnot": "md_bnot",
        "pow": "md_pow",
    }

    def visit_IfExp(self, node: ast.IfExp) -> ast.Call:
        """Convert ternary if to md_if function call.

        Python: x if cond else y
        MilkDrop: md_if(cond, x, y)
        """
        return ast.Call(
            func=ast.Name(id="md_if", ctx=ast.Load()),
            args=[self.visit(node.test), self.visit(node.body), self.visit(node.orelse)],
            keywords=[],
        )

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Transform function calls."""
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Map to MilkDrop function if needed
            if func_name in self.FUNC_MAP:
                node.func.id = self.FUNC_MAP[func_name]

        # Recursively visit arguments
        node.args = [self.visit(arg) for arg in node.args]
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:
        """Convert boolean operations to proper GLSL logic."""
        if isinstance(node.op, ast.And) and len(node.values) == 2:
            a_above = ast.Call(
                func=ast.Name(id="md_above", ctx=ast.Load()),
                args=[self.visit(node.values[0]), ast.Constant(value=0.0)],
                keywords=[],
            )
            b_above = ast.Call(
                func=ast.Name(id="md_above", ctx=ast.Load()),
                args=[self.visit(node.values[1]), ast.Constant(value=0.0)],
                keywords=[],
            )
            return ast.Call(
                func=ast.Name(id="md_band", ctx=ast.Load()), args=[a_above, b_above], keywords=[]
            )
        elif isinstance(node.op, ast.Or) and len(node.values) == 2:
            a_above = ast.Call(
                func=ast.Name(id="md_above", ctx=ast.Load()),
                args=[self.visit(node.values[0]), ast.Constant(value=0.0)],
                keywords=[],
            )
            b_above = ast.Call(
                func=ast.Name(id="md_above", ctx=ast.Load()),
                args=[self.visit(node.values[1]), ast.Constant(value=0.0)],
                keywords=[],
            )
            return ast.Call(
                func=ast.Name(id="md_bor", ctx=ast.Load()), args=[a_above, b_above], keywords=[]
            )

        # Keep original for other cases
        return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> ast.Call:
        """Convert comparison to md_above/below/equal if appropriate."""
        if len(node.ops) == 1 and len(node.comparators) == 1:
            op = node.ops[0]
            if isinstance(op, ast.Gt):
                return ast.Call(
                    func=ast.Name(id="md_above", ctx=ast.Load()),
                    args=[self.visit(node.left), self.visit(node.comparators[0])],
                    keywords=[],
                )
            elif isinstance(op, ast.Lt):
                return ast.Call(
                    func=ast.Name(id="md_below", ctx=ast.Load()),
                    args=[self.visit(node.left), self.visit(node.comparators[0])],
                    keywords=[],
                )
            elif isinstance(op, ast.Eq):
                return ast.Call(
                    func=ast.Name(id="md_equal", ctx=ast.Load()),
                    args=[self.visit(node.left), self.visit(node.comparators[0])],
                    keywords=[],
                )

        # Keep original comparison for other cases
        return self.generic_visit(node)


def parse_and_transform(expr: str) -> ast.AST:
    """Parse expression and apply AST transformations.

    This is the main entry point for the parser pipeline.
    """
    parser = V9Parser()
    tree = parser.parse(expr)
    transformer = ASTTransformer()
    transformed = transformer.visit(tree)
    ast.fix_missing_locations(transformed)
    return transformed
