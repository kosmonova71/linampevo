#!/usr/bin/env python3
"""
ASCEND V9 - Tokenizer (Strict, Deterministic)

No regex-heavy normalization. Clean tokenization for MilkDrop expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar


class TokenType(Enum):
    """Token types for V9 tokenizer."""

    NUMBER = auto()
    IDENT = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    OP = auto()
    COMMA = auto()
    SEMICOLON = auto()
    SPACE = auto()
    EOF = auto()


@dataclass
class Token:
    """Single token with type and value."""

    type: TokenType
    value: str
    pos: int  # Position in original string

    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}', pos={self.pos})"


class V9Tokenizer:
    """Strict tokenizer for MilkDrop expressions.

    Rules:
    - No implicit multiplication during tokenization
    - No string hacks after tokenization
    - Deterministic output
    - Track positions for error reporting
    """

    # Token patterns (ordered by priority)
    PATTERNS: ClassVar[list[tuple[TokenType, str]]] = [
        (TokenType.NUMBER, r"\d+\.?\d*"),  # Numbers: 123, 1.5
        (TokenType.IDENT, r"[a-zA-Z_][a-zA-Z0-9_]*"),  # Identifiers
        (TokenType.LPAREN, r"\("),
        (TokenType.RPAREN, r"\)"),
        (TokenType.LBRACKET, r"\["),
        (TokenType.RBRACKET, r"\]"),
        (TokenType.OP, r"[+\-*/%^|&~!<>=]+"),  # Operators
        (TokenType.COMMA, r","),
        (TokenType.SEMICOLON, r";"),
        (TokenType.SPACE, r"\s+"),
    ]

    def __init__(self):
        # Compile regex pattern
        pattern_parts = [f"(?P<{t.name}>{p})" for t, p in self.PATTERNS]
        self.regex = re.compile("|".join(pattern_parts))

    def tokenize(self, expr: str) -> list[Token]:
        """Tokenize expression into list of tokens.

        Args:
            expr: Input expression string

        Returns:
            List of tokens in order
        """
        tokens = []
        pos = 0

        for match in self.regex.finditer(expr):
            token_type = TokenType[match.lastgroup]
            value = match.group()
            pos = match.start()

            # Skip spaces
            if token_type == TokenType.SPACE:
                continue

            tokens.append(Token(token_type, value, pos))

        # Add EOF token
        if tokens:
            tokens.append(Token(TokenType.EOF, "", len(expr)))

        return tokens

    def tokenize_with_implicit_mult(self, expr: str) -> list[Token]:
        """Tokenize with implicit multiplication insertion.

        This is the ONLY place where implicit multiplication is handled.
        Uses lookahead to prevent mangling ternary expressions.
        """
        tokens = self.tokenize(expr)

        if not tokens:
            return tokens

        # Insert implicit multiplication with lookahead
        result = []
        python_keywords = {"if", "else", "and", "or", "not", "in", "is", "for", "while", "return"}
        milkdrop_functions = {
            "sin",
            "cos",
            "tan",
            "sqrt",
            "abs",
            "asin",
            "acos",
            "atan",
            "atan2",
            "log",
            "log10",
            "exp",
            "floor",
            "ceil",
            "round",
            "sign",
            "min",
            "max",
            "sqr",
            "pow",
            "mod",
            "band",
            "bor",
            "bnot",
            "if",
            "above",
            "below",
            "equal",
            "rand",
        }

        for i, token in enumerate(tokens):
            # Get next token for lookahead
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None

            # Check if we should insert implicit multiplication
            if next_token and token.type != TokenType.EOF:
                should_insert = self._should_insert_implicit_mult(
                    token, next_token, python_keywords, milkdrop_functions
                )

                if should_insert:
                    result.append(Token(TokenType.OP, "*", token.pos + len(token.value)))

            result.append(token)

        return result

    def _should_insert_implicit_mult(
        self, current: Token, next_token: Token, python_keywords: set, milkdrop_functions: set
    ) -> bool:
        """Check if implicit multiplication should be inserted between tokens."""
        # Cases where implicit mult is valid:
        # NUMBER followed by IDENT/LPAREN: 2x, 2(x)
        # IDENT followed by IDENT/LPAREN (if not function call): xy, x(y)
        # RPAREN followed by IDENT/LPAREN: (x)y, (x)(y)

        # Check if current is keyword or function
        is_current_keyword = current.value in python_keywords
        is_current_func = current.value in milkdrop_functions

        # Check if next is keyword
        is_next_keyword = next_token.value in python_keywords

        # Function call: IDENT followed by LPAREN
        is_function_call = (
            current.type == TokenType.IDENT
            and next_token.type == TokenType.LPAREN
            and current.value in milkdrop_functions
        )

        if is_function_call:
            return False

        if is_current_keyword or is_next_keyword or is_current_func:
            return False

        # Valid implicit mult patterns
        valid_pairs = [
            (TokenType.NUMBER, TokenType.IDENT),
            (TokenType.NUMBER, TokenType.LPAREN),
            (TokenType.IDENT, TokenType.IDENT),
            (TokenType.IDENT, TokenType.LPAREN),
            (TokenType.RPAREN, TokenType.IDENT),
            (TokenType.RPAREN, TokenType.LPAREN),
        ]

        return (current.type, next_token.type) in valid_pairs

    def tokens_to_string(self, tokens: list[Token]) -> str:
        """Reconstruct string from tokens (for debugging)."""
        return "".join(t.value for t in tokens if t.type != TokenType.EOF)
