"""evo.milkdrop -- load and compile Milkdrop .milk presets.

This sub-package turns the classic Milkdrop preset format ([preset00]
header, per_frame_ / per_pixel_ equations, and the flat
key=value parameter block) into a structured MilkdropPreset that the
renderer can consume.

The compilation pipeline reuses the existing V9 engine
(parser / compiler / ir / gl)
so the Milkdrop expression syntax is parsed deterministically and lowered
to SSA IR before being emitted as GLSL.
"""

from __future__ import annotations

from .compiler import MilkdropCompiler, compile_preset_to_glsl
from .preset import MilkdropPreset, parse_milkdrop_file, parse_milkdrop_text
from .renderer import MilkdropShaderRenderer

__all__ = [
    "MilkdropCompiler",
    "MilkdropPreset",
    "MilkdropShaderRenderer",
    "compile_preset_to_glsl",
    "parse_milkdrop_file",
    "parse_milkdrop_text",
]
