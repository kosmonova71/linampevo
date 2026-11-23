#!/usr/bin/env python3

"""evo_shader_graph.py

Dynamic shader invention for the EvoRenderer project.

Instead of mutating constants, this module evolves shader *structure* by
representing the shader as a graph of typed nodes.  Nodes can duplicate,
split, merge, or disappear, creating genuinely new rendering algorithms.

Architecture
------------
* ``ShaderNode``      - one operation (noise, warp, palette, mix, sample, ...)
* ``ShaderGraph``     - DAG of nodes with typed edges
* ``ShaderGraphGenome`` - evolutionary wrapper (mutate, crossover, clone)
* ``ShaderGraphCompiler`` - converts a graph into valid GLSL 330 fragment shader
* ``NodeLibrary``     - catalog of safe, validated node templates

Integration
-----------
The graph compiler produces GLSL source that can be fed into the existing
``_compile_program`` pipeline in ``evo_visualizer.py``.
"""

from __future__ import annotations

import copy
import logging
import random
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Type system
# ============================================================================

VALID_TYPES = {"float", "vec2", "vec3", "vec4", "sampler2D", "bool"}
# Cast table keyed by (from_type, to_type). Missing entries mean the variable
# is used as-is (same type). These keep the generated shader valid even when
# evolution wires an edge whose source type differs from the destination's
# declared input type -- previously that produced invalid GLSL (e.g. passing a
# vec3 into a float argument) and a shader compile failure.
TYPE_CAST = {
    ("float", "vec2"): "vec2({0}, {0})",
    ("float", "vec3"): "vec3({0})",
    ("float", "vec4"): "vec4({0})",
    ("float", "bool"): "bool({0})",
    ("vec2", "float"): "{0}.x",
    ("vec2", "vec3"): "vec3({0}, 0.0)",
    ("vec2", "vec4"): "vec4({0}, 0.0, 1.0)",
    ("vec3", "float"): "{0}.x",
    ("vec3", "vec2"): "{0}.xy",
    ("vec3", "vec4"): "vec4({0}, 1.0)",
    ("vec4", "float"): "{0}.x",
    ("vec4", "vec2"): "{0}.xy",
    ("vec4", "vec3"): "{0}.xyz",
    ("bool", "float"): "float({0})",
}

# Uniform metadata types — every node can declare what uniforms it needs
# so the compiler auto-generates their declarations instead of hardcoding.
# (name, glsl_type, default_value)  — default is only used when no node
# explicitly drives it; e.g. previous_frame defaults to sampler2D(0).
DEFAULT_UNIFORMS: dict[str, tuple[str, Any]] = {
    "u_time": ("float", 0.0),
    "u_resolution": ("vec2", (1.0, 1.0)),
    "bass": ("float", 0.0),
    "mid": ("float", 0.0),
    "treble": ("float", 0.0),
    "previous_frame": ("sampler2D", 0),
}

# ============================================================================
# Node templates
# ============================================================================

NODE_LIBRARY: dict[str, dict[str, Any]] = {
    # ---- Inputs -----------------------------------------------------------
    "input_uv": {
        "category": "input",
        "outputs": {"uv": "vec2"},
        "params": {},
        "uniforms": {},
    },
    "input_time": {
        "category": "input",
        "outputs": {"t": "float"},
        "params": {},
        "uniforms": {"u_time": "float"},
    },
    "input_audio_bass": {
        "category": "input",
        "outputs": {"bass": "float"},
        "params": {},
        "uniforms": {"bass": "float"},
    },
    "input_audio_mid": {
        "category": "input",
        "outputs": {"mid": "float"},
        "params": {},
        "uniforms": {"mid": "float"},
    },
    "input_audio_treble": {
        "category": "input",
        "outputs": {"treble": "float"},
        "params": {},
        "uniforms": {"treble": "float"},
    },
    # ---- Fields -----------------------------------------------------------
    "noise_fbm": {
        "category": "field",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"n": "float"},
        "params": {"scale": ("float", 3.0), "layers": ("int", 4)},
        "uniforms": {},
    },
    "noise_voronoi": {
        "category": "field",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"d": "float"},
        "params": {"scale": ("float", 5.0), "jitter": ("float", 1.0)},
        "uniforms": {},
    },
    "noise_rd": {
        "category": "field",
        "inputs": {"uv": "vec2", "t": "float"},
        "outputs": {"n": "float"},
        "params": {"feed": ("float", 0.055), "kill": ("float", 0.062)},
        "uniforms": {},
    },
    "noise_mandelbrot": {
        "category": "field",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"n": "float"},
        "params": {"max_iter": ("int", 64), "zoom": ("float", 1.0)},
        "uniforms": {},
    },
    "sdf_field": {
        "category": "field",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"d": "float", "surface": "vec3"},
        "params": {
            "primitive": ("int", 0),
            "march_steps": ("int", 32),
            "camera_dist": ("float", 3.5),
        },
        "uniforms": {"u_resolution": "vec2"},
    },
    # ---- Warps ------------------------------------------------------------
    "warp_curl": {
        "category": "warp",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"wp": "vec2"},
        "params": {"strength": ("float", 0.5), "scale": ("float", 1.0)},
        "uniforms": {},
    },
    "warp_vortex": {
        "category": "warp",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"wp": "vec2"},
        "params": {"strength": ("float", 0.3), "arms": ("float", 3.0)},
        "uniforms": {},
    },
    "warp_twist": {
        "category": "warp",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"wp": "vec2"},
        "params": {"amount": ("float", 0.5)},
        "uniforms": {},
    },
    "curl_noise": {
        "category": "warp",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"wp": "vec2"},
        "params": {"scale": ("float", 3.0), "strength": ("float", 0.5), "octaves": ("int", 4)},
        "uniforms": {},
    },
    "domain_warp": {
        "category": "warp",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"wp": "vec2"},
        "params": {"strength": ("float", 0.5), "scale": ("float", 2.0), "octaves": ("int", 3)},
        "uniforms": {},
    },
    # ---- Palettes ---------------------------------------------------------
    "palette_cosine": {
        "category": "palette",
        "inputs": {"hue": "float", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"palette_id": ("int", 0), "shift": ("float", 0.0)},
        "uniforms": {},
    },
    "palette_spectral": {
        "category": "palette",
        "inputs": {"hue": "float", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"wl_shift": ("float", 0.0), "dispersion": ("float", 0.0)},
        "uniforms": {},
    },
    "palette_plasma": {
        "category": "palette",
        "inputs": {"hue": "float", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"speed": ("float", 1.0), "complexity": ("float", 1.0)},
        "uniforms": {},
    },
    # ---- Post effects -----------------------------------------------------
    "post_glow": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "prev": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {"intensity": ("float", 0.5)},
        "uniforms": {"previous_frame": "sampler2D"},
    },
    "post_bloom": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "prev": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {"threshold": ("float", 0.6), "intensity": ("float", 0.5)},
        "uniforms": {"previous_frame": "sampler2D"},
    },
    "post_chromatic": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "prev": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {"amount": ("float", 0.003)},
        "uniforms": {"previous_frame": "sampler2D"},
    },
    "post_feedback": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "prev": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {"alpha": ("float", 0.85)},
        "uniforms": {"previous_frame": "sampler2D"},
    },
    "post_distortion": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "prev": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {"amount": ("float", 0.5)},
        "uniforms": {},
    },
    # New post modules from evo/post/ .glsl files
    "post_polar_galaxy": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"swirl_strength": ("float", 0.35)},
        "uniforms": {"previous_frame": "sampler2D", "u_time": "float"},
    },
    "post_cells": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"mix_amount": ("float", 0.35)},
        "uniforms": {"u_time": "float"},
    },
    "post_nonlinear_trails": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"ghost_gain": ("float", 0.25)},
        "uniforms": {"previous_frame": "sampler2D", "u_time": "float"},
    },
    "post_audio_topology": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"bass_warp": ("float", 0.15), "treble_shear": ("float", 0.05)},
        "uniforms": {
            "previous_frame": "sampler2D",
            "bass": "float",
            "mid": "float",
            "treble": "float",
            "u_time": "float",
        },
    },
    "post_depth_cloud": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"taps": ("int", 6), "mix": ("float", 0.4)},
        "uniforms": {"previous_frame": "sampler2D", "u_time": "float"},
    },
    "post_neon_edge": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"edge_strength": ("float", 2.0)},
        "uniforms": {"previous_frame": "sampler2D"},
    },
    # ---- Feedback transforms (UV warping for feedback sample) -------------
    "feedback_drift": {
        "category": "feedback",
        "inputs": {"uv": "vec2", "t": "float"},
        "outputs": {"fb_uv": "vec2"},
        "params": {"drift_x_speed": ("float", 0.2), "drift_y_speed": ("float", 0.17)},
        "uniforms": {"u_time": "float"},
    },
    "feedback_kaleido": {
        "category": "feedback",
        "inputs": {"uv": "vec2", "t": "float"},
        "outputs": {"fb_uv": "vec2"},
        "params": {"symmetry": ("int", 6)},
        "uniforms": {"u_time": "float"},
    },
    "feedback_mirror": {
        "category": "feedback",
        "inputs": {"uv": "vec2", "t": "float"},
        "outputs": {"fb_uv": "vec2"},
        "params": {"jitter": ("float", 0.01)},
        "uniforms": {"u_time": "float"},
    },
    "feedback_spiral": {
        "category": "feedback",
        "inputs": {"uv": "vec2", "t": "float"},
        "outputs": {"fb_uv": "vec2"},
        "params": {"rotate_speed": ("float", 0.3), "breathe_speed": ("float", 0.5)},
        "uniforms": {"u_time": "float"},
    },
    "feedback_zoom": {
        "category": "feedback",
        "inputs": {"uv": "vec2", "t": "float"},
        "outputs": {"fb_uv": "vec2"},
        "params": {"zoom_speed": ("float", 0.6), "zoom_amount": ("float", 0.012)},
        "uniforms": {"u_time": "float"},
    },
    # ---- Math helpers -----------------------------------------------------
    "math_sin": {
        "category": "math",
        "inputs": {"x": "float"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_cos": {
        "category": "math",
        "inputs": {"x": "float"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_mix": {
        "category": "math",
        "inputs": {"a": "vec3", "b": "vec3", "t": "float"},
        "outputs": {"r": "vec3"},
        "params": {},
        "uniforms": {},
    },
    "math_smoothstep": {
        "category": "math",
        "inputs": {"edge0": "float", "edge1": "float", "x": "float"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_mul": {
        "category": "math",
        "inputs": {"a": "float", "b": "float"},
        "outputs": {"r": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_add": {
        "category": "math",
        "inputs": {"a": "float", "b": "float"},
        "outputs": {"r": "float"},
        "params": {},
        "uniforms": {},
    },
    # ---- Sample / feedback read -------------------------------------------
    "sample_prev": {
        "category": "sample",
        "inputs": {"uv": "vec2"},
        "outputs": {"col": "vec3"},
        "params": {},
        "uniforms": {"previous_frame": "sampler2D"},
    },
    # ---- Outputs ----------------------------------------------------------
    "output_color": {
        "category": "output",
        "inputs": {"col": "vec3"},
        "outputs": {},
        "params": {},
        "uniforms": {},
    },
    "output_feedback": {
        "category": "output",
        "inputs": {"col": "vec3"},
        "outputs": {},
        "params": {"blend_mode": ("int", 0), "input_source": ("int", 0)},
        "uniforms": {},
    },
    "output_post": {
        "category": "output",
        "inputs": {"col": "vec3"},
        "outputs": {},
        "params": {"blend_mode": ("int", 0), "input_source": ("int", 0)},
        "uniforms": {},
    },
    # ---- Field multi-output variants --------------------------------------
    "voronoi_cells": {
        "category": "field",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"cells": "vec3", "dist": "float"},
        "params": {"scale": ("float", 5.0), "jitter": ("float", 1.0), "speed": ("float", 0.5)},
        "uniforms": {},
    },
    "reaction_diffusion": {
        "category": "field",
        "inputs": {"uv": "vec2", "t": "float"},
        "outputs": {"pattern": "float"},
        "params": {
            "feed": ("float", 0.055),
            "kill": ("float", 0.062),
            "iterations": ("int", 8),
            "speed": ("float", 1.0),
        },
        "uniforms": {},
    },
    "voronoi_edge": {
        "category": "field",
        "inputs": {"p": "vec2", "t": "float"},
        "outputs": {"edge": "float"},
        "params": {"scale": ("float", 5.0), "thickness": ("float", 0.05)},
        "uniforms": {},
    },
    # ---- Lighting ----------------------------------------------------------
    "lighting_point": {
        "category": "lighting",
        "inputs": {"col": "vec3", "p": "vec2", "n": "float"},
        "outputs": {"col": "vec3"},
        "params": {"intensity": ("float", 0.8), "falloff": ("float", 2.0)},
        "uniforms": {"u_resolution": "vec2"},
    },
    "lighting_ambient": {
        "category": "lighting",
        "inputs": {"col": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {"intensity": ("float", 0.3), "color": ("vec3", (0.5, 0.5, 0.5))},
        "uniforms": {},
    },
    "lighting_bloom": {
        "category": "lighting",
        "inputs": {"col": "vec3", "uv": "vec2"},
        "outputs": {"col": "vec3"},
        "params": {
            "threshold": ("float", 0.6),
            "intensity": ("float", 0.5),
            "radius": ("float", 0.4),
        },
        "uniforms": {"previous_frame": "sampler2D"},
    },
    # ---- Particles ---------------------------------------------------------
    "particle_field": {
        "category": "particle",
        "inputs": {"uv": "vec2", "t": "float"},
        "outputs": {"col": "vec3", "depth": "float"},
        "params": {"count": ("int", 32), "size": ("float", 0.02), "speed": ("float", 0.5)},
        "uniforms": {"u_time": "float", "u_resolution": "vec2"},
    },
    "particle_trails": {
        "category": "particle",
        "inputs": {"col": "vec3", "uv": "vec2"},
        "outputs": {"col": "vec3"},
        "params": {"length": ("float", 0.3), "fade": ("float", 0.9)},
        "uniforms": {"previous_frame": "sampler2D", "u_time": "float"},
    },
    # ---- Volume / fog ------------------------------------------------------
    "volume_fog": {
        "category": "volume",
        "inputs": {"col": "vec3", "uv": "vec2", "depth": "float"},
        "outputs": {"col": "vec3"},
        "params": {"density": ("float", 0.5), "falloff": ("float", 1.0), "steps": ("int", 8)},
        "uniforms": {"u_time": "float"},
    },
    "volume_cloud": {
        "category": "volume",
        "inputs": {"col": "vec3", "uv": "vec2", "t": "float"},
        "outputs": {"col": "vec3"},
        "params": {"scale": ("float", 3.0), "density": ("float", 0.4), "speed": ("float", 0.2)},
        "uniforms": {"u_time": "float"},
    },
    # ---- Camera / projection -----------------------------------------------
    "camera_lens": {
        "category": "camera",
        "inputs": {"uv": "vec2", "t": "float"},
        "outputs": {"uv": "vec2", "dist": "float"},
        "params": {"distortion": ("float", 0.1), "vignette": ("float", 0.3)},
        "uniforms": {"u_resolution": "vec2", "u_time": "float"},
    },
    "camera_dof": {
        "category": "camera",
        "inputs": {"col": "vec3", "uv": "vec2", "depth": "float"},
        "outputs": {"col": "vec3"},
        "params": {"focus": ("float", 0.5), "blur": ("float", 0.1), "radius": ("float", 0.02)},
        "uniforms": {"previous_frame": "sampler2D", "u_resolution": "vec2"},
    },
    # ---- Composite / blend -------------------------------------------------
    "composite_add": {
        "category": "composite",
        "inputs": {"a": "vec3", "b": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {},
        "uniforms": {},
    },
    "composite_mul": {
        "category": "composite",
        "inputs": {"a": "vec3", "b": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {},
        "uniforms": {},
    },
    "composite_screen": {
        "category": "composite",
        "inputs": {"a": "vec3", "b": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {},
        "uniforms": {},
    },
    "composite_overlay": {
        "category": "composite",
        "inputs": {"a": "vec3", "b": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {},
        "uniforms": {},
    },
    "math_clamp": {
        "category": "math",
        "inputs": {"x": "float", "lo": "float", "hi": "float"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_fract": {
        "category": "math",
        "inputs": {"x": "float"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_mod": {
        "category": "math",
        "inputs": {"x": "float", "y": "float"},
        "outputs": {"r": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_abs": {
        "category": "math",
        "inputs": {"x": "float"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_pow": {
        "category": "math",
        "inputs": {"x": "float", "exp": "float"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_sqrt": {
        "category": "math",
        "inputs": {"x": "float"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_length": {
        "category": "math",
        "inputs": {"v": "vec2"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_dot": {
        "category": "math",
        "inputs": {"a": "vec2", "b": "vec2"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "math_normalize": {
        "category": "math",
        "inputs": {"v": "vec2"},
        "outputs": {"r": "vec2"},
        "params": {},
        "uniforms": {},
    },
    "math_step": {
        "category": "math",
        "inputs": {"edge": "float", "x": "float"},
        "outputs": {"y": "float"},
        "params": {},
        "uniforms": {},
    },
    "sample_blur": {
        "category": "sample",
        "inputs": {"uv": "vec2", "col": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {"radius": ("float", 0.01), "quality": ("int", 3)},
        "uniforms": {"previous_frame": "sampler2D"},
    },
    "output_bloom": {
        "category": "output",
        "inputs": {"col": "vec3"},
        "outputs": {},
        "params": {"blend_mode": ("int", 0), "input_source": ("int", 0)},
        "uniforms": {},
    },
    "output_composite": {
        "category": "output",
        "inputs": {"col": "vec3"},
        "outputs": {},
        "params": {"blend_mode": ("int", 0), "input_source": ("int", 0)},
        "uniforms": {},
    },
    # ---- Legacy aliases (for backward compat) -----------------------------
    "glow_node": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "prev": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {
            "intensity": ("float", 0.5),
            "radius": ("float", 0.4),
            "threshold": ("float", 0.3),
        },
        "uniforms": {"previous_frame": "sampler2D"},
    },
    "bloom_node": {
        "category": "post",
        "inputs": {"col": "vec3", "uv": "vec2", "prev": "vec3"},
        "outputs": {"col": "vec3"},
        "params": {
            "intensity": ("float", 0.5),
            "threshold": ("float", 0.6),
            "radius": ("float", 0.5),
        },
        "uniforms": {"previous_frame": "sampler2D"},
    },
}

CATEGORY_ORDER = [
    "input",
    "field",
    "warp",
    "palette",
    "math",
    "sample",
    "lighting",
    "particle",
    "volume",
    "camera",
    "composite",
    "post",
    "output",
]

NODE_COST_METRICS: dict[str, dict[str, int]] = {
    "input_uv": {"instructions": 0, "alu": 0, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "input_time": {"instructions": 0, "alu": 0, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "input_audio_bass": {
        "instructions": 0,
        "alu": 0,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "input_audio_mid": {
        "instructions": 0,
        "alu": 0,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "input_audio_treble": {
        "instructions": 0,
        "alu": 0,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "noise_fbm": {"instructions": 16, "alu": 14, "texture_reads": 0, "branches": 0, "fps_cost": 2},
    "noise_voronoi": {
        "instructions": 18,
        "alu": 16,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 3,
    },
    "noise_rd": {"instructions": 64, "alu": 48, "texture_reads": 0, "branches": 0, "fps_cost": 5},
    "noise_mandelbrot": {
        "instructions": 128,
        "alu": 100,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 8,
    },
    "sdf_field": {"instructions": 96, "alu": 80, "texture_reads": 0, "branches": 0, "fps_cost": 6},
    "warp_curl": {"instructions": 12, "alu": 10, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "warp_vortex": {"instructions": 8, "alu": 6, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "warp_twist": {"instructions": 6, "alu": 4, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "curl_noise": {
        "instructions": 12,
        "alu": 10,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "domain_warp": {
        "instructions": 14,
        "alu": 12,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 2,
    },
    "palette_cosine": {
        "instructions": 6,
        "alu": 4,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "palette_spectral": {
        "instructions": 4,
        "alu": 2,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "palette_plasma": {
        "instructions": 8,
        "alu": 6,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "voronoi_cells": {
        "instructions": 20,
        "alu": 16,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 3,
    },
    "reaction_diffusion": {
        "instructions": 64,
        "alu": 48,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 5,
    },
    "voronoi_edge": {
        "instructions": 16,
        "alu": 12,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 2,
    },
    "math_sin": {"instructions": 2, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_cos": {"instructions": 2, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_mix": {"instructions": 2, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_smoothstep": {
        "instructions": 3,
        "alu": 2,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "math_mul": {"instructions": 1, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_add": {"instructions": 1, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_clamp": {"instructions": 2, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_fract": {"instructions": 1, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_mod": {"instructions": 2, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_abs": {"instructions": 1, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_pow": {"instructions": 4, "alu": 3, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "math_sqrt": {"instructions": 2, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_length": {"instructions": 3, "alu": 2, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_dot": {"instructions": 3, "alu": 2, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "math_normalize": {
        "instructions": 5,
        "alu": 3,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "math_step": {"instructions": 2, "alu": 1, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "sample_prev": {"instructions": 1, "alu": 0, "texture_reads": 1, "branches": 0, "fps_cost": 1},
    "sample_blur": {"instructions": 5, "alu": 2, "texture_reads": 1, "branches": 0, "fps_cost": 2},
    "post_glow": {"instructions": 3, "alu": 2, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "post_bloom": {"instructions": 3, "alu": 2, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "post_chromatic": {
        "instructions": 4,
        "alu": 2,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "post_feedback": {
        "instructions": 2,
        "alu": 1,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "post_distortion": {
        "instructions": 6,
        "alu": 4,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "post_polar_galaxy": {
        "instructions": 8,
        "alu": 6,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 2,
    },
    "post_cells": {"instructions": 6, "alu": 4, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "post_nonlinear_trails": {
        "instructions": 3,
        "alu": 2,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "post_audio_topology": {
        "instructions": 8,
        "alu": 6,
        "texture_reads": 1,
        "branches": 0,
        "fps_cost": 2,
    },
    "post_depth_cloud": {
        "instructions": 10,
        "alu": 6,
        "texture_reads": 2,
        "branches": 0,
        "fps_cost": 3,
    },
    "post_neon_edge": {
        "instructions": 20,
        "alu": 10,
        "texture_reads": 4,
        "branches": 0,
        "fps_cost": 4,
    },
    "feedback_drift": {
        "instructions": 4,
        "alu": 3,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "feedback_kaleido": {
        "instructions": 6,
        "alu": 4,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "feedback_mirror": {
        "instructions": 4,
        "alu": 3,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "feedback_spiral": {
        "instructions": 6,
        "alu": 4,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "feedback_zoom": {
        "instructions": 4,
        "alu": 3,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "lighting_point": {
        "instructions": 4,
        "alu": 3,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "lighting_ambient": {
        "instructions": 3,
        "alu": 2,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "lighting_bloom": {
        "instructions": 2,
        "alu": 1,
        "texture_reads": 1,
        "branches": 0,
        "fps_cost": 2,
    },
    "particle_field": {
        "instructions": 4,
        "alu": 3,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "particle_trails": {
        "instructions": 2,
        "alu": 1,
        "texture_reads": 1,
        "branches": 0,
        "fps_cost": 1,
    },
    "volume_fog": {"instructions": 3, "alu": 2, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "volume_cloud": {
        "instructions": 6,
        "alu": 4,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 1,
    },
    "camera_lens": {"instructions": 4, "alu": 3, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "camera_dof": {"instructions": 2, "alu": 1, "texture_reads": 1, "branches": 0, "fps_cost": 2},
    "composite_add": {
        "instructions": 1,
        "alu": 1,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "composite_mul": {
        "instructions": 1,
        "alu": 1,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "composite_screen": {
        "instructions": 2,
        "alu": 1,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "composite_overlay": {
        "instructions": 4,
        "alu": 2,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "output_color": {
        "instructions": 1,
        "alu": 0,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "output_feedback": {
        "instructions": 1,
        "alu": 0,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "output_post": {"instructions": 1, "alu": 0, "texture_reads": 0, "branches": 0, "fps_cost": 0},
    "output_bloom": {
        "instructions": 1,
        "alu": 0,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "output_composite": {
        "instructions": 1,
        "alu": 0,
        "texture_reads": 0,
        "branches": 0,
        "fps_cost": 0,
    },
    "glow_node": {"instructions": 3, "alu": 2, "texture_reads": 0, "branches": 0, "fps_cost": 1},
    "bloom_node": {"instructions": 3, "alu": 2, "texture_reads": 0, "branches": 0, "fps_cost": 1},
}


# ============================================================================
# Graph classes
# ============================================================================


class ShaderNode:
    """One shader operation node in the graph."""

    def __init__(
        self, ntype: str, node_id: str | None = None, params: dict[str, Any] | None = None
    ):
        self.ntype = ntype
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.params: dict[str, Any] = dict(params or {})
        self.usage_count: int = 0
        self.last_used_generation: int = 0
        tpl = NODE_LIBRARY.get(self.ntype, {})
        for k, (_, default) in tpl.get("params", {}).items():
            self.params.setdefault(k, default)

    def clone(self) -> ShaderNode:
        dup = ShaderNode(self.ntype, self.node_id, copy.deepcopy(self.params))
        dup.usage_count = self.usage_count
        dup.last_used_generation = self.last_used_generation
        return dup

    def mutate(self, strength: float = 0.15) -> list[tuple[str, Any, Any]]:
        tpl = NODE_LIBRARY.get(self.ntype, {})
        changes: list[tuple[str, Any, Any]] = []
        for k, (ptype, default) in tpl.get("params", {}).items():
            if random.random() < strength:
                old = self.params[k]
                if ptype == "int":
                    new_val = max(1, min(128, int(self.params[k] + random.uniform(-1, 1))))
                elif isinstance(default, tuple):
                    new_val = list(old)
                    for i in range(len(old)):
                        if random.random() < strength:
                            new_val[i] = max(
                                0.0,
                                min(
                                    1.0,
                                    old[i] + random.uniform(-strength, strength) * abs(default[i]),
                                ),
                            )
                    new_val = tuple(new_val)
                else:
                    v = float(self.params[k]) + random.uniform(-strength, strength) * abs(
                        float(default)
                    )
                    new_val = max(0.0, v)
                if new_val != old:
                    self.params[k] = new_val
                    changes.append((k, old, new_val))
        return changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "ntype": self.ntype,
            "node_id": self.node_id,
            "params": dict(self.params),
            "usage_count": self.usage_count,
            "last_used_generation": self.last_used_generation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShaderNode:
        node = cls(d["ntype"], d.get("node_id"), d.get("params"))
        node.usage_count = int(d.get("usage_count", 0))
        node.last_used_generation = int(d.get("last_used_generation", 0))
        return node


# Edge weight — every connection can carry blend/weight params that modulate
# the signal strength.  Must be defined before ShaderEdge uses it.
DEFAULT_EDGE_WEIGHT = 1.0


class ShaderEdge:
    """Typed connection between two nodes, with an evolved weight.

    The weight gene modulates how strongly the source signal affects the
    destination (multiplied in the compiled GLSL).  Weight = 0 is a dead edge
    (pruned), weight < 0 inverts the signal, weight > 1 amplifies.
    """

    def __init__(
        self,
        src_id: str,
        src_port: str,
        dst_id: str,
        dst_port: str,
        dtype: str = "float",
        weight: float = DEFAULT_EDGE_WEIGHT,
    ):
        self.src_id = src_id
        self.src_port = src_port
        self.dst_id = dst_id
        self.dst_port = dst_port
        self.dtype = dtype
        self.weight = float(weight)

    def clone(self) -> ShaderEdge:
        return ShaderEdge(
            self.src_id, self.src_port, self.dst_id, self.dst_port, self.dtype, self.weight
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_id": self.src_id,
            "src_port": self.src_port,
            "dst_id": self.dst_id,
            "dst_port": self.dst_port,
            "dtype": self.dtype,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShaderEdge:
        return cls(
            d["src_id"],
            d["src_port"],
            d["dst_id"],
            d["dst_port"],
            d.get("dtype", "float"),
            d.get("weight", DEFAULT_EDGE_WEIGHT),
        )


class ShaderGraph:
    """DAG of shader nodes with typed edges."""

    def __init__(self):
        self.nodes: dict[str, ShaderNode] = {}
        self.edges: list[ShaderEdge] = []
        self._next_id = 0

    def add_node(self, ntype: str, params: dict[str, Any] | None = None) -> ShaderNode:
        node = ShaderNode(ntype, params=params)
        self.nodes[node.node_id] = node
        return node

    def remove_node(self, node_id: str) -> None:
        self.edges = [e for e in self.edges if e.src_id != node_id and e.dst_id != node_id]
        self.nodes.pop(node_id, None)

    def connect(
        self, src_id: str, src_port: str, dst_id: str, dst_port: str, dtype: str = "float"
    ) -> None:
        self.edges.append(ShaderEdge(src_id, src_port, dst_id, dst_port, dtype))

    def disconnect(self, src_id: str, dst_id: str) -> None:
        self.edges = [e for e in self.edges if not (e.src_id == src_id and e.dst_id == dst_id)]

    def get_inputs(self, node_id: str) -> dict[str, ShaderNode]:
        out: dict[str, ShaderNode] = {}
        for e in self.edges:
            if e.dst_id == node_id:
                out[e.dst_port] = self.nodes.get(e.src_id)
        return out

    def get_outputs(self, node_id: str) -> list[ShaderEdge]:
        return [e for e in self.edges if e.src_id == node_id]

    def node_by_type(self, ntype: str) -> list[ShaderNode]:
        return [n for n in self.nodes.values() if n.ntype == ntype]

    def clone(self) -> ShaderGraph:
        g = ShaderGraph()
        g._next_id = self._next_id
        for n in self.nodes.values():
            g.nodes[n.node_id] = n.clone()
        for e in self.edges:
            g.edges.append(e.clone())
        return g

    @classmethod
    def default_pipeline(cls) -> ShaderGraph:
        g = cls()
        uv = g.add_node("input_uv")
        t = g.add_node("input_time")

        curl = g.add_node("warp_curl")
        vor = g.add_node("noise_voronoi")
        rd = g.add_node("reaction_diffusion")
        pal = g.add_node("palette_cosine")

        glow = g.add_node("post_glow")
        bloom = g.add_node("post_bloom")
        fb = g.add_node("post_feedback")
        sample = g.add_node("sample_prev")

        out = g.add_node("output_color")

        g.connect(uv.node_id, "uv", curl.node_id, "p", "vec2")
        g.connect(t.node_id, "t", curl.node_id, "t", "float")
        g.connect(curl.node_id, "wp", vor.node_id, "p", "vec2")
        g.connect(t.node_id, "t", vor.node_id, "t", "float")
        g.connect(vor.node_id, "d", pal.node_id, "hue", "float")
        g.connect(t.node_id, "t", pal.node_id, "t", "float")

        g.connect(pal.node_id, "col", glow.node_id, "col", "vec3")
        g.connect(uv.node_id, "uv", glow.node_id, "uv", "vec2")
        g.connect(glow.node_id, "col", bloom.node_id, "col", "vec3")
        g.connect(uv.node_id, "uv", bloom.node_id, "uv", "vec2")

        g.connect(bloom.node_id, "col", fb.node_id, "col", "vec3")
        g.connect(uv.node_id, "uv", fb.node_id, "uv", "vec2")

        g.connect(uv.node_id, "uv", sample.node_id, "uv", "vec2")
        g.connect(sample.node_id, "col", fb.node_id, "prev", "vec3")

        g.connect(fb.node_id, "col", out.node_id, "col", "vec3")

        g.connect(uv.node_id, "uv", rd.node_id, "uv", "vec2")
        g.connect(t.node_id, "t", rd.node_id, "t", "float")

        return g

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShaderGraph:
        g = cls()
        for nd in d.get("nodes", []):
            n = ShaderNode.from_dict(nd)
            g.nodes[n.node_id] = n
        for ed in d.get("edges", []):
            g.edges.append(ShaderEdge.from_dict(ed))
        return g

    def validate(self) -> bool:
        for e in self.edges:
            if e.src_id not in self.nodes or e.dst_id not in self.nodes:
                return False
            src = self.nodes.get(e.src_id)
            if src is not None:
                src_tpl = NODE_LIBRARY.get(src.ntype, {})
                if e.src_port not in src_tpl.get("outputs", {}):
                    return False
            dst = self.nodes.get(e.dst_id)
            if dst is not None:
                dst_tpl = NODE_LIBRARY.get(dst.ntype, {})
                if e.dst_port not in dst_tpl.get("inputs", {}):
                    return False
        return True

    def feedback_edges(self) -> list[ShaderEdge]:
        """Edges that form a feedback loop (dst or src is a feedback/post/sample node)."""
        fb_types = (
            "post_feedback",
            "post_glow",
            "post_bloom",
            "post_chromatic",
            "post_distortion",
            "sample_prev",
        )
        out: list[ShaderEdge] = []
        for e in self.edges:
            src = self.nodes.get(e.src_id)
            dst = self.nodes.get(e.dst_id)
            if src is None or dst is None:
                continue
            if src.ntype in fb_types or dst.ntype in fb_types:
                out.append(e)
        return out

    def reachable_from_output(self) -> set[str]:
        """Set of node ids reachable (following edges backward) from output_color."""
        out_nodes = [n for n in self.nodes.values() if n.ntype == "output_color"]
        seen: set[str] = set()
        stack = [n.node_id for n in out_nodes]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            for e in self.edges:
                if e.dst_id == nid and e.src_id not in seen:
                    stack.append(e.src_id)
        return seen

    def has_unapproved_cycle(self) -> bool:
        """True if there is a cycle that is NOT an explicitly allowed feedback loop.

        Feedback edges (those touching post_feedback / post_* / sample_prev) may
        legitimately close a loop, so we check for cycles only on the subgraph
        with those edges removed.
        """
        fb = {id(e) for e in self.feedback_edges()}
        WHITE, GREY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in self.nodes}
        # Build adjacency over non-feedback edges.
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            if id(e) in fb:
                continue
            if e.src_id in adj and e.dst_id in adj:
                adj[e.src_id].append(e.dst_id)

        def dfs(nid: str) -> bool:
            color[nid] = GREY
            for nxt in adj[nid]:
                if color[nxt] == GREY:
                    return True
                if color[nxt] == WHITE and dfs(nxt):
                    return True
            color[nid] = BLACK
            return False

        for nid in list(self.nodes.keys()):
            if color[nid] == WHITE and dfs(nid):
                return True
        return False

    def max_depth(self) -> int:
        """Longest path (by edges) from any input node to output_color."""
        parents: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            if e.src_id in parents and e.dst_id in parents:
                parents[e.dst_id].append(e.src_id)
        memo: dict[str, int] = {}

        def depth(nid: str) -> int:
            if nid in memo:
                return memo[nid]
            ps = parents[nid]
            memo[nid] = 0
            if not ps:
                return 0
            memo[nid] = 1 + max(depth(p) for p in ps)
            return memo[nid]

        out_nodes = [n for n in self.nodes.values() if n.ntype == "output_color"]
        if not out_nodes:
            return 0
        return max(depth(n.node_id) for n in out_nodes)

    def prune_dead_nodes(self) -> int:
        """Remove nodes not reachable from output_color. Returns count removed."""
        if not any(n.ntype == "output_color" for n in self.nodes.values()):
            return 0
        keep = self.reachable_from_output()
        removable = [nid for nid in list(self.nodes.keys()) if nid not in keep]
        for nid in removable:
            self.remove_node(nid)
        return len(removable)

    def required_inputs_satisfied(self) -> bool:
        """Every non-input, non-output node must have at least one incoming edge
        feeding a declared input port, and the output_color must be driven."""
        for n in self.nodes.values():
            tpl = NODE_LIBRARY.get(n.ntype, {})
            in_ports = list(tpl.get("inputs", {}).keys())
            if not in_ports:
                continue  # pure input or output with no required inputs
            connected = {e.dst_port for e in self.edges if e.dst_id == n.node_id}
            # Allow output_color to be empty (compiler substitutes black).
            if n.ntype == "output_color":
                continue
            if not connected:
                return False
        return True

    def validate_for_compile(self, max_nodes: int = 24, max_depth: int = 12) -> bool:
        """Compile-safety validation: every node reachable from output, no
        unapproved cycles, required inputs connected, size within limits."""
        if not self.validate():
            return False
        if "output_color" not in [n.ntype for n in self.nodes.values()]:
            return False
        if len(self.nodes) > max_nodes:
            return False
        if self.max_depth() > max_depth:
            return False
        if self.has_unapproved_cycle():
            return False
        if self.prune_dead_nodes() > 0:
            # Pruning changed the graph; re-check size/cycle after cleanup.
            if len(self.nodes) > max_nodes or self.max_depth() > max_depth:
                return False
            if self.has_unapproved_cycle():
                return False
        return self.required_inputs_satisfied()

    def topological_order(self) -> list[ShaderNode]:
        visited: set[str] = set()
        order: list[ShaderNode] = []

        def visit(nid: str, stack: set[str]) -> None:
            if nid in visited or nid not in self.nodes:
                return
            if nid in stack:
                return
            stack.add(nid)
            for e in self.edges:
                if e.dst_id == nid:
                    visit(e.src_id, stack)
            stack.remove(nid)
            visited.add(nid)
            order.append(self.nodes[nid])

        for nid in list(self.nodes.keys()):
            visit(nid, set())
        return order

    def random_graph(self, max_nodes: int = 6) -> None:
        self.nodes.clear()
        self.edges.clear()
        available = [
            "input_uv",
            "input_time",
            "input_audio_bass",
            "input_audio_mid",
            "input_audio_treble",
            "noise_fbm",
            "noise_voronoi",
            "noise_rd",
            "noise_mandelbrot",
            "warp_curl",
            "warp_vortex",
            "warp_twist",
            "curl_noise",
            "domain_warp",
            "palette_cosine",
            "palette_spectral",
            "palette_plasma",
            "voronoi_cells",
            "reaction_diffusion",
            "voronoi_edge",
            "sdf_field",
            "math_sin",
            "math_cos",
            "math_mix",
            "math_smoothstep",
            "math_mul",
            "math_add",
            "math_clamp",
            "math_fract",
            "math_mod",
            "math_abs",
            "math_pow",
            "math_length",
            "math_dot",
            "math_normalize",
            "math_step",
            "sample_prev",
            "sample_blur",
            "post_glow",
            "post_bloom",
            "post_chromatic",
            "post_feedback",
            "post_polar_galaxy",
            "post_cells",
            "post_nonlinear_trails",
            "post_audio_topology",
            "post_depth_cloud",
            "post_neon_edge",
            "feedback_drift",
            "feedback_kaleido",
            "feedback_mirror",
            "feedback_spiral",
            "feedback_zoom",
            "lighting_point",
            "lighting_ambient",
            "lighting_bloom",
            "particle_field",
            "particle_trails",
            "volume_fog",
            "volume_cloud",
            "camera_lens",
            "camera_dof",
            "composite_add",
            "composite_mul",
            "composite_screen",
            "composite_overlay",
            "output_color",
            "output_feedback",
            "output_post",
            "output_bloom",
            "output_composite",
        ]
        n_nodes = random.randint(3, max(3, max_nodes))
        inputs = [
            "input_uv",
            "input_time",
            "input_audio_bass",
            "input_audio_mid",
            "input_audio_treble",
        ]
        self.add_node(random.choice(inputs))
        for _ in range(n_nodes - 1):
            prev = random.choice(list(self.nodes.values()))
            candidates = [
                k
                for k in available
                if k not in PROTECTED_NODES
                and k != prev.ntype
                and k not in [n.ntype for n in self.nodes.values()]
            ]
            if not candidates:
                continue
            new_type = random.choice(candidates)
            new_node = self.add_node(new_type)
            src_ports = list(NODE_LIBRARY.get(prev.ntype, {}).get("outputs", {}).keys())
            dst_ports = list(NODE_LIBRARY.get(new_type, {}).get("inputs", {}).keys())
            if src_ports and dst_ports:
                sp = random.choice(src_ports)
                dp = random.choice(dst_ports)
                actual_type = NODE_LIBRARY.get(prev.ntype, {}).get("outputs", {}).get(sp, "float")
                self.connect(prev.node_id, sp, new_node.node_id, dp, actual_type)
        # Ensure output_color is present
        if "output_color" not in [n.ntype for n in self.nodes.values()]:
            sink = self.add_node("output_color")
            for prev in list(self.nodes.values()):
                if prev.ntype == "output_color":
                    continue
                dst_ports = list(NODE_LIBRARY.get("output_color", {}).get("inputs", {}).keys())
                src_ports = list(NODE_LIBRARY.get(prev.ntype, {}).get("outputs", {}).keys())
                if dst_ports and src_ports:
                    sp = random.choice(src_ports)
                    dp = dst_ports[0]
                    dtype = NODE_LIBRARY.get("output_color", {}).get("inputs", {}).get(dp, "vec3")
                    self.connect(prev.node_id, sp, sink.node_id, dp, dtype)
                    break


# ============================================================================
# Genome
# ============================================================================

PROTECTED_NODES = (
    "input_uv",
    "input_time",
    "input_audio_bass",
    "input_audio_mid",
    "input_audio_treble",
    "output_color",
    "output_feedback",
    "output_post",
    "output_bloom",
    "output_composite",
)

# Replacable operator families: nodes that occupy a similar "slot" so one can
# be swapped for a related one without breaking port shapes.
OPERATOR_FAMILIES = {
    "field": [
        "noise_fbm",
        "noise_voronoi",
        "noise_rd",
        "noise_mandelbrot",
        "voronoi_cells",
        "reaction_diffusion",
        "voronoi_edge",
        "sdf_field",
    ],
    "warp": ["warp_curl", "warp_vortex", "warp_twist", "curl_noise", "domain_warp"],
    "palette": ["palette_cosine", "palette_spectral", "palette_plasma"],
    "post": [
        "post_glow",
        "post_bloom",
        "post_chromatic",
        "post_feedback",
        "post_distortion",
        "glow_node",
        "bloom_node",
        "post_polar_galaxy",
        "post_cells",
        "post_nonlinear_trails",
        "post_audio_topology",
        "post_depth_cloud",
        "post_neon_edge",
    ],
    "feedback": [
        "feedback_drift",
        "feedback_kaleido",
        "feedback_mirror",
        "feedback_spiral",
        "feedback_zoom",
    ],
    "math": [
        "math_sin",
        "math_cos",
        "math_mix",
        "math_smoothstep",
        "math_mul",
        "math_add",
        "math_clamp",
        "math_fract",
        "math_mod",
        "math_abs",
        "math_pow",
        "math_sqrt",
        "math_length",
        "math_dot",
        "math_normalize",
        "math_step",
    ],
    "lighting": ["lighting_point", "lighting_ambient", "lighting_bloom"],
    "particle": ["particle_field", "particle_trails"],
    "volume": ["volume_fog", "volume_cloud"],
    "camera": ["camera_lens", "camera_dof"],
    "composite": ["composite_add", "composite_mul", "composite_screen", "composite_overlay"],
    "sample": ["sample_prev", "sample_blur"],
    "output": [
        "output_color",
        "output_feedback",
        "output_post",
        "output_bloom",
        "output_composite",
    ],
}

MAX_GRAPH_NODES = 32
MAX_GRAPH_DEPTH = 16

# Edge weight gene — every connection can carry blend/weight params
DEFAULT_EDGE_WEIGHT = 1.0

# Weighted-for-random-selection mutation kind table
# Structural mutations dominate because graph topology evolution
# produces genuinely new rendering techniques, not just parameter tweaks.
_MUTATION_KINDS = [
    ("insert_node", 20),
    ("rewire_edge", 15),
    ("remove_node", 12),
    ("duplicate_subgraph", 10),
    ("change_operator", 10),
    ("parallel_branch", 10),
    ("mutate_edge_weight", 8),
    ("insert_pipeline_node", 8),
    ("swap_pipeline_segment", 5),
    ("split_pipeline", 4),
    ("merge_pipeline", 4),
    ("create_feedback", 5),
    ("remove_feedback", 4),
    ("inject_macro", 5),
    ("split_node", 3),
    ("merge_nodes", 2),
    ("change_input", 3),
    ("evolve_field", 3),
    # Parameter mutation is last because tweaking constants produces the
    # least interesting variation compared to structural changes.
    ("parameter", 15),
]


def _weighted_mutation_kind() -> str:
    total = sum(w for _, w in _MUTATION_KINDS)
    r = random.uniform(0, total)
    upto = 0.0
    for kind, w in _MUTATION_KINDS:
        upto += w
        if r <= upto:
            return kind
    return "parameter"


class SubgraphLibrary:
    """Stores high-performing subgraphs (building blocks) discovered during
    evolution so they can be reused as macros instead of rediscovered."""

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.blocks: list[dict[str, Any]] = []

    def add(self, graph: ShaderGraph, fitness: float, generation: int) -> None:
        sig = _graph_signature(graph)
        if any(b["sig"] == sig for b in self.blocks):
            return
        self.blocks.append(
            {
                "sig": sig,
                "graph": graph.clone(),
                "fitness": float(fitness),
                "generation": generation,
            }
        )
        self.blocks.sort(key=lambda b: b["fitness"], reverse=True)
        if len(self.blocks) > self.capacity:
            self.blocks = self.blocks[: self.capacity]

    def best(self) -> ShaderGraph | None:
        return self.blocks[0]["graph"].clone() if self.blocks else None

    def random(self) -> ShaderGraph | None:
        return random.choice(self.blocks)["graph"].clone() if self.blocks else None

    def __len__(self) -> int:
        return len(self.blocks)


def _graph_signature(graph: ShaderGraph) -> str:
    parts = []
    for n in sorted(graph.nodes.values(), key=lambda x: x.node_id):
        parts.append(f"{n.ntype}({n.node_id})")
    for e in graph.edges:
        parts.append(f"{e.src_id}.{e.src_port}->{e.dst_id}.{e.dst_port}")
    return "|".join(parts)


def _graph_shader_signature(graph: ShaderGraph) -> dict[str, Any]:
    """Return the graph state that can affect generated GLSL."""
    nodes = [
        {
            "ntype": node.ntype,
            "node_id": node.node_id,
            "params": copy.deepcopy(node.params),
        }
        for node in sorted(graph.nodes.values(), key=lambda item: item.node_id)
    ]
    edges = [
        {
            "src_id": edge.src_id,
            "src_port": edge.src_port,
            "dst_id": edge.dst_id,
            "dst_port": edge.dst_port,
            "dtype": edge.dtype,
            "weight": float(edge.weight),
        }
        for edge in sorted(
            graph.edges,
            key=lambda item: (
                item.src_id,
                item.src_port,
                item.dst_id,
                item.dst_port,
            ),
        )
    ]
    return {"nodes": nodes, "edges": edges}


class ShaderGraphGenome:
    """Evolutionary wrapper around a ShaderGraph.

    Performs *structural* graph evolution: it can insert/remove nodes, rewire
    edges, duplicate whole subgraphs, swap operators, change inputs, merge or
    split nodes, and create/remove feedback loops -- in addition to the ordinary
    numeric parameter mutation. All mutations preserve a compilable, reachable,
    acyclic (modulo allowed feedback) graph.
    """

    def __init__(self, graph: ShaderGraph | None = None, library: SubgraphLibrary | None = None):
        self.graph = graph or ShaderGraph.default_pipeline()
        self.lineage = str(uuid.uuid4())[:8]
        self.generation = 0
        self.parent_id: str | None = None
        from evo_bio.pareto import Objectives

        self.objectives = Objectives()
        self.mutations: list[str] = []
        self._strength = 0.3
        self._recent_accepts: list[int] = []
        self.library = library if library is not None else SubgraphLibrary()
        self.last_event: dict[str, Any] = {}
        self.age = 0
        self.lifespan = 500
        self._usage_tracker: dict[str, int] = {}
        self._prune_every: int = 8
        self._generations_since_prune: int = 0

    @property
    def objectives(self):
        from evo_bio.pareto import Objectives

        obj = getattr(self, "_objectives", None)
        if obj is None:
            self._objectives = Objectives().to_dict()
            obj = self._objectives
        if isinstance(obj, dict):
            return Objectives.from_dict(obj)
        return obj

    @objectives.setter
    def objectives(self, value):
        from evo_bio.pareto import Objectives

        if isinstance(value, Objectives):
            self._objectives = value.to_dict()
        elif isinstance(value, dict):
            self._objectives = dict(value)
        else:
            self._objectives = Objectives().to_dict()

    @property
    def fitness(self) -> float:
        return self.objectives.scalar()

    @fitness.setter
    def fitness(self, value: float) -> None:
        from evo_bio.pareto import Objectives

        self._objectives = Objectives(
            beauty=float(value) * 0.4,
            novelty=float(value) * 0.15,
            audio_sync=float(value) * 0.1,
            performance=1.0,
            species_health=float(value) * 0.1,
            motion=float(value) * 0.1,
            complexity=float(value) * 0.1,
            memory=float(value) * 0.05,
        ).to_dict()

    # -- public API ---------------------------------------------------------
    def mutate(self, strength: float | None = None, large: bool = False) -> dict[str, Any]:
        """Apply one weighted mutation. Returns a structured event dict.

        Structural mutations are wrapped in a compile-safety net: if a mutation
        would leave the graph unreachable, cyclic (modulo allowed feedback), or
        otherwise uncompilable, the change is rolled back and the genome falls
        back to a safe parameter tweak. This guarantees the genome never keeps a
        graph that cannot be turned into valid GLSL.
        """
        if strength is not None:
            self._strength = float(strength)
        if large:
            self._strength = min(self._strength * 2.0 + 0.1, 1.0)
        before_nodes = len(self.graph.nodes)
        before_edges = len(self.graph.edges)
        kind = _weighted_mutation_kind()
        fn = getattr(self, f"_mutate_{kind}", self._mutate_parameter)
        snapshot = None
        if kind != "parameter":
            snapshot = (self.graph.to_dict(), list(self.mutations))
        detail = fn() or ""
        self.graph.prune_dead_nodes()
        valid = self.graph.validate_for_compile(
            max_nodes=MAX_GRAPH_NODES, max_depth=MAX_GRAPH_DEPTH
        )
        if snapshot is not None and not valid:
            self.graph = ShaderGraph.from_dict(snapshot[0])
            self.mutations = snapshot[1]
            kind = "parameter"
            detail = self._mutate_parameter() or ""
            self.graph.prune_dead_nodes()
            valid = self.graph.validate_for_compile(
                max_nodes=MAX_GRAPH_NODES, max_depth=MAX_GRAPH_DEPTH
            )
        self.track_usage()
        self._maybe_prune()
        after_nodes = len(self.graph.nodes)
        after_edges = len(self.graph.edges)
        event = {
            "generation": self.generation,
            "mutation": kind,
            "detail": detail,
            "nodes": (before_nodes, after_nodes),
            "edges": (before_edges, after_edges),
            "valid": valid,
            "fitness": self.objectives.scalar() if hasattr(self, "objectives") else 0.0,
            "novelty": 0.0,
            "metrics": self.compute_shader_metrics(),
        }
        self.last_event = event
        return event

    # -- structural mutations ----------------------------------------------
    def _mutate_insert_node(self) -> str:
        candidates = [
            k
            for k in NODE_LIBRARY
            if k not in PROTECTED_NODES and k not in [n.ntype for n in self.graph.nodes.values()]
        ]
        if not candidates:
            candidates = [k for k in NODE_LIBRARY if k not in PROTECTED_NODES]
        if not candidates:
            return self._mutate_parameter()
        ntype = random.choice(candidates)
        node = self.graph.add_node(ntype)
        my_outs = list(NODE_LIBRARY.get(ntype, {}).get("outputs", {}).keys())
        my_ins = list(NODE_LIBRARY.get(ntype, {}).get("inputs", {}).keys())
        # Insert "between" a random non-output node and its downstream consumer:
        # take a random upstream node and connect it -> new node -> (one of its
        # downstream consumers).
        upstream = [
            n
            for n in self.graph.nodes.values()
            if n.node_id != node.node_id and n.ntype not in ("output_color",)
        ]
        if my_ins and upstream:
            up = random.choice(upstream)
            up_outs = list(NODE_LIBRARY.get(up.ntype, {}).get("outputs", {}).keys())
            if up_outs and my_ins:
                sp = random.choice(up_outs)
                dp = random.choice(my_ins)
                actual_type = NODE_LIBRARY.get(up.ntype, {}).get("outputs", {}).get(sp, "float")
                self.graph.connect(up.node_id, sp, node.node_id, dp, actual_type)
        if my_outs:
            # Connect the new node's output into a random consumer if compatible.
            consumers = [
                n
                for n in self.graph.nodes.values()
                if n.node_id != node.node_id
                and n.ntype not in ("input_uv", "input_time", "input_audio_bass")
            ]
            random.shuffle(consumers)
            for c in consumers:
                c_ins = list(NODE_LIBRARY.get(c.ntype, {}).get("inputs", {}).keys())
                if c_ins and my_outs:
                    sp = random.choice(my_outs)
                    dp = random.choice(c_ins)
                    actual_type = NODE_LIBRARY.get(ntype, {}).get("outputs", {}).get(sp, "float")
                    self.graph.connect(node.node_id, sp, c.node_id, dp, actual_type)
                    between = c.ntype
                    break
            else:
                between = "??"
        else:
            between = "??"
        from_name = upstream[0].ntype if upstream else "??"
        self.mutations.append(f"node_added:{ntype}")
        return f"Inserted {ntype} between {from_name} -> {between}"

    def _mutate_remove_node(self) -> str:
        candidates = [n for n in self.graph.nodes.values() if n.ntype not in PROTECTED_NODES]
        removable = [
            n for n in candidates if len(self.graph.nodes) > 3 or n.ntype not in ("output_color",)
        ]
        if not removable:
            return self._mutate_parameter()
        target = random.choice(removable)
        self.graph.remove_node(target.node_id)
        self.mutations.append(f"node_removed:{target.ntype}")
        return f"Removed {target.ntype}"

    def _mutate_rewire_edge(self) -> str:
        nodes = [
            n
            for n in self.graph.nodes.values()
            if n.ntype not in ("input_uv", "input_time", "input_audio_bass")
        ]
        src = random.choice(nodes) if nodes else None
        dst = random.choice(nodes) if nodes else None
        if src is None or dst is None:
            return self._mutate_parameter()
        s_outs = list(NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).keys())
        d_ins = list(NODE_LIBRARY.get(dst.ntype, {}).get("inputs", {}).keys())
        if not s_outs or not d_ins:
            return self._mutate_parameter()
        sp = random.choice(s_outs)
        dp = random.choice(d_ins)
        dtype = NODE_LIBRARY.get(dst.ntype, {}).get("inputs", {}).get(dp, "float")
        self.graph.connect(src.node_id, sp, dst.node_id, dp, dtype)
        self.mutations.append("edge_rewired")
        return f"Rewired {src.ntype}.{sp} -> {dst.ntype}.{dp}"

    def _mutate_duplicate_subgraph(self) -> str:
        """Clone a node and the subtree feeding into it (bounded)."""
        roots = [n for n in self.graph.nodes.values() if n.ntype not in PROTECTED_NODES]
        if not roots:
            return self._mutate_parameter()
        root = random.choice(roots)
        # Collect the feed subtree (bounded BFS). Preference for reachable set.
        feed = self._feed_subtree(root.node_id, max_size=6)
        id_map = {}
        for nid in feed:
            old = self.graph.nodes.get(nid)
            if old is None:
                continue
            dup = old.clone()
            dup.node_id = str(uuid.uuid4())[:8]
            self.graph.nodes[dup.node_id] = dup
            id_map[nid] = dup.node_id
        for e in list(self.graph.edges):
            if e.src_id in id_map and e.dst_id in id_map:
                self.graph.connect(
                    id_map[e.src_id], e.src_port, id_map[e.dst_id], e.dst_port, e.dtype
                )
        count = len(id_map)
        self.mutations.append("subgraph_duplicated")
        return f"Duplicated subtree under {root.ntype} ({count} nodes)"

    def _mutate_change_operator(self) -> str:
        """Swap a node for a related operator keeping the same port shapes."""
        swappable = [n for n in self.graph.nodes.values() if n.ntype not in PROTECTED_NODES]
        if not swappable:
            return self._mutate_parameter()
        node = random.choice(swappable)
        fam = None
        for f, members in OPERATOR_FAMILIES.items():
            if node.ntype in members:
                fam = f
                break
        if fam is None:
            return self._mutate_parameter()
        choices = [m for m in OPERATOR_FAMILIES[fam] if m != node.ntype]
        if not choices:
            return self._mutate_parameter()
        new_type = random.choice(choices)
        node.ntype = new_type
        tpl = NODE_LIBRARY.get(new_type, {})
        node.params = {}
        for k, (_, default) in tpl.get("params", {}).items():
            node.params.setdefault(k, default)
        self.mutations.append(f"node_replaced:{node.ntype}")
        return f"Replaced {node.ntype} with {new_type}"

    def _mutate_change_input(self) -> str:
        """Re-point one input port of a node to a different compatible source."""
        targets = [
            n
            for n in self.graph.nodes.values()
            if n.ntype not in ("input_uv", "input_time", "input_audio_bass", "output_color")
        ]
        if not targets:
            return self._mutate_parameter()
        node = random.choice(targets)
        in_ports = list(NODE_LIBRARY.get(node.ntype, {}).get("inputs", {}).keys())
        if not in_ports:
            return self._mutate_parameter()
        dp = random.choice(in_ports)
        # Candidate sources: any node with a compatible output type.
        want_type = NODE_LIBRARY.get(node.ntype, {}).get("inputs", {}).get(dp, "float")
        sources = []
        for n in self.graph.nodes.values():
            outs = NODE_LIBRARY.get(n.ntype, {}).get("outputs", {})
            if any(v == want_type for v in outs.values()):
                sources.append(n)
        if not sources:
            return self._mutate_parameter()
        src = random.choice(sources)
        sp = random.choice(list(NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).keys()))
        actual_type = NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).get(sp, want_type)
        # Remove existing edges into this port, then add the new one.
        self.graph.edges = [
            e for e in self.graph.edges if not (e.dst_id == node.node_id and e.dst_port == dp)
        ]
        self.graph.connect(src.node_id, sp, node.node_id, dp, actual_type)
        self.mutations.append("input_changed")
        return f"Repointed {node.ntype}.{dp} <- {src.ntype}.{sp}"

    def _mutate_merge_nodes(self) -> str:
        nodes = [n for n in self.graph.nodes.values() if n.ntype not in PROTECTED_NODES]
        if len(nodes) < 2:
            return self._mutate_parameter()
        a, b = random.sample(nodes, 2)
        self.graph.disconnect(b.node_id, a.node_id)
        self.graph.disconnect(a.node_id, b.node_id)
        for e in list(self.graph.edges):
            if e.dst_id == b.node_id:
                self.graph.connect(e.src_id, e.src_port, a.node_id, e.dst_port, e.dtype)
        self.graph.remove_node(b.node_id)
        self.mutations.append(f"branch_merged:{a.ntype}+{b.ntype}")
        return f"Merged {b.ntype} into {a.ntype}"

    def _mutate_split_node(self) -> str:
        """Clone a node to fan a single source out to multiple consumers."""
        candidates = [
            n
            for n in self.graph.nodes.values()
            if n.ntype not in PROTECTED_NODES
            and any(e.src_id == n.node_id for e in self.graph.edges)
        ]
        if not candidates:
            return self._mutate_parameter()
        src = random.choice(candidates)
        dup = src.clone()
        dup.node_id = str(uuid.uuid4())[:8]
        self.graph.nodes[dup.node_id] = dup
        # Mirror the original's input wiring so the clone is a functional
        # duplicate (otherwise it has unconnected required inputs).
        for e in self.graph.edges:
            if e.dst_id == src.node_id:
                self.graph.connect(e.src_id, e.src_port, dup.node_id, e.dst_port, e.dtype)
        consumers = [e for e in self.graph.edges if e.src_id == src.node_id]
        moved = 0
        for e in consumers:
            if random.random() < 0.5:
                self.graph.connect(dup.node_id, e.src_port, e.dst_id, e.dst_port, e.dtype)
                moved += 1
        self.mutations.append("node_split")
        return f"Split {src.ntype} ({moved} edges fanned out)"

    def _mutate_create_feedback(self) -> str:
        """Close a loop through a feedback-capable node (post_feedback/sample)."""
        fb_types = (
            "post_feedback",
            "post_glow",
            "post_bloom",
            "post_chromatic",
            "post_distortion",
            "sample_prev",
        )
        has_fb = any(n.ntype in fb_types for n in self.graph.nodes.values())
        if not has_fb:
            fb = self.graph.add_node("post_feedback")
            outs = list(NODE_LIBRARY.get("post_feedback", {}).get("outputs", {}).keys())
            ins = list(NODE_LIBRARY.get("post_feedback", {}).get("inputs", {}).keys())
            if outs and ins:
                srcs = [
                    n
                    for n in self.graph.nodes.values()
                    if n.ntype
                    not in ("input_uv", "input_time", "input_audio_bass", "output_color")
                    and NODE_LIBRARY.get(n.ntype, {}).get("outputs")
                ]
                if srcs:
                    src = random.choice(srcs)
                    sp = random.choice(
                        list(NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).keys())
                    )
                    actual_type = (
                        NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).get(sp, "vec3")
                    )
                    self.graph.connect(src.node_id, sp, fb.node_id, ins[0], actual_type)
            self.mutations.append("feedback_created")
            return "Created feedback loop via post_feedback"
        fb_node = next(n for n in self.graph.nodes.values() if n.ntype in fb_types)
        ins = list(NODE_LIBRARY.get(fb_node.ntype, {}).get("inputs", {}).keys())
        if len(ins) < 2:
            return self._mutate_parameter()
        sp = self.graph.add_node("sample_prev")
        upstream = [
            n
            for n in self.graph.nodes.values()
            if n.ntype not in ("input_uv", "input_time", "input_audio_bass", "output_color")
            and n.node_id != sp.node_id
        ]
        if upstream:
            src = random.choice(upstream)
            src_outs = list(NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).keys())
            if src_outs:
                self.graph.connect(src.node_id, src_outs[0], sp.node_id, "uv", "vec2")
        self.graph.connect(sp.node_id, "col", fb_node.node_id, ins[-1], "vec3")
        self.mutations.append("feedback_created")
        return "Wired feedback edge on existing feedback node"

    def _mutate_remove_feedback(self) -> str:
        fb_types = (
            "post_feedback",
            "post_glow",
            "post_bloom",
            "post_chromatic",
            "post_distortion",
            "sample_prev",
        )
        self_fb = [
            e
            for e in self.graph.edges
            if (
                self.graph.nodes.get(e.src_id) is not None
                and self.graph.nodes.get(e.dst_id) is not None
                and (
                    self.graph.nodes[e.src_id].ntype in fb_types
                    or self.graph.nodes[e.dst_id].ntype in fb_types
                )
            )
        ]
        self_loops = [e for e in self_fb if e.src_id == e.dst_id]
        if self_loops:
            self.graph.edges.remove(random.choice(self_loops))
            self.mutations.append("feedback_removed")
            return "Removed self feedback loop"
        if self_fb:
            self.graph.edges.remove(random.choice(self_fb))
            self.mutations.append("feedback_removed")
            return "Removed feedback edge"
        return self._mutate_parameter()

    def _pipeline_order(self) -> list[str]:
        """Return node ids in canonical pipeline category order, following
        edges from input_uv / input_time toward output_color."""
        cat_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
        order = self.graph.topological_order()
        ordered = [
            n.node_id
            for n in order
            if NODE_LIBRARY.get(n.ntype, {}).get("category")
            in (
                "field",
                "warp",
                "palette",
                "math",
                "sample",
                "lighting",
                "particle",
                "volume",
                "camera",
                "composite",
                "post",
            )
        ]
        ordered.sort(
            key=lambda nid: cat_rank.get(
                NODE_LIBRARY.get(self.graph.nodes[nid].ntype, {}).get("category"), 99
            )
        )
        return ordered

    def _insert_into_pipeline(self, ntype: str, after_cat: str) -> ShaderNode | None:
        """Insert a node into the canonical pipeline directly after the last
        node of category ``after_cat`` (or at the start if none). Returns the
        new node, or None if insertion is not feasible."""
        pipe = self._pipeline_order()
        anchor = None
        for nid in reversed(pipe):
            if NODE_LIBRARY.get(self.graph.nodes[nid].ntype, {}).get("category") == after_cat:
                anchor = nid
                break
        if anchor is None:
            # No node of the requested category yet; anchor to the nearest
            # available upstream category so insertion still succeeds.
            order = ["field", "warp", "palette", "post"]
            if after_cat in order:
                for cat in reversed(order[: order.index(after_cat) + 1]):
                    for nid in reversed(pipe):
                        if (
                            NODE_LIBRARY.get(self.graph.nodes[nid].ntype, {}).get("category")
                            == cat
                        ):
                            anchor = nid
                            break
                    if anchor is not None:
                        break
        node = self.graph.add_node(ntype)
        tpl = NODE_LIBRARY.get(ntype, {})
        # Wire inputs from a compatible upstream node (prefer the anchor, then
        # any upstream node whose output type matches; otherwise leave the port
        # unconnected and let the compiler substitute a default value).
        for dp, dt in tpl.get("inputs", {}).items():
            candidates = [
                nid
                for nid in pipe
                if nid != node.node_id
                and any(
                    v == dt
                    for v in NODE_LIBRARY.get(self.graph.nodes[nid].ntype, {})
                    .get("outputs", {})
                    .values()
                )
            ]
            if anchor is not None and anchor not in candidates:
                candidates = [anchor, *candidates]
            for cand in candidates:
                outs = NODE_LIBRARY.get(self.graph.nodes[cand].ntype, {}).get("outputs", {})
                if any(v == dt for v in outs.values()):
                    sp = next(k for k, v in outs.items() if v == dt)
                    self.graph.connect(cand, sp, node.node_id, dp, dt)
                    break
        # Wire outputs toward downstream consumers / output.
        # A node's output may only feed nodes that are strictly downstream of
        # the anchor (forward in the pipeline), never anything upstream, so we
        # never close a cycle by routing the new node's output back into an
        # earlier node's ``p``/``uv`` input.
        downstream = set()
        if anchor is not None:
            stack = [anchor]
            seen_d = set()
            while stack:
                cur = stack.pop()
                if cur in seen_d:
                    continue
                seen_d.add(cur)
                for e in self.graph.edges:
                    if e.src_id == cur and e.dst_id not in seen_d:
                        stack.append(e.dst_id)
            downstream = seen_d
        for sp, st in tpl.get("outputs", {}).items():
            consumers = [
                nid
                for nid in pipe
                if nid != node.node_id
                and (anchor is None or nid in downstream)
                and (
                    anchor is None
                    or NODE_LIBRARY.get(self.graph.nodes[nid].ntype, {}).get("category")
                    != after_cat
                )
                and any(
                    v == st
                    for v in NODE_LIBRARY.get(self.graph.nodes[nid].ntype, {})
                    .get("inputs", {})
                    .values()
                )
            ]
            if consumers:
                c = consumers[0]
                dp = next(
                    k
                    for k, v in NODE_LIBRARY.get(self.graph.nodes[c].ntype, {})
                    .get("inputs", {})
                    .items()
                    if v == st
                )
                self.graph.connect(node.node_id, sp, c, dp, st)
                break
            out_node = next(
                (n for n in self.graph.nodes.values() if n.ntype == "output_color"), None
            )
            if out_node is not None and st == "vec3":
                self.graph.connect(node.node_id, sp, out_node.node_id, "col", "vec3")
                break
        if not any(e.dst_id == node.node_id or e.src_id == node.node_id for e in self.graph.edges):
            self.graph.remove_node(node.node_id)
            return None
        return node

    def _mutate_insert_pipeline_node(self) -> str:
        """Insert a node at a random valid position in the canonical pipeline:
        input -> field -> warp -> palette -> post -> output."""
        slots = [
            ("field", "curl_noise", "field"),
            ("field", "voronoi_cells", "field"),
            ("field", "reaction_diffusion", "field"),
            ("warp", "domain_warp", "warp"),
            ("palette", "palette_spectral", "palette"),
            ("post", "glow_node", "post"),
            ("post", "bloom_node", "post"),
        ]
        after_cat, ntype, _ = random.choice(slots)
        node = self._insert_into_pipeline(ntype, after_cat)
        if node is None:
            return self._mutate_parameter()
        self.mutations.append(f"pipeline_insert:{ntype}")
        return f"Inserted {ntype} into pipeline after {after_cat}"

    def _mutate_swap_pipeline_segment(self) -> str:
        """Swap two adjacent nodes in the pipeline category order."""
        pipe = self._pipeline_order()
        if len(pipe) < 2:
            return self._mutate_parameter()
        i = random.randint(0, len(pipe) - 2)
        a, b = pipe[i], pipe[i + 1]
        na, nb = self.graph.nodes[a], self.graph.nodes[b]
        ea = [e for e in self.graph.edges if e.src_id == a or e.dst_id == a]
        eb = [e for e in self.graph.edges if e.src_id == b or e.dst_id == b]
        # Only swap when it keeps category order valid (the two nodes are
        # adjacent in the topological chain). Rewire so output flows b->a.
        # Collect external edges touching a and b.
        ext_a_in = [
            (e.src_id, e.src_port, e.dst_port, e.dtype)
            for e in ea
            if e.dst_id == a and e.src_id != b
        ]
        ext_a_out = [
            (e.dst_id, e.src_port, e.dst_port, e.dtype)
            for e in ea
            if e.src_id == a and e.dst_id != b
        ]
        ext_b_in = [
            (e.src_id, e.src_port, e.dst_port, e.dtype)
            for e in eb
            if e.dst_id == b and e.src_id != a
        ]
        ext_b_out = [
            (e.dst_id, e.src_port, e.dst_port, e.dtype)
            for e in eb
            if e.src_id == b and e.dst_id != a
        ]
        # Remove existing a/b edges.
        self.graph.edges = [
            e
            for e in self.graph.edges
            if e.src_id not in (a, b)
            or e.dst_id not in (a, b)
            or (e.src_id == a and e.dst_id == a)
            or (e.src_id == b and e.dst_id == b)
        ]
        # Reconnect: external -> b (now earlier), b -> a, a -> external.
        for src_id, sp, dp, dt in ext_a_in:
            self.graph.connect(src_id, sp, b, dp, dt)
        for src_id, sp, dp, dt in ext_b_in:
            self.graph.connect(src_id, sp, b, dp, dt)
        for dst_id, sp, dp, dt in ext_a_out:
            self.graph.connect(a, sp, dst_id, dp, dt)
        for dst_id, sp, dp, dt in ext_b_out:
            self.graph.connect(b, sp, dst_id, dp, dt)
        # Internal b -> a.
        b_outs = list(NODE_LIBRARY.get(nb.ntype, {}).get("outputs", {}).keys())
        a_ins = list(NODE_LIBRARY.get(na.ntype, {}).get("inputs", {}).keys())
        if b_outs and a_ins:
            self.graph.connect(
                b,
                b_outs[0],
                a,
                a_ins[0],
                NODE_LIBRARY.get(nb.ntype, {}).get("outputs", {}).get(b_outs[0], "float"),
            )
        self.mutations.append(f"pipeline_swap:{na.ntype}<>{nb.ntype}")
        return f"Swapped pipeline segment {na.ntype} <-> {nb.ntype}"

    def _mutate_split_pipeline(self) -> str:
        """Split a post-processing node into two (e.g., bloom into bloom + glow)."""
        posts = [
            n
            for n in self.graph.nodes.values()
            if NODE_LIBRARY.get(n.ntype, {}).get("category") == "post"
            and n.ntype not in PROTECTED_NODES
            and any(e.src_id == n.node_id for e in self.graph.edges)
        ]
        if not posts:
            return self._mutate_parameter()
        src = random.choice(posts)
        # Choose a sibling post type to append after src.
        fam = OPERATOR_FAMILIES.get("post", [])
        choices = [m for m in fam if m != src.ntype]
        if not choices:
            return self._mutate_parameter()
        new_type = random.choice(choices)
        # Mirror src inputs onto the new node.
        node = self.graph.add_node(new_type)
        for e in list(self.graph.edges):
            if e.dst_id == src.node_id:
                self.graph.connect(e.src_id, e.src_port, node.node_id, e.dst_port, e.dtype)
        # Wire src -> new node, then new node -> src's old downstream.
        src_outs = list(NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).keys())
        node_ins = list(NODE_LIBRARY.get(new_type, {}).get("inputs", {}).keys())
        node_outs = list(NODE_LIBRARY.get(new_type, {}).get("outputs", {}).keys())
        if src_outs and node_ins:
            self.graph.connect(
                src.node_id,
                src_outs[0],
                node.node_id,
                node_ins[0],
                NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).get(src_outs[0], "vec3"),
            )
        for e in list(self.graph.edges):
            if e.src_id == src.node_id and src_outs and node_outs:
                self.graph.connect(node.node_id, node_outs[0], e.dst_id, e.dst_port, e.dtype)
        self.mutations.append(f"pipeline_split:{src.ntype}+{new_type}")
        return f"Split {src.ntype} -> {src.ntype} + {new_type}"

    def _mutate_merge_pipeline(self) -> str:
        """Merge two adjacent post-processing nodes into one."""
        posts = [
            n
            for n in self.graph.nodes.values()
            if NODE_LIBRARY.get(n.ntype, {}).get("category") == "post"
            and n.ntype not in PROTECTED_NODES
        ]
        if len(posts) < 2:
            return self._mutate_parameter()
        # Pick two posts where one feeds the other.
        pairs = [
            (a, b)
            for a in posts
            for b in posts
            if a is not b
            and any(e.src_id == a.node_id and e.dst_id == b.node_id for e in self.graph.edges)
        ]
        if not pairs:
            return self._mutate_parameter()
        a, b = random.choice(pairs)
        # Re-point everything that consumed b to consume a instead.
        for e in list(self.graph.edges):
            if e.dst_id == b.node_id:
                self.graph.connect(e.src_id, e.src_port, a.node_id, e.dst_port, e.dtype)
        # Drop internal a->b edge and remove b.
        self.graph.edges = [
            e for e in self.graph.edges if not (e.src_id == a.node_id and e.dst_id == b.node_id)
        ]
        self.graph.remove_node(b.node_id)
        self.mutations.append(f"pipeline_merge:{a.ntype}+{b.ntype}")
        return f"Merged {b.ntype} into {a.ntype}"

    # -- NEW: Edge weight evolution -------------------------------------------
    def _mutate_mutate_edge_weight(self) -> str:
        """Mutate a random edge's weight gene (how strongly the source signal
        modulates the destination)."""
        if not self.graph.edges:
            return self._mutate_parameter()
        edge = random.choice(self.graph.edges)
        old = edge.weight
        delta = random.uniform(-self._strength, self._strength) * 0.5
        new_weight = max(-2.0, min(2.0, old + delta))
        edge.weight = new_weight
        self.mutations.append(f"edge_weight:{new_weight:+.3f}")
        return f"Edge weight {old:.3f} -> {new_weight:.3f}"

    # -- NEW: Parallel branch mutation -----------------------------------------
    def _mutate_parallel_branch(self) -> str:
        """Fan out a node's output to two different downstream paths, creating
        parallel processing branches.  Picks a node with at least one consumer,
        clones the consumer, and rewires half of the edges to the clone."""
        src_candidates = [
            n
            for n in self.graph.nodes.values()
            if n.ntype not in PROTECTED_NODES
            and any(e.src_id == n.node_id for e in self.graph.edges)
        ]
        if not src_candidates:
            return self._mutate_parameter()
        src = random.choice(src_candidates)
        consumers = [e for e in self.graph.edges if e.src_id == src.node_id]
        if len(consumers) < 2:
            return self._mutate_parameter()
        target_edge = random.choice(consumers)
        target = self.graph.nodes.get(target_edge.dst_id)
        if target is None or target.ntype in PROTECTED_NODES:
            return self._mutate_parameter()
        clone = target.clone()
        clone.node_id = str(uuid.uuid4())[:8]
        self.graph.nodes[clone.node_id] = clone
        for e in self.graph.edges:
            if e.dst_id == target.node_id and e.src_id != src.node_id:
                self.graph.connect(e.src_id, e.src_port, clone.node_id, e.dst_port, e.dtype)
        self.graph.connect(
            src.node_id,
            target_edge.src_port,
            clone.node_id,
            target_edge.dst_port,
            target_edge.dtype,
        )
        target_outs = [e for e in self.graph.edges if e.src_id == target.node_id]
        moved = 0
        for e in target_outs:
            if random.random() < 0.4:
                self.graph.connect(clone.node_id, e.src_port, e.dst_id, e.dst_port, e.dtype)
                moved += 1
        self.mutations.append(f"parallel_branch:{target.ntype}")
        return f"Parallel branch: cloned {target.ntype} -> {clone.node_id} ({moved} edges)"

    # -- NEW: Macro injection from SubgraphLibrary -----------------------------
    def _mutate_inject_macro(self) -> str:
        """Insert a high-performing subgraph from the library as a reusable
        building block.  Wraps self.inject_macro() with mutation logging."""
        if not self.library:
            return self._mutate_parameter()
        if self.inject_macro():
            return "Injected macro from SubgraphLibrary"
        return self._mutate_parameter()

    def _mutate_evolve_field(self) -> str:
        """Replace a field node with a different field type from the same family
        (fbm -> voronoi -> reaction_diffusion)."""
        fields = [
            n
            for n in self.graph.nodes.values()
            if NODE_LIBRARY.get(n.ntype, {}).get("category") == "field"
            and n.ntype not in PROTECTED_NODES
        ]
        if not fields:
            return self._mutate_parameter()
        node = random.choice(fields)
        fam = OPERATOR_FAMILIES.get("field", [])
        choices = [m for m in fam if m != node.ntype]
        if not choices:
            return self._mutate_parameter()
        new_type = random.choice(choices)
        node.ntype = new_type
        tpl = NODE_LIBRARY.get(new_type, {})
        node.params = {}
        for k, (_, default) in tpl.get("params", {}).items():
            node.params.setdefault(k, default)
        # Drop now-incompatible input edges (compiler will substitute defaults).
        ins = set(tpl.get("inputs", {}).keys())
        self.graph.edges = [
            e for e in self.graph.edges if not (e.dst_id == node.node_id and e.dst_port not in ins)
        ]
        self.mutations.append(f"field_evolved:{node.ntype}")
        return f"Evolved field into {new_type}"

    def _mutate_duplicate_branch(self) -> str:
        """Clone an entire processing branch: pick a node, clone it and all its
        downstream consumers, creating a parallel processing path."""
        candidates = [
            n
            for n in self.graph.nodes.values()
            if n.ntype not in PROTECTED_NODES
            and n.ntype
            not in (
                "output_color",
                "output_feedback",
                "output_post",
                "output_bloom",
                "output_composite",
            )
        ]
        if not candidates:
            return self._mutate_parameter()
        src = random.choice(candidates)
        # BFS downstream subtree
        downstream: list[str] = []
        seen: set[str] = set()
        stack = [src.node_id]
        while stack and len(downstream) < 8:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            downstream.append(nid)
            for e in self.graph.edges:
                if e.src_id == nid and e.dst_id not in seen:
                    stack.append(e.dst_id)
        # Clone all nodes in the branch
        id_map: dict[str, str] = {}
        for nid in downstream:
            old = self.graph.nodes.get(nid)
            if old is None:
                continue
            dup = old.clone()
            dup.node_id = str(uuid.uuid4())[:8]
            self.graph.nodes[dup.node_id] = dup
            id_map[nid] = dup.node_id
        # Replicate edges within the branch
        for e in self.graph.edges:
            if e.src_id in id_map and e.dst_id in id_map:
                self.graph.connect(
                    id_map[e.src_id], e.src_port, id_map[e.dst_id], e.dst_port, e.dtype
                )
        # Wire clone output to a new or existing output
        src_dup_id = id_map.get(src.node_id)
        if src_dup_id:
            src_dup = self.graph.nodes[src_dup_id]
            src_outs = list(NODE_LIBRARY.get(src_dup.ntype, {}).get("outputs", {}).keys())
            out_nodes = [
                n
                for n in self.graph.nodes.values()
                if n.ntype
                in (
                    "output_color",
                    "output_feedback",
                    "output_post",
                    "output_bloom",
                    "output_composite",
                )
            ]
            if src_outs and out_nodes:
                out = random.choice(out_nodes)
                out_ins = list(NODE_LIBRARY.get(out.ntype, {}).get("inputs", {}).keys())
                if out_ins:
                    sp = src_outs[0]
                    dp = random.choice(out_ins)
                    dtype = NODE_LIBRARY.get(src_dup.ntype, {}).get("outputs", {}).get(sp, "float")
                    self.graph.connect(src_dup_id, sp, out.node_id, dp, dtype)
        count = len(id_map)
        self.mutations.append(f"branch_duplicated:{src.ntype}")
        return f"Duplicated branch under {src.ntype} ({count} nodes)"

    def _mutate_crossover_branches(self) -> str:
        """Swap two parallel branches between two output nodes to create new
        combinations of rendering techniques."""
        out_types = (
            "output_color",
            "output_feedback",
            "output_post",
            "output_bloom",
            "output_composite",
        )
        out_nodes = [n for n in self.graph.nodes.values() if n.ntype in out_types]
        if len(out_nodes) < 2:
            return self._mutate_parameter()
        # Pick two outputs and swap one upstream node feeding each
        a_out, b_out = random.sample(out_nodes, 2)
        a_ins = [e for e in self.graph.edges if e.dst_id == a_out.node_id]
        b_ins = [e for e in self.graph.edges if e.dst_id == b_out.node_id]
        if not a_ins or not b_ins:
            return self._mutate_parameter()
        a_edge = random.choice(a_ins)
        b_edge = random.choice(b_ins)
        # Swap the connections
        self.graph.edges = [
            e
            for e in self.graph.edges
            if not (e.dst_id == a_out.node_id and e.src_id == a_edge.src_id)
            and not (e.dst_id == b_out.node_id and e.src_id == b_edge.src_id)
        ]
        dtype_a = a_edge.dtype
        dtype_b = b_edge.dtype
        self.graph.connect(b_edge.src_id, b_edge.src_port, a_out.node_id, a_edge.dst_port, dtype_b)
        self.graph.connect(a_edge.src_id, a_edge.src_port, b_out.node_id, b_edge.dst_port, dtype_a)
        self.mutations.append("branches_crossovered")
        return f"Crossovered branches feeding {a_out.ntype} <-> {b_out.ntype}"

    def _mutate_parameter(self) -> str:
        parts: list[str] = []
        # Mutate node parameters
        for n in self.graph.nodes.values():
            changes = n.mutate(self._strength)
            for k, old, new_val in changes:
                if isinstance(new_val, tuple):
                    parts.append(f"{n.ntype}:{k}={new_val}")
                else:
                    parts.append(f"{n.ntype}:{k}={new_val:.3g}")
        # Also mutate edge weights as parameter drift
        for edge in self.graph.edges:
            if random.random() < self._strength * 0.3:
                old = edge.weight
                delta = random.uniform(-self._strength, self._strength) * 0.3
                new_weight = max(-2.0, min(2.0, old + delta))
                edge.weight = new_weight
                parts.append(f"edge_weight:{old:.3f}->{new_weight:.3f}")
        if parts:
            self.mutations.append(f"parameter:{';'.join(parts[:6])}")
            return f"Tuned {len(parts)} param(s): {';'.join(parts[:3])}"
        self.mutations.append("parameter")
        return "No parameter changed"

    # -- helpers ------------------------------------------------------------
    def _feed_subtree(self, root_id: str, max_size: int = 6) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        stack = [root_id]
        while stack and len(out) < max_size:
            nid = stack.pop()
            if nid in seen or nid not in self.graph.nodes:
                continue
            seen.add(nid)
            out.append(nid)
            for e in self.graph.edges:
                if e.dst_id == nid and e.src_id not in seen:
                    stack.append(e.src_id)
        return out

    def record_accept(self, accepted: bool) -> None:
        self._recent_accepts.append(1 if accepted else 0)
        if len(self._recent_accepts) > 8:
            self._recent_accepts.pop(0)
        if accepted:
            # Preserve useful innovations as reusable building blocks.
            self.library.add(self.graph, self.fitness, self.generation)

    def adapt_strength(self, lo: float, hi: float) -> None:
        if not self._recent_accepts:
            return
        rate = sum(self._recent_accepts) / len(self._recent_accepts)
        if rate < 0.2:
            self._strength = max(lo, self._strength * 0.7)
        elif rate > 0.6:
            self._strength = min(hi, self._strength * 1.3)

    def inject_macro(self) -> bool:
        """Insert a high-performing stored subgraph as a reusable building block.
        Returns True if a macro was inserted."""
        if not self.library:
            return False
        macro = self.library.random()
        if macro is None:
            return False
        # Re-key the macro's node ids and graft it in, then connect the macro's
        # output (if any) toward output_color if output is currently empty.
        id_map = {}
        for old in list(macro.nodes.values()):
            dup = old.clone()
            dup.node_id = str(uuid.uuid4())[:8]
            self.graph.nodes[dup.node_id] = dup
            id_map[old.node_id] = dup.node_id
        for e in macro.edges:
            if e.src_id in id_map and e.dst_id in id_map:
                self.graph.connect(
                    id_map[e.src_id], e.src_port, id_map[e.dst_id], e.dst_port, e.dtype
                )
        # Route macro outputs toward any output_color / feedback consumer.
        macro_outs = [
            nid for nid, oid in id_map.items() if any(e.src_id == nid for e in macro.edges)
        ]
        out_node = next((n for n in self.graph.nodes.values() if n.ntype == "output_color"), None)
        if out_node is not None and macro_outs:
            src_nid = id_map[macro_outs[0]]
            src = self.graph.nodes[src_nid]
            sp = list(NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).keys())
            if sp:
                actual_type = NODE_LIBRARY.get(src.ntype, {}).get("outputs", {}).get(sp[0], "vec3")
                self.graph.connect(src_nid, sp[0], out_node.node_id, "col", actual_type)
        self.mutations.append("macro_inserted")
        return True

    def novelty(self, basis: list[str] | None = None) -> float:
        """Simple structural novelty: fraction of node/edge signatures not in
        ``basis`` (a list of _graph_signature() strings). 0..1."""
        if not basis:
            return 0.0
        sig = _graph_signature(self.graph)
        if sig in basis:
            return 0.0
        return 1.0

    def distance(self, other: ShaderGraphGenome) -> float:
        """Normalized structural distance to another genome (0..1)."""

        def _node_counts(g):
            counts: dict[str, int] = {}
            for n in g.nodes.values():
                counts[n.ntype] = counts.get(n.ntype, 0) + 1
            return counts

        def _edge_counts(g):
            counts: dict[str, int] = {}
            for e in g.edges:
                src = g.nodes.get(e.src_id)
                dst = g.nodes.get(e.dst_id)
                if src is not None and dst is not None:
                    key = f"{src.ntype}->{dst.ntype}"
                    counts[key] = counts.get(key, 0) + 1
            return counts

        def _count_distance(a, b):
            keys = set(a) | set(b)
            if not keys:
                return 0.0
            total = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
            max_total = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
            return total / max(max_total, 1.0)

        d_nodes = _count_distance(_node_counts(self.graph), _node_counts(other.graph))
        d_edges = _count_distance(_edge_counts(self.graph), _edge_counts(other.graph))
        return (d_nodes + d_edges) / 2.0

    def crossover(self, other: ShaderGraphGenome) -> ShaderGraphGenome:
        child = ShaderGraphGenome(library=self.library)
        # Guarantee a minimal, connected, compilable scaffold first.
        uv = child.graph.add_node("input_uv")
        t = child.graph.add_node("input_time")
        fbm = child.graph.add_node("noise_fbm")
        pal = child.graph.add_node("palette_cosine")
        out = child.graph.add_node("output_color")
        child.graph.connect(uv.node_id, "uv", fbm.node_id, "p", "vec2")
        child.graph.connect(t.node_id, "t", fbm.node_id, "t", "float")
        child.graph.connect(fbm.node_id, "n", pal.node_id, "hue", "float")
        child.graph.connect(pal.node_id, "col", out.node_id, "col", "vec3")

        # Per-category independent crossover: for each shader function category,
        # pick one parent and graft ALL its nodes of that category. This allows
        # field(), lighting(), palette(), camera(), noise(), feedback(),
        # raymarch(), particles() to crossover independently instead of
        # whole-shader random subtree grafting.
        crossover_categories = {
            "field",
            "warp",
            "palette",
            "lighting",
            "particle",
            "camera",
            "feedback",
            "post",
            "volume",
            "composite",
            "math",
            "sample",
        }

        # Track cloned nodes: original_id -> cloned_id
        id_map: dict[str, str] = {}

        for cat in crossover_categories:
            donor = random.choice([self, other])
            for src in donor.graph.nodes.values():
                if NODE_LIBRARY.get(src.ntype, {}).get("category") != cat:
                    continue
                if src.node_id in id_map:
                    continue
                keeper = src.clone()
                old_id = keeper.node_id
                keeper.node_id = str(uuid.uuid4())[:8]
                child.graph.nodes[keeper.node_id] = keeper
                id_map[old_id] = keeper.node_id

        # Preserve edges between cloned nodes from the same donor.
        for donor in (self, other):
            for edge in donor.graph.edges:
                src_child = id_map.get(edge.src_id)
                dst_child = id_map.get(edge.dst_id)
                if src_child and dst_child:
                    child.graph.connect(
                        src_child,
                        edge.src_port,
                        dst_child,
                        edge.dst_port,
                        edge.dtype,
                    )

        # Wire unconnected inputs of cloned nodes from existing nodes.
        for node in child.graph.nodes.values():
            if node.node_id not in id_map.values():
                continue
            for dp in NODE_LIBRARY.get(node.ntype, {}).get("inputs", {}):
                already_connected = any(
                    e.dst_id == node.node_id and e.dst_port == dp for e in child.graph.edges
                )
                if already_connected:
                    continue
                cand = [
                    n
                    for n in child.graph.nodes.values()
                    if n.node_id != node.node_id and n.ntype not in ("output_color",)
                ]
                if cand:
                    c = random.choice(cand)
                    outs = list(NODE_LIBRARY.get(c.ntype, {}).get("outputs", {}).keys())
                    if outs:
                        sp = random.choice(outs)
                        actual_type = (
                            NODE_LIBRARY.get(c.ntype, {}).get("outputs", {}).get(sp, "float")
                        )
                        child.graph.connect(c.node_id, sp, node.node_id, dp, actual_type)

        child.graph.prune_dead_nodes()
        child.generation = max(self.generation, other.generation) + 1
        child.parent_id = self.lineage
        return child

    def clone(self) -> ShaderGraphGenome:
        g = ShaderGraphGenome(self.graph.clone(), library=self.library)
        g.lineage = self.lineage
        g.generation = self.generation
        g.parent_id = self.parent_id
        g.fitness = self.fitness
        g.mutations = list(self.mutations)
        g._strength = self._strength
        g._recent_accepts = list(self._recent_accepts)
        g.last_event = dict(self.last_event)
        g.age = getattr(self, "age", 0)
        g.lifespan = getattr(self, "lifespan", 500)
        g._usage_tracker = dict(self._usage_tracker)
        g._prune_every = getattr(self, "_prune_every", 8)
        g._generations_since_prune = getattr(self, "_generations_since_prune", 0)
        for node in g.graph.nodes.values():
            src = self.graph.nodes.get(node.node_id)
            if src is not None:
                node.usage_count = src.usage_count
                node.last_used_generation = src.last_used_generation
        return g

    def track_usage(self) -> None:
        """Mark every node currently reachable from output_color as used."""
        reachable = self.graph.reachable_from_output()
        for nid in reachable:
            node = self.graph.nodes.get(nid)
            if node is not None:
                node.usage_count += 1
                node.last_used_generation = self.generation
                self._usage_tracker[nid] = self._usage_tracker.get(nid, 0) + 1

    def compute_shader_metrics(self) -> dict[str, int]:
        """Return estimated instruction-level metrics for the current graph."""
        compiler = ShaderGraphCompiler()
        return compiler.compute_shader_metrics(self.graph)

    def fps_contribution(self) -> float:
        """Estimated FPS cost of the current graph (higher = slower)."""
        total_cost = 0.0
        for node in self.graph.nodes.values():
            cost = NODE_COST_METRICS.get(
                node.ntype,
                {
                    "instructions": 4,
                    "alu": 3,
                    "texture_reads": 0,
                    "branches": 0,
                    "fps_cost": 1,
                },
            )
            total_cost += float(cost.get("fps_cost", 1))
        return total_cost

    def prune_unused_nodes(self, min_usage: int = 2, max_age: int = 20) -> int:
        """Remove nodes that have not been used recently.

        A node is considered unused if its ``usage_count`` is below
        ``min_usage`` AND it has not been used for at least ``max_age``
        generations.  Protected nodes are never pruned.

        Returns the number of nodes removed.
        """
        if not any(n.ntype == "output_color" for n in self.graph.nodes.values()):
            return 0
        candidates = [
            n
            for n in self.graph.nodes.values()
            if n.ntype not in PROTECTED_NODES
            and n.usage_count < min_usage
            and (self.generation - n.last_used_generation) > max_age
        ]
        for node in candidates:
            self.graph.remove_node(node.node_id)
            self._usage_tracker.pop(node.node_id, None)
        if candidates:
            self.graph.prune_dead_nodes()
        return len(candidates)

    def _maybe_prune(self) -> None:
        """Periodically prune unused nodes based on internal scheduling."""
        self._generations_since_prune += 1
        if self._generations_since_prune >= self._prune_every:
            self._generations_since_prune = 0
            removed = self.prune_unused_nodes()
            if removed:
                self.mutations.append(f"self_pruned:{removed}")

    def environmental_affinity(self, audio: dict[str, float]) -> float:
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        score = 0.0
        for n in self.graph.nodes.values():
            if n.ntype in ("warp_curl", "curl_noise"):
                score += 0.05 * bass
            if n.ntype in ("glow_node", "bloom_node", "post_glow", "post_bloom"):
                score += 0.03 * treb
        return min(score, 1.0)

    def age_one_generation(self) -> None:
        self.age += 1
        if self.age > self.lifespan:
            try:
                self.mutate(large=True)
            except TypeError:
                self.mutate()
            self.age = 0
            return

        ratio = self.age / max(self.lifespan, 1)
        if ratio < 0.25:
            try:
                self.mutate(large=True)
            except TypeError:
                self.mutate()
        elif ratio < 0.75:
            try:
                self.mutate()
            except TypeError:
                pass
        else:
            try:
                self.mutate(large=True)
                self.mutate(large=True)
            except TypeError:
                try:
                    self.mutate()
                except Exception as e:
                    logger.warning("mutation failed: %s", e)

    def metamorphosis(self) -> None:
        self.graph = ShaderGraph.default_pipeline()
        self._usage_tracker = {}
        self._generations_since_prune = 0
        self.mutations.append("metamorphosis")

    def shader_signature(self) -> dict[str, Any]:
        return _graph_shader_signature(self.graph)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "_lineage": self.lineage,
            "_generation": self.generation,
            "_fitness": self.objectives.to_dict() if hasattr(self, "objectives") else {},
            "_parent_id": self.parent_id,
            "_mutations": list(self.mutations),
            "_library": [b["sig"] for b in self.library.blocks],
            "_usage_tracker": dict(self._usage_tracker),
            "_prune_every": getattr(self, "_prune_every", 8),
            "_generations_since_prune": getattr(self, "_generations_since_prune", 0),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShaderGraphGenome:
        g = cls(ShaderGraph.from_dict(d.get("graph", {})))
        g.lineage = d.get("_lineage", g.lineage or str(uuid.uuid4())[:8])
        g.generation = int(d.get("_generation", 0))
        obj_data = d.get("_fitness", {})
        if isinstance(obj_data, dict) and obj_data:
            from evo_bio.pareto import Objectives

            g.objectives = Objectives.from_dict(obj_data)
        elif isinstance(obj_data, (int, float)):
            g.fitness = float(obj_data)
        g.parent_id = d.get("_parent_id")
        g.mutations = list(d.get("_mutations", []))
        g._usage_tracker = {str(k): int(v) for k, v in d.get("_usage_tracker", {}).items()}
        g._prune_every = int(d.get("_prune_every", 8))
        g._generations_since_prune = int(d.get("_generations_since_prune", 0))
        return g


# ============================================================================
# Compiler
# ============================================================================


class ShaderGraphCompiler:
    """Converts a ShaderGraph into valid GLSL 330 fragment shader source."""

    def __init__(self):
        self._func_cache: dict[str, str] = {}

    def compile(
        self, graph: ShaderGraph, brightness: float = 1.0, feedback_alpha: float = 0.85
    ) -> str:
        self.graph = graph
        if not graph.validate():
            return self._fallback_shader(brightness, feedback_alpha)
        fb = max(0.0, min(0.999, float(feedback_alpha)))
        br = max(0.0, float(brightness))

        # Multi-pass: if there are multiple distinct output contexts, generate
        # one shader program per output node so the renderer can chain them.
        out_nodes = [
            n
            for n in graph.nodes.values()
            if NODE_LIBRARY.get(n.ntype, {}).get("category") == "output"
        ]
        if len(out_nodes) > 1:
            return self.compile_multi_pass(graph, out_nodes, brightness, feedback_alpha)
        return self._compile_single(graph, brightness, feedback_alpha)

    def _compile_single(
        self,
        graph: ShaderGraph,
        brightness: float = 1.0,
        feedback_alpha: float = 0.85,
        accumulate_outputs: bool = False,
    ) -> str:
        self.graph = graph
        fb = max(0.0, min(0.999, float(feedback_alpha)))
        br = max(0.0, float(brightness))
        uniforms: dict[str, str] = {}
        funcs: list[str] = []
        calls: list[str] = []
        decls: list[str] = []
        input_vars: dict[str, str] = {}

        self._emitted: dict[str, str] = {}

        pre_declarations: list[str] = []
        seen_pre: set = set()
        for node in graph.topological_order():
            tpl = NODE_LIBRARY.get(node.ntype, {})
            out_tpl = tpl.get("outputs", {})
            for port, ptype in out_tpl.items():
                var = f"{port}_{node.node_id}"
                if var in seen_pre:
                    continue
                seen_pre.add(var)
                default_val = self._default_for_type(ptype)
                pre_declarations.append(f"{ptype} {var} = {default_val};")
        if "outCol" not in seen_pre:
            pre_declarations.append("vec3 outCol = vec3(0.0);")

        for node in graph.topological_order():
            tpl = NODE_LIBRARY.get(node.ntype, {})
            cat = tpl.get("category", "")
            if cat == "input":
                self._gen_input(node, uniforms, decls, input_vars)
            elif cat in (
                "field",
                "warp",
                "palette",
                "post",
                "math",
                "sample",
                "feedback",
                "lighting",
                "particle",
                "volume",
                "camera",
                "composite",
            ):
                self._gen_op(node, tpl, uniforms, funcs, calls)
            elif cat == "output":
                self._gen_output(node, tpl, calls, accumulate=accumulate_outputs)

        uniform_decls = "\n".join(decls)
        func_block = "\n\n".join(funcs)
        call_block = "\n    ".join(c for c in calls if c)
        pre_decls = "\n    ".join(pre_declarations) if pre_declarations else ""

        outcol_guard = ""

        helper_block = GLSL_HELPERS

        return f"""#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform float bass;
uniform float mid;
uniform float treble;
uniform sampler2D previous_frame;

{uniform_decls}

{helper_block}

{func_block}

void main() {{
    vec2 uv = evo_safe_uv(v_uv);
    vec2 sample_uv = uv;
    float t = u_time;

    {pre_decls}

    {call_block}
    {outcol_guard}

    outCol = evo_safe_color(outCol * {br:.8g});
    vec3 prev_main = texture(previous_frame, uv).rgb;
    if (any(isnan(prev_main)) || any(isinf(prev_main))) prev_main = vec3(0.0);
    outCol = mix(outCol, prev_main, {fb:.8g});
    fragColor = vec4(outCol, 1.0);
}}
"""

    def compile_multi_pass(
        self,
        graph: ShaderGraph,
        out_nodes: list[ShaderNode],
        brightness: float = 1.0,
        feedback_alpha: float = 0.85,
    ) -> str:
        """Generate a single shader that handles multiple output contexts by
        blending all output nodes additively into ``outCol``.  True multi-pass
        (separate FBOs per output) is left to the caller (MultiPassPipeline).
        """
        return self._compile_single(
            graph, brightness, feedback_alpha, accumulate_outputs=True
        )

    def _gen_input(
        self,
        node: ShaderNode,
        uniforms: dict[str, str],
        decls: list[str],
        input_vars: dict[str, str],
    ) -> None:
        if node.ntype == "input_uv":
            uniforms["v_uv"] = "v_uv"
            input_vars[node.node_id] = "v_uv"
        elif node.ntype == "input_time":
            uniforms["u_time"] = "u_time"
            input_vars[node.node_id] = "u_time"
        elif node.ntype == "input_audio_bass":
            uniforms["bass"] = "bass"
            input_vars[node.node_id] = "bass"
        elif node.ntype == "input_audio_mid":
            uniforms["mid"] = "mid"
            input_vars[node.node_id] = "mid"
        elif node.ntype == "input_audio_treble":
            uniforms["treble"] = "treble"
            input_vars[node.node_id] = "treble"

    def _gen_op(
        self,
        node: ShaderNode,
        tpl: dict[str, Any],
        uniforms: dict[str, str],
        funcs: list[str],
        calls: list[str],
    ) -> None:
        out_tpl = tpl.get("outputs", {})
        out_vars = list(out_tpl.keys())
        primary_out = out_vars[0] if out_vars else "y"
        primary_type = out_tpl.get(primary_out, "float")

        in_vars: dict[str, str] = {}
        for port, port_type in tpl.get("inputs", {}).items():
            name, src_type, weight = self.graph_get_input_with_weight(node, port)
            if name is None or name == "":
                expr = self._resolve_default(port, port_type)
            else:
                expr = self._cast_var(name, src_type or port_type, port_type)
                if abs(weight - 1.0) > 0.001:
                    if port_type in ("vec2", "vec3", "vec4"):
                        expr = f"({weight} * {expr})"
                    else:
                        expr = f"({weight} * {expr})"
            in_vars[port] = expr

        func_name = f"sg_{node.ntype}_{node.node_id}"

        if node.ntype == "noise_fbm":
            funcs.append(f"""
float {func_name}(vec2 p, float t) {{
    float v = 0.0; float a = 0.5;
    for(int i=0;i<{int(node.params.get("layers", 4))};i++){{
        v += a * fract(sin(dot(floor(p), vec2(12.9898,78.233))) * 43758.5453);
        p = p * 2.01; a *= 0.5;
    }}
    return v;
}}""")
            calls.append(
                f"{primary_out}_{node.node_id} = {func_name}({in_vars.get('p', 'uv')}, {in_vars.get('t', 't')});"
            )
        elif node.ntype == "noise_voronoi":
            funcs.append(f"""
float {func_name}(vec2 p, float t) {{
    vec2 i = floor(p); vec2 f = fract(p);
    float d = 1.0;
    for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++){{
        vec2 nb = vec2(float(x),float(y));
        vec2 pt = vec2(fract(sin(dot(i+nb, vec2(12.9898,78.233))) * 43758.5453));
        d = min(d, length(nb + pt - f));
    }}
    return d;
}}""")
            calls.append(
                f"{primary_out}_{node.node_id} = {func_name}({in_vars.get('p', 'uv')}, {in_vars.get('t', 't')});"
            )
        elif node.ntype == "warp_curl":
            funcs.append(f"""
vec2 {func_name}(vec2 p, float t) {{
    float e = 0.01;
    float n1 = fract(sin(dot(floor(p+vec2(0,e)), vec2(12.9898,78.233))) * 43758.5453);
    float n2 = fract(sin(dot(floor(p-vec2(0,e)), vec2(12.9898,78.233))) * 43758.5453);
    float n3 = fract(sin(dot(floor(p+vec2(e,0)), vec2(12.9898,78.233))) * 43758.5453);
    float n4 = fract(sin(dot(floor(p-vec2(e,0)), vec2(12.9898,78.233))) * 43758.5453);
    return vec2(n1-n2, n4-n3) * {node.params.get("strength", 0.5)};
}}""")
            calls.append(
                f"{primary_out}_{node.node_id} = {func_name}({in_vars.get('p', 'uv')}, {in_vars.get('t', 't')});"
            )
        elif node.ntype == "noise_rd":
            calls.append(
                f"{primary_out}_{node.node_id} = reaction_diffusion({in_vars.get('uv', 'uv')}, {node.params.get('feed', 0.055)}, {node.params.get('kill', 0.062)}, {int(node.params.get('iterations', 8))});"
            )
        elif node.ntype == "noise_mandelbrot":
            calls.append(
                f"{primary_out}_{node.node_id} = mandelbrot({in_vars.get('p', 'uv')} * {node.params.get('zoom', 1.0)}, {int(node.params.get('max_iter', 64))});"
            )
        elif node.ntype == "warp_vortex":
            calls.append(
                f"{primary_out}_{node.node_id} = {func_name}({in_vars.get('p', 'uv')}, {in_vars.get('t', 't')});"
            )
            funcs.append(f"""
vec2 {func_name}(vec2 p, float t) {{
    float r = length(p);
    float a = atan(p.y, p.x) + {node.params.get("strength", 0.3)} * r + t * 0.1;
    float arms = max(1.0, {node.params.get("arms", 3.0)});
    a = mod(a, 6.28318530718 / arms);
    return vec2(cos(a), sin(a)) * r;
}}""")
        elif node.ntype == "warp_twist":
            calls.append(
                f"{primary_out}_{node.node_id} = {func_name}({in_vars.get('p', 'uv')}, {in_vars.get('t', 't')});"
            )
            funcs.append(f"""
vec2 {func_name}(vec2 p, float t) {{
    float a = {node.params.get("amount", 0.5)} * length(p) + t * 0.1;
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c) * p;
}}""")
        elif node.ntype == "palette_cosine":
            calls.append(
                f"{primary_out}_{node.node_id} = 0.5 + 0.5*cos(EVO_TWO_PI*({in_vars.get('hue', '0.5')} + vec3(0.0,0.33,0.67)));"
            )

        elif node.ntype == "palette_spectral":
            calls.append(
                f"{primary_out}_{node.node_id} = vec3({in_vars.get('hue', '0.5')}, 1.0-{in_vars.get('hue', '0.5')}, sin({in_vars.get('hue', '0.5')}*3.14));"
            )
        elif node.ntype == "palette_plasma":
            calls.append(
                f"{primary_out}_{node.node_id} = vec3(sin({in_vars.get('hue', '0.5')}*6.0), sin({in_vars.get('hue', '0.5')}*6.0+2.0), sin({in_vars.get('hue', '0.5')}*6.0+4.0)) * 0.5 + 0.5;"
            )
        elif node.ntype == "math_sin":
            calls.append(f"{primary_out}_{node.node_id} = sin({in_vars.get('x', '0.0')});")
        elif node.ntype == "math_cos":
            calls.append(f"{primary_out}_{node.node_id} = cos({in_vars.get('x', '0.0')});")
        elif node.ntype == "math_mix":
            calls.append(
                f"{primary_out}_{node.node_id} = mix({in_vars.get('a', 'vec3(0.0)')}, {in_vars.get('b', 'vec3(0.0)')}, {in_vars.get('t', '0.0')});"
            )
        elif node.ntype == "math_smoothstep":
            calls.append(
                f"{primary_out}_{node.node_id} = smoothstep({in_vars.get('edge0', '0.0')}, {in_vars.get('edge1', '1.0')}, {in_vars.get('x', '0.0')});"
            )
        elif node.ntype == "sample_prev":
            calls.append(
                f"{primary_out}_{node.node_id} = texture(previous_frame, evo_safe_uv({in_vars.get('uv', 'v_uv')})).rgb;"
            )
        elif node.ntype == "post_glow":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('col', 'vec3(0.0)')} + evo_safe_color({in_vars.get('prev', 'vec3(0.0)')}) * {node.params.get('intensity', 0.5)} * 0.15;"
            )
        elif node.ntype == "post_bloom":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('col', 'vec3(0.0)')} + evo_safe_color({in_vars.get('prev', 'vec3(0.0)')}) * {node.params.get('intensity', 0.5)} * 0.1;"
            )
        elif node.ntype == "post_chromatic":
            calls.append(
                f"{primary_out}_{node.node_id} = vec3(evo_safe_color({in_vars.get('prev', 'vec3(0.0)')}).r, {in_vars.get('col', 'vec3(0.0)')}.g, evo_safe_color({in_vars.get('prev', 'vec3(0.0)')}).b);"
            )
        elif node.ntype == "post_feedback":
            calls.append(
                f"{primary_out}_{node.node_id} = mix(evo_safe_color({in_vars.get('col', 'vec3(0.0)')}), evo_safe_color({in_vars.get('prev', 'vec3(0.0)')}), clamp({node.params.get('alpha', 0.85)}, 0.0, 0.999));"
            )

        elif node.ntype == "post_distortion":
            uv_var = in_vars.get("uv", "v_uv")
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('col', 'vec3(0.0)')} + vec3(sin({uv_var}.x * 10.0 + t), sin({uv_var}.y * 10.0 + t), 0.0) * 0.1 * {node.params.get('amount', 0.5)};"
            )
        elif node.ntype == "curl_noise":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('p', 'uv')} + curl_noise({in_vars.get('p', 'uv')} * {node.params.get('scale', 3.0)}, {node.params.get('strength', 0.5)});"
            )
        elif node.ntype == "domain_warp":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('p', 'uv')} + curl_noise({in_vars.get('p', 'uv')} * {node.params.get('scale', 2.0)}, {node.params.get('strength', 0.5)}) * float({int(node.params.get('octaves', 3))});"
            )
        elif node.ntype == "voronoi_cells":
            calls.append(
                f"cells_{node.node_id} = voronoi({in_vars.get('p', 'uv')} * {node.params.get('scale', 5.0)} + {in_vars.get('t', 't')} * {node.params.get('speed', 0.5)}, {node.params.get('jitter', 1.0)});"
            )
            calls.append(f"dist_{node.node_id} = cells_{node.node_id}.z;")
        elif node.ntype == "voronoi_edge":
            calls.append(
                f"{primary_out}_{node.node_id} = smoothstep(0.0, {node.params.get('thickness', 0.05)}, voronoi({in_vars.get('p', 'uv')} * {node.params.get('scale', 5.0)}, 1.0).z);"
            )
        elif node.ntype == "reaction_diffusion":
            calls.append(
                f"{primary_out}_{node.node_id} = reaction_diffusion({in_vars.get('uv', 'uv')}, {node.params.get('feed', 0.055)}, {node.params.get('kill', 0.062)}, {int(node.params.get('iterations', 8))});"
            )
        elif node.ntype == "glow_node":
            calls.append(
                f"{primary_out}_{node.node_id} = bloom({in_vars.get('col', 'vec3(0.0)')} + {in_vars.get('prev', 'vec3(0.0)')} * {node.params.get('intensity', 0.5)}, {in_vars.get('uv', 'v_uv')}, {node.params.get('threshold', 0.3)}, {node.params.get('intensity', 0.5)}, {node.params.get('radius', 0.4)});"
            )
        elif node.ntype == "bloom_node":
            calls.append(
                f"{primary_out}_{node.node_id} = bloom({in_vars.get('col', 'vec3(0.0)')} + {in_vars.get('prev', 'vec3(0.0)')} * {node.params.get('intensity', 0.5)}, {in_vars.get('uv', 'v_uv')}, {node.params.get('threshold', 0.6)}, {node.params.get('intensity', 0.5)}, {node.params.get('radius', 0.5)});"
            )
        # ---- New post modules from evo/post/ -----------------------------------
        elif node.ntype == "post_polar_galaxy":
            calls.append(
                f"{primary_out}_{node.node_id} = polar_galaxy({in_vars.get('col', 'vec3(0.0)')}, {in_vars.get('uv', 'v_uv')}, {in_vars.get('t', 't')}, {node.params.get('swirl_strength', 0.35)});"
            )
        elif node.ntype == "post_cells":
            calls.append(
                f"{primary_out}_{node.node_id} = organic_cells({in_vars.get('col', 'vec3(0.0)')}, {in_vars.get('uv', 'v_uv')}, {in_vars.get('t', 't')}, {node.params.get('mix_amount', 0.35)});"
            )
        elif node.ntype == "post_nonlinear_trails":
            prev_col = in_vars.get("prev", "vec3(0.0)")
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('col', 'vec3(0.0)')} + {prev_col} * {node.params.get('ghost_gain', 0.25)};"
            )
        elif node.ntype == "post_audio_topology":
            calls.append(
                f"{primary_out}_{node.node_id} = audio_topology({in_vars.get('col', 'vec3(0.0)')}, {in_vars.get('uv', 'v_uv')}, {in_vars.get('t', 't')}, bass, mid, treble, {node.params.get('bass_warp', 0.15)}, {node.params.get('treble_shear', 0.05)});"
            )
        elif node.ntype == "post_depth_cloud":
            calls.append(
                f"{primary_out}_{node.node_id} = depth_cloud({in_vars.get('col', 'vec3(0.0)')}, {in_vars.get('uv', 'v_uv')}, {in_vars.get('t', 't')}, texture(previous_frame, {in_vars.get('uv', 'v_uv')}).rgb, {int(node.params.get('taps', 6))}, {node.params.get('mix', 0.4)});"
            )
        elif node.ntype == "post_neon_edge":
            calls.append(
                f"{primary_out}_{node.node_id} = neon_edge({in_vars.get('col', 'vec3(0.0)')}, {in_vars.get('uv', 'v_uv')}, texture(previous_frame, {in_vars.get('uv', 'v_uv')}).rgb, {node.params.get('edge_strength', 2.0)});"
            )
        # ---- Feedback transform nodes -----------------------------------------
        elif node.ntype == "feedback_drift":
            funcs.append(f"""
vec2 {func_name}(vec2 uv, float t) {{
    float dx = sin(t * {node.params.get("drift_x_speed", 0.2)}) * 0.02;
    float dy = cos(t * {node.params.get("drift_y_speed", 0.17)}) * 0.02;
    return uv + vec2(dx, dy);
}}""")
            calls.append(
                f"{primary_out}_{node.node_id} = {func_name}({in_vars.get('uv', 'v_uv')}, {in_vars.get('t', 't')});"
            )
        elif node.ntype == "feedback_kaleido":
            calls.append(
                f"{primary_out}_{node.node_id} = kaleidoscope_fb({in_vars.get('uv', 'v_uv')}, {in_vars.get('t', 't')}, {int(node.params.get('symmetry', 6))});"
            )
        elif node.ntype == "feedback_mirror":
            calls.append(
                f"{primary_out}_{node.node_id} = mirror_fb({in_vars.get('uv', 'v_uv')}, {in_vars.get('t', 't')}, {node.params.get('jitter', 0.01)});"
            )
        elif node.ntype == "feedback_spiral":
            calls.append(
                f"{primary_out}_{node.node_id} = spiral_fb({in_vars.get('uv', 'v_uv')}, {in_vars.get('t', 't')}, {node.params.get('rotate_speed', 0.3)}, {node.params.get('breathe_speed', 0.5)});"
            )
        elif node.ntype == "feedback_zoom":
            calls.append(
                f"{primary_out}_{node.node_id} = zoom_fb({in_vars.get('uv', 'v_uv')}, {in_vars.get('t', 't')}, {node.params.get('zoom_speed', 0.6)}, {node.params.get('zoom_amount', 0.012)});"
            )
        # ---- SDF field --------------------------------------------------------
        elif node.ntype == "sdf_field":
            calls.append(
                f"{primary_out}_{node.node_id} = sdf_scene({in_vars.get('p', 'uv')}, {in_vars.get('t', 't')}, {int(node.params.get('march_steps', 32))}, {node.params.get('camera_dist', 3.5)}, {int(node.params.get('primitive', 0))});"
            )
        # ---- Math helpers -----------------------------------------------------
        elif node.ntype == "math_mul":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('a', '0.0')} * {in_vars.get('b', '0.0')};"
            )
        elif node.ntype == "math_add":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('a', '0.0')} + {in_vars.get('b', '0.0')};"
            )
        # ---- Audio inputs -----------------------------------------------------
        elif node.ntype == "input_audio_treble":
            calls.append(f"treble_{node.node_id} = treble;")
        elif node.ntype == "input_audio_mid":
            calls.append(f"mid_{node.node_id} = mid;")
        elif node.ntype == "input_audio_bass":
            calls.append(f"bass_{node.node_id} = bass;")
        # ---- Lighting nodes ---------------------------------------------------
        elif node.ntype == "lighting_ambient":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('col', 'vec3(0.0)')} * {node.params.get('intensity', 0.3)} + vec3({node.params.get('color', '0.5,0.5,0.5')});"
            )
        elif node.ntype == "lighting_point":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('col', 'vec3(0.0)')} + vec3({node.params.get('intensity', 0.8)}) * exp(-{node.params.get('falloff', 2.0)} * {in_vars.get('n', '0.0')});"
            )
        elif node.ntype == "lighting_bloom":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('col', 'vec3(0.0)')} + texture(previous_frame, {in_vars.get('uv', 'v_uv')}).rgb * {node.params.get('intensity', 0.5)};"
            )
        # ---- Particle nodes ---------------------------------------------------
        elif node.ntype == "particle_field":
            calls.append(
                f"{primary_out}_{node.node_id} = vec3(sin({in_vars.get('t', 't')} * {node.params.get('speed', 0.5)}) * 0.5 + 0.5);"
            )
        elif node.ntype == "particle_trails":
            calls.append(
                f"{primary_out}_{node.node_id} = mix({in_vars.get('col', 'vec3(0.0)')}, texture(previous_frame, {in_vars.get('uv', 'v_uv')}).rgb, {node.params.get('fade', 0.9)});"
            )
        # ---- Volume nodes -----------------------------------------------------
        elif node.ntype == "volume_fog":
            calls.append(
                f"{primary_out}_{node.node_id} = mix({in_vars.get('col', 'vec3(0.0)')}, vec3({node.params.get('density', 0.5)}), {node.params.get('falloff', 1.0)} * {in_vars.get('depth', '0.5')});"
            )
        elif node.ntype == "volume_cloud":
            calls.append(
                f"{primary_out}_{node.node_id} = mix({in_vars.get('col', 'vec3(0.0)')}, vec3(0.8), {node.params.get('density', 0.4)} * (0.5 + 0.5 * sin({in_vars.get('uv', 'v_uv')}.x * {node.params.get('scale', 3.0)} + {in_vars.get('t', 't')} * {node.params.get('speed', 0.2)})));"
            )
        # ---- Camera nodes -----------------------------------------------------
        elif node.ntype == "camera_lens":
            calls.append(
                f"vec2 {func_name}_out = {in_vars.get('uv', 'v_uv')} * (1.0 + {node.params.get('distortion', 0.1)} * dot({in_vars.get('uv', 'v_uv')} - 0.5, {in_vars.get('uv', 'v_uv')} - 0.5));"
            )
            calls.append(f"{primary_out}_{node.node_id} = {func_name}_out;")
        elif node.ntype == "camera_dof":
            calls.append(
                f"{primary_out}_{node.node_id} = mix({in_vars.get('col', 'vec3(0.0)')}, texture(previous_frame, {in_vars.get('uv', 'v_uv')}).rgb, {node.params.get('blur', 0.1)});"
            )
        # ---- Composite nodes --------------------------------------------------
        elif node.ntype == "composite_add":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('a', 'vec3(0.0)')} + {in_vars.get('b', 'vec3(0.0)')};"
            )
        elif node.ntype == "composite_mul":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('a', 'vec3(0.0)')} * {in_vars.get('b', 'vec3(0.0)')};"
            )
        elif node.ntype == "composite_screen":
            calls.append(
                f"{primary_out}_{node.node_id} = 1.0 - (1.0 - {in_vars.get('a', 'vec3(0.0)')}) * (1.0 - {in_vars.get('b', 'vec3(0.0)')});"
            )
        elif node.ntype == "composite_overlay":
            calls.append(
                f"{primary_out}_{node.node_id} = mix(2.0 * {in_vars.get('a', 'vec3(0.0)')} * {in_vars.get('b', 'vec3(0.0)')}, 1.0 - 2.0 * (1.0 - {in_vars.get('a', 'vec3(0.0)')}) * (1.0 - {in_vars.get('b', 'vec3(0.0)')}), step(0.5, {in_vars.get('a', 'vec3(0.0)')}));"
            )
        # ---- More math nodes --------------------------------------------------
        elif node.ntype == "math_clamp":
            calls.append(
                f"{primary_out}_{node.node_id} = clamp({in_vars.get('x', '0.0')}, {in_vars.get('lo', '0.0')}, {in_vars.get('hi', '1.0')});"
            )
        elif node.ntype == "math_fract":
            calls.append(f"{primary_out}_{node.node_id} = fract({in_vars.get('x', '0.0')});")
        elif node.ntype == "math_mod":
            calls.append(
                f"{primary_out}_{node.node_id} = mod({in_vars.get('x', '0.0')}, max(abs({in_vars.get('y', '1.0')}), 1e-5) * sign({in_vars.get('y', '1.0')} + 1e-5));"
            )
        elif node.ntype == "math_abs":
            calls.append(f"{primary_out}_{node.node_id} = abs({in_vars.get('x', '0.0')});")
        elif node.ntype == "math_pow":
            calls.append(
                f"{primary_out}_{node.node_id} = pow(max({in_vars.get('x', '0.0')}, 0.0), {in_vars.get('exp', '2.0')});"
            )
        elif node.ntype == "math_sqrt":
            calls.append(f"{primary_out}_{node.node_id} = sqrt(max({in_vars.get('x', '0.0')}, 0.0));")
        elif node.ntype == "math_length":
            calls.append(
                f"{primary_out}_{node.node_id} = length({in_vars.get('v', 'vec2(0.0)')});"
            )
        elif node.ntype == "math_dot":
            calls.append(
                f"{primary_out}_{node.node_id} = dot({in_vars.get('a', 'vec2(0.0)')}, {in_vars.get('b', 'vec2(0.0)')});"
            )
        elif node.ntype == "math_normalize":
            calls.append(
                f"{{ vec2 _v = {in_vars.get('v', 'vec2(0.0)')}; float _l = length(_v); {primary_out}_{node.node_id} = _l > 1e-5 ? _v / _l : vec2(0.0); }}"
            )

        elif node.ntype == "math_step":
            calls.append(
                f"{primary_out}_{node.node_id} = step({in_vars.get('edge', '0.5')}, {in_vars.get('x', '0.0')});"
            )
        # ---- Sample nodes -----------------------------------------------------
        elif node.ntype == "sample_blur":
            calls.append(
                f"{primary_out}_{node.node_id} = {in_vars.get('col', 'vec3(0.0)')} + texture(previous_frame, {in_vars.get('uv', 'v_uv')}).rgb * {node.params.get('radius', 0.01)};"
            )
        # ---- Output nodes with edge weights -----------------------------------
        elif node.ntype in ("output_feedback", "output_post", "output_bloom", "output_composite"):
            blend = int(node.params.get("blend_mode", 0))
            col_in = "vec3(0.0)"
            for e in self.graph.edges:
                if e.dst_id != node.node_id or e.dst_port != "col":
                    continue
                if e.src_id == e.dst_id:
                    continue
                src = self.graph.nodes.get(e.src_id)
                if src is None:
                    continue
                src_tpl = NODE_LIBRARY.get(src.ntype, {})
                src_outs = src_tpl.get("outputs", {})
                if e.src_port in src_outs:
                    sp = e.src_port
                    st = src_outs[e.src_port]
                else:
                    sp = next(iter(src_outs.keys()), "col")
                    st = src_outs.get(sp, "vec3")
                if src.ntype in ("input_uv", "input_time", "input_audio_bass", "input_audio_mid", "input_audio_treble"):
                    name, typ = {
                        "input_uv": ("v_uv", "vec2"),
                        "input_time": ("u_time", "float"),
                        "input_audio_bass": ("bass", "float"),
                        "input_audio_mid": ("mid", "float"),
                        "input_audio_treble": ("treble", "float"),
                    }[src.ntype]
                    src_var = self._cast_var(name, typ, "vec3")
                else:
                    src_var = self._cast_var(f"{sp}_{src.node_id}", st, "vec3")
                if blend == 0:
                    col_in = src_var
                elif blend == 1:
                    col_in = f"mix(vec3(0.0), {src_var}, 0.5)"
                elif blend == 2:
                    col_in = f"{src_var} + vec3(0.1)"
                else:
                    col_in = f"1.0 - (1.0 - {src_var})"
                break
            calls.append(f"outCol = evo_safe_color({col_in});")

        # ---- Audio inputs -----------------------------------------------------

        self._emitted[f"{primary_out}_{node.node_id}"] = primary_type

        for e in self.graph.edges:
            if e.src_id != node.node_id:
                continue
            port = e.src_port
            if port == primary_out:
                continue
            alias = f"{port}_{node.node_id}"
            if alias in self._emitted or any(alias in c for c in calls):
                continue
            ptype = out_tpl.get(port, primary_type)
            expr = (
                self._cast_var(f"{primary_out}_{node.node_id}", primary_type, ptype)
                if ptype != primary_type
                else f"{primary_out}_{node.node_id}"
            )
            self._emitted[alias] = ptype
            calls.append(f"{alias} = {expr};")

    def _gen_output(
        self, node: ShaderNode, tpl: dict[str, Any], calls: list[str], accumulate: bool = False
    ) -> None:
        col_exprs: list[str] = []
        for e in self.graph.edges:
            if e.dst_id != node.node_id or e.dst_port != "col":
                continue
            if e.src_id == e.dst_id:
                continue
            src = self.graph.nodes.get(e.src_id)
            if src is not None and src.ntype in (
                "input_uv",
                "input_time",
                "input_audio_bass",
                "input_audio_mid",
                "input_audio_treble",
            ):
                name, typ = {
                    "input_uv": ("v_uv", "vec2"),
                    "input_time": ("u_time", "float"),
                    "input_audio_bass": ("bass", "float"),
                    "input_audio_mid": ("mid", "float"),
                    "input_audio_treble": ("treble", "float"),
                }[src.ntype]
                col_exprs.append(self._cast_var(name, typ, "vec3"))
            elif src is not None:
                src_tpl = NODE_LIBRARY.get(src.ntype, {})
                src_outs = src_tpl.get("outputs", {})
                if e.src_port in src_outs:
                    port_type = src_outs[e.src_port]
                    var_name = f"{e.src_port}_{e.src_id}"
                else:
                    out_ports = list(src_outs.keys())
                    primary_port = out_ports[0] if out_ports else "col"
                    port_type = src_outs.get(primary_port, "vec3")
                    var_name = f"{primary_port}_{e.src_id}"
                col_exprs.append(self._cast_var(var_name, port_type, "vec3"))
        if not col_exprs:
            col_exprs = ["vec3(0.0)"]
        col_in = " + ".join(col_exprs)
        if accumulate:
            calls.append(f"outCol += evo_safe_color({col_in});")
        else:
            calls.append(f"outCol = {col_in};")

    def _default_for_type(self, ptype: str) -> str:
        return {
            "float": "0.0",
            "vec2": "vec2(0.0)",
            "vec3": "vec3(0.0)",
            "vec4": "vec4(0.0)",
            "bool": "false",
        }.get(ptype, "0.0")

    def _cast_var(self, name: str, from_type: str, to_type: str) -> str:
        """Cast ``name`` from ``from_type`` to ``to_type`` if they differ."""
        if from_type == to_type:
            return name
        tmpl = TYPE_CAST.get((from_type, to_type))
        if tmpl is None:
            return name
        return tmpl.format(name)

    def graph_get_inputs(self, node: ShaderNode) -> dict[str, tuple[str, str]]:
        mapping: dict[str, tuple[str, str]] = {}
        for e in self.graph.edges:
            if e.dst_id != node.node_id:
                continue
            if e.src_id == node.node_id:
                continue
            src = self.graph.nodes.get(e.src_id)
            if src is not None and src.ntype in (
                "input_uv",
                "input_time",
                "input_audio_bass",
                "input_audio_mid",
                "input_audio_treble",
            ):
                name, typ = {
                    "input_uv": ("v_uv", "vec2"),
                    "input_time": ("u_time", "float"),
                    "input_audio_bass": ("bass", "float"),
                    "input_audio_mid": ("mid", "float"),
                    "input_audio_treble": ("treble", "float"),
                }[src.ntype]
                mapping[e.dst_port] = (name, typ)
            elif src is not None:
                src_tpl = NODE_LIBRARY.get(src.ntype, {})
                src_outs = src_tpl.get("outputs", {})
                if e.src_port in src_outs:
                    out_typ = src_outs[e.src_port]
                    mapping[e.dst_port] = (f"{e.src_port}_{e.src_id}", out_typ)
                else:
                    primary_out = next(iter(src_outs.keys())) if src_outs else "out"
                    out_typ = src_outs.get(primary_out, e.dtype)
                    mapping[e.dst_port] = (f"{primary_out}_{e.src_id}", out_typ)
        return mapping

    def graph_get_input_with_weight(self, node: ShaderNode, port: str) -> tuple[str, str, float]:
        """Resolve an input port to (expression, type, edge_weight)."""
        tpl = NODE_LIBRARY.get(node.ntype, {})
        port_type = tpl.get("inputs", {}).get(port, "float")
        for e in self.graph.edges:
            if e.dst_id == node.node_id and e.dst_port == port:
                if e.src_id == node.node_id:
                    continue
                name, typ = self.graph_get_inputs(node).get(port, (self._resolve_default(port, port_type), port_type))
                return name, typ, e.weight
        return self._resolve_default(port, port_type), port_type, 1.0


    def _resolve_default(self, port, port_type):
        defaults = {
            "float": "0.0",
            "vec2": "vec2(0.0)",
            "vec3": "vec3(0.0)",
            "vec4": "vec4(0.0)",
            "sampler2D": "previous_frame",
            "bool": "false",
        }
        return defaults.get(port_type, "0.0")

    def compute_shader_metrics(self, graph: ShaderGraph) -> dict[str, int]:
        """Parse a compiled shader graph into rough instruction-level metrics.

        Returns a dict with keys: ``instructions``, ``alu``, ``texture_reads``,
        ``branches``. These are heuristic estimates derived from the node
        library cost table, not from a real GPU compiler backend.
        """
        metrics = {"instructions": 0, "alu": 0, "texture_reads": 0, "branches": 0}
        for node in graph.nodes.values():
            cost = NODE_COST_METRICS.get(
                node.ntype,
                {
                    "instructions": 4,
                    "alu": 3,
                    "texture_reads": 0,
                    "branches": 0,
                    "fps_cost": 1,
                },
            )
            for key in metrics:
                metrics[key] += cost.get(key, 0)
        return metrics

    def _fallback_shader(self, brightness=1.0, feedback_alpha=0.85):
        fb = max(0.0, min(0.999, float(feedback_alpha)))
        br = max(0.0, float(brightness))
        return (
            "#version 330 core\n"
            "in vec2 v_uv;\n"
            "out vec4 fragColor;\n"
            "uniform float u_time;\n"
            "uniform vec2 u_resolution;\n"
            "uniform float bass;\n"
            "uniform float mid;\n"
            "uniform float treble;\n"
            "uniform sampler2D previous_frame;\n\n"
            f"{GLSL_HELPERS}\n\n"
            "void main() {\n"
            "    vec2 uv = evo_safe_uv(v_uv);\n"
            "    float t = u_time;\n"
            "    float n = 0.5 + 0.5 * sin(uv.x * 3.0 + uv.y * 5.0 + t);\n"
            "    vec3 col = 0.5 + 0.5 * cos(6.28318 * (n + vec3(0.0, 0.33, 0.67)));\n"
            "    vec3 prev = texture(previous_frame, uv).rgb;\n"
            "    if (any(isnan(prev)) || any(isinf(prev))) prev = vec3(0.0);\n"
            f"    col = mix(evo_safe_color(col * {br:.8g}), prev, {fb:.8g});\n"
            "    fragColor = vec4(evo_safe_color(col), 1.0);\n"
            "}\n"
        )



GLSL_HELPERS = """// GLSL helper functions shared by all evolved shader graphs
#define EVO_TWO_PI 6.28318530718

vec3 evo_sat(vec3 c) {
    c = max(c, vec3(0.0));
    // Tone-map hot values before they poison feedback/history frames.
    return c / (vec3(1.0) + c);
}

vec3 evo_safe_color(vec3 c) {
    c = clamp(c, vec3(-16.0), vec3(16.0));
    if (any(isnan(c)) || any(isinf(c))) return vec3(0.0);
    return evo_sat(c);
}

vec2 evo_safe_uv(vec2 uv) {
    uv = clamp(uv, vec2(-2.0), vec2(2.0));
    if (any(isnan(uv)) || any(isinf(uv))) return vec2(0.5);
    return fract(uv);
}

vec3 voronoi(vec2 x, float jitter) {
    vec2 n = floor(x);
    vec2 f = fract(x);
    float md = 8.0;
    vec2 mr;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = vec2(fract(sin(dot(n + g, vec2(12.9898, 78.233))) * 43758.5453),
                          fract(sin(dot(n + g, vec2(43.1243, 67.8976))) * 43758.5453));
            o = -0.5 + jitter * o;
            vec2 d = g + o - f;
            float dist = dot(d, d);
            if (dist < md) { md = dist; mr = o; }
        }
    }
    return vec3(md, mr);
}
float reaction_diffusion(vec2 uv, float feed, float kill, int steps) {
    float u = 0.5, v = 0.5; vec2 p = uv;
    for (int i = 0; i < 64; i++) {
        if (i >= steps) break;
        vec2 n = vec2(fract(sin(dot(floor(p*4.0), vec2(12.9898,78.233)))*43758.5453),
                      fract(sin(dot(floor(p*4.0+100.0), vec2(12.9898,78.233)))*43758.5453));
        float lap = n.x - u;
        float du = feed*u - kill*u*v + 0.2*lap;
        float dv = feed*v + 0.1*lap;
        u = clamp(u+du*0.1,0.0,1.0); v = clamp(v+dv*0.1,0.0,1.0);
        p += vec2(0.01);
    }
    return u;
}
float mandelbrot(vec2 p, int max_iter) {
    vec2 z = vec2(0.0); float it = 0.0;
    for (int i = 0; i < 256; i++) {
        if (i >= max_iter) break;
        z = vec2(z.x*z.x-z.y*z.y, 2.0*z.x*z.y) + p;
        if (dot(z,z) > 4.0) break;
        it += 1.0;
    }
    return it/float(max_iter);
}
vec2 curl_noise(vec2 p, float strength) {
    float e = 0.01;
    float n1 = fract(sin(dot(floor(p+vec2(0.0,e)), vec2(12.9898,78.233)))*43758.5453);
    float n2 = fract(sin(dot(floor(p-vec2(0.0,e)), vec2(12.9898,78.233)))*43758.5453);
    float n3 = fract(sin(dot(floor(p+vec2(e,0.0)), vec2(12.9898,78.233)))*43758.5453);
    float n4 = fract(sin(dot(floor(p-vec2(e,0.0)), vec2(12.9898,78.233)))*43758.5453);
    return vec2(n1-n2, n4-n3)*strength;
}
vec3 bloom(vec3 col, vec2 uv, float threshold, float intensity, float radius) {
    float bright = max(col.r, max(col.g, col.b));
    float mask = max(bright-threshold,0.0)/max(1.0-threshold,0.001);
    return col + col*mask*intensity;
}
vec3 neon_edge(vec3 col, vec2 uv, vec3 prev, float edge_strength) {
    float e = 0.0035;
    vec3 a = texture(previous_frame, uv + vec2(e, 0.0)).rgb;
    vec3 b = texture(previous_frame, uv - vec2(e, 0.0)).rgb;
    vec3 c = texture(previous_frame, uv + vec2(0.0, e)).rgb;
    vec3 d = texture(previous_frame, uv - vec2(0.0, e)).rgb;
    float edge = length(a - b) + length(c - d);
    return col + vec3(edge) * edge_strength;
}
vec3 audio_topology(vec3 col, vec2 uv, float t, float bass, float mid, float treble, float bass_warp, float treble_shear) {
    float warp = bass * bass_warp + mid * 0.05;
    float shear = treble * treble_shear;
    vec2 duv = uv + vec2(sin(uv.y * 10.0 + t * 2.0 + shear), cos(uv.x * 10.0 + t * 2.0)) * warp;
    return mix(col, texture(previous_frame, fract(duv)).rgb, 0.2 + bass * 0.3);
}
vec3 polar_galaxy(vec3 col, vec2 uv, float t, float swirl_strength) {
    vec2 centered = uv - 0.5;
    float r = length(centered);
    float a = atan(centered.y, centered.x);
    float swirl = a + r * swirl_strength + t * 0.1;
    vec2 suv = vec2(cos(swirl), sin(swirl)) * r + 0.5;
    return mix(col, texture(previous_frame, fract(suv)).rgb, 0.5);
}
vec3 organic_cells(vec3 col, vec2 uv, float t, float mix_amount) {
    float n = fract(sin(dot(floor(uv * 8.0), vec2(12.9898, 78.233))) * 43758.5453);
    vec2 off = vec2(cos(n * 6.28 + t), sin(n * 6.28 + t)) * 0.02;
    return mix(col, texture(previous_frame, fract(uv + off)).rgb, mix_amount);
}
vec3 depth_cloud(vec3 col, vec2 uv, float t, vec3 prev, int taps, float mix_val) {
    vec3 acc = vec3(0.0);
    float tw = 0.0;
    for (int i = 0; i < 12; i++) {
        if (i >= taps) break;
        float fi = float(i);
        vec2 off = vec2(sin(t + fi * 1.7), cos(t + fi * 2.3)) * 0.01 * (fi + 1.0);
        float w = 1.0 / (fi + 1.0);
        acc += texture(previous_frame, fract(uv + off)).rgb * w;
        tw += w;
    }
    acc /= max(tw, 0.001);
    return mix(col, acc, mix_val);
}
vec2 mirror_fb(vec2 uv, float t, float jitter) {
    float j = sin(t * 3.0) * jitter;
    uv = fract(uv + j);
    if (uv.x > 0.5) uv.x = 1.0 - uv.x;
    if (uv.y > 0.5) uv.y = 1.0 - uv.y;
    return uv;
}
vec2 kaleidoscope_fb(vec2 uv, float t, int symmetry) {
    vec2 centered = uv - 0.5;
    float r = length(centered);
    float a = atan(centered.y, centered.x);
    float seg = 6.28318 / float(symmetry);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    return vec2(cos(a), sin(a)) * r + 0.5;
}
vec2 spiral_fb(vec2 uv, float t, float rotate_speed, float breathe_speed) {
    vec2 centered = uv - 0.5;
    float r = length(centered);
    float a = atan(centered.y, centered.x) + t * rotate_speed;
    float scale = 1.0 + sin(t * breathe_speed) * 0.1;
    return vec2(cos(a), sin(a)) * r * scale + 0.5;
}
vec2 zoom_fb(vec2 uv, float t, float zoom_speed, float zoom_amount) {
    vec2 centered = uv - 0.5;
    float scale = 1.0 + sin(t * zoom_speed) * zoom_amount;
    return centered / max(scale, 0.01) + 0.5;
}
float sdf_sphere(vec3 p, float r) { return length(p) - r; }
float sdf_box(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float sdf_torus(vec3 p, vec2 t) { vec2 q = vec2(length(p.xz) - t.x, p.y); return length(q) - t.y; }
float sdf_cyl(vec3 p, float r) { vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, 1.0); return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); }
float noise3d(vec3 p) { vec3 i = floor(p); vec3 f = fract(p); f = f*f*(3.0-2.0*f); float a = fract(sin(dot(i, vec3(12.9898,78.233,45.164)))*43758.5453); float b = fract(sin(dot(i+vec3(1.0,0.0,0.0), vec3(12.9898,78.233,45.164)))*43758.5453); float c = fract(sin(dot(i+vec3(0.0,1.0,0.0), vec3(12.9898,78.233,45.164)))*43758.5453); float d = fract(sin(dot(i+vec3(1.0,1.0,0.0), vec3(12.9898,78.233,45.164)))*43758.5453); float e = fract(sin(dot(i+vec3(0.0,0.0,1.0), vec3(12.9898,78.233,45.164)))*43758.5453); float ff = fract(sin(dot(i+vec3(1.0,0.0,1.0), vec3(12.9898,78.233,45.164)))*43758.5453); float g = fract(sin(dot(i+vec3(0.0,1.0,1.0), vec3(12.9898,78.233,45.164)))*43758.5453); float h = fract(sin(dot(i+vec3(1.0,1.0,1.0), vec3(12.9898,78.233,45.164)))*43758.5453); return mix(mix(mix(a,b,f.x),mix(c,d,f.x),f.y), mix(mix(e,ff,f.x),mix(g,h,f.x),f.y), f.z); }
float sdf_scene(vec2 p, float t, int march_steps, float camera_dist, int primitive) {
    vec3 pos = vec3(p * camera_dist, -camera_dist * 2.0);
    pos.xz *= mat2(cos(t*0.3), -sin(t*0.3), sin(t*0.3), cos(t*0.3));
    pos.y += sin(t*0.5) * 0.3;
    float n = noise3d(pos * 0.8 + t*0.2);
    vec3 rp = pos;
    rp.xy *= mat2(cos(t*0.2), -sin(t*0.2), sin(t*0.2), cos(t*0.2));
    float d;
    if(primitive == 0) d = sdf_sphere(rp, 1.0 + n*0.3);
    else if(primitive == 1) d = sdf_box(rp, vec3(0.8 + n*0.2));
    else if(primitive == 2) d = sdf_torus(rp, vec2(0.7 + n*0.15, 0.2 + n*0.1));
    else if(primitive == 3) d = sdf_cyl(rp, 0.6 + n*0.2);
    else d = sdf_sphere(rp, 1.0 + n*0.3);
    for(int i=0;i<4;i++){
        rp.xy *= mat2(cos(0.5), -sin(0.5), sin(0.5), cos(0.5));
        d = abs(d) - 0.5;
    }
    return d;
}
vec3 lighting_point_glsl(vec3 col, vec3 light_pos, float intensity, float falloff) {
    float dist = length(light_pos);
    return col + intensity * exp(-falloff * dist);
}
vec3 lighting_ambient_glsl(vec3 col, float intensity, vec3 color) {
    return col * intensity + color * 0.1;
}
float particle_field_glsl(vec2 uv, float t, float count, float size, float speed) {
    float n = fract(sin(dot(floor(uv * count), vec2(12.9898, 78.233))) * 43758.5453);
    float particle = smoothstep(size, 0.0, length(fract(uv * count + t * speed) - 0.5));
    return particle * (0.5 + 0.5 * sin(t + n * 6.28));
}
vec3 volume_fog_glsl(vec3 col, float density, float falloff, float depth) {
    float fog = exp(-falloff * depth);
    return mix(col, vec3(0.6, 0.7, 0.8), density * (1.0 - fog));
}
vec2 camera_lens_glsl(vec2 uv, float distortion, float vignette) {
    vec2 centered = uv - 0.5;
    float r = length(centered);
    float a = atan(centered.y, centered.x);
    float new_r = r * (1.0 + distortion * r * r);
    vec2 new_uv = vec2(cos(a), sin(a)) * new_r + 0.5;
    float vig = 1.0 - vignette * r * r;
    return new_uv * vig;
}
"""
