"""Shared constants for the evo_visualizer package."""

from __future__ import annotations

TWO_PI = 6.28318
GLSL_FLOAT_PRECISION = 6
MAX_MUTATION_HISTORY = 64

# Geometry constants
FULLSCREEN_QUAD_VERTICES = [
    -1.0,
    -1.0,
    0.0,
    0.0,
    1.0,
    -1.0,
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    0.0,
    1.0,
]
QUAD_VERTEX_COUNT = 6
QUAD_STRIDE_BYTES = 16  # 4 floats * 4 bytes

# Audio constants
AUDIO_KEYS = ("bass", "mid", "treb", "beat")
AUDIO_CLAMP_MAX = 1.5
AUDIO_CLAMP_MIN = 0.0

# Palette function constants
PALETTE_A = (0.15, 0.20, 0.35)
PALETTE_B = (0.65, 0.25, 0.75)
PALETTE_C = (1.00, 0.95, 0.85)
PALETTE_D = (0.20, 0.40, 0.60)

# Noise/FBM constants
NOISE_FREQ_X = 123.34
NOISE_FREQ_Y = 456.21
NOISE_OFFSET = 45.32
FBM_SCALE_X = 1.6
FBM_SCALE_Y = 1.2
FBM_SCALE_NEG_Y = -1.2
