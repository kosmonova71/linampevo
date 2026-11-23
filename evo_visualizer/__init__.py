#!/usr/bin/env python3

"""evo_visualizer.py
Minimum living renderer scaffold.
Milestone 1:
  - Single organism that can mutate live.
  - Visual genome parameters -> generated GLSL fragment shader.
  - Render via OpenGL feedback ping-pong FBO.
This file intentionally avoids MilkDrop preset loading, transitions, VM,
expression parsings.
GTK dependency: Gtk.GLArea.
Features:
  - Real-time genome evolution with adaptive mutation strength
  - Audio-reactive rendering with feedback effects
  - Human feedback steering (like/dislike)
  - Preset/checkpoint system
  -
  - Comprehensive logging and error handling
"""

from __future__ import annotations

import logging

import gi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None

gi.require_version("Gtk", "4.0")

# Shared constants
# Biological systems
from evo_visualizer.bio import (  # noqa: E402
    OrganismBrain,
    Organs,
    Senses,
    Signal,
    SpeciesDef,
)

# Configuration
from evo_visualizer.config import (  # noqa: E402
    EvoConfig,
    MultiResTier,
    _lerp_cfg,
)
from evo_visualizer.constants import (  # noqa: E402
    AUDIO_CLAMP_MAX,
    AUDIO_CLAMP_MIN,
    AUDIO_KEYS,
    FBM_SCALE_NEG_Y,
    FBM_SCALE_X,
    FBM_SCALE_Y,
    FULLSCREEN_QUAD_VERTICES,
    GLSL_FLOAT_PRECISION,
    MAX_MUTATION_HISTORY,
    NOISE_FREQ_X,
    NOISE_FREQ_Y,
    NOISE_OFFSET,
    PALETTE_A,
    PALETTE_B,
    PALETTE_C,
    PALETTE_D,
    QUAD_STRIDE_BYTES,
    QUAD_VERTEX_COUNT,
    TWO_PI,
)

# Spatial ecosystem
from evo_visualizer.ecosystem import (  # noqa: E402
    AreaMemory,
    BodyPart,
    Climate,
    DynamicMaterial,
    EnhancedSpatialEcosystem,
    EnvironmentalEvent,
    ExtinctionArchive,
    FossilRecord,
    NicheDefinition,
    NicheManager,
    PersistentWorldTextures,
    PlantSpecies,
    ProceduralSkeleton,
    ResourceField,
    SkeletonNode,
    SoftBody,
    SpatialEcosystem,
    SpatialOrganism,
)

# Evolving post-effects (effects/*.frag shader pool)
from evo_visualizer.effects import (  # noqa: E402
    EffectLibrary,
    EffectManager,
    EffectSpec,
)

# Genomes
from evo_visualizer.genomes import (  # noqa: E402
    VisualGenome,
)

# Multi-resolution evolution
from evo_visualizer.multi_res import (  # noqa: E402
    MultiResolutionEvolution,
)

# Population and memory
from evo_visualizer.population import (  # noqa: E402
    MemoryDNA,
    Population,
    _hsv_to_rgb,
)

# Main renderer
from evo_visualizer.renderer import EvoRenderer  # noqa: E402

# Rendering support
from evo_visualizer.renderer_support import (  # noqa: E402
    AudioState,
    FeedbackBuffer,
    FitnessEvaluator,
    GeneFitnessTracker,
    MutationHeatmap,
    _HeatRow,
)

__all__ = [
    "AUDIO_CLAMP_MAX",
    "AUDIO_CLAMP_MIN",
    "AUDIO_KEYS",
    "FBM_SCALE_NEG_Y",
    "FBM_SCALE_X",
    "FBM_SCALE_Y",
    "FULLSCREEN_QUAD_VERTICES",
    "GLSL_FLOAT_PRECISION",
    "MAX_MUTATION_HISTORY",
    "NOISE_FREQ_X",
    "NOISE_FREQ_Y",
    "NOISE_OFFSET",
    "PALETTE_A",
    "PALETTE_B",
    "PALETTE_C",
    "PALETTE_D",
    "QUAD_STRIDE_BYTES",
    "QUAD_VERTEX_COUNT",
    "TWO_PI",
    "AreaMemory",
    "AudioState",
    "BodyPart",
    "Climate",
    "DynamicMaterial",
    "EffectLibrary",
    "EffectManager",
    "EffectSpec",
    "EnhancedSpatialEcosystem",
    "EnvironmentalEvent",
    "EvoConfig",
    "EvoRenderer",
    "ExtinctionArchive",
    "FeedbackBuffer",
    "FitnessEvaluator",
    "FossilRecord",
    "GeneFitnessTracker",
    "MemoryDNA",
    "MultiResTier",
    "MultiResolutionEvolution",
    "MutationHeatmap",
    "NicheDefinition",
    "NicheManager",
    "OrganismBrain",
    "Organs",
    "PersistentWorldTextures",
    "PlantSpecies",
    "Population",
    "ProceduralSkeleton",
    "ResourceField",
    "Senses",
    "Signal",
    "SkeletonNode",
    "SoftBody",
    "SpatialEcosystem",
    "SpatialOrganism",
    "SpeciesDef",
    "VisualGenome",
    "_HeatRow",
    "_hsv_to_rgb",
    "_lerp_cfg",
]
