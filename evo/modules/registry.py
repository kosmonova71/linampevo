#!/usr/bin/env python3

"""Module registry — maps genome selections to module classes.

The :class:`ModuleRegistry` is a singleton-like mapping from module category
and name to the corresponding :class:`ShaderModule` subclass.  It uses a
simple decorator-based registration pattern.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import ModuleConfig, ShaderModule
from .boids_module import BoidsModule
from .camera_module import CameraModule
from .cellular_automata_module import CellularAutomataModule
from .erosion_module import ErosionModule
from .feedback_module import (
    DriftFeedbackModule,
    KaleidoFeedbackModule,
    MirrorFeedbackModule,
    SpiralFeedbackModule,
    ZoomFeedbackModule,
)
from .fluid_feedback_module import FluidFeedbackModule
from .heat_diffusion_module import HeatDiffusionModule
from .lighting_module import LightingModule
from .noise_module import (
    FBMFieldModule,
    MandelbrotFieldModule,
    ReactionDiffusionModule,
    SDFFieldModule,
    VoronoiFieldModule,
)
from .palette_module import (
    CosinePaletteModule,
    PlasmaPaletteModule,
    SpectralPaletteModule,
)
from .particle_module import ParticleModule
from .physarum_module import PhysarumModule
from .post_module import (
    AudioTopologyPostModule,
    BloomPostModule,
    CellsPostModule,
    ChromaticAberrationPostModule,
    DepthCloudPostModule,
    DistortionPostModule,
    FeedbackPostModule,
    GlowPostModule,
    NeonEdgePostModule,
    NonlinearTrailsPostModule,
    PolarGalaxyPostModule,
)
from .slime_mold_module import SlimeMoldModule
from .volume_module import VolumeModule
from .warp_module import (
    CurlWarpModule,
    NoneWarpModule,
    TwistWarpModule,
    VortexWarpModule,
)
from .wave_equation_module import WaveEquationModule

logger = logging.getLogger(__name__)


# ============================================================================
# Registry
# ============================================================================

#: Default categories and module name -> class mappings.
_DEFAULT_FIELD_MODULES: dict[str, type[ShaderModule]] = {
    "fbm": FBMFieldModule,
    "voronoi": VoronoiFieldModule,
    "reaction_diffusion": ReactionDiffusionModule,
    "mandelbrot": MandelbrotFieldModule,
    "sdf": SDFFieldModule,
    "fluid_feedback": FluidFeedbackModule,
    "wave_equation": WaveEquationModule,
    "heat_diffusion": HeatDiffusionModule,
    "cellular_automata": CellularAutomataModule,
    "slime_mold": SlimeMoldModule,
    "boids": BoidsModule,
    "physarum": PhysarumModule,
    "erosion": ErosionModule,
}

_DEFAULT_WARP_MODULES: dict[str, type[ShaderModule]] = {
    "curl": CurlWarpModule,
    "vortex": VortexWarpModule,
    "twist": TwistWarpModule,
    "none": NoneWarpModule,
}

_DEFAULT_PALETTE_MODULES: dict[str, type[ShaderModule]] = {
    "cosine": CosinePaletteModule,
    "spectral": SpectralPaletteModule,
    "plasma": PlasmaPaletteModule,
}

_DEFAULT_FEEDBACK_MODULES: dict[str, type[ShaderModule]] = {
    "drift": DriftFeedbackModule,
    "mirror": MirrorFeedbackModule,
    "spiral": SpiralFeedbackModule,
    "zoom": ZoomFeedbackModule,
    "kaleido": KaleidoFeedbackModule,
}

_DEFAULT_POST_MODULES: dict[str, type[ShaderModule]] = {
    "bloom": BloomPostModule,
    "glow": GlowPostModule,
    "chromatic_aberration": ChromaticAberrationPostModule,
    "distortion": DistortionPostModule,
    "feedback": FeedbackPostModule,
    "polar_galaxy": PolarGalaxyPostModule,
    "cells": CellsPostModule,
    "nonlinear_trails": NonlinearTrailsPostModule,
    "audio_topology": AudioTopologyPostModule,
    "depth_cloud": DepthCloudPostModule,
    "neon_edge": NeonEdgePostModule,
}

_DEFAULT_PASS_MODULES: dict[str, type[ShaderModule]] = {
    "lighting": LightingModule,
    "particle": ParticleModule,
    "volume": VolumeModule,
    "camera": CameraModule,
}


class ModuleRegistry:
    """Maps genome module selections to ShaderModule classes.

    Usage::

        reg = ModuleRegistry()
        # Field module for genome.field_type
        field_cls = reg.get("field", genome.field_type)
        field_mod = field_cls(ModuleConfig("field", genome.field_type, genome.field_params))

        # Warp module
        warp_cls = reg.get("warp", genome.warp)
        ...

        # Post modules (list)
        post_mods = reg.get_post_modules(genome.post, genome.post_params)
    """

    def __init__(self):
        self._categories: dict[str, dict[str, type[ShaderModule]]] = {
            "field": dict(_DEFAULT_FIELD_MODULES),
            "warp": dict(_DEFAULT_WARP_MODULES),
            "palette": dict(_DEFAULT_PALETTE_MODULES),
            "feedback": dict(_DEFAULT_FEEDBACK_MODULES),
            "post": dict(_DEFAULT_POST_MODULES),
            "pass": dict(_DEFAULT_PASS_MODULES),
        }

    # ---- Registration --------------------------------------------------

    def register(self, category: str, name: str, module_cls: type[ShaderModule]) -> None:
        """Register a module class.

        Args:
            category: One of "field", "warp", "palette", "feedback", "post", "pass".
            name: The module name (matches genome attribute values).
            module_cls: The ShaderModule subclass.
        """
        cat = self._categories.setdefault(category, {})
        cat[name] = module_cls

    # ---- Lookup --------------------------------------------------------

    def get(self, category: str, name: str) -> type[ShaderModule] | None:
        """Look up a module class by category and name.

        Returns None if not found.
        """
        cat = self._categories.get(category, {})
        cls = cat.get(name)
        if cls is None:
            logger.warning("Module not found: %s/%s", category, name)
        return cls

    def has(self, category: str, name: str) -> bool:
        return name in self._categories.get(category, {})

    # ---- Factories -----------------------------------------------------

    def create(
        self, category: str, name: str, params: dict[str, Any] | None = None, **extra_kwargs
    ) -> ShaderModule | None:
        """Create a module instance for the given category and name.

        Args:
            category: e.g. "field", "warp", "palette"
            name: e.g. "fbm", "curl", "cosine"
            params: The genome's parameter dict for this category.
            **extra_kwargs: Extra kwargs for ModuleConfig (raymarch_steps, etc.)

        Returns:
            A ShaderModule instance, or None if the module is not registered.
        """
        cls = self.get(category, name)
        if cls is None:
            return None
        config = ModuleConfig(
            category=category,
            name=name,
            params=dict(params or {}),
            **extra_kwargs,
        )
        return cls(config)

    def create_post_modules(
        self,
        post_names: list[str],
        post_params: dict[str, Any] | None = None,
    ) -> list[ShaderModule]:
        """Create a list of post-processing modules from genome selections."""
        modules: list[ShaderModule] = []
        for name in post_names:
            m = self.create("post", name, post_params)
            if m is not None:
                modules.append(m)
        return modules

    def create_pass_module(
        self,
        pass_type: str,
        params: dict[str, Any] | None = None,
    ) -> ShaderModule | None:
        """Create a pass-level module (lighting, particle, volume, camera)."""
        return self.create("pass", pass_type, params)

    # ---- Introspection -------------------------------------------------

    @property
    def available_fields(self) -> list[str]:
        return list(self._categories.get("field", {}).keys())

    @property
    def available_warps(self) -> list[str]:
        return list(self._categories.get("warp", {}).keys())

    @property
    def available_palettes(self) -> list[str]:
        return list(self._categories.get("palette", {}).keys())

    @property
    def available_feedback(self) -> list[str]:
        return list(self._categories.get("feedback", {}).keys())

    @property
    def available_posts(self) -> list[str]:
        return list(self._categories.get("post", {}).keys())

    @property
    def available_pass_types(self) -> list[str]:
        return list(self._categories.get("pass", {}).keys())


# ============================================================================
# Global default registry
# ============================================================================

_DEFAULT_REGISTRY: ModuleRegistry | None = None


def get_default_registry() -> ModuleRegistry:
    """Return the global default registry (lazily created)."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ModuleRegistry()
        # Register animation module
        from .animation_module import register as register_animation
        register_animation(_DEFAULT_REGISTRY)
    return _DEFAULT_REGISTRY


def module(category: str, name: str):
    """Decorator to register a ShaderModule subclass.

    Usage::

        @module("field", "fbm")
        class FBMFieldModule(ShaderModule):
            ...
    """

    def decorator(cls: type[ShaderModule]) -> type[ShaderModule]:
        reg = get_default_registry()
        reg.register(category, name, cls)
        return cls

    return decorator
