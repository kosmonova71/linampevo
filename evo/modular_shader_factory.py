#!/usr/bin/env python3

"""modular_shader_factory.py

Assembles GLSL fragment shaders from modular .glsl files based on
a ModularGenome's module selections.

Architecture:
  Genome -> ModuleRegistry -> ShaderModule instances -> ShaderAssembler.build()
                                                                  ↓
                                                         Complete GLSL

This is now a thin delegation layer over the ``evo/modules`` package.
The monolithic ``generate()`` / ``generate_pipeline()`` methods delegate to
the :class:`ShaderAssembler` which uses discrete, composable module classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evo.modules.assembler import ShaderAssembler
from evo.modules.module_source_cache import ModuleSourceCache
from evo.modules.registry import ModuleRegistry, get_default_registry
from evo.pipeline_genome import PipelineGenome

logger = logging.getLogger(__name__)


class ModularShaderFactory:
    """Assembles fragment shaders from modular GLSL files.

    Delegates all module logic to the :class:`ShaderAssembler` and the
    discrete module classes in ``evo/modules/``.

    The ``generate()`` and ``generate_pipeline()`` methods are kept for
    backward compatibility — they now delegate to the ``ShaderAssembler``.
    """

    def __init__(
        self,
        evo_dir: Path | None = None,
        registry: ModuleRegistry | None = None,
        source_cache: ModuleSourceCache | None = None,
    ):
        if evo_dir is None:
            evo_dir = Path(__file__).parent.parent / "evo"
        self.evo_dir = Path(evo_dir)
        self.registry = registry or get_default_registry()
        self.assembler = ShaderAssembler(self.registry, source_cache=source_cache)

    @property
    def source_cache(self) -> ModuleSourceCache | None:
        return self.assembler.source_cache

    @source_cache.setter
    def source_cache(self, value: ModuleSourceCache | None) -> None:
        self.assembler.source_cache = value

    # ------------------------------------------------------------------
    # Public API: generate() and generate_pipeline()
    # ------------------------------------------------------------------

    def generate(
        self,
        genome,
        brightness: float = 1.0,
        feedback_alpha: float = 0.85,
        morphology: dict | None = None,
        raymarch_steps: int | None = None,
        fractal_layers: int | None = None,
        history_depth: int = 0,
    ):
        """Generate a complete GLSL fragment shader from a modular genome.

        Delegates to :meth:`ShaderAssembler.build`.

        Args:
            genome: ModularGenome instance
            brightness: Global brightness multiplier
            feedback_alpha: Feedback blending factor
            morphology: Optional dict of evolved-body metrics
            raymarch_steps: Cap on raymarch iterations (quality tier)
            fractal_layers: Cap on fractal layers (quality tier)
            history_depth: Number of temporal history frames

        Returns:
            ``(source, uniforms)`` where ``source`` is the complete GLSL
            fragment shader source and ``uniforms`` maps uniform name ->
            ``{"value", "kind"}``.
        """
        return self.assembler.build(
            genome=genome,
            brightness=brightness,
            feedback_alpha=feedback_alpha,
            morphology=morphology,
            raymarch_steps=raymarch_steps,
            fractal_layers=fractal_layers,
            history_depth=history_depth,
        )

    def generate_pipeline(
        self,
        pipeline: PipelineGenome,
        brightness: float = 1.0,
        feedback_alpha: float = 0.85,
        morphology: dict | None = None,
        history_depth: int = 0,
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        """Generate a complete GLSL fragment shader from a PipelineGenome.

        Delegates to :meth:`ShaderAssembler.build_pipeline`.

        Returns:
            ``(source, uniforms)`` tuple.
        """
        return self.assembler.build_pipeline(
            pipeline=pipeline,
            brightness=brightness,
            feedback_alpha=feedback_alpha,
            morphology=morphology,
            history_depth=history_depth,
        )

    # ------------------------------------------------------------------
    # Utility: expose the TOPOLOGY_DEFINES set so external code can
    # query which parameters change shader topology.
    # ------------------------------------------------------------------

    TOPOLOGY_DEFINES = frozenset(
        {
            "K_FRACTAL_LAYERS",
            "K_RD_ITERATIONS",
            "K_MB_MAX_ITER",
            "K_SDF_MARCH",
            "K_CURL_OCTAVES",
            "K_TWIST_ITER",
            "K_MB_JULIA",
            "K_SDF_PRIMITIVE",
            "K_SDF_SHAPE",
            "K_PAL_ID",
            "K_KALEIDO_SYM",
        }
    )
