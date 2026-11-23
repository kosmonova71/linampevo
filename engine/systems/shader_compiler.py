"""engine/systems/shader_compiler.py

Shader compilation system for the evolution architecture.

Wraps the existing modular shader factory to provide a clean interface
that the evolution system can use without knowing about GLSL details.
Supports both single-pass (ModularGenome) and multi-pass (PipelineGenome)
compilation.

The compiler produces a ShaderArtifact with source, uniforms, and a
topology key for caching. Two artifacts with the same topology_key differ
only in numeric parameters, enabling cheap uniform uploads instead of
full recompiles.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from evo.modular_shader_factory import ModularShaderFactory

from .interfaces import (
    IGenome,
    RenderPassConfig,
    ShaderArtifact,
)

logger = logging.getLogger(__name__)


class SystemShaderCompiler:
    """Compile genomes into shader artifacts via the modular factory.

    Supports:
      - ModularGenome: single-pass shader
      - PipelineGenome: multi-pass shader (all passes in one GLSL program)
      - RenderPassConfig: per-pass compilation for true multi-FBO pipelines

    The compiler caches topology keys to avoid redundant work.
    """

    def __init__(
        self,
        factory: ModularShaderFactory | None = None,
        shader_cache: Any | None = None,
    ):
        self.factory = factory or ModularShaderFactory()
        self._shader_cache = shader_cache  # Optional ShaderCache from genome_base

    def compile(self, genome: IGenome) -> ShaderArtifact:
        """Compile any genome into a shader artifact.

        Dispatches to the appropriate compilation path based on genome type.
        """
        # Check for PipelineGenome first (multi-pass)
        ptype = type(genome).__name__
        if ptype == "PipelineGenome":
            return self._compile_pipeline(genome)

        # Single-pass: try ModularGenome path
        try:
            from evo.modular_genome import ModularGenome

            if isinstance(genome, ModularGenome):
                return self._compile_modular(genome)
        except ImportError:
            pass

        # Fallback: wrap in a default ModularGenome
        try:
            from evo.modular_genome import ModularGenome

            default = ModularGenome()
            return self._compile_modular(default)
        except ImportError:
            return ShaderArtifact(source="", uniforms={}, topology_key=())

    def compile_pass(
        self,
        pass_config: RenderPassConfig,
        global_params: dict[str, Any],
    ) -> ShaderArtifact:
        """Compile a single pass's shader section for multi-FBO rendering."""
        t0 = time.perf_counter()

        try:
            modules = pass_config.modules
            if modules is None:
                from evo.modular_genome import ModularGenome

                modules = ModularGenome()

            source, uniforms = self.factory.generate(
                modules,
                brightness=float(global_params.get("brightness", 1.0)),
                feedback_alpha=float(global_params.get("feedback_alpha", 0.85)),
                history_depth=int(global_params.get("history_depth", 0)),
            )

            topo_key = (
                pass_config.pass_type,
                modules.field_type,
                modules.warp,
                modules.palette,
                tuple(modules.post),
                tuple(sorted(modules.field_params.items())),
                tuple(sorted(modules.warp_params.items())),
                tuple(sorted(modules.palette_params.items())),
                tuple(sorted(modules.post_params.items())),
            )

            return ShaderArtifact(
                source=source,
                uniforms=uniforms,
                topology_key=topo_key,
                compile_time_seconds=time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.warning("pass compile failed for %s: %s", pass_config.pass_id, exc)
            return ShaderArtifact(
                source="",
                uniforms={},
                topology_key=(),
                compile_time_seconds=time.perf_counter() - t0,
            )

    def _compile_modular(self, genome) -> ShaderArtifact:
        """Compile a ModularGenome into a single-pass shader artifact."""
        t0 = time.perf_counter()

        try:
            source, uniforms = self.factory.generate(
                genome,
                brightness=1.0,
                feedback_alpha=0.85,
                history_depth=0,
            )

            topo_key = (
                genome.field_type,
                genome.warp,
                genome.palette,
                tuple(genome.post),
                genome.field_params.get("fractal_layers", 0),
                genome.field_params.get("iterations", 0),
                genome.field_params.get("march_steps", 0),
                genome.field_params.get("max_iter", 0),
                genome.field_params.get("octaves", 0),
                genome.warp_params.get("octaves", 0),
                genome.warp_params.get("iterations", 0),
                genome.palette_params.get("palette_id", 0),
                genome.field_params.get("julia_mode", 0),
                genome.field_params.get("primitive_type", 0),
                genome.field_params.get("sdf_shape", 0),
                genome.field_params.get("kaleido_sym", 0),
            )

            return ShaderArtifact(
                source=source,
                uniforms=uniforms,
                topology_key=topo_key,
                compile_time_seconds=time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.warning("modular compile failed: %s", exc)
            return ShaderArtifact(
                source="",
                uniforms={},
                topology_key=(),
                compile_time_seconds=time.perf_counter() - t0,
            )

    def _compile_pipeline(self, genome) -> ShaderArtifact:
        """Compile a PipelineGenome into a multi-pass shader artifact."""
        t0 = time.perf_counter()

        try:
            source, uniforms = self.factory.generate_pipeline(
                genome,
                brightness=1.0,
                feedback_alpha=0.85,
                history_depth=0,
            )

            topo_key = tuple(
                (
                    p.pass_type,
                    p.modules.field_type,
                    p.modules.warp,
                    p.modules.palette,
                    tuple(p.modules.post),
                    p.input_source,
                    p.blend_mode,
                    p.feedback_route,
                )
                for p in genome.passes
            )

            return ShaderArtifact(
                source=source,
                uniforms=uniforms,
                topology_key=topo_key,
                compile_time_seconds=time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.warning("pipeline compile failed: %s", exc)
            return ShaderArtifact(
                source="",
                uniforms={},
                topology_key=(),
                compile_time_seconds=time.perf_counter() - t0,
            )

    def topology_key(self, genome: IGenome) -> tuple:
        """Return a topology key for cache lookups without full compilation."""
        try:
            sig = genome.shader_signature()
            return hash(json.dumps(sig, sort_keys=True, default=str))
        except Exception:
            return hash(id(genome))
