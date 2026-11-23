from engine.audio.analyzer import AudioAnalyzer, AudioFeatures
from engine.compiler.compiler_v9 import V9Compiler
from engine.gl.async_compiler import AsyncShaderCompiler
from engine.gl.glsl_generator import GLSLGenerator, ir_to_glsl
from engine.gl.passes import MultiPassRenderer, PassType, RenderPass
from engine.gl.shader_cache import ShaderCache, get_shader_cache
from engine.ir.lowering import lower_to_ssa
from engine.ir.ssa import IRBlock, IRBuilder, IRInstruction, IROptimizer, hash_ir
from engine.parser.parser import ASTTransformer, V9Parser
from engine.parser.tokenizer import V9Tokenizer
from engine.runtime.executor_cpu import CPUJITExecutor
from engine.runtime.executor_gpu import GPUExecutor
from engine.systems.audio_system import AudioSystem
from engine.systems.evolution import EvolutionSystem
from engine.systems.fitness import FitnessEvaluator

# Decoupled system architecture
from engine.systems.interfaces import (
    AudioState,
    EvolutionConfig,
    EvolutionEvent,
    IAudioProvider,
    IFitnessEvaluator,
    IGenome,
    IPopulation,
    IRenderPipeline,
    IShaderCompiler,
    ISpeciesManager,
    IWorldSimulation,
    RenderPassConfig,
    ShaderArtifact,
)
from engine.systems.population import Population
from engine.systems.render_pipeline import SystemRenderPipeline
from engine.systems.shader_compiler import SystemShaderCompiler
from engine.systems.species_manager import SpeciesManager
from engine.systems.world_simulation import WorldSimulation

__all__ = [
    "ASTTransformer",
    "AsyncShaderCompiler",
    # Audio
    "AudioAnalyzer",
    "AudioFeatures",
    "AudioState",
    "AudioSystem",
    # Runtime
    "CPUJITExecutor",
    "EvolutionConfig",
    "EvolutionEvent",
    "EvolutionSystem",
    "FitnessEvaluator",
    # GL
    "GLSLGenerator",
    "GPUExecutor",
    "IAudioProvider",
    "IFitnessEvaluator",
    # Decoupled systems
    "IGenome",
    "IPopulation",
    # IR
    "IRBlock",
    "IRBuilder",
    "IRInstruction",
    "IROptimizer",
    "IRenderPipeline",
    "IShaderCompiler",
    "ISpeciesManager",
    "IWorldSimulation",
    "MultiPassRenderer",
    "PassType",
    "Population",
    "RenderPass",
    "RenderPassConfig",
    "ShaderArtifact",
    "ShaderCache",
    "SpeciesManager",
    "SystemRenderPipeline",
    "SystemShaderCompiler",
    # Compiler
    "V9Compiler",
    "V9Parser",
    # Parser
    "V9Tokenizer",
    "WorldSimulation",
    "get_shader_cache",
    "hash_ir",
    "ir_to_glsl",
    "lower_to_ssa",
]
