"""engine/systems/interfaces.py

Protocol/ABC interfaces for the decoupled evolution architecture.

Each system is independently testable and swappable. The renderer is just a
view that consumes system outputs. Evolution no longer reaches into the
renderer's internals.

Multi-pass render graph evolution:

    A PipelineGenome now describes a full render graph:
      - pass ordering (sequence)
      - per-pass input_source (previous, feedback, first, custom)
      - per-pass blend_mode (alpha, add, multiply, screen, overlay)
      - per-pass feedback_route (none, self, previous, first)
      - feedback routing between passes

    This lets evolution invent passes, orderings, blending, and feedback
    routing rather than only parameters.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Genome interface (extends GenomeBase from evo.genome_base)
# ---------------------------------------------------------------------------


class IGenome(abc.ABC):
    """Unified contract for every evolved organism.

    All genome types used by the system layer must satisfy this interface.
    """

    lineage: str
    fitness: float
    generation: int
    parent_id: str | None
    mutations: list
    age: int
    lifespan: int

    @abc.abstractmethod
    def mutate(self, strength: float | None = None, large: bool = False) -> None:
        """Mutate this organism in place."""

    @abc.abstractmethod
    def clone(self) -> IGenome:
        """Return a deep copy of this organism."""

    @abc.abstractmethod
    def crossover(self, other: IGenome) -> IGenome:
        """Return a child combining this organism with another."""

    def distance(self, other: IGenome) -> float:
        return 0.0

    def environmental_affinity(self, audio: dict[str, float]) -> float:
        return 0.0

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
                except Exception:
                    pass

    def shader_signature(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Shader artifact
# ---------------------------------------------------------------------------


@dataclass
class ShaderArtifact:
    """Output of shader compilation.

    Attributes:
        source: Complete GLSL fragment shader source.
        uniforms: Map of uniform_name -> {"value", "kind", "field", "key"}.
        topology_key: Hashable key identifying the shader topology. Two
            artifacts with the same topology_key differ only in numeric
            parameters, so the renderer can avoid recompiling.
        compile_time_seconds: Time taken to compile (for cost estimation).
    """

    source: str = ""
    uniforms: dict[str, dict[str, Any]] = field(default_factory=dict)
    topology_key: tuple = ()
    compile_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Render pass configuration (multi-pass render graph)
# ---------------------------------------------------------------------------


@dataclass
class RenderPassConfig:
    """Configuration for a single render pass in the graph.

    Attributes:
        pass_id: Unique identifier for this pass (e.g. "field_0").
        pass_type: Type tag: field, lighting, feedback, post, particle,
            composite, reaction_diffusion, flow_field.
        input_source: Where this pass reads its input texture from.
            - "previous": the output of the immediately preceding pass
            - "feedback": the ping-pong feedback buffer
            - "first": the original input (first pass output, never overwritten)
            - "none": no input (generates from scratch)
        blend_mode: How this pass blends its output with its input.
            - "alpha": alpha blend with input
            - "add": additive blend
            - "multiply": multiply blend
            - "screen": screen blend
            - "overlay": overlay blend
            - "replace": replace input entirely (default for most passes)
        feedback_route: Whether and how this pass feeds back to itself or others.
            - "none": no feedback
            - "self": this pass reads its own previous output
            - "previous": feeds back to the previous pass
            - "first": feeds back to the first pass in the chain
        modules: The ModularGenome for this pass's shader logic.
        enabled: Whether this pass is active.
    """

    pass_id: str = ""
    pass_type: str = "field"
    input_source: str = "previous"
    blend_mode: str = "replace"
    feedback_route: str = "none"
    modules: Any = None  # ModularGenome
    enabled: bool = True


# ---------------------------------------------------------------------------
# Evolution configuration and events
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    """Configuration for the evolution system.

    Attributes:
        population_size: Number of genomes in the active population.
        elite_frac: Fraction of top performers kept unchanged.
        mutation_strength: Base mutation step size.
        silence_threshold: Average audio energy below which dream mode activates.
        silence_hold_secs: Seconds of silence before dream mode.
        dream_evolutions_per_tick: Number of evolution steps per tick in dream mode.
        enable_species: Whether to track species formation.
        enable_researcher: Whether to run the autonomous researcher.
        enable_identity: Whether to use the identity engine.
        enable_audio_pressure: Whether to bias mutations from audio analysis.
        gpu_cost_penalty: Weight of GPU cost in fitness scoring.
    """

    population_size: int = 3
    elite_frac: float = 0.25
    mutation_strength: float = 0.12
    silence_threshold: float = 0.08
    silence_hold_secs: float = 5.0
    dream_evolutions_per_tick: int = 2
    enable_species: bool = True
    enable_researcher: bool = False
    enable_identity: bool = False
    enable_audio_pressure: bool = False
    gpu_cost_penalty: float = 0.25


@dataclass
class EvolutionEvent:
    """Event emitted by the evolution system after a tick.

    Attributes:
        event_type: One of "evolved", "dream", "champion", "rejected",
            "species_new", "researcher_theory".
        genome: The genome involved (may be None for some event types).
        score: Fitness score (may be None).
        delta: Score change from parent (may be None).
        metadata: Arbitrary extra data.
    """

    event_type: str = "evolved"
    genome: IGenome | None = None
    score: float | None = None
    delta: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# System interfaces (Protocols for duck typing)
# ---------------------------------------------------------------------------


class IPopulation(Protocol):
    """Population management interface."""

    def get_champion(self) -> IGenome | None:
        """Return the highest-fitness genome, or None."""
        ...

    def select_parents(self, n: int) -> list[IGenome]:
        """Return n parent genomes selected by fitness."""
        ...

    def replace(self, new_genomes: list[IGenome]) -> None:
        """Replace the population with new genomes."""
        ...

    def add(self, genome: IGenome) -> None:
        """Add a genome to the population."""
        ...

    def remove(self, genome: IGenome) -> None:
        """Remove a genome from the population."""
        ...

    def size(self) -> int:
        """Return the current population size."""
        ...

    def all(self) -> list[IGenome]:
        """Return all genomes in the population."""
        ...


class IFitnessEvaluator(Protocol):
    """Fitness evaluation interface."""

    def evaluate(
        self,
        genome: IGenome,
        audio: AudioState,
        frame: float,
        *,
        novelty: float | None = None,
        gpu_cost: float = 0.0,
    ) -> float:
        """Score a genome given current audio and time context."""
        ...


class IShaderCompiler(Protocol):
    """Shader compilation interface."""

    def compile(self, genome: IGenome) -> ShaderArtifact:
        """Compile a genome into a shader artifact.

        Returns a ShaderArtifact with GLSL source, uniform metadata, and
        a topology key for cache lookups.
        """
        ...

    def compile_pass(
        self, pass_config: RenderPassConfig, global_params: dict[str, Any]
    ) -> ShaderArtifact:
        """Compile a single pass's shader code (for multi-pass rendering)."""
        ...


class IRenderPipeline(Protocol):
    """Render pipeline interface.

    Abstracts the GPU rendering so evolution can propose render graphs
    without knowing OpenGL details.
    """

    def execute_graph(self, passes: list[RenderPassConfig]) -> None:
        """Execute the full multi-pass render graph."""
        ...

    def get_feedback_texture(self) -> int:
        """Return the OpenGL texture ID of the current feedback buffer."""
        ...

    def resize_feedback(self, resolution: int) -> None:
        """Resize the feedback buffer (called when genome requests it)."""
        ...

    def upload_uniforms(self, uniforms: dict[str, dict[str, Any]]) -> None:
        """Upload uniform values to the active shader program."""
        ...


class IAudioProvider(Protocol):
    """Audio state provider interface."""

    def get_state(self) -> AudioState:
        """Return the current smoothed audio state."""
        ...


class IWorldSimulation(Protocol):
    """World / physics simulation interface."""

    def step(self, dt: float, audio: AudioState) -> None:
        """Advance the simulation by dt seconds."""
        ...

    def get_summaries(self) -> dict[str, float]:
        """Return scalar summaries for shader uniforms."""
        ...


class ISpeciesManager(Protocol):
    """Species management interface."""

    def classify(self, genome: IGenome) -> str:
        """Return a species identifier for the genome."""
        ...

    def record(self, genome: IGenome, fitness: float) -> None:
        """Record a genome observation for species tracking."""
        ...

    def get_species_summary(self) -> dict[str, int]:
        """Return counts of genomes per species."""
        ...


# ---------------------------------------------------------------------------
# Audio state (used across all systems)
# ---------------------------------------------------------------------------


class AudioState:
    """Smoothed audio features for reactive rendering.

    Lightweight dataclass so it can be constructed from dicts for testing.
    """

    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    energy: float = 0.0
    beat: float = 0.0

    def __init__(
        self,
        bass: float = 0.0,
        mid: float = 0.0,
        treble: float = 0.0,
        energy: float = 0.0,
        beat: float = 0.0,
    ):
        self.bass = max(0.0, min(2.0, float(bass)))
        self.mid = max(0.0, min(2.0, float(mid)))
        self.treble = max(0.0, min(2.0, float(treble)))
        self.energy = max(0.0, min(2.0, float(energy)))
        self.beat = max(0.0, min(2.0, float(beat)))

    def set_levels(self, levels: dict) -> None:
        smoothing = float(levels.get("smoothing", 0.85))
        raw_bass = max(0.0, min(2.0, float(levels.get("bass", 0.0))))
        raw_mid = max(0.0, min(2.0, float(levels.get("mid", 0.0))))
        raw_treble = max(0.0, min(2.0, float(levels.get("treble", levels.get("treb", 0.0)))))
        raw_beat = max(0.0, min(2.0, float(levels.get("beat", 0.0))))
        self.bass = self.bass * smoothing + raw_bass * (1.0 - smoothing)
        self.mid = self.mid * smoothing + raw_mid * (1.0 - smoothing)
        self.treble = self.treble * smoothing + raw_treble * (1.0 - smoothing)
        self.energy = self.bass * 0.5 + self.mid * 0.3 + self.treble * 0.2
        beat_smoothing = 0.80
        self.beat = self.beat * beat_smoothing + raw_beat * (1.0 - beat_smoothing)
        self.bass = max(0.0, min(2.0, self.bass))
        self.mid = max(0.0, min(2.0, self.mid))
        self.treble = max(0.0, min(2.0, self.treble))
        self.energy = max(0.0, min(2.0, self.energy))
        self.beat = max(0.0, min(2.0, self.beat))
