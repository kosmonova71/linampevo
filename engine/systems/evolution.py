"""engine/systems/evolution.py

Decoupled evolution system.

This replaces the tightly-coupled EvolutionEngine from evo_renderer/evolution.py.
The EvolutionSystem:

  - Takes interfaces (population, fitness, compiler, audio, world, species)
    rather than reaching into renderer internals
  - Emits EvolutionEvents instead of directly mutating renderer state
  - Is independently testable with mock implementations
  - Supports true multi-pass render graph evolution

Evolution loop:
  1. Collect audio state from audio provider
  2. Score all genomes via fitness evaluator
  3. Select elites + parents
  4. Mutate/crossover to create children
  5. Compile children via shader compiler
  6. Accept/reject via guardian gate
  7. Emit events for external consumers to apply

External consumers (renderer, identity, researcher) subscribe to events
and apply approved changes themselves. The evolution system never touches
GPU state.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from .interfaces import (
    AudioState,
    EvolutionConfig,
    EvolutionEvent,
    IAudioProvider,
    IFitnessEvaluator,
    IGenome,
    IShaderCompiler,
    ISpeciesManager,
    IWorldSimulation,
)

logger = logging.getLogger(__name__)


class EvolutionSystem:
    """Decoupled evolution orchestrator.

    Does not know about OpenGL, GLFW, GTK, or any rendering framework.
    Emits EvolutionEvents that a renderer or other consumer can act on.

    Multi-pass render graph evolution:
      - Evolves PipelineGenome with per-pass input_source, blend_mode,
        feedback_route
      - The shader compiler generates GLSL that respects the graph topology
      - The render pipeline executes the graph via multi-FBO rendering
    """

    def __init__(
        self,
        population: Any,  # IPopulation - avoiding Protocol isinstance checks
        fitness_evaluator: IFitnessEvaluator,
        shader_compiler: IShaderCompiler,
        audio_provider: IAudioProvider,
        world_simulation: IWorldSimulation | None = None,
        species_manager: ISpeciesManager | None = None,
        config: EvolutionConfig | None = None,
    ):
        self.population = population
        self.fitness_evaluator = fitness_evaluator
        self.shader_compiler = shader_compiler
        self.audio_provider = audio_provider
        self.world_simulation = world_simulation
        self.species_manager = species_manager
        self.config = config or EvolutionConfig()

        self.elapsed_time: float = 0.0
        self._was_silent: bool = False
        self._silent_accum: float = 0.0

        self._listeners: list[Any] = []
        self._failed_signatures: set = set()
        self._last_event: EvolutionEvent | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, dt: float) -> list[EvolutionEvent]:
        """Run one evolution tick.

        Returns a list of EvolutionEvents for consumers to act on.
        Consumers (renderer, identity, researcher) subscribe via add_listener
        or process the returned events directly.
        """
        dt = max(0.0, min(dt, 0.1))
        self.elapsed_time += dt

        audio = self.audio_provider.get_state()
        events: list[EvolutionEvent] = []

        # Silence detection
        avg_energy = (
            float(getattr(audio, "bass", 0.0))
            + float(getattr(audio, "mid", 0.0))
            + float(getattr(audio, "treble", 0.0))
        ) / 3.0

        if avg_energy < self.config.silence_threshold:
            self._silent_accum += dt
        else:
            self._silent_accum = 0.0

        is_dream = self._silent_accum >= self.config.silence_hold_secs
        just_woke = self._was_silent and not is_dream
        self._was_silent = is_dream

        if just_woke:
            return events

        # Continuous evolution every frame.
        if is_dream:
            for _ in range(self.config.dream_evolutions_per_tick):
                ev = self._evolve_once(audio=audio, dream=True)
                if ev:
                    events.append(ev)
        else:
            ev = self._evolve_once(audio=audio, dream=False)
            if ev:
                events.append(ev)

        # Notify listeners
        for ev in events:
            self._notify_listeners(ev)

        return events

    def add_listener(self, listener: Any) -> None:
        """Add an event listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Any) -> None:
        """Remove an event listener."""
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def get_champion(self) -> IGenome | None:
        """Return the current champion genome."""
        return self.population.get_champion()

    def get_population_size(self) -> int:
        return self.population.size()

    def set_population(self, genomes: list[IGenome]) -> None:
        """Replace the entire population."""
        self.population.replace(genomes)

    def propose_mutation(self, genome: IGenome, mutation_id: str) -> IGenome:
        """Create a mutated copy of genome for the caller to evaluate."""
        child = genome.clone()
        child.mutate(strength=self.config.mutation_strength)
        return child

    def record_feedback(self, genome: IGenome, accepted: bool) -> None:
        """Record whether a genome mutation was accepted."""
        try:
            genome.record_accept(accepted)
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Internal evolution logic
    # ------------------------------------------------------------------

    def _evolve_once(self, audio: AudioState, dream: bool = False) -> EvolutionEvent | None:
        """Run a single evolution step.

        1. Score current champion
        2. Select parents
        3. Create child via mutation/crossover
        4. Compile child shader
        5. Score child
        6. Accept or reject via guardian logic
        7. Emit event
        """
        parent = self.population.get_champion()
        if parent is None:
            return None

        # Score parent
        before = self._score_genome(parent, audio)

        # Create child
        child = self._create_child(parent, audio)

        # Compile child
        artifact = self.shader_compiler.compile(child)
        if not artifact.source:
            return EvolutionEvent(
                event_type="rejected",
                genome=child,
                score=before,
                delta=0.0,
                metadata={"reason": "compile_failed"},
            )

        # Score child
        after = self._score_genome(child, audio, artifact=artifact)

        # Guardian acceptance
        delta = after - before
        accepted = self._guardian_approve(after, delta, before)

        if accepted:
            self.population.record_fitness(child, after)
            self.population.add(child)

            # Species tracking
            if self.species_manager is not None:
                try:
                    self.species_manager.record(child, after)
                except Exception as exc:
                    logger.debug("Species tracking failed: %s", exc)

            event_type = "dream" if dream else "evolved"
            return EvolutionEvent(
                event_type=event_type,
                genome=child,
                score=after,
                delta=delta,
                metadata={
                    "mutation_strength": self.config.mutation_strength,
                    "topology_changed": True,
                },
            )
        else:
            self.record_feedback(child, accepted=False)
            return EvolutionEvent(
                event_type="rejected",
                genome=child,
                score=before,
                delta=delta,
                metadata={"reason": "guardian_rejected"},
            )

    def _score_genome(
        self,
        genome: IGenome,
        audio: AudioState,
        *,
        artifact: Any | None = None,
        concept: Any = None,
    ) -> float:
        """Score a genome using the fitness evaluator."""
        novelty = None
        if concept is not None and hasattr(self.fitness_evaluator, "concept_archive"):
            try:
                concept_id = concept.stable_id() if hasattr(concept, "stable_id") else None
                novelty_res = self.fitness_evaluator.concept_archive.novelty(
                    concept, concept_id=concept_id
                )
                novelty = novelty_res.novelty
            except Exception as exc:
                logger.debug("Novelty scoring failed: %s", exc)
                novelty = None

        gpu_cost = 0.0
        if artifact is not None and hasattr(artifact, "compile_time_seconds"):
            gpu_cost = float(artifact.compile_time_seconds)

        return self.fitness_evaluator.evaluate(
            genome=genome,
            audio=audio,
            frame=self.elapsed_time,
            novelty=novelty,
            gpu_cost=gpu_cost,
        )

    def _create_child(self, parent: IGenome, audio: AudioState) -> IGenome:
        """Create a mutated child from a parent genome."""
        child = parent.clone()

        # Audio-driven mutation pressure
        strength = self.config.mutation_strength
        try:
            pressure = self.audio_provider.get_audio_pressure()
            if pressure.get("is_silent", False):
                strength *= 1.5  # Explore more during silence
            elif pressure.get("energy", 0.0) > 0.7:
                strength *= 0.7  # Be conservative during intense audio
        except Exception as exc:
            logger.debug("Audio pressure fetch failed: %s", exc)

        # Environmental affinity
        try:
            affinity = parent.environmental_affinity(
                {
                    "bass": float(getattr(audio, "bass", 0.0)),
                    "treb": float(getattr(audio, "treble", 0.0)),
                }
            )
            strength += affinity * 0.1
        except Exception as exc:
            logger.debug("Environmental affinity failed: %s", exc)

        child.mutate(strength=float(max(0.01, min(1.0, strength))))
        return child

    def _guardian_approve(self, score: float, delta: float, parent_score: float) -> bool:
        """Simple guardian gate: accept improvements above a threshold."""
        if score <= 0.0:
            return False
        if delta > 0.0:
            return True
        # Accept small regressions occasionally for exploration
        return random.random() < 0.15

    def _notify_listeners(self, event: EvolutionEvent) -> None:
        """Notify all registered listeners of an evolution event."""
        self._last_event = event
        for listener in self._listeners:
            try:
                listener.on_evolution_event(event)
            except Exception as exc:
                logger.debug("Listener notification failed: %s", exc)
