"""Identity engine — glues VisualIdentity into the renderer/evolution pipeline.
This is the entry point for Stage 12 integration. The EvolutionEngine creates
an IdentityEngine and ticks it each frame. The identity engine then feeds
behavioural parameters back into the evolution loop.

The identity engine is intentionally a *thin* façade: almost every method
forwards to :class:`.state.VisualIdentity`.  All audio boundary normalisation
(accepting both ``treble`` and ``treb``) lives here so the rest of the package
can assume a single attribute name.
"""

from __future__ import annotations

import logging
from typing import Any

from .state import IdentityGenome, VisualIdentity

logger = logging.getLogger(__name__)


def _audio_energy(audio: Any) -> float:
    """Return a 0-1-ish energy level from an audio analyse object.

    Accepts both ``treble`` and ``treb`` attribute names -- the renderer's
    audio dict uses ``treb`` while the rest of the codebase uses ``treble``.
    """
    bass = float(getattr(audio, "bass", 0.0) or 0.0)
    mid = float(getattr(audio, "mid", 0.0) or 0.0)
    treble = float(getattr(audio, "treble", getattr(audio, "treb", 0.0)) or 0.0)
    return (bass + mid + treble) / 3.0


class IdentityEngine:
    """Bridge between the renderer's evolution loop and the persistent identity.
    Responsibilities:
      - Tick identity (age, mood, goals, memory) each frame.
      - Translate identity mood/goals into mutation parameters.
      - Run dream simulations during silence/idle.
      - Expose lineage and preference summaries.
      - Persist identity state periodically.
    """

    def __init__(
        self,
        renderer: Any,
        state_path: str | None = None,
        max_memory_events: int = 10000,
        dream_interval_secs: float = 8.0,
        dream_max_steps: int = 20,
        persist_interval_secs: float = 30.0,
    ):
        self.renderer = renderer
        self.identity = VisualIdentity(
            state_path=state_path,
            max_memory_events=max_memory_events,
        )
        self.dream_interval_secs = float(dream_interval_secs)
        self.dream_max_steps = int(dream_max_steps)
        self.persist_interval_secs = float(persist_interval_secs)
        # Silence accumulation is the *single* dream-gating timer.  The old
        # ``_dream_timer`` was incremented and never read -- dead state.
        self._silence_accum = 0.0
        self.silence_threshold = 0.08
        self._persist_timer = 0.0

    def update(self, dt: float, audio: Any | None = None) -> dict[str, Any]:
        self.identity.tick(dt, audio=audio)

        if audio is not None:
            avg_energy = _audio_energy(audio)
        else:
            avg_energy = 0.0

        if avg_energy < self.silence_threshold:
            self._silence_accum += float(dt)
        else:
            if self._silence_accum > 0.0:
                self.identity.apply_silence(self._silence_accum)
            self._silence_accum = 0.0

        # A single silence-accumulation gate fires the dream.  No separate
        # wall-clock timer to keep in sync.
        if self._silence_accum >= self.dream_interval_secs:
            self._run_dream()

        self._persist_timer += float(dt)
        if self._persist_timer >= self.persist_interval_secs:
            self._persist_timer = 0.0
            self.identity.save()

        return self.identity.summary()

    def record_mutation(self, mutation_id: str, before: float, after: float) -> None:
        self.identity.record_experience(mutation_id, before, after)

    def record_champion(self, genome: Any, score: float, parent: Any | None = None) -> None:
        self.identity.record_champion(genome, score, parent)

    def record_modules(self, modules: list[str], delta: float) -> None:
        self.identity.record_modules(modules, delta)

    def human_feedback(self, sentiment: str) -> None:
        self.identity.apply_human_feedback(sentiment)

    def identity_mutation_strength(self) -> float:
        return self.identity.identity_mutation_strength()

    def visual_mutation_strength(self) -> float:
        return self.identity.mutation_strength()

    def exploration_probability(self) -> float:
        return self.identity.exploration_probability()

    def prefer_known_species(self) -> bool:
        return self.identity.prefer_known_species()

    def should_explore_unknown(self) -> bool:
        return self.identity.should_explore_unknown()

    def identity_genome(self) -> IdentityGenome:
        return self.identity.identity_genome

    def mutate_identity_genome(self) -> None:
        self.identity.identity_genome.mutate(strength=self.identity.identity_mutation_strength())

    def consume_dreams(self) -> list[dict[str, Any]]:
        return self.identity.consume_dreams()

    def evolution_params(self) -> dict[str, Any]:
        """Return the identity's behavioural parameters for one evolution step.

        The renderer's evolution loop should call this once per generation
        (or per frame) and feed the returned values into its mutation /
        selection step.  Without this, the identity is a passive log: it
        records history but never influences it.

        Returns:
            A dict with keys ``mutation_strength``, ``exploration``,
            ``prefer_known``, ``explore_unknown``, and
            ``identity_mutation_strength``.
        """
        return {
            "mutation_strength": self.identity.mutation_strength(),
            "exploration": self.identity.exploration_probability(),
            "prefer_known": self.identity.prefer_known_species(),
            "explore_unknown": self.identity.should_explore_unknown(),
            "identity_mutation_strength": self.identity.identity_mutation_strength(),
        }

    def apply_evolution_outcome(
        self,
        mutation_id: str,
        before: float,
        after: float,
        *,
        modules: list[str] | None = None,
        delta: float | None = None,
    ) -> None:
        """Record the outcome of one evolution step.

        Convenience wrapper that records both the mutation experience (for
        preference learning) and the module usage.  The renderer should call
        this after each mutation it applies, so the identity can learn which
        operators help.

        Args:
            mutation_id: Identifier of the mutation operator used.
            before: Fitness before the mutation.
            after: Fitness after the mutation.
            modules: Active modules on the mutated genome.
            delta: Optional explicit fitness delta (defaults to
                ``after - before``).
        """
        self.identity.record_experience(mutation_id, before, after)
        if modules is not None:
            self.identity.record_modules(
                list(modules), float(delta) if delta is not None else (after - before)
            )

    def record_hall_of_fame(
        self,
        genome: Any,
        score: float,
        lineage_id: str | None = None,
        modules: list[str] | None = None,
        parent: Any | None = None,
    ) -> None:
        self.identity.record_hall_of_fame(
            genome,
            score,
            lineage_id=lineage_id,
            modules=modules,
            parent=parent,
        )

    def record_museum_specimen(
        self,
        genome: Any,
        score: float,
        tags: list[str] | None = None,
        notes: str = "",
        lineage_id: str | None = None,
    ) -> None:
        self.identity.record_museum_specimen(
            genome,
            score,
            tags=tags,
            notes=notes,
            lineage_id=lineage_id,
        )

    def record_ancestor_lineage(self, lineage_id: str, chain: list[dict[str, Any]]) -> None:
        self.identity.record_ancestor_lineage(lineage_id, chain)

    def archive_extinct_species(
        self,
        species_id: str,
        genome: Any,
        score: float,
        cause: str = "unknown",
        lineage_id: str | None = None,
        generation: int = 0,
        modules: list[str] | None = None,
    ) -> None:
        self.identity.archive_extinct_species(
            species_id,
            genome,
            score,
            cause=cause,
            lineage_id=lineage_id,
            generation=generation,
            modules=modules,
        )

    def replay_ancestors(self, lineage_id: str) -> list[dict[str, Any]]:
        return self.identity.replay_ancestors(lineage_id)

    def get_evolution_tree(self, lineage_id: str | None = None) -> dict[str, Any]:
        return self.identity.get_evolution_tree(lineage_id)

    def find_common_ancestor(self, lineage_a: str, lineage_b: str) -> dict[str, Any] | None:
        return self.identity.find_common_ancestor(lineage_a, lineage_b)

    def extinct_lineage_chain(self, lineage_id: str) -> list[dict[str, Any]]:
        return self.identity.extinct_lineage_chain(lineage_id)

    def summary(self) -> dict[str, Any]:
        return self.identity.summary()

    def save(self) -> None:
        self.identity.save()

    def _run_dream(self) -> None:
        try:
            self.identity.dream(self.renderer, max_steps=self.dream_max_steps)
        except Exception as exc:  # dream simulation may raise anything
            logger.debug("Dream simulation failed: %s", exc)
        finally:
            self._silence_accum = 0.0
