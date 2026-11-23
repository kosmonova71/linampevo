"""genome_base.py

Single unified organism interface for the evolutionary renderer.

Both :class:`VisualGenome` (evo_visualizer.py) and :class:`ModularGenome`
(evo/modular_genome.py) subclass ``GenomeBase`` so the Population, the
FitnessEvaluator and the renderer never have to special-case a concrete
genome type. Anything that can mutate, clone, crossover and report a fitness
is a valid organism.

This removes a future failure point: previously ``Population`` was written
against ``VisualGenome`` methods while the renderer drove a ``ModularGenome``,
so the population layer was more advanced than the organism abstraction it
operated on.
"""

from __future__ import annotations

import abc
from collections import OrderedDict
from typing import Any

# Mutation weighting for the numeric/parametric genome (VisualGenome).
# Higher weight = more likely to be mutated. Palette/colour genes carry the
# most visible change, structural genes (symmetry) the least.
MUTATION_WEIGHTS: dict[str, int] = {
    "warp": 5,
    "feedback": 5,
    "field_type": 2,
    "colour_palette": 8,
    "rotation": 3,
    "symmetry": 1,
    # 3D / fake-raymarch genes
    "dimension": 3,
    "camera_speed": 4,
    "camera_orbit": 3,
    "camera_depth": 3,
    "camera_roll": 3,
    "depth_scale": 4,
    "ray_steps": 2,
    "fog_density": 3,
}


class GenomeBase(abc.ABC):
    """Unified contract for every evolved organism.

    Subclasses must provide real mutable attributes for the evolutionary
    trackers declared here (lineage, fitness, generation, parent_id,
    mutations, age, lifespan).
    """

    # --- Evolutionary trackers (concrete attribute types set by subclasses) ---
    lineage: str
    objectives: Any  # Objectives instance from evo_bio.pareto
    generation: int
    parent_id: str | None
    mutations: list

    # --- Lifespan / ageing ---
    age: int
    lifespan: int

    __slots__ = ()

    @abc.abstractmethod
    def mutate(self, strength: float | None = None, large: bool = False) -> None:
        """Mutate this organism in place.

        Args:
            strength: Optional override for the mutation step size.
            large: When True, apply a stronger / multi-step mutation (used when
                an organism outlives its lifespan to force succession).
        """

    @abc.abstractmethod
    def clone(self) -> GenomeBase:
        """Return a deep copy of this organism."""

    @abc.abstractmethod
    def crossover(self, other: GenomeBase) -> GenomeBase:
        """Return a child combining this organism with ``other``."""

    def distance(self, other: GenomeBase) -> float:
        """Return a normalized 0..1 phenotypic distance to another genome.

        Larger means more different. Default is neutral so a genome that does
        not implement a metric still participates in diversity pressure.
        """
        return 0.0

    def environmental_affinity(self, audio: dict[str, float]) -> float:
        """Return a bonus describing how well this genome suits the current
        audio environment.

        Audio is expected to be a dict with at least ``bass``/``treb`` keys in
        0..1 range. Default is neutral (0.0).
        """
        return 0.0

    def _age_mutation_factor(self) -> float:
        if self.lifespan <= 0:
            return 1.0
        ratio = self.age / self.lifespan
        if ratio < 0.25:
            return 2.5
        elif ratio < 0.75:
            return 0.3
        return 4.0

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
                except TypeError:
                    pass


class ShaderCache:
    """Caches compiled GL programs keyed by a genome's shader signature.

    Recompiling a fragment shader for every candidate in a population is the
    single biggest performance bottleneck of the evolution loop. Identical
    genomes (and repeated clones/crossovers) hit the same source, so we cache
    the compiled program and reuse it.

    The cache is *bounded* and **LRU**. When it is full, the least-recently-used
    entry (not the oldest by insertion) is evicted and its GL program is deleted
    via ``on_evict``. Every :meth:`get` refreshes an entry's recency so the
    programs an organism actually renders with survive pressure, while one-off
    scoring programs for discarded candidates get recycled first. Eviction
    never deletes a program that is currently bound (see :meth:`protect`), so
    the active champion program is safe. This is critical: leak #1 in the OOM
    post-mortem was the previous code evicting programs with ``del`` but never
    releasing the underlying GL object, which accumulated until the OOM killer
    fired.
    """

    def __init__(self, max_size: int = 256, on_evict=None):
        # OrderedDict gives us O(1) LRU: move_to_end on access, popitem(last=False)
        # for the least-recently-used entry.
        self.cache: OrderedDict[int, int] = OrderedDict()
        self.max_size = max(1, int(max_size))
        # Callable(program_id) used to release a GL program. GL-agnostic so this
        # module has no OpenGL dependency.
        self.on_evict = on_evict
        self._protected: int | None = None
        self._evictions = 0

    def protect(self, program: int | None) -> None:
        """Mark ``program`` as currently bound so eviction must skip it."""
        self._protected = int(program) if program is not None else None

    def unprotect(self) -> None:
        self._protected = None

    def key(self, genome: Any) -> int:
        """Stable integer key for a genome.

        Prefers ``genome.shader_signature()`` (only the parameters that affect
        the generated source, so the same phenotype reuses its compiled
        program) and falls back to the full serialized dict.

        Important: the key must NOT include lineage / generation / fitness /
        mutation history. Those change on every mutation and previously churned
        the cache, forcing a recompile (and an eventual eviction leak) each
        generation even when the shader was identical.
        """
        try:
            sig_factory = getattr(genome, "shader_signature", None)
            if sig_factory is not None:
                return hash(json_dumps(sig_factory()))
            return hash(json_dumps(genome.to_dict()))
        except TypeError:
            return id(genome)

    def get(self, key: int) -> int | None:
        prog = self.cache.get(key)
        if prog is not None:
            # Refresh recency: this program is being used right now, so it
            # should be the last one evicted under pressure.
            self.cache.move_to_end(key)
        return prog

    def store(self, key: int, program: int) -> None:
        if key in self.cache:
            # Same genome recompiled (e.g. re-accepted champion): release the
            # now-orphaned program unless it is the one currently bound.
            old = self.cache[key]
            self.cache[key] = program
            # Touch recency for the updated entry.
            self.cache.move_to_end(key)
            if old != program and old != self._protected and self.on_evict is not None:
                self.on_evict(old)
            return
        if len(self.cache) >= self.max_size:
            self._evict_one()
        self.cache[key] = program

    def _evict_one(self) -> None:
        # Drop the least-recently-used entry that is not currently bound.
        for k in list(self.cache.keys()):
            prog = self.cache[k]
            if prog == self._protected:
                continue
            del self.cache[k]
            self._evictions += 1
            if self.on_evict is not None:
                self.on_evict(prog)
            return
        # Every entry is protected (degenerate but safe): nothing to free.

    def programs(self):
        return list(self.cache.values())

    def stats(self) -> dict[str, int]:
        """Return cache size and total LRU eviction count for diagnostics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "evictions": self._evictions,
        }

    def trim_to(self, target: int) -> int:
        """Evict least-recently-used entries until ``len <= target``.

        Used by the proactive memory watchdog to reclaim GPU programs *before*
        the kernel OOM killer fires. Returns the number of programs freed.

        The currently-protected program is never evicted; if all entries are
        protected the method frees nothing and returns 0.
        """
        target = max(0, int(target))
        freed = 0
        while len(self.cache) > target:
            evicted = False
            for k in list(self.cache.keys()):
                prog = self.cache[k]
                if prog == self._protected:
                    continue
                del self.cache[k]
                self._evictions += 1
                freed += 1
                if self.on_evict is not None:
                    self.on_evict(prog)
                evicted = True
                break
            if not evicted:
                break
        return freed

    def clear(self) -> None:
        if self.on_evict is not None:
            for prog in self.cache.values():
                self.on_evict(prog)
        self.cache.clear()


def json_dumps(obj: Any) -> str:
    """json.dumps with stable key ordering and a tolerant default serializer."""
    import json

    return json.dumps(obj, sort_keys=True, default=str)
