"""Multi-resolution evolution: a pyramid of evolving populations at different resolutions.

Each tier runs its own Population and FitnessEvaluator backed by a small
dedicated fitness FBO.  Successful organisms are periodically promoted upward
to higher-resolution tiers; poor performers are demoted or replaced.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from typing import TYPE_CHECKING, Any

from OpenGL import GL

from evo_visualizer.config import MultiResTier
from evo_visualizer.constants import QUAD_VERTEX_COUNT

if TYPE_CHECKING:
    from evo_renderer import EvoRenderer

from evo_bio.pareto import Objectives
from evo_gpu_fitness import (
    OUT_BEAUTY,
    OUT_COVERAGE,
    OUT_ENERGY,
    OUT_ENTROPY,
    OUT_MOTION,
    OUT_NOVELTY,
    OUT_SYMMETRY,
    GpuFitnessPipeline,
)

logger = logging.getLogger(__name__)


class MultiResolutionEvolution:
    """Manages a pyramid of evolving populations at different resolutions.

    Each tier runs its own :class:`Population` and :class:`FitnessEvaluator`
    backed by a small dedicated fitness FBO.  Successful organisms are
    periodically promoted upward to higher-resolution tiers; poor performers
    are demoted or replaced.  Because fitness-readback cost scales linearly
    with pixel count, running the exploration tier at 32x32 instead of
    512x512 yields roughly a 256x speed-up in evaluation per organism.
    """

    def __init__(self, renderer: EvoRenderer, init_gl: bool = True):
        from evo_visualizer.population import Population

        self.renderer = renderer
        self._frame = 0
        cfg = renderer._config
        self.tiers: list[MultiResTier] = list(getattr(cfg, "res_tiers", []))
        if not self.tiers:
            self.tiers = [
                MultiResTier(
                    name="exploration",
                    res=32,
                    population_size=24,
                    mutation_mult=2.0,
                    promote_interval=120,
                ),
                MultiResTier(
                    name="normal",
                    res=128,
                    population_size=8,
                    mutation_mult=1.0,
                    promote_interval=180,
                ),
                MultiResTier(
                    name="champions",
                    res=512,
                    population_size=2,
                    mutation_mult=0.5,
                    promote_interval=300,
                ),
            ]
        self.populations: dict[str, Population] = {}
        self._fitness_pipelines: dict[str, GpuFitnessPipeline] = {}
        self._fitness_fbos: dict[str, tuple[int, int]] = {}
        self._previous_fitness_tex: dict[str, int] = {}
        self._gl_initialized: bool = False
        self._lock = __import__("threading").RLock()

        champion = renderer.genome
        for tier in self.tiers:
            _genome_type = type(renderer.genome)
            pop = Population(
                size=tier.population_size,
                crossover_rate=cfg.crossover_rate,
                genome_factory=lambda _gt=_genome_type: _gt(),
                elite_fraction=getattr(cfg, "elite_fraction", 0.15),
                tournament_size=getattr(cfg, "tournament_size", 4),
            )
            for i, org in enumerate(pop.organisms):
                if i == 0:
                    org = champion.clone()
                else:
                    org = champion.clone()
                    org.mutate(strength=0.15)
                    org.compute_entropy()
                pop.organisms[i] = org
            self.populations[tier.name] = pop

        if init_gl:
            self.init_gl()

    def init_gl(self) -> None:
        """Allocate per-tier fitness FBOs and GPU fitness pipelines.

        Must be called from inside a live GL context (after ``make_current``
        in :meth:`EvoRenderer.on_realize`).  Safe to call only once;
        subsequent calls are no-ops.
        """
        if self._gl_initialized:
            return
        for tier in self.tiers:
            try:
                fbo, tex = self._alloc_fbo(tier.res)
                self._fitness_fbos[tier.name] = (fbo, tex)
                pipeline = GpuFitnessPipeline(res=tier.res)
                pipeline.init_gl()
                self._fitness_pipelines[tier.name] = pipeline
                self._previous_fitness_tex[tier.name] = 0
            except Exception as exc:
                logger.debug("Multi-res FBO alloc failed for %s: %s", tier.name, exc)
        self._gl_initialized = True

    def _alloc_fbo(self, res: int) -> tuple[int, int]:
        fbo = GL.glGenFramebuffers(1)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA16F,
            res,
            res,
            0,
            GL.GL_RGBA,
            GL.GL_HALF_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            tex,
            0,
        )
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Multi-res FBO incomplete: {hex(status)}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return int(fbo), int(tex)

    def step(self) -> None:
        self._frame += 1
        renderer = self.renderer
        t = time.time() - renderer._start_time
        bass, mid, treb, beat = renderer.audio.uvec()

        for idx, tier in enumerate(self.tiers):
            interval = max(1, 4 // (idx + 1))
            if self._frame % interval != 0:
                continue
            self._evolve_tier(tier, t, bass, mid, treb, beat)

        if self._frame % max(1, self.tiers[-1].promote_interval) == 0:
            self._promote_adjacent()

    def _evolve_tier(
        self,
        tier: MultiResTier,
        t: float,
        bass: float,
        mid: float,
        treb: float,
        beat: float,
    ) -> None:
        pop = self.populations.get(tier.name)
        if pop is None:
            return

        fbo, tex = self._fitness_fbos.get(tier.name, (0, 0))
        if fbo == 0:
            return

        organisms = pop.organisms
        n = len(organisms)
        if n == 0:
            return

        with self._lock:
            n_mut = max(1, int(2 * tier.mutation_mult))
            candidates = []
            for _ in range(n_mut):
                parent = random.choice(organisms)
                cand = parent.clone()
                cand.mutate(strength=0.03 * tier.mutation_mult)
                candidates.append(cand)

            scored: list[tuple[float, Any]] = []
            for cand in candidates:
                self._render_async(cand, tier, fbo, tex, t, bass, mid, treb, beat)
                # Sync and score immediately after each render so the
                # metrics are read before the next candidate overwrites
                # the shared pipeline FBO / readback buffers.
                self._sync_gpu()
                obj = self._collect_and_score(tier, cand)
                scored.append((obj.scalar(), cand))

            if not scored:
                return

            # Flush every outstanding GL draw before the final scoring /
            # replacement loop reads back fitness metrics and mutates the
            # population.  _sync_gpu() fences per-candidate, but a final
            # barrier guarantees the last candidate's dispatch is visible
            # before any readback below.
            try:
                GL.glFinish()
            except Exception:
                logger.debug("glFinish failed in _evolve_tier")

            scored.sort(key=lambda item: item[0], reverse=True)
            keep = max(1, int(n * 0.5))
            # Ascending sort: weakest organisms come first. Intentional — we
            # pair each top-scoring candidate with the weakest organism so the
            # strongest candidate can replace it via Pareto dominance.
            ranked = sorted(
                organisms,
                key=lambda org: float(
                    getattr(org, "objectives", Objectives()).scalar()
                    if getattr(org, "objectives", None) is not None
                    else 0.0
                ),
            )
            for candidate, weakest in zip(scored[:keep], ranked):
                candidate_genome = candidate[1]
                candidate_obj = getattr(candidate_genome, "objectives", None)
                if candidate_obj is None or not isinstance(candidate_obj, Objectives):
                    candidate_obj = Objectives()
                    candidate_genome.objectives = candidate_obj
                weakest_obj = getattr(weakest, "objectives", None)
                if weakest_obj is None or not isinstance(weakest_obj, Objectives):
                    weakest_obj = Objectives()
                if candidate_obj.dominates(weakest_obj):
                    weakest.objectives = candidate_obj
                    if hasattr(weakest, "params") and hasattr(candidate[1], "params"):
                        weakest.params = copy.deepcopy(candidate[1].params)
                    elif hasattr(weakest, "field_type"):
                        self.renderer._copy_phenotype(weakest, candidate[1])

            for org in organisms:
                if hasattr(org, "compute_entropy"):
                    org.compute_entropy()

    def _render_async(
        self,
        genome,
        tier: MultiResTier,
        fbo: int,
        tex: int,
        t: float,
        bass: float,
        mid: float,
        treb: float,
        beat: float,
    ) -> None:
        renderer = self.renderer
        res = tier.res
        prog = renderer._get_program(genome, allow_async=False)
        if not prog:
            return

        gl_state = renderer._snapshot_gl_state()
        old_locs = {
            "_loc_time": renderer._loc_time,
            "_loc_res": renderer._loc_res,
            "_loc_quality": getattr(renderer, "_loc_quality", -1),
            "_loc_bass": renderer._loc_bass,
            "_loc_mid": renderer._loc_mid,
            "_loc_treble": renderer._loc_treble,
            "_loc_previous_frame": renderer._loc_previous_frame,
        }
        old_program = renderer._program

        renderer._set_program(int(prog))
        GL.glUseProgram(int(prog))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, res, res)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        renderer._apply_render_uniforms(
            bass=bass,
            mid=mid,
            treb=treb,
            t=t,
            device_w=res,
            device_h=res,
            feedback_res=res,
            previous_frame_tex=0,
        )

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

        pipeline = self._fitness_pipelines.get(tier.name)
        if pipeline is not None and pipeline.available:
            prev_tex = self._previous_fitness_tex.get(tier.name, 0)
            pipeline.dispatch(tex, previous_tex=prev_tex)
            self._previous_fitness_tex[tier.name] = tex

        renderer._restore_gl_state(gl_state)
        renderer._program = old_program
        for attr, val in old_locs.items():
            setattr(renderer, attr, val)

    def _sync_gpu(self) -> None:
        """Block until all previously dispatched GL work has completed.

        The multi-res fitness loop renders several candidates into the
        *same* FBO/texture and reads metrics back with a shared readback
        PBO. Without a barrier the metrics read for candidate N can come
        from candidate N+1's render. A fence sync is preferred over
        ``glFinish`` because it only waits on the queued work rather than
        flushing the whole pipeline, but ``glFinish`` is the safe
        fallback when ARB_sync is unavailable.
        """
        glFenceSync = getattr(GL, "glFenceSync", None)
        glClientWaitSync = getattr(GL, "glClientWaitSync", None)
        if glFenceSync is not None and glClientWaitSync is not None:
            try:
                sync = glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
                if sync:
                    glClientWaitSync(sync, GL.GL_SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000)
                    GL.glDeleteSync(sync)
                    return
            except Exception:
                logger.debug("Fence sync failed, falling back to glFinish")
        try:
            GL.glFinish()
        except Exception:
            logger.debug("glFinish failed in _sync_gpu")

    def _collect_and_score(self, tier: MultiResTier, genome: Any) -> Objectives:
        from evo_bio.pareto import Objectives

        pipeline = self._fitness_pipelines.get(tier.name)
        if pipeline is None or not pipeline.available:
            return Objectives()
        metrics = pipeline.read_metrics(block=False)
        if metrics is None:
            metrics = pipeline.read_metrics(block=True)
        if metrics is None:
            return Objectives()
        try:
            obj = Objectives(
                beauty=float(metrics[OUT_BEAUTY]),
                novelty=float(metrics[OUT_NOVELTY]),
                audio_sync=float(metrics[OUT_ENERGY]),
                performance=1.0,
                species_health=float(metrics[OUT_COVERAGE]),
                motion=float(metrics[OUT_MOTION]),
                complexity=float(metrics[OUT_ENTROPY]),
                memory=float(metrics[OUT_SYMMETRY]),
            )
            genome.objectives = obj
            return obj
        except Exception:
            return Objectives()

    def _promote_adjacent(self) -> None:
        with self._lock:
            names = [t.name for t in self.tiers]
            for i in range(len(names) - 1):
                self._promote(names[i], names[i + 1])

    def _promote(self, src_name: str, dst_name: str) -> None:
        with self._lock:
            src_pop = self.populations.get(src_name)
            dst_pop = self.populations.get(dst_name)
            if not src_pop or not dst_pop:
                return

            def best_obj(pop):
                best, best_obj = None, None
                for o in pop.organisms:
                    obj = getattr(o, "objectives", None)
                    if obj is None:
                        # Fabricate a single-objective vector from the scalar
                        # fitness so promotion always has a comparable
                        # Objectives instance.  This avoids the silent
                        # promotion-noise caused by a missing .fitness attr.
                        try:
                            f = float(getattr(o, "fitness", 0.0))
                        except Exception:
                            f = 0.0
                        from evo_bio.pareto import Objectives

                        obj = Objectives(
                            beauty=f * 0.4,
                            novelty=f * 0.15,
                            audio_sync=f * 0.1,
                            performance=1.0,
                            species_health=f * 0.1,
                            motion=f * 0.1,
                            complexity=f * 0.1,
                            memory=f * 0.05,
                        )
                    if best is None or obj.scalar() > best_obj.scalar():
                        best, best_obj = o, obj
                return best, best_obj

            def worst_obj(pop):
                worst, worst_obj = None, None
                for o in pop.organisms:
                    obj = getattr(o, "objectives", None)
                    if obj is None:
                        try:
                            f = float(getattr(o, "fitness", 0.0))
                        except Exception:
                            f = 0.0
                        from evo_bio.pareto import Objectives

                        obj = Objectives(
                            beauty=f * 0.4,
                            novelty=f * 0.15,
                            audio_sync=f * 0.1,
                            performance=1.0,
                            species_health=f * 0.1,
                            motion=f * 0.1,
                            complexity=f * 0.1,
                            memory=f * 0.05,
                        )
                    if worst is None or obj.scalar() < worst_obj.scalar():
                        worst, worst_obj = o, obj
                return worst, worst_obj

            src_best, src_obj = best_obj(src_pop)
            dst_worst, dst_obj = worst_obj(dst_pop)
            if src_best is None or dst_worst is None:
                return

            try:
                promotes = src_obj.dominates(dst_obj)
            except Exception:
                promotes = src_obj.scalar() > dst_obj.scalar()

            if promotes:
                dst_worst.objectives = src_obj
                if hasattr(dst_worst, "params") and hasattr(src_best, "params"):
                    dst_worst.params = copy.deepcopy(src_best.params)
                elif hasattr(dst_worst, "field_type"):
                    self.renderer._copy_phenotype(dst_worst, src_best)
                self.renderer._logger.info(
                    f"Promoted {src_name} -> {dst_name}: "
                    f"best={src_obj.scalar():.3f} > worst={dst_obj.scalar():.3f}"
                )

    def champion(self) -> Any:
        names = [t.name for t in self.tiers]
        for name in reversed(names):
            pop = self.populations.get(name)
            if pop:
                best = pop.best()
                if best is not None:
                    return best
        return self.renderer.genome

    def cleanup(self) -> None:
        for name, (fbo, tex) in list(self._fitness_fbos.items()):
            try:
                GL.glDeleteFramebuffers(1, [fbo])
            except Exception:
                logger.debug("Suppressed exception in cleanup")
            try:
                GL.glDeleteTextures(1, [tex])
            except Exception:
                logger.debug("Suppressed exception in cleanup")
            self._fitness_fbos[name] = (0, 0)
        for pipeline in self._fitness_pipelines.values():
            try:
                pipeline.cleanup()
            except Exception:
                logger.debug("Suppressed exception in cleanup")
        self._fitness_pipelines.clear()
