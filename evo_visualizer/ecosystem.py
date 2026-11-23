#!/usr/bin/env python3
"""Spatial ecosystem classes: organisms, niches, climate, physics."""

from __future__ import annotations

import array
import copy
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from OpenGL import GL

from evo.modular_genome import ModularGenome
from evo_advanced import AdvancedEcosystem
from evo_bio.cultural_evolution import Culture
from evo_multi_pass import RegionLayout
from evo_visualizer.bio import (
    OrganismBrain,
    Organs,
    Senses,
    Signal,
    SpeciesDef,
)
from evo_visualizer.population import COMPLEXITY_TIERS, _hsv_to_rgb

if TYPE_CHECKING:
    from evo_visualizer.config import EvoConfig

logger = logging.getLogger(__name__)


class ResourceField:
    """Invisible resource fields (spec 48) represented as textures (spec 60).

    Channels: bass, treble, harmony, motion. Deposited from audio at moving
    sources, diffused, decayed, and consumed by organisms.

    The four channels are stored as persistent ``np.ndarray`` buffers so the
    per-step diffusion does not rebuild Python lists from scratch.  All public
    access goes through :meth:`as_array` / :meth:`from_array` so the numpy
    storage stays coherent with the Python-level read/write helpers.
    """

    def __init__(self, w: int, h: int):
        self.w = max(4, int(w))
        self.h = max(4, int(h))
        n = self.w * self.h
        # Persistent numpy storage -- avoids rebuilding Python lists every step.
        # The public ``bass``/``treb``/``harm``/``mot`` attributes are views over
        # the same buffers so legacy index-based callers stay coherent without
        # an extra copy.
        self._bass = np.zeros(n, dtype=np.float32)
        self._treb = np.zeros(n, dtype=np.float32)
        self._harm = np.zeros(n, dtype=np.float32)
        self._mot = np.zeros(n, dtype=np.float32)
        self.bass = self._bass
        self.treb = self._treb
        self.harm = self._harm
        self.mot = self._mot
        self._t = 0.0

    def _sync_lists(self) -> None:
        """Push numpy state back to the Python list views (rarely needed)."""
        self.bass = self._bass.tolist()
        self.treb = self._treb.tolist()
        self.harm = self._harm.tolist()
        self.mot = self._mot.tolist()

    def _arr(self, ch: str) -> np.ndarray:
        return getattr(self, "_" + ch)

    def _idx(self, x: int, y: int) -> int:
        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))
        return y * self.w + x

    def deposit(self, ch: str, nx: float, ny: float, amount: float) -> None:
        # Reject non-finite inputs: a single NaN deposit would poison the whole
        # channel permanently (diffusion spreads it and subtraction then warns).
        if not (math.isfinite(nx) and math.isfinite(ny) and math.isfinite(amount)):
            return
        arr = self._arr(ch)
        cx = nx * (self.w - 1)
        cy = ny * (self.h - 1)
        R = 4
        sigma = 2.0
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                x = round(cx) + dx
                y = round(cy) + dy
                if 0 <= x < self.w and 0 <= y < self.h:
                    d2 = dx * dx + dy * dy
                    idx = y * self.w + x
                    arr[idx] = min(
                        1.5, float(arr[idx]) + amount * math.exp(-d2 / (2.0 * sigma * sigma))
                    )
        # Keep the legacy list view in sync.
        getattr(self, ch)[...] = arr

    def sample(self, ch: str, nx: float, ny: float) -> float:
        arr = self._arr(ch)
        x = max(0, min(self.w - 1, int(nx * (self.w - 1))))
        y = max(0, min(self.h - 1, int(ny * (self.h - 1))))
        return float(arr[y * self.w + x])

    def deplete(self, frac: float) -> None:
        f = max(0.0, min(1.0, frac))
        for arr in (self._bass, self._treb, self._harm, self._mot):
            arr *= 1.0 - f
        for ch in ("bass", "treb", "harm", "mot"):
            getattr(self, ch)[...] = getattr(self, "_" + ch)

    def consume(self, nx: float, ny: float, frac: float = 0.25) -> dict[str, float]:
        out = {}
        empty = {ch: 0.0 for ch in ("bass", "treb", "harm", "mot")}
        if not (math.isfinite(nx) and math.isfinite(ny) and math.isfinite(frac)):
            return dict(empty)
        for ch in ("bass", "treb", "harm", "mot"):
            arr = self._arr(ch)
            x = max(0, min(self.w - 1, int(nx * (self.w - 1))))
            y = max(0, min(self.h - 1, int(ny * (self.h - 1))))
            i = y * self.w + x
            val = float(arr[i])
            if not math.isfinite(val):
                # Scrub any pre-existing corruption instead of propagating it.
                val = 0.0
                arr[i] = 0.0
            amt = val * frac
            arr[i] = val - amt
            getattr(self, ch)[i] = val - amt
            out[ch] = amt
        return out

    def gradient(self, ch: str, nx: float, ny: float, eps: float = 0.02) -> tuple[float, float]:
        gx = self.sample(ch, nx + eps, ny) - self.sample(ch, nx - eps, ny)
        gy = self.sample(ch, nx, ny + eps) - self.sample(ch, nx, ny - eps)
        return (gx, gy)

    def update(self, audio: dict[str, float], dt: float) -> None:
        self._t += dt
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        mid = float(audio.get("mid", 0.0))
        beat = float(audio.get("beat", 0.0))
        # Bass drifts left when loud, treble right (spec 58 migration drivers).
        bx = 0.5 - bass * 0.45 + 0.08 * math.sin(self._t * 0.25)
        tx = 0.5 + treb * 0.45 + 0.08 * math.cos(self._t * 0.21)
        hx = 0.5 + 0.12 * math.sin(self._t * 0.17 + mid)
        my = 0.5 + beat * 0.4
        self.deposit("bass", bx, 0.5, bass * 0.9 * dt * 12.0)
        self.deposit("treb", tx, 0.5, treb * 0.9 * dt * 12.0)
        self.deposit("harm", hx, 0.5, mid * 0.7 * dt * 12.0)
        self.deposit("mot", 0.5, my, beat * 0.8 * dt * 12.0)
        self._diffuse_decay()

    def _diffuse_decay(self) -> None:
        """Optimized diffusion using vectorized numpy operations.

        Operates directly on the persistent ``np.ndarray`` buffers instead of
        rebuilding Python lists every step.  The list views are refreshed at
        the end so legacy index-based callers stay coherent.
        """
        if np is None:
            return

        for ch in ("bass", "treb", "harm", "mot"):
            arr = self._arr(ch)
            w, h = self.w, self.h

            # 3x3 diffusion kernel (center weight = 4, neighbours = 1).
            kernel = np.ones((3, 3), dtype=np.float32)
            kernel[1, 1] = 4.0

            _has_scipy = False
            _scipy_convolve = None
            try:
                from scipy.ndimage import convolve as _scipy_convolve

                _has_scipy = True
            except ImportError:
                _has_scipy = False
            if _has_scipy and _scipy_convolve is not None:
                convolved = _scipy_convolve(arr.reshape((h, w)), kernel, mode="reflect")
                neighbor_count = np.ones((h, w), dtype=np.float32) * 8.0
                neighbor_count[0, :] -= 3
                neighbor_count[-1, :] -= 3
                neighbor_count[:, 0] -= 3
                neighbor_count[:, -1] -= 3
                neighbor_count[0, 0] += 1
                neighbor_count[0, -1] += 1
                neighbor_count[-1, 0] += 1
                neighbor_count[-1, -1] += 1
                result = (convolved / (neighbor_count + 4.0)) * 0.992
            else:
                arr_np = arr.reshape((h, w))
                padded = np.pad(arr_np, 1, mode="reflect")
                neighborhoods = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        shifted = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
                        neighborhoods.append(shifted)
                convolved = sum(neighborhoods)
                convolved += arr_np * 3.0
                neighbor_count = np.ones((h, w), dtype=np.float32) * 8.0
                neighbor_count[0, :] -= 3
                neighbor_count[-1, :] -= 3
                neighbor_count[:, 0] -= 3
                neighbor_count[:, -1] -= 3
                neighbor_count[0, 0] += 1
                neighbor_count[0, -1] += 1
                neighbor_count[-1, 0] += 1
                neighbor_count[-1, -1] += 1
                result = (convolved / (neighbor_count + 4.0)) * 0.992

            # Write back into the persistent numpy buffer in-place.  Any
            # non-finite value is scrubbed here so a single bad deposit cannot
            # spread across the whole channel via the diffusion kernel.
            if not np.isfinite(result).all():
                result = np.nan_to_num(result, nan=0.0, posinf=1.5, neginf=0.0).astype(
                    result.dtype, copy=False
                )
            arr[:] = result.flatten()
            getattr(self, ch)[...] = arr

    def rgba_array(self) -> list[float]:
        out = []
        for i in range(self.w * self.h):
            out.append(min(1.0, float(self._bass[i])))
            out.append(min(1.0, float(self._treb[i])))
            out.append(min(1.0, float(self._harm[i])))
            out.append(min(1.0, float(self._mot[i])))
        return out


class Climate:
    """Seasonal music climate (spec 52 + spec 24 environmental events)."""

    def __init__(self):
        self.temperature = 0.5
        self.season = 0.0
        self.light_level = 0.7
        self.humidity = 0.5
        self.turbulence = 0.3
        self.bass_influence = 0.0
        self.mid_influence = 0.0
        self.treble_influence = 0.0
        self.beat_phase = 0.0
        self.events: list[EnvironmentalEvent] = []
        self._event_timer = random.uniform(8.0, 25.0)

    def update(self, audio: dict[str, float], dt: float) -> None:
        bass = float(audio.get("bass", 0.0))
        mid = float(audio.get("mid", 0.0))
        treb = float(audio.get("treb", 0.0))
        beat = float(audio.get("beat", 0.0))
        self.temperature += (bass * 0.4 - self.temperature * 0.3) * dt
        self.light_level += (mid * 0.5 + 0.3 - self.light_level * 0.2) * dt
        self.humidity += (treb * 0.3 - self.humidity * 0.1) * dt
        self.turbulence += (beat * 0.6 - self.turbulence * 0.4) * dt
        self.bass_influence = bass
        self.mid_influence = mid
        self.treble_influence = treb
        self.beat_phase += dt * (0.5 + beat)
        self.temperature = max(0.0, min(1.0, self.temperature))
        self.light_level = max(0.0, min(1.0, self.light_level))
        self.humidity = max(0.0, min(1.0, self.humidity))
        self.turbulence = max(0.0, min(1.0, self.turbulence))
        self.season += dt / 600.0
        self._event_timer -= dt
        if self._event_timer <= 0.0:
            self._event_timer = random.uniform(12.0, 35.0)
            self._maybe_spawn_event(audio)

    def _maybe_spawn_event(self, audio: dict[str, float]) -> None:
        candidates = []
        weights = []
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        beat = float(audio.get("beat", 0.0))
        if bass > 0.6:
            candidates.append(EnvironmentalEvent.FLOOD)
            weights.append(0.25)
            candidates.append(EnvironmentalEvent.SOLAR_FLARE)
            weights.append(0.15)
        if treb > 0.6:
            candidates.append(EnvironmentalEvent.STORM)
            weights.append(0.2)
        if beat > 0.7:
            candidates.append(EnvironmentalEvent.MUSIC_PULSE)
            weights.append(0.35)
        candidates.append(EnvironmentalEvent.COLD)
        weights.append(0.15)
        candidates.append(EnvironmentalEvent.FIRE)
        weights.append(0.1)
        candidates.append(EnvironmentalEvent.DARKNESS)
        weights.append(0.1)
        candidates.append(EnvironmentalEvent.RESOURCE_COLLAPSE)
        weights.append(0.08)
        if not candidates:
            return
        etype = random.choices(candidates, weights=weights, k=1)[0]
        duration = random.uniform(4.0, 12.0)
        intensity = random.uniform(0.3, 1.0)
        ev = EnvironmentalEvent(etype, duration, intensity)
        self.events.append(ev)

    def step_events(self, dt: float, eco: SpatialEcosystem, audio: dict[str, float]) -> None:
        for ev in self.events:
            ev.step(dt)
            ev.apply(eco, audio)
        self.events = [ev for ev in self.events if ev.active]

    def biome_weight(self, field_type: str) -> float:
        warm = {
            "fbm": 1.1,
            "reaction_diffusion": 1.2,
            "voronoi": 0.9,
            "mandelbrot": 0.85,
            "sdf": 0.95,
        }.get(field_type, 1.0)
        cold = {
            "fbm": 0.95,
            "reaction_diffusion": 0.9,
            "voronoi": 1.2,
            "mandelbrot": 1.15,
            "sdf": 1.1,
        }.get(field_type, 1.0)
        return warm * self.temperature + cold * (1.0 - self.temperature)

    def complexity_bonus(self) -> float:
        return 0.3 + 0.7 * self.temperature


class ExtinctionArchive:
    """Dormant store of extinct species (spec 59)."""

    def __init__(self):
        self.dormant: dict[int, dict] = {}
        self.revival_cooldown = 0.0

    def archive(self, species: SpeciesDef, genome_dict: dict) -> None:
        self.dormant[species.sid] = {
            "color": species.color,
            "diet": list(species.diet),
            "biome_affinity": species.biome_affinity,
            "genome": genome_dict,
            "avg_complexity": species.avg_complexity,
        }

    def maybe_revive(self, dt: float) -> int | None:
        self.revival_cooldown -= dt
        if self.revival_cooldown > 0.0 or not self.dormant:
            return None
        self.revival_cooldown = 20.0
        if random.random() < 0.8:
            return random.choice(list(self.dormant.keys()))
        return None


@dataclass
class NicheDefinition:
    """Defines one environmental niche: name, colour palette, and
    the genes it favours with per-gene weight modifiers."""

    name: str
    colour: tuple[float, float, float]
    favours: dict[str, float] = field(default_factory=dict)
    penalises: dict[str, float] = field(default_factory=dict)
    env_intensity: float = 1.0

    def fitness_modifier(self, gene_key: str, gene_value: float) -> float:
        """Return a multiplicative modifier for a gene based on this niche."""
        modifier = 1.0
        if gene_key in self.favours:
            modifier *= 1.0 + self.favours[gene_key] * gene_value
        if gene_key in self.penalises:
            modifier *= 1.0 - self.penalises[gene_key] * abs(gene_value)
        return max(0.1, min(3.0, modifier))


class NicheManager:
    """Manages screen-region-to-environment mappings and applies
    niche-based selection pressure to organisms.

    Seven base niches mapped onto screen regions:
      desert  - hot, arid; favours energy_efficiency, heat tolerance
      ocean   - deep, cold, high-pressure; favours flow, symmetry
      forest  - lush, moderate; favours complexity, growth
      lava    - extreme heat, volatile; favours aggression, speed
      ice     - freezing, stable; favours resilience, cold tolerance
      space   - vacuum, radiation; favours adaptability, novelty
      organic - living, nutrient-rich; favours reproduction, metabolism
    """

    DEFAULT_NICHES: ClassVar[list[NicheDefinition]] = [
        NicheDefinition(
            name="desert",
            colour=(0.9, 0.7, 0.3),
            # Gene keys aligned to the real VisualGenome params (see
            # evo_visualizer.genomes.__init__.VisualGenome.defaults) so the
            # fitness modifiers are no-ops.
            favours={"warp": 0.8, "noise_scale": 0.6, "contrast": 0.7},
            penalises={"glow": 0.3, "feedback": 0.4},
        ),
        NicheDefinition(
            name="ocean",
            colour=(0.2, 0.4, 0.8),
            favours={"symmetry": 0.9, "flow_speed": 0.8, "glow": 0.5},
            penalises={"contrast": 0.3, "warp": 0.2},
        ),
        NicheDefinition(
            name="forest",
            colour=(0.2, 0.7, 0.3),
            favours={"fractal_layers": 0.8, "turbulence": 0.7, "symmetry": 0.4},
            penalises={"warp": 0.2},
        ),
        NicheDefinition(
            name="lava",
            colour=(0.9, 0.2, 0.1),
            favours={"vortex": 0.9, "glow": 0.8, "particle_density": 0.7},
            penalises={"symmetry": 0.4, "glow": 0.5},
        ),
        NicheDefinition(
            name="ice",
            colour=(0.7, 0.9, 1.0),
            favours={"feedback": 0.8, "contrast": 0.4, "domain_warp": 0.5},
            penalises={"vortex": 0.3, "noise_scale": 0.3},
        ),
        NicheDefinition(
            name="space",
            colour=(0.05, 0.02, 0.1),
            favours={"turbulence": 0.9, "domain_warp": 0.8, "warp": 0.6},
            penalises={"glow": 0.4, "fractal_layers": 0.5},
        ),
        NicheDefinition(
            name="organic",
            colour=(0.3, 0.6, 0.2),
            favours={"particle_density": 0.9, "glow": 0.8, "turbulence": 0.7},
            penalises={"contrast": 0.2, "vortex": 0.3},
        ),
    ]

    LAYOUT_PATTERNS: ClassVar[dict[str, list[str]]] = {
        "stripes": ["desert", "ocean", "forest", "lava", "ice", "space", "organic"],
        "blocks": ["desert", "desert", "ocean", "ocean", "forest", "forest", "lava"],
        "quadrants": ["desert", "ocean", "forest", "lava"],
        "radial": ["organic", "forest", "ocean", "desert", "lava", "ice", "space"],
        "random": [],
    }

    def __init__(self, count: int = 5, layout: str = "stripes"):
        self.count = max(1, min(12, count))
        self.layout = layout
        self.niches: list[NicheDefinition] = []
        self.regions: list[dict[str, Any]] = []
        self._assign_niches()

    def _assign_niches(self) -> None:
        pattern = self.LAYOUT_PATTERNS.get(self.layout, [])
        if pattern and self.layout != "random":
            cycle = (pattern * ((self.count // len(pattern)) + 1))[: self.count]
            self.niches = [
                next((n for n in self.DEFAULT_NICHES if n.name == name), self.DEFAULT_NICHES[0])
                for name in cycle
            ]
        elif self.layout == "random":
            self.niches = random.sample(
                self.DEFAULT_NICHES, min(self.count, len(self.DEFAULT_NICHES))
            )
            while len(self.niches) < self.count:
                self.niches.append(random.choice(self.DEFAULT_NICHES))
        else:
            self.niches = [
                self.DEFAULT_NICHES[i % len(self.DEFAULT_NICHES)] for i in range(self.count)
            ]

    def set_layout(self, layout: str) -> None:
        self.layout = layout
        self._assign_niches()

    def set_count(self, count: int) -> None:
        self.count = max(1, min(12, count))
        self._assign_niches()

    def get_niche_for_position(self, x: float, y: float) -> tuple[NicheDefinition, int]:
        """Return the niche and region index for a normalized (0..1) position."""
        if not self.regions:
            return self.niches[0], 0
        for i, region in enumerate(self.regions):
            rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return self.niches[i % len(self.niches)], i
        return self.niches[0], 0

    def update_regions(self, regions: list[dict[str, Any]]) -> None:
        """Update region bounds (called from the region renderer)."""
        self.regions = regions

    def niche_fitness_bonus(
        self, organism_pos: tuple[float, float], gene_key: str, gene_value: float
    ) -> float:
        """Return a fitness bonus multiplier for a gene given the organism's screen position."""
        niche, _ = self.get_niche_for_position(organism_pos[0], organism_pos[1])
        return niche.fitness_modifier(gene_key, gene_value)

    def region_colour(self, region_index: int) -> tuple[float, float, float]:
        """Return the RGB colour for a region's niche overlay."""
        if region_index < len(self.niches):
            return self.niches[region_index].colour
        return (0.5, 0.5, 0.5)


class SpatialOrganism:
    """A living organism occupying its own territory (specs 46, 47, 53-58, 51, 56-57, 59, 63)."""

    EMOTION_HUNGRY = "hungry"
    EMOTION_FEAR = "fear"
    EMOTION_AGGRESSIVE = "aggressive"
    EMOTION_CURIOUS = "curious"
    EMOTION_SLEEPING = "sleeping"
    EMOTION_MATING = "mating"

    def __init__(
        self,
        genome,
        species: int,
        *,
        energy: float = 1.0,
        pos: tuple[float, float] | None = None,
        generation: int = 0,
    ):
        self.genome = genome
        self.species = species
        self.pos = list(pos) if pos else [random.random(), random.random()]
        self.vel = [random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)]
        self.radius = random.uniform(0.06, 0.12)
        self.energy = energy
        self.age = 0.0
        self.life_span = random.uniform(60.0, 240.0)
        self.senses = Senses()
        self.organs = Organs()
        self.brain = OrganismBrain(seed=random.randint(0, 1 << 30))
        self.complexity_level = 0
        from evo_bio.pareto import Objectives

        self.objectives = Objectives()
        self.alive = True
        self.generation = generation
        self.preferred = random.choice(("bass", "treb", "harm", "mot"))
        self.signal_cooldown = 0.0
        self._prev_energy = energy
        self.soft_body = SoftBody(complexity=random.randint(2, 5))
        self.memory = AreaMemory()
        self.material = DynamicMaterial()
        self.skeleton = ProceduralSkeleton(depth=random.randint(2, 4))
        self.organs_damage = 0.0
        self.last_pos = list(self.pos)

        # Spec 51 (Time Dilation): per-organism tick rate.
        # 1 update/sec (slow — forest) to 100 updates/sec (fast — insect).
        self.tick_rate = random.uniform(1.0, 100.0)
        self._tick_accumulator = 0.0
        self._last_tick_dt = 0.0

        # Spec 14 (Adaptive LOD): 2=near, 1=mid, 0=far.
        self._lod_tier: int = 2

        # Infection state (continuous evolution: disease spreads every frame).
        self.infected: bool = False
        self.infection_timer: float = 0.0
        self.infection_strength: float = 0.0

        # Spec 56/57 (Parasites + Symbiosis): nested organism relationship.
        self.symbiont: SpatialOrganism | None = None
        self.host: SpatialOrganism | None = None
        self.symbiosis_type: str = "none"  # "none", "parasite", "mutualism"

        # Spec 59 (Emotional States): internal state biases decisions.
        self.emotional_state: str = self.EMOTION_CURIOUS
        self._emotion_timer = 0.0
        self._emotion_cooldown = random.uniform(3.0, 12.0)

        # Spec 63 (Innovation Protection): prevents breakthrough genomes
        # from being mutated away immediately.
        self.innovation_protected = False
        self.innovation_age = 0
        self.innovation_protection_duration = 15.0

        self.max_energy = 3.0

        # Cultural evolution: each organism carries a culture that
        # can acquire, transmit, and refine memes (behaviours/techniques)
        # independently of genetic mutation.
        self.culture: Culture = Culture()

        # Spatial grouping (colonies / islands / veins).
        self._group_id = 0
        self._group_center = (self.pos[0], self.pos[1])
        self._vein_connections: list[int] = []

        # Environmental niche tracking.
        self.niche: str = "deep_field"
        self.niche_index: int = 0
        self.niche_fitness_bonus: float = 1.0
        self.niche_migration_cooldown: float = 0.0

    @property
    def fitness(self) -> float:
        return self.objectives.scalar()

    @fitness.setter
    def fitness(self, value: float) -> None:
        from evo_bio.pareto import Objectives

        self.objectives = Objectives(
            beauty=float(value) * 0.4,
            novelty=float(value) * 0.1,
            audio_sync=float(value) * 0.05,
            performance=1.0,
            species_health=float(value) * 0.1,
            motion=float(value) * 0.15,
            complexity=float(value) * 0.1,
            memory=float(value) * 0.1,
        )

    @property
    def lineage(self) -> str:
        return self.genome.lineage

    @lineage.setter
    def lineage(self, value: str) -> None:
        self.genome.lineage = value

    def clone_child(self) -> SpatialOrganism:
        g = self.genome.clone()
        g.mutate(strength=0.12)
        child = SpatialOrganism(
            g,
            self.species,
            energy=self.energy * 0.4,
            pos=(
                self.pos[0] + random.uniform(-0.05, 0.05),
                self.pos[1] + random.uniform(-0.05, 0.05),
            ),
            generation=self.generation + 1,
        )
        child.senses = self.senses.clone()
        child.senses.mutate(0.1)
        child.organs = self.organs.clone()
        child.organs.mutate(0.1)
        child.brain = self.brain.clone()
        child.brain.mutate(0.15)
        child.complexity_level = self.complexity_level
        child.soft_body = copy.deepcopy(self.soft_body)
        child.soft_body.mutate(strength=0.08)
        child.skeleton = copy.deepcopy(self.skeleton)
        child.skeleton.mutate(strength=0.1)
        child.material = copy.deepcopy(self.material)
        child.material.mutate(strength=0.1)
        child.tick_rate = self.tick_rate * random.uniform(0.8, 1.25)
        child.tick_rate = max(1.0, min(100.0, child.tick_rate))
        child.emotional_state = random.choice(
            [
                SpatialOrganism.EMOTION_HUNGRY,
                SpatialOrganism.EMOTION_FEAR,
                SpatialOrganism.EMOTION_AGGRESSIVE,
                SpatialOrganism.EMOTION_CURIOUS,
                SpatialOrganism.EMOTION_SLEEPING,
                SpatialOrganism.EMOTION_MATING,
            ]
        )
        # Spec 31: let the child inherit the parent's evolving map dynamics
        # (decay/diffusion/range). AdvancedEcosystem consumes this hint when it
        # first creates the child's developmental state.
        parent_maps = getattr(self, "_own_maps", None)
        if parent_maps is not None:
            child._parent_maps = parent_maps
        return child

    def perceive(
        self, audio: dict[str, float], field: ResourceField, signals: list[Signal], neighbours: int
    ) -> list[float]:
        s = self.senses
        vals = {
            "bass": float(audio.get("bass", 0.0)) * s.bass_sensitivity,
            "treb": float(audio.get("treb", 0.0)) * s.treble_sensitivity,
            "harm": float(audio.get("mid", 0.0)) * s.harmony_sensitivity,
            "mot": float(audio.get("beat", 0.0)) * s.motion_sensitivity,
        }
        return [
            vals["bass"],
            vals["treb"],
            max(0.0, min(1.0, self.energy / 3.0)),
            min(1.0, neighbours / 8.0),
            min(1.0, self.age / self.life_span),
            max(0.0, min(1.0, self.objectives.scalar())),
        ]

    def step(
        self,
        dt: float,
        eco: SpatialEcosystem,
        audio: dict[str, float],
        neighbours: list[SpatialOrganism],
    ) -> None:
        if not self.alive:
            return

        # Spec 51 (Time Dilation): accumulate fractional steps.
        self._tick_accumulator += dt * self.tick_rate
        effective_dt = dt
        if self._tick_accumulator < 1.0:
            return
        effective_dt = min(0.1, self._tick_accumulator)
        self._tick_accumulator -= 1.0
        self._last_tick_dt = effective_dt

        # Innovation protection ages out.
        if self.innovation_protected:
            self.innovation_age += dt
            if self.innovation_age >= self.innovation_protection_duration:
                self.innovation_protected = False
                self.innovation_age = 0.0

        # Spec 56/57 (Symbiosis): parasite drains host; mutualism shares energy.
        if self.symbiosis_type == "parasite" and self.host is not None and self.host.alive:
            drain = 0.015 * effective_dt
            self.host.energy -= drain
            self.energy += drain * 0.8
        elif self.symbiosis_type == "mutualism" and self.host is not None and self.host.alive:
            if self.energy > 1.0 and self.host.energy < 2.0:
                share = 0.02 * effective_dt
                self.energy -= share
                self.host.energy += share
            if self.host.energy > 1.0 and self.energy < 2.0:
                share = 0.02 * effective_dt
                self.host.energy -= share
                self.energy += share

        self.age += effective_dt
        self.signal_cooldown = max(0.0, self.signal_cooldown - effective_dt)

        # Spec 14 (Adaptive LOD): skip expensive subsystems for far organisms.
        if self._lod_tier <= 0:
            recall = 0.0
            move_a = 0.0
            grow_a = 0.0
            repro_a = 0.0
            mutate_a = 0.0
            shader_a = 0.0
            intake = self._consume(eco.field, effective_dt)
            self.energy = max(0.0, self.energy - 0.005 * effective_dt)
            self.objectives.beauty = (
                self.objectives.beauty * 0.95 + (max(0.0, self.energy / 3.0)) * 0.05
            )
            if self.energy <= 0.0 or self.age >= self.life_span:
                self.alive = False
            self._sanitize_state()
            return

        # Soft body animation (feature 20)
        self.soft_body.step(effective_dt, audio)

        # Memory recall (feature 25)
        recall = self.memory.recall(self.pos[0], self.pos[1])

        n_in = len(neighbours)
        inputs = self.perceive(audio, eco.field, eco.signals, n_in)
        acts = [0.0, 0.0, 0.0, 0.0, 0.0]
        if eco.config.enable_brain:
            acts = self.brain.process(inputs)
        move_a, grow_a, repro_a, mutate_a, shader_a = acts

        # Spec 59 (Emotional States): bias decisions from internal state.
        emotion_bias = self._emotion_bias(audio, neighbours)
        move_a += emotion_bias.get("move", 0.0)
        repro_a += emotion_bias.get("reproduce", 0.0)
        mutate_a += emotion_bias.get("mutate", 0.0)

        self._move(effective_dt, eco, audio, neighbours, move_a, recall)
        intake = self._consume(eco.field, effective_dt)
        self._deposit(eco.field, effective_dt)
        self._pay_costs(effective_dt, eco)
        self._act(effective_dt, eco, audio, grow_a, repro_a, mutate_a, shader_a, intake)
        self._emit_signal(eco, audio, intake)

        # Spec 14 (Adaptive LOD): skip expensive per-frame updates for far organisms.
        if self._lod_tier > 0:
            # Spec 59 (Emotional States): update emotional state periodically.
            self._emotion_timer += effective_dt
            if self._emotion_timer >= self._emotion_cooldown:
                self._emotion_timer = 0.0
                self._emotion_cooldown = random.uniform(3.0, 12.0)
                self._update_emotional_state(eco, audio, neighbours, intake)

            # Carrying capacity: energy cannot grow without bound (spec 56).
            self.energy = min(eco.max_energy * 1.15, self.energy)

            # Record memory from intake (feature 25)
            if intake > 0.03:
                self.memory.record(self.pos[0], self.pos[1], intake)
            elif intake < 0.005 and self.energy < 0.8:
                self.memory.record(self.pos[0], self.pos[1], -0.1)

            # rolling fitness
            inst = (
                max(0.0, min(1.0, self.energy / 3.0)) * 0.5
                + min(1.0, intake) * 0.3
                + eco.climate.biome_weight(getattr(self.genome, "field_type", "fbm")) * 0.2
            )
            # Environmental niche fitness modifier.
            if eco.niche_manager is not None and eco.config.enable_niches:
                niche, _ = eco.niche_manager.get_niche_for_position(self.pos[0], self.pos[1])
                self.niche = niche.name
                self.niche_index = _
                niche_bonus = niche.fitness_modifier("warp", 0.5)
                inst *= niche_bonus
            self.objectives.beauty = self.objectives.beauty * 0.9 + inst * 0.1
            self.brain.fit = self.brain.fit * 0.95 + (inst - 0.5) * 0.05

        if self.energy <= 0.0 or self.age >= self.life_span:
            # Detach symbiont if host dies.
            if self.symbiont is not None:
                self.symbiont.host = None
                self.symbiont.symbiosis_type = "none"
                self.symbiont = None
            if self.host is not None:
                self.host.symbiont = None
                self.host = None
            self.alive = False

        self._sanitize_state()

    def _sanitize_state(self) -> None:
        """Repair non-finite simulation state before it corrupts the renderer.

        NaN/Inf can enter via audio levels, gradient math, or numerical drift
        and would otherwise crash the spatial pass with 'cannot convert float
        NaN to integer' when a territory rectangle is cast to int.
        """
        if not math.isfinite(self.pos[0]) or not math.isfinite(self.pos[1]):
            self.pos[0] = random.uniform(0.05, 0.95)
            self.pos[1] = random.uniform(0.05, 0.95)
            self.vel[0] = 0.0
            self.vel[1] = 0.0
        else:
            self.pos[0] = max(0.02, min(0.98, self.pos[0]))
            self.pos[1] = max(0.02, min(0.98, self.pos[1]))
        if not math.isfinite(self.vel[0]):
            self.vel[0] = 0.0
        if not math.isfinite(self.vel[1]):
            self.vel[1] = 0.0
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            self.radius = 0.08
        else:
            self.radius = max(0.02, min(0.5, self.radius))
        if not math.isfinite(self.energy):
            self.energy = 1.0
        else:
            self.energy = max(
                0.0, min(self.max_energy * 1.15 if self.max_energy else 3.0, self.energy)
            )
        from evo_bio.pareto import Objectives

        if not math.isfinite(self.objectives.beauty):
            self.objectives = Objectives()
        if not math.isfinite(self.age) or self.age < 0.0:
            self.age = 0.0
        if not math.isfinite(self.complexity_level):
            self.complexity_level = 1

    def _emotion_bias(
        self, audio: dict[str, float], neighbours: list[SpatialOrganism]
    ) -> dict[str, float]:
        bias: dict[str, float] = {"move": 0.0, "reproduce": 0.0, "mutate": 0.0}
        state = self.emotional_state
        if state == self.EMOTION_HUNGRY:
            bias["move"] = 0.4
            bias["mutate"] = -0.2
        elif state == self.EMOTION_FEAR:
            bias["move"] = 0.6
            bias["reproduce"] = -0.5
        elif state == self.EMOTION_AGGRESSIVE:
            bias["move"] = 0.3
            bias["mutate"] = 0.2
        elif state == self.EMOTION_CURIOUS:
            bias["move"] = 0.15
            bias["mutate"] = 0.1
        elif state == self.EMOTION_MATING:
            bias["reproduce"] = 0.8
            bias["move"] = 0.1
        elif state == self.EMOTION_SLEEPING:
            bias["move"] = -0.8
            bias["mutate"] = -0.3
        # Tick rate bias: fast organisms are more excitable.
        tick_bonus = (self.tick_rate / 100.0) * 0.3
        bias["move"] += tick_bonus
        return bias

    def _update_emotional_state(
        self,
        eco: SpatialEcosystem,
        audio: dict[str, float],
        neighbours: list[SpatialOrganism],
        intake: float,
    ) -> None:
        bass = float(audio.get("bass", 0.0))
        beat = float(audio.get("beat", 0.0))
        n_predators = sum(
            1
            for o in neighbours
            if o.alive
            and o.species != self.species
            and self.species in eco.species.get(o.species, SpeciesDef(0, (0, 0, 0))).diet
        )
        if self.energy < 0.3:
            self.emotional_state = self.EMOTION_HUNGRY
        elif n_predators > 0 and self.energy < 1.5:
            self.emotional_state = self.EMOTION_FEAR
        elif self.energy > 1.4 and self.age > self.life_span * 0.3:
            self.emotional_state = self.EMOTION_MATING
        elif beat > 0.7 and bass > 0.5:
            self.emotional_state = self.EMOTION_AGGRESSIVE
        elif self.energy > 2.0:
            self.emotional_state = self.EMOTION_CURIOUS
        else:
            self.emotional_state = self.EMOTION_SLEEPING

    def _move(self, dt, eco, audio, neighbours, move_a, recall=(0.0, 0.0)) -> None:
        ax, ay = recall
        mv = self.organs.movement if eco.config.enable_organs else 0.5
        rng = self.senses.hearing_range if eco.config.enable_senses else 0.35

        # Flocking (spec 51): alignment, cohesion, separation.
        if eco.config.enable_flocking and neighbours:
            avx = avy = 0.0
            cx = cy = 0.0
            sx = sy = 0.0
            for o in neighbours:
                avx += o.vel[0]
                avy += o.vel[1]
                cx += o.pos[0]
                cy += o.pos[1]
                d = math.hypot(o.pos[0] - self.pos[0], o.pos[1] - self.pos[1]) or 1e-3
                if d < rng * 0.5:
                    sx += (self.pos[0] - o.pos[0]) / d
                    sy += (self.pos[1] - o.pos[1]) / d
            n = len(neighbours)
            ax += (avx / n) * 0.4 + sx * 0.6
            ay += (avy / n) * 0.4 + sy * 0.6
            ax += (cx / n - self.pos[0]) * 0.3
            ay += (cy / n - self.pos[1]) * 0.3

        # Migration (spec 58): climb the preferred resource gradient.
        if eco.config.enable_migration:
            gx, gy = eco.field.gradient(self.preferred, self.pos[0], self.pos[1])
            ax += gx * 0.8 * (self.senses.motion_sensitivity if eco.config.enable_senses else 1.0)
            ay += gy * 0.8
            # Lateral music current.
            ax += (float(audio.get("treb", 0.0)) - float(audio.get("bass", 0.0))) * 0.05

        # Signal reactions (spec 50).
        if eco.config.enable_signals:
            for sig in eco.signals:
                d = math.hypot(sig.x - self.pos[0], sig.y - self.pos[1])
                if d > rng:
                    continue
                w = 1.0 - d / rng
                if sig.stype == "danger":
                    ax += (self.pos[0] - sig.x) * w * 1.2
                    ay += (self.pos[1] - sig.y) * w * 1.2
                elif sig.stype in ("food", "music", "light"):
                    ax += (sig.x - self.pos[0]) * w * 0.8
                    ay += (sig.y - self.pos[1]) * w * 0.8
                elif sig.stype == "noise":
                    ax += (self.pos[0] - sig.x) * w * 1.0
                    ay += (self.pos[1] - sig.y) * w * 1.0
                elif sig.stype == "mate" and sig.species == self.species:
                    ax += (sig.x - self.pos[0]) * w * 1.0
                    ay += (sig.y - self.pos[1]) * w * 1.0

        # Spec 31: steer by the organism's own evolving maps -- climb its food
        # map, flee its danger map. These are the same maps rendered into its
        # territory, so what you see is what drives it.
        own_maps = getattr(self, "_own_maps", None)
        if own_maps is not None:
            try:
                fgx, fgy = own_maps.grad_food
                dgx, dgy = own_maps.grad_danger
                hunger = 1.0 - max(0.0, min(1.0, self.energy / self.max_energy))
                ax += fgx * (0.6 + 0.8 * hunger)
                ay += fgy * (0.6 + 0.8 * hunger)
                ax -= dgx * 1.1
                ay -= dgy * 1.1
            except Exception:
                logger.debug("Suppressed exception in cleanup")

        self.vel[0] = (self.vel[0] + ax * dt * 0.6) * 0.96
        self.vel[1] = (self.vel[1] + ay * dt * 0.6) * 0.96
        sp = math.hypot(self.vel[0], self.vel[1])
        maxsp = 0.15 * (0.4 + mv)
        if sp > maxsp:
            self.vel[0] *= maxsp / sp
            self.vel[1] *= maxsp / sp
        self.pos[0] += self.vel[0] * dt * (0.5 + 0.5 * abs(move_a))
        self.pos[1] += self.vel[1] * dt * (0.5 + 0.5 * abs(move_a))
        # Bounce off world edges.
        if self.pos[0] < 0.02:
            self.pos[0] = 0.02
            self.vel[0] = abs(self.vel[0])
        if self.pos[0] > 0.98:
            self.pos[0] = 0.98
            self.vel[0] = -abs(self.vel[0])
        if self.pos[1] < 0.02:
            self.pos[1] = 0.02
            self.vel[1] = abs(self.vel[1])
        if self.pos[1] > 0.98:
            self.pos[1] = 0.98
            self.vel[1] = -abs(self.vel[1])

    def _consume(self, field: ResourceField, dt: float) -> float:
        sens = self.senses
        nx, ny = self.pos[0], self.pos[1]
        if not math.isfinite(nx) or not math.isfinite(ny):
            nx = random.uniform(0.05, 0.95)
            ny = random.uniform(0.05, 0.95)
            self.pos[0], self.pos[1] = nx, ny
        eaten = field.consume(nx, ny, 0.2)
        gain = (
            eaten["bass"] * sens.bass_sensitivity
            + eaten["treb"] * sens.treble_sensitivity
            + eaten["harm"] * sens.harmony_sensitivity
            + eaten["mot"] * sens.motion_sensitivity
        )
        gain = min(1.5, gain * 1.5 * dt * 8.0)
        self.energy += gain
        return gain

    def _deposit(self, field: ResourceField, dt: float) -> None:
        if self.energy <= 1.2:
            return
        amt = min(0.8, (self.energy - 1.2) * 0.15 * dt)
        field.deposit("bass", self.pos[0], self.pos[1], amt)
        self.energy -= amt

    def _visual_cost(self, eco) -> float:
        g = self.genome
        ft = getattr(g, "field_type", "fbm")
        base = {
            "fbm": 0.2,
            "voronoi": 0.3,
            "reaction_diffusion": 0.5,
            "mandelbrot": 0.6,
            "sdf": 1.0,
        }.get(ft, 0.2)
        posts = getattr(g, "post", None) or []
        base += 0.08 * len(posts)
        if eco.config.enable_organs:
            base += 0.5 * self.organs.energy_cost()
        base += 0.12 * self.complexity_level
        return base

    def _pay_costs(self, dt: float, eco) -> None:
        if not eco.config.enable_energy_economy:
            return
        cost = (0.012 + self._visual_cost(eco)) * dt
        # Luxury metabolism: high energy is expensive to maintain (spec 56),
        # which bounds the population's energy and keeps GPU-heavy organisms in check.
        cost += 0.08 * max(0.0, self.energy - 1.0) * dt
        self.energy -= cost

    def _act(self, dt, eco, audio, grow_a, repro_a, mutate_a, shader_a, intake) -> None:
        # Grow (spec 47): radius tracks energy.
        en = max(0.0, min(1.0, self.energy / 3.0))
        target_r = 0.05 + 0.10 * en + 0.03 * self.complexity_level
        self.radius += (target_r - self.radius) * 0.1

        # Spec 63 (Innovation Protection): skip genome mutation if protected.
        can_mutate_genome = not self.innovation_protected

        # Mutation (spec 57 via brain, plus time-based drift).
        if (
            eco.config.enable_brain
            and shader_a > 0.4
            and random.random() < 0.05
            and can_mutate_genome
        ):
            self.genome.mutate(strength=0.1)
        elif random.random() < 0.02 and can_mutate_genome:
            self.genome.mutate(strength=0.08)

        # Soft body mutation (feature 20)
        if random.random() < 0.01:
            self.soft_body.mutate(strength=0.08)

        # Skeleton mutation (feature 27)
        if random.random() < 0.008:
            self.skeleton.mutate(strength=0.1)

        # Material mutation (feature 28)
        if random.random() < 0.012:
            self.material.mutate(strength=0.1)

        # Spec 56/57 (Symbiosis): attempt to attach parasite or form mutualism.
        if self.symbiont is None and self.host is None and random.random() < 0.005:
            self._try_form_symbiosis(eco)

        # Evolution of complexity (spec 57): acquire higher tiers when energy
        # is plentiful and the climate allows it.
        if (
            eco.config.enable_complexity
            and self.energy > 1.3
            and self.complexity_level < 7
            and random.random() < 0.02 * (0.5 + eco.climate.complexity_bonus())
        ):
            self.complexity_level += 1
            ft, wp = COMPLEXITY_TIERS[self.complexity_level]
            try:
                self.genome.field_type = ft
                self.genome.warp = wp
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug("Failed to apply complexity tier: %s", e)
            self.energy -= 0.3

        # Reproduction (spec 46): spawn a child when thriving.
        if (
            self.energy > 1.3
            and (not eco.config.enable_brain or repro_a > 0.2)
            and random.random() < 0.10
        ):
            child = self.clone_child()
            eco.spawn_child(child)
            self.energy -= 0.4

    def _try_form_symbiosis(self, eco: SpatialEcosystem) -> None:
        candidates = [
            o
            for o in eco.organisms
            if o is not self
            and o.alive
            and o.symbiont is None
            and o.host is None
            and math.hypot(o.pos[0] - self.pos[0], o.pos[1] - self.pos[1]) < 0.12
        ]
        if not candidates:
            return
        target = random.choice(candidates)
        if self.energy > target.energy * 1.5:
            self.symbiont = target
            target.host = self
            self.symbiosis_type = "parasite"
            target.symbiosis_type = "parasite"
        elif abs(self.energy - target.energy) < 0.3:
            self.host = target
            target.symbiont = self
            self.symbiosis_type = "mutualism"
            target.symbiosis_type = "mutualism"

    def _emit_signal(self, eco, audio, intake) -> None:
        if not eco.config.enable_signals or self.signal_cooldown > 0.0:
            return
        stype = None
        if (self.emotional_state == self.EMOTION_FEAR and self.energy < 0.4) or (
            self.emotional_state == self.EMOTION_HUNGRY and intake < 0.01
        ):
            stype = "danger"
        elif intake < 0.01 and self.energy > 1.0:
            stype = "food"
        elif (
            self.emotional_state == self.EMOTION_MATING
            and self.energy > 1.4
            and self.age > self.life_span * 0.4
        ):
            stype = "mate"
        elif float(audio.get("beat", 0.0)) > 0.7:
            stype = "music"
        elif self.energy > 2.0 and float(audio.get("treb", 0.0)) > 0.3:
            stype = "light"
        elif float(audio.get("bass", 0.0)) > 0.6 and float(audio.get("treb", 0.0)) > 0.6:
            stype = "noise"
        if stype:
            eco.signals.append(Signal(self.pos[0], self.pos[1], stype, self.species, 1.0, 2.5))
            self.signal_cooldown = 3.0


class SpatialEcosystem:
    """CPU-supervised spatial ecosystem (specs 46-60)."""

    def __init__(self, config: EvoConfig):
        self.config = config
        self.max_energy = 3.0
        self.t = 0.0
        self.step_count = 0
        self.field = ResourceField(config.spatial_world_w, config.spatial_world_h)
        self.climate = Climate()
        self.signals: list[Signal] = []
        self.archive = ExtinctionArchive()
        self.species: dict[int, SpeciesDef] = {}
        self.organisms: list[SpatialOrganism] = []
        self._next_sid = 0
        self._spawned_this_step: list[SpatialOrganism] = []

        self.niche_manager: NicheManager | None = None
        if getattr(config, "enable_niches", True):
            self.niche_manager = NicheManager(
                count=getattr(config, "region_count", 5),
                layout=getattr(config, "niche_layout", "stripes"),
            )

        self.culture = Culture()

        n_species = 4
        for i in range(n_species):
            self._new_species(diet=self._random_diet(n_species, i))

        for _ in range(config.spatial_population_size):
            sid = random.randrange(n_species)
            org = SpatialOrganism(ModularGenome(), sid, energy=random.uniform(1.0, 1.6))
            org.culture = self.culture
            self.organisms.append(org)

        self._seed_initial_culture()

    # NOTE: niche assignment lives in the single merged _assign_niches()
    # defined further below (after _get_region_bounds). Do not add another
    # definition here: Python would silently shadow one of them and drop the
    # niche_fitness_bonus bookkeeping.
    def _new_species(self, diet: set | None = None) -> int:
        sid = self._next_sid
        self._next_sid += 1
        hue = (sid * 0.137) % 1.0
        color = _hsv_to_rgb(hue, 0.7, 1.0)
        self.species[sid] = SpeciesDef(
            sid=sid,
            color=color,
            diet=set(diet or set()),
            biome_affinity=random.random(),
            born_at=self.t,
        )
        return sid

    def _random_diet(self, n: int, i: int) -> set:
        if n <= 1:
            return set()
        prey = (i + 1) % n
        return {prey} if random.random() < 0.35 else set()

    def _seed_initial_culture(self) -> None:
        """Give each species a few starter memes so cultural evolution has content to spread."""
        starter_names = [
            "spiral_drift",
            "pulse_sync",
            "territory_mark",
            "food_trail",
            "flee_pattern",
            "hunt_coordinate",
            "rest_cycle",
            "signal_echo",
            "color_shift",
        ]
        for sid in self.species:
            for _ in range(3):
                name = random.choice(starter_names)
                strength = random.uniform(0.3, 0.8)
                self.culture.discover(str(sid), name, strength=strength)

    def spawn_child(self, child: SpatialOrganism) -> None:
        child.culture = self.culture
        self._spawned_this_step.append(child)

    def _neighbours(self, org: SpatialOrganism) -> list[SpatialOrganism]:
        rng = getattr(org.senses, "hearing_range", 0.35)
        out = []
        for o in self.organisms:
            if o is org or not o.alive:
                continue
            if math.hypot(o.pos[0] - org.pos[0], o.pos[1] - org.pos[1]) < rng:
                out.append(o)
        return out

    def step(self, dt: float, audio: dict[str, float], music_dna: Any = None) -> None:
        dt = min(0.1, max(1e-3, dt))
        self.t += dt
        self.step_count += 1

        self.field.update(audio, dt)
        if self.config.enable_climate:
            self.climate.update(audio, dt)

        for sig in self.signals:
            sig.ttl -= dt
            sig.strength = max(0.0, sig.strength - dt * 0.3)
        self.signals = [s for s in self.signals if s.ttl > 0.0 and s.strength > 0.01]

        # Spec 51 (Time Dilation): each organism steps at its own tick rate.
        for org in self.organisms:
            if not org.alive:
                continue
            if hasattr(org, "genome") and hasattr(org.genome, "FLOAT_GENES"):
                continue
            org.step(dt, self, audio, self._neighbours(org))
            # Spec 63 (Innovation Protection): protect breakthrough genomes.
            if (
                org.complexity_level >= 5
                and not org.innovation_protected
                and random.random() < 0.3
            ):
                org.innovation_protected = True
                org.innovation_age = 0.0

        # Environmental niches: assign organisms to niches and
        # apply niche-based migration pressure.
        if self.niche_manager is not None and self.config.enable_niches:
            self.niche_manager.update_regions(self._get_region_bounds())
            self._assign_niches()
            self._niche_migration(dt)

        if self.config.enable_predator_prey:
            self._predation()
        if self.config.enable_infection:
            self._infection()

        # Integrate births.
        if self._spawned_this_step:
            for c in self._spawned_this_step:
                if len(self.organisms) < self.config.spatial_population_size:
                    self.organisms.append(c)
                else:
                    weakest = min(self.organisms, key=lambda o: o.objectives.scalar())
                    self.organisms.remove(weakest)
                    self.organisms.append(c)
            self._spawned_this_step = []

        self._extinction_check()
        if self.config.enable_extinction:
            alive_count = sum(1 for o in self.organisms if o.alive)
            self._maybe_revive(dt, force=alive_count < 4)
        if random.random() < 0.004:
            self._maybe_speciate()
        self._mutate_diets()

        self.organisms = [o for o in self.organisms if o.alive]
        self._update_species_stats()

        # Spec 48: Cultural evolution step.
        if getattr(self.config, "enable_cultural_evolution", True):
            self._cultural_evolution_step()

    def _cultural_evolution_step(self) -> None:
        """Run one step of cultural evolution across all species."""
        species_order = list({str(o.species) for o in self.organisms if o.alive})
        if len(species_order) < 2:
            return
        copy_rate = getattr(self.config, "cultural_copy_rate", 0.3)
        imitate_rate = getattr(self.config, "cultural_imitate_rate", 0.2)
        steal_rate = getattr(self.config, "cultural_steal_rate", 0.05)
        modify_rate = getattr(self.config, "cultural_modify_rate", 0.15)
        teach_rate = getattr(self.config, "cultural_teach_rate", 0.1)
        self.culture.step(
            species_order,
            copy_rate=copy_rate,
            imitate_rate=imitate_rate,
            steal_rate=steal_rate,
            modify_rate=modify_rate,
            teach_rate=teach_rate,
        )
        # Cross-species cultural transmission via signals.
        for sig in self.signals:
            if sig.strength < 0.5:
                continue
            src_org = None
            for o in self.organisms:
                if o.alive and o.pos[0] == sig.x and o.pos[1] == sig.y:
                    src_org = o
                    break
            if src_org is None:
                continue
            for tgt in self.organisms:
                if tgt is src_org or not tgt.alive:
                    continue
                dist = math.hypot(tgt.pos[0] - sig.x, tgt.pos[1] - sig.y)
                if dist > sig.range * 2:
                    continue
                mech = random.choices(
                    ["copy", "imitate", "steal", "teach"],
                    weights=[copy_rate, imitate_rate, steal_rate, teach_rate],
                    k=1,
                )[0]
                if mech == "copy":
                    self.culture.copy(str(src_org.species), str(tgt.species))
                elif mech == "imitate":
                    self.culture.imitate(str(src_org.species), str(tgt.species))
                elif mech == "steal":
                    self.culture.steal(str(src_org.species), str(tgt.species))
                elif mech == "teach":
                    self.culture.teach(str(src_org.species), str(tgt.species))

    def _get_region_bounds(self) -> list[dict[str, float]]:
        """Return region bounds for the current niche layout."""
        n = self.niche_manager.count if self.niche_manager else 1
        layout = getattr(self.config, "region_layout", "split_left_right")
        if layout == "split_top_bottom":
            regions = RegionLayout.split_top_bottom(n)
        elif layout == "split_quadrants":
            regions = RegionLayout.split_quadrants(n)
        elif layout == "split_grid":
            regions = RegionLayout.split_grid(n)
        else:
            regions = RegionLayout.split_left_right(n)
        return [{"x": r.x, "y": r.y, "w": r.w, "h": r.h} for r in regions]

    def _assign_niches(self) -> None:
        """Assign each organism to its current niche, keeping fitness bonuses.

        Merged implementation: earlier versions of this method also maintained
        ``org.niche_fitness_bonus`` while a later duplicate only assigned the
        name/index. Python kept only the later definition, silently dropping
        the bonus. This single method preserves both behaviours.
        """
        if self.niche_manager is None:
            for org in self.organisms:
                org.niche = "deep_field"
                org.niche_index = 0
                org.niche_fitness_bonus = 1.0
            return
        for org in self.organisms:
            if not org.alive:
                continue
            niche, idx = self.niche_manager.get_niche_for_position(org.pos[0], org.pos[1])
            org.niche = niche.name
            org.niche_index = idx
            org.niche_fitness_bonus = niche.fitness_modifier("warp", 0.5)

    def _niche_migration(self, dt: float) -> None:
        """Organisms may migrate to adjacent niches based on niche misfit."""
        if self.niche_manager is None:
            return
        rate = getattr(self.config, "niche_migration_rate", 0.01)
        for org in self.organisms:
            if not org.alive:
                continue
            if hasattr(org, "genome") and hasattr(org.genome, "FLOAT_GENES"):
                continue
            if org.niche_migration_cooldown > 0.0:
                org.niche_migration_cooldown -= dt
                continue
            if random.random() < rate * dt:
                niche, idx = self.niche_manager.get_niche_for_position(org.pos[0], org.pos[1])
                if niche.name != org.niche:
                    org.niche = niche.name
                    org.niche_index = idx
                    org.niche_migration_cooldown = random.uniform(2.0, 8.0)

    def _predation(self) -> None:
        for pred in self.organisms:
            if not pred.alive or pred.innovation_protected:
                continue
            if hasattr(pred, "genome") and hasattr(pred.genome, "FLOAT_GENES"):
                continue
            diet = self.species.get(pred.species)
            if not diet or not diet.diet:
                continue
            for prey in self.organisms:
                if not prey.alive or prey is pred or prey.innovation_protected:
                    continue
                if hasattr(prey, "genome") and hasattr(prey.genome, "FLOAT_GENES"):
                    continue
                if prey.species not in diet.diet:
                    continue
                d = math.hypot(prey.pos[0] - pred.pos[0], prey.pos[1] - pred.pos[1])
                if d < pred.radius + prey.radius + 0.02 and pred.energy < 2.2:
                    eat = min(0.06, prey.energy * 0.25)
                    pred.energy = min(self.max_energy, pred.energy + eat * 0.8)
                    prey.energy -= eat
                    if prey.energy <= 0.0:
                        prey.alive = False
                        if prey.symbiont is not None:
                            prey.symbiont.host = None
                            prey.symbiont.symbiosis_type = "none"
                            prey.symbiont = None

    def _infection(self) -> None:
        for org in self.organisms:
            if not org.alive:
                continue
            if hasattr(org, "genome") and hasattr(org.genome, "FLOAT_GENES"):
                continue
            if org.infected:
                org.infection_timer -= 0.016
                org.energy -= 0.003
                if random.random() < 0.03:
                    org.genome.mutate(strength=0.04)
                if org.infection_timer <= 0.0:
                    org.infected = False
                    org.infection_strength = 0.0
                continue
            for other in self.organisms:
                if not other.alive or other is org or not other.infected:
                    continue
                d = math.hypot(org.pos[0] - other.pos[0], org.pos[1] - other.pos[1])
                if d < org.radius + other.radius + 0.04 and random.random() < 0.02:
                    org.infected = True
                    org.infection_timer = random.uniform(3.0, 12.0)
                    org.infection_strength = other.infection_strength

    def _extinction_check(self) -> None:
        present = {o.species for o in self.organisms if o.alive}
        for sid in list(self.species.keys()):
            if sid not in present:
                best = max(
                    (o for o in self.organisms if o.species == sid),
                    key=lambda o: o.objectives.scalar(),
                    default=None,
                )
                if best is not None:
                    self.archive.archive(self.species[sid], best.genome.to_dict())
                del self.species[sid]

    def _maybe_revive(self, dt: float, force: bool = False) -> None:
        if force:
            # World is too empty: revive a dormant species immediately (spec 59 recovery).
            if self.archive.dormant:
                sid = random.choice(list(self.archive.dormant.keys()))
                self.archive.revival_cooldown = 20.0
            else:
                sid = None
        else:
            sid = self.archive.maybe_revive(dt)
        if sid is None or sid in self.species:
            return
        rec = self.archive.dormant[sid]
        new_sid = self._new_species(diet=set(rec["diet"]))
        self.species[new_sid].color = (
            min(1.0, rec["color"][0] + random.uniform(-0.2, 0.2)),
            min(1.0, rec["color"][1] + random.uniform(-0.2, 0.2)),
            min(1.0, rec["color"][2] + random.uniform(-0.2, 0.2)),
        )
        g = ModularGenome()
        try:
            g.from_dict(rec["genome"])
        except (ValueError, KeyError, TypeError) as e:
            logger.debug("Failed to revive from lineage record: %s", e)
        for _ in range(2):
            org = SpatialOrganism(g.clone(), new_sid, energy=random.uniform(0.9, 1.3))
            lvl = int(min(7, rec.get("avg_complexity", 0.0) * 7 + random.uniform(-1, 1)))
            org.complexity_level = max(0, lvl)
            org.culture = self.culture
            self.organisms.append(org)

    def _maybe_speciate(self) -> None:
        if len(self.species) >= 8:
            return
        sid = self._new_species(diet=self._random_diet(len(self.species) + 1, 0))
        org = SpatialOrganism(ModularGenome(), sid, energy=random.uniform(1.0, 1.4))
        org.culture = self.culture
        self.organisms.append(org)

    def _mutate_diets(self) -> None:
        if len(self.species) < 2:
            return
        for sp in self.species.values():
            if random.random() < 0.01:
                others = [s for s in self.species if s != sp.sid]
                if others:
                    tgt = random.choice(others)
                    if tgt in sp.diet:
                        sp.diet.discard(tgt)
                    else:
                        sp.diet.add(tgt)

    def _update_species_stats(self) -> None:
        for sid, sp in self.species.items():
            members = [o for o in self.organisms if o.alive and o.species == sid]
            if members:
                sp.avg_complexity = sum(o.complexity_level for o in members) / len(members)

    def alive_organisms(self) -> list[SpatialOrganism]:
        return [o for o in self.organisms if o.alive]


# ===========================================================================
# Feature 16 — Persistent GPU World
# ===========================================================================
class PersistentWorldTextures:
    """Persistent world textures that accumulate across frames.

    Two RGBA16F textures encode seven semantic channels:

      Texture A (u_world_tex):
        R = heat       — thermal accumulation from metabolism
        G = food       — resource deposits left by organisms
        B = life       — organism presence / life-force
        A = species    — species territorial colour

      Texture B (u_world_tex2):
        R = damage     — injury / energy-loss marks
        G = memory     — good/bad area recall overlay
        B = history    — long-term fossil / footprint layer
        A = reserved

    Unlike the feedback buffer these are NOT cleared each frame; they
    decay slowly so the renderer becomes a true persistent landscape.
    """

    def __init__(self, size: int = 256, decay: float = 0.995):
        self.size = max(32, int(size))
        self._decay_rate = max(0.9, min(1.0, float(decay)))
        self.tex_a: int | None = None
        self.tex_b: int | None = None
        self.fbo_a: int | None = None
        self.fbo_b: int | None = None
        self._write_prog: int | None = None
        self._write_loc: dict[str, int] = {}
        self._vao: int | None = None
        self._vbo: int | None = None
        self._pool: Any | None = None

    @property
    def tex(self) -> int | None:
        return self.tex_a

    @property
    def tex2(self) -> int | None:
        return self.tex_b

    def init_gl(self, pool: Any | None = None) -> None:
        # Remember the pool so re-init / resize paths reuse pooled objects.
        self._pool = pool
        # Re-initialising while the previous pairs are still held would leak
        # them from the pool, so return them first.
        if self.fbo_a is not None or self.fbo_b is not None:
            self._release_pairs()
        if pool is not None:
            self.fbo_a, self.tex_a = pool.acquire_fbo_with_texture(self.size, self.size)
            self.fbo_b, self.tex_b = pool.acquire_fbo_with_texture(self.size, self.size)
        else:
            self._create_texture_pair()

        self._create_write_program()
        self._create_quad()

    def _create_texture_pair(self) -> None:
        if self._pool is not None:
            # Reuse a pooled (fbo, tex) pair instead of allocating a fresh one.
            self.fbo_a, self.tex_a = self._pool.acquire_fbo_with_texture(self.size, self.size)
            self.fbo_b, self.tex_b = self._pool.acquire_fbo_with_texture(self.size, self.size)
            return
        for i in (0, 1):
            tex = GL.glGenTextures(1)
            fbo = GL.glGenFramebuffers(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA16F,
                self.size,
                self.size,
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
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
            )
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"World FBO {i} incomplete: {hex(status)}")
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            if i == 0:
                self.fbo_a, self.tex_a = fbo, tex
            else:
                self.fbo_b, self.tex_b = fbo, tex

    def _create_write_program(self) -> None:
        vs = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;
out vec2 v_uv;
void main(){
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""
        fs = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform vec2 u_center;
uniform float u_radius;
uniform vec4 u_color;
void main() {
    float d = length(v_uv - u_center);
    float a = 1.0 - smoothstep(0.0, u_radius, d);
    fragColor = u_color * a;
}
"""
        try:
            self._write_prog = self._compile(vs, fs)
            self._write_loc = {
                "center": GL.glGetUniformLocation(self._write_prog, "u_center"),
                "radius": GL.glGetUniformLocation(self._write_prog, "u_radius"),
                "color": GL.glGetUniformLocation(self._write_prog, "u_color"),
            }
        except Exception as e:
            logger.warning(f"World write shader compile failed: {e}")
            self._write_prog = None

    def _compile(self, vs_src: str, fs_src: str) -> int:
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vs_src)
        GL.glCompileShader(vs)
        if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
            err = GL.glGetShaderInfoLog(vs)
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Write VS: {err}")
        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, fs_src)
        GL.glCompileShader(fs)
        if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
            err = GL.glGetShaderInfoLog(fs)
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Write FS: {err}")
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            err = GL.glGetProgramInfoLog(prog)
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Write program link: {err}")
        return int(prog)

    def _create_quad(self) -> None:
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        quad = array.array(
            "f",
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
            ],
        )
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(quad) * 4, quad.tobytes(), GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 16, 0)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 16, 8)
        GL.glBindVertexArray(0)

    def write_organism(self, org: SpatialOrganism, eco: SpatialEcosystem) -> None:
        if self._write_prog is None or self.tex_a is None:
            return

        gx = org.pos[0]
        gy = org.pos[1]
        radius = max(0.005, org.radius * 2.5 / self.size)

        energy = max(0.0, min(1.0, org.energy / 3.0))
        n_species = max(1, len(eco.species))
        species_val = float(org.species) / n_species

        heat = energy * 0.7
        food = min(1.0, max(0.0, (org.energy - 1.2) * 0.6)) if org.energy > 1.2 else 0.0
        life = energy
        damage = 1.0 - energy

        recall = org.memory.recall(org.pos[0], org.pos[1])
        memory = min(1.0, math.hypot(recall[0], recall[1]) * 0.4)

        history = 0.03

        color_a = (heat, food, life, species_val)
        color_b = (damage, memory, history, 0.0)

        self._draw_blob(0, gx, gy, radius, color_a)
        self._draw_blob(1, gx, gy, radius * 0.7, color_b)

    def _draw_blob(
        self,
        tex_idx: int,
        cx: float,
        cy: float,
        radius: float,
        color: tuple[float, float, float, float],
    ) -> None:
        fbo = self.fbo_a if tex_idx == 0 else self.fbo_b
        if fbo is None or self._write_prog is None:
            return

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, self.size, self.size)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glUseProgram(self._write_prog)
        GL.glUniform2f(self._write_loc["center"], cx, cy)
        GL.glUniform1f(self._write_loc["radius"], radius)
        GL.glUniform4f(self._write_loc["color"], *color)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def decay(self) -> None:
        for fbo in (self.fbo_a, self.fbo_b):
            if fbo is None:
                continue
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glViewport(0, 0, self.size, self.size)
            GL.glBlendFunc(GL.GL_ZERO, GL.GL_SRC_COLOR)
            GL.glEnable(GL.GL_BLEND)
            GL.glClearColor(self._decay_rate, self._decay_rate, self._decay_rate, self._decay_rate)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glDisable(GL.GL_BLEND)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _release_pairs(self) -> None:
        """Return the two (fbo, tex) pairs to the pool, or delete them.

        Pooled pairs must go back through :meth:`FboPool.release` so the
        pool's bookkeeping stays coherent; deleting them directly leaves the
        pool reporting them as live forever.
        """
        if self._pool is not None:
            for fbo, tex in ((self.fbo_a, self.tex_a), (self.fbo_b, self.tex_b)):
                if fbo is not None or tex is not None:
                    try:
                        self._pool.release(fbo, tex)
                    except Exception as e:
                        logger.debug(f"World pair release error: {e}")
            self.fbo_a = self.fbo_b = None
            self.tex_a = self.tex_b = None
            return
        for tex in (self.tex_a, self.tex_b):
            if tex is not None:
                try:
                    GL.glDeleteTextures(1, [tex])
                except Exception as e:
                    logger.debug(f"World tex cleanup error: {e}")
        for fbo in (self.fbo_a, self.fbo_b):
            if fbo is not None:
                try:
                    GL.glDeleteFramebuffers(1, [fbo])
                except Exception as e:
                    logger.debug(f"World fbo cleanup error: {e}")
        self.fbo_a = self.fbo_b = None
        self.tex_a = self.tex_b = None

    def cleanup(self) -> None:
        self._release_pairs()
        if self._write_prog is not None:
            try:
                GL.glDeleteProgram(self._write_prog)
            except Exception as e:
                logger.debug(f"World write prog cleanup error: {e}")
        if self._vbo is not None:
            try:
                GL.glDeleteBuffers(1, [self._vbo])
            except Exception as e:
                logger.debug(f"World vbo cleanup error: {e}")
        if self._vao is not None:
            try:
                GL.glDeleteVertexArrays(1, [self._vao])
            except Exception as e:
                logger.debug(f"World vao cleanup error: {e}")
        self.tex_a = self.tex_b = None
        self.fbo_a = self.fbo_b = None
        self._write_prog = None
        self._write_loc = {}
        self._vao = self._vbo = None


# ===========================================================================
# Feature 24 — Environmental Events
# ===========================================================================
class EnvironmentalEvent:
    """A transient world event that modifies the ecosystem."""

    FLOOD = "flood"
    FIRE = "fire"
    COLD = "cold"
    STORM = "storm"
    DARKNESS = "darkness"
    SOLAR_FLARE = "solar_flare"
    RESOURCE_COLLAPSE = "resource_collapse"
    MUSIC_PULSE = "music_pulse"

    def __init__(self, etype: str, duration: float, intensity: float = 1.0):
        self.etype = etype
        self.duration = max(0.5, float(duration))
        self.intensity = max(0.0, min(1.0, float(intensity)))
        self.elapsed = 0.0
        self.active = True

    def step(self, dt: float) -> None:
        self.elapsed += dt
        if self.elapsed >= self.duration:
            self.active = False

    def remaining(self) -> float:
        return max(0.0, self.duration - self.elapsed)

    def apply(self, eco: SpatialEcosystem, audio: dict[str, float]) -> None:
        if not self.active:
            return
        t = self.elapsed / max(self.duration, 1e-3)
        fade = 1.0 - t
        power = self.intensity * fade
        if self.etype == EnvironmentalEvent.FLOOD:
            for org in eco.organisms:
                if org.alive:
                    org.energy -= 0.03 * power * eco.climate.temperature
                    org.vel[0] += random.uniform(-0.02, 0.02) * power
                    org.vel[1] += random.uniform(-0.01, 0.01) * power
        elif self.etype == EnvironmentalEvent.FIRE:
            for org in eco.organisms:
                if org.alive:
                    dist = math.hypot(org.pos[0] - 0.5, org.pos[1] - 0.5)
                    heat = max(0.0, 1.0 - dist * 1.5) * power
                    org.energy -= 0.05 * heat
        elif self.etype == EnvironmentalEvent.COLD:
            eco.climate.temperature = max(0.0, eco.climate.temperature - 0.02 * power)
            for org in eco.organisms:
                if org.alive:
                    org.energy -= 0.015 * power
        elif self.etype == EnvironmentalEvent.STORM:
            for org in eco.organisms:
                if org.alive:
                    org.vel[0] += random.uniform(-0.05, 0.05) * power
                    org.vel[1] += random.uniform(-0.05, 0.05) * power
        elif self.etype == EnvironmentalEvent.DARKNESS:
            eco.climate.light_level = max(0.0, eco.climate.light_level - 0.03 * power)
        elif self.etype == EnvironmentalEvent.SOLAR_FLARE:
            for org in eco.organisms:
                if org.alive:
                    org.energy += 0.02 * power
                    if random.random() < 0.1 * power:
                        org.genome.mutate(strength=0.15)
        elif self.etype == EnvironmentalEvent.RESOURCE_COLLAPSE:
            eco.field.deplete(power * 0.3)
        elif self.etype == EnvironmentalEvent.MUSIC_PULSE:
            beat = float(audio.get("beat", 0.0))
            for org in eco.organisms:
                if org.alive:
                    org.energy += 0.01 * beat * power
                    if beat > 0.7 and random.random() < 0.2 * power:
                        org.genome.mutate(strength=0.08)


# ===========================================================================
# Feature 25 — Enhanced Memory (bad/good area recall)
# ===========================================================================
class AreaMemory:
    """Organism remembers good and bad locations."""

    def __init__(self, capacity: int = 16):
        self.capacity = max(4, int(capacity))
        self.good: list[tuple[float, float, float]] = []
        self.bad: list[tuple[float, float, float]] = []

    def record(self, x: float, y: float, reward: float) -> None:
        entry = (x, y, max(0.0, reward))
        if reward > 0.0:
            self.good.append(entry)
            if len(self.good) > self.capacity:
                self.good = self.good[-self.capacity :]
        else:
            self.bad.append(entry)
            if len(self.bad) > self.capacity:
                self.bad = self.bad[-self.capacity :]

    def recall(self, x: float, y: float) -> tuple[float, float]:
        gx = gy = 0.0
        for gx2, gy2, w in self.good:
            d = math.hypot(gx2 - x, gy2 - y)
            if d < 1e-3:
                continue
            influence = w / (d * 8.0)
            gx += (gx2 - x) * influence
            gy += (gy2 - y) * influence
        bx = by = 0.0
        for bx2, by2, w in self.bad:
            d = math.hypot(bx2 - x, by2 - y)
            if d < 1e-3:
                continue
            influence = w / (d * 8.0)
            bx += (x - bx2) * influence
            by += (y - by2) * influence
        return (gx + bx, gy + by)


# ===========================================================================
# Feature 20 — Soft Body Organisms (procedural body parts)
# ===========================================================================
class BodyPart:
    """One procedural body segment."""

    HEAD = "head"
    BODY = "body"
    TAIL = "tail"
    TENTACLE = "tentacle"
    LEAF = "leaf"
    BRANCH = "branch"
    ROOT = "root"

    def __init__(self, ptype: str, length: float = 0.04, width: float = 0.015):
        self.ptype = ptype
        self.length = max(0.005, float(length))
        self.width = max(0.002, float(width))
        self.angle = random.uniform(0.0, math.tau)
        self.phase = random.uniform(0.0, math.tau)

    def step(self, dt: float, audio: dict[str, float]) -> None:
        bass = float(audio.get("bass", 0.0))
        self.angle += math.sin(time.time() * 2.0 + self.phase) * 0.3 * dt
        self.angle += bass * 0.5 * dt


class SoftBody:
    """Procedurally generated body made of parts."""

    def __init__(self, complexity: int = 3):
        self.parts: list[BodyPart] = []
        self.complexity = max(1, min(7, int(complexity)))
        self._generate()

    def _generate(self) -> None:
        self.parts.append(BodyPart(BodyPart.HEAD, 0.03, 0.025))
        for i in range(self.complexity):
            if i % 3 == 0:
                self.parts.append(BodyPart(BodyPart.BODY, 0.04, 0.018))
            elif i % 3 == 1:
                self.parts.append(BodyPart(BodyPart.TENTACLE, 0.05, 0.008))
            else:
                self.parts.append(BodyPart(BodyPart.LEAF, 0.03, 0.012))
        if self.complexity > 3:
            self.parts.append(BodyPart(BodyPart.TAIL, 0.06, 0.01))

    def step(self, dt: float, audio: dict[str, float]) -> None:
        for part in self.parts:
            part.step(dt, audio)

    def mutate(self, strength: float = 0.1) -> None:
        if random.random() < strength:
            self.complexity = max(1, min(7, self.complexity + random.choice([-1, 0, 1])))
            self._generate()
        for part in self.parts:
            if random.random() < strength:
                part.length = max(0.005, part.length + random.uniform(-0.01, 0.01))
                part.width = max(0.002, part.width + random.uniform(-0.005, 0.005))


# ===========================================================================
# Feature 27 — Procedural Skeletons (L-system grammar)
# ===========================================================================
class SkeletonNode:
    def __init__(self, length: float, angle: float, depth: int):
        self.length = max(0.005, float(length))
        self.angle = float(angle)
        self.depth = int(depth)
        self.children: list[SkeletonNode] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "angle": self.angle,
            "depth": self.depth,
            "children": [c.to_dict() for c in self.children],
        }


def _grow_skeleton(depth: int, length: float, angle: float) -> SkeletonNode:
    node = SkeletonNode(length, angle, depth)
    if depth <= 0:
        return node
    n_branches = random.randint(2, 3)
    for _ in range(n_branches):
        child_len = length * random.uniform(0.6, 0.85)
        child_angle = angle + random.uniform(-0.8, 0.8)
        node.children.append(_grow_skeleton(depth - 1, child_len, child_angle))
    return node


class ProceduralSkeleton:
    """L-system-like grammar for organism body structure."""

    def __init__(self, depth: int = 3, base_length: float = 0.08):
        self.depth = max(1, min(5, int(depth)))
        self.base_length = max(0.01, float(base_length))
        self.root: SkeletonNode | None = None
        self.leaf_count = 0
        self.branch_count = 0
        self.generate()

    def generate(self) -> None:
        self.root = _grow_skeleton(self.depth, self.base_length, 0.0)
        self._count()

    def _count(self) -> None:
        if self.root is None:
            return
        stack = [self.root]
        self.leaf_count = 0
        self.branch_count = 0
        while stack:
            node = stack.pop()
            if node.depth == 0:
                self.leaf_count += 1
            else:
                self.branch_count += 1
            stack.extend(node.children)

    def mutate(self, strength: float = 0.1) -> None:
        if random.random() < strength:
            self.depth = max(1, min(5, self.depth + random.choice([-1, 0, 1])))
        if random.random() < strength:
            self.base_length = max(0.01, self.base_length + random.uniform(-0.01, 0.01))
        if self.root and random.random() < strength * 0.5:
            self._mutate_node(self.root)
        self.generate()

    def _mutate_node(self, node: SkeletonNode) -> None:
        if random.random() < 0.3 and node.children:
            idx = random.randrange(len(node.children))
            node.children[idx].length *= random.uniform(0.8, 1.25)
            node.children[idx].angle += random.uniform(-0.3, 0.3)
        if random.random() < 0.2:
            node.angle += random.uniform(-0.2, 0.2)


# ===========================================================================
# Feature 28 — Dynamic Materials
# ===========================================================================
MATERIAL_TYPES = ["organic", "metal", "glass", "plasma", "smoke", "crystal", "water", "neon"]

MATERIAL_COLOURS = {
    "organic": (0.2, 0.6, 0.3),
    "metal": (0.7, 0.7, 0.75),
    "glass": (0.6, 0.8, 0.9),
    "plasma": (0.4, 0.2, 0.8),
    "smoke": (0.4, 0.4, 0.4),
    "crystal": (0.5, 0.3, 0.9),
    "water": (0.2, 0.4, 0.8),
    "neon": (0.1, 0.9, 0.4),
}


class DynamicMaterial:
    """Evolving material properties for body parts."""

    def __init__(self, mat_type: str = "organic"):
        self.mat_type = mat_type if mat_type in MATERIAL_TYPES else "organic"
        self.colour = list(MATERIAL_COLOURS.get(self.mat_type, (0.5, 0.5, 0.5)))
        self.metallic = 0.0 if self.mat_type == "organic" else random.uniform(0.3, 0.9)
        self.roughness = random.uniform(0.1, 0.9)
        self.emission = random.uniform(0.0, 0.5)
        self.transparency = (
            0.0 if self.mat_type in ("glass", "water") else random.uniform(0.0, 0.2)
        )

    def mutate(self, strength: float = 0.1) -> None:
        if random.random() < strength:
            self.mat_type = random.choice(MATERIAL_TYPES)
            self.colour = list(MATERIAL_COLOURS.get(self.mat_type, (0.5, 0.5, 0.5)))
        for attr in ("metallic", "roughness", "emission", "transparency"):
            if random.random() < strength:
                val = getattr(self, attr)
                setattr(self, attr, max(0.0, min(1.0, val + random.uniform(-0.1, 0.1))))

    def to_uniforms(self) -> dict[str, Any]:
        return {
            "u_mat_type": MATERIAL_TYPES.index(self.mat_type),
            "u_mat_colour": tuple(self.colour),
            "u_mat_metallic": self.metallic,
            "u_mat_roughness": self.roughness,
            "u_mat_emission": self.emission,
            "u_mat_transparency": self.transparency,
        }


# ===========================================================================
# Feature 30 — Long-Term Artificial Evolution (fossil record)
# ===========================================================================
class FossilRecord:
    """Persistent evolutionary history across sessions."""

    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else Path("fossil_record.json")
        self.generations: list[dict[str, Any]] = []
        self.extinct_species: list[dict[str, Any]] = []
        self.evolutionary_branches: list[dict[str, Any]] = []
        self.major_innovations: list[dict[str, Any]] = []
        self.environmental_history: list[dict[str, Any]] = []
        self._load()

    def record_generation(
        self, gen: int, best_genome: dict[str, Any], best_fitness: float, species_count: int
    ) -> None:
        entry = {
            "generation": gen,
            "best_genome": best_genome,
            "best_fitness": best_fitness,
            "species_count": species_count,
            "timestamp": time.time(),
        }
        self.generations.append(entry)
        if len(self.generations) > 10000:
            self.generations = self.generations[-5000:]
        self._detect_innovation(entry)
        self._save()

    def record_extinction(self, species_id: str, genome: dict[str, Any], reason: str) -> None:
        entry = {
            "species_id": species_id,
            "genome": genome,
            "reason": reason,
            "timestamp": time.time(),
        }
        self.extinct_species.append(entry)
        self._save()

    def record_branch(self, parent: str, child: str, innovation: str) -> None:
        entry = {
            "parent": parent,
            "child": child,
            "innovation": innovation,
            "timestamp": time.time(),
        }
        self.evolutionary_branches.append(entry)
        self._save()

    def record_environment(self, event: str, intensity: float) -> None:
        entry = {
            "event": event,
            "intensity": intensity,
            "timestamp": time.time(),
        }
        self.environmental_history.append(entry)
        if len(self.environmental_history) > 2000:
            self.environmental_history = self.environmental_history[-1000:]
        self._save()

    def _detect_innovation(self, entry: dict[str, Any]) -> None:
        if not self.generations:
            return
        prev = self.generations[-1]
        fit_gain = entry["best_fitness"] - prev.get("best_fitness", 0.0)
        if fit_gain > 0.15:
            self.major_innovations.append(
                {
                    "at_generation": entry["generation"],
                    "fitness_gain": fit_gain,
                    "genome": entry["best_genome"],
                    "timestamp": time.time(),
                }
            )

    def lineage_summary(self) -> str:
        return (
            f"FossilRecord: {len(self.generations)} generations, "
            f"{len(self.extinct_species)} extinct, "
            f"{len(self.evolutionary_branches)} branches, "
            f"{len(self.major_innovations)} innovations"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generations": self.generations[-100:],
            "extinct_species": self.extinct_species[-50:],
            "evolutionary_branches": self.evolutionary_branches[-50:],
            "major_innovations": self.major_innovations[-20:],
            "environmental_history": self.environmental_history[-100:],
        }

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        except (OSError, TypeError) as e:
            logger.debug("Failed to save extinction archive: %s", e)

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                self.generations = data.get("generations", [])
                self.extinct_species = data.get("extinct_species", [])
                self.evolutionary_branches = data.get("evolutionary_branches", [])
                self.major_innovations = data.get("major_innovations", [])
                self.environmental_history = data.get("environmental_history", [])
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Failed to load extinction archive: %s", e)


# ===========================================================================
# Feature 22 — Multi-Species Ecology (enhanced species roles)
# ===========================================================================
class PlantSpecies:
    """Static energy-producing organism."""

    def __init__(self, pos: tuple[float, float]):
        self.pos = list(pos)
        self.energy = random.uniform(0.8, 1.5)
        self.radius = random.uniform(0.02, 0.05)
        self.alive = True
        self.growth_rate = random.uniform(0.01, 0.04)

    def step(self, dt: float, eco: SpatialEcosystem) -> None:
        if not self.alive:
            return
        self.energy += self.growth_rate * eco.climate.light_level * dt
        self.energy = min(2.5, self.energy)
        if random.random() < 0.005 * dt:
            self._spread(eco)

    def _spread(self, eco: SpatialEcosystem) -> None:
        if len(eco.plants) >= 20:
            return
        angle = random.uniform(0.0, math.tau)
        dist = random.uniform(0.05, 0.15)
        nx = max(0.02, min(0.98, self.pos[0] + math.cos(angle) * dist))
        ny = max(0.02, min(0.98, self.pos[1] + math.sin(angle) * dist))
        eco.plants.append(PlantSpecies((nx, ny)))


class EnhancedSpatialEcosystem(SpatialEcosystem):
    """Extended ecosystem with plants, events, world texture, and fossil record."""

    def __init__(self, config: EvoConfig):
        super().__init__(config)
        self.plants: list[PlantSpecies] = []
        self.world_tex = PersistentWorldTextures(
            size=self.config.world_texture_size,
            decay=self.config.world_texture_decay,
        )
        self.fossil = FossilRecord()
        self.event_log: list[str] = []
        for _ in range(8):
            self.plants.append(PlantSpecies((random.random(), random.random())))
        # Specs 16-30: developmental evolutionary supervisor.
        self.advanced = AdvancedEcosystem(self, config)

    def step(self, dt: float, audio: dict[str, float], music_dna: Any = None) -> None:
        super().step(dt, audio, music_dna)
        # Drive the advanced developmental systems (GRN, development, novelty,
        # emotion, sexual selection, renderer/materials evolution, niches,
        # replay, archaeology, self-modifying evolution) over this ecosystem.
        try:
            self.advanced.step(dt, audio)
        except Exception as e:
            logger.debug("Advanced ecosystem step failed: %s", e)
        if hasattr(self.climate, "step_events"):
            self.climate.step_events(dt, self, audio)
        for plant in self.plants:
            plant.step(dt, self)
        self.plants = [p for p in self.plants if p.alive]
        if len(self.plants) < 3 and random.random() < 0.1:
            self.plants.append(PlantSpecies((random.random(), random.random())))
        if self.world_tex.tex is not None:
            for org in self.organisms:
                if org.alive:
                    self.world_tex.write_organism(org, self)
            self.world_tex.decay()
        if self.step_count % 30 == 0:
            best = max(
                (o for o in self.organisms if o.alive), key=lambda o: o.fitness, default=None
            )
            if best is not None:
                self.fossil.record_generation(
                    self.step_count,
                    best.genome.to_dict() if hasattr(best.genome, "to_dict") else {},
                    best.fitness,
                    len(self.species),
                )
            for ev in getattr(self.climate, "events", []):
                if ev.active and ev.etype not in self.event_log:
                    self.event_log.append(ev.etype)
                    self.fossil.record_environment(ev.etype, ev.intensity)
                    if len(self.event_log) > 100:
                        self.event_log = self.event_log[-50:]
        if random.random() < dt / max(
            getattr(self.config, "champion_migration_interval_secs", 30.0), 5.0
        ):
            self._migrate_champions()

    def _migrate_champions(self) -> None:
        """Exchange champion genomes between random species (island migration)."""
        from evo_bio.pareto import ParetoFront

        if len(self.species) < 2:
            return
        alive = [o for o in self.organisms if o.alive]
        if not alive:
            return
        by_species: dict[int, list[SpatialOrganism]] = {}
        for o in alive:
            by_species.setdefault(o.species, []).append(o)
        if len(by_species) < 2:
            return
        sids = list(by_species.keys())
        a, b = random.sample(sids, 2)
        front_a = ParetoFront()
        front_b = ParetoFront()
        for o in by_species[a]:
            front_a.add(o.lineage, o.objectives)
        for o in by_species[b]:
            front_b.add(o.lineage, o.objectives)
        front_a.build()
        front_b.build()
        best_a_id = front_a.best()
        best_b_id = front_b.best()
        champ_a = next(o for o in by_species[a] if o.lineage == best_a_id)
        champ_b = next(o for o in by_species[b] if o.lineage == best_b_id)
        if champ_a.objectives.dominates(champ_b.objectives):
            champ_b.genome = champ_a.genome.clone()
            champ_b.genome.mutate(strength=0.05)
        else:
            champ_a.genome = champ_b.genome.clone()
            champ_a.genome.mutate(strength=0.05)
