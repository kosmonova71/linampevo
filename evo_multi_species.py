#!/usr/bin/env python3

"""evo_multi_species.py

Multi-species renderer with niche differentiation for the EvoRenderer project.

Expands the spatial ecosystem from ~4-8 species to up to 20, each occupying
a distinct ecological niche with unique rendering behaviours:

  * fog_layers       - thick volumetric fog, low contrast
  * trail_makers     - heavy feedback, long persistent trails
  * geometry_guild   - SDF / mandelbrot field types, structural
  * colour_weavers   - plasma / spectral palettes, high colour variance
  * parasites        - small organisms that drain host energy
  * deep_field       - fbm, slow, low-frequency
  * bright_field     - glow-heavy, bloom, high brightness
  * chaotic_region   - vortex, turbulence, high motion
  * calm_region      - reaction_diffusion, low turbulence
  * music_hotspot    - audio-reactive, beat-driven mutations
  * resource_valley  - deposit resources, feed others

Integration
-----------
Designed to be imported by ``evo_visualizer.py``.  It extends the existing
``SpatialEcosystem`` and ``SpatialOrganism`` classes via composition rather
than deep inheritance so the existing code keeps working.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

MAX_SPECIES = 20
SPECIATION_DISTANCE = 0.35
NICHE_RECOMBINE_INTERVAL = 45.0

# Enhanced speciation parameters
MIN_GENETIC_DISTANCE_FOR_SPECIATION = 0.25  # Minimum genetic divergence to speciate
SPECIATION_ENERGY_THRESHOLD = 1.5  # Energy needed to attempt speciation
SPECIATION_COOLDOWN = 30.0  # Minimum time between speciation events per species
MAX_SPECIES_PER_NICHE = 3  # Maximum species allowed in one niche
SPECIATION_NICHE_DIVERGENCE = 0.15  # Niche fitness difference to trigger speciation

NICHE_TYPES = (
    "fog_layers",
    "trail_makers",
    "geometry_guild",
    "colour_weavers",
    "parasites",
    "deep_field",
    "bright_field",
    "chaotic_region",
    "calm_region",
    "music_hotspot",
    "resource_valley",
)

NICHE_FIELD_AFFINITY: dict[str, list[str]] = {
    "fog_layers": ["fbm", "reaction_diffusion"],
    "trail_makers": ["fbm", "voronoi"],
    "geometry_guild": ["sdf", "mandelbrot"],
    "colour_weavers": ["plasma", "spectral", "cosine"],
    "parasites": ["fbm", "voronoi"],
    "deep_field": ["fbm"],
    "bright_field": ["fbm", "voronoi"],
    "chaotic_region": ["voronoi", "reaction_diffusion"],
    "calm_region": ["reaction_diffusion", "fbm"],
    "music_hotspot": ["mandelbrot", "sdf"],
    "resource_valley": ["fbm", "reaction_diffusion"],
}

NICHE_WARP_AFFINITY: dict[str, list[str]] = {
    "fog_layers": ["none", "curl"],
    "trail_makers": ["vortex", "twist"],
    "geometry_guild": ["none", "curl"],
    "colour_weavers": ["none", "curl"],
    "parasites": ["none", "curl"],
    "deep_field": ["none", "curl"],
    "bright_field": ["vortex", "twist"],
    "chaotic_region": ["vortex", "twist"],
    "calm_region": ["none", "curl"],
    "music_hotspot": ["vortex", "twist"],
    "resource_valley": ["none", "curl"],
}

NICHE_POST_AFFINITY: dict[str, list[str]] = {
    "fog_layers": ["glow", "feedback"],
    "trail_makers": ["feedback", "nonlinear_trails", "distortion"],
    "geometry_guild": ["glow", "neon_edge", "depth_cloud"],
    "colour_weavers": ["chromatic_aberration", "bloom", "polar_galaxy"],
    "parasites": ["feedback", "glow"],
    "deep_field": ["feedback", "glow"],
    "bright_field": ["bloom", "chromatic_aberration"],
    "chaotic_region": ["distortion", "nonlinear_trails", "feedback"],
    "calm_region": ["glow", "feedback"],
    "music_hotspot": ["audio_topology", "neon_edge"],
    "resource_valley": ["feedback", "cells"],
}

NICHE_PALETTE_AFFINITY: dict[str, list[str]] = {
    "fog_layers": ["cosine"],
    "trail_makers": ["cosine", "plasma"],
    "geometry_guild": ["cosine", "spectral"],
    "colour_weavers": ["plasma", "spectral"],
    "parasites": ["cosine"],
    "deep_field": ["cosine"],
    "bright_field": ["cosine", "spectral"],
    "chaotic_region": ["plasma"],
    "calm_region": ["cosine", "spectral"],
    "music_hotspot": ["plasma", "spectral"],
    "resource_valley": ["cosine"],
}

NICHE_COLOR: dict[str, tuple[float, float, float]] = {
    "fog_layers": (0.6, 0.6, 0.7),
    "trail_makers": (0.9, 0.4, 0.2),
    "geometry_guild": (0.3, 0.8, 0.9),
    "colour_weavers": (0.9, 0.3, 0.8),
    "parasites": (0.2, 0.2, 0.2),
    "deep_field": (0.2, 0.3, 0.5),
    "bright_field": (1.0, 0.9, 0.7),
    "chaotic_region": (0.9, 0.2, 0.3),
    "calm_region": (0.3, 0.7, 0.5),
    "music_hotspot": (0.8, 0.6, 1.0),
    "resource_valley": (0.4, 0.8, 0.3),
}


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class Niche:
    """A world niche with type, position, radius and affinity."""

    ntype: str
    cx: float = 0.5
    cy: float = 0.5
    radius: float = 0.25
    affinity: float = 0.5
    energy_bonus: float = 0.0

    def in_range(self, x: float, y: float) -> bool:
        return math.hypot(x - self.cx, y - self.cy) <= self.radius

    def suitability(
        self, audio: dict[str, float], field_type: str, warp: str, post: list[str]
    ) -> float:
        score = 0.0
        if field_type in NICHE_FIELD_AFFINITY.get(self.ntype, []):
            score += 0.4
        if warp in NICHE_WARP_AFFINITY.get(self.ntype, []):
            score += 0.3
        overlap = len(set(post) & set(NICHE_POST_AFFINITY.get(self.ntype, [])))
        score += 0.3 * min(1.0, overlap / 2.0)
        return min(1.0, score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ntype": self.ntype,
            "cx": self.cx,
            "cy": self.cy,
            "radius": self.radius,
            "affinity": self.affinity,
            "energy_bonus": self.energy_bonus,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Niche:
        return cls(
            ntype=d.get("ntype", "deep_field"),
            cx=float(d.get("cx", 0.5)),
            cy=float(d.get("cy", 0.5)),
            radius=float(d.get("radius", 0.25)),
            affinity=float(d.get("affinity", 0.5)),
            energy_bonus=float(d.get("energy_bonus", 0.0)),
        )


@dataclass
class SpeciesDef:
    """A species definition with colour, diet, biome affinity and niche."""

    sid: int
    color: tuple[float, float, float]
    diet: set[int] = dc_field(default_factory=set)
    biome_affinity: float = 0.5
    born_at: float = 0.0
    avg_complexity: float = 0.0
    niche: str = "deep_field"
    niche_score: float = 0.0
    extinction_timer: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sid": self.sid,
            "color": list(self.color),
            "diet": list(self.diet),
            "biome_affinity": self.biome_affinity,
            "born_at": self.born_at,
            "avg_complexity": self.avg_complexity,
            "niche": self.niche,
            "niche_score": self.niche_score,
            "extinction_timer": self.extinction_timer,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpeciesDef:
        return cls(
            sid=int(d.get("sid", 0)),
            color=tuple(d.get("color", (0.5, 0.5, 0.5))),
            diet=set(d.get("diet", [])),
            biome_affinity=float(d.get("biome_affinity", 0.5)),
            born_at=float(d.get("born_at", 0.0)),
            avg_complexity=float(d.get("avg_complexity", 0.0)),
            niche=d.get("niche", "deep_field"),
            niche_score=float(d.get("niche_score", 0.0)),
            extinction_timer=float(d.get("extinction_timer", 0.0)),
        )


# ============================================================================
# Niche map
# ============================================================================


class NicheMap:
    """Discovers and maintains ecosystem niches over time."""

    def __init__(self, target: int = 8):
        self.target = max(2, min(MAX_SPECIES // 2, int(target)))
        self.niches: list[Niche] = []
        self._spawn_timer = 0.0
        for _ in range(self.target):
            self.niches.append(self._random_niche())

    def _random_niche(self) -> Niche:
        ntype = random.choice(NICHE_TYPES)
        return Niche(
            ntype=ntype,
            cx=random.uniform(0.1, 0.9),
            cy=random.uniform(0.1, 0.9),
            radius=random.uniform(0.15, 0.4),
        )

    def step(self, audio: dict[str, float], dt: float, intensity: float = 0.5) -> None:
        self._spawn_timer -= dt
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        for n in self.niches:
            n.cx = max(0.05, min(0.95, n.cx + (treb - bass) * 0.02 * dt))
            n.cy = max(0.05, min(0.95, n.cy + (bass - treb) * 0.02 * dt))
            n.affinity = 0.9 * n.affinity + 0.1 * intensity
        if self._spawn_timer <= 0.0 and len(self.niches) < self.target + 3:
            self._spawn_timer = random.uniform(20.0, 50.0)
            self.niches.append(self._random_niche())
        self.niches = [n for n in self.niches if n.affinity > 0.08 or random.random() < 0.995]
        while len(self.niches) < self.target:
            self.niches.append(self._random_niche())

    def best_for(self, field_type: str, warp: str, post: list[str]) -> Niche | None:
        best, best_fit = None, -1.0
        for n in self.niches:
            fit = n.suitability({}, field_type, warp, post)
            if fit > best_fit:
                best_fit = fit
                best = n
        return best

    def assign_species_niche(self, species_id: int, genome: Any) -> str:
        field = getattr(genome, "field_type", "fbm")
        warp = getattr(genome, "warp", "none")
        post = getattr(genome, "post", []) or []
        best = self.best_for(field, warp, post)
        return best.ntype if best else random.choice(NICHE_TYPES)

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "niches": [n.to_dict() for n in self.niches]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NicheMap:
        m = cls(d.get("target", 8))
        m.niches = [Niche.from_dict(n) for n in d.get("niches", [])]
        return m


# ============================================================================
# Multi-species ecosystem extensions
# ============================================================================


class MultiSpeciesEcosystem:
    """Extends SpatialEcosystem with multi-species rendering behaviours.

    This class is intentionally composition-based: it holds references to the
    host ecosystem and adds species-level rendering logic without modifying the
    host's step() loop.
    """

    def __init__(self, host: Any, config: Any):
        self.host = host
        self.config = config
        self.niche_map = NicheMap(target=min(10, max(4, config.spatial_population_size // 3)))
        self._speciation_timer = 0.0
        self._recombine_timer = 0.0

    # ------------------------------------------------------------------
    # Species creation / speciation
    # ------------------------------------------------------------------

    def maybe_speciate(self, parent_org: Any) -> int | None:
        """Return a new species id if ``parent_org`` should speciate.

        Speciation criteria:
        1. Population has room for new species (MAX_SPECIES)
        2. Parent has sufficient energy (SPECIATION_ENERGY_THRESHOLD)
        3. Genetic distance from parent species centroid exceeds threshold
        4. Niche has capacity for new species (MAX_SPECIES_PER_NICHE)
        5. Sufficient time since last speciation event (SPECIATION_COOLDOWN)
        6. Niche fitness divergence suggests adaptive radiation
        """
        if len(self.host.species) >= MAX_SPECIES:
            return None

        # Energy threshold - only thriving organisms can speciate
        if parent_org.energy < SPECIATION_ENERGY_THRESHOLD:
            return None

        # Check speciation cooldown for this species
        sp = self.host.species.get(parent_org.species)
        if sp is not None:
            last_speciation = getattr(sp, 'last_speciation_time', 0.0)
            if self.host.t - last_speciation < SPECIATION_COOLDOWN:
                return None

        # Check niche capacity
        if self.niche_map is not None:
            niche_counts: dict[str, int] = {}
            for s in self.host.species.values():
                if s.niche in NICHE_TYPES:
                    niche_counts[s.niche] = niche_counts.get(s.niche, 0) + 1
            parent_niche = sp.niche if sp else None
            if parent_niche and niche_counts.get(parent_niche, 0) >= MAX_SPECIES_PER_NICHE:
                # Try to find a niche with capacity
                available_niches = [n for n in NICHE_TYPES if niche_counts.get(n, 0) < MAX_SPECIES_PER_NICHE]
                if not available_niches:
                    return None
            else:
                available_niches = [parent_niche] if parent_niche else list(NICHE_TYPES)

        # Calculate genetic distance from species centroid
        genetic_dist = self._genetic_distance_from_species(parent_org)
        if genetic_dist < MIN_GENETIC_DISTANCE_FOR_SPECIATION:
            return None

        # Check niche fitness divergence (adaptive radiation)
        niche_divergence = self._niche_fitness_divergence(parent_org)
        if niche_divergence < SPECIATION_NICHE_DIVERGENCE:
            # Not enough niche differentiation
            pass  # Still allow, but lower probability

        # Probability increases with genetic distance and niche divergence
        speciation_prob = 0.003 * (1.0 + genetic_dist * 2.0 + niche_divergence * 3.0)
        if random.random() > speciation_prob:
            return None

        # Determine target niche for new species
        target_niche = None
        if self.niche_map is not None and available_niches:
            # Prefer niches with fewer species
            target_niche = min(available_niches, key=lambda n: niche_counts.get(n, 0))

        # Create new species with mutated genome
        sid = self._new_species(niche=target_niche)
        genome = parent_org.genome.clone()
        # Stronger mutation for speciation event
        genome.mutate(strength=0.35)
        from evo.modular_genome import ModularGenome

        if isinstance(genome, ModularGenome):
            genome.metamorphosis()

        # Record speciation time
        if sp is not None:
            sp.last_speciation_time = self.host.t

        # Bias new species genome toward its niche
        self.niche_bias_genome(genome)

        return sid

    def _genetic_distance_from_species(self, org: Any) -> float:
        """Calculate genetic distance from the species centroid."""
        sp = self.host.species.get(org.species)
        if sp is None:
            return 1.0

        # Find all living members of this species
        members = [o for o in self.host.organisms if o.alive and o.species == org.species]
        if len(members) < 2:
            return 1.0

        # Calculate centroid genome
        centroid = self._compute_species_centroid(members)
        if centroid is None:
            return 1.0

        # Distance from centroid
        return org.genome.distance(centroid)

    def _compute_species_centroid(self, members: list[Any]) -> Any | None:
        """Compute the centroid genome for a list of organisms."""
        if not members:
            return None

        # Use first member as template
        centroid = members[0].genome.clone()
        param_sums: dict[str, float] = {}
        param_counts: dict[str, int] = {}

        for m in members:
            params = m.genome.params()
            for k, v in params.items():
                param_sums[k] = param_sums.get(k, 0.0) + v
                param_counts[k] = param_counts.get(k, 0) + 1

        for k in param_sums:
            if param_counts[k] > 0:
                setattr(centroid, k, param_sums[k] / param_counts[k])

        centroid.clamp()
        return centroid

    def _niche_fitness_divergence(self, org: Any) -> float:
        """Calculate how much this organism's niche fitness differs from species average."""
        sp = self.host.species.get(org.species)
        if sp is None or self.niche_map is None:
            return 0.0

        members = [o for o in self.host.organisms if o.alive and o.species == org.species]
        if not members:
            return 0.0

        # Get niche for this organism
        org_niche, _ = self.niche_map.best_for(
            getattr(org.genome, 'field_type', 'fbm'),
            getattr(org.genome, 'warp', 'none'),
            getattr(org.genome, 'post', []) or []
        )
        if org_niche is None:
            return 0.0

        # Calculate average niche suitability for species
        suitabilities = []
        for m in members:
            niche, _ = self.niche_map.best_for(
                getattr(m.genome, 'field_type', 'fbm'),
                getattr(m.genome, 'warp', 'none'),
                getattr(m.genome, 'post', []) or []
            )
            if niche:
                suitabilities.append(niche.suitability(
                    {},  # audio not needed for static check
                    getattr(m.genome, 'field_type', 'fbm'),
                    getattr(m.genome, 'warp', 'none'),
                    getattr(m.genome, 'post', []) or []
                ))

        if not suitabilities:
            return 0.0

        avg_suitability = sum(suitabilities) / len(suitabilities)
        org_suitability = org_niche.suitability(
            {},
            getattr(org.genome, 'field_type', 'fbm'),
            getattr(org.genome, 'warp', 'none'),
            getattr(org.genome, 'post', []) or []
        )

        return abs(org_suitability - avg_suitability)

    # ------------------------------------------------------------------
    # Species-level fitness and extinction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Species-level fitness and extinction
    # ------------------------------------------------------------------

    def species_fitness(self, sid: int) -> float:
        """Calculate species fitness using multi-objective Pareto evaluation."""
        members = [o for o in self.host.organisms if o.alive and o.species == sid]
        if not members:
            return 0.0

        # Use Pareto scalar fitness if available
        total = 0.0
        for m in members:
            if hasattr(m, 'objectives') and hasattr(m.objectives, 'scalar'):
                total += m.objectives.scalar()
            else:
                total += m.fitness
        return total / len(members)

    def species_niche_fitness(self, sid: int) -> float:
        """Calculate how well-adapted the species is to its niche."""
        sp = self.host.species.get(sid)
        if sp is None or self.niche_map is None:
            return 0.5

        members = [o for o in self.host.organisms if o.alive and o.species == sid]
        if not members:
            return 0.0

        niche = sp.niche
        if niche not in NICHE_TYPES:
            return 0.5

        # Average niche suitability
        suitabilities = []
        for m in members:
            niche_obj, _ = self.niche_map.best_for(
                getattr(m.genome, 'field_type', 'fbm'),
                getattr(m.genome, 'warp', 'none'),
                getattr(m.genome, 'post', []) or []
            )
            if niche_obj and niche_obj.ntype == niche:
                suit = niche_obj.suitability(
                    {},
                    getattr(m.genome, 'field_type', 'fbm'),
                    getattr(m.genome, 'warp', 'none'),
                    getattr(m.genome, 'post', []) or []
                )
                suitabilities.append(suit)

        if not suitabilities:
            return 0.5
        return sum(suitabilities) / len(suitabilities)

    def extinction_check(self, dt: float) -> None:
        present = {o.species for o in self.host.organisms if o.alive}
        for sid in list(self.host.species.keys()):
            if sid in present:
                self.host.species[sid].extinction_timer = 0.0
                # Update species stats
                self._update_species_stats(sid)
                continue

            self.host.species[sid].extinction_timer += dt
            # Extinction threshold varies by species fitness
            sp = self.host.species[sid]
            niche_fitness = self.species_niche_fitness(sid)
            # Well-adapted species get more time
            threshold = 60.0 * (0.5 + niche_fitness)
            if sp.extinction_timer > threshold:
                best = max(
                    (o for o in self.host.organisms if o.species == sid),
                    key=lambda o: o.fitness,
                    default=None,
                )
                if best is not None:
                    self.host.archive.archive(
                        self.host.species[sid],
                        best.genome.to_dict() if hasattr(best.genome, "to_dict") else {},
                    )
                del self.host.species[sid]

    def _update_species_stats(self, sid: int) -> None:
        """Update species statistics."""
        sp = self.host.species.get(sid)
        if sp is None:
            return
        members = [o for o in self.host.organisms if o.alive and o.species == sid]
        if members:
            sp.avg_complexity = sum(o.complexity_level for o in members) / len(members)
            # Track niche fitness
            sp.niche_score = self.species_niche_fitness(sid)
            # Track genetic diversity within species
            if len(members) > 1:
                centroid = self._compute_species_centroid(members)
                if centroid:
                    avg_dist = sum(m.genome.distance(centroid) for m in members) / len(members)
                    sp.genetic_diversity = avg_dist

    # ------------------------------------------------------------------
    # Niche-aware genome bias
    # ------------------------------------------------------------------

    def niche_bias_genome(self, genome: Any) -> None:
        """Mutate genome toward its species niche."""
        sp = self.host.species.get(getattr(genome, "species", 0))
        if sp is None:
            return
        niche = sp.niche
        if niche not in NICHE_TYPES:
            return

        field_choices = NICHE_FIELD_AFFINITY.get(niche)
        warp_choices = NICHE_WARP_AFFINITY.get(niche)
        post_choices = NICHE_POST_AFFINITY.get(niche)
        palette_choices = NICHE_PALETTE_AFFINITY.get(niche)

        if field_choices and hasattr(genome, "field_type") and random.random() < 0.15:
            genome.field_type = random.choice(field_choices)
            if hasattr(genome, "_default_field_params"):
                genome.field_params = genome._default_field_params()
        if warp_choices and hasattr(genome, "warp") and random.random() < 0.12:
            genome.warp = random.choice(warp_choices)
        if post_choices and hasattr(genome, "post") and random.random() < 0.10:
            genome.post = random.sample(
                post_choices, k=min(len(post_choices), random.randint(1, 3))
            )
        if palette_choices and hasattr(genome, "palette") and random.random() < 0.08:
            genome.palette = random.choice(palette_choices)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def species_shader_injections(self, org: Any, src: str) -> str:
        """Inject species/niche-specific shader code."""
        sid = getattr(org, "species", 0)
        sp = self.host.species.get(sid)
        if sp is None:
            return src

        niche = sp.niche

        if niche == "fog_layers":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "float fog = smoothstep(0.0, 0.7, length(v_uv - 0.5));\n"
                "col = mix(col, vec3(0.6, 0.6, 0.7), fog * 0.4);\n"
                "col *= mix(0.5, 1.0, clamp(u_org_energy, 0.0, 1.0));\n"
                "fragColor = vec4(col, 1.0);",
            )
        elif niche == "trail_makers":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "col += texture(previous_frame, v_uv * 0.98 + vec2(0.002)).rgb * 0.35;\n"
                "col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.55);\n"
                "fragColor = vec4(col, 1.0);",
            )
        elif niche == "geometry_guild":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "col = pow(col, vec3(0.9));\n"
                "col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.55);\n"
                "fragColor = vec4(col, 1.0);",
            )
        elif niche == "colour_weavers":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.7);\n"
                "col += 0.08 * sin(v_uv.x * 20.0 + u_time * 2.0);\n"
                "fragColor = vec4(col, 1.0);",
            )
        elif niche == "parasites":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "col *= 0.7 + 0.3 * u_org_energy;\n"
                "col = mix(col, col * u_org_tint * 1.5, 0.6);\n"
                "fragColor = vec4(col, 1.0);",
            )
        elif niche == "bright_field":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "col *= 1.3;\n"
                "col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.55);\n"
                "fragColor = vec4(col, 1.0);",
            )
        elif niche == "chaotic_region":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "col += 0.15 * sin(v_uv.x * 30.0 + u_time * 5.0) * sin(v_uv.y * 30.0 - u_time * 3.0);\n"
                "col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.55);\n"
                "fragColor = vec4(col, 1.0);",
            )
        elif niche == "calm_region":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "col = mix(col, vec3(dot(col, vec3(0.299, 0.587, 0.114))), 0.2);\n"
                "col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.55);\n"
                "fragColor = vec4(col, 1.0);",
            )
        elif niche == "music_hotspot":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "col += 0.2 * bass * sin(v_uv.x * 15.0 + u_time);\n"
                "col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.55);\n"
                "fragColor = vec4(col, 1.0);",
            )
        elif niche == "resource_valley":
            src = src.replace(
                "fragColor = vec4(col, 1.0);",
                "col = mix(col, col * vec3(0.8, 1.0, 0.8), 0.15);\n"
                "col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.55);\n"
                "fragColor = vec4(col, 1.0);",
            )

        return src

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self._speciation_timer += dt
        self._recombine_timer += dt
        self.assign_niches()
        self.extinction_check(dt)
        self.maybe_revive(dt)
        if self._speciation_timer > 15.0:
            self._speciation_timer = 0.0
            for org in self.host.organisms:
                if org.alive and random.random() < 0.02:
                    new_sid = self.maybe_speciate(org)
                    if new_sid is not None:
                        org.species = new_sid
        if self._recombine_timer > NICHE_RECOMBINE_INTERVAL:
            self._recombine_timer = 0.0
            self.niche_map.step(audio, dt, float(audio.get("bass", 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "niche_map": self.niche_map.to_dict(),
            "species": {str(k): v.to_dict() for k, v in self.host.species.items()},
        }

    @classmethod
    def from_dict(cls, host: Any, config: Any, d: dict[str, Any]) -> MultiSpeciesEcosystem:
        obj = cls(host, config)
        if d.get("niche_map"):
            obj.niche_map = NicheMap.from_dict(d["niche_map"])
        return obj


# ============================================================================
# Parasite system
# ============================================================================


class ParasiteSystem:
    """Manages parasite-host relationships between organisms."""

    def __init__(self, host_eco: Any):
        self.host = host_eco
        self.parasite_drain = 0.012

    def step(self, dt: float) -> None:
        for org in self.host.organisms:
            if not org.alive:
                continue
            if getattr(org, "symbiosis_type", "none") != "parasite":
                continue
            h = getattr(org, "host", None)
            if h is None or not h.alive:
                continue
            drain = self.parasite_drain * dt
            h.energy = max(0.0, h.energy - drain)
            org.energy = min(getattr(self.host, "max_energy", 3.0), org.energy + drain * 0.7)

    def try_attach(self, org: Any) -> None:
        if getattr(org, "symbiont", None) is not None:
            return
        if getattr(org, "host", None) is not None:
            return
        if random.random() > 0.003:
            return
        candidates = [
            o
            for o in self.host.organisms
            if o is not org
            and o.alive
            and getattr(o, "symbiont", None) is None
            and getattr(o, "host", None) is None
            and math.hypot(o.pos[0] - org.pos[0], o.pos[1] - org.pos[1]) < 0.1
        ]
        if not candidates:
            return
        target = random.choice(candidates)
        if org.energy > target.energy * 1.4:
            org.symbiont = target
            target.host = org
            org.symbiosis_type = "parasite"
            target.symbiosis_type = "host"
        elif abs(org.energy - target.energy) < 0.4:
            org.host = target
            target.symbiont = org
            org.symbiosis_type = "mutualism"
            target.symbiosis_type = "mutualism"


# ============================================================================
# Helpers
# ============================================================================


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, t)
    if i == 4:
        return (t, p, q)
    return (v, p, q)
