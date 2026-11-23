"""Ecosystem — main entry point for the physics simulation.

This package is a **standalone demo world** that runs a minimal
agent-based ecosystem: entities evolve under physics forces driven by
audio input.  It is *not* coupled to ``evo_multi_species`` or the
spatial organism simulators in ``evo_visualizer``.

Renderer integration
--------------------
The grid data exposed via :attr:`Ecosystem.world_state` (energy,
density, flow, temperature) is designed to be uploaded to shader
uniforms or textures.  If you intend to drive GPU rendering from
these grids, copy them into a GPU-friendly buffer (e.g. a flat
``float`` array or a texture) rather than passing the nested Python
lists directly — they are serialisation-friendly but not
zero-copy / buffer-protocol compatible.
"""

from __future__ import annotations

import json
import os
import random

from .field import Field
from .forces import Forces
from .particles import Entity, EntityDNA
from .world_state import WorldState


class Ecosystem:
    def __init__(
        self,
        width: int,
        height: int,
        seed_entities: int = 16,
        max_entities: int = 96,
    ):
        self.width = int(width)
        self.height = int(height)

        self.field = Field(self.width, self.height)
        self.forces = Forces()

        self.entities: list[Entity] = []
        self.max_entities = int(max_entities)

        for _ in range(int(seed_entities)):
            self.entities.append(
                Entity(
                    x=random.uniform(0, self.width),
                    y=random.uniform(0, self.height),
                    vx=0.0,
                    vy=0.0,
                    dna=EntityDNA.random(),
                )
            )

        self.world_state = WorldState(
            width=self.width,
            height=self.height,
            energy=self.field.energy,
            temperature=[[0.0 for _ in range(self.width)] for _ in range(self.height)],
            density=[[0.0 for _ in range(self.width)] for _ in range(self.height)],
            flow=[[0.0 for _ in range(self.width)] for _ in range(self.height)],
            age=0.0,
        )

        self.age = 0.0

    def inject_from_audio(self, bass: float, mid: float, treble: float) -> None:
        """Map audio levels to field pressure and inject energy bursts.

        Bursts land at random hotspots *and* near entity positions,
        so territory and survival are coupled to the audio signal.
        """
        bass = float(bass)
        mid = float(mid)
        treble = float(treble)

        pressure = 0.35 * bass + 0.20 * mid + 0.15 * treble
        vib = 0.10 * treble
        beat = 0.25 * bass * max(0.0, treble)

        # Random spatial bursts.
        n_bursts = 2 + int(max(0.0, pressure) * 2.0)
        for _ in range(n_bursts):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            self.field.inject(x, y, amount=pressure + vib)

        # Territory-driven bursts: inject near entities so audio level
        # directly influences where entities can forage.
        if self.entities and bass > 0.1:
            n_entity_bursts = 1 + int(bass * 3.0)
            for _ in range(n_entity_bursts):
                e = random.choice(self.entities)
                cx = int(e.x) % self.width if self.width else 0
                cy = int(e.y) % self.height if self.height else 0
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
                nx = (cx + dx) % self.width if self.width else 0
                ny = (cy + dy) % self.height if self.height else 0
                self.field.inject(nx, ny, amount=pressure * 0.5 + vib * 0.5)

        # Occasional beat burst.
        if random.random() < 0.02 + bass * 0.01:
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            self.field.inject(x, y, amount=beat + pressure * 0.5)

    def step(self, dt: float, audio_bass: float, audio_mid: float, audio_treble: float) -> None:
        dt = float(dt)
        self.age += dt
        self.world_state.age = self.age

        # store audio for force stage
        self.world_state.audio_bass = float(audio_bass)
        self.world_state.audio_mid = float(audio_mid)
        self.world_state.audio_treble = float(audio_treble)

        # audio -> field energy
        self.inject_from_audio(audio_bass, audio_mid, audio_treble)

        # field diffusion (swaps internal buffers)
        self.field.diffuse(amount=0.25 + 0.15 * min(1.0, float(audio_bass)))
        # Rebind shared reference so WorldState sees the post-diffuse grid.
        self.world_state.energy = self.field.energy

        # entity dynamics — forces + lifecycle (position, local energy, drag)
        new_entities: list[Entity] = []
        for e in list(self.entities):
            self.forces.apply(e, self.world_state, dt)
            e.step_lifecycle(dt, field=self.field)

            # reproduction (split)
            child_energy = e.maybe_split(dt)
            if (
                child_energy is not None
                and len(self.entities) + len(new_entities) < self.max_entities
            ):
                child_dna = e.dna.clone()
                child_dna.mutate(strength=0.15)

                child = Entity(
                    x=e.x,
                    y=e.y,
                    vx=-e.vx * 0.3,
                    vy=-e.vy * 0.3,
                    dna=child_dna,
                    life=60.0,
                    age=0.0,
                    energy=child_energy,
                )
                new_entities.append(child)

            # dead entities are skipped for splitting (removed below)
            if e.is_dead():
                continue

        # remove dead entities
        self.entities = [e for e in self.entities if not e.is_dead()]

        # competition: cull by fitness (lowest energy+life first) when over cap
        if len(self.entities) > self.max_entities:
            self.entities.sort(key=lambda e: e.fitness())
            self.entities = self.entities[-self.max_entities :]

        self.entities.extend(new_entities)

        # stamp population density into the grid for flow / shader use
        self.world_state.stamp_density(self.entities)

        # derived fields for shader interface
        self.world_state.update_derived(entities_count=len(self.entities))

    def world_summaries(self) -> dict[str, float]:
        """Return derived world statistics from the last :meth:`step`."""
        return {
            "world_energy_mean": float(self.world_state.energy_mean),
            "world_energy_max": float(self.world_state.energy_max),
            "world_entity_count": float(len(self.entities)),
            "world_age": float(self.world_state.age),
        }

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save_world(self, root: str, world_name: str = "earth_001") -> None:
        base = os.path.join(root, world_name)
        os.makedirs(base, exist_ok=True)

        physics_path = os.path.join(base, "physics.json")
        species_path = os.path.join(base, "species.json")

        # physics.json: store energy stats, age, and full grid snapshot.
        payload_physics = {
            "width": self.width,
            "height": self.height,
            "age": self.world_state.age,
            "energy_mean": float(self.world_state.energy_mean),
            "energy_max": float(self.world_state.energy_max),
            "energy": [list(row) for row in self.field.energy],
        }
        with open(physics_path, "w", encoding="utf-8") as f:
            json.dump(payload_physics, f)

        # species.json: store entity DNA summary via to_dict for stable format.
        payload_species = {
            "entities": [
                {
                    "x": e.x,
                    "y": e.y,
                    "life": e.life,
                    "dna": e.dna.to_dict(),
                }
                for e in self.entities
            ]
        }
        with open(species_path, "w", encoding="utf-8") as f:
            json.dump(payload_species, f)

    @classmethod
    def load_world(cls, root: str, world_name: str = "earth_001") -> Ecosystem | None:
        """Reconstruct an :class:`Ecosystem` from a saved world.

        Returns ``None`` if the save files are missing or unreadable.
        """
        base = os.path.join(root, world_name)
        physics_path = os.path.join(base, "physics.json")
        species_path = os.path.join(base, "species.json")

        if not os.path.isfile(physics_path) or not os.path.isfile(species_path):
            return None

        try:
            with open(physics_path, encoding="utf-8") as f:
                physics = json.load(f)
            with open(species_path, encoding="utf-8") as f:
                species = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        width = int(physics.get("width", 16))
        height = int(physics.get("height", 16))

        # Start with zero seed entities; we'll load from file instead.
        world = cls(width=width, height=height, seed_entities=0, max_entities=96)
        world.age = float(physics.get("age", 0.0))
        world.world_state.age = world.age

        # Restore energy grid.
        energy_grid = physics.get("energy")
        if energy_grid:
            for y in range(min(height, len(energy_grid))):
                for x in range(min(width, len(energy_grid[y]))):
                    world.field.energy[y][x] = float(energy_grid[y][x])
            world.world_state.energy = world.field.energy

        # Restore entities.
        for e in species.get("entities", []):
            dna = EntityDNA.from_dict(e.get("dna", {}))
            world.entities.append(
                Entity(
                    x=float(e.get("x", 0.0)),
                    y=float(e.get("y", 0.0)),
                    vx=0.0,
                    vy=0.0,
                    dna=dna,
                    life=float(e.get("life", Entity._START_LIFE)),
                )
            )

        # Enforce cap from fittest.
        if len(world.entities) > world.max_entities:
            world.entities.sort(key=lambda e: e.fitness())
            world.entities = world.entities[-world.max_entities :]

        world.world_state.update_derived(entities_count=len(world.entities))
        return world
