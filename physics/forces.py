"""Forces — physics force model applied to entities each step."""

from __future__ import annotations

import math

from .particles import Entity
from .world_state import WorldState


class Forces:
    """Apply physics forces to entities.

    The force model combines three influences, each weighted by the
    entity's DNA:

    * **Gravity** — a downward force proportional to ``dna.gravity``
      and ``dna.mass``.  In this coordinate system **+Y is down**,
      matching screen-space convention so a positive ``gravity``
      value pulls entities toward the bottom of the grid.
    * **Audio coupling** — forces derived from the world-state audio
      levels and the entity's frequency affinities.
    * **Energy gradient** — the entity is attracted towards nearby
      high-energy cells.

    Forces update the entity's velocity; position integration is left
    to the caller (or a subsequent ``Entity.step_lifecycle``).
    """

    def __init__(self) -> None:
        # Tunable constants — kept small to stay stable at fixed dt.
        self._gravity_scale: float = 0.3
        self._audio_scale: float = 0.8
        self._energy_scale: float = 0.15
        # Maximum velocity change (impulse) per axis per step.  Clamps
        # ``a * dt`` so large or variable dt values cannot create
        # unstable spikes.
        self._max_impulse: float = 0.5

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def apply(self, entity: Entity, world_state: WorldState, dt: float) -> None:
        """Apply forces to *entity* for time-step *dt*."""
        dt = float(dt)
        dna = entity.dna

        # --- gravity ---------------------------------------------------
        # +Y is "down" (screen-space), so gravity adds to fy.
        fx = 0.0
        fy = 0.0
        fy += float(dna.gravity) * float(dna.mass) * self._gravity_scale

        # --- audio coupling -------------------------------------------
        bass = float(getattr(world_state, "audio_bass", 0.0))
        mid = float(getattr(world_state, "audio_mid", 0.0))
        treble = float(getattr(world_state, "audio_treble", 0.0))

        # Per-axis jitter from audio; frequency-specific affinities gate
        # how strongly each band pushes the entity.
        fx += (
            bass * float(dna.bass_affinity) - treble * float(dna.treble_affinity)
        ) * self._audio_scale
        fy += (mid * float(dna.mid_affinity) - bass * float(dna.bass_affinity)) * self._audio_scale

        # Occasional beat burst — strong orthogonal kick on strong bass.
        if bass > 0.7:
            beat = 0.4 * (bass - 0.7)
            if fx >= 0.0:
                fx += beat
            else:
                fx -= beat

        # --- energy gradient attraction -------------------------------
        energy = world_state.energy
        if energy and 0 <= entity.y < len(energy) and 0 <= entity.x < len(energy[0]):
            cx = int(entity.x)
            cy = int(entity.y)
            e0 = float(energy[cy][cx])

            # Compare with 4-connected neighbours and bias towards the
            # steepest uphill direction.
            neighbours = []
            w = len(energy[0])
            h = len(energy)
            if cx > 0:
                neighbours.append((float(energy[cy][cx - 1]) - e0, -1, 0))
            if cx < w - 1:
                neighbours.append((float(energy[cy][cx + 1]) - e0, 1, 0))
            if cy > 0:
                neighbours.append((float(energy[cy - 1][cx]) - e0, 0, -1))
            if cy < h - 1:
                neighbours.append((float(energy[cy + 1][cx]) - e0, 0, 1))

            if neighbours:
                # Pick the neighbour with the strongest energy delta.
                best = max(neighbours, key=lambda n: n[0])
                delta = best[0]
                if delta > 0.0:
                    fx += best[1] * delta * self._energy_scale * float(dna.attraction)
                    fy += best[2] * delta * self._energy_scale * float(dna.attraction)

        # --- integrate velocity ---------------------------------------
        # F = ma  →  a = F/m ;  cap mass to avoid division issues.
        mass = max(float(dna.mass), 0.01)
        ax = fx / mass
        ay = fy / mass

        # Clamp impulse (a * dt) so a single large dt cannot spike
        # velocity beyond design limits.
        ix = ax * dt
        iy = ay * dt
        mag = math.hypot(ix, iy)
        if mag > self._max_impulse and mag > 0.0:
            scale = self._max_impulse / mag
            ix *= scale
            iy *= scale

        # Apply velocity cap based on dna.speed so entities don't
        # accelerate without bound.
        max_speed = float(dna.speed) + 1.0
        entity.vx += ix
        entity.vy += iy

        speed = math.hypot(entity.vx, entity.vy)
        if speed > max_speed and speed > 0.0:
            entity.vx = entity.vx / speed * max_speed
            entity.vy = entity.vy / speed * max_speed
