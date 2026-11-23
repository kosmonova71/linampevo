"""WorldState — aggregated simulation state shared across subsystems.

``WorldState`` holds references to the field energy grid (shared with
:class:`~physics.Field`), the temperature / density / flow scalar
fields, and the current audio levels.  It also stores derived values
(like population density) that are updated via
:meth:`WorldState.update_derived`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .particles import Entity


class WorldState:
    """Container for cross-cutting simulation state.

    Parameters
    ----------
    width, height
        Grid dimensions.
    energy
        Reference to the field energy grid (shared with ``Field``).
        After each ``Field.diffuse`` call the caller should rebind
        this to ``Field.energy`` so it stays in sync (diffusion swaps
        buffers).
    temperature, density, flow
        Additional scalar grids used by the shader interface.
    age
        World simulation age in seconds.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        energy: list[list[float]],
        temperature: list[list[float]],
        density: list[list[float]],
        flow: list[list[float]],
        age: float = 0.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.energy = energy
        self.temperature = temperature
        self.density = density
        self.flow = flow

        self.age = float(age)

        # Audio levels — updated each step by Ecosystem.
        self.audio_bass: float = 0.0
        self.audio_mid: float = 0.0
        self.audio_treble: float = 0.0

        # Derived values (populated by update_derived).
        self.population_density: float = 0.0
        self.energy_mean: float = 0.0
        self.energy_max: float = 0.0
        self.total_entities: int = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def stamp_density(self, entities: list[Entity]) -> None:
        """Reset the density grid and stamp entity positions into it.

        Each entity contributes +1 to the cell it occupies (toroidally
        wrapped).  This must be called before :meth:`update_derived`
        so that the flow field and population density reflect the
        current population distribution.
        """
        for row in self.density:
            for i in range(len(row)):
                row[i] = 0.0
        if not self.density or not self.density[0]:
            return
        dh = len(self.density)
        dw = len(self.density[0])
        for e in entities:
            x = int(e.x) % self.width if self.width else 0
            y = int(e.y) % self.height if self.height else 0
            if 0 <= y < dh and 0 <= x < dw:
                self.density[y][x] += 1.0

    def update_derived(self, *, entities_count: int = 0) -> None:
        """Recompute cheap derived fields from the energy grid.

        Called once per step after :meth:`stamp_density` and all
        entities have been processed.  Uses the shared ``energy``
        grid reference for fresh values.
        """
        self.total_entities = int(entities_count)

        # Population density: normalised to [0, 1] relative to a
        # notional carrying capacity.
        capacity = max(self.width * self.height * 4, 1)
        self.population_density = min(1.0, float(entities_count) / float(capacity))

        # Energy statistics — single pass over the grid.
        total = 0.0
        max_val = 0.0
        count = 0
        if self.energy:
            for row in self.energy:
                for v in row:
                    val = float(v)
                    total += val
                    max_val = max(max_val, val)
                    count += 1
        self.energy_mean = float(total / count) if count else 0.0
        self.energy_max = float(max_val)

        # Propagate temperature — slowly equilibrate towards local energy.
        if self.temperature and self.energy_mean > 0.0:
            factor = 0.02
            for y, row in enumerate(self.temperature):
                for x in range(len(row)):
                    # Temperature is driven by local energy but smoothed.
                    local_energy = (
                        float(self.energy[y][x])
                        if y < len(self.energy) and x < len(self.energy[y])
                        else 0.0
                    )
                    self.temperature[y][x] += (local_energy - self.temperature[y][x]) * factor

        # Update flow from density gradients (very lightweight).
        if self.density and self.flow:
            for y in range(self.height):
                if y >= len(self.density) or y >= len(self.flow):
                    continue
                drow = self.density[y]
                frow = self.flow[y]
                for x in range(self.width):
                    if x >= len(drow) or x >= len(frow):
                        continue
                    e = (
                        float(self.energy[y][x])
                        if y < len(self.energy) and x < len(self.energy[y])
                        else 0.0
                    )
                    frow[x] = 0.3 * frow[x] + 0.7 * (e - float(drow[x])) * 0.01

    def __repr__(self) -> str:
        return (
            f"WorldState(width={self.width}, height={self.height}, "
            f"age={self.age:.2f}, entities={self.total_entities})"
        )
