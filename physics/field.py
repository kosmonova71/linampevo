"""Field — 2D energy grid for the physics ecosystem."""

from __future__ import annotations


class Field:
    """A 2D grid representing spatially-distributed energy.

    The energy field is the primary resource that entities interact
    with: they draw energy from it to survive and reproduce.  Audio
    input injects bursts of energy at random positions, while the
    ``diffuse`` method spreads energy across neighbouring cells each
    step.

    The grid is stored in two ping-pong buffers (``_grid_a`` /
    ``_grid_b``) so that ``diffuse`` can swap buffers instead of
    allocating a fresh grid every step.  ``energy`` always points to
    the *active* (front) buffer.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)

        # Base energy grid — seeded with a small positive value so that
        # entities always have something to draw from.
        self._grid_a: list[list[float]] = [
            [1.0 for _ in range(self.width)] for _ in range(self.height)
        ]
        self._grid_b: list[list[float]] = [
            [0.0 for _ in range(self.width)] for _ in range(self.height)
        ]
        # ``energy`` is a reference to whichever buffer is currently
        # active (front).  After ``diffuse`` swaps buffers this is
        # updated so external holders (e.g. WorldState) see the new
        # grid.
        self.energy: list[list[float]] = self._grid_a

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def inject(self, x: int, y: int, amount: float) -> None:
        """Add *amount* of energy at grid position ``(x, y)``.

        Coordinates are clamped to valid grid bounds.  Energy is
        clamped to a minimum of 0 so the field never goes negative.
        """
        if not (0 <= y < self.height and 0 <= x < self.width):
            return
        self.energy[y][x] = max(0.0, self.energy[y][x] + float(amount))

    def withdraw(self, x: int, y: int, amount: float) -> None:
        """Remove *amount* of energy from cell ``(x, y)``.

        This models entities consuming local field energy.  The cell
        is clamped to a minimum of 0.  Coordinates are bounds-checked
        and silently ignored if out of range.
        """
        if not (0 <= y < self.height and 0 <= x < self.width):
            return
        self.energy[y][x] = max(0.0, self.energy[y][x] - float(amount))

    def sample(self, x: int, y: int) -> float:
        """Return the energy at cell ``(x, y)``, or 0.0 if out of bounds."""
        if 0 <= y < self.height and 0 <= x < self.width:
            return float(self.energy[y][x])
        return 0.0

    def diffuse(self, amount: float = 0.25) -> None:
        """Apply one Jacobi diffusion step to the energy grid.

        Each cell is blended towards the average of itself and its four
        von-Neumann neighbours.  ``amount`` controls the blend factor:
        higher values spread energy more aggressively per step.

        Uses a ping-pong buffer swap to avoid per-step allocation.
        """
        amount = max(0.0, min(1.0, float(amount)))
        if amount <= 0.0:
            return

        w = self.width
        h = self.height
        if w <= 0 or h <= 0:
            return

        src = self.energy
        # Pick the buffer that is *not* currently active.
        dst = self._grid_b if self.energy is self._grid_a else self._grid_a

        for y in range(h):
            src_row = src[y]
            dst_row = dst[y]
            for x in range(w):
                total = src_row[x]
                count = 1
                if x > 0:
                    total += src[y][x - 1]
                    count += 1
                if x < w - 1:
                    total += src[y][x + 1]
                    count += 1
                if y > 0:
                    total += src[y - 1][x]
                    count += 1
                if y < h - 1:
                    total += src[y + 1][x]
                    count += 1
                dst_row[x] = src_row[x] + amount * (total / count - src_row[x])

        # Swap: the destination buffer becomes the new front buffer.
        self.energy = dst

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def total_energy(self) -> float:
        """Return the sum of all cells in the grid."""
        return sum(sum(row) for row in self.energy)

    def mean_energy(self) -> float:
        """Return the mean energy per cell."""
        count = self.width * self.height
        return self.total_energy() / count if count else 0.0

    def clear(self) -> None:
        """Reset every cell in the active buffer to zero."""
        for row in self.energy:
            for i in range(len(row)):
                row[i] = 0.0
