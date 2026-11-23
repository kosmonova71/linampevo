"""Physics ecosystem simulation package.

Provides a minimal agent-based ecosystem where entities evolve under
physics forces driven by audio input.  The package exposes:

- :class:`Field` -- 2D energy grid with injection and diffusion.
- :class:`Forces` -- Applies physics forces (gravity, audio, energy
  gradients) to entities each step.
- :class:`Entity` / :class:`EntityDNA` -- Simulated particles with DNA.
- :class:`WorldState` -- Aggregated state shared between subsystems.

The main entry point is :class:`physics.Ecosystem`.
"""

from __future__ import annotations

from .ecosystem import Ecosystem
from .field import Field
from .forces import Forces
from .particles import Entity, EntityDNA
from .world_state import WorldState

__all__ = [
    "Ecosystem",
    "Entity",
    "EntityDNA",
    "Field",
    "Forces",
    "WorldState",
]
