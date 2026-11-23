"""Biological system classes: senses, organs, brains, species, signals."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Senses:
    """Evolved sensory apparatus (spec 53).

    Different organisms perceive different parts of the music and the world, so
    they do not all react to the same stimuli.
    """

    hearing_range: float = 0.35
    bass_sensitivity: float = 1.0
    treble_sensitivity: float = 1.0
    harmony_sensitivity: float = 1.0
    motion_sensitivity: float = 1.0
    beat_prediction: float = 0.0
    colour_sensitivity: float = 1.0

    def clone(self) -> Senses:
        return Senses(
            hearing_range=self.hearing_range,
            bass_sensitivity=self.bass_sensitivity,
            treble_sensitivity=self.treble_sensitivity,
            harmony_sensitivity=self.harmony_sensitivity,
            motion_sensitivity=self.motion_sensitivity,
            beat_prediction=self.beat_prediction,
            colour_sensitivity=self.colour_sensitivity,
        )

    def mutate(self, strength: float = 0.15) -> None:
        def rnd(attr, lo, hi):
            v = getattr(self, attr) + random.uniform(-strength, strength) * (hi - lo)
            return max(lo, min(hi, v))

        self.hearing_range = rnd("hearing_range", 0.1, 0.8)
        self.bass_sensitivity = rnd("bass_sensitivity", 0.0, 2.0)
        self.treble_sensitivity = rnd("treble_sensitivity", 0.0, 2.0)
        self.harmony_sensitivity = rnd("harmony_sensitivity", 0.0, 2.0)
        self.motion_sensitivity = rnd("motion_sensitivity", 0.0, 2.0)
        self.beat_prediction = rnd("beat_prediction", 0.0, 1.0)
        self.colour_sensitivity = rnd("colour_sensitivity", 0.2, 2.0)

    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain dict (for spatial organism persistence)."""
        return {
            "hearing_range": float(self.hearing_range),
            "bass_sensitivity": float(self.bass_sensitivity),
            "treble_sensitivity": float(self.treble_sensitivity),
            "harmony_sensitivity": float(self.harmony_sensitivity),
            "motion_sensitivity": float(self.motion_sensitivity),
            "beat_prediction": float(self.beat_prediction),
            "colour_sensitivity": float(self.colour_sensitivity),
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Senses:
        """Reconstruct from a serialised dict (tolerant of missing keys)."""
        return cls(
            hearing_range=float(d.get("hearing_range", 0.35)),
            bass_sensitivity=float(d.get("bass_sensitivity", 1.0)),
            treble_sensitivity=float(d.get("treble_sensitivity", 1.0)),
            harmony_sensitivity=float(d.get("harmony_sensitivity", 1.0)),
            motion_sensitivity=float(d.get("motion_sensitivity", 1.0)),
            beat_prediction=float(d.get("beat_prediction", 0.0)),
            colour_sensitivity=float(d.get("colour_sensitivity", 1.0)),
        )


@dataclass
class Organs:
    """Internal functional modules (spec 54).

    Movement, skeleton, skin, glow, particles and memory evolve separately and
    each draws from the shared energy pool.
    """

    movement: float = 0.5
    skeleton: float = 0.5
    skin: float = 0.5
    glow: float = 0.3
    particles: float = 0.3
    memory: float = 0.5

    def clone(self) -> Organs:
        return Organs(
            self.movement, self.skeleton, self.skin, self.glow, self.particles, self.memory
        )

    def mutate(self, strength: float = 0.15) -> None:
        for a in ("movement", "skeleton", "skin", "glow", "particles", "memory"):
            lo, hi = 0.0, 1.0
            v = getattr(self, a) + random.uniform(-strength, strength) * (hi - lo)
            setattr(self, a, max(lo, min(hi, v)))

    def energy_cost(self) -> float:
        return 0.4 * self.glow + 0.4 * self.particles + 0.15 * self.skeleton + 0.1 * self.skin

    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain dict (for spatial organism persistence)."""
        return {
            "movement": float(self.movement),
            "skeleton": float(self.skeleton),
            "skin": float(self.skin),
            "glow": float(self.glow),
            "particles": float(self.particles),
            "memory": float(self.memory),
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Organs:
        """Reconstruct from a serialised dict (tolerant of missing keys)."""
        return cls(
            movement=float(d.get("movement", 0.5)),
            skeleton=float(d.get("skeleton", 0.5)),
            skin=float(d.get("skin", 0.5)),
            glow=float(d.get("glow", 0.3)),
            particles=float(d.get("particles", 0.3)),
            memory=float(d.get("memory", 0.5)),
        )


class OrganismBrain:
    """Tiny neural controller (spec 55).

    Inputs:  bass, treble, energy, nearby pressure, age, fitness.
    Outputs: move, grow, reproduce, mutate, change_shader.
    """

    IN = 6
    HID = 5
    OUT = 5
    OUT_LABELS = ("move", "grow", "reproduce", "mutate", "change_shader")

    def __init__(self, seed: int | None = None):
        rnd = random.Random(seed)
        self.w1 = [rnd.uniform(-1.0, 1.0) for _ in range(self.IN * self.HID)]
        self.w2 = [rnd.uniform(-1.0, 1.0) for _ in range(self.HID * self.OUT)]
        self.b1 = [0.0] * self.HID
        self.b2 = [0.0] * self.OUT
        self._last_h: list[float] | None = None
        self._last_in: list[float] | None = None
        self.fit = 0.0

    def clone(self) -> OrganismBrain:
        b = OrganismBrain()
        b.w1 = list(self.w1)
        b.w2 = list(self.w2)
        b.b1 = list(self.b1)
        b.b2 = list(self.b2)
        return b

    def mutate(self, strength: float = 0.2) -> None:
        for k in range(len(self.w1)):
            self.w1[k] += random.uniform(-strength, strength)
        for k in range(len(self.w2)):
            self.w2[k] += random.uniform(-strength, strength)

    def process(self, inputs: list[float]) -> list[float]:
        h = []
        for j in range(self.HID):
            s = self.b1[j]
            base = j * self.IN
            for i in range(self.IN):
                s += self.w1[base + i] * inputs[i]
            h.append(math.tanh(s))
        out = []
        for k in range(self.OUT):
            s = self.b2[k]
            base = k * self.HID
            for j in range(self.HID):
                s += self.w2[base + j] * h[j]
            out.append(math.tanh(s))
        self._last_h = h
        self._last_in = list(inputs)
        return out

    def reinforce(self, reward: float, lr: float = 0.04) -> None:
        if self._last_h is None or self._last_in is None:
            return
        for k in range(self.OUT):
            base = k * self.HID
            for j in range(self.HID):
                self.w2[base + j] += lr * reward * self._last_h[j]
        for j in range(self.HID):
            base = j * self.IN
            for i in range(self.IN):
                self.w1[base + i] += lr * reward * self._last_in[i] * self._last_h[j]

    def fitness(self) -> float:
        return float(self.fit)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the network weights/biases (for persistence)."""
        return {
            "w1": list(self.w1),
            "w2": list(self.w2),
            "b1": list(self.b1),
            "b2": list(self.b2),
            "fit": float(self.fit),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OrganismBrain:
        """Reconstruct from a serialised dict (tolerant of missing keys)."""
        b = cls()
        b.w1 = [float(x) for x in d.get("w1", b.w1)]
        b.w2 = [float(x) for x in d.get("w2", b.w2)]
        b.b1 = [float(x) for x in d.get("b1", b.b1)]
        b.b2 = [float(x) for x in d.get("b2", b.b2)]
        b.fit = float(d.get("fit", 0.0))
        return b


@dataclass
class SpeciesDef:
    """A species: colour, diet (predator/prey, spec 49) and biome affinity."""

    sid: int
    color: tuple[float, float, float]
    diet: set[int] = field(default_factory=set)
    biome_affinity: float = 0.5
    born_at: float = 0.0
    avg_complexity: float = 0.0
    niche: str = "deep_field"
    niche_score: float = 0.0
    extinction_timer: float = 0.0


@dataclass
class Signal:
    """A broadcast message (spec 50): danger / food / mate / music / light / noise."""

    x: float
    y: float
    stype: str
    species: int
    strength: float
    ttl: float
