#!/usr/bin/env python3

"""evo_bio.visual_behaviours

Spec 16 — Evolving visual behaviours.

Instead of evolving appearance (colours, warp, noise), evolve *behaviour*.
Each behaviour is a dynamical system that produces an emergent spatial
field.  The shader renders that field as a viewport into an ecosystem.

Families:
    0  school       — boids-like schooling
    1  explosion     — radial burst with shockwave
    2  coral_growth  — recursive branching skeleton
    3  crystal       — ordered lattice growth
    4  storm         — turbulent vortical flow
    5  galaxy        — gravitational collapse + rotation
    6  lava          — viscous flow with thermal cooling
    7  swarm         — chaotic collective cloud
    8  vine          — directional climbing growth
    9  neuron        — branching with signal propagation
    10 fungal        — mycelial network expansion

The engine composites all active behaviours into a single field texture
that the renderer uploads to the GPU each frame.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

FAMILY_NAMES = (
    "school",
    "explosion",
    "coral_growth",
    "crystal",
    "storm",
    "galaxy",
    "lava",
    "swarm",
    "vine",
    "neuron",
    "fungal",
)
FAMILY_COUNT = len(FAMILY_NAMES)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _field_mean(field: list[list[float]]) -> float:
    if not field or not field[0]:
        return 0.0
    total = sum(sum(row) for row in field)
    count = len(field) * len(field[0])
    return total / max(1, count)


# ---------------------------------------------------------------------------
# Base behaviour
# ---------------------------------------------------------------------------


class VisualBehaviour(ABC):
    """Abstract base for a single visual behaviour."""

    def __init__(self, genome: Any, width: int = 64, height: int = 64, *, seed: int | None = None):
        self.genome = genome
        self.width = max(8, int(width))
        self.height = max(8, int(height))
        self.rng = random.Random(seed)
        self.age: float = 0.0
        self.field: list[list[float]] = [[0.0] * self.width for _ in range(self.height)]
        self.agent_positions: list[tuple[float, float]] = []

    # -- interface ----------------------------------------------------------

    @abstractmethod
    def step(self, dt: float, audio: dict[str, float]) -> None:
        """Advance one tick.  Writes into ``self.field``."""

    def reset(self) -> None:
        """Clear state (useful when genome family changes)."""
        self.field = [[0.0] * self.width for _ in range(self.height)]
        self.agent_positions = []
        self.age = 0.0

    def field_copy(self) -> list[list[float]]:
        return [row[:] for row in self.field]

    def _g(self, name: str) -> float:
        """Convenience accessor for genome params dict."""
        params = getattr(self.genome, "params", dict)()
        return float(params.get(name, 0.5))

    def _apply_audio(self, audio: dict[str, float]) -> float:
        bass = float(audio.get("bass", 0.0))
        mid = float(audio.get("mid", 0.0))
        treb = float(audio.get("treb", 0.0))
        beat = float(audio.get("beat", 0.0))
        reactivity = self._g("audio_reactivity")
        return (0.4 * bass + 0.3 * mid + 0.2 * treb + 0.5 * beat) * reactivity

    def _write_trail(self, x: float, y: float, strength: float, radius: float = 0.02) -> None:
        ix = int(x * self.width)
        iy = int(y * self.height)
        r = max(1, int(radius * max(self.width, self.height)))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px = ix + dx
                py = iy + dy
                if 0 <= px < self.width and 0 <= py < self.height:
                    d = math.hypot(dx, dy)
                    if d <= r:
                        falloff = 1.0 - d / max(r, 1)
                        self.field[py][px] = min(1.0, self.field[py][px] + strength * falloff)

    def _decay_field(self, rate: float = 0.02) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self.field[y][x] = max(0.0, self.field[y][x] - rate)


# ---------------------------------------------------------------------------
# Behaviour factory
# ---------------------------------------------------------------------------


def make_behaviour(
    genome: Any, width: int = 64, height: int = 64, *, seed: int | None = None
) -> VisualBehaviour:
    family = int(getattr(genome, "behaviour_family", 0)) % FAMILY_COUNT
    cls = _BEHAVIOUR_CLASSES[family]
    return cls(genome, width, height, seed=seed)


# ---------------------------------------------------------------------------
# Concrete behaviours
# ---------------------------------------------------------------------------


class SchoolBehaviour(VisualBehaviour):
    """Boids-like schooling: coherent group motion with alignment,
    cohesion and separation.  Agents leave soft trails."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = 12 + int(self._g("intensity") * 20)
        self.agents: list[dict[str, float]] = []
        for _ in range(n):
            self.agents.append(
                {
                    "x": self.rng.random(),
                    "y": self.rng.random(),
                    "vx": self.rng.uniform(-0.01, 0.01),
                    "vy": self.rng.uniform(-0.01, 0.01),
                }
            )
        self.agent_positions = [(a["x"], a["y"]) for a in self.agents]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        speed = 0.3 + 2.0 * self._g("speed")
        ali = 0.5 + 1.5 * self._g("cohesion")
        coh = 0.3 + 1.2 * self._g("cohesion")
        chaos = self._g("chaos")

        new_v = []
        for a in self.agents:
            ax = ay = 0.0
            count = 0
            for b in self.agents:
                if b is a:
                    continue
                dx = a["x"] - b["x"]
                dy = a["y"] - b["y"]
                d = math.hypot(dx, dy)
                if d > 0.15 or d <= 0:
                    continue
                count += 1
                ax += dx / max(d, 1e-6)
                ay += dy / max(d, 1e-6)
                ax += b["vx"] * ali
                ay += b["vy"] * ali
                ax += b["x"] * coh * 0.1
                ay += b["y"] * coh * 0.1
            if count:
                pass
            else:
                ax = self.rng.uniform(-chaos, chaos)
                ay = self.rng.uniform(-chaos, chaos)
            new_v.append((ax, ay))

        for a, (ax, ay) in zip(self.agents, new_v):
            a["vx"] = _clamp(a["vx"] + ax * dt * 0.1, -speed, speed)
            a["vy"] = _clamp(a["vy"] + ay * dt * 0.1, -speed, speed)
            a["x"] = (a["x"] + a["vx"] * dt + audio_kick * 0.01) % 1.0
            a["y"] = (a["y"] + a["vy"] * dt + audio_kick * 0.01) % 1.0

        self._decay_field(0.01 + 0.02 * (1.0 - self._g("persistence")))
        for a in self.agents:
            self._write_trail(a["x"], a["y"], 0.3 + 0.4 * self._g("intensity"), 0.015)
        self.agent_positions = [(a["x"], a["y"]) for a in self.agents]


class ExplosionBehaviour(VisualBehaviour):
    """Radial burst from seed points with shockwave propagation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bursts: list[dict[str, float]] = []
        self._burst_timer = 0.0
        self.agent_positions = []

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        self._burst_timer += dt
        burst_interval = 0.5 + 3.0 * (1.0 - self._g("speed"))
        if self._burst_timer >= burst_interval or not self.bursts:
            self._burst_timer = 0.0
            self.bursts.append(
                {
                    "x": 0.2 + 0.6 * self.rng.random(),
                    "y": 0.2 + 0.6 * self.rng.random(),
                    "radius": 0.0,
                    "strength": 0.5 + 0.5 * self._g("intensity"),
                    "speed": 0.3 + 0.7 * self._g("speed"),
                    "particles": [
                        {
                            "x": 0.5,
                            "y": 0.5,
                            "vx": self.rng.uniform(-1, 1),
                            "vy": self.rng.uniform(-1, 1),
                            "life": 1.0,
                        }
                        for _ in range(8 + int(self._g("intensity") * 20))
                    ],
                }
            )

        decay = 0.01 + 0.03 * (1.0 - self._g("persistence"))
        for b in self.bursts:
            b["radius"] += b["speed"] * dt * (1.0 + audio_kick)
            b["strength"] *= 1.0 - decay
            for p in b["particles"]:
                p["x"] += p["vx"] * dt * 0.5
                p["y"] += p["vy"] * dt * 0.5
                p["life"] *= 1.0 - decay * 2.0
                if 0 <= p["x"] <= 1 and 0 <= p["y"] <= 1:
                    self._write_trail(p["x"], p["y"], p["life"] * b["strength"] * 0.2, 0.01)
            self._write_trail(b["x"], b["y"], b["strength"] * 0.4, b["radius"] * 0.3)

        self.bursts = [b for b in self.bursts if b["strength"] > 0.01]
        self.agent_positions = [
            (p["x"], p["y"]) for b in self.bursts for p in b["particles"] if p["life"] > 0.01
        ][:50]


class CoralGrowthBehaviour(VisualBehaviour):
    """Recursive branching growth from seed points on the boundary."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branches: list[dict[str, Any]] = []
        self._seed_points: list[tuple[float, float, float]] = []
        for _ in range(3 + int(self._g("branching") * 5)):
            side = self.rng.choice(["top", "bottom", "left", "right"])
            if side == "top":
                x, y = self.rng.random(), 1.0
                angle = self.rng.uniform(math.pi * 0.25, math.pi * 0.75)
            elif side == "bottom":
                x, y = self.rng.random(), 0.0
                angle = self.rng.uniform(-math.pi * 0.75, -math.pi * 0.25)
            elif side == "left":
                x, y = 0.0, self.rng.random()
                angle = self.rng.uniform(-math.pi * 0.25, math.pi * 0.25)
            else:
                x, y = 1.0, self.rng.random()
                angle = self.rng.uniform(math.pi * 0.75, math.pi * 1.25)
            self._seed_points.append((x, y, angle))
            self.branches.append(
                {
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "length": 0.0,
                    "thickness": 0.5 + 0.5 * self._g("intensity"),
                    "children": [],
                    "alive": True,
                }
            )
        self.agent_positions = [(s[0], s[1]) for s in self._seed_points]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        speed = 0.1 + 0.4 * self._g("speed")
        max_branch = 0.3 + 0.7 * self._g("branching")
        decay = 0.005 + 0.02 * (1.0 - self._g("persistence"))

        new_branches = []
        for b in self.branches:
            if not b["alive"]:
                continue
            b["length"] += speed * dt * (1.0 + audio_kick * 0.5)
            nx = b["x"] + math.cos(b["angle"]) * speed * dt
            ny = b["y"] + math.sin(b["angle"]) * speed * dt
            if 0 <= nx <= 1 and 0 <= ny <= 1:
                self._write_trail(nx, ny, b["thickness"] * 0.3, 0.008)
                b["x"], b["y"] = nx, ny
            else:
                b["alive"] = False
                continue

            if (
                b["length"] > 0.2
                and len(b["children"]) < 3
                and self.rng.random() < max_branch * dt
            ):
                angle_var = 0.3 + 0.5 * self._g("chaos")
                child_angle = b["angle"] + self.rng.uniform(-angle_var, angle_var)
                child = {
                    "x": b["x"],
                    "y": b["y"],
                    "angle": child_angle,
                    "length": 0.0,
                    "thickness": b["thickness"] * 0.7,
                    "children": [],
                    "alive": True,
                }
                new_branches.append(child)
                b["children"].append(child)

        self.branches.extend(new_branches)
        self._decay_field(decay)
        self.agent_positions = [(b["x"], b["y"]) for b in self.branches if b["alive"]][:50]


class CrystalBehaviour(VisualBehaviour):
    """Ordered lattice growth from seed points with symmetric expansion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crystals: list[dict[str, Any]] = []
        symmetry = 4 + int(self._g("symmetry") * 4)
        for _ in range(2 + int(self._g("intensity") * 3)):
            cx, cy = self.rng.random(), self.rng.random()
            facets = []
            for i in range(symmetry):
                angle = (i / symmetry) * 2.0 * math.pi
                facets.append(
                    {
                        "angle": angle,
                        "length": 0.0,
                        "speed": 0.2 + 0.3 * self._g("speed"),
                    }
                )
            self.crystals.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "facets": facets,
                    "size": 0.0,
                    "alive": True,
                }
            )
        self.agent_positions = [(c["cx"], c["cy"]) for c in self.crystals]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        decay = 0.01 + 0.02 * (1.0 - self._g("persistence"))

        for c in self.crystals:
            if not c["alive"]:
                continue
            c["size"] += (0.2 + 0.5 * self._g("speed")) * dt * (1.0 + audio_kick)
            for f in c["facets"]:
                f["length"] += f["speed"] * dt * (1.0 + audio_kick * 0.3)
                ex = c["cx"] + math.cos(f["angle"]) * f["length"]
                ey = c["cy"] + math.sin(f["angle"]) * f["length"]
                self._write_trail(ex, ey, 0.5, 0.01)
            # fill centre
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    px = int(c["cx"] * self.width) + dx
                    py = int(c["cy"] * self.height) + dy
                    if 0 <= px < self.width and 0 <= py < self.height:
                        self.field[py][px] = min(1.0, self.field[py][px] + 0.1)

        self._decay_field(decay)
        self.agent_positions = [(c["cx"], c["cy"]) for c in self.crystals][:20]


class StormBehaviour(VisualBehaviour):
    """Turbulent vortical flow driven by multiple vortex centres."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = 3 + int(self._g("intensity") * 8)
        self.vortices: list[dict[str, float]] = []
        for _ in range(n):
            self.vortices.append(
                {
                    "x": self.rng.random(),
                    "y": self.rng.random(),
                    "strength": self.rng.uniform(-1, 1),
                    "radius": 0.1 + 0.2 * self.rng.random(),
                    "phase": self.rng.uniform(0, math.tau),
                }
            )
        self.agent_positions = [(v["x"], v["y"]) for v in self.vortices]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        viscosity = 0.2 + 0.8 * self._g("viscosity")
        chaos = self._g("chaos")

        for v in self.vortices:
            v["phase"] += dt * (0.5 + 2.0 * self._g("speed")) * (1.0 + audio_kick)
            v["x"] += math.cos(v["phase"]) * dt * chaos * 0.1
            v["y"] += math.sin(v["phase"] * 1.3) * dt * chaos * 0.1
            v["x"] = _clamp(v["x"], 0.0, 1.0)
            v["y"] = _clamp(v["y"], 0.0, 1.0)

        # Write vortex centres and connecting streams
        self._decay_field(0.005 + 0.01 * viscosity)
        for v in self.vortices:
            ix = int(v["x"] * self.width)
            iy = int(v["y"] * self.height)
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    px, py = ix + dx, iy + dy
                    if 0 <= px < self.width and 0 <= py < self.height:
                        d = math.hypot(dx, dy)
                        if d < 4:
                            self.field[py][px] = min(
                                1.0,
                                self.field[py][px] + 0.05 * abs(v["strength"]) / max(d, 1),
                            )
            # stream lines
            for i in range(6):
                ang = v["phase"] + (i / 6) * math.tau
                sx = v["x"] + math.cos(ang) * v["radius"] * 0.5
                sy = v["y"] + math.sin(ang) * v["radius"] * 0.5
                self._write_trail(sx, sy, 0.15 * abs(v["strength"]), 0.01)

        self.agent_positions = [(v["x"], v["y"]) for v in self.vortices]


class GalaxyBehaviour(VisualBehaviour):
    """Gravitational collapse with orbital rotation — spiral arms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arms = 2 + int(self._g("symmetry") * 4)
        self.centre = (0.5, 0.5)
        self.particles: list[dict[str, float]] = []
        n = 200 + int(self._g("intensity") * 400)
        for i in range(n):
            arm = i % self.arms
            base_angle = (arm / self.arms) * 2.0 * math.pi
            dist = self.rng.uniform(0.05, 0.45)
            angle = base_angle + dist * (3.0 + 2.0 * self._g("gravity"))
            x = 0.5 + math.cos(angle) * dist
            y = 0.5 + math.sin(angle) * dist
            self.particles.append(
                {
                    "x": x,
                    "y": y,
                    "vx": 0.0,
                    "vy": 0.0,
                    "mass": 0.5 + 0.5 * self.rng.random(),
                }
            )
        self.agent_positions = [(p["x"], p["y"]) for p in self.particles[:20]]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        speed = 0.1 + 0.5 * self._g("speed")
        gravity = 0.5 + 1.5 * self._g("gravity")

        self._decay_field(0.005 + 0.01 * (1.0 - self._g("persistence")))
        cx, cy = self.centre
        for p in self.particles:
            dx = cx - p["x"]
            dy = cy - p["y"]
            dist = max(math.hypot(dx, dy), 1e-6)
            # gravitational pull + orbital tangent
            pull = gravity * p["mass"] / (dist * dist + 0.01)
            tx = -dy / dist
            ty = dx / dist
            p["vx"] += (pull * (dx / dist) + tx * speed * 0.3) * dt
            p["vy"] += (pull * (dy / dist) + ty * speed * 0.3) * dt
            # audio kick
            p["vx"] += audio_kick * 0.02 * (self.rng.random() - 0.5)
            p["vy"] += audio_kick * 0.02 * (self.rng.random() - 0.5)
            p["x"] = _clamp(p["x"] + p["vx"] * dt, 0.0, 1.0)
            p["y"] = _clamp(p["y"] + p["vy"] * dt, 0.0, 1.0)
            self._write_trail(p["x"], p["y"], 0.15 * p["mass"], 0.006)

        self.agent_positions = [(p["x"], p["y"]) for p in self.particles[:20]]


class LavaBehaviour(VisualBehaviour):
    """Slow viscous flow with thermal cooling and bright emission."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blobs: list[dict[str, float]] = []
        for _ in range(5 + int(self._g("intensity") * 10)):
            self.blobs.append(
                {
                    "x": self.rng.random(),
                    "y": 0.3 + 0.7 * self.rng.random(),
                    "vx": self.rng.uniform(-0.005, 0.005),
                    "vy": self.rng.uniform(-0.01, 0.0),
                    "temp": 0.7 + 0.3 * self.rng.random(),
                    "size": 0.02 + 0.04 * self.rng.random(),
                }
            )
        self.agent_positions = [(b["x"], b["y"]) for b in self.blobs]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        viscosity = 0.3 + 0.7 * self._g("viscosity")
        speed = 0.1 + 0.3 * self._g("speed")

        self._decay_field(0.005 + 0.01 * (1.0 - self._g("persistence")))
        for b in self.blobs:
            b["vx"] += self.rng.uniform(-0.002, 0.002) * (1.0 - viscosity)
            b["vy"] -= speed * dt * 0.1 * viscosity
            b["vx"] = _clamp(b["vx"], -0.02, 0.02)
            b["vy"] = _clamp(b["vy"], -0.03, 0.01)
            b["x"] = _clamp(b["x"] + b["vx"] * dt * (1.0 + audio_kick * 0.5), 0.0, 1.0)
            b["y"] = _clamp(b["y"] + b["vy"] * dt, 0.0, 1.0)
            b["temp"] *= 1.0 - 0.01 * dt
            if b["y"] < 0.02:
                b["y"] = 0.95
                b["x"] = self.rng.random()
                b["temp"] = 0.8 + 0.2 * self.rng.random()
            r = max(1, int(b["size"] * max(self.width, self.height)))
            ix = int(b["x"] * self.width)
            iy = int(b["y"] * self.height)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    px, py = ix + dx, iy + dy
                    if 0 <= px < self.width and 0 <= py < self.height:
                        d = math.hypot(dx, dy)
                        if d <= r:
                            falloff = 1.0 - d / max(r, 1)
                            self.field[py][px] = min(
                                1.0,
                                self.field[py][px] + b["temp"] * falloff * 0.1,
                            )

        self.agent_positions = [(b["x"], b["y"]) for b in self.blobs]


class SwarmBehaviour(VisualBehaviour):
    """Chaotic collective cloud with weak alignment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = 30 + int(self._g("intensity") * 80)
        self.agents: list[dict[str, float]] = []
        for _ in range(n):
            self.agents.append(
                {
                    "x": self.rng.random(),
                    "y": self.rng.random(),
                    "vx": self.rng.uniform(-0.02, 0.02),
                    "vy": self.rng.uniform(-0.02, 0.02),
                    "phase": self.rng.uniform(0, math.tau),
                }
            )
        self.agent_positions = [(a["x"], a["y"]) for a in self.agents]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        speed = 0.5 + 2.0 * self._g("speed")
        chaos = self._g("chaos")

        for a in self.agents:
            a["phase"] += dt * (1.0 + chaos * 3.0)
            a["vx"] += math.cos(a["phase"]) * chaos * dt * 0.5
            a["vy"] += math.sin(a["phase"] * 1.7) * chaos * dt * 0.5
            a["vx"] = _clamp(a["vx"], -speed, speed)
            a["vy"] = _clamp(a["vy"], -speed, speed)
            a["x"] = (a["x"] + a["vx"] * dt * (1.0 + audio_kick * 0.3)) % 1.0
            a["y"] = (a["y"] + a["vy"] * dt * (1.0 + audio_kick * 0.3)) % 1.0

        self._decay_field(0.005 + 0.015 * (1.0 - self._g("persistence")))
        for a in self.agents:
            self._write_trail(a["x"], a["y"], 0.1 + 0.2 * self._g("intensity"), 0.008)
        self.agent_positions = [(a["x"], a["y"]) for a in self.agents]


class VineGrowthBehaviour(VisualBehaviour):
    """Directional climbing growth with coiling tendrils."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tendrils: list[dict[str, Any]] = []
        for _ in range(3 + int(self._g("branching") * 6)):
            side = self.rng.choice(["left", "right", "bottom"])
            if side == "left":
                x, y, angle = 0.0, self.rng.random(), self.rng.uniform(-0.3, 0.3)
            elif side == "right":
                x, y, angle = (
                    1.0,
                    self.rng.random(),
                    self.rng.uniform(math.pi - 0.3, math.pi + 0.3),
                )
            else:
                x, y, angle = (
                    self.rng.random(),
                    0.0,
                    self.rng.uniform(math.pi * 0.3, math.pi * 0.7),
                )
            self.tendrils.append(
                {
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "length": 0.0,
                    "thickness": 0.3 + 0.3 * self._g("intensity"),
                    "alive": True,
                }
            )
        self.agent_positions = [(t["x"], t["y"]) for t in self.tendrils]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        speed = 0.2 + 0.5 * self._g("speed")
        self._decay_field(0.003 + 0.01 * (1.0 - self._g("persistence")))

        for t in self.tendrils:
            if not t["alive"]:
                continue
            t["angle"] += math.sin(self.age * 2.0 + t["x"] * 10.0) * dt * 0.3
            t["x"] += math.cos(t["angle"]) * speed * dt * (1.0 + audio_kick * 0.3)
            t["y"] += math.sin(t["angle"]) * speed * dt * (1.0 + audio_kick * 0.3)
            t["length"] += speed * dt
            if 0 <= t["x"] <= 1 and 0 <= t["y"] <= 1:
                self._write_trail(t["x"], t["y"], t["thickness"] * 0.2, 0.006)
            else:
                t["alive"] = False

        self.agent_positions = [(t["x"], t["y"]) for t in self.tendrils if t["alive"]][:30]


class NeuronBehaviour(VisualBehaviour):
    """Branching cell body with signal pulses travelling along branches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_bodies: list[dict[str, Any]] = []
        for _ in range(1 + int(self._g("intensity") * 4)):
            cx, cy = 0.2 + 0.6 * self.rng.random(), 0.2 + 0.6 * self.rng.random()
            branches = []
            n_branch = 2 + int(self._g("branching") * 5)
            for i in range(n_branch):
                angle = (i / n_branch) * 2.0 * math.pi + self.rng.uniform(-0.2, 0.2)
                branches.append(
                    {
                        "angle": angle,
                        "points": [(cx, cy)],
                        "signal_t": 0.0,
                        "signal_speed": 0.5 + self._g("speed"),
                    }
                )
            self.cell_bodies.append(
                {
                    "x": cx,
                    "y": cy,
                    "branches": branches,
                    "pulse_phase": self.rng.uniform(0, math.tau),
                }
            )
        self.agent_positions = [(c["x"], c["y"]) for c in self.cell_bodies]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        speed = 0.3 + 0.7 * self._g("speed")
        self._decay_field(0.003 + 0.008 * (1.0 - self._g("persistence")))

        for c in self.cell_bodies:
            c["pulse_phase"] += dt * (1.0 + audio_kick)
            pulse = 0.5 + 0.5 * math.sin(c["pulse_phase"])
            # Draw cell body
            self._write_trail(c["x"], c["y"], 0.3 + 0.3 * pulse, 0.02)
            for br in c["branches"]:
                br["signal_t"] += dt * br["signal_speed"] * (1.0 + audio_kick * 0.5)
                if br["signal_t"] > 1.0:
                    br["signal_t"] -= 1.0
                # extend branch
                last = br["points"][-1]
                nx = last[0] + math.cos(br["angle"]) * speed * dt * 0.3
                ny = last[1] + math.sin(br["angle"]) * dt * 0.3
                if 0 <= nx <= 1 and 0 <= ny <= 1:
                    br["points"].append((nx, ny))
                    self._write_trail(nx, ny, 0.1 + 0.2 * pulse, 0.005)
                # draw signal pulse
                if br["points"]:
                    pi = min(len(br["points"]) - 1, int(br["signal_t"] * len(br["points"])))
                    px, py = br["points"][pi]
                    self._write_trail(px, py, 0.5 * pulse, 0.012)

        self.agent_positions = [(c["x"], c["y"]) for c in self.cell_bodies]


class FungalGrowthBehaviour(VisualBehaviour):
    """Mycelial network expansion from spore points with merging branches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network: list[dict[str, Any]] = []
        for _ in range(2 + int(self._g("intensity") * 5)):
            sx = self.rng.random()
            sy = self.rng.random()
            angle = self.rng.uniform(0, math.tau)
            self.network.append(
                {
                    "x": sx,
                    "y": sy,
                    "angle": angle,
                    "length": 0.0,
                    "thickness": 0.3 + 0.4 * self._g("branching"),
                    "alive": True,
                }
            )
        self.agent_positions = [(n["x"], n["y"]) for n in self.network]

    def step(self, dt: float, audio: dict[str, float]) -> None:
        self.age += dt
        audio_kick = self._apply_audio(audio)
        speed = 0.1 + 0.3 * self._g("speed")
        self._decay_field(0.003 + 0.008 * (1.0 - self._g("persistence")))

        new_hyphae = []
        for n in self.network:
            if not n["alive"]:
                continue
            n["length"] += speed * dt * (1.0 + audio_kick * 0.2)
            nx = n["x"] + math.cos(n["angle"]) * speed * dt
            ny = n["y"] + math.sin(n["angle"]) * speed * dt
            n["angle"] += self.rng.uniform(-0.2, 0.2) * self._g("chaos")
            if 0 <= nx <= 1 and 0 <= ny <= 1:
                self._write_trail(nx, ny, n["thickness"] * 0.15, 0.005)
                n["x"], n["y"] = nx, ny
            else:
                n["alive"] = False
                continue
            if self.rng.random() < self._g("branching") * dt * 0.5:
                child = {
                    "x": n["x"],
                    "y": n["y"],
                    "angle": n["angle"] + self.rng.uniform(-0.8, 0.8),
                    "length": 0.0,
                    "thickness": n["thickness"] * 0.8,
                    "alive": True,
                }
                new_hyphae.append(child)

        self.network.extend(new_hyphae)
        self.agent_positions = [(n["x"], n["y"]) for n in self.network if n["alive"]][:40]


# ---------------------------------------------------------------------------
# Behaviour registry
# ---------------------------------------------------------------------------

_BEHAVIOUR_CLASSES = [
    SchoolBehaviour,
    ExplosionBehaviour,
    CoralGrowthBehaviour,
    CrystalBehaviour,
    StormBehaviour,
    GalaxyBehaviour,
    LavaBehaviour,
    SwarmBehaviour,
    VineGrowthBehaviour,
    NeuronBehaviour,
    FungalGrowthBehaviour,
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VisualBehaviourEngine:
    """Manages a single active behaviour and composites it into a field.

    The engine owns one behaviour at a time (the current champion).  When the
    genome family changes the behaviour is replaced.  ``step`` advances the
    simulation; ``get_field`` returns the current 2D float grid for upload to
    the GPU.
    """

    def __init__(self, width: int = 64, height: int = 64, *, seed: int | None = None):
        self.width = max(8, int(width))
        self.height = max(8, int(height))
        self.rng = random.Random(seed)
        self.behaviour: VisualBehaviour | None = None
        self.genome: Any | None = None

    def set_genome(self, genome: Any) -> None:
        """Set the current behaviour genome and rebuild the behaviour."""
        if genome is None:
            self.behaviour = None
            self.genome = None
            return
        self.genome = genome
        family = int(getattr(genome, "behaviour_family", 0)) % FAMILY_COUNT
        if (
            self.behaviour is None
            or int(getattr(self.behaviour.genome, "behaviour_family", -1)) != family
        ):
            self.behaviour = make_behaviour(
                genome, self.width, self.height, seed=self.rng.randint(0, 1 << 30)
            )
        else:
            self.behaviour.genome = genome

    def step(self, dt: float, audio: dict[str, float]) -> None:
        if self.behaviour is None:
            return
        self.behaviour.step(dt, audio)

    def get_field(self) -> list[list[float]]:
        if self.behaviour is None:
            return [[0.0] * self.width for _ in range(self.height)]
        return self.behaviour.field_copy()

    def reset(self) -> None:
        if self.behaviour is not None:
            self.behaviour.reset()

    def summary(self) -> dict[str, Any]:
        if self.behaviour is None:
            return {"family": "none", "mean": 0.0, "agents": 0}
        return {
            "family": getattr(self.behaviour.genome, "family_name", lambda: "unknown")(),
            "mean": _field_mean(self.behaviour.field),
            "agents": len(getattr(self.behaviour, "agent_positions", [])),
        }
