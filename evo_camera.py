#!/usr/bin/env python3

"""evo_camera.py

Emergent camera system for the EvoRenderer project.

The camera becomes another evolving organism.  Instead of static camera genes,
the camera has its own genome, brain, energy, age, and reproductive cycle.
Multiple camera organisms compete; the best one drives the renderer view.

Behaviours:
  * chase        - follow the most active organism
  * flee         - move away from predator-like activity
  * orbit        - circle interesting regions
  * zoom         - zoom into high-activity areas
  * fly_through  - navigate through scene depth
  * spiral       - spiral around center
  * inspect      - zoom into newborn organisms

Extended cinematography genome (spec 10 - "go much further"):
  * focus        - depth of field / focus distance + aperture
  * focus_pull   - speed of focus transitions
  * tracking     - how tightly the camera locks onto / follows subjects
  * exposure     - tonal exposure + automatic exposure response
  * shake        - handheld / seismic shake amount
  * handheld     - gait-based body motion shake (distinct from tremor)
  * framing      - rule-of-thirds placement + headroom bias
  * cinemat      - school of cinematography selected (dolly, handheld, drone,
                   steadicam, snorri, bullet-time, crane, drift...))
  * crane        - vertical arc movement
  * dolly        - lateral/in-out dolly motion
  * drift        - slow lateral wander
  * parallax     - depth-layer offset for pseudo-3D feel
  * pan          - horizontal pan rotation
  * tilt         - vertical tilt rotation
  * shot_length  - hold duration before re-evaluating target
  * transition   - cross-blend strength between states

The Genome is organised into three layers so evolution can act on each
independently:

  * KINEMATIC genes  - where the camera is (orbit / zoom / fov / roll / dolly).
  * OPTICAL  genes    - how the lens sees (focus / aperture / exposure / shake).
  * DIRECTING genes   - what it points at and why (tracking / framing / cinemat
                        style / behaviour blends).
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from collections import deque
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

logger = logging.getLogger(__name__)

# Cinematography "schools" the camera can evolve toward.  Each is a distinct
# grammar of camera grammar that biases the organism's default motion so the
# population can discover whole *styles* of moving the camera rather than just
# single parameters.  Purely biological: a school is just a constellation of
# latent biases, mutated and crossed over like any other gene.
CINEMATOGRAPHY_SCHOOLS = (
    "observational",  # slow, stable, wide - David Attenborough
    "handheld",  # restless, shaky, intimate - doc / horror
    "steadicam",  # smooth gliding follow - Scorsese / Fincher
    "drone",  # high, sweeping, slow descent - aerial
    "dolly",  # mechanical lateral / push-in - Kubrick
    "snorri",  # aggressive whip-pans + vertigo - Fincher / Reznor
    "bullet_time",  # near-frozen orbit around subject - Matrix
    "crane",  # sweeping vertical arc - Vertigo / Spielberg
    "drift",  # slow lateral wander, lazy cine-eye - Wenders
    "parallax",  # layered depth, foreground/background stress - Resnais
)

# Number of schools (kept as a constant so brains can predict label count).
N_SCHOOLS = len(CINEMATOGRAPHY_SCHOOLS)

# ============================================================================
# Camera genome
# ============================================================================


@dataclass
class CameraGenome:
    """Genetic blueprint for camera behaviour.

    All float genes are normalised to roughly 0..1 so mutation, crossover and
    the neural brain share a common scale.  ``cinemat_school`` indexes
    :data:`CINEMATOGRAPHY_SCHOOLS`.
    """

    # --- kinematic (where the camera is) ---------------------------------
    orbit_speed: float = 0.5
    orbit_radius: float = 0.5
    zoom: float = 0.5
    fov: float = 0.5
    roll: float = 0.0
    dolly_speed: float = 0.3
    look_at_bias: float = 0.5
    crane: float = 0.0
    drift: float = 0.0
    parallax: float = 0.3
    pan: float = 0.3
    tilt: float = 0.3
    shot_length: float = 1.0
    dolly: float = 0.5

    # --- optical (how the lens sees) -------------------------------------
    focus_distance: float = 0.5  # DoF focal plane (0 far .. 1 near)
    aperture: float = 0.2  # bokeh / blur strength
    focus_pull: float = 0.3  # speed of focus transitions
    exposure: float = 0.5  # base tonal exposure
    auto_exposure: float = 0.4  # how strongly brightness tracks the scene
    shake: float = 0.15  # handheld / seismic shake amount
    shake_frequency: float = 0.5  # tremor speed
    handheld: float = 0.2  # gait-based body motion shake

    # --- directing (what it points at + why) -----------------------------
    tracking: float = 0.5  # how tightly it follows a subject
    framing: float = 0.5  # rule-of-thirds bias (0 center .. 1 offset)
    headroom: float = 0.5  # vertical placement of subject
    transition: float = 0.3  # cross-blend between camera states
    cinemat_school: int = 0  # index into CINEMATOGRAPHY_SCHOOLS
    cinemat_mix: float = 0.7  # strength of the school bias

    # --- behaviour blends ------------------------------------------------
    chase_attraction: float = 0.5
    flee_repulsion: float = 0.3
    orbit_attraction: float = 0.5
    zoom_activity: float = 0.4
    tunnel_follow: float = 0.2
    spiral_strength: float = 0.3
    inspect_newborn: float = 0.4

    lineage: str = ""
    generation: int = 0
    parent_id: str | None = None
    fitness: float = 0.0
    mutations: list[str] = dc_field(default_factory=list)

    # Continuous (non-normalised) genes - mutated in place, not clamped 0..1.
    CONTINUOUS = frozenset({"cinemat_school"})
    INT = frozenset({"cinemat_school"})

    def __post_init__(self):
        if not self.lineage:
            self.lineage = str(uuid.uuid4())[:8]
        self.cinemat_school = int(self.cinemat_school) % N_SCHOOLS

    # All normalised float genes (everything except lineage/ids/fitness/mutations
    # and the integer cinemat_school).
    FLOAT_GENES = (
        "orbit_speed",
        "orbit_radius",
        "zoom",
        "fov",
        "roll",
        "dolly_speed",
        "look_at_bias",
        "crane",
        "drift",
        "parallax",
        "pan",
        "tilt",
        "shot_length",
        "dolly",
        "focus_distance",
        "aperture",
        "focus_pull",
        "exposure",
        "auto_exposure",
        "shake",
        "shake_frequency",
        "handheld",
        "tracking",
        "framing",
        "headroom",
        "transition",
        "cinemat_mix",
        "chase_attraction",
        "flee_repulsion",
        "orbit_attraction",
        "zoom_activity",
        "tunnel_follow",
        "spiral_strength",
        "inspect_newborn",
    )

    def mutate(self, strength: float = 0.15, large: bool = False) -> None:
        # Normalised float genes.
        for attr in self.FLOAT_GENES:
            if random.random() < (0.45 if large else 0.3):
                v = getattr(self, attr) + random.uniform(-strength, strength)
                setattr(self, attr, max(0.0, min(1.0, v)))
        # Integer school gene - occasionally jump to a different school.
        if random.random() < (0.25 if large else 0.08):
            self.cinemat_school = (int(self.cinemat_school) + random.choice([-1, 1])) % N_SCHOOLS
        self.mutations.append(f"mutate{'!' if large else ''}")
        if len(self.mutations) > 64:
            self.mutations = self.mutations[-64:]

    def clone(self) -> CameraGenome:
        g = type(self)(
            orbit_speed=self.orbit_speed,
            orbit_radius=self.orbit_radius,
            zoom=self.zoom,
            fov=self.fov,
            roll=self.roll,
            dolly_speed=self.dolly_speed,
            look_at_bias=self.look_at_bias,
            crane=self.crane,
            drift=self.drift,
            parallax=self.parallax,
            pan=self.pan,
            tilt=self.tilt,
            shot_length=self.shot_length,
            dolly=self.dolly,
            focus_distance=self.focus_distance,
            aperture=self.aperture,
            focus_pull=self.focus_pull,
            exposure=self.exposure,
            auto_exposure=self.auto_exposure,
            shake=self.shake,
            shake_frequency=self.shake_frequency,
            handheld=self.handheld,
            tracking=self.tracking,
            framing=self.framing,
            headroom=self.headroom,
            transition=self.transition,
            cinemat_school=self.cinemat_school,
            cinemat_mix=self.cinemat_mix,
            chase_attraction=self.chase_attraction,
            flee_repulsion=self.flee_repulsion,
            orbit_attraction=self.orbit_attraction,
            zoom_activity=self.zoom_activity,
            tunnel_follow=self.tunnel_follow,
            spiral_strength=self.spiral_strength,
            inspect_newborn=self.inspect_newborn,
            lineage=self.lineage,
            generation=self.generation,
            parent_id=self.parent_id,
        )
        g.fitness = self.fitness
        g.mutations = list(self.mutations)
        return g

    def crossover(self, other: CameraGenome) -> CameraGenome:
        child = type(self)()
        for attr in self.FLOAT_GENES:
            setattr(
                child, attr, getattr(self, attr) if random.random() < 0.5 else getattr(other, attr)
            )
        child.cinemat_school = (
            self.cinemat_school if random.random() < 0.5 else other.cinemat_school
        )
        child.generation = max(self.generation, other.generation) + 1
        child.parent_id = self.lineage
        child.lineage = str(uuid.uuid4())[:8]
        return child

    def school_name(self) -> str:
        return CINEMATOGRAPHY_SCHOOLS[int(self.cinemat_school) % N_SCHOOLS]

    def to_dict(self) -> dict[str, Any]:
        return {
            "orbit_speed": self.orbit_speed,
            "orbit_radius": self.orbit_radius,
            "zoom": self.zoom,
            "fov": self.fov,
            "roll": self.roll,
            "dolly_speed": self.dolly_speed,
            "look_at_bias": self.look_at_bias,
            "crane": self.crane,
            "drift": self.drift,
            "parallax": self.parallax,
            "pan": self.pan,
            "tilt": self.tilt,
            "shot_length": self.shot_length,
            "dolly": self.dolly,
            "focus_distance": self.focus_distance,
            "aperture": self.aperture,
            "focus_pull": self.focus_pull,
            "exposure": self.exposure,
            "auto_exposure": self.auto_exposure,
            "shake": self.shake,
            "shake_frequency": self.shake_frequency,
            "handheld": self.handheld,
            "tracking": self.tracking,
            "framing": self.framing,
            "headroom": self.headroom,
            "transition": self.transition,
            "cinemat_school": self.cinemat_school,
            "cinemat_mix": self.cinemat_mix,
            "chase_attraction": self.chase_attraction,
            "flee_repulsion": self.flee_repulsion,
            "orbit_attraction": self.orbit_attraction,
            "zoom_activity": self.zoom_activity,
            "tunnel_follow": self.tunnel_follow,
            "spiral_strength": self.spiral_strength,
            "inspect_newborn": self.inspect_newborn,
            "_lineage": self.lineage,
            "_generation": self.generation,
            "_fitness": self.fitness,
            "_parent_id": self.parent_id,
            "_mutations": list(self.mutations),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraGenome:
        g = cls(
            orbit_speed=d.get("orbit_speed", 0.5),
            orbit_radius=d.get("orbit_radius", 0.5),
            zoom=d.get("zoom", 0.5),
            fov=d.get("fov", 0.5),
            roll=d.get("roll", 0.0),
            dolly_speed=d.get("dolly_speed", 0.3),
            look_at_bias=d.get("look_at_bias", 0.5),
            crane=d.get("crane", 0.0),
            drift=d.get("drift", 0.0),
            parallax=d.get("parallax", 0.3),
            pan=d.get("pan", 0.3),
            tilt=d.get("tilt", 0.3),
            shot_length=d.get("shot_length", 1.0),
            dolly=d.get("dolly", 0.5),
            focus_distance=d.get("focus_distance", 0.5),
            aperture=d.get("aperture", 0.2),
            focus_pull=d.get("focus_pull", 0.3),
            exposure=d.get("exposure", 0.5),
            auto_exposure=d.get("auto_exposure", 0.4),
            shake=d.get("shake", 0.15),
            shake_frequency=d.get("shake_frequency", 0.5),
            handheld=d.get("handheld", 0.2),
            tracking=d.get("tracking", 0.5),
            framing=d.get("framing", 0.5),
            headroom=d.get("headroom", 0.5),
            transition=d.get("transition", 0.3),
            cinemat_school=int(d.get("cinemat_school", 0)) % N_SCHOOLS,
            cinemat_mix=d.get("cinemat_mix", 0.7),
            chase_attraction=d.get("chase_attraction", 0.5),
            flee_repulsion=d.get("flee_repulsion", 0.3),
            orbit_attraction=d.get("orbit_attraction", 0.5),
            zoom_activity=d.get("zoom_activity", 0.4),
            tunnel_follow=d.get("tunnel_follow", 0.2),
            spiral_strength=d.get("spiral_strength", 0.3),
            inspect_newborn=d.get("inspect_newborn", 0.4),
        )
        g.lineage = d.get("_lineage", g.lineage)
        g.generation = int(d.get("_generation", 0))
        g.fitness = float(d.get("_fitness", 0.0))
        g.parent_id = d.get("_parent_id")
        g.mutations = list(d.get("_mutations", []))
        return g


# ============================================================================
# Camera brain
# ============================================================================


class CameraBrain:
    """Tiny neural controller for camera organism.

    The brain modulates the genome's home values for every kinematic, optical
    and directing gene, producing *targets* the organism eases toward.  This
    lets evolution sculpt both the baseline genome AND the reactive controller
    that wobbles it in response to the scene.
    """

    IN = 12
    HID = 10
    OUT = 18
    OUT_LABELS = (
        "target_orbit",
        "target_zoom",
        "target_fov",
        "target_roll",
        "dolly_x",
        "dolly_y",
        "dolly_z",
        "target_focus",
        "target_aperture",
        "target_exposure",
        "target_shake",
        "target_tracking",
        "target_framing",
        "target_headroom",
        "target_crane",
        "target_drift",
        "target_pan",
        "target_tilt",
    )

    def __init__(self, seed: int | None = None):
        rnd = random.Random(seed)
        self.w1 = [rnd.uniform(-1.0, 1.0) for _ in range(self.IN * self.HID)]
        self.w2 = [rnd.uniform(-1.0, 1.0) for _ in range(self.HID * self.OUT)]
        self.b1 = [0.0] * self.HID
        self.b2 = [0.0] * self.OUT
        self._last_h: list[float] | None = None
        self._last_in: list[float] | None = None
        self.fit = 0.0

    def clone(self) -> CameraBrain:
        b = CameraBrain()
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


# ============================================================================
# Camera organism
# ============================================================================


class CameraOrganism:
    """An emergent camera organism.

    Drives the renderer view.  Has energy (from scene interestingness),
    territory (viewport dominance), metabolism, age, lifespan, genome,
    brain, and can reproduce / die.  True ecological selection: cameras
    survive by outcompeting others for the driver seat, not by a
    hand-crafted fitness score.
    """

    def __init__(
        self, genome: CameraGenome | None = None, seed: int | None = None, species_id: int = 0
    ):
        self.genome = genome or CameraGenome()
        self.brain = CameraBrain(seed=seed)
        self.energy = random.uniform(0.8, 1.6)
        self.metabolism = random.uniform(0.04, 0.12)
        self.age = 0.0
        self.life_span = random.uniform(80.0, 200.0)
        self.alive = True
        self.generation = 0
        self.offspring_count = 0
        self.territory = random.uniform(0.3, 0.7)
        self.dominance = 0.0
        self.species_id = species_id
        self.species = species_id
        self.radius = random.uniform(0.06, 0.12)
        self.vel = [random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02)]
        self.pos = [0.5, 0.5, 0.0]
        self.orbit = 1.0
        self.zoom = 1.0
        self.fov = 1.5
        self.roll = 0.0
        self.focus = genome.focus_distance if genome else 0.5
        self.aperture = genome.aperture if genome else 0.2
        self.exposure = genome.exposure if genome else 0.5
        self.shake = genome.shake if genome else 0.15
        self.handheld = genome.handheld if genome else 0.2
        self.tracking = genome.tracking if genome else 0.5
        self.framing = genome.framing if genome else 0.5
        self.headroom = genome.headroom if genome else 0.5
        self.dolly = genome.dolly if genome else 0.5
        self.crane = genome.crane if genome else 0.0
        self.drift = genome.drift if genome else 0.0
        self.parallax = genome.parallax if genome else 0.3
        self.pan = genome.pan if genome else 0.3
        self.tilt = genome.tilt if genome else 0.3
        self.focus_target = self.focus
        self.vel_orbit = 0.0
        self.vel_zoom = 0.0
        self._shot_timer = 0.0
        self._prev_target_orbit = 0.0
        self._prev_target_zoom = 0.0
        self._prev_target_fov = 0.0
        self._transition_blend = 1.0
        self.shake_phase = random.uniform(0.0, 6.2831)
        self.target_look_at = [0.5, 0.5]
        self.recently_born: list[tuple[float, float, float]] = []
        self._birth_memory: list[tuple[float, float, float]] = []
        self._energy_history: deque = deque(maxlen=32)
        self._dominance_timer = 0.0
        self._last_food = 0.0

    def perceive(
        self,
        audio: dict[str, float],
        scene_entropy: float,
        scene_motion: float,
        organisms: list[Any],
        births: list[tuple[float, float, float]],
    ) -> list[float]:
        bass = float(audio.get("bass", 0.0))
        mid = float(audio.get("mid", 0.0))
        treb = float(audio.get("treb", 0.0))
        beat = float(audio.get("beat", 0.0))
        n_pred = sum(
            1 for o in organisms if getattr(o, "alive", False) and getattr(o, "energy", 0) > 2.0
        )
        activity = min(1.0, scene_entropy * 0.5 + scene_motion * 0.5)
        self._birth_memory = births[-8:]
        n_organisms = max(1, len(organisms))
        avg_energy = sum(getattr(o, "energy", 0.0) for o in organisms) / n_organisms
        return [
            bass,
            mid,
            treb,
            beat,
            activity,
            min(1.0, n_pred / n_organisms),
            min(1.0, self.energy / 3.0),
            min(1.0, self.age / max(self.life_span, 1.0)),
            scene_entropy,  # 9  bright / busy -> exposure
            scene_motion,  # 10 motion -> shake / tracking
            min(1.0, avg_energy),  # 11 crowd energy -> interest
            min(1.0, len(self._birth_memory) / 5.0),  # 12 births -> inspect
        ]

    def _school_bias(self, g: CameraGenome) -> dict[str, float]:
        """Latent biases contributed by the chosen cinematography school.

        Returns multipliers (0..~1.5) applied to the respective gene targets.
        This is what lets a whole *style* evolve as a single heritable gene.
        """
        school = int(g.cinemat_school) % N_SCHOOLS
        base = {
            "shake": 1.0,
            "orbit": 1.0,
            "zoom": 1.0,
            "fov": 1.0,
            "tracking": 1.0,
            "focus": 1.0,
            "exposure": 1.0,
            "framing": 1.0,
            "dolly": 1.0,
            "spiral": 1.0,
            "crane": 1.0,
            "drift": 1.0,
            "parallax": 1.0,
            "pan": 1.0,
            "tilt": 1.0,
            "handheld": 1.0,
            "focus_pull": 1.0,
            "transition": 1.0,
        }
        if school == 0:  # observational - calm, wide, steady
            base.update(
                shake=0.15,
                orbit=0.7,
                fov=1.3,
                tracking=0.5,
                focus=0.7,
                dolly=0.3,
                handheld=0.2,
                parallax=0.6,
            )
        elif school == 1:  # handheld - restless, shaky, intimate
            base.update(
                shake=1.6,
                orbit=1.2,
                fov=0.9,
                tracking=1.2,
                framing=1.3,
                dolly=1.2,
                handheld=2.0,
                pan=1.4,
                drift=1.3,
            )
        elif school == 2:  # steadicam - smooth gliding follow
            base.update(
                shake=0.3,
                tracking=1.5,
                dolly=1.4,
                orbit=0.8,
                focus=0.9,
                handheld=0.6,
                drift=0.8,
                transition=1.2,
            )
        elif school == 3:  # drone - high sweeping descent
            base.update(
                shake=0.2,
                dolly=1.4,
                orbit=1.3,
                fov=1.2,
                tracking=0.6,
                framing=1.1,
                crane=1.6,
                drift=1.2,
                parallax=1.1,
            )
        elif school == 4:  # dolly - mechanical lateral / push-in
            base.update(
                shake=0.1,
                dolly=1.6,
                tracking=0.9,
                zoom=1.2,
                orbit=0.5,
                framing=0.8,
                drift=1.1,
                focus_pull=1.3,
                transition=1.5,
            )
        elif school == 5:  # snorri - aggressive whip-pans + vertigo
            base.update(
                shake=0.8,
                orbit=1.6,
                zoom=1.5,
                fov=1.4,
                tracking=1.3,
                framing=1.2,
                spiral=1.4,
                pan=1.8,
                tilt=1.6,
            )
        elif school == 6:  # bullet_time - near-frozen orbit around subject
            base.update(
                shake=0.05,
                orbit=1.8,
                tracking=1.6,
                zoom=0.7,
                dolly=0.2,
                fov=0.8,
                spiral=1.2,
                focus_pull=0.4,
                transition=0.8,
                parallax=1.5,
            )
        elif school == 7:  # crane - sweeping vertical arc
            base.update(
                shake=0.05,
                crane=1.8,
                dolly=0.4,
                orbit=0.6,
                fov=1.3,
                tracking=0.7,
                framing=1.1,
                parallax=1.4,
                transition=1.6,
                focus_pull=1.2,
            )
        elif school == 8:  # drift - slow lateral wander, lazy cine-eye
            base.update(
                shake=0.08,
                orbit=0.3,
                drift=2.0,
                dolly=0.5,
                fov=1.1,
                tracking=0.4,
                framing=1.0,
                parallax=0.9,
                transition=1.8,
                handheld=0.7,
            )
        elif school == 9:  # parallax - layered depth, foreground/background stress
            base.update(
                parallax=2.0,
                drift=1.2,
                dolly=0.6,
                orbit=0.7,
                fov=1.4,
                tracking=0.8,
                focus_pull=1.4,
                transition=1.3,
                handheld=0.5,
            )
        return base

    def step(
        self,
        dt: float,
        audio: dict[str, float],
        scene_entropy: float,
        scene_motion: float,
        organisms: list[Any],
        births: list[tuple[float, float, float]],
    ) -> None:
        if not self.alive:
            return
        self.age += dt
        inputs = self.perceive(audio, scene_entropy, scene_motion, organisms, births)
        acts = self.brain.process(inputs)

        g = self.genome
        school = self._school_bias(g)
        mix = g.cinemat_mix

        def biased(gene: str, val: float) -> float:
            return val * (1.0 + (school.get(gene, 1.0) - 1.0) * mix)

        # --- kinematic targets (brain wobbles genome home values) --------
        target_orbit = biased("orbit", g.orbit_radius) + acts[0] * 0.3
        target_zoom = biased("zoom", g.zoom) + acts[1] * 0.3
        target_fov = biased("fov", g.fov) + acts[2] * 0.3
        target_roll = g.roll + acts[3] * 0.2
        dolly_z = acts[6] * biased("dolly", g.dolly_speed) * 0.02
        target_crane = biased("crane", g.crane) + acts[14] * 0.2
        target_drift = acts[15] * biased("drift", g.drift) * 0.3
        target_pan = biased("pan", g.pan) + acts[16] * 0.2
        target_tilt = biased("tilt", g.tilt) + acts[17] * 0.2

        # --- shot_length: hold current target until timer expires -------------
        hold_time = max(0.1, g.shot_length)
        self._shot_timer += dt
        if self._shot_timer >= hold_time:
            self._shot_timer = 0.0
            self._prev_target_orbit = target_orbit
            self._prev_target_zoom = target_zoom
            self._prev_target_fov = target_fov
            blend_t = g.transition
            self._transition_blend = blend_t
        else:
            t_b = min(1.0, self._shot_timer / max(0.1, hold_time))
            target_orbit = self._prev_target_orbit * (
                1.0 - t_b * self._transition_blend
            ) + target_orbit * (t_b * self._transition_blend)
            target_zoom = self._prev_target_zoom * (
                1.0 - t_b * self._transition_blend
            ) + target_zoom * (t_b * self._transition_blend)
            target_fov = self._prev_target_fov * (
                1.0 - t_b * self._transition_blend
            ) + target_fov * (t_b * self._transition_blend)

        # --- optical targets ---------------------------------------------
        self.focus_target = g.focus_distance + target_tilt * 0.2
        focus_speed = 0.02 + g.focus_pull * 0.12
        self.focus += (self.focus_target - self.focus) * focus_speed * (0.5 + g.dolly_speed)
        target_aperture = g.aperture + acts[8] * 0.2
        target_exposure = biased("exposure", g.exposure) + acts[9] * 0.2

        # Handheld: gait-based whole-body shake (low freq, high amp) distinct
        # from seismic tremor (high freq, low amp).  Both layer together.
        if g.handheld > 0.01:
            hs = g.handheld * g.shake
            gait = math.sin(self.age * 3.7) * math.cos(self.age * 2.3)
            gait += 0.5 * math.sin(self.age * 5.1 + 1.7)
            target_shake_cmp = g.shake + acts[10] * 0.2 + gait * hs
        else:
            target_shake_cmp = g.shake + acts[10] * 0.2

        # --- directing targets -------------------------------------------
        target_tracking = biased("tracking", g.tracking) + acts[11] * 0.2
        target_framing = biased("framing", g.framing) + acts[12] * 0.2
        target_headroom = g.headroom + acts[13] * 0.2

        # Parallax bias on look target (depth layer separation)
        parallax_offset = (g.parallax - 0.5) * 0.3

        # Behaviour blending of the look-at target.
        look_x, look_y = 0.5 + parallax_offset * 0.3, 0.5 + parallax_offset * 0.3
        if organisms and g.chase_attraction > 0.1:
            best = max(organisms, key=lambda o: getattr(o, "fitness", 0.0), default=None)
            if best is not None:
                bx, by = best.pos[0], best.pos[1]
                look_x += (bx - 0.5) * g.chase_attraction * target_tracking * 0.8
                look_y += (by - 0.5) * g.chase_attraction * target_tracking * 0.8

        if g.flee_repulsion > 0.1:
            preds = [o for o in organisms if getattr(o, "energy", 0) > 2.0]
            if preds:
                px = sum(o.pos[0] for o in preds) / len(preds)
                py = sum(o.pos[1] for o in preds) / len(preds)
                look_x -= (px - 0.5) * g.flee_repulsion * 0.6
                look_y -= (py - 0.5) * g.flee_repulsion * 0.6

        if g.orbit_attraction > 0.1:
            self.vel_orbit += g.orbit_attraction * 0.02 * dt * 60.0
        if g.spiral_strength > 0.1:
            self.vel_orbit += g.spiral_strength * school.get("spiral", 1.0) * 0.01 * dt * 60.0

        if g.zoom_activity > 0.1:
            target_zoom = max(0.2, min(3.0, target_zoom + scene_entropy * g.zoom_activity * 0.5))

        if g.inspect_newborn > 0.1 and births:
            latest = births[-1]
            look_x, look_y = latest[0], latest[1]
            target_zoom = min(2.5, target_zoom + g.inspect_newborn * 0.5)

        # Rule-of-thirds framing: bias the look-at target toward a power-point
        # of the frame rather than dead centre.  framing gene controls how far.
        third = (target_framing - 0.5) * 0.5
        look_x = max(0.05, min(0.95, look_x + third))
        look_y = max(0.05, min(0.95, look_y + (target_headroom - 0.5) * 0.4))

        # --- smooth update ------------------------------------------------
        speed = 0.05 * (0.5 + g.dolly_speed)
        self.orbit += (target_orbit - self.orbit) * speed
        self.zoom += (target_zoom - self.zoom) * speed
        self.fov += (target_fov - self.fov) * speed
        self.roll += (target_roll - self.roll) * speed * 0.5
        self.crane += (target_crane - self.crane) * speed * 0.3
        self.drift += (target_drift - self.drift) * speed * 0.2
        self.pan += (target_pan - self.pan) * speed * 0.3
        self.tilt += (target_tilt - self.tilt) * speed * 0.3
        self.aperture += (target_aperture - self.aperture) * speed
        self.exposure += (target_exposure - self.exposure) * speed
        self.shake += (max(0.0, target_shake_cmp) - self.shake) * speed
        self.tracking += (target_tracking - self.tracking) * speed
        self.framing += (target_framing - self.framing) * speed
        self.headroom += (target_headroom - self.headroom) * speed
        self.pos[0] += (look_x - self.pos[0]) * speed * (0.4 + self.tracking)
        self.pos[1] += (look_y - self.pos[1]) * speed * (0.4 + self.tracking)
        self.pos[2] += dolly_z * speed
        self.vel_orbit *= 0.95
        self.orbit += self.vel_orbit * dt

        self.orbit = max(0.2, min(5.0, self.orbit))
        self.zoom = max(0.2, min(4.0, self.zoom))
        self.fov = max(0.5, min(3.0, self.fov))
        self.roll = max(-3.14, min(3.14, self.roll))
        self.crane = max(-2.0, min(2.0, self.crane))
        self.drift = max(-1.0, min(1.0, self.drift))
        self.pan = max(-3.14, min(3.14, self.pan))
        self.tilt = max(-3.14, min(3.14, self.tilt))
        self.focus = max(0.0, min(1.0, self.focus))
        self.aperture = max(0.0, min(1.0, self.aperture))
        self.exposure = max(0.0, min(1.5, self.exposure))
        self.shake = max(0.0, min(1.5, self.shake))
        self.tracking = max(0.0, min(1.0, self.tracking))
        self.framing = max(0.0, min(1.0, self.framing))
        self.headroom = max(0.0, min(1.0, self.headroom))

        # Ecological energy: the scene is the environment.  Interesting
        # scenes provide food.  The dominant camera (highest energy) gets
        # the full bounty; subordinate cameras get only scraps.
        scene_interest = min(
            1.0,
            scene_entropy * 0.3
            + scene_motion * 0.3
            + float(audio.get("beat", 0.0)) * 0.2
            + min(1.0, len(self._birth_memory) / 5.0) * 0.2,
        )
        # Heritable style bonus: without this, energy depended only on
        # metabolism and dominance, so the camera genome/brain had zero
        # effect on survival and selection could not act on behaviour.
        style = self._style_bonus()
        if style > 0.0:
            scene_interest = min(1.0, scene_interest + style)
        food = scene_interest * (0.2 + 0.8 * self.dominance)
        self._last_food = food
        self._energy_history.append(self.energy)
        # ``food`` is a per-second intake *rate* and must be scaled by dt just
        # like the metabolism drain.  Adding it unscaled every frame pinned
        # energy at the 4.0 cap within ~1s, which collapsed dominance to zero
        # for the whole population and froze camera selection.
        self.energy = max(
            0.0,
            min(4.0, self.energy + (food * 8.0 - self.metabolism * 12.0) * dt),
        )

        # Territory: dominant camera expands its control, subordinates shrink.
        if self.dominance > 0.5:
            self.territory = min(1.0, self.territory + dt * 0.1)
        else:
            self.territory = max(0.05, self.territory - dt * 0.15)

        if self.age >= self.life_span or self.energy <= 0.0:
            self.alive = False

    def _style_bonus(self) -> float:
        """0..~0.4 food bonus derived from this camera's own shooting style.

        Base organisms are style-neutral; subclasses override with metrics
        that reflect what the genome/brain actually produce (shot variety,
        cut rhythm, engagement).
        """
        return 0.0

    @property
    def fitness(self) -> float:
        return max(0.0, self.energy)

    @fitness.setter
    def fitness(self, value: float) -> None:
        self.energy = max(0.0, min(4.0, float(value)))

    def clone_child(self) -> CameraOrganism:
        child = CameraOrganism(
            self.genome.clone(),
            seed=random.randint(0, 1 << 30),
            species_id=self.species_id,
        )
        child.genome.mutate(strength=0.12)
        child.brain = self.brain.clone()
        child.brain.mutate(strength=0.15)
        child.energy = self.energy * 0.4
        child.metabolism = self.metabolism * random.uniform(0.8, 1.25)
        child.generation = self.generation + 1
        child.offspring_count = 0
        child.territory = self.territory * random.uniform(0.8, 1.2)
        child.dominance = 0.0
        child.orbit = self.orbit * random.uniform(0.8, 1.2)
        child.zoom = self.zoom * random.uniform(0.8, 1.2)
        child.focus = self.focus * random.uniform(0.8, 1.2)
        child.aperture = self.aperture * random.uniform(0.8, 1.2)
        child.exposure = self.exposure * random.uniform(0.8, 1.2)
        child.shake = self.shake * random.uniform(0.8, 1.2)
        child.handheld = self.handheld * random.uniform(0.8, 1.2)
        child.tracking = self.tracking * random.uniform(0.8, 1.2)
        child.framing = self.framing * random.uniform(0.8, 1.2)
        child.headroom = self.headroom * random.uniform(0.8, 1.2)
        child.crane = self.crane * random.uniform(0.8, 1.2)
        child.drift = self.drift * random.uniform(0.8, 1.2)
        child.parallax = self.parallax * random.uniform(0.8, 1.2)
        child.pan = self.pan * random.uniform(0.8, 1.2)
        child.tilt = self.tilt * random.uniform(0.8, 1.2)
        child.dolly = self.dolly * random.uniform(0.8, 1.2)
        self.offspring_count += 1
        return child

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": self.genome.to_dict(),
            "brain": {
                "w1": self.brain.w1,
                "w2": self.brain.w2,
                "b1": self.brain.b1,
                "b2": self.brain.b2,
            },
            "energy": self.energy,
            "metabolism": self.metabolism,
            "age": self.age,
            "life_span": self.life_span,
            "alive": self.alive,
            "generation": self.generation,
            "offspring_count": self.offspring_count,
            "territory": self.territory,
            "dominance": self.dominance,
            "pos": list(self.pos),
            "orbit": self.orbit,
            "zoom": self.zoom,
            "fov": self.fov,
            "roll": self.roll,
            "crane": self.crane,
            "drift": self.drift,
            "pan": self.pan,
            "tilt": self.tilt,
            "focus": self.focus,
            "focus_target": self.focus_target,
            "aperture": self.aperture,
            "exposure": self.exposure,
            "shake": self.shake,
            "handheld": self.handheld,
            "tracking": self.tracking,
            "framing": self.framing,
            "headroom": self.headroom,
            "parallax": self.parallax,
            "shake_phase": self.shake_phase,
            "lineage": self.genome.lineage,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraOrganism:
        obj = cls(CameraGenome.from_dict(d.get("genome", {})))
        obj.brain.w1 = d.get("brain", {}).get("w1", obj.brain.w1)
        obj.brain.w2 = d.get("brain", {}).get("w2", obj.brain.w2)
        obj.brain.b1 = d.get("brain", {}).get("b1", obj.brain.b1)
        obj.brain.b2 = d.get("brain", {}).get("b2", obj.brain.b2)
        obj.energy = float(d.get("energy", 1.0))
        obj.metabolism = float(d.get("metabolism", 0.08))
        obj.age = float(d.get("age", 0.0))
        obj.life_span = float(d.get("life_span", 120.0))
        obj.alive = bool(d.get("alive", True))
        obj.generation = int(d.get("generation", 0))
        obj.offspring_count = int(d.get("offspring_count", 0))
        obj.territory = float(d.get("territory", 0.5))
        obj.dominance = float(d.get("dominance", 0.0))
        obj.pos = list(d.get("pos", [0.5, 0.5, 0.0]))
        obj.orbit = float(d.get("orbit", 1.0))
        obj.zoom = float(d.get("zoom", 1.0))
        obj.fov = float(d.get("fov", 1.5))
        obj.roll = float(d.get("roll", 0.0))
        obj.crane = float(d.get("crane", 0.0))
        obj.drift = float(d.get("drift", 0.0))
        obj.dolly = float(d.get("dolly", 0.5))
        obj.pan = float(d.get("pan", 0.0))
        obj.tilt = float(d.get("tilt", 0.0))
        obj.transition = float(d.get("transition", 0.3))
        obj.focus = float(d.get("focus", 0.5))
        obj.focus_target = float(d.get("focus_target", obj.focus))
        obj.aperture = float(d.get("aperture", 0.2))
        obj.exposure = float(d.get("exposure", 0.5))
        obj.shake = float(d.get("shake", 0.15))
        obj.handheld = float(d.get("handheld", 0.2))
        obj.tracking = float(d.get("tracking", 0.5))
        obj.framing = float(d.get("framing", 0.5))
        obj.headroom = float(d.get("headroom", 0.5))
        obj.parallax = float(d.get("parallax", 0.3))
        obj.shake_phase = float(d.get("shake_phase", 0.0))
        return obj


# ============================================================================
# Camera population
# ============================================================================


class CameraPopulation:
    """Manages competing camera organisms with true ecological selection.

    Cameras compete for the driver seat.  The dominant camera (highest
    energy) controls the viewport and receives the full bounty of the
    scene.  Subordinate cameras receive only scraps and slowly starve.
    Reproduction and death are governed by energy, not by a hand-crafted
    fitness score.
    """

    def __init__(self, size: int = 3):
        self.size = max(1, min(6, int(size)))
        self.cameras: list[CameraOrganism] = []
        self._generation = 0
        self._spawn_timer = 0.0
        self._total_births = 0
        self._total_deaths = 0
        # Driver-seat term limits: without them the energy leader is
        # unbeatable while alive (winner-take-all food), so children always
        # starve and "evolution" degenerates into random restarts.
        self._last_dominant: CameraOrganism | None = None
        self._reign_timer: float = 0.0
        self._reign_term: float = random.uniform(25.0, 45.0)
        self._usurper: CameraOrganism | None = None
        self._usurper_timer: float = 0.0
        for i in range(self.size):
            self.cameras.append(CameraOrganism(species_id=1000 + i))

    def _update_dominance(self) -> None:
        alive = [c for c in self.cameras if c.alive]
        if not alive:
            return
        if len(alive) == 1:
            alive[0].dominance = 1.0
            return
        max_energy = max(c.energy for c in alive)
        min_energy = min(c.energy for c in alive)
        span = max_energy - min_energy
        if span < 0.05:
            # Energies are effectively tied (fresh spawn or saturation):
            # break the tie by rank so the leader still receives the dominant
            # food share instead of every camera getting identical scraps.
            ranked = sorted(alive, key=lambda c: c.energy, reverse=True)
            for rank, cam in enumerate(ranked):
                cam.dominance = 1.0 - rank / max(1, len(ranked) - 1)
            return
        for cam in alive:
            cam.dominance = max(0.0, min(1.0, (cam.energy - min_energy) / span))

    def dominant(self) -> CameraOrganism:
        alive = [c for c in self.cameras if c.alive]
        if not alive:
            return self.cameras[0] if self.cameras else CameraOrganism()
        # During a coup the crowned challenger holds the driver seat even if
        # its raw energy has not (yet) overtaken the old leader's — otherwise
        # term limits would shift food but never the visible camera.
        usurper = getattr(self, "_usurper", None)
        if usurper is not None and usurper.alive and getattr(self, "_usurper_timer", 0.0) > 0.0:
            return usurper
        return max(alive, key=lambda c: c.energy)

    def best(self) -> CameraOrganism:
        return self.dominant()

    def step(
        self,
        dt: float,
        audio: dict[str, float],
        scene_entropy: float,
        scene_motion: float,
        organisms: list[Any],
        births: list[tuple[float, float, float]],
    ) -> None:
        births_list = list(births)

        # Update dominance ranking before stepping so each camera knows
        # whether it is the dominant driver or a subordinate observer.
        self._update_dominance()

        # Term limits: periodically crown the strongest challenger for a
        # short reign and suppress the old leader's food share, so the
        # driver seat can actually change hands and selection has something
        # to act on.
        alive_now = [c for c in self.cameras if c.alive]
        if len(alive_now) > 1:
            leader = self.dominant()
            if leader is self._last_dominant:
                self._reign_timer += dt
            else:
                self._last_dominant = leader
                self._reign_timer = 0.0
            if self._usurper_timer > 0.0:
                self._usurper_timer -= dt
                if self._usurper is not None and self._usurper.alive:
                    self._usurper.dominance = max(self._usurper.dominance, 0.9)
                    if leader is not self._usurper:
                        leader.dominance = min(leader.dominance, 0.4)
                else:
                    self._usurper = None
                    self._usurper_timer = 0.0
            elif self._reign_timer > self._reign_term:
                challenger = max(
                    (c for c in alive_now if c is not leader),
                    key=lambda c: c.energy,
                    default=None,
                )
                if challenger is not None:
                    self._usurper = challenger
                    self._usurper_timer = self._reign_term * 0.5
                    self._reign_timer = 0.0

        for cam in self.cameras:
            if cam.alive:
                cam.step(dt, audio, scene_entropy, scene_motion, organisms, births_list)

        # Ecological turnover: remove the dead.
        before = len(self.cameras)
        self.cameras = [c for c in self.cameras if c.alive]
        after = len(self.cameras)
        self._total_deaths += before - after

        # Maintain minimum population: spawn new random cameras if extinct.
        while len(self.cameras) < 1:
            self.cameras.append(CameraOrganism())

        # Reproduction: dominant camera with excess energy spawns child.
        self._spawn_timer += dt
        if self._spawn_timer > 8.0:
            self._spawn_timer = 0.0
            dom = self.dominant()
            if dom.energy > 1.0 and random.random() < 0.7:
                child = dom.clone_child()
                child.energy = dom.energy * 0.45
                child.generation = self._generation
                self._on_child_born(child)
                self.cameras.append(child)
                self._total_births += 1
                # Cap population: weakest camera dies when over capacity.
                if len(self.cameras) > self.size + 3:
                    weakest = min(self.cameras, key=lambda c: c.energy)
                    weakest.alive = False
                    self.cameras = [c for c in self.cameras if c.alive]

        # Trim excess cameras beyond soft cap.
        if len(self.cameras) > self.size + 4:
            surplus = len(self.cameras) - (self.size + 2)
            for _ in range(surplus):
                weakest = min(self.cameras, key=lambda c: c.energy)
                weakest.alive = False
            self.cameras = [c for c in self.cameras if c.alive]

        if random.random() < dt / 50.0:
            self._generation += 1
            for cam in self.cameras:
                cam.generation = self._generation

    def _on_child_born(self, child: CameraOrganism) -> None:
        """Hook invoked when a child is spawned. Subclasses may speciate."""
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "cameras": [c.to_dict() for c in self.cameras],
            "generation": self._generation,
            "total_births": self._total_births,
            "total_deaths": self._total_deaths,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraPopulation:
        obj = cls(d.get("size", 3))
        obj.cameras = [CameraOrganism.from_dict(c) for c in d.get("cameras", [])]
        obj._generation = int(d.get("generation", 0))
        obj._total_births = int(d.get("total_births", 0))
        obj._total_deaths = int(d.get("total_deaths", 0))
        return obj


# ============================================================================
# Feature extension — camera as evolving species
# ============================================================================


@dataclass
class CameraSpeciesDef:
    """A camera species: distinct visual style / grammar that can speciate."""

    sid: int
    name: str
    color: tuple[float, float, float]
    born_at: float = 0.0
    avg_fitness: float = 0.0
    extinction_timer: float = 0.0
    niche: str = "general"
    cut_style: str = "hold"  # "hold", "cut", "whip", "jump"
    movement_style: str = "smooth"  # "smooth", "handheld", "drone", "snorri"


# ============================================================================
# Extended camera genome — anticipation, cuts, variety, boring-view avoidance
# ============================================================================


class ExtendedCameraGenome(CameraGenome):
    """Camera genome extended with species-level and editorial genes.

    Added genes:
      anticipation    — predictive lead on moving subjects (0 = reactive, 1 = oracle)
      lead_factor     — how far ahead of a subject the camera aims
      cut_frequency   — probability of cutting vs holding on beat / scene change
      whip_pan        — speed of whip-pan transitions between subjects
      jump_cut        — tendency to make sudden discontinuous cuts
      hold_strength   — resistance to breaking an established shot
      restlessness    — tendency to move when scene entropy is low
      boredom_threshold — entropy level below which the camera gets bored
      follow_smoothing — inertia of tracking movement (low = snappy, high = floaty)
      cut_on_beat     — synchronise cuts to musical downbeats
      variety_bonus   — weight given to visual variety vs coherence
    """

    FLOAT_GENES = (
        *CameraGenome.FLOAT_GENES,
        "anticipation",
        "lead_factor",
        "cut_frequency",
        "whip_pan",
        "jump_cut",
        "hold_strength",
        "restlessness",
        "boredom_threshold",
        "follow_smoothing",
        "cut_on_beat",
        "variety_bonus",
    )

    def __init__(self, **kwargs):
        self.anticipation: float = kwargs.pop("anticipation", 0.4)
        self.lead_factor: float = kwargs.pop("lead_factor", 0.3)
        self.cut_frequency: float = kwargs.pop("cut_frequency", 0.3)
        self.whip_pan: float = kwargs.pop("whip_pan", 0.2)
        self.jump_cut: float = kwargs.pop("jump_cut", 0.1)
        self.hold_strength: float = kwargs.pop("hold_strength", 0.5)
        self.restlessness: float = kwargs.pop("restlessness", 0.3)
        self.boredom_threshold: float = kwargs.pop("boredom_threshold", 0.25)
        self.follow_smoothing: float = kwargs.pop("follow_smoothing", 0.4)
        self.cut_on_beat: float = kwargs.pop("cut_on_beat", 0.4)
        self.variety_bonus: float = kwargs.pop("variety_bonus", 0.3)
        super().__init__(**kwargs)

    def mutate(self, strength: float = 0.15, large: bool = False) -> None:
        super().mutate(strength=strength, large=large)
        for attr in self.FLOAT_GENES:
            if attr in (
                "anticipation",
                "lead_factor",
                "cut_frequency",
                "whip_pan",
                "jump_cut",
                "hold_strength",
                "restlessness",
                "boredom_threshold",
                "follow_smoothing",
                "cut_on_beat",
                "variety_bonus",
            ) and random.random() < (0.45 if large else 0.3):
                v = getattr(self, attr) + random.uniform(-strength, strength)
                setattr(self, attr, max(0.0, min(1.0, v)))

    def clone(self) -> ExtendedCameraGenome:
        g = type(self)()
        for attr in self.FLOAT_GENES:
            setattr(g, attr, getattr(self, attr))
        g.cinemat_school = self.cinemat_school
        g.lineage = self.lineage
        g.generation = self.generation
        g.parent_id = self.parent_id
        g.fitness = self.fitness
        g.mutations = list(self.mutations)
        return g

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "anticipation": self.anticipation,
                "lead_factor": self.lead_factor,
                "cut_frequency": self.cut_frequency,
                "whip_pan": self.whip_pan,
                "jump_cut": self.jump_cut,
                "hold_strength": self.hold_strength,
                "restlessness": self.restlessness,
                "boredom_threshold": self.boredom_threshold,
                "follow_smoothing": self.follow_smoothing,
                "cut_on_beat": self.cut_on_beat,
                "variety_bonus": self.variety_bonus,
            }
        )
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtendedCameraGenome:
        g = super().from_dict(d)
        g.anticipation = float(d.get("anticipation", 0.4))
        g.lead_factor = float(d.get("lead_factor", 0.3))
        g.cut_frequency = float(d.get("cut_frequency", 0.3))
        g.whip_pan = float(d.get("whip_pan", 0.2))
        g.jump_cut = float(d.get("jump_cut", 0.1))
        g.hold_strength = float(d.get("hold_strength", 0.5))
        g.restlessness = float(d.get("restlessness", 0.3))
        g.boredom_threshold = float(d.get("boredom_threshold", 0.25))
        g.follow_smoothing = float(d.get("follow_smoothing", 0.4))
        g.cut_on_beat = float(d.get("cut_on_beat", 0.4))
        g.variety_bonus = float(d.get("variety_bonus", 0.3))
        return g


# ============================================================================
# Extended camera organism — species, shots, boredom, cuts
# ============================================================================


class ExtendedCameraOrganism(CameraOrganism):
    """Camera organism with species identity, shot history, and boredom tracking."""

    def __init__(
        self,
        genome: ExtendedCameraGenome | None = None,
        seed: int | None = None,
        species_id: int = 0,
    ):
        super().__init__(genome=genome, seed=seed, species_id=species_id)
        self.species_id = species_id
        self.shot_history: list[dict[str, float]] = []
        self.boredom_timer: float = 0.0
        self.last_cut_time: float = 0.0
        self.cut_log: list[dict[str, Any]] = []
        self.frame_signatures: deque = deque(maxlen=64)
        self._last_shot_hash: int | None = None
        self._hold_timer: float = 0.0
        self._whip_phase: float = 0.0

    def _record_shot(self) -> None:
        sig = {
            "orbit": self.orbit,
            "zoom": self.zoom,
            "fov": self.fov,
            "roll": self.roll,
            "focus": self.focus,
            "aperture": self.aperture,
            "exposure": self.exposure,
            "shake": self.shake,
            "tracking": self.tracking,
            "framing": self.framing,
            "headroom": self.headroom,
            "pan": self.pan,
            "tilt": self.tilt,
            "crane": self.crane,
            "drift": self.drift,
            "dolly": self.dolly,
            "pos_x": self.pos[0],
            "pos_y": self.pos[1],
        }
        h = hash(tuple(round(v, 3) for v in sig.values()))
        self._last_shot_hash = h
        self.shot_history.append(sig)
        if len(self.shot_history) > 128:
            self.shot_history = self.shot_history[-128:]
        self.frame_signatures.append(h)

    def _should_cut(self, dt: float, audio: dict[str, float], scene_entropy: float) -> bool:
        g = self.genome
        beat = float(audio.get("beat", 0.0))
        beat_delta = abs(beat - getattr(self, "_prev_beat", 0.0))
        self._prev_beat = beat

        self._hold_timer += dt
        hold_time = max(0.2, g.shot_length * (1.0 + (1.0 - g.hold_strength)))

        cut = False
        reason = "hold"

        if self._hold_timer >= hold_time:
            cut = True
            reason = "hold_expired"

        if g.cut_on_beat > 0.1 and beat_delta > 0.15 and self._hold_timer > 0.3:
            cut = True
            reason = "beat"

        if scene_entropy < g.boredom_threshold and g.restlessness > 0.2:
            self.boredom_timer += dt
            if self.boredom_timer > 1.5:
                cut = True
                reason = "bored"
                self.boredom_timer = 0.0
        else:
            self.boredom_timer = max(0.0, self.boredom_timer - dt * 2.0)

        if random.random() < g.jump_cut * dt * 2.0:
            cut = True
            reason = "jump"

        if cut:
            self.last_cut_time = self.age
            self._hold_timer = 0.0
            self.cut_log.append(
                {
                    "time": self.age,
                    "reason": reason,
                    "beat": beat,
                    "entropy": scene_entropy,
                    "shot_hash": self._last_shot_hash,
                }
            )
            if len(self.cut_log) > 64:
                self.cut_log = self.cut_log[-64:]
        return cut

    def _anticipation_target(self, organisms: list[Any], dt: float) -> tuple[float, float]:
        """Predict where the most interesting subject will be."""
        if not organisms or self.genome.anticipation < 0.05:
            return self.pos[0], self.pos[1]
        best = max(organisms, key=lambda o: getattr(o, "fitness", 0.0), default=None)
        if best is None:
            return self.pos[0], self.pos[1]
        look_x, look_y = best.pos[0], best.pos[1]
        if hasattr(best, "vel") and len(best.vel) >= 2:
            lead = self.genome.lead_factor * self.genome.anticipation
            look_x += best.vel[0] * lead * 3.0
            look_y += best.vel[1] * lead * 3.0
        return max(0.05, min(0.95, look_x)), max(0.05, min(0.95, look_y))

    def _follow_movement(self, organisms: list[Any], dt: float) -> None:
        """Update tracking target with predictive lead and smoothing."""
        if not organisms or self.genome.tracking < 0.05:
            return
        tx, ty = self._anticipation_target(organisms, dt)
        sm = 0.5 + self.genome.follow_smoothing
        self.pos[0] += (tx - self.pos[0]) * sm * dt * (0.3 + self.tracking * 0.5)
        self.pos[1] += (ty - self.pos[1]) * sm * dt * (0.3 + self.tracking * 0.5)

    def step(
        self,
        dt: float,
        audio: dict[str, float],
        scene_entropy: float,
        scene_motion: float,
        organisms: list[Any],
        births: list[tuple[float, float, float]],
    ) -> None:
        if not self.alive:
            return
        self.age += dt

        cut = self._should_cut(dt, audio, scene_entropy)
        if cut and self.shot_history:
            prev = self.shot_history[-1]
            px = prev.get("pos_x", 0.5)
            py = prev.get("pos_y", 0.5)
            if self.genome.whip_pan > 0.1:
                wp = self.genome.whip_pan * (1.0 + random.uniform(-0.3, 0.3))
                self._whip_phase = max(0.0, min(1.0, wp))
            elif self.genome.jump_cut > 0.1 and random.random() < 0.3:
                self.pos[0] = random.uniform(0.1, 0.9)
                self.pos[1] = random.uniform(0.1, 0.9)
            else:
                blend = self.genome.transition
                self.pos[0] = px * (1.0 - blend) + self.pos[0] * blend
                self.pos[1] = py * (1.0 - blend) + self.pos[1] * blend

        if self._whip_phase > 0.0:
            self._whip_phase = max(0.0, self._whip_phase - dt * 3.0)
            wp = self._whip_phase * self.genome.whip_pan
            self.pan += wp * random.uniform(-0.5, 0.5) * dt * 10.0
            self.tilt += wp * random.uniform(-0.3, 0.3) * dt * 8.0

        super().step(dt, audio, scene_entropy, scene_motion, organisms, births)

        self._follow_movement(organisms, dt)
        self._record_shot()

        if scene_entropy < self.genome.boredom_threshold and self.genome.restlessness > 0.1:
            restlessness_drive = (
                self.genome.boredom_threshold - scene_entropy
            ) * self.genome.restlessness
            self.orbit += random.uniform(-restlessness_drive, restlessness_drive) * dt * 2.0
            self.zoom += random.uniform(-restlessness_drive * 0.5, restlessness_drive * 0.5) * dt
            self.orbit = max(0.2, min(5.0, self.orbit))
            self.zoom = max(0.2, min(4.0, self.zoom))

        variety = 0.0
        if len(self.frame_signatures) >= 8:
            recent = list(self.frame_signatures)[-8:]
            unique = len(set(recent))
            variety = unique / 8.0
        variety_gain = variety * self.genome.variety_bonus * 0.15
        self.energy = max(0.0, min(4.0, self.energy + variety_gain * dt))

    def _style_bonus(self) -> float:
        """Reward shooting styles that stay visually varied and keep cutting.

        Shot-signature variety and recent cut activity are the observable
        products of the genome/brain, so this gives selection a gradient it
        can actually climb — unlike raw energy, which only reflected
        metabolism and rank.
        """
        variety = 0.0
        if len(self.frame_signatures) >= 8:
            recent = list(self.frame_signatures)[-8:]
            variety = len(set(recent)) / 8.0
        cut_rate = min(1.0, len(self.cut_log) / 16.0)
        engaged = 0.0 if self.boredom_timer > 0.5 else 0.1
        return min(0.4, variety * 0.25 + cut_rate * 0.10 + engaged)

    def clone_child(self) -> ExtendedCameraOrganism:
        child = super().clone_child()
        if isinstance(child, ExtendedCameraOrganism):
            child.species_id = self.species_id
            child.boredom_timer = 0.0
            child._hold_timer = 0.0
            child._whip_phase = 0.0
            child.cut_log = []
            child.frame_signatures = deque(maxlen=64)
        return child

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "species_id": self.species_id,
                "boredom_timer": self.boredom_timer,
                "last_cut_time": self.last_cut_time,
                "cut_log": self.cut_log[-16:],
                "_hold_timer": self._hold_timer,
                "_whip_phase": self._whip_phase,
            }
        )
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtendedCameraOrganism:
        obj = super().from_dict(d)
        obj.species_id = int(d.get("species_id", obj.species_id))
        obj.species = obj.species_id
        obj.boredom_timer = float(d.get("boredom_timer", 0.0))
        obj.last_cut_time = float(d.get("last_cut_time", 0.0))
        obj.cut_log = list(d.get("cut_log", []))
        obj._hold_timer = float(d.get("_hold_timer", 0.0))
        obj._whip_phase = float(d.get("_whip_phase", 0.0))
        return obj


# ============================================================================
# Extended camera population — speciation, species competition, extinction
# ============================================================================


class ExtendedCameraPopulation(CameraPopulation):
    """Camera population with speciation and species-level competition."""

    def __init__(self, size: int = 3):
        super().__init__(size=size)
        self.species: dict[int, CameraSpeciesDef] = {}
        self._next_sid = 1
        self._archive: dict[int, CameraSpeciesDef] = {}
        for cam in self.cameras:
            if cam.species_id == 0:
                cam.species_id = self._next_sid
                self._next_sid += 1
            self.species[cam.species_id] = CameraSpeciesDef(
                sid=cam.species_id,
                name=f"camera_species_{cam.species_id}",
                color=(random.random(), random.random(), random.random()),
                born_at=0.0,
            )

    def _maybe_speciate(self, parent: ExtendedCameraOrganism) -> int | None:
        if random.random() > 0.02 or len(self.species) >= 6:
            return None
        sid = self._next_sid
        self._next_sid += 1
        self.species[sid] = CameraSpeciesDef(
            sid=sid,
            name=f"camera_species_{sid}",
            color=(
                max(0.0, min(1.0, parent.genome.cinemat_mix + random.uniform(-0.2, 0.2))),
                max(0.0, min(1.0, parent.genome.cinemat_mix * 0.7 + random.uniform(-0.2, 0.2))),
                max(0.0, min(1.0, 1.0 - parent.genome.cinemat_mix + random.uniform(-0.2, 0.2))),
            ),
            born_at=parent.age,
            niche=random.choice(["observational", "chase", "abstract", "editorial"]),
        )
        return sid

    def best(self) -> ExtendedCameraOrganism:
        # Respect usurpation: during a coup the reigning challenger drives the
        # view even before its raw energy overtakes the old leader's.
        return self.dominant()

    def step(
        self,
        dt: float,
        audio: dict[str, float],
        scene_entropy: float,
        scene_motion: float,
        organisms: list[Any],
        births: list[tuple[float, float, float]],
    ) -> None:
        super().step(dt, audio, scene_entropy, scene_motion, organisms, births)

    def _on_child_born(self, child: CameraOrganism) -> None:
        sid = self._maybe_speciate(child)
        if sid is not None and isinstance(child, ExtendedCameraOrganism):
            child.species_id = sid
            child.species = sid

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["species"] = {
            sid: {
                "sid": sp.sid,
                "name": sp.name,
                "color": sp.color,
                "born_at": sp.born_at,
                "avg_fitness": sp.avg_fitness,
                "extinction_timer": sp.extinction_timer,
                "niche": sp.niche,
                "cut_style": sp.cut_style,
                "movement_style": sp.movement_style,
            }
            for sid, sp in self.species.items()
        }
        d["next_sid"] = self._next_sid
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtendedCameraPopulation:
        obj = cls(d.get("size", 3))
        obj.cameras = [ExtendedCameraOrganism.from_dict(c) for c in d.get("cameras", [])]
        obj._generation = int(d.get("generation", 0))
        obj._next_sid = int(d.get("next_sid", 1))
        sp_data = d.get("species", {})
        obj.species = {}
        for sid_s, sp_d in sp_data.items():
            sid = int(sid_s)
            obj.species[sid] = CameraSpeciesDef(
                sid=sid,
                name=sp_d.get("name", f"camera_species_{sid}"),
                color=tuple(sp_d.get("color", (0.5, 0.5, 0.5))),
                born_at=sp_d.get("born_at", 0.0),
                avg_fitness=sp_d.get("avg_fitness", 0.0),
                extinction_timer=sp_d.get("extinction_timer", 0.0),
                niche=sp_d.get("niche", "general"),
                cut_style=sp_d.get("cut_style", "hold"),
                movement_style=sp_d.get("movement_style", "smooth"),
            )
        return obj


# ============================================================================
# Backwards-compatible aliases
# ============================================================================

CameraGenome = ExtendedCameraGenome
CameraOrganism = ExtendedCameraOrganism
CameraPopulation = ExtendedCameraPopulation
