"""evo_bio.cultural_evolution

Spec 48 - Cultural Evolution.

Ideas spread independently of DNA. A species discovers "spiral behaviour";
nearby species copy and *improve* it. Evolution happens much faster than mutation
alone. This is distinct from horizontal gene transfer (genetic): here the unit of
transmission is a *behaviour / meme* that is learned, then refined.

Mechanisms:
  - copy:     organism replicates a meme from another species
  - imitate:  organism copies a successful technique and refines it
  - steal:    organism takes a meme from another species without permission
  - modify:   organism alters a meme it already knows
  - teach:    organism actively transmits a meme to another organism

Convergence is much faster than genetic mutation alone because successful
techniques propagate horizontally across the population.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Meme:
    """A cultural behaviour that can be copied and improved."""

    name: str
    fidelity: float = 0.9  # how faithfully it is copied
    innovation: float = 0.1  # chance a copy is improved
    owner: str = ""
    strength: float = 0.5  # effectiveness of the technique (0..1)
    generation: int = 0  # how many times the meme has been copied
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fidelity": self.fidelity,
            "innovation": self.innovation,
            "owner": self.owner,
            "strength": self.strength,
            "generation": self.generation,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Meme:
        return cls(
            d["name"],
            float(d.get("fidelity", 0.9)),
            float(d.get("innovation", 0.1)),
            d.get("owner", ""),
            strength=float(d.get("strength", 0.5)),
            generation=int(d.get("generation", 0)),
            tags=d.get("tags", []),
        )


class Culture:
    """Tracks memes and transmits them between species with possible improvement.

    Supports five cultural transmission mechanisms:
      copy, imitate, steal, modify, teach
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        # species -> list of memes it knows
        self.known: dict[str, list[Meme]] = {}
        # track convergence speed: generations to reach fitness threshold
        self.convergence_history: list[float] = []
        self._step_count = 0

    # ------------------------------------------------------------------
    # Core discovery
    # ------------------------------------------------------------------

    def discover(self, species: str, meme: str, strength: float = 0.5) -> Meme:
        m = Meme(name=meme, owner=species, strength=strength)
        self.known.setdefault(species, []).append(m)
        return m

    # ------------------------------------------------------------------
    # Mechanism 1: copy
    # ------------------------------------------------------------------

    def copy(self, source: str, target: str) -> Meme | None:
        """Copy a meme from source to target with fidelity gate.

        The copy is faithful with probability equal to the meme's fidelity.
        If the copy fails the fidelity gate, the meme is garbled and may
        be lost entirely.
        """
        src_memes = self.known.get(source, [])
        if not src_memes:
            return None
        parent = self.rng.choice(src_memes)
        faithful = self.rng.random() < parent.fidelity
        if not faithful:
            return None
        child = Meme(
            name=parent.name,
            owner=target,
            fidelity=parent.fidelity,
            innovation=parent.innovation,
            strength=parent.strength,
            generation=parent.generation + 1,
            tags=list(parent.tags),
        )
        self.known.setdefault(target, []).append(child)
        return child

    # ------------------------------------------------------------------
    # Mechanism 2: imitate
    # ------------------------------------------------------------------

    def imitate(self, source: str, target: str) -> Meme | None:
        """Imitate a successful technique from source and refine it.

        Imitation is like copy but with an additional improvement step:
        the imitator refines the copied meme, increasing its strength
        and potentially its fidelity and innovation.
        """
        child = self.copy(source, target)
        if child is None:
            return None
        # Refine the imitation
        child.strength = min(1.0, child.strength + 0.1)
        if self.rng.random() < child.innovation:
            child.fidelity = min(1.0, child.fidelity + 0.05)
            child.innovation = min(1.0, child.innovation + 0.05)
        return child

    # ------------------------------------------------------------------
    # Mechanism 3: steal
    # ------------------------------------------------------------------

    def steal(self, source: str, target: str) -> Meme | None:
        """Steal a meme from source to target without permission.

        Stealing bypasses the fidelity gate but the stolen meme starts
        with reduced strength because the thief doesn't fully understand it.
        """
        src_memes = self.known.get(source, [])
        if not src_memes:
            return None
        parent = self.rng.choice(src_memes)
        # Stealing bypasses fidelity but reduces strength
        child = Meme(
            name=parent.name,
            owner=target,
            fidelity=max(0.3, parent.fidelity - 0.2),
            innovation=parent.innovation,
            strength=parent.strength * 0.6,
            generation=parent.generation + 1,
            tags=list(parent.tags),
        )
        self.known.setdefault(target, []).append(child)
        return child

    # ------------------------------------------------------------------
    # Mechanism 4: modify
    # ------------------------------------------------------------------

    def modify(self, species: str, meme_name: str, strength_delta: float = 0.1) -> Meme | None:
        """Modify a meme the species already knows.

        Randomly perturbs the meme's strength, fidelity, and innovation
        to explore variations on an existing technique.
        """
        memes = self.known.get(species, [])
        if not memes:
            return None
        candidates = [m for m in memes if m.name == meme_name]
        if not candidates:
            return None
        meme = self.rng.choice(candidates)
        meme.strength = max(
            0.0, min(1.0, meme.strength + strength_delta * self.rng.uniform(-1, 1))
        )
        meme.fidelity = max(0.3, min(1.0, meme.fidelity + 0.05 * self.rng.uniform(-1, 1)))
        meme.innovation = max(0.0, min(1.0, meme.innovation + 0.05 * self.rng.uniform(-1, 1)))
        return meme

    # ------------------------------------------------------------------
    # Mechanism 5: teach
    # ------------------------------------------------------------------

    def teach(self, source: str, target: str, meme_name: str | None = None) -> Meme | None:
        """Actively teach a meme from source to target.

        Teaching is the most reliable transmission mechanism: the teacher
        deliberately selects and transmits a meme with high fidelity,
        bypassing the random fidelity gate. The learner gains full
        understanding of the technique.
        """
        src_memes = self.known.get(source, [])
        if not src_memes:
            return None
        if meme_name is not None:
            candidates = [m for m in src_memes if m.name == meme_name]
            if not candidates:
                return None
            parent = self.rng.choice(candidates)
        else:
            parent = self.rng.choice(src_memes)
        # Teaching bypasses fidelity gate - full transmission
        child = Meme(
            name=parent.name,
            owner=target,
            fidelity=parent.fidelity,
            innovation=parent.innovation,
            strength=parent.strength,
            generation=parent.generation + 1,
            tags=list(parent.tags),
        )
        self.known.setdefault(target, []).append(child)
        return child

    # ------------------------------------------------------------------
    # Population-level spread
    # ------------------------------------------------------------------

    def transmit(self, source: str, target: str) -> Meme | None:
        """Copy a meme from source to target with possible improvement.

        This is the default transmission that combines imitation with
        a chance of innovation, preserving backward compatibility.
        """
        src_memes = self.known.get(source, [])
        if not src_memes:
            return None
        parent = self.rng.choice(src_memes)
        faithful = self.rng.random() < parent.fidelity
        child = Meme(
            name=parent.name,
            owner=target,
            fidelity=parent.fidelity if faithful else max(0.3, parent.fidelity - 0.3),
            innovation=parent.innovation,
            strength=parent.strength,
            generation=parent.generation + 1,
            tags=list(parent.tags),
        )
        if self.rng.random() < parent.innovation:
            child.fidelity = min(1.0, child.fidelity + 0.1)
            child.innovation = min(1.0, child.innovation + 0.1)
            child.strength = min(1.0, child.strength + 0.05)
        self.known.setdefault(target, []).append(child)
        return child

    def spread(self, species_order: list[str]) -> int:
        """Each species tries to learn from its neighbour; returns # transmissions."""
        transfers = 0
        for i in range(len(species_order)):
            src = species_order[i]
            tgt = species_order[(i + 1) % len(species_order)]
            if self.transmit(src, tgt) is not None:
                transfers += 1
        return transfers

    def memes_of(self, species: str) -> list[Meme]:
        return list(self.known.get(species, []))

    def species_with_meme(self, meme_name: str) -> list[str]:
        """Return all species that know a meme with the given name."""
        return [sp for sp, memes in self.known.items() if any(m.name == meme_name for m in memes)]

    def best_meme(self, species: str) -> Meme | None:
        """Return the strongest meme known by a species."""
        memes = self.known.get(species, [])
        if not memes:
            return None
        return max(memes, key=lambda m: m.strength)

    def convergence_speed(self, generations: int = 10) -> float:
        """Estimate how quickly cultural evolution converges.

        Returns the average number of new memes acquired per generation
        over the last ``generations`` steps. Higher means faster convergence.
        """
        if len(self.convergence_history) < 2:
            return 0.0
        window = self.convergence_history[-generations:]
        return sum(window) / len(window)

    def step(
        self,
        species_order: list[str],
        copy_rate: float = 0.3,
        imitate_rate: float = 0.2,
        steal_rate: float = 0.05,
        modify_rate: float = 0.15,
        teach_rate: float = 0.1,
    ) -> int:
        """Advance cultural evolution one step across all species.

        Each species attempts cultural transmission using the configured
        rates for each mechanism. Returns the total number of transmissions.
        """
        self._step_count += 1
        total_transmissions = 0
        new_memes_count = 0

        for i, species in enumerate(species_order):
            if species not in self.known or not self.known[species]:
                continue

            # Determine target species (next in order, wrapping)
            target = species_order[(i + 1) % len(species_order)]

            # Copy: baseline horizontal transmission
            if self.rng.random() < copy_rate:
                result = self.copy(species, target)
                if result is not None:
                    total_transmissions += 1

            # Imitate: copy + refine (higher success rate for good memes)
            if self.rng.random() < imitate_rate:
                result = self.imitate(species, target)
                if result is not None:
                    total_transmissions += 1
                    new_memes_count += 1

            # Steal: take without permission (risky but fast)
            if self.rng.random() < steal_rate:
                result = self.steal(species, target)
                if result is not None:
                    total_transmissions += 1

            # Modify: alter existing memes (drives local innovation)
            if self.rng.random() < modify_rate:
                memes = self.known.get(species, [])
                if memes:
                    meme = self.rng.choice(memes)
                    self.modify(species, meme.name, strength_delta=0.05 * self.rng.uniform(-1, 1))

            # Teach: active, reliable transmission
            if self.rng.random() < teach_rate:
                result = self.teach(species, target)
                if result is not None:
                    total_transmissions += 1
                    new_memes_count += 1

        # Record convergence metric
        if self._step_count % 10 == 0:
            self.convergence_history.append(new_memes_count / max(1, len(species_order)))

        return total_transmissions

    def to_dict(self) -> dict:
        return {
            "known": {sp: [m.to_dict() for m in memes] for sp, memes in self.known.items()},
            "convergence_history": self.convergence_history,
            "step_count": self._step_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Culture:
        c = cls()
        for sp, memes in d.get("known", {}).items():
            c.known[sp] = [Meme.from_dict(m) for m in memes]
        c.convergence_history = d.get("convergence_history", [])
        c._step_count = d.get("step_count", 0)
        return c
