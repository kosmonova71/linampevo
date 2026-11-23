from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConceptTreeNode:
    """Tree node used as a concept-level intermediate representation.
    This is intentionally generic for Stage 8 incremental work.
    Later stages can map tree nodes into ShaderDNA operations.
    """

    op: str
    params: dict[str, Any] = field(default_factory=dict)
    children: tuple[ConceptTreeNode, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "params": dict(self.params),
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ConceptDNA:
    """Concept DNA (Stage 8) — currently used for novelty/archaeology.
    The initial integration keeps this simple and deterministic.
    """

    primitives: list[str]
    relationships: list[str]
    rules: list[str]
    behaviours: list[str]
    tree: ConceptTreeNode

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitives": list(self.primitives),
            "relationships": list(self.relationships),
            "rules": list(self.rules),
            "behaviours": list(self.behaviours),
            "tree": self.tree.to_dict(),
        }

    def stable_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def random_concept_dna(rng: random.Random | None = None) -> ConceptDNA:
    """Create a constrained random concept.
    In Option 1 integration this is mostly used as a novelty carrier.
    """
    rng = rng or random.Random()
    primitives = rng.sample(
        ["noise", "warp", "colour", "feedback", "flow", "reflect", "grow"], k=3
    )
    relationships = rng.sample(["repeat", "combine", "sequence", "nest", "mirror"], k=2)
    rules = rng.sample(["audio_modulate", "feedback_persist", "field_condition"], k=2)
    behaviours = rng.sample(
        ["organic_bloom", "spectral_memory", "gravity_bloom", "recursive_growth"],
        k=2,
    )

    def node(op: str, depth: int) -> ConceptTreeNode:
        if depth <= 0:
            return ConceptTreeNode(op=op, params={"seed": rng.random()})
        child_count = rng.choice([1, 2])
        children = tuple(
            node(op=rng.choice(["flow", "transform", "react", "grow"]), depth=depth - 1)
            for _ in range(child_count)
        )
        return ConceptTreeNode(op=op, params={"seed": rng.random()}, children=children)

    root = node(op=rng.choice(["repeat", "combine", "transform"]), depth=2)
    return ConceptDNA(
        primitives=primitives,
        relationships=relationships,
        rules=rules,
        behaviours=behaviours,
        tree=root,
    )
