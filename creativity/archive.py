from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass
class NoveltyResult:
    novelty: float
    max_similarity: float
    nearest_id: str | None


class ConceptArchaeology:
    """Visual archaeology / archive.
    Option 1 integration: we store/retrieve novelty carrier strings in the
    existing SQLite DB when possible.
    Similarity is computed over a stable concept id string (if present).
    For now we use a simple scheme:
      - identical id => similarity 1
      - else => similarity decreases with overlap of components in dict
    This keeps novelty bounded even without image readback.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    concept_id TEXT NOT NULL,
                    concept_json TEXT NOT NULL,
                    score REAL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_concept_archive_concept_id
                ON concept_archive(concept_id);
                """
            )

    @staticmethod
    def _concept_to_json(concept: Any) -> dict[str, Any]:
        if concept is None:
            return {}
        if hasattr(concept, "to_dict"):
            return concept.to_dict()
        if isinstance(concept, dict):
            return concept
        return {"raw": str(concept)}

    def record(self, concept: Any, *, concept_id: str, score: float | None = None) -> None:
        payload = self._concept_to_json(concept)
        concept_json = json.dumps(payload, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO concept_archive (ts, concept_id, concept_json, score)
                VALUES (strftime('%s','now'), ?, ?, ?)
                """,
                (str(concept_id), concept_json, float(score) if score is not None else None),
            )

    @staticmethod
    def _similarity_dict(a: dict[str, Any], b: dict[str, Any]) -> float:
        keys = ["primitives", "relationships", "rules", "behaviours"]
        sims: list[float] = []
        for k in keys:
            av = a.get(k, [])
            bv = b.get(k, [])
            if not isinstance(av, list) or not isinstance(bv, list):
                continue
            aset = set(map(str, av))
            bset = set(map(str, bv))
            if not aset and not bset:
                sims.append(1.0)
                continue
            inter = len(aset & bset)
            union = max(1, len(aset | bset))
            sims.append(inter / union)
        return float(sum(sims) / max(1, len(sims)))

    def novelty(self, concept: Any, *, concept_id: str | None = None) -> NoveltyResult:
        concept_json = self._concept_to_json(concept)
        best_sim = 0.0
        best_id: str | None = None
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT concept_id, concept_json
                FROM concept_archive
                ORDER BY score DESC NULLS LAST, ts DESC
                LIMIT 50
                """
            ).fetchall()
        if not rows:
            return NoveltyResult(novelty=1.0, max_similarity=0.0, nearest_id=None)
        for cid, cjson in rows:
            if cid and concept_id and cid == concept_id:
                sim = 1.0
            else:
                try:
                    other = json.loads(cjson)
                except json.JSONDecodeError:
                    other = {}
                sim = self._similarity_dict(concept_json, other)
            if sim > best_sim:
                best_sim = float(sim)
                best_id = cid
        novelty = max(0.0, min(1.0, 1.0 - best_sim))
        return NoveltyResult(novelty=novelty, max_similarity=best_sim, nearest_id=best_id)
