"""
SQLite persistence for rubrics and grades.

Rubrics are versioned and move ``proposed → approved``; approving a new
version supersedes the previous one. Grading refuses to run against anything
but an approved rubric. Every grade stores what is needed to reconstruct it
during an appeal: rubric version + per-question hash, model, provider model,
prompt version, temperature and the raw model output.

A grade is *stale* when the approved rubric's hash for its question no longer
matches the hash it was graded under; ``stale_grades`` lists those for re-grade.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.grading.rubric import Rubric

DB_PATH = Path(os.getenv("STORAGE_ROOT", "storage")) / "grading.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rubrics (
    rubric_id    TEXT NOT NULL,
    version      INTEGER NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('proposed','approved','superseded','rejected')),
    source       TEXT,
    created_at   TEXT NOT NULL,
    approved_at  TEXT,
    approved_by  TEXT,
    PRIMARY KEY (rubric_id, version)
);
CREATE TABLE IF NOT EXISTS grades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    booklet_id      TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    score           REAL,
    max_score       REAL NOT NULL,
    rubric_id       TEXT NOT NULL,
    rubric_version  INTEGER NOT NULL,
    question_hash   TEXT NOT NULL,
    model           TEXT,
    provider_model  TEXT,
    prompt_version  TEXT,
    temperature     REAL,
    raw_response    TEXT,
    evidence        TEXT,
    needs_review    INTEGER NOT NULL DEFAULT 0,
    review_reasons  TEXT,
    status          TEXT NOT NULL DEFAULT 'current' CHECK (status IN ('current','superseded')),
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_grades_current ON grades (booklet_id, question_id, status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class GradeRecord:
    booklet_id: str
    question_id: str
    score: Optional[float]
    max_score: float
    rubric_id: str
    rubric_version: int
    question_hash: str
    model: str = ""
    provider_model: str = ""
    prompt_version: str = ""
    temperature: float = 0.0
    raw_response: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    created_at: str = ""


class GradingStore:
    def __init__(self, path: str | Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- rubrics -----------------------------------------------------------

    def propose_rubric(self, rubric: Rubric) -> int:
        """Store a rubric as proposed; identical content returns the existing version."""
        h = rubric.content_hash()
        row = self.conn.execute(
            "SELECT version FROM rubrics WHERE rubric_id=? AND content_hash=? "
            "AND status IN ('proposed','approved') ORDER BY version DESC LIMIT 1",
            (rubric.rubric_id, h),
        ).fetchone()
        if row:
            return int(row["version"])
        latest = self.conn.execute(
            "SELECT MAX(version) AS v FROM rubrics WHERE rubric_id=?", (rubric.rubric_id,)
        ).fetchone()["v"]
        version = int(latest or 0) + 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO rubrics (rubric_id, version, content, content_hash, status, source, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (rubric.rubric_id, version, json.dumps(rubric.to_dict(), ensure_ascii=False),
                 h, "proposed", rubric.source, _now()),
            )
        return version

    def approve_rubric(self, rubric_id: str, version: int, approved_by: str) -> None:
        row = self.conn.execute(
            "SELECT content FROM rubrics WHERE rubric_id=? AND version=?", (rubric_id, version)
        ).fetchone()
        if row is None:
            raise KeyError(f"no rubric {rubric_id} v{version}")
        errors = Rubric.from_dict(json.loads(row["content"])).validate()
        if errors:
            raise ValueError("rubric does not validate: " + "; ".join(errors))
        with self.conn:
            self.conn.execute(
                "UPDATE rubrics SET status='superseded' WHERE rubric_id=? AND status='approved'",
                (rubric_id,),
            )
            self.conn.execute(
                "UPDATE rubrics SET status='approved', approved_at=?, approved_by=? "
                "WHERE rubric_id=? AND version=?",
                (_now(), approved_by, rubric_id, version),
            )

    def approved_rubric(self, rubric_id: str) -> Optional[tuple[Rubric, int]]:
        row = self.conn.execute(
            "SELECT content, version FROM rubrics WHERE rubric_id=? AND status='approved'",
            (rubric_id,),
        ).fetchone()
        if row is None:
            return None
        return Rubric.from_dict(json.loads(row["content"])), int(row["version"])

    def rubric_versions(self, rubric_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT version, content_hash, status, source, created_at, approved_at, approved_by "
            "FROM rubrics WHERE rubric_id=? ORDER BY version", (rubric_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_rubric(self, rubric_id: str, version: int) -> Optional[Rubric]:
        row = self.conn.execute(
            "SELECT content FROM rubrics WHERE rubric_id=? AND version=?", (rubric_id, version)
        ).fetchone()
        return Rubric.from_dict(json.loads(row["content"])) if row else None

    # -- grades ------------------------------------------------------------

    def record_grade(self, g: GradeRecord) -> int:
        """Insert a grade; any previous current grade for the same question is superseded."""
        with self.conn:
            self.conn.execute(
                "UPDATE grades SET status='superseded' WHERE booklet_id=? AND question_id=? "
                "AND rubric_id=? AND status='current'",
                (g.booklet_id, g.question_id, g.rubric_id),
            )
            cur = self.conn.execute(
                "INSERT INTO grades (booklet_id, question_id, score, max_score, rubric_id, rubric_version, "
                "question_hash, model, provider_model, prompt_version, temperature, raw_response, evidence, "
                "needs_review, review_reasons, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (g.booklet_id, g.question_id, g.score, g.max_score, g.rubric_id, g.rubric_version,
                 g.question_hash, g.model, g.provider_model, g.prompt_version, g.temperature,
                 g.raw_response, json.dumps(g.evidence, ensure_ascii=False), int(g.needs_review),
                 json.dumps(g.review_reasons, ensure_ascii=False), "current", g.created_at or _now()),
            )
        return int(cur.lastrowid)

    def current_grades(self, rubric_id: str, booklet_id: Optional[str] = None) -> List[GradeRecord]:
        sql = "SELECT * FROM grades WHERE rubric_id=? AND status='current'"
        args: list = [rubric_id]
        if booklet_id is not None:
            sql += " AND booklet_id=?"
            args.append(booklet_id)
        rows = self.conn.execute(sql + " ORDER BY booklet_id, CAST(question_id AS INTEGER)", args).fetchall()
        return [self._row_to_grade(r) for r in rows]

    def stale_grades(self, rubric_id: str) -> List[GradeRecord]:
        """Current grades whose question rubric changed since they were produced."""
        approved = self.approved_rubric(rubric_id)
        if approved is None:
            return []
        rubric, _ = approved
        return [
            g for g in self.current_grades(rubric_id)
            if g.question_hash != rubric.question_hash(g.question_id)
        ]

    @staticmethod
    def _row_to_grade(r: sqlite3.Row) -> GradeRecord:
        return GradeRecord(
            booklet_id=r["booklet_id"], question_id=r["question_id"], score=r["score"],
            max_score=r["max_score"], rubric_id=r["rubric_id"], rubric_version=r["rubric_version"],
            question_hash=r["question_hash"], model=r["model"] or "", provider_model=r["provider_model"] or "",
            prompt_version=r["prompt_version"] or "", temperature=r["temperature"] or 0.0,
            raw_response=r["raw_response"] or "", evidence=json.loads(r["evidence"] or "{}"),
            needs_review=bool(r["needs_review"]), review_reasons=json.loads(r["review_reasons"] or "[]"),
            created_at=r["created_at"],
        )
