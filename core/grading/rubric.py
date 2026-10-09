"""
Rubric model: milestones with point weights, alternative solution paths, and
multiple-choice keys per exam version.

A rubric is data, not code. It is *proposed* (by extraction or an LLM) and
becomes the grading standard only after the instructor approves it in the
store (``core.grading.store``). Any edit produces a new content hash, which is
how stale grades are detected and re-graded.

Milestone expressions reference earlier milestones by id, e.g. ``expected =
"m1 * 2"`` with ``depends_on = ["m1"]``. The consequential-error engine
re-evaluates such expressions with the student's own earlier values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from core.grading.equivalence import evaluate_with, to_sympy

logger = logging.getLogger(__name__)

RUBRIC_SCHEMA_VERSION = "rubric_v1"
_MILESTONE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class Milestone:
    id: str
    description: str
    points: float
    expected: Optional[str] = None          # value/expression; None → qualitative step
    depends_on: List[str] = field(default_factory=list)
    abs_tol: Optional[float] = None         # instructor-set tolerance, e.g. 0.5 for "nearest integer"

    @property
    def is_value(self) -> bool:
        return self.expected is not None


@dataclass
class SolutionPath:
    path_id: str
    description: str
    milestones: List[Milestone]
    source: str = "instructor"              # instructor | proposed

    def total(self) -> float:
        return sum(m.points for m in self.milestones)

    def resolve_expected(self) -> Dict[str, float]:
        """Correct numeric value of every value milestone, evaluated in order."""
        values: Dict[str, float] = {}
        for m in self.milestones:
            if not m.is_value:
                continue
            v = evaluate_with(m.expected, {d: values[d] for d in m.depends_on if d in values})
            if v is not None:
                values[m.id] = v
        return values


@dataclass
class QuestionRubric:
    question_id: str
    max_points: float
    kind: str = "open"                      # open | mcq
    question_text: str = ""
    paths: List[SolutionPath] = field(default_factory=list)
    # mcq only: exam version → correct letter, and the master option texts
    mcq_keys: Dict[str, str] = field(default_factory=dict)
    mcq_options: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.kind == "mcq":
            if not self.mcq_keys:
                errors.append(f"Q{self.question_id}: no MCQ keys")
            return errors
        if not self.paths:
            errors.append(f"Q{self.question_id}: open question without solution paths")
        for p in self.paths:
            if abs(p.total() - self.max_points) > 1e-6:
                errors.append(
                    f"Q{self.question_id}/{p.path_id}: milestones sum to {p.total()}, "
                    f"max is {self.max_points}"
                )
            seen: List[str] = []
            for m in p.milestones:
                if not _MILESTONE_ID_RE.match(m.id):
                    errors.append(f"Q{self.question_id}/{p.path_id}: bad milestone id {m.id!r}")
                for d in m.depends_on:
                    if d not in seen:
                        errors.append(
                            f"Q{self.question_id}/{p.path_id}/{m.id}: depends on {d!r}, "
                            "which is not an earlier milestone"
                        )
                if m.is_value:
                    expr = to_sympy(m.expected)
                    if expr is None:
                        errors.append(f"Q{self.question_id}/{p.path_id}/{m.id}: cannot parse {m.expected!r}")
                    else:
                        free = {s.name for s in expr.free_symbols}
                        if not free <= set(m.depends_on):
                            errors.append(
                                f"Q{self.question_id}/{p.path_id}/{m.id}: expression uses "
                                f"{sorted(free - set(m.depends_on))} not listed in depends_on"
                            )
                seen.append(m.id)
            resolved = p.resolve_expected()
            for m in p.milestones:
                if m.is_value and m.id not in resolved:
                    errors.append(f"Q{self.question_id}/{p.path_id}/{m.id}: expected value does not evaluate")
        return errors


@dataclass
class Rubric:
    rubric_id: str
    exam_id: str
    questions: List[QuestionRubric]
    source: str = ""
    schema_version: str = RUBRIC_SCHEMA_VERSION

    def question(self, question_id: str) -> Optional[QuestionRubric]:
        return next((q for q in self.questions if q.question_id == question_id), None)

    def validate(self) -> List[str]:
        return [e for q in self.questions for e in q.validate()]

    def to_dict(self) -> dict:
        return asdict(self)

    def content_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def question_hash(self, question_id: str) -> str:
        q = self.question(question_id)
        canonical = json.dumps(asdict(q) if q else None, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "Rubric":
        questions = []
        for q in d.get("questions", []):
            paths = [
                SolutionPath(
                    path_id=p["path_id"],
                    description=p.get("description", ""),
                    milestones=[Milestone(**m) for m in p.get("milestones", [])],
                    source=p.get("source", "instructor"),
                )
                for p in q.get("paths", [])
            ]
            questions.append(QuestionRubric(
                question_id=str(q["question_id"]),
                max_points=float(q["max_points"]),
                kind=q.get("kind", "open"),
                question_text=q.get("question_text", ""),
                paths=paths,
                mcq_keys=dict(q.get("mcq_keys", {})),
                mcq_options=dict(q.get("mcq_options", {})),
                notes=q.get("notes", ""),
            ))
        return cls(
            rubric_id=d["rubric_id"],
            exam_id=d.get("exam_id", ""),
            questions=questions,
            source=d.get("source", ""),
            schema_version=d.get("schema_version", RUBRIC_SCHEMA_VERSION),
        )


def mcq_rubric(
    rubric_id: str,
    exam_id: str,
    master,                                 # McqAnswerKey
    version_letters: Dict[str, Dict[str, str]],   # version → question id → letter
    *,
    points_per_question: float = 10.0,
    source: str = "",
) -> Rubric:
    """Build an all-MCQ rubric from the master key and derived per-version keys."""
    questions = []
    for mq in master.questions:
        keys = {"master": mq.correct_letter or ""}
        keys.update({v: letters[mq.question_id] for v, letters in version_letters.items()
                     if mq.question_id in letters})
        questions.append(QuestionRubric(
            question_id=mq.question_id,
            max_points=points_per_question,
            kind="mcq",
            mcq_keys=keys,
            mcq_options=dict(mq.options),
            notes=f"correct option value: {mq.correct_text}",
        ))
    return Rubric(rubric_id=rubric_id, exam_id=exam_id, questions=questions, source=source)


# ---------------------------------------------------------------------------
# LLM rubric proposal (open questions)
# ---------------------------------------------------------------------------

PROPOSE_PROMPT_VERSION = "rubric_propose_v1"
PROPOSE_PROMPT = """\
You are helping an instructor turn a worked solution into a grading rubric.

Question (id {question_id}, worth {max_points} points):
{question_text}

Instructor's solution and scoring notes:
{solution_text}

Split the solution into ordered milestones whose points sum to exactly
{max_points}. For a milestone that produces a quantity, give "expected" as a
plain math expression (e.g. "0.6*0.95", "m1/(m1+m2)") that may reference
EARLIER milestone ids listed in "depends_on"; this is how follow-through
credit is computed. For a purely qualitative step, use "expected": null.
Milestone ids are lowercase like m1, m2. If the solution mentions an
alternative method, add it as a second path.

Return ONLY JSON:
{{"paths": [{{"path_id": "main", "description": "...", "milestones": [
  {{"id": "m1", "description": "...", "points": 2, "expected": "0.6*0.95", "depends_on": []}}
]}}]}}
"""


def propose_open_question(
    question_id: str,
    question_text: str,
    solution_text: str,
    max_points: float,
    *,
    model: Optional[str] = None,
):
    """LLM proposal of milestones; returns (QuestionRubric, raw_text, errors). Needs approval."""
    from core.vlm import generate_text, parse_json_response

    resp = generate_text(
        PROPOSE_PROMPT.format(
            question_id=question_id, max_points=max_points,
            question_text=question_text, solution_text=solution_text,
        ),
        model=model, json_mode=True, purpose="rubric_propose",
    )
    try:
        payload = parse_json_response(resp.text)
        paths = [
            SolutionPath(
                path_id=str(p.get("path_id", f"path{i+1}")),
                description=str(p.get("description", "")),
                milestones=[
                    Milestone(
                        id=str(m["id"]), description=str(m.get("description", "")),
                        points=float(m.get("points", 0)), expected=m.get("expected"),
                        depends_on=[str(x) for x in m.get("depends_on") or []],
                    )
                    for m in p.get("milestones", [])
                ],
                source="proposed",
            )
            for i, p in enumerate(payload.get("paths", []))
        ]
    except Exception as exc:
        return None, resp.text, [f"unparseable proposal: {exc}"]
    q = QuestionRubric(question_id=question_id, max_points=max_points,
                       question_text=question_text, paths=paths)
    return q, resp.text, q.validate()
