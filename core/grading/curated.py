"""
Curated gate sets for the two hardest requirements: valid alternative methods
must get full credit, and an early slip must be deducted exactly once.

Cases live in ``tests/fixtures/phase4/curated_cases.json`` with the rubric in
``curated_rubric.json``. Controls (wrong answers) are included so a grader that
hands out full credit to everything cannot pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from core.grading.open_grader import OpenGradeResult
from core.grading.rubric import QuestionRubric, Rubric

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "phase4"


def load_curated(
    fixtures: Path = FIXTURES, cases_file: str = "curated_cases.json"
) -> tuple[Rubric, List[dict]]:
    rubric = Rubric.from_dict(json.loads((fixtures / "curated_rubric.json").read_text(encoding="utf-8")))
    cases = json.loads((fixtures / cases_file).read_text(encoding="utf-8"))["cases"]
    return rubric, cases


@dataclass
class CaseOutcome:
    case_id: str
    set: str
    passed: bool
    score: float
    max_points: float
    best_path: Optional[str]
    deductions: List[str]
    needs_review: bool
    detail: str = ""


@dataclass
class CuratedReport:
    outcomes: List[CaseOutcome] = field(default_factory=list)
    cost_usd: float = 0.0

    def rate(self, set_name: str) -> Optional[float]:
        rows = [o for o in self.outcomes if o.set == set_name]
        return sum(o.passed for o in rows) / len(rows) if rows else None


def evaluate_case(case: dict, q: QuestionRubric, result: OpenGradeResult) -> CaseOutcome:
    best = next((p for p in result.path_scores if p.path_id == result.best_path), None)
    deductions = best.deductions if best else []
    full = abs(result.score - q.max_points) < 1e-9
    kind = case["set"]
    if kind == "alternative":
        passed = full and (case.get("declared", True) or result.needs_review)
        detail = "" if passed else (
            "undeclared method not routed to review" if full else f"score {result.score}/{q.max_points}"
        )
    elif kind == "consequential":
        passed = deductions == [case["slip"]] and abs(result.score - case["expected_score"]) < 1e-9
        detail = "" if passed else f"deductions {deductions}, expected [{case['slip']}]"
    else:
        passed = not full
        detail = "" if passed else "wrong answer received full credit"
    return CaseOutcome(case["id"], kind, passed, result.score, q.max_points, result.best_path,
                       deductions, result.needs_review, detail)


def run_curated(grade_fn: Callable[[QuestionRubric, str], OpenGradeResult],
                fixtures: Path = FIXTURES, cases_file: str = "curated_cases.json") -> CuratedReport:
    rubric, cases = load_curated(fixtures, cases_file)
    report = CuratedReport()
    for case in cases:
        q = rubric.question(case["question_id"])
        result = grade_fn(q, case["transcript"])
        report.cost_usd += result.cost_usd
        report.outcomes.append(evaluate_case(case, q, result))
    return report
