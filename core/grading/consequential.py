"""
Deterministic milestone scoring with consequential error ("ציון נגרר").

The LLM only *reports* what the student wrote for each milestone; points are
decided here, by arithmetic, so the same extraction always yields the same
grade:

    correct        student value equals the rubric's value          → full points
    consequential  wrong overall, but equals the rubric expression
                   re-evaluated with the student's own earlier values → full points
    wrong          neither                                          → 0
    missing        no value reported                                 → 0

The originating slip loses its milestone's points exactly once; every
downstream step that is correct relative to the slip keeps full credit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.grading.equivalence import evaluate_with, to_sympy
from core.grading.rubric import SolutionPath

# Students round. A value matches when it equals the target to the precision
# the student wrote (half a unit in the last written decimal) and is within
# ROUNDING_REL_TOL relatively, so "0.43" matches 0.42727 but "0.4" does not.
ROUNDING_REL_TOL: float = 0.01
EXACT_REL_TOL: float = 1e-6
_DECIMALS_RE = re.compile(r"\d\.(\d+)")


@dataclass
class MilestoneOutcome:
    milestone_id: str
    status: str                 # correct | consequential | wrong | missing | step_ok | step_missing
    awarded: float
    max_points: float
    student_value: Optional[str] = None
    expected_value: Optional[float] = None
    carried_value: Optional[float] = None
    detail: str = ""


@dataclass
class PathScore:
    path_id: str
    score: float
    max_points: float
    outcomes: List[MilestoneOutcome] = field(default_factory=list)

    @property
    def deductions(self) -> List[str]:
        return [o.milestone_id for o in self.outcomes if o.awarded < o.max_points]

    def to_dict(self) -> dict:
        return {
            "path_id": self.path_id,
            "score": self.score,
            "max_points": self.max_points,
            "outcomes": [o.__dict__ for o in self.outcomes],
        }


_RESULT_SPLIT_RE = re.compile(r"\\approx|≈|=|\\simeq|→|⇒|\\to\b|\\rightarrow|\\Rightarrow|->")


def _result_text(text: Optional[str]) -> str:
    parts = [p.strip() for p in _RESULT_SPLIT_RE.split(str(text or "")) if p.strip()]
    return parts[-1] if parts else ""


_LAST_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?%?")


def student_number(text: Optional[str]) -> Optional[float]:
    """Numeric value of what the student wrote; for "0.4·0.95=0.38" that is the final result."""
    result = _result_text(text)
    if not result:
        return None
    expr = to_sympy(result)
    if expr is None:
        # Trailing words ("270.7, כלומר 271"): the last number written is the result.
        numbers = _LAST_NUMBER_RE.findall(result)
        expr = to_sympy(numbers[-1]) if numbers else None
    if expr is None or expr.free_symbols:
        return None
    try:
        v = complex(expr.evalf())
    except (TypeError, ValueError):
        return None
    return v.real if abs(v.imag) < 1e-12 else None


def value_matches(
    student_text: str, student: float, target: float, abs_tol: Optional[float] = None
) -> bool:
    """Exact match, within the rubric's tolerance, or a faithful rounding at the student's precision."""
    scale = max(abs(student), abs(target), 1e-12)
    if abs(student - target) <= EXACT_REL_TOL * scale:
        return True
    if abs_tol is not None and abs(student - target) <= abs_tol:
        return True
    tail = _result_text(student_text)
    numbers = _LAST_NUMBER_RE.findall(tail)
    m = _DECIMALS_RE.search(numbers[-1] if numbers and to_sympy(tail) is None else tail)
    if not m:
        return False
    half_unit = 0.5 * 10 ** (-len(m.group(1)))
    return abs(student - target) <= half_unit + 1e-12 and abs(student - target) <= ROUNDING_REL_TOL * scale


def score_path(
    path: SolutionPath,
    student_values: Dict[str, Optional[str]],
    step_present: Dict[str, bool],
) -> PathScore:
    correct = path.resolve_expected()
    effective: Dict[str, float] = {}       # value the student carried forward
    outcomes: List[MilestoneOutcome] = []

    for m in path.milestones:
        if not m.is_value:
            ok = bool(step_present.get(m.id))
            outcomes.append(MilestoneOutcome(
                m.id, "step_ok" if ok else "step_missing", m.points if ok else 0.0, m.points,
            ))
            continue

        target = correct.get(m.id)
        raw = student_values.get(m.id)
        sv = student_number(raw)
        out = MilestoneOutcome(m.id, "missing", 0.0, m.points, student_value=raw, expected_value=target)
        if sv is None or target is None:
            if raw:
                out.detail = "student value not numeric"
            if target is not None:
                effective[m.id] = target     # a skipped write-down does not poison later steps
            outcomes.append(out)
            continue

        effective[m.id] = sv
        if value_matches(raw, sv, target, m.abs_tol):
            out.status, out.awarded = "correct", m.points
        elif m.depends_on:
            carried = evaluate_with(m.expected, {d: effective.get(d, correct.get(d)) for d in m.depends_on})
            out.carried_value = carried
            upstream_wrong = any(
                d in effective and d in correct and not _close(effective[d], correct[d])
                for d in m.depends_on
            )
            if carried is not None and value_matches(raw, sv, carried, m.abs_tol):
                out.awarded = m.points
                if upstream_wrong:
                    out.status = "consequential"
                    out.detail = "correct follow-through from the student's earlier value"
                else:
                    out.status = "correct"
                    out.detail = "follows from the student's rounded earlier value"
            else:
                out.status = "wrong"
        else:
            out.status = "wrong"
        outcomes.append(out)

    return PathScore(
        path_id=path.path_id,
        score=sum(o.awarded for o in outcomes),
        max_points=path.total(),
        outcomes=outcomes,
    )


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-9, ROUNDING_REL_TOL * max(abs(a), abs(b)))
