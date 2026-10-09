"""
Phase 4 exit-gate metrics.

Per-question scores are normalised to a 10-point scale before comparison, as
the plan specifies, so questions of different weight are comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

GATE_MAE_MAX = 1.0
GATE_WITHIN1_MIN = 0.85
GATE_PEARSON_MIN = 0.90
GATE_SIGNED_ABS_MAX = 0.3
GATE_ALT_FULL_MIN = 0.90
GATE_CONSEQ_ONCE_MIN = 0.90

Scores = Dict[str, Dict[str, Optional[float]]]   # booklet → question → score


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


@dataclass
class QuestionGate:
    n: int
    mae: float
    within1: float
    signed: float

    @property
    def passed(self) -> bool:
        return (self.mae <= GATE_MAE_MAX and self.within1 >= GATE_WITHIN1_MIN
                and abs(self.signed) <= GATE_SIGNED_ABS_MAX)


def question_gate(system: Scores, reference: Scores, max_points: Dict[str, float]) -> QuestionGate:
    diffs: List[float] = []
    for bid, ref_q in reference.items():
        for qid, ref in ref_q.items():
            sys_score = system.get(bid, {}).get(qid)
            if ref is None or sys_score is None:
                continue
            scale = 10.0 / max_points.get(qid, 10.0)
            diffs.append((sys_score - ref) * scale)
    n = len(diffs)
    if not n:
        return QuestionGate(0, float("nan"), float("nan"), float("nan"))
    return QuestionGate(
        n=n,
        mae=sum(abs(d) for d in diffs) / n,
        within1=sum(abs(d) <= 1.0 + 1e-9 for d in diffs) / n,
        signed=sum(diffs) / n,
    )


@dataclass
class TotalGate:
    n: int
    pearson: Optional[float]
    mae_total: float
    signed_total: float
    exact_matches: int
    disagreements: Dict[str, tuple] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.pearson is not None and self.pearson >= GATE_PEARSON_MIN


def total_gate(system_totals: Dict[str, Optional[float]], reference_totals: Dict[str, float]) -> TotalGate:
    pairs = [(b, system_totals[b], reference_totals[b]) for b in sorted(reference_totals)
             if system_totals.get(b) is not None]
    xs = [s for _, s, _ in pairs]
    ys = [r for _, _, r in pairs]
    n = len(pairs)
    return TotalGate(
        n=n,
        pearson=pearson(xs, ys),
        mae_total=sum(abs(s - r) for _, s, r in pairs) / n if n else float("nan"),
        signed_total=sum(s - r for _, s, r in pairs) / n if n else float("nan"),
        exact_matches=sum(abs(s - r) < 1e-9 for _, s, r in pairs),
        disagreements={b: (s, r) for b, s, r in pairs if abs(s - r) >= 1e-9},
    )
