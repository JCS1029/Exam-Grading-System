"""
Grading orchestration.

Multiple-choice booklets:
    cover sheet → CV grid reading + VLM cross-check → reconciled answers
    cover sheet → suit symbol → version key from the approved rubric
    question pages → printed options → this booklet's own key (consistency check)
    answer == key → full points; anything uncertain is flagged for review,
    never silently resolved.

Every question becomes a ``GradeRecord`` carrying the rubric version/hash,
models, prompt versions, temperature and raw model output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.grading.answer_key import McqAnswerKey
from core.grading.omr import (
    VLM_GRID_PROMPT_VERSION,
    crop_grid,
    read_answer_grid,
    reconcile,
    vlm_read_grid,
)
from core.grading.rubric import Rubric
from core.grading.store import GradeRecord
from core.grading.versions import (
    OPTIONS_PROMPT_VERSION,
    detect_suit,
    map_to_master,
    read_printed_options,
)

logger = logging.getLogger(__name__)

MCQ_PIPELINE_VERSION = "mcq_pipeline_v1"


@dataclass
class BookletResult:
    booklet_id: str
    suit: Optional[str]
    grades: List[GradeRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def total(self) -> Optional[float]:
        if any(g.score is None for g in self.grades):
            return None
        return sum(g.score for g in self.grades)

    @property
    def review_count(self) -> int:
        return sum(g.needs_review for g in self.grades)


def grade_mcq_booklet(
    booklet_dir: str | Path,
    rubric: Rubric,
    rubric_version: int,
    *,
    master: Optional[McqAnswerKey] = None,
    crosscheck_model: Optional[str] = None,
    use_vlm: bool = True,
) -> BookletResult:
    """Grade the multiple-choice cover sheet of one anonymised booklet."""
    from core.vlm import VLM_TEMPERATURE, get_default_model

    booklet_dir = Path(booklet_dir)
    bid = booklet_dir.name
    cover = booklet_dir / "page_001.png"
    out = BookletResult(booklet_id=bid, suit=None)

    cv = read_answer_grid(cover)
    if cv.errors:
        out.errors.extend(cv.errors)
    vlm = None
    model = crosscheck_model or get_default_model()
    if use_vlm and cv.grid_bbox:
        jpeg = crop_grid(cover, cv)
        if jpeg:
            vlm = vlm_read_grid(jpeg, model=model)
            out.cost_usd += vlm.cost_usd
            out.errors.extend(vlm.errors)
    rows = {r.question_id: r for r in reconcile(cv, vlm)}

    suit = detect_suit(cover, cv.grid_bbox) if cv.grid_bbox else None
    out.suit = suit.suit if suit else None

    # The booklet's own printed pages must agree with the approved key.
    own_letters: Dict[str, str] = {}
    own_issues: Dict[str, str] = {}
    if use_vlm and master is not None:
        pages = sorted(p for p in booklet_dir.glob("page_*.png") if p.name != cover.name)
        printed = read_printed_options(bid, pages, model=model)
        out.cost_usd += printed.cost_usd
        out.errors.extend(printed.errors)
        own_letters, own_issues = map_to_master(printed.questions, master)

    for q in rubric.questions:
        if q.kind != "mcq":
            continue
        reasons: List[str] = []
        row = rows.get(q.question_id)
        answer = row.answer if row else None
        if row is None:
            reasons.append("question row not read")
        elif row.needs_review:
            reasons.extend(row.reasons)

        key = q.mcq_keys.get(out.suit or "")
        if suit is None or suit.needs_review or out.suit is None:
            reasons.append(f"exam version unknown: {suit.reason if suit else 'no grid'}")
        elif key is None:
            reasons.append(f"no approved key for version {out.suit!r}")
        if own_letters.get(q.question_id) and key and own_letters[q.question_id] != key:
            reasons.append(
                f"booklet's printed options give key {own_letters[q.question_id]}, "
                f"approved {out.suit} key is {key}"
            )
        if q.question_id in own_issues:
            reasons.append(f"printed options unreadable: {own_issues[q.question_id]}")

        score = None if key is None else (q.max_points if answer == key else 0.0)
        evidence = {
            "pipeline": MCQ_PIPELINE_VERSION,
            "suit": suit.to_dict() if suit else None,
            "key_letter": key,
            "answer": answer,
            "cv_row": next((r.to_dict() for r in cv.rows if r.question_id == q.question_id), None),
            "vlm_row": next(
                ({"marks": [list(m) for m in r.marks], "final_answer": r.final_answer, "note": r.note}
                 for r in (vlm.rows if vlm else []) if r.question_id == q.question_id),
                None,
            ),
            "own_key_letter": own_letters.get(q.question_id),
        }
        out.grades.append(GradeRecord(
            booklet_id=bid,
            question_id=q.question_id,
            score=score,
            max_score=q.max_points,
            rubric_id=rubric.rubric_id,
            rubric_version=rubric_version,
            question_hash=rubric.question_hash(q.question_id),
            model=f"cv:omr+vlm:{model}" if vlm else "cv:omr",
            provider_model=(vlm.provider_model or model) if vlm else "",
            prompt_version=f"{VLM_GRID_PROMPT_VERSION}+{OPTIONS_PROMPT_VERSION}" if vlm else "",
            temperature=VLM_TEMPERATURE,
            raw_response=vlm.raw_text if vlm else "",
            evidence=evidence,
            needs_review=bool(reasons),
            review_reasons=reasons,
        ))
    return out
