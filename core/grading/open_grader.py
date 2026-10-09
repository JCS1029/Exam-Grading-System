"""
Open-question grading against an approved rubric.

Split of responsibilities, so that grades are reproducible and defensible:
    LLM (temperature 0)  reads the student's transcript and reports, per
                         milestone of every declared path, the value the
                         student actually wrote — errors included — and
                         whether qualitative steps are present.
    consequential.py     turns those reports into points, deterministically.

Alternative paths: every instructor-declared path is scored and the best one
counts. A method outside all declared paths is the open-validity case: the
LLM's soundness opinion is recorded and used for the *proposed* score, but the
question is always routed to human review.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.grading.consequential import PathScore, score_path, student_number, value_matches
from core.grading.rubric import QuestionRubric

logger = logging.getLogger(__name__)

OPEN_PROMPT_VERSION = "open_grade_v2"
CACHE_DIR = Path(os.getenv("STORAGE_ROOT", "storage")) / "cache" / "open_grade"

OPEN_PROMPT = """\
You are reading a student's handwritten exam answer (already transcribed) to
report what the student did. You are NOT grading and must NOT correct errors.

Question:
{question_text}

Student's answer (transcript):
<<<
{transcript}
>>>

The instructor's rubric lists these milestones, grouped by solution path:
{milestones}

For every milestone key above report:
  - in "values": for milestones marked [value], what the STUDENT wrote for that
    quantity, exactly as written, even if it is wrong; null if the student
    never computed it. If the student wrote an expression AND evaluated it,
    copy the whole chain including the student's own numeric result, e.g.
    "0.4*0.95=0.36" or "\\binom{{5}}{{2}}0.1^2\\approx 0.07" — the student's
    evaluation may contain the error, so never drop it.
  - in "steps": for milestones marked [step], true if the student's work
    contains that step, otherwise false.
  - in "evidence": a short quote from the transcript supporting each answer.
Also report:
  - "path_followed": the path id the student follows, "other" if the student
    uses a substantially different method from every listed path, or "none".
  - "final_answer": the student's final answer as written (or null).
  - "undeclared_method": {{"used": bool, "description": str,
    "reasoning_valid": bool or null, "reaches_correct_conclusion": bool or null}}
    — fill reasoning fields only when used is true, judging the student's own
    argument on its mathematical merits.

Return ONLY JSON with keys: path_followed, values, steps, evidence,
final_answer, undeclared_method.
"""


@dataclass
class OpenGradeResult:
    question_id: str
    score: float
    max_points: float
    best_path: Optional[str]
    path_scores: List[PathScore] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    extraction: dict = field(default_factory=dict)
    raw_response: str = ""
    model: str = ""
    provider_model: str = ""
    cost_usd: float = 0.0

    def evidence(self) -> dict:
        return {
            "best_path": self.best_path,
            "paths": [p.to_dict() for p in self.path_scores],
            "extraction": self.extraction,
        }


def _milestone_listing(q: QuestionRubric) -> str:
    lines = []
    for p in q.paths:
        lines.append(f"Path {p.path_id}: {p.description}")
        for m in p.milestones:
            kind = "value" if m.is_value else "step"
            lines.append(f"  {p.path_id}.{m.id} [{kind}] {m.description}")
    return "\n".join(lines)


def build_prompt(q: QuestionRubric, transcript: str) -> str:
    return OPEN_PROMPT.format(
        question_text=q.question_text or f"Question {q.question_id}",
        transcript=transcript.strip(),
        milestones=_milestone_listing(q),
    )


def score_extraction(q: QuestionRubric, extraction: dict) -> OpenGradeResult:
    """Deterministic part: points from an extraction dict (testable without an LLM)."""
    values = extraction.get("values") or {}
    steps = extraction.get("steps") or {}
    path_scores = []
    for p in q.paths:
        sv = {m.id: _as_text(values.get(f"{p.path_id}.{m.id}")) for m in p.milestones}
        st = {m.id: bool(steps.get(f"{p.path_id}.{m.id}")) for m in p.milestones}
        path_scores.append(score_path(p, sv, st))

    best = max(path_scores, key=lambda s: s.score, default=None)
    result = OpenGradeResult(
        question_id=q.question_id,
        score=best.score if best else 0.0,
        max_points=q.max_points,
        best_path=best.path_id if best else None,
        path_scores=path_scores,
        extraction=extraction,
    )

    followed = extraction.get("path_followed")
    undeclared = extraction.get("undeclared_method") or {}
    if followed == "other" or undeclared.get("used"):
        result.needs_review = True
        result.review_reasons.append(
            "open-validity: student method not among declared paths"
            + (f" ({undeclared.get('description')})" if undeclared.get("description") else "")
        )
        # Correctness of the result is checked arithmetically; the LLM only judges soundness.
        if undeclared.get("reasoning_valid") is True and _final_answer_correct(
            q, extraction.get("final_answer")
        ):
            result.score = q.max_points
            result.review_reasons.append(
                "proposed full credit: LLM judges the method sound and the final answer is correct"
            )
    if best and followed not in (None, "none", "other") and followed != best.path_id:
        result.review_reasons.append(f"LLM says path {followed!r}, best-scoring path is {best.path_id!r}")
    return result


def _as_text(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _final_answer_correct(q: QuestionRubric, final_answer) -> bool:
    text = _as_text(final_answer)
    sv = student_number(text)
    if sv is None:
        return False
    for p in q.paths:
        values = p.resolve_expected()
        last = next((m for m in reversed(p.milestones) if m.is_value), None)
        if last and last.id in values and value_matches(text, sv, values[last.id], last.abs_tol):
            return True
    return False


def grade_open_question(
    q: QuestionRubric,
    transcript: str,
    *,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> OpenGradeResult:
    from core.vlm import generate_text, get_default_model, parse_json_response

    model = model or get_default_model()
    prompt = build_prompt(q, transcript)
    key = hashlib.sha256(f"{OPEN_PROMPT_VERSION}\n{model}\n{prompt}".encode("utf-8")).hexdigest()[:24]
    cache = CACHE_DIR / f"{key}.json"
    if use_cache and cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        text, provider_model, cost = cached["raw_text"], cached.get("provider_model", ""), 0.0
    else:
        resp = generate_text(prompt, model=model, json_mode=True, purpose=f"open_grade:Q{q.question_id}")
        text, provider_model, cost = resp.text, resp.provider_model, resp.cost_usd
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"raw_text": text, "provider_model": provider_model},
                                        ensure_ascii=False), encoding="utf-8")
    try:
        extraction = parse_json_response(text)
        if not isinstance(extraction, dict):
            raise ValueError("extraction is not a JSON object")
    except Exception as exc:
        result = OpenGradeResult(q.question_id, 0.0, q.max_points, None,
                                 needs_review=True, review_reasons=[f"unparseable LLM output: {exc}"])
    else:
        result = score_extraction(q, extraction)
    result.raw_response = text
    result.model, result.provider_model, result.cost_usd = model, provider_model, cost
    logger.debug("Q%s graded %.2f/%.2f via %s", q.question_id, result.score, q.max_points,
                 json.dumps(result.evidence(), ensure_ascii=False)[:200])
    return result
