"""
Multiple-choice answer key extraction from a digital solution PDF.

The instructor's solution PDF marks the correct option of every question in
red. Because the PDF is born-digital, the key is read from the text layer —
deterministic and free, no VLM involved.

Usage:
    from core.grading.answer_key import extract_mcq_key

    key = extract_mcq_key("sol.pdf")
    key.questions[0].correct_letter   # "ד"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf

logger = logging.getLogger(__name__)

OPTION_LETTERS: tuple[str, ...] = ("א", "ב", "ג", "ד")
RED_MIN: int = 0xC0          # red channel ≥ this and green/blue ≤ GB_MAX → "red"
GB_MAX: int = 0x60
ROW_TOLERANCE_PT: float = 6.0

_LABEL_RE = re.compile(r"^[()]{0,2}\s*([אבגד])\s*[()]{0,2}$")
_QUESTION_RE = re.compile(r"(\d+)\s*שאלה|שאלה\s*(\d+)")
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


@dataclass
class McqQuestion:
    question_id: str
    options: Dict[str, str] = field(default_factory=dict)   # letter → option text
    correct_letter: Optional[str] = None

    @property
    def correct_text(self) -> Optional[str]:
        return self.options.get(self.correct_letter or "")


@dataclass
class McqAnswerKey:
    source: str
    version_id: str = "master"
    questions: List[McqQuestion] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(q.correct_letter for q in self.questions)

    def letters(self) -> Dict[str, str]:
        return {q.question_id: q.correct_letter or "" for q in self.questions}

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "version_id": self.version_id,
            "questions": [
                {
                    "question_id": q.question_id,
                    "options": q.options,
                    "correct_letter": q.correct_letter,
                    "correct_text": q.correct_text,
                }
                for q in self.questions
            ],
            "errors": self.errors,
        }


def _is_red(color: int) -> bool:
    r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return r >= RED_MIN and g <= GB_MAX and b <= GB_MAX


def _join_spans(spans: List[dict]) -> str:
    """Join value spans in reading order: RTL for Hebrew, LTR for math/numbers."""
    if not spans:
        return ""
    texts = [s["text"].strip() for s in spans]
    if any(_HEBREW_RE.search(t) for t in texts):
        ordered = sorted(spans, key=lambda s: -s["bbox"][0])
        return " ".join(s["text"].strip() for s in ordered if s["text"].strip())
    ordered = sorted(spans, key=lambda s: s["bbox"][0])
    return "".join(s["text"].strip() for s in ordered)


def _page_spans(page: pymupdf.Page) -> List[dict]:
    spans: List[dict] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip():
                    spans.append(span)
    return spans


def extract_mcq_key(pdf_path: str | Path, *, version_id: str = "master") -> McqAnswerKey:
    """
    Read every question's options and the red-marked correct option.

    Any question without exactly one red option is reported in ``errors`` —
    an ambiguous key must halt grading, never guess.
    """
    pdf_path = Path(pdf_path)
    key = McqAnswerKey(source=str(pdf_path), version_id=version_id)
    doc = pymupdf.open(str(pdf_path))

    by_id: Dict[str, McqQuestion] = {}
    order: List[str] = []
    red_count: Dict[str, int] = {}

    for page in doc:
        spans = _page_spans(page)
        headers = []
        for s in spans:
            m = _QUESTION_RE.search(s["text"].replace(":", " "))
            if m and "שאלה" in s["text"]:
                headers.append((s["bbox"][1], m.group(1) or m.group(2)))
        headers.sort()

        labels = [s for s in spans if _LABEL_RE.match(s["text"].strip())]
        for label in labels:
            letter = _LABEL_RE.match(label["text"].strip()).group(1)
            y = label["bbox"][1]
            owner = None
            for hy, qid in headers:
                if hy <= y + 1:
                    owner = qid
            if owner is None:
                continue
            row = [
                s for s in spans
                if abs(s["bbox"][1] - y) <= ROW_TOLERANCE_PT and s is not label
            ]
            row_labels = sorted(
                (s for s in row if _LABEL_RE.match(s["text"].strip())),
                key=lambda s: s["bbox"][0],
            )
            # Value sits left of its label, right of the next label to the left.
            left_bound = max(
                (s["bbox"][2] for s in row_labels if s["bbox"][2] <= label["bbox"][0]),
                default=0.0,
            )
            value_spans = [
                s for s in row
                if not _LABEL_RE.match(s["text"].strip())
                and left_bound - 1 <= s["bbox"][0] < label["bbox"][0]
            ]
            text = _join_spans(value_spans)

            q = by_id.get(owner)
            if q is None:
                q = McqQuestion(question_id=owner)
                by_id[owner] = q
                order.append(owner)
            q.options[letter] = text
            if _is_red(label["color"]) or any(_is_red(s["color"]) for s in value_spans):
                red_count[owner] = red_count.get(owner, 0) + 1
                q.correct_letter = letter

    doc.close()
    key.questions = [by_id[qid] for qid in sorted(order, key=lambda x: int(x))]
    for q in key.questions:
        n_red = red_count.get(q.question_id, 0)
        if n_red != 1:
            key.errors.append(f"Question {q.question_id}: {n_red} red options (expected exactly 1)")
            q.correct_letter = None
        if set(q.options) != set(OPTION_LETTERS):
            key.errors.append(
                f"Question {q.question_id}: options {sorted(q.options)} (expected א–ד)"
            )
    if not key.questions:
        key.errors.append("No questions found — is this a digital (text-layer) PDF?")
    logger.info("Extracted MCQ key from %s: %d questions, %d errors",
                pdf_path.name, len(key.questions), len(key.errors))
    return key
