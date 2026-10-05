"""Canonical transcript schema for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TranscriptResult:
    """Structured transcription of one question crop."""

    crop_id: str
    booklet_id: str
    page_name: str
    question_id: str
    question_label: str
    crop_path: str
    state: str

    hebrew_text: str = ""
    math_latex: List[str] = field(default_factory=list)
    interleaved_markdown: str = ""
    flagged_tokens: List[str] = field(default_factory=list)
    strikethrough_regions: List[str] = field(default_factory=list)

    model: str = ""
    prompt_version: str = ""
    latency_sec: float = 0.0
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    cache_hit: bool = False

    # Weak signal — never used as primary confidence
    self_reported_confidence: Optional[float] = None
    legibility: str = ""

    # Objective quality signals (computed locally)
    latex_eval: dict = field(default_factory=dict)
    objective_confidence: float = 0.0
    confidence_signals: dict = field(default_factory=dict)

    raw: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(
            self.interleaved_markdown or self.hebrew_text or self.math_latex
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_id": self.crop_id,
            "booklet_id": self.booklet_id,
            "page_name": self.page_name,
            "question_id": self.question_id,
            "question_label": self.question_label,
            "crop_path": self.crop_path,
            "state": self.state,
            "hebrew_text": self.hebrew_text,
            "math_latex": self.math_latex,
            "interleaved_markdown": self.interleaved_markdown,
            "flagged_tokens": self.flagged_tokens,
            "strikethrough_regions": self.strikethrough_regions,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "latency_sec": self.latency_sec,
            "prompt_tokens": self.prompt_tokens,
            "candidate_tokens": self.candidate_tokens,
            "cache_hit": self.cache_hit,
            "self_reported_confidence": self.self_reported_confidence,
            "legibility": self.legibility,
            "latex_eval": self.latex_eval,
            "objective_confidence": self.objective_confidence,
            "confidence_signals": self.confidence_signals,
            "raw": self.raw,
            "errors": self.errors,
            "ok": self.ok,
        }
