"""
Objective confidence scoring — do NOT trust self-reported model confidence.

Primary signals (Phase 3):
  - LaTeX SymPy parse success rate
  - Presence of flagged_tokens (penalty)
  - JSON completeness
  - Optional cross-model agreement (when two transcripts supplied)
"""

from __future__ import annotations

from typing import Optional


def compute_objective_confidence(
    *,
    latex_eval: dict,
    flagged_tokens: list,
    has_content: bool,
    is_blank: bool,
    self_reported: Optional[float] = None,
    cross_model_agreement: Optional[float] = None,
) -> tuple[float, dict]:
    """
    Return (score 0..1, signal breakdown).

    Self-reported confidence is logged but capped to a weak 10% weight max.
    """
    signals: dict = {}

    if is_blank:
        signals["blank_crop"] = 1.0
        return 0.95, signals

    if not has_content:
        signals["empty_transcript"] = 0.0
        return 0.15, signals

    parse_rate = float(latex_eval.get("parse_rate") or 0.0)
    total_formulas = int(latex_eval.get("total") or 0)
    if total_formulas == 0:
        # Prose-only answer — no LaTeX penalty
        latex_score = 0.92
        signals["latex_parse_rate"] = None
    else:
        latex_score = parse_rate
        signals["latex_parse_rate"] = parse_rate

    flag_penalty = min(0.35, 0.07 * len(flagged_tokens))
    signals["flagged_token_count"] = len(flagged_tokens)
    signals["flag_penalty"] = round(flag_penalty, 3)

    score = latex_score - flag_penalty
    score = max(0.05, min(0.98, score))

    if cross_model_agreement is not None:
        # High agreement boosts; low agreement pulls down hard
        agree = max(0.0, min(1.0, cross_model_agreement))
        signals["cross_model_agreement"] = agree
        score = 0.6 * score + 0.4 * agree

    if self_reported is not None:
        # Weak prior only — never let 0.95 alone inflate the score
        weak = max(0.0, min(1.0, float(self_reported)))
        signals["self_reported_confidence"] = weak
        score = 0.9 * score + 0.1 * min(weak, 0.85)

    signals["final"] = round(score, 4)
    return round(score, 4), signals
