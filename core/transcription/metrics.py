"""Phase 3 gate metrics: Hebrew CER, LaTeX parse rate, cost projection."""

from __future__ import annotations

import re
import statistics
import unicodedata
from typing import Optional, Sequence

import jiwer


# Rough OpenRouter Gemini Flash pricing (USD per 1M tokens) — 3.6-flash tier
_FLASH_INPUT_PER_M = 0.75
_FLASH_OUTPUT_PER_M = 3.75


def normalize_hebrew_for_cer(text: str) -> str:
    """Strip math, whitespace, and punctuation for prose CER."""
    if not text:
        return ""
    # Remove LaTeX spans
    t = re.sub(r"\$\$?[^$]+\$\$?", " ", text)
    t = re.sub(r"\[\[deleted:[^\]]+\]\]", " ", t, flags=re.IGNORECASE)
    # Remove niqqud / cantillation
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    # Keep Hebrew letters, digits, basic punctuation
    t = re.sub(r"[^\u0590-\u05FF0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def hebrew_cer(reference: str, hypothesis: str) -> float:
    """Character error rate on normalized Hebrew prose (0 = perfect)."""
    ref = normalize_hebrew_for_cer(reference)
    hyp = normalize_hebrew_for_cer(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    return float(jiwer.cer(ref, hyp))


def estimate_cohort_cost_usd(
    *,
    n_crops: int,
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
) -> float:
    """Project full-cohort transcription cost from measured per-crop averages."""
    prompt = n_crops * avg_prompt_tokens
    completion = n_crops * avg_completion_tokens
    return round(
        (prompt / 1_000_000) * _FLASH_INPUT_PER_M
        + (completion / 1_000_000) * _FLASH_OUTPUT_PER_M,
        4,
    )


def _median_or_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) >= 3:
        return float(statistics.median(values))
    return float(sum(values) / len(values))


def aggregate_gate_metrics(
    results: Sequence[dict],
    *,
    ground_truth: Optional[dict[str, str]] = None,
    cohort_pages: int = 3600,
    crops_per_page: float = 1.5,
) -> dict:
    """
    Summarise Phase 3 exit-gate metrics from transcription result dicts.

    Cost projection uses the median token counts of successful non-cache calls
    so a few pathological long generations cannot inflate the cohort estimate.
    """
    gt = ground_truth or {}
    cer_values: list[float] = []
    parse_rates: list[float] = []
    api_errors = 0
    unhandled_503 = 0
    prompt_samples: list[float] = []
    completion_samples: list[float] = []

    for r in results:
        if r.get("errors"):
            api_errors += 1
            for err in r["errors"]:
                if "503" in str(err) and "retry" not in str(err).lower():
                    unhandled_503 += 1

        le = r.get("latex_eval") or {}
        if le.get("total", 0) > 0 and not r.get("errors"):
            parse_rates.append(float(le.get("parse_rate") or 0.0))

        # Cost: only successful paid calls; median resists runaway completions
        if (
            not r.get("cache_hit")
            and not r.get("errors")
            and (
                r.get("ok")
                or (
                    r.get("hebrew_text")
                    or r.get("interleaved_markdown")
                    or r.get("math_latex")
                )
            )
        ):
            pt = int(r.get("prompt_tokens") or 0)
            ct = int(r.get("candidate_tokens") or 0)
            if pt > 0 or ct > 0:
                prompt_samples.append(float(pt))
                completion_samples.append(float(ct))

        cid = r.get("crop_id") or ""
        if cid in gt and not r.get("errors"):
            ref = gt[cid]
            hyp = r.get("hebrew_text") or r.get("interleaved_markdown") or ""
            cer_values.append(hebrew_cer(ref, hyp))

    avg_prompt = _median_or_mean(prompt_samples)
    avg_completion = _median_or_mean(completion_samples)
    projected_crops = int(cohort_pages * crops_per_page)
    projected_cost = estimate_cohort_cost_usd(
        n_crops=projected_crops,
        avg_prompt_tokens=avg_prompt,
        avg_completion_tokens=avg_completion,
    )

    mean_cer = round(sum(cer_values) / len(cer_values), 4) if cer_values else None
    mean_parse = round(sum(parse_rates) / len(parse_rates), 4) if parse_rates else None

    gates = {
        "hebrew_cer_le_8pct": mean_cer is not None and mean_cer <= 0.08,
        "latex_parse_ge_90pct": mean_parse is not None and mean_parse >= 0.90,
        "zero_unhandled_503": unhandled_503 == 0,
        "cohort_cost_under_25usd": projected_cost < 25.0,
    }

    return {
        "n_results": len(results),
        "n_ok": sum(
            1
            for r in results
            if r.get("ok")
            or (
                not r.get("errors")
                and (r.get("hebrew_text") or r.get("interleaved_markdown"))
            )
        ),
        "n_with_ground_truth": len(cer_values),
        "mean_hebrew_cer": mean_cer,
        "mean_latex_parse_rate": mean_parse,
        "api_error_count": api_errors,
        "unhandled_503_count": unhandled_503,
        "avg_prompt_tokens_per_call": round(avg_prompt, 1),
        "avg_completion_tokens_per_call": round(avg_completion, 1),
        "projected_cohort_crops": projected_crops,
        "projected_cohort_cost_usd": projected_cost,
        "gates": gates,
        "all_gates_pass": all(gates.values()) if mean_cer is not None else False,
    }
