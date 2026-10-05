"""Recompute latex_eval + objective confidence + gates on existing phase3 results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.transcription.confidence import compute_objective_confidence
from core.transcription.latex import collect_all_formulas, validate_latex_formulas
from core.transcription.metrics import aggregate_gate_metrics

RESULTS = Path("storage/reports/phase3_results.json")
GT = Path("storage/eval/phase3_ground_truth.json")


def main() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    results = payload["results"]
    for r in results:
        if r.get("errors"):
            continue
        formulas = collect_all_formulas(r.get("math_latex") or [], r.get("interleaved_markdown") or "")
        le = validate_latex_formulas(formulas)
        r["latex_eval"] = le
        is_blank = bool((r.get("raw") or {}).get("is_blank")) if isinstance(r.get("raw"), dict) else False
        has_content = bool(r.get("interleaved_markdown") or r.get("hebrew_text") or r.get("math_latex"))
        score, signals = compute_objective_confidence(
            latex_eval=le,
            flagged_tokens=r.get("flagged_tokens") or [],
            has_content=has_content,
            is_blank=is_blank,
            self_reported=r.get("self_reported_confidence"),
        )
        r["objective_confidence"] = score
        r["confidence_signals"] = signals

    gt = {}
    if GT.exists():
        raw = json.loads(GT.read_text(encoding="utf-8"))
        gt = {k: v for k, v in raw.items() if not str(k).startswith("_") and isinstance(v, str)}

    models = sorted({r.get("model") for r in results if r.get("model")})
    gate_by_model = {
        m: aggregate_gate_metrics([r for r in results if r.get("model") == m], ground_truth=gt)
        for m in models
    }
    primary = "gemini-3.6-flash" if "gemini-3.6-flash" in models else models[0]
    payload["summary"] = {
        **(payload.get("summary") or {}),
        "models": models,
        "primary_model": primary,
        "gate_metrics": gate_by_model,
        "gate_primary": gate_by_model[primary],
        "latex_recomputed": True,
    }
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for m, g in gate_by_model.items():
        print(m, "parse", g.get("mean_latex_parse_rate"), "cer", g.get("mean_hebrew_cer"),
              "cost", g.get("projected_cohort_cost_usd"), "errors", g.get("api_error_count"),
              "gates", g.get("gates"), "PASS" if g.get("all_gates_pass") else "FAIL")


if __name__ == "__main__":
    main()
