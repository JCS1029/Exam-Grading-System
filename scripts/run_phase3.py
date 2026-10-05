"""
Phase 3 Pipeline Runner — VLM Transcription on Segmented Crops.

Token-conservative by default (see .env.example): single model gemini-3.6-flash,
cached responses, no dual bake-off unless you pass --models.

Usage:
    python scripts/run_phase3.py
    python scripts/run_phase3.py --booklets 02,05 --limit 10
    python scripts/run_phase3.py --models gemini-3.6-flash
    python scripts/run_phase3.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.transcription.discovery import discover_crops
from core.transcription.metrics import aggregate_gate_metrics
from core.transcription.transcriber import bakeoff_models, save_transcript, transcribe_crop
from core.vlm import GEMINI_API_KEY

GROUND_TRUTH_PATH = Path("storage/eval/phase3_ground_truth.json")
RESULTS_PATH = Path("storage/reports/phase3_results.json")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def load_ground_truth() -> dict[str, str]:
    if not GROUND_TRUTH_PATH.exists():
        return {}
    data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, str)}


def run_phase3(
    *,
    booklet_ids: list[str] | None = None,
    models: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    states: list[str] | None = None,
) -> dict:
    logger = logging.getLogger("phase3")
    start = time.time()
    model_list = models or bakeoff_models()

    logger.info("=" * 60)
    logger.info("PHASE 3: VLM Transcription Bake-Off")
    logger.info("=" * 60)
    logger.info("Models: %s", ", ".join(model_list))
    logger.info("API key: %s", "present" if GEMINI_API_KEY else "MISSING")

    crops = discover_crops(booklet_ids=booklet_ids, states=states or ["answered"])
    if limit:
        crops = crops[:limit]

    logger.info("Crops to transcribe: %d (answered)", len(crops))

    if dry_run:
        return {
            "dry_run": True,
            "n_crops": len(crops),
            "models": model_list,
            "crop_ids": [c.crop_id for c in crops],
        }

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set — cannot run live transcription.")
        return {"error": "missing_api_key", "n_crops": len(crops)}

    all_results: list[dict] = []
    by_model: dict[str, list[dict]] = {m: [] for m in model_list}

    for model in model_list:
        logger.info("--- Model: %s ---", model)
        for i, crop in enumerate(crops, 1):
            logger.info("[%d/%d] %s", i, len(crops), crop.crop_id)
            tr = transcribe_crop(crop, model=model)
            rec = tr.to_dict()
            rec["model"] = model
            all_results.append(rec)
            by_model[model].append(rec)
            if tr.ok:
                save_transcript(tr, model_slug=model)
            if tr.errors:
                for err in tr.errors:
                    logger.warning("  error: %s", err)
            else:
                logger.info(
                    "  ok | obj_conf=%.2f latex=%.0f%% latency=%.1fs cache=%s",
                    tr.objective_confidence,
                    (tr.latex_eval.get("parse_rate") or 0) * 100,
                    tr.latency_sec,
                    tr.cache_hit,
                )

    gt = load_ground_truth()
    gate_by_model = {
        m: aggregate_gate_metrics(recs, ground_truth=gt) for m, recs in by_model.items()
    }

    # Pick primary model gate (first in list)
    primary = model_list[0]
    summary = {
        "n_crops": len(crops),
        "models": model_list,
        "primary_model": primary,
        "elapsed_sec": round(time.time() - start, 2),
        "gate_metrics": gate_by_model,
        "gate_primary": gate_by_model.get(primary, {}),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Merge with prior results so filtered runs do not wipe the cohort file
    prior: list[dict] = []
    if RESULTS_PATH.exists():
        try:
            prior = json.loads(RESULTS_PATH.read_text(encoding="utf-8")).get("results") or []
        except (OSError, json.JSONDecodeError):
            prior = []
    merged: dict[tuple[str, str], dict] = {}
    for rec in prior:
        key = (str(rec.get("crop_id") or ""), str(rec.get("model") or ""))
        if key[0]:
            merged[key] = rec
    for rec in all_results:
        key = (str(rec.get("crop_id") or ""), str(rec.get("model") or ""))
        if key[0]:
            merged[key] = rec
    merged_results = list(merged.values())
    all_models = sorted({str(r.get("model")) for r in merged_results if r.get("model")})
    full_gates = {
        m: aggregate_gate_metrics(
            [r for r in merged_results if r.get("model") == m], ground_truth=gt
        )
        for m in all_models
    }
    full_primary = primary if primary in full_gates else (all_models[0] if all_models else primary)
    payload = {
        "summary": {
            **summary,
            "n_crops_total": len({r.get("crop_id") for r in merged_results}),
            "n_results_total": len(merged_results),
            "models": all_models,
            "gate_metrics": full_gates,
            "gate_primary": full_gates.get(full_primary, summary.get("gate_primary")),
            "this_run_crop_ids": [c.crop_id for c in crops],
        },
        "ground_truth_path": str(GROUND_TRUTH_PATH),
        "n_ground_truth_entries": len(gt),
        "results": merged_results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 60)
    for model, gm in gate_by_model.items():
        logger.info("Model %s:", model)
        logger.info("  Mean Hebrew CER:     %s", gm.get("mean_hebrew_cer"))
        logger.info("  Mean LaTeX parse:    %s", gm.get("mean_latex_parse_rate"))
        logger.info("  Unhandled 503s:      %s", gm.get("unhandled_503_count"))
        logger.info("  Projected cohort $:  %s", gm.get("projected_cohort_cost_usd"))
        logger.info("  Gates:               %s", gm.get("gates"))
    logger.info("Results: %s", RESULTS_PATH)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — VLM Transcription Bake-Off")
    parser.add_argument("--booklets", type=str, default=None, help="Comma-separated booklet IDs")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model IDs")
    parser.add_argument("--limit", type=int, default=None, help="Max crops to transcribe")
    parser.add_argument("--states", type=str, default="answered", help="Crop states (default: answered)")
    parser.add_argument("--dry-run", action="store_true", help="List crops without API calls")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    booklet_ids = (
        [b.strip() for b in args.booklets.split(",") if b.strip()] if args.booklets else None
    )
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    states = [s.strip() for s in args.states.split(",") if s.strip()]

    summary = run_phase3(
        booklet_ids=booklet_ids,
        models=models,
        limit=args.limit,
        dry_run=args.dry_run,
        states=states,
    )

    if summary.get("error"):
        sys.exit(1)
    gp = summary.get("gate_primary") or {}
    if gp and not gp.get("all_gates_pass") and gp.get("n_with_ground_truth", 0) > 0:
        logging.getLogger("phase3").warning(
            "Primary model did not pass all exit gates — review phase3_review.html"
        )


if __name__ == "__main__":
    main()
