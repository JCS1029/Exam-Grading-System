"""Repair previously failed Phase 3 transcripts using hardened JSON parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.transcription.discovery import CropRef
from core.transcription.metrics import aggregate_gate_metrics
from core.transcription.transcriber import (
    _parse_transcript_payload,
    save_transcript,
    transcribe_crop,
)
from core.vlm import parse_json_response

RESULTS_PATH = Path("storage/reports/phase3_results.json")
GROUND_TRUTH_PATH = Path("storage/eval/phase3_ground_truth.json")


def load_gt() -> dict[str, str]:
    if not GROUND_TRUTH_PATH.exists():
        return {}
    data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not str(k).startswith("_") and isinstance(v, str)}


def repair_from_raw(results: list[dict]) -> tuple[int, list[dict]]:
    """Re-parse failed rows that still have raw_text. Returns (n_repaired, still_failed)."""
    repaired = 0
    still: list[dict] = []
    for r in results:
        if not r.get("errors"):
            continue
        raw = (r.get("raw") or {}).get("raw_text") or ""
        if not raw.strip():
            still.append(r)
            continue
        try:
            parsed = parse_json_response(raw)
        except Exception:
            still.append(r)
            continue

        crop = CropRef(
            crop_id=r["crop_id"],
            booklet_id=r["booklet_id"],
            page_name=r["page_name"],
            question_id=r["question_id"],
            question_label=r.get("question_label") or "",
            crop_path=Path(r["crop_path"]),
            manifest_path=Path("."),
            state=r.get("state") or "answered",
        )
        filled = _parse_transcript_payload(parsed, crop)
        new = filled.to_dict()
        # Preserve billing / model metadata from the original call
        for k in (
            "model",
            "latency_sec",
            "prompt_tokens",
            "candidate_tokens",
            "cache_hit",
            "crop_path",
            "prompt_version",
        ):
            if k in r:
                new[k] = r[k]
        new["errors"] = []
        new["repaired_from_raw"] = True
        # Replace in-place
        idx = results.index(r)
        results[idx] = new
        repaired += 1
        if filled.ok:
            save_transcript(filled, model_slug=str(r.get("model") or "unknown"))
    return repaired, still


def retry_api(still: list[dict], results: list[dict]) -> int:
    """Live re-transcribe remaining failures with the new prompt (no cache)."""
    fixed = 0
    for r in still:
        crop_path = Path(r["crop_path"])
        if not crop_path.exists():
            print(f"  missing crop: {crop_path}")
            continue
        crop = CropRef(
            crop_id=r["crop_id"],
            booklet_id=r["booklet_id"],
            page_name=r["page_name"],
            question_id=r["question_id"],
            question_label=r.get("question_label") or "",
            crop_path=crop_path,
            manifest_path=Path("."),
            state=r.get("state") or "answered",
        )
        model = r.get("model") or "gemini-3.6-flash"
        print(f"  retry API {model} {crop.crop_id}")
        tr = transcribe_crop(crop, model=model, use_cache=False)
        new = tr.to_dict()
        new["model"] = model
        # Replace matching (crop_id, model)
        for i, old in enumerate(results):
            if old.get("crop_id") == r["crop_id"] and old.get("model") == model:
                results[i] = new
                break
        if tr.ok:
            save_transcript(tr, model_slug=model)
            fixed += 1
        else:
            print(f"    still failed: {tr.errors}")
    return fixed


def main() -> None:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results: list[dict] = payload["results"]
    before_err = sum(1 for r in results if r.get("errors"))
    print(f"Before: {before_err} errors / {len(results)} results")

    repaired, still = repair_from_raw(results)
    print(f"Repaired from raw_text: {repaired}")
    print(f"Still need API retry: {len(still)}")

    if still:
        fixed = retry_api(still, results)
        print(f"API retries fixed: {fixed}")

    after_err = sum(1 for r in results if r.get("errors"))
    print(f"After: {after_err} errors")

    gt = load_gt()
    models = sorted({r.get("model") for r in results if r.get("model")})
    gate_by_model = {
        m: aggregate_gate_metrics([r for r in results if r.get("model") == m], ground_truth=gt)
        for m in models
    }
    primary = "gemini-3.6-flash" if "gemini-3.6-flash" in models else (models[0] if models else "")
    summary = {
        "n_crops": len({r["crop_id"] for r in results}),
        "models": models,
        "primary_model": primary,
        "gate_metrics": gate_by_model,
        "gate_primary": gate_by_model.get(primary, {}),
        "repaired": True,
    }
    payload["summary"] = summary
    payload["results"] = results
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary.get("gate_primary"), indent=2, ensure_ascii=False))
    for m, g in gate_by_model.items():
        print(f"\n{m}: parse={g.get('mean_latex_parse_rate')} cost=${g.get('projected_cohort_cost_usd')} cer={g.get('mean_hebrew_cer')} errors={g.get('api_error_count')} gates={g.get('gates')}")


if __name__ == "__main__":
    main()
