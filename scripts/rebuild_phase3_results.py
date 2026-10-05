"""Rebuild storage/reports/phase3_results.json from saved transcript JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.transcription.metrics import aggregate_gate_metrics

TRANSCRIPTS = Path("storage/transcripts")
RESULTS = Path("storage/reports/phase3_results.json")
GT = Path("storage/eval/phase3_ground_truth.json")


def main() -> None:
    by_key: dict[tuple[str, str], dict] = {}

    # Prefer on-disk transcripts
    for path in sorted(TRANSCRIPTS.rglob("transcript_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        model = rec.get("model") or "unknown"
        # Filename often has model slug; prefer field
        cid = rec.get("crop_id")
        if not cid:
            continue
        by_key[(cid, model)] = rec

    # Overlay anything currently in results.json (newer sol run)
    if RESULTS.exists():
        cur = json.loads(RESULTS.read_text(encoding="utf-8"))
        for rec in cur.get("results") or []:
            cid = rec.get("crop_id")
            model = rec.get("model") or "unknown"
            if cid:
                by_key[(cid, model)] = rec

    results = list(by_key.values())
    gt = {}
    if GT.exists():
        raw = json.loads(GT.read_text(encoding="utf-8"))
        gt = {k: v for k, v in raw.items() if not str(k).startswith("_") and isinstance(v, str)}

    models = sorted({r.get("model") for r in results if r.get("model")})
    gate_by_model = {
        m: aggregate_gate_metrics([r for r in results if r.get("model") == m], ground_truth=gt)
        for m in models
    }
    primary = "gemini-3.6-flash" if "gemini-3.6-flash" in models else (models[0] if models else "")
    payload = {
        "summary": {
            "n_crops": len({r.get("crop_id") for r in results}),
            "models": models,
            "primary_model": primary,
            "gate_metrics": gate_by_model,
            "gate_primary": gate_by_model.get(primary, {}),
            "rebuilt_from_transcripts": True,
        },
        "ground_truth_path": str(GT),
        "n_ground_truth_entries": len(gt),
        "results": results,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rebuilt {RESULTS}: {len(results)} results, {len(models)} models, {payload['summary']['n_crops']} crops")


if __name__ == "__main__":
    main()
