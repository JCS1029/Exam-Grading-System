"""Dump candidate ground-truth seeds from dual-model agreement (UTF-8)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.transcription.metrics import hebrew_cer, normalize_hebrew_for_cer

results = json.loads(Path("storage/reports/phase3_results.json").read_text(encoding="utf-8"))["results"]
by: dict = {}
for r in results:
    if r.get("errors"):
        continue
    by.setdefault(r["crop_id"], {})[r["model"]] = r

lines = []
candidates = {}
for cid, models in sorted(by.items()):
    a = models.get("gemini-3.5-flash")
    b = models.get("gemini-3.6-flash")
    if not a or not b:
        continue
    ha = (a.get("hebrew_text") or "").strip()
    hb = (b.get("hebrew_text") or "").strip()
    # Prefer prose fields; fall back to interleaved with math stripped
    if len(normalize_hebrew_for_cer(ha)) < 8 and len(normalize_hebrew_for_cer(hb)) < 8:
        ha = normalize_hebrew_for_cer(a.get("interleaved_markdown") or "")
        hb = normalize_hebrew_for_cer(b.get("interleaved_markdown") or "")
    if len(normalize_hebrew_for_cer(ha)) < 8 and len(normalize_hebrew_for_cer(hb)) < 8:
        continue
    cer = hebrew_cer(ha or hb, hb or ha)
    lines.append(f"=== {cid}  cross_cer={cer:.3f}")
    lines.append(f"35: {ha[:300]}")
    lines.append(f"36: {hb[:300]}")
    lines.append("")
    # High agreement → use longer text as provisional GT (human should still review)
    if cer <= 0.08 and max(len(ha), len(hb)) >= 8:
        ref = ha if len(normalize_hebrew_for_cer(ha)) >= len(normalize_hebrew_for_cer(hb)) else hb
        candidates[cid] = ref

out = Path("storage/eval/_gt_candidates.txt")
out.write_text("\n".join(lines), encoding="utf-8")
gt_path = Path("storage/eval/phase3_ground_truth.json")
payload = {
    "_comment": "Provisional GT seeded from dual-model agreement (CER<=8% between models). Replace with human labels when available.",
    "_format": "crop_id → reference hebrew prose",
}
payload.update(candidates)
gt_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {out} ({len(lines)} lines)")
print(f"seeded {len(candidates)} GT entries → {gt_path}")
