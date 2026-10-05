"""Export test8sol/test9sol transcripts to a readable Markdown file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path("storage/reports/phase3_results.json")
OUT = Path("storage/reports/phase3_sol_transcripts.md")
BOOKLETS = {"test8sol", "test9sol"}
MODEL = "gemini-3.6-flash"


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8")).get("results") or []
    rows = [
        r
        for r in results
        if r.get("booklet_id") in BOOKLETS and r.get("model") == MODEL and not r.get("errors")
    ]
    rows.sort(key=lambda r: r.get("crop_id") or "")

    lines = [
        "# Phase 3 transcripts — test8sol / test9sol",
        "",
        f"Model: `{MODEL}` · {len(rows)} questions",
        "",
        "Open the side-by-side review: [`phase3_sol_review.html`](./phase3_sol_review.html)",
        "",
    ]
    for r in rows:
        lines.append(f"## {r.get('crop_id')}")
        lines.append("")
        lines.append(f"- Label: {r.get('question_label')}")
        lines.append(f"- Objective confidence: {r.get('objective_confidence')}")
        le = r.get("latex_eval") or {}
        if le.get("total"):
            lines.append(
                f"- LaTeX parse: {le.get('parsed')}/{le.get('total')} ({100*(le.get('parse_rate') or 0):.0f}%)"
            )
        lines.append(f"- Crop: `{r.get('crop_path')}`")
        lines.append("")
        lines.append("### Hebrew")
        lines.append("")
        lines.append(r.get("hebrew_text") or "_(empty)_")
        lines.append("")
        lines.append("### Interleaved (Hebrew + LaTeX)")
        lines.append("")
        lines.append(r.get("interleaved_markdown") or "_(empty)_")
        lines.append("")
        math = r.get("math_latex") or []
        if math:
            lines.append("### Math formulas")
            lines.append("")
            for m in math:
                lines.append(f"- `${m}$`")
            lines.append("")
        flags = r.get("flagged_tokens") or []
        if flags:
            lines.append(f"**Flagged:** {', '.join(flags)}")
            lines.append("")
        lines.append("---")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(rows)} questions)")


if __name__ == "__main__":
    main()
