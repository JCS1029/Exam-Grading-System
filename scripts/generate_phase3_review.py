"""
Generate Phase 3 transcription review HTML — crop vs transcript vs ground truth.

Loads storage/reports/phase3_results.json and crop images via relative paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_JSON = Path("storage/reports/phase3_results.json")
GROUND_TRUTH_JSON = Path("storage/eval/phase3_ground_truth.json")
OUTPUT_HTML = Path("storage/reports/phase3_review.html")


def load_items(booklet_ids: list[str] | None = None) -> tuple[list[dict], dict, dict]:
    if not RESULTS_JSON.exists():
        return [], {}, {}

    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    summary = payload.get("summary") or {}

    if booklet_ids:
        allowed = set(booklet_ids)
        results = [r for r in results if r.get("booklet_id") in allowed]
        summary = {
            **summary,
            "filter_booklets": booklet_ids,
            "n_filtered": len(results),
        }

    gt: dict = {}
    if GROUND_TRUTH_JSON.exists():
        raw = json.loads(GROUND_TRUTH_JSON.read_text(encoding="utf-8"))
        gt = {k: v for k, v in raw.items() if not str(k).startswith("_")}

    items: list[dict] = []
    for r in results:
        crop_path = Path(r.get("crop_path") or "")
        rel_crop = None
        if crop_path.exists():
            parts = crop_path.parts
            if "crops" in parts:
                idx = parts.index("crops")
                rel_crop = "../" + "/".join(parts[idx:]).replace("\\", "/")
        elif r.get("crop_path"):
            # Still emit a relative path guess for browser loading
            p = Path(r["crop_path"])
            parts = p.parts
            if "crops" in parts:
                idx = parts.index("crops")
                rel_crop = "../" + "/".join(parts[idx:]).replace("\\", "/")

        items.append({
            **r,
            "crop_src": rel_crop,
            "ground_truth": gt.get(r.get("crop_id") or ""),
        })
    return items, summary, gt


def generate_html(items: list[dict], summary: dict, gt: dict) -> str:
    items_json = json.dumps(items, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False)
    gt_json = json.dumps(gt, ensure_ascii=False)
    n = len(items)
    models = summary.get("models") or []

    return f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 3 Transcription Review — Crop / Transcript / Ground Truth</title>
    <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0d1117; --card: #161b22; --border: #30363d;
            --blue: #58a6ff; --green: #3fb950; --orange: #d29922; --red: #f85149;
            --text: #c9d1d9; --muted: #8b949e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: var(--bg); color: var(--text); font-family: 'Assistant', sans-serif; height: 100vh; display: flex; flex-direction: column; }}
        header {{ padding: 12px 20px; border-bottom: 1px solid var(--border); display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
        header h1 {{ font-size: 1.1rem; font-weight: 600; }}
        .badge {{ background: #21262d; border: 1px solid var(--border); padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; color: var(--muted); }}
        .main {{ display: flex; flex: 1; overflow: hidden; }}
        .sidebar {{ width: 320px; border-left: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; }}
        .sidebar-item {{ padding: 10px 14px; border-bottom: 1px solid var(--border); cursor: pointer; font-size: 0.85rem; }}
        .sidebar-item:hover, .sidebar-item.active {{ background: #21262d; }}
        .sidebar-item .id {{ font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--blue); }}
        .content {{ flex: 1; overflow-y: auto; padding: 20px; }}
        .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        @media (max-width: 900px) {{ .split {{ grid-template-columns: 1fr; }} }}
        .panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
        .panel h3 {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 12px; }}
        .crop-img {{ max-width: 100%; border-radius: 4px; border: 1px solid var(--border); }}
        .hebrew {{ direction: rtl; text-align: right; line-height: 1.7; white-space: pre-wrap; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
        .metric {{ background: #21262d; padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; }}
        .metric.good {{ border: 1px solid var(--green); color: var(--green); }}
        .metric.warn {{ border: 1px solid var(--orange); color: var(--orange); }}
        .metric.bad {{ border: 1px solid var(--red); color: var(--red); }}
        textarea {{ width: 100%; min-height: 100px; background: #0d1117; color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-family: 'Assistant', sans-serif; direction: rtl; }}
        .flags {{ color: var(--orange); font-size: 0.85rem; margin-top: 8px; }}
        .gate-box {{ margin-top: 16px; padding: 12px; background: #21262d; border-radius: 8px; font-size: 0.85rem; }}
    </style>
</head>
<body>
    <header>
        <h1>Phase 3 — Transcription Review{(' — ' + ', '.join(summary.get('filter_booklets') or [])) if summary.get('filter_booklets') else ''}</h1>
        <span class="badge">{n} transcripts</span>
        <span class="badge">Models: {', '.join(models) if models else 'none'}</span>
        <span class="badge" id="gateBadge">Gate metrics loading…</span>
    </header>
    <div class="main">
        <div class="content" id="mainContent">
            <p style="color: var(--muted);">Select a crop from the sidebar.</p>
        </div>
        <div class="sidebar" id="sidebar"></div>
    </div>
    <script>
    const items = {items_json};
    const summary = {summary_json};
    const groundTruth = {gt_json};

    function cerClass(cer) {{
        if (cer == null) return '';
        if (cer <= 0.08) return 'good';
        if (cer <= 0.15) return 'warn';
        return 'bad';
    }}

    function normalizeHebrew(t) {{
        return (t || '').replace(/\\$[^$]+\\$/g, ' ').replace(/\\s+/g, ' ').trim();
    }}

    function simpleCER(ref, hyp) {{
        ref = normalizeHebrew(ref);
        hyp = normalizeHebrew(hyp);
        if (!ref && !hyp) return 0;
        if (!ref || !hyp) return 1;
        const m = ref.length, n = hyp.length;
        const dp = Array.from({{length: m+1}}, (_, i) => Array(n+1).fill(0));
        for (let i=0;i<=m;i++) dp[i][0]=i;
        for (let j=0;j<=n;j++) dp[0][j]=j;
        for (let i=1;i<=m;i++) for (let j=1;j<=n;j++) {{
            dp[i][j] = ref[i-1]===hyp[j-1] ? dp[i-1][j-1] : 1+Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
        }}
        return dp[m][n] / m;
    }}

    function renderGateBadge() {{
        const gp = summary.gate_primary || {{}};
        const el = document.getElementById('gateBadge');
        if (!gp.gates) {{ el.textContent = 'Run phase3 first'; return; }}
        const pass = gp.all_gates_pass ? 'PASS' : 'REVIEW';
        el.textContent = `Gate: ${{pass}} | CER=${{gp.mean_hebrew_cer ?? 'n/a'}} | LaTeX=${{gp.mean_latex_parse_rate ?? 'n/a'}}`;
    }}

    function renderSidebar() {{
        const sb = document.getElementById('sidebar');
        sb.innerHTML = items.map((it, i) => `
            <div class="sidebar-item" data-idx="${{i}}">
                <div class="id">${{it.crop_id}}</div>
                <div>${{it.model}} | obj=${{(it.objective_confidence||0).toFixed(2)}}</div>
            </div>`).join('');
        sb.querySelectorAll('.sidebar-item').forEach(el => {{
            el.addEventListener('click', () => selectItem(+el.dataset.idx));
        }});
    }}

    function selectItem(idx) {{
        document.querySelectorAll('.sidebar-item').forEach((el, i) => {{
            el.classList.toggle('active', i === idx);
        }});
        const it = items[idx];
        const gt = it.ground_truth || groundTruth[it.crop_id] || '';
        const hyp = it.hebrew_text || it.interleaved_markdown || '';
        const cer = gt ? simpleCER(gt, hyp) : null;
        const latex = it.latex_eval || {{}};
        const parsePct = latex.total ? Math.round((latex.parse_rate||0)*100) : 'n/a';

        document.getElementById('mainContent').innerHTML = `
            <div class="metrics">
                <span class="metric">model: ${{it.model}}</span>
                <span class="metric ${{it.objective_confidence >= 0.7 ? 'good' : 'warn'}}">objective conf: ${{(it.objective_confidence||0).toFixed(3)}}</span>
                <span class="metric">self-reported: ${{it.self_reported_confidence ?? 'n/a'}}</span>
                <span class="metric ${{latex.parse_rate >= 0.9 ? 'good' : latex.total ? 'warn' : ''}}">LaTeX parse: ${{parsePct}}%</span>
                <span class="metric ${{cerClass(cer)}}">CER: ${{cer != null ? (cer*100).toFixed(1)+'%' : 'no GT'}}</span>
                <span class="metric">${{it.cache_hit ? 'cached' : it.latency_sec+'s'}}</span>
            </div>
            <div class="split">
                <div class="panel">
                    <h3>Question crop</h3>
                    ${{it.crop_src ? `<img class="crop-img" src="${{it.crop_src}}" alt="crop">` : '<p>No image</p>'}}
                </div>
                <div class="panel">
                    <h3>Transcription (interleaved)</h3>
                    <div class="hebrew">${{it.interleaved_markdown || it.hebrew_text || '(empty)'}}</div>
                    ${{it.flagged_tokens?.length ? `<div class="flags">Flagged: ${{it.flagged_tokens.join(', ')}}</div>` : ''}}
                </div>
            </div>
            <div class="panel" style="margin-top:16px">
                <h3>Ground truth (edit & copy to phase3_ground_truth.json)</h3>
                <textarea id="gtEdit">${{gt}}</textarea>
                <p style="font-size:0.8rem;color:var(--muted);margin-top:8px">Key: <code>${{it.crop_id}}</code></p>
            </div>
            <div class="gate-box" id="signalsBox"></div>
        `;
        if (window.MathJax?.typesetPromise) MathJax.typesetPromise();
        const sig = it.confidence_signals || {{}};
        document.getElementById('signalsBox').innerHTML = '<strong>Objective signals:</strong> ' + JSON.stringify(sig);
    }}

    renderGateBadge();
    renderSidebar();
    if (items.length) selectItem(0);
    </script>
</body>
</html>"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate Phase 3 transcription review HTML")
    parser.add_argument(
        "--booklets",
        type=str,
        default=None,
        help="Comma-separated booklet IDs to include (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_HTML),
        help="Output HTML path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Keep only this model (e.g. gemini-3.6-flash)",
    )
    args = parser.parse_args()

    booklet_ids = (
        [b.strip() for b in args.booklets.split(",") if b.strip()] if args.booklets else None
    )
    items, summary, gt = load_items(booklet_ids)
    if args.model:
        items = [it for it in items if it.get("model") == args.model]
        summary = {**summary, "models": [args.model], "filter_model": args.model}

    out = Path(args.output)
    html = generate_html(items, summary, gt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(items)} items)")


if __name__ == "__main__":
    main()
