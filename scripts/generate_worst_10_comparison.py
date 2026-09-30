"""
Generate Phase 1 Interactive Visual Comparison UI for Worst Dataset Renders.

Compares Before (Raw Ingestion) vs After (Preprocessed / Anonymized) for the 10
dataset pages with the highest fixing score (most fixing needed / done).

Ranks dynamically from ``*.metrics.json`` sidecars written by the preprocessor
(white-point clamp + CLAHE + deskew).

Features:
  - Ranked Top 10 Worst Scans with fixing scores and breakdown
  - Side-by-side Before (Raw Ingestion) vs After (Preprocessed) vs Anonymized
  - Metric chips: contrast, faint strokes, lighting, white-point clamp
  - Interactive page switcher and comparison modes
"""

from __future__ import annotations

import base64
import io
import json
import sqlite3
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STORAGE_ROOT = Path("storage")
REPORTS_DIR = STORAGE_ROOT / "reports"
PSEUDONYM_DB = STORAGE_ROOT / "pseudonym_map.db"
OUTPUT_HTML = REPORTS_DIR / "phase1_worst_10_comparison.html"
TOP_N = 10


def _reason_from_metrics(m: dict) -> str:
    parts = []
    if m.get("white_point_clamped"):
        parts.append(
            f"White-point clamp crushed {100.0 * float(m.get('clamped_frac', 0.0)):.1f}% "
            f"of pixels (paper {m.get('paper_white', 0):.0f}, thr {m.get('white_point_threshold', 0):.0f}) "
            f"to kill bleed-through before CLAHE."
        )
    faint = float(m.get("faint_ratio", 0.0))
    if faint >= 0.30:
        parts.append(f"Elevated faint-stroke ratio ({100.0 * faint:.1f}%).")
    spread = float(m.get("lighting_spread", 0.0))
    if spread >= 6.0:
        parts.append(f"Non-uniform illumination (spread {spread:.2f}).")
    std0 = float(m.get("orig_contrast_std", 0.0))
    if std0 and std0 < 32.0:
        parts.append(f"Low baseline contrast (std {std0:.2f}).")
    if not parts:
        parts.append("Highest composite fixing score in this batch.")
    return " ".join(parts)


def get_pseudonym_map() -> dict[str, str]:
    if not PSEUDONYM_DB.exists():
        return {}
    conn = sqlite3.connect(PSEUDONYM_DB)
    rows = conn.execute("SELECT booklet_id, pseudonym FROM pseudonym_map").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}

def image_to_base64_thumbnail(image_path: Path, max_dim: int = 1400) -> str:
    """Read image, downsample to web-friendly resolution, encode to base64 JPEG."""
    if not image_path.exists():
        return ""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

def load_page_records() -> list[dict]:
    """Load every preprocessed page that has a metrics sidecar."""
    records: list[dict] = []
    prep_root = STORAGE_ROOT / "preprocessed"
    if not prep_root.is_dir():
        return records

    for metrics_path in sorted(prep_root.glob("*/*.metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        booklet_id = payload.get("booklet_id") or metrics_path.parent.name
        page_name = metrics_path.name.replace(".metrics.json", ".png")
        metrics = payload.get("metrics") or {}
        records.append({
            "booklet_id": booklet_id,
            "page_name": page_name,
            "fixing_score": float(payload.get("fixing_score") or 0.0),
            "metrics": metrics,
            "reason": _reason_from_metrics(metrics),
        })
    return records


def prepare_worst_10_pages() -> tuple[list[dict], int]:
    p_map = get_pseudonym_map()
    records = load_page_records()
    records.sort(key=lambda r: r["fixing_score"], reverse=True)
    total = len(records)
    pages = []

    for rank, item in enumerate(records[:TOP_N], start=1):
        b_id = item["booklet_id"]
        p_name = item["page_name"]
        raw_f = STORAGE_ROOT / "raw" / b_id / p_name
        prep_f = STORAGE_ROOT / "preprocessed" / b_id / p_name
        anon_f = STORAGE_ROOT / "anonymized" / b_id / p_name

        raw_b64 = image_to_base64_thumbnail(raw_f)
        prep_b64 = image_to_base64_thumbnail(prep_f) if prep_f.exists() else raw_b64
        anon_b64 = image_to_base64_thumbnail(anon_f) if anon_f.exists() else prep_b64

        pages.append({
            "rank": rank,
            "booklet_id": b_id,
            "page_name": p_name,
            "label": f"#{rank} — Booklet {b_id} ({p_name.replace('.png', '')})",
            "pseudonym": p_map.get(b_id, f"STUDENT_{b_id}"),
            "fixing_score": item["fixing_score"],
            "metrics": item["metrics"],
            "reason": item["reason"],
            "is_page1": ("001" in p_name),
            "raw_b64": raw_b64,
            "prep_b64": prep_b64,
            "anon_b64": anon_b64,
        })
    return pages, total

def generate_html(pages: list[dict], output_path: Path, total_analyzed: int = 0):
    pages_json = json.dumps(pages)
    analyzed_label = f"{total_analyzed} Pages Analyzed"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 1 Quality Assessment — Top 10 Most Fixed Renders</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0a0d12;
            --bg-card: #121820;
            --bg-sidebar: #0e1319;
            --border: #232d3b;
            --border-highlight: #3b82f6;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-purple: #a855f7;
            --accent-yellow: #fbbf24;
            --accent-red: #f87171;
            --text-main: #f1f5f9;
            --text-sub: #94a3b8;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        header {{
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .logo-group {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
        }}
        .badge-red {{
            background: rgba(248, 113, 113, 0.15);
            color: var(--accent-red);
            border: 1px solid rgba(248, 113, 113, 0.35);
        }}
        .badge-green {{
            background: rgba(52, 211, 153, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(52, 211, 153, 0.3);
        }}
        .badge-blue {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}
        .badge-purple {{
            background: rgba(168, 85, 247, 0.15);
            color: var(--accent-purple);
            border: 1px solid rgba(168, 85, 247, 0.3);
        }}
        .header-stats {{
            display: flex;
            gap: 20px;
            font-size: 0.85rem;
            color: var(--text-sub);
        }}
        .header-stats span strong {{
            color: var(--text-main);
            font-family: 'JetBrains Mono', monospace;
        }}
        .app-body {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}
        /* Sidebar */
        .sidebar {{
            width: 340px;
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .sidebar-header {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .page-list {{
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .page-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.18s ease;
        }}
        .page-card:hover {{
            border-color: var(--accent-blue);
            background: #182230;
        }}
        .page-card.active {{
            border-color: var(--border-highlight);
            background: #16243a;
            box-shadow: 0 0 0 1px var(--border-highlight);
        }}
        .page-card-title {{
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }}
        .page-card-score {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 0.85rem;
            color: #f87171;
            background: rgba(239, 68, 68, 0.15);
            padding: 2px 6px;
            border-radius: 4px;
        }}
        .page-card-sub {{
            font-size: 0.78rem;
            color: var(--text-sub);
            font-family: 'JetBrains Mono', monospace;
        }}
        /* Main Workspace */
        .workspace {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background: #07090d;
        }}
        .workspace-toolbar {{
            padding: 10px 24px;
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .toolbar-info {{
            font-size: 0.9rem;
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .toolbar-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .btn {{
            padding: 6px 14px;
            background: #1e293b;
            color: var(--text-main);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.82rem;
            font-weight: 500;
            transition: all 0.15s ease;
        }}
        .btn:hover {{
            background: #2b394e;
            border-color: var(--text-sub);
        }}
        .btn.active {{
            background: var(--border-highlight);
            border-color: var(--border-highlight);
            color: #fff;
        }}
        /* Metric Detail Bar */
        .metric-bar {{
            background: #0d131a;
            border-bottom: 1px solid var(--border);
            padding: 8px 24px;
            display: flex;
            gap: 16px;
            align-items: center;
            font-size: 0.8rem;
            color: var(--text-sub);
        }}
        .metric-chip {{
            background: #16202c;
            border: 1px solid var(--border);
            padding: 3px 8px;
            border-radius: 4px;
            display: inline-flex;
            gap: 6px;
            align-items: center;
        }}
        .metric-chip strong {{
            color: var(--accent-blue);
            font-family: 'JetBrains Mono', monospace;
        }}
        /* Comparison Panes */
        .comparison-grid {{
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            overflow: hidden;
            position: relative;
        }}
        .comparison-grid.three-panes {{
            grid-template-columns: 1fr 1fr 1fr;
        }}
        .pane {{
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border);
            overflow: hidden;
            background: #03060a;
        }}
        .pane:last-child {{
            border-right: none;
        }}
        .pane-header {{
            background: #111721;
            padding: 10px 16px;
            font-size: 0.82rem;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }}
        .pane-content {{
            flex: 1;
            overflow: auto;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding: 24px;
        }}
        .pane-content img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.8);
            border: 1px solid #1e293b;
        }}
        .mask-legend {{
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
        }}
    </style>
</head>
<body>

<header>
    <div class="logo-group">
        <h2 style="font-size: 1.15rem; font-weight: 700;">Phase 1 Image Quality Assessment</h2>
        <span class="badge badge-red">Worst 10 Renders Focus</span>
        <span class="badge badge-blue">White-Point Clamp + CLAHE</span>
        <span class="badge badge-green">{analyzed_label}</span>
    </div>
    <div class="header-stats">
        <span>Dataset: <strong>docs/Dataset/</strong></span>
        <span>Bleed-through: <strong>Crush gray ≥ paper-white band before CLAHE</strong></span>
        <span>Storage: <strong>storage/preprocessed/</strong></span>
    </div>
</header>

<div class="app-body">
    <!-- Sidebar -->
    <aside class="sidebar">
        <div class="sidebar-header">
            <span>Ranked by Fixing Need</span>
            <span class="badge badge-red">Top 10</span>
        </div>
        <div class="page-list" id="pageList"></div>
    </aside>

    <!-- Workspace -->
    <main class="workspace">
        <div class="workspace-toolbar">
            <div class="toolbar-info">
                <span id="activePageLabel" style="font-weight: 600; font-size: 1rem;">Select Page</span>
                <span id="activeScoreBadge" class="badge badge-red">Fixing Score: --</span>
                <span id="activePseudonymBadge" class="badge badge-purple">Pseudonym: --</span>
                <span id="maskAlert" class="mask-legend" style="display: none;">Privacy Mask Applied (Top 15%)</span>
            </div>
            <div class="toolbar-controls">
                <button class="btn active" id="btnSideBySide" onclick="setViewMode('side-by-side')">Before vs After</button>
                <button class="btn" id="btnThreeWay" onclick="setViewMode('three-way')">3-Way (Raw / Clean / Anon)</button>
                <button class="btn" onclick="zoomFit()">Reset Scroll</button>
            </div>
        </div>

        <div class="metric-bar" id="metricBar">
            <span>Assessment: <span id="reasonText" style="color: var(--text-main); font-weight: 500;">--</span></span>
        </div>

        <div class="comparison-grid" id="comparisonGrid">
            <!-- Pane 1: Raw Ingestion -->
            <div class="pane">
                <div class="pane-header">
                    <span>1. BEFORE (RAW SCAN)</span>
                    <span style="color: var(--accent-red); font-size: 0.75rem;">Faint Strokes / Shadow</span>
                </div>
                <div class="pane-content" id="rawPane">
                    <img id="rawImg" src="" alt="Raw Scan">
                </div>
            </div>

            <!-- Pane 2: Preprocessed (Deskewed + Contrast) -->
            <div class="pane" id="prepPaneContainer">
                <div class="pane-header">
                    <span>2. AFTER (PREPROCESSED)</span>
                    <span style="color: var(--accent-blue); font-size: 0.75rem;">White-Point Clamp + CLAHE</span>
                </div>
                <div class="pane-content" id="prepPane">
                    <img id="prepImg" src="" alt="Preprocessed Scan">
                </div>
            </div>

            <!-- Pane 3: Anonymized -->
            <div class="pane" id="anonPaneContainer" style="display: none;">
                <div class="pane-header">
                    <span>3. ANONYMIZED (VLM INPUT)</span>
                    <span style="color: var(--accent-green); font-size: 0.75rem;">Masked & Pseudonymized</span>
                </div>
                <div class="pane-content" id="anonPane">
                    <img id="anonImg" src="" alt="Anonymized Scan">
                </div>
            </div>
        </div>
    </main>
</div>

<script>
    const pages = {pages_json};
    let activeIdx = 0;
    let viewMode = 'side-by-side';

    function init() {{
        renderPageList();
        if (pages.length > 0) {{
            selectPage(0);
        }}
    }}

    function renderPageList() {{
        const container = document.getElementById('pageList');
        container.innerHTML = '';
        pages.forEach((p, idx) => {{
            const card = document.createElement('div');
            card.className = 'page-card' + (idx === activeIdx ? ' active' : '');
            card.onclick = () => selectPage(idx);
            
            card.innerHTML = `
                <div class="page-card-title">
                    <span>${{p.label}}</span>
                    <span class="page-card-score">${{p.fixing_score}}</span>
                </div>
                <div class="page-card-sub">ID: ${{p.pseudonym}} &bull; Faint: ${{Math.round((p.metrics.faint_ratio || 0) * 100)}}% &bull; Clamp: ${{((p.metrics.clamped_frac || 0) * 100).toFixed(1)}}%</div>
            `;
            container.appendChild(card);
        }});
    }}

    function selectPage(idx) {{
        activeIdx = idx;
        const page = pages[idx];

        document.querySelectorAll('.page-card').forEach((el, i) => {{
            el.classList.toggle('active', i === idx);
        }});

        document.getElementById('activePageLabel').innerText = page.label;
        document.getElementById('activeScoreBadge').innerText = 'Fixing Score: ' + page.fixing_score + ' / 100';
        document.getElementById('activePseudonymBadge').innerText = 'Assigned: ' + page.pseudonym;
        document.getElementById('maskAlert').style.display = page.is_page1 ? 'inline-block' : 'none';
        const m = page.metrics || {{}};
        const faintPct = ((m.faint_ratio || 0) * 100).toFixed(1);
        const clampPct = ((m.clamped_frac || 0) * 100).toFixed(1);
        document.getElementById('reasonText').innerText = page.reason
            + ' (Faint ' + faintPct + '%, lighting spread ' + (m.lighting_spread || 0)
            + ', paper white ' + (m.paper_white || '--')
            + ', clamp thr ' + (m.white_point_threshold || '--')
            + ', crushed ' + clampPct + '%)';

        document.getElementById('rawImg').src = page.raw_b64;
        document.getElementById('prepImg').src = page.prep_b64;
        document.getElementById('anonImg').src = page.anon_b64;
    }}

    function setViewMode(mode) {{
        viewMode = mode;
        const grid = document.getElementById('comparisonGrid');
        const anonContainer = document.getElementById('anonPaneContainer');
        const prepContainer = document.getElementById('prepPaneContainer');
        const btnSide = document.getElementById('btnSideBySide');
        const btnThree = document.getElementById('btnThreeWay');

        if (mode === 'three-way') {{
            grid.classList.add('three-panes');
            anonContainer.style.display = 'flex';
            prepContainer.style.display = 'flex';
            btnThree.classList.add('active');
            btnSide.classList.remove('active');
        }} else {{
            grid.classList.remove('three-panes');
            anonContainer.style.display = 'none';
            prepContainer.style.display = 'flex';
            btnSide.classList.add('active');
            btnThree.classList.remove('active');
        }}
    }}

    function zoomFit() {{
        document.getElementById('rawPane').scrollTop = 0;
        document.getElementById('prepPane').scrollTop = 0;
        document.getElementById('anonPane').scrollTop = 0;
    }}

    window.onload = init;
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated worst 10 comparison UI report at: {output_path}")

if __name__ == "__main__":
    print("Preparing worst 10 dataset comparison pages...")
    pages, total = prepare_worst_10_pages()
    if not pages:
        print("No preprocessed metrics found under storage/preprocessed/. Run Phase 1 first.")
        sys.exit(1)
    generate_html(pages, OUTPUT_HTML, total_analyzed=total)
