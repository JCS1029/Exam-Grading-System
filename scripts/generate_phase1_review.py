"""
Generate Phase 1 before/after review HTML (same role as phase0_review.html).

Writes a standalone inspector to storage/reports/phase1_review.html that loads
raw vs preprocessed pages via relative paths (no embedded images).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STORAGE_ROOT = Path("storage")
RAW_ROOT = STORAGE_ROOT / "raw"
PREP_ROOT = STORAGE_ROOT / "preprocessed"
OUTPUT_HTML = STORAGE_ROOT / "reports" / "phase1_review.html"


def collect_pages() -> list[dict]:
    pages: list[dict] = []
    if not RAW_ROOT.is_dir():
        return pages

    for booklet_dir in sorted(RAW_ROOT.iterdir()):
        if not booklet_dir.is_dir():
            continue
        booklet_id = booklet_dir.name
        for raw_path in sorted(booklet_dir.glob("page_*.png")):
            page_name = raw_path.name
            prep_path = PREP_ROOT / booklet_id / page_name
            metrics_path = PREP_ROOT / booklet_id / page_name.replace(".png", ".metrics.json")
            metrics: dict = {}
            fixing_score = 0.0
            clamped = False
            if metrics_path.exists():
                try:
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                    metrics = payload.get("metrics") or {}
                    fixing_score = float(payload.get("fixing_score") or 0.0)
                    clamped = bool(payload.get("white_point_clamped") or metrics.get("white_point_clamped"))
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass

            pages.append({
                "booklet_id": booklet_id,
                "page_name": page_name,
                "page_label": page_name.replace(".png", ""),
                "raw_src": f"../raw/{booklet_id}/{page_name}",
                "prep_src": f"../preprocessed/{booklet_id}/{page_name}" if prep_path.exists() else "",
                "has_prep": prep_path.exists(),
                "fixing_score": round(fixing_score, 1),
                "clamped": clamped,
                "metrics": metrics,
            })
    return pages


def generate_html(pages: list[dict], output_path: Path) -> None:
    pages_json = json.dumps(pages, ensure_ascii=False)
    n = len(pages)
    n_clamped = sum(1 for p in pages if p.get("clamped"))
    booklets = sorted({p["booklet_id"] for p in pages})

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 1 Preprocess Review — Before / After</title>
    <link href="https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #0d1117;
            --bg-card: #161b22;
            --border-color: #30363d;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-orange: #d29922;
            --accent-red: #f85149;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-main);
            color: var(--text-primary);
            font-family: 'Assistant', -apple-system, BlinkMacSystemFont, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        header {{
            background: #161b22;
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .badge {{
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 12px;
            background: rgba(88, 166, 255, 0.2);
            color: var(--accent-blue);
            border: 1px solid var(--accent-blue);
            font-family: 'JetBrains Mono', monospace;
        }}
        .badge-green {{
            background: rgba(63, 185, 80, 0.15);
            color: var(--accent-green);
            border-color: var(--accent-green);
        }}
        .container {{ flex: 1; display: flex; overflow: hidden; }}
        .sidebar {{
            width: 320px;
            background: #161b22;
            border-left: 1px solid var(--border-color);
            overflow-y: auto;
            padding: 16px;
        }}
        .page-item {{
            background: #21262d;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .page-item:hover, .page-item.active {{
            border-color: var(--accent-blue);
            background: #282e38;
        }}
        .page-item-title {{ font-weight: 600; color: #fff; margin-bottom: 4px; }}
        .page-item-desc {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }}
        .main-view {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .toolbar {{
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
        }}
        .toolbar-info {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
        .btn {{
            padding: 6px 12px;
            background: #21262d;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            font-family: 'Assistant', sans-serif;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .btn.active {{
            background: var(--accent-blue);
            color: #fff;
            border-color: var(--accent-blue);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            padding: 12px 16px;
            background: #0d1117;
            border-bottom: 1px solid var(--border-color);
        }}
        .metric-card {{
            background: var(--bg-card);
            padding: 10px 12px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .metric-card .val {{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--accent-green);
            font-family: 'JetBrains Mono', monospace;
        }}
        .metric-card .lbl {{ font-size: 0.75rem; color: var(--text-secondary); }}
        .reason {{
            padding: 8px 16px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
        }}
        .panes {{
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            overflow: hidden;
            direction: ltr;
        }}
        .panes.overlay-mode {{ grid-template-columns: 1fr; }}
        .pane {{
            display: flex;
            flex-direction: column;
            overflow: hidden;
            border-right: 1px solid var(--border-color);
            background: #010409;
        }}
        .pane-header {{
            padding: 8px 14px;
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
        }}
        .pane-content {{
            flex: 1;
            overflow: auto;
            padding: 16px;
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }}
        .pane-content img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #30363d;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        }}
        .overlay-wrap {{
            position: relative;
            max-width: 100%;
            line-height: 0;
        }}
        .overlay-wrap img {{
            display: block;
            max-width: 100%;
            height: auto;
        }}
        .overlay-wrap .after-clip {{
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            overflow: hidden;
            border-right: 2px solid var(--accent-blue);
        }}
        .overlay-wrap .after-clip img {{
            max-width: none;
        }}
        .manual-notes {{
            background: #1c2128;
            border-top: 1px solid var(--accent-orange);
            padding: 12px 16px;
        }}
        .manual-notes h4 {{ color: var(--accent-orange); margin-bottom: 6px; }}
        textarea {{
            width: 100%;
            height: 64px;
            background: #0d1117;
            border: 1px solid #30363d;
            color: #fff;
            padding: 8px;
            border-radius: 6px;
            font-family: 'Assistant', sans-serif;
            resize: vertical;
        }}
        .save-btn {{
            background: var(--accent-green);
            color: #fff;
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 8px;
        }}
    </style>
</head>
<body>
<header>
    <div class="header-title">
        <span>מערכת בדיקת מבחנים — ניקוי סריקות (Phase 1)</span>
        <span class="badge">Before / After Inspector</span>
        <span class="badge badge-green">White-point clamp + CLAHE</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-secondary);">
        {n} pages · {len(booklets)} booklets · {n_clamped} clamped · {", ".join(booklets) or "none"}
    </div>
</header>

<div class="container">
    <div class="sidebar" id="sidebar"></div>
    <div class="main-view">
        <div class="toolbar">
            <div class="toolbar-info">
                <span id="activeLabel" style="font-weight:700;color:#fff;">בחר עמוד</span>
                <span class="badge" id="clampBadge">clamp --</span>
            </div>
            <div>
                <button class="btn active" id="btnSide" onclick="setMode('side')">לפני / אחרי</button>
                <button class="btn" id="btnOverlay" onclick="setMode('overlay')">סליידר</button>
                <button class="btn" onclick="syncReset()">איפוס גלילה</button>
            </div>
        </div>
        <div class="metrics-grid">
            <div class="metric-card"><div class="val" id="mClamp">--</div><div class="lbl">פיקסלים שנמחקו (phantom / bleed)</div></div>
            <div class="metric-card"><div class="val" id="mThr">--</div><div class="lbl">סף לבן (threshold)</div></div>
            <div class="metric-card"><div class="val" id="mFaint">--</div><div class="lbl">יחס כתב חלש (faint strokes)</div></div>
            <div class="metric-card"><div class="val" id="mFix">--</div><div class="lbl">Fixing score</div></div>
            <div class="metric-card"><div class="val" id="mPaper">--</div><div class="lbl">Paper white</div></div>
        </div>
        <div class="reason" id="reasonText">השווה כתב רפאים בסריקה המקורית מול העמוד אחרי ה-clamp.</div>
        <div class="panes" id="panes">
            <div class="pane" id="rawPane">
                <div class="pane-header"><span>BEFORE — RAW</span><span>סריקה מקורית</span></div>
                <div class="pane-content" id="rawScroll"><img id="rawImg" alt="Raw scan"></div>
            </div>
            <div class="pane" id="prepPane">
                <div class="pane-header"><span>AFTER — PREPROCESSED</span><span>Deskew + clamp + CLAHE</span></div>
                <div class="pane-content" id="prepScroll"><img id="prepImg" alt="Preprocessed scan"></div>
            </div>
            <div class="pane" id="overlayPane" style="display:none;">
                <div class="pane-header">
                    <span>OVERLAY SLIDER</span>
                    <label style="font-weight:400;">גרור · <input id="splitRange" type="range" min="0" max="100" value="50" oninput="setSplit(this.value)"></label>
                </div>
                <div class="pane-content">
                    <div class="overlay-wrap" id="overlayWrap">
                        <img id="overlayBefore" alt="Before">
                        <div class="after-clip" id="afterClip"><img id="overlayAfter" alt="After"></div>
                    </div>
                </div>
            </div>
        </div>
        <div class="manual-notes">
            <h4>הערות בדיקה ידנית (Manual Quality Check)</h4>
            <p style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:8px;">
                חפש כתב רפאים (bleed-through) בצד BEFORE — אמור להיעלם ב-AFTER בלי למחוק עיפרון אמיתי.
            </p>
            <textarea id="manualCheckInput" placeholder="רשום אם ה-clamp מחק כתב אמיתי, או אם נשאר כתב רפאים..."></textarea>
            <button class="save-btn" onclick="saveManualReview()">שמור הערה</button>
        </div>
    </div>
</div>

<script>
    const pages = {pages_json};
    let idx = 0;
    let mode = 'side';

    function pct(v) {{
        return ((Number(v) || 0) * 100).toFixed(1) + '%';
    }}

    function init() {{
        renderSidebar();
        const raw = document.getElementById('rawScroll');
        const prep = document.getElementById('prepScroll');
        raw.addEventListener('scroll', () => {{
            if (mode !== 'side') return;
            prep.scrollTop = raw.scrollTop;
            prep.scrollLeft = raw.scrollLeft;
        }});
        if (pages.length) loadItem(0);
    }}

    function renderSidebar() {{
        const sidebar = document.getElementById('sidebar');
        sidebar.innerHTML = '';
        pages.forEach((p, i) => {{
            const div = document.createElement('div');
            div.className = 'page-item' + (i === idx ? ' active' : '');
            div.onclick = () => loadItem(i);
            const clamp = pct(p.metrics && p.metrics.clamped_frac);
            div.innerHTML = `
                <div class="page-item-title">חוברת ${{p.booklet_id}} · ${{p.page_label}}</div>
                <div class="page-item-desc">clamp ${{clamp}} · fix ${{p.fixing_score}}</div>
            `;
            sidebar.appendChild(div);
        }});
    }}

    function loadItem(i) {{
        idx = i;
        renderSidebar();
        const p = pages[i];
        const m = p.metrics || {{}};
        document.getElementById('activeLabel').innerText = 'חוברת ' + p.booklet_id + ' — ' + p.page_label;
        document.getElementById('clampBadge').innerText = p.clamped ? ('clamped ' + pct(m.clamped_frac)) : 'no clamp';
        document.getElementById('mClamp').innerText = pct(m.clamped_frac);
        document.getElementById('mThr').innerText = (m.white_point_threshold != null ? m.white_point_threshold : '--');
        document.getElementById('mFaint').innerText = pct(m.faint_ratio);
        document.getElementById('mFix').innerText = p.fixing_score;
        document.getElementById('mPaper').innerText = (m.paper_white != null ? m.paper_white : '--');
        document.getElementById('reasonText').innerText = p.clamped
            ? ('White-point clamp crushed ' + pct(m.clamped_frac) + ' of pixels (paper ' + m.paper_white + ', thr ' + m.white_point_threshold + ') before CLAHE.')
            : 'No white-point clamp recorded for this page.';

        document.getElementById('rawImg').src = p.raw_src;
        document.getElementById('prepImg').src = p.prep_src || p.raw_src;
        document.getElementById('overlayBefore').src = p.raw_src;
        document.getElementById('overlayAfter').src = p.prep_src || p.raw_src;
        syncReset();
        setSplit(document.getElementById('splitRange').value);
    }}

    function setMode(next) {{
        mode = next;
        document.getElementById('btnSide').classList.toggle('active', next === 'side');
        document.getElementById('btnOverlay').classList.toggle('active', next === 'overlay');
        document.getElementById('rawPane').style.display = next === 'side' ? 'flex' : 'none';
        document.getElementById('prepPane').style.display = next === 'side' ? 'flex' : 'none';
        document.getElementById('overlayPane').style.display = next === 'overlay' ? 'flex' : 'none';
        document.getElementById('panes').classList.toggle('overlay-mode', next === 'overlay');
        if (next === 'overlay') setSplit(document.getElementById('splitRange').value);
    }}

    function setSplit(val) {{
        const clip = document.getElementById('afterClip');
        const wrap = document.getElementById('overlayWrap');
        const before = document.getElementById('overlayBefore');
        const after = document.getElementById('overlayAfter');
        clip.style.width = val + '%';
        after.style.width = before.clientWidth + 'px';
        wrap.style.width = before.clientWidth + 'px';
    }}

    function syncReset() {{
        document.getElementById('rawScroll').scrollTop = 0;
        document.getElementById('prepScroll').scrollTop = 0;
    }}

    function saveManualReview() {{
        const text = document.getElementById('manualCheckInput').value;
        if (!text.trim()) return;
        const p = pages[idx];
        alert('נשמרה הערה עבור ' + p.booklet_id + ' / ' + p.page_label + ':\\n' + text);
        document.getElementById('manualCheckInput').value = '';
    }}

    window.onload = init;
    window.addEventListener('resize', () => {{
        if (mode === 'overlay') setSplit(document.getElementById('splitRange').value);
    }});
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Generated Phase 1 review HTML at: {output_path}")


if __name__ == "__main__":
    pages = collect_pages()
    if not pages:
        print("No pages found under storage/raw/. Run Phase 1 first.")
        sys.exit(1)
    generate_html(pages, OUTPUT_HTML)
    print(f"Pages: {len(pages)}")
