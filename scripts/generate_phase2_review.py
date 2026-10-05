"""
Generate Phase 2 layout review HTML (sidebar + page overlay of columns/questions).

Loads page images and crop overlays via relative paths from storage/reports/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STORAGE_ROOT = Path("storage")
CROPS_ROOT = STORAGE_ROOT / "crops"
ANON_ROOT = STORAGE_ROOT / "anonymized"
PREP_ROOT = STORAGE_ROOT / "preprocessed"
OUTPUT_HTML = STORAGE_ROOT / "reports" / "phase2_review.html"
RESULTS_JSON = STORAGE_ROOT / "reports" / "phase2_results.json"


def collect_pages() -> list[dict]:
    pages: list[dict] = []
    if RESULTS_JSON.exists():
        payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        records = payload.get("pages") or []
    else:
        records = []
        for manifest in sorted(CROPS_ROOT.glob("*/*/manifest.json")):
            data = json.loads(manifest.read_text(encoding="utf-8"))
            records.append({
                "booklet_id": data["booklet_id"],
                "page_name": data["page_name"],
                "n_columns": len(data.get("columns") or []),
                "n_questions": len(data.get("questions") or []),
                "n_tables": len(data.get("tables") or []),
                "is_multi_column": data.get("is_multi_column", False),
                "coverage": data.get("coverage") or {},
                "halted": data.get("halted", False),
                "halt_reasons": data.get("halt_reasons") or [],
                "metrics": {},
                "output_dir": str(manifest.parent),
            })

    for rec in records:
        bid = rec["booklet_id"]
        page_name = rec["page_name"]
        stem = page_name.replace(".png", "")
        manifest_path = CROPS_ROOT / bid / stem / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        src = ANON_ROOT / bid / page_name
        rel_src = f"../anonymized/{bid}/{page_name}"
        if not src.exists():
            rel_src = f"../preprocessed/{bid}/{page_name}"

        pages.append({
            **rec,
            "method": rec.get("method") or manifest.get("method") or "?",
            "page_src": rel_src,
            "manifest": manifest,
            "crop_dir": f"../crops/{bid}/{stem}",
            "coverage": rec.get("coverage") or manifest.get("coverage") or {},
            "n_columns": rec.get("n_columns") or len(manifest.get("columns") or []) or 1,
            "n_questions": rec.get("n_questions") or len(manifest.get("questions") or []),
            "n_tables": rec.get("n_tables") or len(manifest.get("tables") or []),
        })
    return pages


def generate_html(pages: list[dict], output_path: Path) -> None:
    pages_json = json.dumps(pages, ensure_ascii=False)
    n = len(pages)
    multi = sum(1 for p in pages if p.get("is_multi_column"))
    halted = sum(1 for p in pages if p.get("halted"))

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 2 Layout Review — Columns / Questions / Crops</title>
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
            --accent-purple: #a371f7;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-main);
            color: var(--text-primary);
            font-family: 'Assistant', sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        header {{
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title {{
            font-size: 1.2rem;
            font-weight: 700;
            color: #fff;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .badge {{
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 12px;
            border: 1px solid var(--accent-blue);
            color: var(--accent-blue);
            background: rgba(88,166,255,0.15);
            font-family: 'JetBrains Mono', monospace;
        }}
        .container {{ flex: 1; display: flex; overflow: hidden; }}
        .sidebar {{
            width: 300px;
            background: var(--bg-card);
            border-left: 1px solid var(--border-color);
            overflow-y: auto;
            padding: 14px;
        }}
        .page-item {{
            background: #21262d;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            cursor: pointer;
        }}
        .page-item.active, .page-item:hover {{ border-color: var(--accent-blue); }}
        .page-item-title {{ font-weight: 700; color: #fff; }}
        .page-item-desc {{
            font-size: 0.78rem;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            margin-top: 4px;
        }}
        .main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
        .toolbar {{
            padding: 10px 16px;
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
            padding: 10px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        .metric {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 10px;
        }}
        .metric .val {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--accent-green);
        }}
        .metric .lbl {{ font-size: 0.72rem; color: var(--text-secondary); }}
        .workspace {{
            flex: 1;
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            overflow: hidden;
            direction: ltr;
        }}
        .canvas-wrap {{
            overflow: auto;
            background: #010409;
            padding: 16px;
            border-right: 1px solid var(--border-color);
            position: relative;
        }}
        .stage {{
            position: relative;
            display: inline-block;
            max-width: 100%;
            line-height: 0;
        }}
        .stage img.page {{
            max-width: 100%;
            height: auto;
            border: 1px solid #30363d;
        }}
        .overlay {{
            position: absolute;
            border: 2px solid var(--accent-blue);
            background: rgba(88,166,255,0.12);
            pointer-events: none;
        }}
        .overlay.question {{ border-color: var(--accent-green); background: rgba(63,185,80,0.12); }}
        .overlay.table {{ border-color: var(--accent-orange); background: rgba(210,153,34,0.15); }}
        .overlay.gutter {{ border-color: var(--accent-purple); background: rgba(163,113,247,0.2); }}
        .overlay .tag {{
            position: absolute;
            top: 0;
            left: 0;
            transform: translateY(-100%);
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            background: #161b22;
            color: #fff;
            padding: 1px 5px;
            border-radius: 3px;
            white-space: nowrap;
            line-height: 1.4;
        }}
        .crops {{
            overflow-y: auto;
            padding: 12px;
            background: var(--bg-main);
        }}
        .crop-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }}
        .crop-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #30363d;
            margin-top: 6px;
        }}
        .crop-title {{ font-weight: 700; font-size: 0.9rem; }}
        .crop-meta {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }}
        .btn {{
            background: #21262d;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }}
        .btn.active {{ background: var(--accent-blue); color: #fff; border-color: var(--accent-blue); }}
        .halt {{ color: var(--accent-red); font-weight: 700; }}
    </style>
</head>
<body>
<header>
    <div class="header-title">
        <span>Phase 2 — ניתוח פריסה וחיתוך שאלות</span>
        <span class="badge">Layout / Columns / Crops</span>
    </div>
    <div style="font-size:0.85rem;color:var(--text-secondary);">
        {n} pages · {multi} multi-column · {halted} halted
    </div>
</header>
<div class="container">
    <div class="sidebar" id="sidebar"></div>
    <div class="main">
        <div class="toolbar">
            <div>
                <span id="activeLabel" style="font-weight:700;color:#fff;">בחר עמוד</span>
                <span id="haltBadge" class="halt" style="margin-right:10px;display:none;">HALTED</span>
            </div>
            <div>
                <button class="btn active" id="btnAll" onclick="setShow('all')">הכל</button>
                <button class="btn" id="btnCols" onclick="setShow('columns')">עמודות</button>
                <button class="btn" id="btnQs" onclick="setShow('questions')">שאלות</button>
                <button class="btn" id="btnTabs" onclick="setShow('tables')">טבלאות</button>
            </div>
        </div>
        <div class="metrics">
            <div class="metric"><div class="val" id="mCols">--</div><div class="lbl">עמודות</div></div>
            <div class="metric"><div class="val" id="mQs">--</div><div class="lbl">אזורי שאלה</div></div>
            <div class="metric"><div class="val" id="mTabs">--</div><div class="lbl">טבלאות</div></div>
            <div class="metric"><div class="val" id="mAns">--</div><div class="lbl">answered / blank / not_found</div></div>
            <div class="metric"><div class="val" id="mConf">--</div><div class="lbl">gutter confidence</div></div>
        </div>
        <div class="workspace">
            <div class="canvas-wrap">
                <div class="stage" id="stage">
                    <img class="page" id="pageImg" alt="page">
                    <div id="overlays"></div>
                </div>
            </div>
            <div class="crops" id="crops"></div>
        </div>
    </div>
</div>
<script>
    const pages = {pages_json};
    let idx = 0;
    let show = 'all';

    function init() {{
        renderSidebar();
        if (pages.length) loadItem(0);
        document.getElementById('pageImg').onload = redrawOverlays;
        window.addEventListener('resize', redrawOverlays);
    }}

    function renderSidebar() {{
        const sb = document.getElementById('sidebar');
        sb.innerHTML = '';
        pages.forEach((p, i) => {{
            const d = document.createElement('div');
            d.className = 'page-item' + (i === idx ? ' active' : '');
            d.onclick = () => loadItem(i);
            d.innerHTML = `
                <div class="page-item-title">חוברת ${{p.booklet_id}} · ${{p.page_name.replace('.png','')}}</div>
                <div class="page-item-desc">${{p.method || (p.manifest && p.manifest.method) || '?'}} · cols ${{p.n_columns}} · Q [${{Object.keys(p.coverage||{{}}).join(', ')}}]${{p.halted ? ' · HALT' : ''}}</div>
            `;
            sb.appendChild(d);
        }});
    }}

    function covCounts(cov) {{
        let a=0,b=0,n=0;
        Object.values(cov || {{}}).forEach(v => {{
            if (v === 'answered') a++;
            else if (v === 'blank') b++;
            else if (v === 'not_found') n++;
        }});
        return a + '/' + b + '/' + n;
    }}

    function loadItem(i) {{
        idx = i;
        renderSidebar();
        const p = pages[i];
        document.getElementById('activeLabel').innerText = 'חוברת ' + p.booklet_id + ' — ' + p.page_name + ' [' + (p.method || (p.manifest && p.manifest.method) || '?') + ']';
        document.getElementById('haltBadge').style.display = p.halted ? 'inline' : 'none';
        document.getElementById('mCols').innerText = p.n_columns;
        document.getElementById('mQs').innerText = (Object.keys(p.coverage || {{}}).join(', ') || String(p.n_questions));
        document.getElementById('mTabs').innerText = p.n_tables;
        document.getElementById('mAns').innerText = covCounts(p.coverage);
        const confs = (p.manifest && p.manifest.gutter_confidences) || [];
        document.getElementById('mConf').innerText = confs.length ? confs.map(c => c.toFixed(2)).join(', ') : (p.is_multi_column ? '--' : '1.00');
        document.getElementById('pageImg').src = p.page_src;
        renderCrops(p);
        // overlays redrawn on image load
        setTimeout(redrawOverlays, 50);
    }}

    function setShow(mode) {{
        show = mode;
        ['All','Cols','Qs','Tabs'].forEach(k => {{
            const id = 'btn' + k;
            const map = {{All:'all', Cols:'columns', Qs:'questions', Tabs:'tables'}};
            document.getElementById(id).classList.toggle('active', map[k] === mode);
        }});
        redrawOverlays();
    }}

    function redrawOverlays() {{
        const p = pages[idx];
        const img = document.getElementById('pageImg');
        const box = document.getElementById('overlays');
        box.innerHTML = '';
        if (!img.naturalWidth) return;
        const sx = img.clientWidth / img.naturalWidth;
        const sy = img.clientHeight / img.naturalHeight;
        const m = p.manifest || {{}};

        function add(bbox, cls, label) {{
            const [x,y,w,h] = bbox;
            const el = document.createElement('div');
            el.className = 'overlay ' + cls;
            el.style.left = (x * sx) + 'px';
            el.style.top = (y * sy) + 'px';
            el.style.width = (w * sx) + 'px';
            el.style.height = (h * sy) + 'px';
            el.innerHTML = '<span class="tag">' + label + '</span>';
            box.appendChild(el);
        }}

        if (show === 'all' || show === 'columns') {{
            (m.columns || []).forEach(c => add(c.bbox, '', 'col R' + c.reading_order));
            (m.gutters || []).forEach((g, i) => add([g[0], 0, g[1]-g[0], img.naturalHeight], 'gutter', 'gutter ' + i));
        }}
        if (show === 'all' || show === 'questions') {{
            (m.questions || []).forEach(q => add(q.bbox, 'question', q.question_id + ' · ' + q.state));
        }}
        if (show === 'all' || show === 'tables') {{
            (m.tables || []).forEach(t => add(t.bbox, 'table', 'table ' + t.index + ' · ' + t.kind));
        }}
    }}

    function renderCrops(p) {{
        const root = document.getElementById('crops');
        root.innerHTML = '';
        const m = p.manifest || {{}};
        const items = [];
        (m.questions || []).forEach(q => items.push({{
            title: q.question_id + ' — ' + q.label,
            meta: q.state + ' · ink ' + q.ink_fraction + ' · conf ' + q.confidence,
            src: p.crop_dir + '/' + q.crop
        }}));
        (m.tables || []).forEach(t => items.push({{
            title: 'Table ' + t.index + ' (' + t.kind + ')',
            meta: 'H' + t.n_horizontal_lines + ' V' + t.n_vertical_lines + ' · conf ' + t.confidence,
            src: p.crop_dir + '/' + t.crop
        }}));
        (m.columns || []).forEach(c => items.push({{
            title: 'Column reading_order ' + c.reading_order,
            meta: 'bbox ' + c.bbox.join(','),
            src: p.crop_dir + '/column_' + String(c.reading_order).padStart(2,'0') + '.png'
        }}));
        if (!items.length) {{
            root.innerHTML = '<div style="color:var(--text-secondary);padding:12px;">אין חיתוכים לעמוד זה</div>';
            return;
        }}
        items.forEach(it => {{
            const d = document.createElement('div');
            d.className = 'crop-card';
            d.innerHTML = `
                <div class="crop-title">${{it.title}}</div>
                <div class="crop-meta">${{it.meta}}</div>
                <img src="${{it.src}}" alt="${{it.title}}" loading="lazy">
            `;
            root.appendChild(d);
        }});
    }}

    window.onload = init;
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Generated Phase 2 review HTML at: {output_path}")


if __name__ == "__main__":
    pages = collect_pages()
    if not pages:
        print("No Phase 2 crops/manifests found. Run scripts/run_phase2.py first.")
        sys.exit(1)
    generate_html(pages, OUTPUT_HTML)
    print(f"Pages: {len(pages)}")
