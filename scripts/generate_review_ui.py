import os
import json

def generate_review_html(results_json_path, output_html_path):
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    models = data.get("models_evaluated", [])

    items_json_str = json.dumps(items, ensure_ascii=False)
    models_json_str = json.dumps(models)

    html_content = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 0 VLM Transcription Review & Manual Testing</title>
    <!-- MathJax Configuration & Loader -->
    <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #0d1117;
            --bg-card: #161b22;
            --border-color: #30363d;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-orange: #d29922;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
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
        }}
        .container {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}
        /* Left/Right Split */
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
        .page-item-title {{
            font-weight: 600;
            color: #fff;
            margin-bottom: 4px;
        }}
        .page-item-desc {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        .main-view {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}
        .pane-image {{
            flex: 1;
            background: #010409;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            overflow: auto;
            padding: 20px;
            border-left: 1px solid var(--border-color);
        }}
        .pane-image img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
            border: 1px solid #30363d;
        }}
        .pane-results {{
            flex: 1.2;
            background: var(--bg-main);
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .model-tabs {{
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}
        .model-tab {{
            padding: 8px 16px;
            background: #21262d;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .model-tab.active {{
            background: var(--accent-blue);
            color: #fff;
            border-color: var(--accent-blue);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }}
        .metric-card {{
            background: var(--bg-card);
            padding: 12px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .metric-card .val {{
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--accent-green);
        }}
        .metric-card .lbl {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        .transcription-box {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
        }}
        .question-card {{
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .question-label {{
            font-size: 1rem;
            font-weight: 700;
            color: var(--accent-blue);
            margin-bottom: 10px;
            border-bottom: 1px solid #30363d;
            padding-bottom: 6px;
        }}
        .hebrew-text {{
            font-size: 1.05rem;
            line-height: 1.7;
            margin-bottom: 14px;
            direction: rtl;
        }}
        .latex-list {{
            background: #0d1117;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #30363d;
            direction: ltr;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        .manual-notes {{
            background: #1c2128;
            border: 1px solid var(--accent-orange);
            border-radius: 8px;
            padding: 16px;
        }}
        .manual-notes h4 {{
            color: var(--accent-orange);
            margin-bottom: 8px;
        }}
        textarea {{
            width: 100%;
            height: 80px;
            background: #0d1117;
            border: 1px solid #30363d;
            color: #fff;
            padding: 8px;
            border-radius: 6px;
            font-family: 'Assistant', sans-serif;
            resize: vertical;
        }}
        .btn {{
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
        <span>מערכת בדיקת מבחנים - בדיקת היתכנות VLM (Phase 0)</span>
        <span class="badge">Feasibility & Ground-Truth Inspector</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-secondary);">
        12 Pilot Scans | Dual Model Bake-Off
    </div>
</header>

<div class="container">
    <!-- Sidebar -->
    <div class="sidebar" id="sidebar"></div>

    <!-- Main View -->
    <div class="main-view">
        <!-- Original Scan -->
        <div class="pane-image">
            <h3 style="margin-bottom: 12px; color: var(--text-secondary); width: 100%; text-align: right;">סריקת המבחן המקורית</h3>
            <img id="scanImage" src="" alt="Exam Scan">
        </div>

        <!-- Transcription & Extraction -->
        <div class="pane-results">
            <!-- Model Tabs -->
            <div class="model-tabs" id="modelTabs"></div>

            <!-- Metrics -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="val" id="metricLatency">--</div>
                    <div class="lbl">זמן תגובה (Latency)</div>
                </div>
                <div class="metric-card">
                    <div class="val" id="metricFormulas">--</div>
                    <div class="lbl">נוסחאות תקינות (LaTeX Parsed)</div>
                </div>
                <div class="metric-card">
                    <div class="val" id="metricTokens">--</div>
                    <div class="lbl">Tokens (In / Out)</div>
                </div>
            </div>

            <!-- Content -->
            <div class="transcription-box" id="transcriptionBox">
                <!-- Populated dynamically -->
            </div>

            <!-- Manual Test / Verification Section -->
            <div class="manual-notes">
                <h4>הערות בדיקה ידנית (Manual Quality Check)</h4>
                <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 8px;">
                    השווה בין הסריקה המקורית משמאל לבין הפענוח מימין. ניתן לרשום הערות איכות או לתקן את התמלול:
                </p>
                <textarea id="manualCheckInput" placeholder="רשום הערות לגבי איכות הפענוח, מילים שהוחמצו, או נוסחאות בעייתיות..."></textarea>
                <button class="btn" onclick="saveManualReview()">שמור הערה לדוח</button>
            </div>
        </div>
    </div>
</div>

<script>
    const items = {items_json_str};
    const models = {models_json_str};
    let currentItemIdx = 0;
    let currentModel = models[0];

    function init() {{
        renderSidebar();
        renderModelTabs();
        loadItem(0);
    }}

    function renderSidebar() {{
        const sidebar = document.getElementById('sidebar');
        sidebar.innerHTML = '';
        items.forEach((it, idx) => {{
            const div = document.createElement('div');
            div.className = 'page-item' + (idx === currentItemIdx ? ' active' : '');
            div.onclick = () => loadItem(idx);
            div.innerHTML = `
                <div class="page-item-title">${{it.metadata.id}}: ${{it.metadata.pdf.split('/').pop()}} (עמ' ${{it.metadata.page + 1}})</div>
                <div class="page-item-desc">${{it.metadata.description}}</div>
            `;
            sidebar.appendChild(div);
        }});
    }}

    function renderModelTabs() {{
        const tabs = document.getElementById('modelTabs');
        tabs.innerHTML = '';
        models.forEach(m => {{
            const btn = document.createElement('button');
            btn.className = 'model-tab' + (m === currentModel ? ' active' : '');
            btn.innerText = m;
            btn.onclick = () => selectModel(m);
            tabs.appendChild(btn);
        }});
    }}

    function selectModel(modelName) {{
        currentModel = modelName;
        renderModelTabs();
        updateDisplay();
    }}

    function loadItem(idx) {{
        currentItemIdx = idx;
        renderSidebar();
        const item = items[idx];
        // Image path relative to reports folder: ../crops/pilot/...
        const imgPath = '../crops/pilot/' + item.metadata.image_filename;
        document.getElementById('scanImage').src = imgPath;
        updateDisplay();
    }}

    function updateDisplay() {{
        const item = items[currentItemIdx];
        const run = item.model_runs[currentModel];
        const transBox = document.getElementById('transcriptionBox');

        if (!run || !run.success) {{
            document.getElementById('metricLatency').innerText = run ? run.latency_sec + 's' : '--';
            document.getElementById('metricFormulas').innerText = 'שגיאה';
            document.getElementById('metricTokens').innerText = '--';
            transBox.innerHTML = `<div style="color: #ff7b72;">שגיאה בקריאה למודל: ${{run ? run.error : 'לא הופעל'}}</div>`;
            return;
        }}

        // Metrics
        document.getElementById('metricLatency').innerText = run.latency_sec + 's';
        const le = run.latex_eval;
        document.getElementById('metricFormulas').innerText = `${{le.parsed}} / ${{le.total}} (${{Math.round(le.parse_rate * 100)}}%)`;
        document.getElementById('metricTokens').innerText = `${{run.prompt_tokens}} / ${{run.candidate_tokens}}`;

        // Parse content
        const p = run.parsed_transcription;
        let html = `<h3 style="color: #fff; margin-bottom: 8px;">תקציר: ${{p.summary || 'אין תקציר'}}</h3>`;
        html += `<div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 16px;">רמת קריאות מוערכת: ${{p.legibility_assessment || 'N/A'}} | מהימנות: ${{p.transcription_confidence || 'N/A'}}</div>`;

        if (p.questions && p.questions.length > 0) {{
            p.questions.forEach(q => {{
                html += `
                    <div class="question-card">
                        <div class="question-label">${{q.question_label || 'שאלה / מקטע'}}</div>
                        <div class="hebrew-text">${{q.full_text || q.hebrew_transcription || ''}}</div>
                        ${{q.latex_formulas && q.latex_formulas.length > 0 ? `
                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 4px;">נוסחאות שחולצו:</div>
                            <div class="latex-list">${{q.latex_formulas.map(f => `<div>$$${{f.replace(/\\$/g, '')}}$$</div>`).join('')}}</div>
                        ` : ''}}
                    </div>
                `;
            }});
        }} else {{
            html += `<pre style="white-space: pre-wrap; font-size: 0.9rem;">${{p.raw_output || 'אין נתונים'}}</pre>`;
        }}

        transBox.innerHTML = html;

        // Trigger MathJax re-render
        if (window.MathJax && window.MathJax.typesetPromise) {{
            window.MathJax.typesetPromise();
        }}
    }}

    function saveManualReview() {{
        const text = document.getElementById('manualCheckInput').value;
        if (!text.trim()) return;
        const currentItem = items[currentItemIdx];
        alert('נשמרה הערת בדיקה עבור ' + currentItem.metadata.id + ':\\n' + text);
        document.getElementById('manualCheckInput').value = '';
    }}

    window.onload = init;
</script>

</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated review HTML at: {output_html_path}")

if __name__ == "__main__":
    import sys
    r_path = "storage/reports/phase0_results.json"
    o_path = "storage/reports/phase0_review.html"
    if os.path.exists(r_path):
        generate_review_html(r_path, o_path)
    else:
        print("phase0_results.json does not exist yet. Run benchmark first.")
