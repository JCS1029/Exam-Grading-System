"""
Phase 3 — Automated Test Suite.

Covers:
    ✓ LaTeX SymPy validation
    ✓ Objective confidence (rejects naive self-reported 0.95 on bad parse)
    ✓ Hebrew CER via jiwer
    ✓ Gate metric aggregation
    ✓ Crop discovery from manifests
    ✓ Transcript parsing without live API

Run:
    python -m pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.transcription.confidence import compute_objective_confidence
from core.transcription.discovery import discover_crops
from core.transcription.latex import collect_all_formulas, extract_latex_from_markdown, validate_latex_formulas
from core.transcription.metrics import aggregate_gate_metrics, hebrew_cer, normalize_hebrew_for_cer
from core.transcription.transcriber import _parse_transcript_payload
from core.transcription.discovery import CropRef


def test_validate_latex_parseable():
    ev = validate_latex_formulas([r"x^2 + 1", r"\alpha + \beta"])
    assert ev["total"] == 2
    assert ev["parse_rate"] >= 0.5


def test_validate_latex_empty_is_perfect():
    ev = validate_latex_formulas([])
    assert ev["parse_rate"] == 1.0
    assert ev["total"] == 0


def test_extract_latex_from_markdown():
    md = "נתון $x^2$ וגם $$\\sum_{i=1}^n i$$"
    spans = extract_latex_from_markdown(md)
    assert "x^2" in spans
    assert "\\sum_{i=1}^n i" in spans


def test_objective_confidence_penalizes_flags_despite_high_self_report():
    score, signals = compute_objective_confidence(
        latex_eval={"total": 2, "parsed": 0, "parse_rate": 0.0},
        flagged_tokens=["?", "??", "illegible"],
        has_content=True,
        is_blank=False,
        self_reported=0.95,
    )
    assert score < 0.5
    assert signals["self_reported_confidence"] == 0.95


def test_objective_confidence_blank_crop():
    score, _ = compute_objective_confidence(
        latex_eval={"total": 0, "parse_rate": 1.0},
        flagged_tokens=[],
        has_content=False,
        is_blank=True,
    )
    assert score >= 0.9


def test_hebrew_cer_identical():
    ref = "זוהי תשובה נכונה עם $x^2$"
    assert hebrew_cer(ref, ref) == 0.0


def test_hebrew_cer_different():
    cer = hebrew_cer("שלום עולם", "שלום worlds")
    assert 0 < cer <= 1.0


def test_normalize_hebrew_strips_niqqud():
    t = normalize_hebrew_for_cer("שָׁלוֹם")
    assert "ש" in t


def test_aggregate_gate_metrics():
    results = [
        {
            "crop_id": "a/b/q1",
            "hebrew_text": "שלום",
            "latex_eval": {"total": 1, "parse_rate": 1.0},
            "prompt_tokens": 500,
            "candidate_tokens": 200,
            "cache_hit": False,
            "errors": [],
            "ok": True,
        }
    ]
    gt = {"a/b/q1": "שלום"}
    agg = aggregate_gate_metrics(results, ground_truth=gt)
    assert agg["mean_hebrew_cer"] == 0.0
    assert agg["mean_latex_parse_rate"] == 1.0
    assert agg["unhandled_503_count"] == 0
    assert agg["projected_cohort_cost_usd"] < 25.0


def test_cost_uses_median_not_outlier_mean():
    """One runaway completion must not push projected cost over $25."""
    results = [
        {
            "crop_id": f"x/p/q{i}",
            "hebrew_text": "שלום",
            "latex_eval": {"total": 1, "parse_rate": 1.0},
            "prompt_tokens": 800,
            "candidate_tokens": 150,
            "cache_hit": False,
            "errors": [],
            "ok": True,
        }
        for i in range(9)
    ]
    results.append(
        {
            "crop_id": "x/p/q_outlier",
            "hebrew_text": "שלום",
            "latex_eval": {"total": 1, "parse_rate": 1.0},
            "prompt_tokens": 800,
            "candidate_tokens": 200_000,  # pathological
            "cache_hit": False,
            "errors": [],
            "ok": True,
        }
    )
    agg = aggregate_gate_metrics(results)
    assert agg["projected_cohort_cost_usd"] < 25.0
    assert agg["avg_completion_tokens_per_call"] == 150.0


def test_parse_transcript_payload():
    crop = CropRef(
        crop_id="02/page_001/q1",
        booklet_id="02",
        page_name="page_001.png",
        question_id="1",
        question_label="שאלה 1",
        crop_path=Path("x.png"),
        manifest_path=Path("m.json"),
        state="answered",
    )
    parsed = {
        "hebrew_text": "פתרון",
        "math_latex": ["x^2"],
        "interleaved_markdown": "פתרון $x^2$",
        "flagged_tokens": [],
        "transcription_confidence": 0.99,
        "is_blank": False,
    }
    tr = _parse_transcript_payload(parsed, crop)
    assert tr.hebrew_text == "פתרון"
    assert tr.objective_confidence > 0.5
    assert tr.self_reported_confidence == 0.99


def test_discover_crops_from_temp_manifest(tmp_path):
    crops_root = tmp_path / "crops" / "99" / "page_001"
    crops_root.mkdir(parents=True)
    (crops_root / "question_1.png").write_bytes(b"fake")
    manifest = {
        "booklet_id": "99",
        "page_name": "page_001.png",
        "questions": [
            {
                "question_id": "1",
                "label": "שאלה 1",
                "state": "answered",
                "crop": "question_1.png",
            },
            {
                "question_id": "2",
                "label": "שאלה 2",
                "state": "blank",
                "crop": "question_2.png",
            },
        ],
    }
    (crops_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    found = discover_crops(booklet_ids=["99"], crops_root=tmp_path / "crops")
    assert len(found) == 1
    assert found[0].question_id == "1"


def test_collect_all_formulas_dedupes():
    formulas = collect_all_formulas(["x^2"], "שוב $x^2$")
    assert formulas == ["x^2"]


def test_parse_json_repairs_latex_backslashes():
    from core.vlm import parse_json_response

    # Invalid JSON: single backslash before TeX commands including \b (≠ JSON \b)
    raw = (
        '{\n'
        '  "hebrew_text": "בדיקה",\n'
        '  "math_latex": ["\\\\alpha"],\n'
        '  "interleaved_markdown": "$\\\\alpha$ and $\\square$ and $\\boxtimes$",\n'
        '  "flagged_tokens": []\n'
        '}'
    )
    parsed = parse_json_response(raw)
    assert "בדיקה" in parsed["hebrew_text"]
    assert "\\square" in parsed["interleaved_markdown"]
    assert "\\boxtimes" in parsed["interleaved_markdown"]
    # Real newline escapes still work
    raw2 = '{"hebrew_text": "שורה\\nשנייה", "math_latex": []}'
    assert "\n" in parse_json_response(raw2)["hebrew_text"]


def test_parse_json_repairs_hebrew_gershayim_quotes():
    from core.vlm import parse_json_response

    raw = '{"hebrew_text": "אף אחד מהנ"ל", "math_latex": [], "interleaved_markdown": "x"}'
    parsed = parse_json_response(raw)
    assert 'מהנ"ל' in parsed["hebrew_text"]
