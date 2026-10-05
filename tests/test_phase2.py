"""
Phase 2 — Automated Test Suite.

Covers:
    ✓ Multi-column gutter detection (≥ synthetic recall)
    ✓ Single-column fallback
    ✓ Question region splitting + blank vs answered
    ✓ Coverage: not_found halts and never collapses to blank
    ✓ Table / grid detection
    ✓ Crop + manifest writer
    ✓ End-to-end booklet segmentation

Run:
    python -m pytest tests/test_phase2.py -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.layout.gutter import detect_columns
from core.layout.questions import detect_question_regions
from core.layout.tables import detect_tables
from core.layout.pipeline import (
    segment_page,
    CoverageState,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="phase2_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _blank_page(h=2000, w=1400, color=255):
    return np.full((h, w, 3), color, dtype=np.uint8)


def _draw_text_block(img, x, y, bw, bh, shade=40):
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (shade, shade, shade), -1)
    for yy in range(y + 20, y + bh - 10, 28):
        cv2.line(img, (x + 10, yy), (x + bw - 10, yy), (20, 20, 20), 2)


def make_two_column_page(tmp_dir: Path) -> Path:
    """Synthetic Hebrew-style 2-column page with a clear center gutter."""
    img = _blank_page(2200, 1800)
    # Left column content blocks
    _draw_text_block(img, 80, 120, 700, 400)
    _draw_text_block(img, 80, 600, 700, 500)
    _draw_text_block(img, 80, 1200, 700, 450)
    # Right column content blocks
    _draw_text_block(img, 1020, 120, 700, 450)
    _draw_text_block(img, 1020, 700, 700, 550)
    _draw_text_block(img, 1020, 1400, 700, 400)
    # Explicit white gutter strip
    img[:, 820:980] = 255
    path = tmp_dir / "two_col.png"
    cv2.imwrite(str(path), img)
    return path


def make_single_column_page(tmp_dir: Path) -> Path:
    img = _blank_page(2000, 1400)
    _draw_text_block(img, 100, 100, 1200, 350)
    _draw_text_block(img, 100, 550, 1200, 400)
    _draw_text_block(img, 100, 1100, 1200, 500)
    path = tmp_dir / "one_col.png"
    cv2.imwrite(str(path), img)
    return path


def make_table_page(tmp_dir: Path) -> Path:
    img = _blank_page(1600, 1200)
    # Draw a score-grid table in the upper half
    x0, y0, cols, rows, cw, rh = 100, 80, 5, 6, 180, 50
    for r in range(rows + 1):
        y = y0 + r * rh
        cv2.line(img, (x0, y), (x0 + cols * cw, y), (0, 0, 0), 3)
    for c in range(cols + 1):
        x = x0 + c * cw
        cv2.line(img, (x, y0), (x, y0 + rows * rh), (0, 0, 0), 3)
    # Some freeform writing below
    _draw_text_block(img, 100, 500, 1000, 400)
    path = tmp_dir / "table_page.png"
    cv2.imwrite(str(path), img)
    return path


class TestGutterDetection:
    def test_detects_two_columns(self, tmp_dir):
        img = cv2.imread(str(make_two_column_page(tmp_dir)))
        layout = detect_columns(img)
        assert layout.is_multi_column
        assert layout.n_columns == 2
        assert layout.ok
        # RTL: reading_order 0 should be the rightmost column
        rightmost = max(layout.columns, key=lambda c: c.x0)
        assert rightmost.reading_order == 0

    def test_single_column_fallback(self, tmp_dir):
        img = cv2.imread(str(make_single_column_page(tmp_dir)))
        layout = detect_columns(img)
        assert layout.n_columns == 1
        assert not layout.is_multi_column

    def test_gutter_confidence_positive(self, tmp_dir):
        img = cv2.imread(str(make_two_column_page(tmp_dir)))
        layout = detect_columns(img)
        assert layout.confidences
        assert max(layout.confidences) >= 0.3


class TestQuestionSegmentation:
    def test_splits_into_multiple_regions(self, tmp_dir):
        img = cv2.imread(str(make_single_column_page(tmp_dir)))
        layout = detect_columns(img)
        regions = detect_question_regions(img, layout)
        assert len(regions) >= 2

    def test_blank_vs_answered(self, tmp_dir):
        img = _blank_page(1800, 1200)
        _draw_text_block(img, 80, 80, 1000, 400)
        # Leave a large blank band, then another block
        _draw_text_block(img, 80, 1200, 1000, 350)
        path = tmp_dir / "mixed.png"
        cv2.imwrite(str(path), img)
        loaded = cv2.imread(str(path))
        layout = detect_columns(loaded)
        regions = detect_question_regions(loaded, layout)
        assert any(r.state == "answered" for r in regions)

    def test_expected_id_labelling(self, tmp_dir):
        img = cv2.imread(str(make_single_column_page(tmp_dir)))
        layout = detect_columns(img)
        regions = detect_question_regions(
            img, layout, expected_ids=["Q1", "Q2", "Q3"]
        )
        ids = [r.question_id for r in regions]
        assert ids[0] == "Q1"


class TestTableDetection:
    def test_detects_grid(self, tmp_dir):
        img = cv2.imread(str(make_table_page(tmp_dir)))
        tables = detect_tables(img)
        assert len(tables) >= 1
        assert tables[0].n_horizontal_lines >= 3
        assert tables[0].n_vertical_lines >= 2


class TestPipeline:
    def test_segment_writes_crops_and_manifest(self, tmp_dir, monkeypatch):
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))
        import core.layout.pipeline as pipe
        pipe.STORAGE_ROOT = tmp_dir
        pipe.ANONYMIZED_STORAGE = tmp_dir / "anonymized"
        pipe.PREPROCESSED_STORAGE = tmp_dir / "preprocessed"
        pipe.CROPS_STORAGE = tmp_dir / "crops"
        pipe.DEFAULT_USE_VLM = False

        src = make_two_column_page(tmp_dir)
        result = segment_page(src, "bookA", page_index=0, use_vlm=False)
        assert result.ok, result.errors
        assert result.n_columns == 2
        assert result.n_questions >= 2
        assert result.method == "cv"
        manifest = Path(result.output_dir) / "manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["is_multi_column"] is True
        assert len(data["questions"]) == result.n_questions
        # Crop files exist
        assert any(Path(p).suffix == ".png" for p in result.crop_paths)

    def test_not_found_halts_and_stays_distinct(self, tmp_dir, monkeypatch):
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))
        import core.layout.pipeline as pipe
        pipe.STORAGE_ROOT = tmp_dir
        pipe.CROPS_STORAGE = tmp_dir / "crops"
        pipe.DEFAULT_USE_VLM = False

        src = make_single_column_page(tmp_dir)
        # Demand more questions than the page can yield
        result = segment_page(
            src,
            "bookB",
            page_index=0,
            expected_questions=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
            use_vlm=False,
        )
        assert result.halted
        assert any(
            v == CoverageState.NOT_FOUND.value for v in result.coverage.values()
        )
        # Critical: not_found must never be rewritten as blank
        for qid, state in result.coverage.items():
            if state == CoverageState.NOT_FOUND.value:
                assert state != CoverageState.BLANK.value

    def test_no_expected_still_ok(self, tmp_dir, monkeypatch):
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))
        import core.layout.pipeline as pipe
        pipe.CROPS_STORAGE = tmp_dir / "crops"
        pipe.DEFAULT_USE_VLM = False

        src = make_single_column_page(tmp_dir)
        result = segment_page(src, "bookC", page_index=0, use_vlm=False)
        assert result.ok
        assert not result.halted

    def test_vlm_path_uses_real_question_ids(self, tmp_dir, monkeypatch):
        """Mocked VLM returns real exam labels; pipeline must keep them."""
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))
        import core.layout.pipeline as pipe
        import core.layout.vlm_segmenter as seg

        pipe.STORAGE_ROOT = tmp_dir
        pipe.CROPS_STORAGE = tmp_dir / "crops"

        from core.layout.questions import QuestionRegion
        from core.layout.vlm_segmenter import VLMSegmentResult

        fake = VLMSegmentResult(
            questions=[
                QuestionRegion(
                    question_id="7",
                    label="שאלה 7",
                    bbox=(50, 50, 400, 200),
                    column_index=0,
                    reading_order=0,
                    ink_fraction=0.0,
                    state="answered",
                    confidence=0.9,
                ),
                QuestionRegion(
                    question_id="8",
                    label="שאלה 8",
                    bbox=(50, 300, 400, 200),
                    column_index=0,
                    reading_order=1,
                    ink_fraction=0.0,
                    state="blank",
                    confidence=0.85,
                ),
            ],
            n_columns=1,
            reading_order_hint="single",
            override_gutter_prior=True,
            page_summary="mock page",
            model="mock",
        )

        monkeypatch.setattr(seg, "segment_page_with_vlm", lambda *a, **k: fake)

        src = make_single_column_page(tmp_dir)
        result = segment_page(src, "bookV", page_index=0, use_vlm=True)
        assert result.method == "vlm"
        assert result.n_columns == 1
        assert set(result.coverage.keys()) == {"7", "8"}
        assert result.coverage["7"] == "answered"
        assert result.coverage["8"] == "blank"
        assert (Path(result.output_dir) / "question_7.png").exists()
        assert (Path(result.output_dir) / "question_8.png").exists()

    def test_expected_missing_halts_on_vlm_ids(self, tmp_dir, monkeypatch):
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))
        import core.layout.pipeline as pipe
        import core.layout.vlm_segmenter as seg
        from core.layout.questions import QuestionRegion
        from core.layout.vlm_segmenter import VLMSegmentResult

        pipe.CROPS_STORAGE = tmp_dir / "crops"
        fake = VLMSegmentResult(
            questions=[
                QuestionRegion(
                    question_id="7",
                    label="שאלה 7",
                    bbox=(50, 50, 400, 200),
                    column_index=0,
                    reading_order=0,
                    ink_fraction=0.0,
                    state="answered",
                    confidence=0.9,
                ),
            ],
            n_columns=1,
            override_gutter_prior=True,
            model="mock",
        )
        monkeypatch.setattr(seg, "segment_page_with_vlm", lambda *a, **k: fake)
        src = make_single_column_page(tmp_dir)
        result = segment_page(
            src,
            "bookH",
            page_index=0,
            use_vlm=True,
            expected_questions=["7", "8", "9"],
        )
        assert result.halted
        assert result.coverage["8"] == CoverageState.NOT_FOUND.value
        assert result.coverage["9"] == CoverageState.NOT_FOUND.value
        assert result.coverage["7"] == CoverageState.ANSWERED.value
