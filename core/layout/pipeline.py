"""
Phase 2 pipeline — segment anonymised pages into question / column / table crops.

Planned Phase 2 behaviour:
  1. OpenCV gutter prior
  2. VLM question segmentation (real שאלה N labels + full-question bboxes)
  3. Coverage validation (answered / blank / not_found)
  4. Crop writer

CV-only mode remains available via ``use_vlm=False`` for offline/debug runs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.layout.gutter import ColumnBand, ColumnLayout, detect_columns
from core.layout.questions import QuestionRegion, detect_question_regions
from core.layout.tables import TableRegion, detect_tables
from core.vlm import GEMINI_API_KEY

logger = logging.getLogger(__name__)

STORAGE_ROOT: Path = Path(os.getenv("STORAGE_ROOT", "storage"))
ANONYMIZED_STORAGE: Path = STORAGE_ROOT / "anonymized"
PREPROCESSED_STORAGE: Path = STORAGE_ROOT / "preprocessed"
CROPS_STORAGE: Path = STORAGE_ROOT / "crops"

# Default: use VLM when an API key is present
DEFAULT_USE_VLM: bool = bool(GEMINI_API_KEY) and os.getenv("PHASE2_USE_VLM", "1") not in (
    "0",
    "false",
    "False",
)


class CoverageState(str, Enum):
    ANSWERED = "answered"
    BLANK = "blank"
    NOT_FOUND = "not_found"


@dataclass
class SegmentResult:
    """Result of segmenting a single page."""

    booklet_id: str
    page_index: int
    page_name: str
    source_path: str
    output_dir: str
    n_columns: int = 1
    n_questions: int = 0
    n_tables: int = 0
    is_multi_column: bool = False
    coverage: Dict[str, str] = field(default_factory=dict)
    crop_paths: List[str] = field(default_factory=list)
    halted: bool = False
    halt_reasons: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    method: str = "cv"  # "vlm" | "cv"
    page_summary: str = ""

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0 and not self.halted


def _safe_name(qid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in qid)


def _layout_from_vlm_columns(
    page_w: int,
    page_h: int,
    n_columns: int,
    reading_order_hint: str,
) -> ColumnLayout:
    """Build a simple column layout when VLM overrides the OpenCV gutter."""
    if n_columns <= 1:
        return ColumnLayout(
            n_columns=1,
            columns=[ColumnBand(index=0, x0=0, x1=page_w, reading_order=0)],
            gutters=[],
            confidences=[],
            is_multi_column=False,
        )
    mid = page_w // 2
    left = ColumnBand(index=0, x0=0, x1=mid, reading_order=1 if reading_order_hint == "rtl" else 0)
    right = ColumnBand(index=1, x0=mid, x1=page_w, reading_order=0 if reading_order_hint == "rtl" else 1)
    cols = sorted([left, right], key=lambda c: c.reading_order)
    return ColumnLayout(
        n_columns=2,
        columns=cols,
        gutters=[(mid - 20, mid + 20)],
        confidences=[0.9],
        is_multi_column=True,
    )


def _tables_from_vlm(raw_tables: List[dict], page_w: int, page_h: int) -> List[TableRegion]:
    from core.layout.vlm_segmenter import _norm_to_bbox

    out: List[TableRegion] = []
    for i, t in enumerate(raw_tables or []):
        try:
            bbox = _norm_to_bbox(t.get("bbox_norm") or [0, 0, 1, 1], page_w, page_h)
            out.append(
                TableRegion(
                    index=i,
                    bbox=bbox,
                    n_horizontal_lines=0,
                    n_vertical_lines=0,
                    confidence=float(t.get("confidence") or 0.7),
                    kind=str(t.get("kind") or "unknown"),
                )
            )
        except Exception:
            continue
    return out


def _apply_coverage(
    questions: List[QuestionRegion],
    expected_questions: Optional[Sequence[str]],
    page_name: str,
) -> Tuple[Dict[str, str], bool, List[str]]:
    coverage: Dict[str, str] = {}
    for q in questions:
        coverage[q.question_id] = (
            CoverageState.BLANK.value if q.state == "blank" else CoverageState.ANSWERED.value
        )

    halted = False
    reasons: List[str] = []
    if expected_questions:
        found = set(coverage.keys())
        for qid in expected_questions:
            qid_s = str(qid)
            if qid_s not in found:
                coverage[qid_s] = CoverageState.NOT_FOUND.value
                halted = True
                reasons.append(
                    f"Expected question '{qid_s}' not found on {page_name} — "
                    f"must not be graded as zero"
                )
    return coverage, halted, reasons


def segment_page(
    image_path: str | Path,
    booklet_id: str,
    page_index: int = 0,
    *,
    expected_questions: Optional[Sequence[str]] = None,
    write_crops: bool = True,
    use_vlm: Optional[bool] = None,
    vlm_model: Optional[str] = None,
) -> SegmentResult:
    """
    Run full Phase 2 segmentation on one page image.

    Prefers VLM segmentation when ``use_vlm`` is True (default if API key set).
    Falls back to classical-CV ink bands only if VLM fails or is disabled.
    """
    image_path = Path(image_path)
    page_name = image_path.name if image_path.suffix else f"page_{page_index + 1:03d}.png"
    out_dir = CROPS_STORAGE / booklet_id / page_name.replace(".png", "")
    result = SegmentResult(
        booklet_id=booklet_id,
        page_index=page_index,
        page_name=page_name,
        source_path=str(image_path),
        output_dir=str(out_dir),
    )

    image = cv2.imread(str(image_path))
    if image is None:
        result.errors.append(f"Failed to load image: {image_path}")
        return result

    h, w = image.shape[:2]
    want_vlm = DEFAULT_USE_VLM if use_vlm is None else use_vlm

    # 1. OpenCV gutter prior
    try:
        layout = detect_columns(image)
    except Exception as exc:
        result.errors.append(f"Column detection failed: {exc}")
        logger.exception("Column detection error on %s", image_path)
        return result

    questions: List[QuestionRegion] = []
    tables: List[TableRegion] = []
    method = "cv"

    # 2. VLM segmentation (planned Phase 2)
    if want_vlm:
        try:
            from core.layout.vlm_segmenter import USE_VLM_CACHE, segment_page_with_vlm

            vlm = segment_page_with_vlm(
                image, layout, model=vlm_model, use_cache=USE_VLM_CACHE
            )
            # Accept VLM result when it finds questions OR tables (cover sheets),
            # or when it cleanly reports an empty content page without hard errors.
            vlm_usable = (
                bool(vlm.questions)
                or bool(vlm.tables)
                or (not vlm.errors and vlm.page_summary)
            )
            if vlm_usable:
                method = "vlm"
                questions = vlm.questions
                result.page_summary = vlm.page_summary
                # Trust VLM column count over OpenCV when they disagree
                if vlm.override_gutter_prior or (
                    vlm.n_columns and vlm.n_columns != layout.n_columns
                ):
                    layout = _layout_from_vlm_columns(
                        w, h, max(1, vlm.n_columns), vlm.reading_order_hint
                    )
                tables = _tables_from_vlm(vlm.tables, w, h)
                # Also run classical table detect if VLM found none but page looks tabular
                if not tables:
                    try:
                        tables = detect_tables(image)
                    except Exception:
                        tables = []
                result.metrics.update(
                    {
                        "vlm_model": vlm.model,
                        "vlm_latency_sec": vlm.latency_sec,
                        "vlm_prompt_tokens": vlm.prompt_tokens,
                        "vlm_candidate_tokens": vlm.candidate_tokens,
                        "vlm_cache_hit": vlm.cache_hit,
                        "vlm_prompt_version": vlm.prompt_version,
                        "vlm_override_gutter": vlm.override_gutter_prior,
                    }
                )
            else:
                logger.warning(
                    "VLM segmentation weak/failed on %s/%s: %s — falling back to CV",
                    booklet_id,
                    page_name,
                    "; ".join(vlm.errors) or "no questions",
                )
                result.errors.extend([f"vlm: {e}" for e in vlm.errors])
        except Exception as exc:
            logger.exception("VLM path crashed on %s — CV fallback", image_path)
            result.errors.append(f"vlm crashed: {exc}")

    # 3. CV fallback (or forced offline mode)
    if method == "cv":
        try:
            tables = detect_tables(image)
        except Exception as exc:
            result.errors.append(f"Table detection failed: {exc}")
            tables = []
        try:
            questions = detect_question_regions(
                image, layout, expected_ids=expected_questions
            )
        except Exception as exc:
            result.errors.append(f"Question segmentation failed: {exc}")
            questions = []

    result.method = method
    result.n_columns = layout.n_columns
    result.is_multi_column = layout.is_multi_column
    result.n_tables = len(tables)
    result.n_questions = len(questions)

    # 4. Coverage validation
    coverage, halted, reasons = _apply_coverage(
        questions, expected_questions, page_name
    )
    result.coverage = coverage
    result.halted = halted
    result.halt_reasons = reasons

    # Clear non-fatal VLM warnings once we have a successful CV fallback
    if method == "cv" and questions:
        result.errors = [e for e in result.errors if not e.startswith("vlm")]

    # 5. Write crops + manifest
    crop_paths: List[str] = []
    if write_crops:
        # Clean previous crops for this page so stale Q1/Q14 files don't linger
        if out_dir.exists():
            for old in out_dir.glob("question_*.png"):
                try:
                    old.unlink()
                except OSError:
                    pass
        out_dir.mkdir(parents=True, exist_ok=True)

        for col in layout.columns:
            x0, x1 = col.x0, col.x1
            crop = image[0:h, x0:x1]
            fname = f"column_{col.reading_order:02d}.png"
            path = out_dir / fname
            cv2.imwrite(str(path), crop)
            crop_paths.append(str(path))

        for q in questions:
            x, y, bw, bh = q.bbox
            # Clamp to page
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))
            crop = image[y : y + bh, x : x + bw]
            fname = f"question_{_safe_name(q.question_id)}.png"
            path = out_dir / fname
            cv2.imwrite(str(path), crop)
            crop_paths.append(str(path))

        for t in tables:
            x, y, bw, bh = t.bbox
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))
            crop = image[y : y + bh, x : x + bw]
            fname = f"table_{t.index:02d}.png"
            path = out_dir / fname
            cv2.imwrite(str(path), crop)
            crop_paths.append(str(path))

        manifest = {
            "booklet_id": booklet_id,
            "page_index": page_index,
            "page_name": page_name,
            "source_path": str(image_path),
            "method": method,
            "page_summary": result.page_summary,
            "page_size": {"width": w, "height": h},
            "columns": [
                {
                    "index": c.index,
                    "reading_order": c.reading_order,
                    "bbox": [c.x0, 0, c.width, h],
                }
                for c in sorted(layout.columns, key=lambda c: c.reading_order)
            ],
            "gutters": [list(g) for g in layout.gutters],
            "is_multi_column": layout.is_multi_column,
            "gutter_confidences": layout.confidences,
            "questions": [
                {
                    "question_id": q.question_id,
                    "label": q.label,
                    "bbox": list(q.bbox),
                    "column_index": q.column_index,
                    "reading_order": q.reading_order,
                    "ink_fraction": q.ink_fraction,
                    "state": q.state,
                    "confidence": q.confidence,
                    "is_subquestion": q.is_subquestion,
                    "crop": f"question_{_safe_name(q.question_id)}.png",
                }
                for q in questions
            ],
            "tables": [
                {
                    "index": t.index,
                    "bbox": list(t.bbox),
                    "n_horizontal_lines": t.n_horizontal_lines,
                    "n_vertical_lines": t.n_vertical_lines,
                    "confidence": t.confidence,
                    "kind": t.kind,
                    "crop": f"table_{t.index:02d}.png",
                }
                for t in tables
            ],
            "coverage": coverage,
            "halted": result.halted,
            "halt_reasons": result.halt_reasons,
            "vlm_metrics": {
                k: result.metrics[k]
                for k in (
                    "vlm_model",
                    "vlm_latency_sec",
                    "vlm_prompt_tokens",
                    "vlm_candidate_tokens",
                    "vlm_cache_hit",
                    "vlm_prompt_version",
                    "vlm_override_gutter",
                )
                if k in result.metrics
            },
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        crop_paths.append(str(manifest_path))

    result.crop_paths = crop_paths
    result.metrics.update(
        {
            "page_width": w,
            "page_height": h,
            "n_columns": layout.n_columns,
            "n_questions": len(questions),
            "n_tables": len(tables),
            "n_answered": sum(
                1 for s in coverage.values() if s == CoverageState.ANSWERED.value
            ),
            "n_blank": sum(1 for s in coverage.values() if s == CoverageState.BLANK.value),
            "n_not_found": sum(
                1 for s in coverage.values() if s == CoverageState.NOT_FOUND.value
            ),
            "mean_question_confidence": round(
                float(np.mean([q.confidence for q in questions])) if questions else 0.0,
                3,
            ),
            "gutter_confidence": round(
                float(np.mean(layout.confidences)) if layout.confidences else 1.0,
                3,
            ),
            "method": method,
        }
    )

    logger.info(
        "Segmented %s/%s [%s] → %d cols, %d questions, %d tables%s",
        booklet_id,
        page_name,
        method,
        result.n_columns,
        result.n_questions,
        result.n_tables,
        " [HALTED]" if result.halted else "",
    )
    return result


def segment_booklet(
    booklet_id: str,
    *,
    expected_questions: Optional[Sequence[str]] = None,
    source_root: Optional[Path] = None,
    use_vlm: Optional[bool] = None,
    vlm_model: Optional[str] = None,
) -> List[SegmentResult]:
    """Segment every page of a booklet (anonymised preferred)."""
    root = source_root or ANONYMIZED_STORAGE / booklet_id
    if not root.is_dir():
        root = PREPROCESSED_STORAGE / booklet_id
    if not root.is_dir():
        logger.error("No pages found for booklet %s", booklet_id)
        return []

    pages = sorted(root.glob("page_*.png"))
    results: List[SegmentResult] = []
    for idx, page_path in enumerate(pages):
        results.append(
            segment_page(
                page_path,
                booklet_id,
                page_index=idx,
                expected_questions=expected_questions,
                use_vlm=use_vlm,
                vlm_model=vlm_model,
            )
        )
    return results


def segment_directory(
    *,
    booklet_ids: Optional[Sequence[str]] = None,
    expected_questions: Optional[Sequence[str]] = None,
    use_vlm: Optional[bool] = None,
    vlm_model: Optional[str] = None,
) -> List[SegmentResult]:
    """Segment all anonymised (or preprocessed) booklets."""
    if booklet_ids is None:
        root = ANONYMIZED_STORAGE if ANONYMIZED_STORAGE.is_dir() else PREPROCESSED_STORAGE
        booklet_ids = sorted(d.name for d in root.iterdir() if d.is_dir())

    all_results: List[SegmentResult] = []
    for bid in booklet_ids:
        all_results.extend(
            segment_booklet(
                bid,
                expected_questions=expected_questions,
                use_vlm=use_vlm,
                vlm_model=vlm_model,
            )
        )
    return all_results
