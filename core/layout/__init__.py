"""
Phase 2 — Layout Analysis, Column Splitting, & Question Segmentation.

Pipeline stages (run in order on anonymised pages):
    1. gutter         — OpenCV vertical ink-density prior
    2. vlm_segmenter  — VLM finds real שאלה N boxes (planned Phase 2)
    3. questions/tables — CV fallbacks when VLM disabled/unavailable
    4. pipeline       — crop writer + coverage states
"""

from core.layout.gutter import detect_columns, ColumnLayout
from core.layout.questions import detect_question_regions, QuestionRegion
from core.layout.tables import detect_tables, TableRegion
from core.layout.pipeline import (
    segment_page,
    segment_booklet,
    SegmentResult,
    CoverageState,
)

__all__ = [
    "detect_columns",
    "ColumnLayout",
    "detect_question_regions",
    "QuestionRegion",
    "detect_tables",
    "TableRegion",
    "segment_page",
    "segment_booklet",
    "SegmentResult",
    "CoverageState",
]
