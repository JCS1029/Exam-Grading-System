"""
Phase 1 — Intake, Preprocessing, Booklet Reconciliation & Anonymisation.

Pipeline stages (run in order):
    1. rasterizer   — PDF / image → 300 DPI PNG pages
    2. preprocessor — Deskew + CLAHE + border crop
    3. reconciler   — Page-count & booklet-ID validation
    4. anonymizer   — Identity masking + pseudonym mapping
"""

from core.intake.rasterizer import rasterize_file, rasterize_directory
from core.intake.preprocessor import preprocess_page, preprocess_booklet
from core.intake.reconciler import reconcile_booklets, ReconciliationReport
from core.intake.anonymizer import anonymize_booklet, PseudonymDB

__all__ = [
    "rasterize_file",
    "rasterize_directory",
    "preprocess_page",
    "preprocess_booklet",
    "reconcile_booklets",
    "ReconciliationReport",
    "anonymize_booklet",
    "PseudonymDB",
]
