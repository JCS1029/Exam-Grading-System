"""
Phase 1 Pipeline Runner — CLI entry point.

Orchestrates the four Phase 1 stages in sequence:
    1. Rasterise   → storage/raw/
    2. Preprocess  → storage/preprocessed/
    3. Reconcile   → validation report
    4. Anonymise   → storage/anonymized/ + pseudonym DB

Usage:
    python scripts/run_phase1.py --input docs/ --expected-pages 8
    python scripts/run_phase1.py --input docs/test1.pdf
    python scripts/run_phase1.py --input docs/ --skip-anonymization
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.intake.rasterizer import rasterize_file, rasterize_directory, BookletManifest
from core.intake.preprocessor import preprocess_booklet, PreprocessResult
from core.intake.reconciler import reconcile_booklets, BookletStatus
from core.intake.anonymizer import anonymize_booklet, PseudonymDB


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_phase1(
    input_path: Path,
    *,
    expected_pages: int | None = None,
    skip_reconciliation: bool = False,
    skip_anonymization: bool = False,
    dpi: int = 300,
) -> dict:
    """
    Run the full Phase 1 pipeline.

    Returns a summary dict with counts and any halted booklets.
    """
    logger = logging.getLogger("phase1")
    start_time = time.time()

    # ── Step 1: Rasterise ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Ingestion & Rasterisation")
    logger.info("=" * 60)

    manifests: list[BookletManifest] = []

    if input_path.is_dir():
        manifests = rasterize_directory(input_path, dpi=dpi)
    elif input_path.is_file():
        manifests = [rasterize_file(input_path, dpi=dpi)]
    else:
        logger.error("Input path does not exist: %s", input_path)
        return {"error": f"Input path not found: {input_path}"}

    total_pages = sum(m.page_count for m in manifests)
    raster_errors = sum(len(m.errors) for m in manifests)
    logger.info(
        "Rasterised %d file(s) → %d pages (%d errors)",
        len(manifests), total_pages, raster_errors,
    )

    # ── Step 2: Preprocess ────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Image Preprocessing (Deskew + CLAHE + Border Crop)")
    logger.info("=" * 60)

    all_preprocess_results: dict[str, list[PreprocessResult]] = {}
    for m in manifests:
        results = preprocess_booklet(m.booklet_id)
        all_preprocess_results[m.booklet_id] = results

    preprocess_ok = sum(
        sum(1 for r in results if r.ok)
        for results in all_preprocess_results.values()
    )
    logger.info("Preprocessed %d/%d pages successfully", preprocess_ok, total_pages)

    # ── Step 3: Reconcile ─────────────────────────────────────────────
    halted_ids: set[str] = set()

    if skip_reconciliation:
        logger.info("")
        logger.info("STEP 3: Reconciliation SKIPPED (--skip-reconciliation)")
    else:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3: Booklet Reconciliation")
        logger.info("=" * 60)

        report = reconcile_booklets(manifests, expected_pages=expected_pages)
        halted_ids = {e.booklet_id for e in report.get_halted()}

        logger.info(report.summary())

    # ── Step 4: Anonymise ─────────────────────────────────────────────
    if skip_anonymization:
        logger.info("")
        logger.info("STEP 4: Anonymisation SKIPPED (--skip-anonymization)")
    else:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 4: Anonymisation")
        logger.info("=" * 60)

        with PseudonymDB() as db:
            for m in manifests:
                if m.booklet_id in halted_ids:
                    logger.warning(
                        "Skipping anonymisation for HALTED booklet: %s",
                        m.booklet_id,
                    )
                    continue

                result = anonymize_booklet(m.booklet_id, db)
                if result.ok:
                    logger.info(
                        "✓ %s → %s (%d pages anonymised)",
                        m.booklet_id, result.pseudonym, result.pages_processed,
                    )
                else:
                    logger.error(
                        "✗ %s: %s",
                        m.booklet_id, "; ".join(result.errors),
                    )

            # Print mapping summary
            mappings = db.get_all_mappings()
            if mappings:
                logger.info("")
                logger.info("Pseudonym Mappings (stored locally):")
                for m in mappings:
                    logger.info(
                        "  %s ↔ booklet %s",
                        m["pseudonym"], m["booklet_id"],
                    )

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 2)
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)
    logger.info("  Files processed:  %d", len(manifests))
    logger.info("  Total pages:      %d", total_pages)
    logger.info("  Pages preprocessed OK: %d", preprocess_ok)
    logger.info("  Booklets halted:  %d", len(halted_ids))
    logger.info("  Elapsed time:     %.2fs", elapsed)

    return {
        "files_processed": len(manifests),
        "total_pages": total_pages,
        "preprocess_ok": preprocess_ok,
        "booklets_halted": len(halted_ids),
        "halted_ids": list(halted_ids),
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 — Intake, Preprocessing, Reconciliation & Anonymisation",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to a PDF/image file or directory of files to process",
    )
    parser.add_argument(
        "--expected-pages", "-p",
        type=int,
        default=None,
        help="Expected number of pages per booklet (triggers HALT on mismatch)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF rasterisation (default: 300)",
    )
    parser.add_argument(
        "--skip-reconciliation",
        action="store_true",
        help="Skip the reconciliation step",
    )
    parser.add_argument(
        "--skip-anonymization",
        action="store_true",
        help="Skip the anonymisation step",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    summary = run_phase1(
        args.input,
        expected_pages=args.expected_pages,
        skip_reconciliation=args.skip_reconciliation,
        skip_anonymization=args.skip_anonymization,
        dpi=args.dpi,
    )

    # Exit with error code if any booklets were halted
    if summary.get("booklets_halted", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
