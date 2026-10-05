"""
Phase 2 Pipeline Runner — Layout Analysis + VLM Question Segmentation.

Usage:
    python scripts/run_phase2.py
    python scripts/run_phase2.py --booklets 02 --pages page_003
    python scripts/run_phase2.py --expected-questions 7,8,9,10
    python scripts/run_phase2.py --no-vlm          # classical CV only (debug)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.layout.pipeline import segment_directory, segment_page, ANONYMIZED_STORAGE, PREPROCESSED_STORAGE
from core.vlm import GEMINI_API_KEY, get_default_model


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_page_path(booklet_id: str, page_name: str) -> Path:
    for root in (ANONYMIZED_STORAGE, PREPROCESSED_STORAGE):
        candidate = root / booklet_id / page_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Page not found: {booklet_id}/{page_name}")


def run_phase2(
    *,
    booklet_ids: list[str] | None = None,
    pages: list[str] | None = None,
    expected_questions: list[str] | None = None,
    use_vlm: bool | None = None,
    vlm_model: str | None = None,
) -> dict:
    logger = logging.getLogger("phase2")
    start = time.time()

    logger.info("=" * 60)
    logger.info("PHASE 2: Layout Analysis + VLM Question Segmentation")
    logger.info("=" * 60)
    effective_vlm = use_vlm if use_vlm is not None else bool(GEMINI_API_KEY)
    logger.info(
        "VLM: %s | model=%s | key=%s",
        "ON" if effective_vlm else "OFF (CV fallback)",
        vlm_model or (get_default_model() if effective_vlm else "n/a"),
        "present" if GEMINI_API_KEY else "MISSING",
    )

    results = []
    if pages:
        if not booklet_ids or len(booklet_ids) != 1:
            raise ValueError("--pages requires exactly one --booklets id")
        bid = booklet_ids[0]
        for i, page_name in enumerate(pages):
            path = _resolve_page_path(bid, page_name)
            results.append(
                segment_page(
                    path,
                    bid,
                    page_index=i,
                    expected_questions=expected_questions,
                    use_vlm=effective_vlm,
                    vlm_model=vlm_model,
                )
            )
    else:
        results = segment_directory(
            booklet_ids=booklet_ids,
            expected_questions=expected_questions,
            use_vlm=effective_vlm,
            vlm_model=vlm_model,
        )

    if not results:
        logger.error("No pages segmented — run Phase 1 first.")
        return {"error": "no_pages", "pages": 0}

    ok = sum(1 for r in results if r.ok)
    halted = [r for r in results if r.halted]
    multi = sum(1 for r in results if r.is_multi_column)
    vlm_n = sum(1 for r in results if r.method == "vlm")
    total_q = sum(r.n_questions for r in results)
    total_t = sum(r.n_tables for r in results)
    answered = sum(r.metrics.get("n_answered", 0) for r in results)
    blank = sum(r.metrics.get("n_blank", 0) for r in results)
    not_found = sum(r.metrics.get("n_not_found", 0) for r in results)
    cache_hits = sum(1 for r in results if r.metrics.get("vlm_cache_hit"))
    prompt_tokens = sum(r.metrics.get("vlm_prompt_tokens") or 0 for r in results)
    completion_tokens = sum(r.metrics.get("vlm_candidate_tokens") or 0 for r in results)
    api_calls = sum(
        1
        for r in results
        if r.method == "vlm" and not r.metrics.get("vlm_cache_hit")
    )

    summary = {
        "pages": len(results),
        "pages_ok": ok,
        "pages_halted": len(halted),
        "multi_column_pages": multi,
        "vlm_pages": vlm_n,
        "cv_fallback_pages": len(results) - vlm_n,
        "total_question_regions": total_q,
        "total_tables": total_t,
        "coverage": {
            "answered": answered,
            "blank": blank,
            "not_found": not_found,
        },
        "elapsed_sec": round(time.time() - start, 2),
        "booklets": sorted({r.booklet_id for r in results}),
        "use_vlm": effective_vlm,
        "model": vlm_model or (get_default_model() if effective_vlm else None),
        "vlm_cache_hits": cache_hits,
        "vlm_api_calls": api_calls,
        "vlm_prompt_tokens": prompt_tokens,
        "vlm_completion_tokens": completion_tokens,
        "vlm_total_tokens": prompt_tokens + completion_tokens,
    }

    reports = Path("storage/reports")
    reports.mkdir(parents=True, exist_ok=True)
    summary_path = reports / "phase2_results.json"
    payload = {
        "summary": summary,
        "pages": [
            {
                "booklet_id": r.booklet_id,
                "page_name": r.page_name,
                "method": r.method,
                "page_summary": r.page_summary,
                "n_columns": r.n_columns,
                "n_questions": r.n_questions,
                "n_tables": r.n_tables,
                "is_multi_column": r.is_multi_column,
                "coverage": r.coverage,
                "question_ids": list(r.coverage.keys()),
                "halted": r.halted,
                "halt_reasons": r.halt_reasons,
                "errors": r.errors,
                "metrics": r.metrics,
                "output_dir": r.output_dir,
            }
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2 COMPLETE")
    logger.info("=" * 60)
    logger.info("  Pages segmented:     %d (%d OK)", len(results), ok)
    logger.info("  Via VLM / CV:        %d / %d", vlm_n, len(results) - vlm_n)
    logger.info("  Multi-column pages:  %d", multi)
    logger.info("  Question regions:    %d", total_q)
    logger.info("  Tables detected:     %d", total_t)
    logger.info(
        "  Coverage answered/blank/not_found: %d / %d / %d",
        answered,
        blank,
        not_found,
    )
    logger.info("  Halted pages:        %d", len(halted))
    if effective_vlm:
        logger.info(
            "  VLM cache hits:      %d/%d (API calls: %d)",
            cache_hits,
            vlm_n,
            api_calls,
        )
        logger.info(
            "  VLM tokens (run):    %d prompt + %d completion = %d total",
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
        )
        if cache_hits:
            logger.info(
                "  Note: cached pages report 0 new tokens; totals above are this run only."
            )
    logger.info("  Summary written:     %s", summary_path)
    logger.info("  Elapsed:             %.2fs", summary["elapsed_sec"])

    for r in results:
        qids = ",".join(r.coverage.keys())
        logger.info(
            "  • %s/%s [%s] cols=%d qs=[%s]",
            r.booklet_id,
            r.page_name,
            r.method,
            r.n_columns,
            qids,
        )

    for r in halted:
        for reason in r.halt_reasons:
            logger.warning("HALT %s/%s: %s", r.booklet_id, r.page_name, reason)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — Layout & VLM Question Segmentation")
    parser.add_argument("--booklets", type=str, default=None, help="Comma-separated booklet IDs")
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Comma-separated page filenames (requires single --booklets)",
    )
    parser.add_argument(
        "--expected-questions",
        type=str,
        default=None,
        help="Comma-separated expected question IDs for coverage validation",
    )
    parser.add_argument("--model", type=str, default=None, help="Override VLM model id")
    parser.add_argument("--no-vlm", action="store_true", help="Force classical-CV only")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    booklet_ids = (
        [b.strip() for b in args.booklets.split(",") if b.strip()] if args.booklets else None
    )
    pages = [p.strip() for p in args.pages.split(",") if p.strip()] if args.pages else None
    expected = (
        [q.strip() for q in args.expected_questions.split(",") if q.strip()]
        if args.expected_questions
        else None
    )

    summary = run_phase2(
        booklet_ids=booklet_ids,
        pages=pages,
        expected_questions=expected,
        use_vlm=False if args.no_vlm else None,
        vlm_model=args.model,
    )
    if summary.get("pages_halted", 0) > 0 or summary.get("error"):
        sys.exit(1)
    # Soft fail if VLM was requested but every page fell back to CV
    if not args.no_vlm and GEMINI_API_KEY and summary.get("vlm_pages", 0) == 0:
        logging.getLogger("phase2").error("VLM was enabled but 0 pages used it.")
        sys.exit(2)


if __name__ == "__main__":
    main()
