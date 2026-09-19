"""
Step 3 — Booklet Reconciliation.

The critical correctness safeguard. Validates that:
    1. Every booklet has the expected number of pages (no silent drops / extras).
    2. No duplicate booklet IDs exist.
    3. Every anomaly is explicitly HALTED — never silently ignored.

Usage:
    from core.intake.reconciler import reconcile_booklets

    report = reconcile_booklets(manifests, expected_pages=8)
    for entry in report.entries:
        print(entry.booklet_id, entry.status)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from core.intake.rasterizer import BookletManifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------
class BookletStatus(str, Enum):
    """Reconciliation outcome for a single booklet."""

    OK = "OK"
    SHORT_BOOKLET = "SHORT_BOOKLET"       # fewer pages than expected
    EXTRA_PAGES = "EXTRA_PAGES"           # more pages than expected
    DUPLICATE_ID = "DUPLICATE_ID"         # booklet ID seen more than once
    RASTERIZE_ERROR = "RASTERIZE_ERROR"   # rasterisation had errors
    HALTED = "HALTED"                     # manually or automatically halted


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BookletEntry:
    """Reconciliation result for a single booklet."""

    booklet_id: str
    source_file: str
    page_count: int
    expected_pages: Optional[int]
    status: BookletStatus
    halt_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == BookletStatus.OK


@dataclass
class ReconciliationReport:
    """Aggregate reconciliation report for all booklets in a batch."""

    entries: List[BookletEntry] = field(default_factory=list)
    expected_pages: Optional[int] = None

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def ok_count(self) -> int:
        return sum(1 for e in self.entries if e.ok)

    @property
    def halted_count(self) -> int:
        return sum(1 for e in self.entries if not e.ok)

    @property
    def all_ok(self) -> bool:
        return self.halted_count == 0

    def get_halted(self) -> List[BookletEntry]:
        """Return all booklets that did NOT pass reconciliation."""
        return [e for e in self.entries if not e.ok]

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Reconciliation Report: {self.ok_count}/{self.total} booklets OK",
        ]
        if self.expected_pages is not None:
            lines.append(f"Expected pages per booklet: {self.expected_pages}")

        halted = self.get_halted()
        if halted:
            lines.append(f"\n⚠ {len(halted)} booklet(s) HALTED:")
            for entry in halted:
                lines.append(
                    f"  • {entry.booklet_id}: {entry.status.value}"
                    f" ({entry.page_count} pages)"
                    f" — {entry.halt_reason or 'unknown reason'}"
                )
        else:
            lines.append("✓ All booklets passed reconciliation.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core reconciliation logic
# ---------------------------------------------------------------------------
def reconcile_booklets(
    manifests: List[BookletManifest],
    *,
    expected_pages: Optional[int] = None,
) -> ReconciliationReport:
    """
    Validate a list of rasterised booklet manifests.

    Checks performed:
        1. **Page count audit** — if *expected_pages* is set, any booklet
           with fewer or more pages is HALTED.
        2. **Duplicate ID detection** — if two manifests share the same
           ``booklet_id``, both are HALTED.
        3. **Rasterisation error check** — booklets with errors from
           rasterisation are HALTED.

    Parameters
    ----------
    manifests : list[BookletManifest]
        Manifests produced by :func:`rasterize_file` or
        :func:`rasterize_directory`.
    expected_pages : int, optional
        If set, every booklet must have exactly this many pages.

    Returns
    -------
    ReconciliationReport
        Per-booklet statuses and an aggregate summary.
    """
    report = ReconciliationReport(expected_pages=expected_pages)
    seen_ids: Dict[str, int] = {}  # booklet_id → count

    # First pass: count IDs for duplicate detection
    for m in manifests:
        seen_ids[m.booklet_id] = seen_ids.get(m.booklet_id, 0) + 1

    # Second pass: evaluate each booklet
    for m in manifests:
        entry = _evaluate_booklet(m, expected_pages=expected_pages, id_count=seen_ids.get(m.booklet_id, 1))
        report.entries.append(entry)

        if not entry.ok:
            logger.warning(
                "HALTED booklet %s: %s — %s",
                entry.booklet_id, entry.status.value, entry.halt_reason,
            )
        else:
            logger.info(
                "Booklet %s: OK (%d pages)",
                entry.booklet_id, entry.page_count,
            )

    logger.info(report.summary())
    return report


def _evaluate_booklet(
    manifest: BookletManifest,
    *,
    expected_pages: Optional[int],
    id_count: int,
) -> BookletEntry:
    """Evaluate a single booklet manifest and return its reconciliation entry."""

    # Check for rasterisation errors first
    if manifest.errors:
        return BookletEntry(
            booklet_id=manifest.booklet_id,
            source_file=manifest.source_file,
            page_count=manifest.page_count,
            expected_pages=expected_pages,
            status=BookletStatus.RASTERIZE_ERROR,
            halt_reason=f"Rasterisation errors: {'; '.join(manifest.errors)}",
        )

    # Check for duplicate IDs
    if id_count > 1:
        return BookletEntry(
            booklet_id=manifest.booklet_id,
            source_file=manifest.source_file,
            page_count=manifest.page_count,
            expected_pages=expected_pages,
            status=BookletStatus.DUPLICATE_ID,
            halt_reason=f"Booklet ID '{manifest.booklet_id}' appears {id_count} times",
        )

    # Check page count
    if expected_pages is not None:
        if manifest.page_count < expected_pages:
            return BookletEntry(
                booklet_id=manifest.booklet_id,
                source_file=manifest.source_file,
                page_count=manifest.page_count,
                expected_pages=expected_pages,
                status=BookletStatus.SHORT_BOOKLET,
                halt_reason=(
                    f"Expected {expected_pages} pages, got {manifest.page_count} "
                    f"(missing {expected_pages - manifest.page_count} page(s))"
                ),
            )
        elif manifest.page_count > expected_pages:
            return BookletEntry(
                booklet_id=manifest.booklet_id,
                source_file=manifest.source_file,
                page_count=manifest.page_count,
                expected_pages=expected_pages,
                status=BookletStatus.EXTRA_PAGES,
                halt_reason=(
                    f"Expected {expected_pages} pages, got {manifest.page_count} "
                    f"({manifest.page_count - expected_pages} extra page(s))"
                ),
            )

    # All checks passed
    return BookletEntry(
        booklet_id=manifest.booklet_id,
        source_file=manifest.source_file,
        page_count=manifest.page_count,
        expected_pages=expected_pages,
        status=BookletStatus.OK,
    )
