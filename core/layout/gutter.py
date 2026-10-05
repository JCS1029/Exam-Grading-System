"""
Column / gutter detection via vertical ink-density projection.

Hebrew multi-column exams typically place the right column first in reading
order. A sustained low-density trough between ink-dense bands is treated as a
gutter. Deterministic, fast, and free — the Phase 2 prior before any VLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INK_THRESHOLD: int = 200                 # gray < this counts as ink
SMOOTH_FRAC: float = 0.012               # smoothing window as fraction of width
MIN_GUTTER_WIDTH_FRAC: float = 0.015     # gutter must be ≥ 1.5% of page width
MAX_GUTTER_WIDTH_FRAC: float = 0.18
MIN_COLUMN_WIDTH_FRAC: float = 0.18      # each column ≥ 18% of page width
TROUGH_RATIO: float = 0.55               # trough ink ≤ 55% of neighbouring peaks
EDGE_MARGIN_FRAC: float = 0.06           # ignore outer 6% (scanner margins)


@dataclass
class ColumnBand:
    """One vertical reading band (column) on a page."""

    index: int
    x0: int
    x1: int
    reading_order: int                   # 0 = first to read (RTL → rightmost)

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    def as_bbox(self, page_h: int) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) covering the full page height."""
        return (self.x0, 0, self.width, page_h)


@dataclass
class ColumnLayout:
    """Result of gutter / column detection for one page."""

    n_columns: int
    columns: List[ColumnBand] = field(default_factory=list)
    gutters: List[Tuple[int, int]] = field(default_factory=list)  # (x0, x1)
    projection: Optional[List[float]] = None
    confidences: List[float] = field(default_factory=list)
    is_multi_column: bool = False

    @property
    def ok(self) -> bool:
        return self.n_columns >= 1 and all(c.width > 0 for c in self.columns)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _vertical_projection(gray: np.ndarray) -> np.ndarray:
    ink = (gray < INK_THRESHOLD).astype(np.float32)
    return ink.sum(axis=0)


def _smooth(proj: np.ndarray, width: int) -> np.ndarray:
    k = max(5, int(width * SMOOTH_FRAC))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(proj, kernel, mode="same")


def _find_gutter_intervals(
    smooth: np.ndarray,
    width: int,
) -> List[Tuple[int, int, float]]:
    """
    Find candidate gutter intervals as sustained troughs in the projection.

    Returns list of (x0, x1, confidence) sorted left→right.
    """
    margin = int(width * EDGE_MARGIN_FRAC)
    search = smooth.copy()
    search[:margin] = np.inf
    search[width - margin :] = np.inf

    peak = float(np.percentile(smooth[margin : width - margin], 90))
    if peak <= 0:
        return []

    trough_ceil = peak * TROUGH_RATIO
    min_w = int(width * MIN_GUTTER_WIDTH_FRAC)
    max_w = int(width * MAX_GUTTER_WIDTH_FRAC)
    min_col = int(width * MIN_COLUMN_WIDTH_FRAC)

    # Binary mask of trough candidates
    is_trough = search < trough_ceil
    intervals: List[Tuple[int, int, float]] = []

    i = margin
    while i < width - margin:
        if not is_trough[i]:
            i += 1
            continue
        j = i
        while j < width - margin and is_trough[j]:
            j += 1
        gw = j - i
        if min_w <= gw <= max_w:
            left_peak = float(np.max(smooth[max(margin, i - min_col) : i])) if i > margin else 0.0
            right_peak = float(np.max(smooth[j : min(width - margin, j + min_col)])) if j < width - margin else 0.0
            trough_val = float(np.mean(smooth[i:j]))
            neighbour = max(left_peak, right_peak, 1.0)
            # Require ink on both sides (real columns), not just page edge
            if left_peak > peak * 0.25 and right_peak > peak * 0.25:
                conf = max(0.0, min(1.0, 1.0 - (trough_val / neighbour)))
                intervals.append((i, j, conf))
        i = j

    # Prefer the strongest single mid-page gutter for 2-column layouts
    if len(intervals) > 1:
        mid = width / 2.0
        intervals.sort(key=lambda t: (-t[2], abs(((t[0] + t[1]) / 2) - mid)))
        # Keep non-overlapping intervals with highest confidence
        kept: List[Tuple[int, int, float]] = []
        for cand in intervals:
            if all(cand[1] < k[0] or cand[0] > k[1] for k in kept):
                # Also require resulting columns wide enough
                xs = sorted([0] + [c[0] for c in kept] + [c[1] for c in kept] + [cand[0], cand[1]] + [width])
                # Defer full column-width check to assembly; keep top 2 for now
                kept.append(cand)
            if len(kept) >= 2:
                break
        intervals = sorted(kept, key=lambda t: t[0])
    else:
        intervals = sorted(intervals, key=lambda t: t[0])

    return intervals


def _columns_from_gutters(
    gutters: List[Tuple[int, int, float]],
    width: int,
) -> Optional[Tuple[List[ColumnBand], List[Tuple[int, int]], List[float]]]:
    """
    Build RTL-ordered columns from a gutter set, or None if widths are invalid.
    """
    if not gutters:
        return None

    min_col = int(width * MIN_COLUMN_WIDTH_FRAC)
    cuts = [0]
    confidences: List[float] = []
    gutter_spans: List[Tuple[int, int]] = []
    for x0, x1, conf in sorted(gutters, key=lambda t: t[0]):
        mid = (x0 + x1) // 2
        cuts.append(mid)
        confidences.append(conf)
        gutter_spans.append((x0, x1))
    cuts.append(width)
    cuts = sorted(set(cuts))

    if len(cuts) < 3:
        return None
    if not all((cuts[i + 1] - cuts[i]) >= min_col for i in range(len(cuts) - 1)):
        return None

    raw_cols: List[ColumnBand] = []
    for i in range(len(cuts) - 1):
        raw_cols.append(ColumnBand(index=i, x0=cuts[i], x1=cuts[i + 1], reading_order=i))
    n = len(raw_cols)
    for col in raw_cols:
        col.reading_order = (n - 1) - col.index
    return (
        sorted(raw_cols, key=lambda c: c.reading_order),
        gutter_spans,
        confidences,
    )


def detect_columns(image: np.ndarray) -> ColumnLayout:
    """
    Detect column bands on a page image.

    Single-column pages return one band spanning the full width.
    Multi-column pages return bands ordered for Hebrew RTL reading
    (rightmost column = reading_order 0).
    """
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    proj = _vertical_projection(gray)
    smooth = _smooth(proj, w)

    gutters = _find_gutter_intervals(smooth, w)

    # Prefer a valid multi-gutter split; if that yields a too-narrow edge
    # strip (common with scanner margins), fall back to the single best
    # mid-page gutter — the usual 2-column Hebrew layout.
    assembled = _columns_from_gutters(gutters, w)
    if assembled is None and gutters:
        mid = w / 2.0
        ranked = sorted(
            gutters,
            key=lambda t: (-t[2], abs(((t[0] + t[1]) / 2) - mid)),
        )
        for cand in ranked:
            assembled = _columns_from_gutters([cand], w)
            if assembled is not None:
                break

    if assembled is None:
        columns = [ColumnBand(index=0, x0=0, x1=w, reading_order=0)]
        return ColumnLayout(
            n_columns=1,
            columns=columns,
            gutters=[],
            projection=None,
            confidences=[],
            is_multi_column=False,
        )

    columns, gutter_spans, confidences = assembled
    n = len(columns)
    return ColumnLayout(
        n_columns=n,
        columns=columns,
        gutters=gutter_spans,
        projection=None,
        confidences=confidences,
        is_multi_column=n > 1,
    )
