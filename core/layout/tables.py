"""
Table / grid detection for cover sheets and score rosters.

Uses morphological line extraction (horizontal + vertical) to find dense
grid regions. Classifies a region as a table when both line directions
form a sufficiently regular lattice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

INK_THRESHOLD: int = 200
MIN_TABLE_AREA_FRAC: float = 0.02
MIN_H_LINES: int = 3
MIN_V_LINES: int = 2
LINE_MIN_FRAC: float = 0.12          # line length ≥ 12% of page dimension


@dataclass
class TableRegion:
    """A detected tabular / grid region on the page."""

    index: int
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    n_horizontal_lines: int = 0
    n_vertical_lines: int = 0
    confidence: float = 0.0
    kind: str = "grid"                # grid | score_roster | unknown

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] * self.bbox[3])


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def detect_tables(image: np.ndarray) -> List[TableRegion]:
    """
    Detect grid-like table regions on a page.

    Returns zero or more :class:`TableRegion` objects sorted top→bottom.
    """
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    # Invert so ink is white for morphology
    binary = cv2.threshold(gray, INK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]

    h_len = max(20, int(w * LINE_MIN_FRAC))
    v_len = max(20, int(h * LINE_MIN_FRAC * 0.5))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    grid = cv2.bitwise_or(h_lines, v_lines)

    # Dilate to merge nearby line fragments into table blobs
    merge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 80), max(5, h // 80)))
    grid = cv2.dilate(grid, merge, iterations=2)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(h * w * MIN_TABLE_AREA_FRAC)
    tables: List[TableRegion] = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < min_area:
            continue
        if bw < w * 0.2 or bh < h * 0.05:
            continue

        roi_h = h_lines[y : y + bh, x : x + bw]
        roi_v = v_lines[y : y + bh, x : x + bw]

        # Count distinct line rows / columns via projection peaks
        h_proj = (roi_h > 0).sum(axis=1)
        v_proj = (roi_v > 0).sum(axis=0)
        n_h = _count_peaks(h_proj, min_gap=max(8, bh // 40))
        n_v = _count_peaks(v_proj, min_gap=max(8, bw // 40))

        if n_h < MIN_H_LINES or n_v < MIN_V_LINES:
            continue

        conf = min(1.0, 0.35 * (n_h / 6.0) + 0.35 * (n_v / 4.0) + 0.3)
        kind = "score_roster" if (y < h * 0.45 and bw > w * 0.4) else "grid"
        tables.append(
            TableRegion(
                index=len(tables),
                bbox=(int(x), int(y), int(bw), int(bh)),
                n_horizontal_lines=n_h,
                n_vertical_lines=n_v,
                confidence=round(conf, 3),
                kind=kind,
            )
        )

    tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    for i, t in enumerate(tables):
        t.index = i
    return tables


def _count_peaks(proj: np.ndarray, min_gap: int) -> int:
    if proj.size == 0:
        return 0
    thr = max(1.0, float(np.percentile(proj, 75)) * 0.5)
    peaks = 0
    last = -min_gap
    for i, v in enumerate(proj):
        if v >= thr and (i - last) >= min_gap:
            peaks += 1
            last = i
    return peaks
