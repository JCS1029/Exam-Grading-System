"""
Question / sub-question region segmentation.

Classical-CV approach (no VLM required for the Phase 2 gate):
  1. Restrict to each column band from the gutter detector.
  2. Build a horizontal ink-density projection.
  3. Split on sustained whitespace valleys into content blocks.
  4. Classify each block as answered / blank by ink density.
  5. Assign provisional labels (Q1, Q2, …) top→bottom within RTL column order.

Coverage states stay distinct: answered, blank, not_found (the last is only
emitted when an expected question ID is missing — see pipeline coverage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.layout.gutter import ColumnBand, ColumnLayout

INK_THRESHOLD: int = 200
SMOOTH_FRAC: float = 0.008
MIN_BLOCK_HEIGHT_FRAC: float = 0.035     # ≥ 3.5% of page height
MIN_GAP_FRAC: float = 0.012              # whitespace valley ≥ 1.2% of height
VALLEY_RATIO: float = 0.28               # valley ≤ 28% of local peak
BLANK_INK_FRAC: float = 0.004            # below this → deliberately blank
EDGE_TRIM_FRAC: float = 0.02


@dataclass
class QuestionRegion:
    """A detected question or sub-question crop region."""

    question_id: str                     # e.g. "Q1", "Q1a", or expected ID
    label: str                           # human-readable provisional label
    bbox: Tuple[int, int, int, int]      # x, y, w, h in page coordinates
    column_index: int
    reading_order: int
    ink_fraction: float
    state: str = "answered"              # answered | blank
    confidence: float = 0.0
    is_subquestion: bool = False

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] * self.bbox[3])


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _horizontal_projection(gray: np.ndarray) -> np.ndarray:
    ink = (gray < INK_THRESHOLD).astype(np.float32)
    return ink.sum(axis=1)


def _smooth(proj: np.ndarray, height: int) -> np.ndarray:
    k = max(5, int(height * SMOOTH_FRAC))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(proj, kernel, mode="same")


def _split_blocks(
    smooth: np.ndarray,
    height: int,
) -> List[Tuple[int, int]]:
    """Return (y0, y1) content blocks separated by whitespace valleys."""
    trim = int(height * EDGE_TRIM_FRAC)
    work = smooth.copy()
    work[:trim] = 0
    work[height - trim :] = 0

    peak = float(np.percentile(work, 90)) if work.size else 0.0
    if peak <= 0:
        return [(0, height)]

    valley_ceil = peak * VALLEY_RATIO
    min_gap = max(8, int(height * MIN_GAP_FRAC))
    min_block = max(20, int(height * MIN_BLOCK_HEIGHT_FRAC))

    is_valley = work < valley_ceil
    # Force edges as valleys so first/last blocks are bounded
    is_valley[:trim] = True
    is_valley[height - trim :] = True

    cuts: List[int] = [0]
    i = trim
    while i < height - trim:
        if not is_valley[i]:
            i += 1
            continue
        j = i
        while j < height - trim and is_valley[j]:
            j += 1
        if (j - i) >= min_gap:
            mid = (i + j) // 2
            if mid - cuts[-1] >= min_block:
                cuts.append(mid)
        i = max(j, i + 1)
    if height - cuts[-1] >= min_block // 2:
        cuts.append(height)
    elif cuts[-1] != height:
        cuts[-1] = height

    blocks: List[Tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a >= min_block:
            blocks.append((a, b))
    if not blocks:
        blocks = [(0, height)]
    return blocks


def _ink_fraction(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    return float(np.mean(gray < INK_THRESHOLD))


def detect_question_regions(
    image: np.ndarray,
    columns: ColumnLayout,
    *,
    expected_ids: Optional[Sequence[str]] = None,
) -> List[QuestionRegion]:
    """
    Detect question-like content blocks inside each column band.

    If *expected_ids* is provided, regions are labelled with those IDs in
    reading order (extra regions get ``Q_extra_N``; missing IDs are handled
    by the pipeline coverage step, not here).
    """
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    regions: List[QuestionRegion] = []
    global_order = 0
    q_counter = 0

    # Process columns in Hebrew reading order (right→left)
    ordered_cols = sorted(columns.columns, key=lambda c: c.reading_order)

    for col in ordered_cols:
        x0, x1 = col.x0, col.x1
        # Slight inset to avoid gutter ink
        inset = max(2, int((x1 - x0) * 0.02))
        xa, xb = x0 + inset, x1 - inset
        if xb <= xa:
            xa, xb = x0, x1
        col_gray = gray[:, xa:xb]
        smooth = _smooth(_horizontal_projection(col_gray), h)
        blocks = _split_blocks(smooth, h)

        for y0, y1 in blocks:
            # Tighten vertically to ink bounds
            strip = col_gray[y0:y1, :]
            ink_rows = np.where((strip < INK_THRESHOLD).any(axis=1))[0]
            if ink_rows.size > 0:
                pad = max(4, int((y1 - y0) * 0.02))
                ty0 = max(y0, y0 + int(ink_rows[0]) - pad)
                ty1 = min(y1, y0 + int(ink_rows[-1]) + pad)
            else:
                ty0, ty1 = y0, y1

            crop = gray[ty0:ty1, xa:xb]
            ink_frac = _ink_fraction(crop)
            state = "blank" if ink_frac < BLANK_INK_FRAC else "answered"

            # Skip near-empty scanner margins at the very top/bottom with no ink
            if state == "blank" and (ty1 - ty0) < int(h * 0.05):
                continue

            q_counter += 1
            if expected_ids and (q_counter - 1) < len(expected_ids):
                qid = str(expected_ids[q_counter - 1])
                label = qid
            else:
                qid = f"Q{q_counter}"
                label = f"שאלה {q_counter}" if state != "blank" else f"ריק {q_counter}"

            # Heuristic: short blocks after a tall one → likely sub-question
            is_sub = (ty1 - ty0) < int(h * 0.08) and q_counter > 1

            conf = 0.55
            if state == "answered":
                conf = min(0.95, 0.55 + ink_frac * 8.0)
            else:
                conf = 0.7  # blank detection is fairly reliable on clean scans

            regions.append(
                QuestionRegion(
                    question_id=qid,
                    label=label,
                    bbox=(int(xa), int(ty0), int(xb - xa), int(ty1 - ty0)),
                    column_index=col.index,
                    reading_order=global_order,
                    ink_fraction=round(ink_frac, 5),
                    state=state,
                    confidence=round(conf, 3),
                    is_subquestion=is_sub,
                )
            )
            global_order += 1

    return regions
