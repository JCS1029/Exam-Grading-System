"""
Answer-grid reader for multiple-choice cover sheets ("סמנו כאן בלבד!").

Classical CV, deterministic and free:
    1. Locate the printed answer table via long horizontal / vertical lines.
    2. Split it into rows (questions) and option columns (א–ד, right-to-left).
    3. Measure each box's ink relative to the sheet's own empty boxes, and
       how solidly the printed checkbox interior is filled.
    4. Resolve each row: one mark → answer; a solid fill next to an X → the
       solid box is a cancellation; anything else → human review.

A VLM cross-check (see ``vlm_read_grid``) runs on the same crop; disagreement
between the two readers is routed to review rather than silently resolved.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Columns left→right on the page; Hebrew tables read right→left, so the
# question-number column is the rightmost one.
OPTION_COLUMNS_LTR: Tuple[str, ...] = ("ד", "ג", "ב", "א")
N_QUESTIONS: int = int(os.getenv("OMR_N_QUESTIONS", "10"))

# Geometry relative to row height; the printed checkbox is ~0.22 × row height.
PROBE_HALF_FRAC: float = 0.30        # square around the checkbox where marks land
CORE_HALF_FRAC: float = 0.07         # strictly inside the checkbox
# Set from the 13 hand-labelled pilot sheets: empty boxes peak at 0.021 excess,
# the faintest real mark sits at 0.033. Re-derive when new sheets are labelled.
MARK_EXCESS_MIN: float = 0.027       # ink above the empty-box baseline → marked
MARGIN_REVIEW: float = 0.005         # |excess - threshold| below this → low confidence
SOLID_GAP_MIN: float = 0.08          # interior-darkness gap that marks a cancellation


class RowStatus(str, Enum):
    ANSWERED = "answered"
    BLANK = "blank"
    AMBIGUOUS = "ambiguous"


@dataclass
class CellReading:
    letter: str
    ink: float              # dark-pixel fraction of the inner cell
    excess: float           # ink minus the sheet's empty-box baseline
    core_fill: float        # dark fraction of the checkbox interior
    marked: bool = False
    solid: bool = False


@dataclass
class RowReading:
    question_id: str
    cells: List[CellReading]
    status: RowStatus = RowStatus.BLANK
    chosen: Optional[str] = None
    cancelled: List[str] = field(default_factory=list)
    low_margin: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "status": self.status.value,
            "chosen": self.chosen,
            "cancelled": self.cancelled,
            "low_margin": self.low_margin,
            "reason": self.reason,
            "cells": [
                {
                    "letter": c.letter,
                    "excess": round(c.excess, 4),
                    "core_fill": round(c.core_fill, 3),
                    "marked": c.marked,
                    "solid": c.solid,
                }
                for c in self.cells
            ],
        }


@dataclass
class GridReading:
    image_path: str
    grid_bbox: Optional[Tuple[int, int, int, int]] = None
    rows: List[RowReading] = field(default_factory=list)
    baseline_ink: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and len(self.rows) == N_QUESTIONS

    def answers(self) -> Dict[str, Optional[str]]:
        return {r.question_id: r.chosen for r in self.rows}

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "grid_bbox": list(self.grid_bbox) if self.grid_bbox else None,
            "baseline_ink": round(self.baseline_ink, 4),
            "rows": [r.to_dict() for r in self.rows],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Grid location
# ---------------------------------------------------------------------------
def _binarize(gray: np.ndarray) -> np.ndarray:
    """Ink = 255. Otsu handles pencil, pen and coloured markers alike."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _line_peaks(mask: np.ndarray, axis: int, min_gap: int) -> List[int]:
    """
    Positions of table lines along one axis of a rectified table crop.

    Peaks are taken relative to the strongest line, so faint scans and thick
    pens both work; the borders at the crop edges are always candidates.
    """
    proj = (mask > 0).mean(axis=axis).astype(np.float32)
    proj = np.convolve(proj, np.ones(5, np.float32) / 5.0, mode="same")
    if proj.max() <= 0:
        return []
    thr = 0.35 * float(proj.max())
    peaks: List[int] = []
    for i in np.argsort(proj)[::-1]:
        if proj[i] < thr:
            break
        if all(abs(int(i) - p) >= min_gap for p in peaks):
            peaks.append(int(i))
    n = proj.size
    for edge in (0, n - 1):
        if all(abs(edge - p) >= min_gap for p in peaks):
            peaks.append(edge)
    return sorted(peaks)


def _line_masks(binary: np.ndarray, h_len: int, v_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Horizontal / vertical ruling masks.

    Thin printed rules on phone scans binarise as dotted runs; closing along
    the line direction first bridges those gaps so the opening keeps them.
    """
    h_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    v_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))
    h_mask = cv2.morphologyEx(
        h_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    )
    v_mask = cv2.morphologyEx(
        v_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    )
    return h_mask, v_mask


def _order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


def _regularise(lines: List[int], expected: int) -> Optional[List[int]]:
    """
    Pick ``expected`` lines that form the most even lattice.

    Stray long strokes (a student's underline, the scan edge) add extra
    candidates; the true table lines are the evenly spaced subset.
    """
    if len(lines) < expected:
        return None
    if len(lines) == expected:
        return lines
    best: Optional[List[int]] = None
    best_cost = float("inf")
    # Lines are sorted; the table is a contiguous run of the candidates once
    # strays outside it are skipped, so test every start/end pair.
    for i in range(len(lines)):
        for j in range(i + expected - 1, len(lines)):
            span = lines[j] - lines[i]
            step = span / (expected - 1)
            chosen = []
            for k in range(expected):
                target = lines[i] + k * step
                chosen.append(min(lines[i : j + 1], key=lambda v: abs(v - target)))
            if len(set(chosen)) != expected:
                continue
            cost = float(np.std(np.diff(chosen))) / max(step, 1.0)
            if cost < best_cost:
                best_cost, best = cost, chosen
    if best is None or best_cost > 0.15:
        return None
    return best


@dataclass
class LocatedGrid:
    table_gray: np.ndarray               # rectified (axis-aligned) table crop
    corners: np.ndarray                  # table corners in page pixels (tl, tr, br, bl)
    row_lines: List[int]                 # in table_gray coordinates
    col_lines: List[int]


def _find_table_corner_candidates(binary: np.ndarray) -> List[np.ndarray]:
    """
    Candidate corner sets for the answer table, best first.

    Stray strokes touching the table (a student's arrow, bleed-through) can
    distort its outline, so several fits are offered and the caller keeps the
    first that yields a regular ruled grid.
    """
    h, w = binary.shape
    h_mask, v_mask = _line_masks(binary, max(15, w // 30), max(15, h // 40))
    # Thin, broken ruling on low-resolution scans must still merge into one
    # table outline; the joint is scaled to the page.
    joint = max(5, w // 120)
    grid = cv2.dilate(cv2.bitwise_or(h_mask, v_mask), np.ones((joint, joint), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 0.3 * w or bh < 0.2 * h or bh > 0.85 * h:
            continue
        area = float(bw * bh)
        if area > best_area:
            best, best_area = cnt, area
    if best is None:
        return []
    candidates: List[np.ndarray] = []
    hull = cv2.convexHull(best)
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(approx) == 4:
        candidates.append(_order_corners(approx.reshape(4, 2).astype(np.float32)))
    candidates.append(_order_corners(cv2.boxPoints(cv2.minAreaRect(best)).astype(np.float32)))
    x, y, bw, bh = cv2.boundingRect(best)
    candidates.append(
        np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
    )
    return candidates


def locate_grid(gray: np.ndarray) -> Tuple[Optional[LocatedGrid], List[str]]:
    """Find, rectify and rule the answer table. Returns (grid, errors)."""
    binary = _binarize(gray)
    # The anonymisation mask is a solid black band; it would read as one huge
    # table and swamp the real lines.
    binary[(binary > 0).mean(axis=1) > 0.9] = 0

    candidates = _find_table_corner_candidates(binary)
    if not candidates:
        return None, ["answer table not found on page"]
    errors: List[str] = []
    for corners in candidates:
        grid, errs = _rule_table(gray, corners)
        if grid is not None:
            return grid, []
        errors = errs
    return None, errors


def _rule_table(gray: np.ndarray, corners: np.ndarray) -> Tuple[Optional[LocatedGrid], List[str]]:
    """Rectify the table at ``corners`` and find its row / column lines."""
    tw = int(round(max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3]))))
    th = int(round(max(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1]))))
    dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    table_gray = cv2.warpPerspective(gray, matrix, (tw, th), borderValue=255)
    table_bin = _binarize(table_gray)

    h_mask, v_mask = _line_masks(table_bin, max(15, tw // 3), max(15, th // 3))
    n_rows = N_QUESTIONS + 2
    rows = _line_peaks(h_mask, axis=1, min_gap=max(4, th // (n_rows * 2)))
    cols = _line_peaks(v_mask, axis=0, min_gap=max(4, tw // 12))

    errors: List[str] = []
    row_lines = _regularise(rows, n_rows)
    col_lines = _regularise_columns(cols)
    if row_lines is None:
        errors.append(f"answer table rows not resolved ({len(rows)} lines, expected {n_rows})")
    if col_lines is None:
        errors.append(
            f"answer table columns not resolved ({len(cols)} lines, "
            f"expected {len(OPTION_COLUMNS_LTR) + 2})"
        )
    if errors:
        return None, errors
    return LocatedGrid(table_gray, corners, row_lines, col_lines), []


def _regularise_columns(cols: List[int]) -> Optional[List[int]]:
    """5 columns → 6 vertical lines; the question column may be wider."""
    n = len(OPTION_COLUMNS_LTR) + 2
    if len(cols) < n:
        return None
    best, best_cost = None, float("inf")
    for i in range(len(cols) - n + 1):
        cand = cols[i : i + n]
        widths = np.diff(cand[:-1])      # the 4 option columns must be even
        cost = float(np.std(widths)) / max(float(np.mean(widths)), 1.0)
        if cost < best_cost:
            best, best_cost = cand, cost
    if best is None or best_cost > 0.12:
        return None
    return best


# ---------------------------------------------------------------------------
# Cell measurement and row resolution
# ---------------------------------------------------------------------------
def _measure_cell(
    binary: np.ndarray, gray: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> Tuple[float, float]:
    """
    (ink, core_fill) for one answer cell.

    ink       — dark fraction of a square around the printed checkbox, where
                marks actually land (the rest of the cell is white space).
    core_fill — mean darkness strictly inside the checkbox: a solid fill is
                near 1, an X or check leaves most of the interior white.
    """
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ch = y1 - y0
    r = max(3, int(ch * PROBE_HALF_FRAC))
    probe = binary[cy - r : cy + r, cx - r : cx + r]
    ink = float(np.mean(probe > 0)) if probe.size else 0.0

    rc = max(2, int(ch * CORE_HALF_FRAC))
    core = gray[cy - rc : cy + rc, cx - rc : cx + rc]
    core_fill = float(1.0 - np.mean(core) / 255.0) if core.size else 0.0
    return ink, core_fill


def _resolve_row(row: RowReading) -> None:
    """
    One mark → the answer. Two marks → the more solidly filled box is a
    cancellation (students black out a box to retract it) and the other is the
    answer. "Solid" is judged *within the row*: some students mark every
    answer with a filled square, so no absolute darkness cut-off is safe.
    """
    marked = [c for c in row.cells if c.marked]
    if not marked:
        row.status, row.reason = RowStatus.BLANK, "no mark"
        return
    if len(marked) == 1:
        row.status, row.chosen = RowStatus.ANSWERED, marked[0].letter
        row.reason = "single mark"
        return
    if len(marked) == 2:
        darker, lighter = sorted(marked, key=lambda c: -c.core_fill)
        gap = darker.core_fill - lighter.core_fill
        if gap >= SOLID_GAP_MIN:
            darker.solid = True
            row.status, row.chosen = RowStatus.ANSWERED, lighter.letter
            row.cancelled = [darker.letter]
            row.low_margin = row.low_margin or gap < SOLID_GAP_MIN + MARGIN_REVIEW
            row.reason = (
                f"correction: {darker.letter} blacked out (interior {darker.core_fill:.2f} vs "
                f"{lighter.core_fill:.2f}); {lighter.letter} chosen"
            )
            return
    row.status = RowStatus.AMBIGUOUS
    row.reason = f"{len(marked)} marked boxes, no clear cancellation — needs review"


def read_answer_grid(image_path: str | Path) -> GridReading:
    """Read the multiple-choice answer table from a cover-sheet image."""
    image_path = Path(image_path)
    result = GridReading(image_path=str(image_path))
    image = cv2.imread(str(image_path))
    if image is None:
        result.errors.append(f"cannot read image: {image_path}")
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grid, errors = locate_grid(gray)
    if grid is None:
        result.errors.extend(errors)
        return result

    xs, ys = grid.corners[:, 0], grid.corners[:, 1]
    result.grid_bbox = (
        int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
    )
    binary = _binarize(grid.table_gray)
    rel_cols, rel_rows = grid.col_lines, grid.row_lines

    measured: List[List[Tuple[float, float]]] = []
    # Row 0 is the header (א ב ג ד / שאלה)
    for qi in range(N_QUESTIONS):
        ya, yb = rel_rows[qi + 1], rel_rows[qi + 2]
        measured.append([
            _measure_cell(binary, grid.table_gray, rel_cols[ci], rel_cols[ci + 1], ya, yb)
            for ci in range(len(OPTION_COLUMNS_LTR))
        ])

    inks = np.array([[m[0] for m in row] for row in measured])
    # Most boxes on a sheet are empty: the lower half is the printed-box baseline
    baseline = float(np.median(np.sort(inks.ravel())[: inks.size // 2]))
    result.baseline_ink = baseline

    for qi, row_m in enumerate(measured):
        cells = []
        for ci, (ink, core) in enumerate(row_m):
            excess = ink - baseline
            cells.append(CellReading(
                letter=OPTION_COLUMNS_LTR[ci], ink=ink, excess=excess, core_fill=core,
                marked=excess >= MARK_EXCESS_MIN,
            ))
        row = RowReading(question_id=str(qi + 1), cells=cells)
        row.low_margin = any(abs(c.excess - MARK_EXCESS_MIN) < MARGIN_REVIEW for c in cells)
        _resolve_row(row)
        result.rows.append(row)
    return result


def crop_grid(image_path: str | Path, reading: GridReading, pad_frac: float = 0.03) -> Optional[bytes]:
    """JPEG bytes of just the answer table — no identity header leaves the machine."""
    if not reading.grid_bbox:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    x, y, w, h = reading.grid_bbox
    px, py = int(w * pad_frac), int(h * pad_frac)
    H, W = image.shape[:2]
    crop = image[max(0, y - py) : min(H, y + h + py), max(0, x - px) : min(W, x + w + px)]
    ch, cw = crop.shape[:2]
    if max(ch, cw) > VLM_GRID_MAX_DIM:
        scale = VLM_GRID_MAX_DIM / float(max(ch, cw))
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else None


# ---------------------------------------------------------------------------
# VLM cross-check
# ---------------------------------------------------------------------------

VLM_GRID_PROMPT_VERSION = "omr_vlm_v1"
VLM_GRID_MAX_DIM: int = int(os.getenv("OMR_VLM_MAX_DIM", "1400"))
VLM_GRID_CACHE_DIR = Path(os.getenv("STORAGE_ROOT", "storage")) / "cache" / "omr_vlm"

VLM_GRID_PROMPT = """\
This image is the multiple-choice answer table from a Hebrew exam cover sheet.
The table reads right-to-left: the rightmost column holds the question number
(1-{n}); the next columns, moving LEFT, are options א, ב, ג, ד (so ד is the
leftmost column). The header row shows these letters. Each cell has a small
printed checkbox.

For every question row report each option box that contains a HANDWRITTEN mark
(ignore the printed empty checkbox outline). Classify each mark:
  "x"      - a cross / X
  "check"  - a check mark (V)
  "filled" - the box is solidly coloured/blacked in
  "circle" - a circle around the box or letter
  "other"  - any other stroke or scribble
Then give the student's final answer for that row. Convention on this exam: when
a row has a solidly filled box AND a box with an X/check, the filled box is a
retracted (cancelled) answer and the X/check box is the final answer. If a row
has exactly one marked box of any kind, that box is the answer. If you cannot
tell, use null for final_answer and explain in "note".

Return ONLY JSON:
{{"rows": [{{"question": 1, "marks": [{{"option": "א", "type": "x"}}],
            "final_answer": "א", "note": ""}}, ...]}}
Include all {n} rows, in order, even blank ones (empty "marks", null answer).
"""


@dataclass
class VlmRowReading:
    question_id: str
    marks: List[Tuple[str, str]]          # (option letter, mark type)
    final_answer: Optional[str]
    note: str = ""


@dataclass
class VlmGridReading:
    model: str
    prompt_version: str
    rows: List[VlmRowReading] = field(default_factory=list)
    raw_text: str = ""
    provider_model: str = ""
    cost_usd: float = 0.0
    cache_hit: bool = False
    errors: List[str] = field(default_factory=list)

    def answers(self) -> Dict[str, Optional[str]]:
        return {r.question_id: r.final_answer for r in self.rows}

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "cache_hit": self.cache_hit,
            "cost_usd": self.cost_usd,
            "errors": self.errors,
            "rows": [
                {
                    "question_id": r.question_id,
                    "marks": [list(m) for m in r.marks],
                    "final_answer": r.final_answer,
                    "note": r.note,
                }
                for r in self.rows
            ],
            "raw_text": self.raw_text,
        }


def _parse_vlm_grid(payload, n: int) -> Tuple[List[VlmRowReading], List[str]]:
    errors: List[str] = []
    by_q: Dict[str, VlmRowReading] = {}
    items = payload if isinstance(payload, list) else (payload or {}).get("rows") or []
    for item in items:
        if not isinstance(item, dict):
            errors.append(f"malformed row: {item!r}")
            continue
        try:
            qid = str(int(item.get("question")))
        except (TypeError, ValueError):
            errors.append(f"row without a valid question number: {item!r}")
            continue
        marks = []
        for m in item.get("marks") or []:
            letter = str(m.get("option", "")).strip()
            if letter in OPTION_COLUMNS_LTR:
                marks.append((letter, str(m.get("type", "other")).strip().lower()))
        answer = item.get("final_answer")
        answer = str(answer).strip() if answer else None
        if answer not in OPTION_COLUMNS_LTR:
            answer = None
        by_q[qid] = VlmRowReading(qid, marks, answer, str(item.get("note") or ""))
    rows = []
    for q in range(1, n + 1):
        qid = str(q)
        if qid not in by_q:
            errors.append(f"VLM omitted question {qid}")
            rows.append(VlmRowReading(qid, [], None, "omitted"))
        else:
            rows.append(by_q[qid])
    return rows, errors


def vlm_read_grid(
    grid_jpeg: bytes,
    *,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> VlmGridReading:
    """Second, independent reading of the answer table by a vision model."""
    import hashlib

    from core.vlm import VLMError, generate_with_image, get_default_model, parse_json_response

    model = model or get_default_model()
    result = VlmGridReading(model=model, prompt_version=VLM_GRID_PROMPT_VERSION)
    key = hashlib.sha256(grid_jpeg).hexdigest()[:24]
    cache_file = VLM_GRID_CACHE_DIR / f"{VLM_GRID_PROMPT_VERSION}_{model.replace('/', '_')}_{key}.json"

    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        result.raw_text = cached["raw_text"]
        result.provider_model = cached.get("provider_model", "")
        result.cache_hit = True
    else:
        try:
            resp = generate_with_image(
                VLM_GRID_PROMPT.format(n=N_QUESTIONS), grid_jpeg,
                model=model, json_mode=True, purpose="omr_crosscheck",
            )
        except VLMError as exc:
            result.errors.append(f"VLM call failed: {exc}")
            return result
        result.raw_text, result.cost_usd = resp.text, resp.cost_usd
        result.provider_model = resp.provider_model
        if use_cache:
            VLM_GRID_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"raw_text": resp.text, "provider_model": resp.provider_model},
                           ensure_ascii=False),
                encoding="utf-8",
            )

    try:
        payload = parse_json_response(result.raw_text)
    except Exception as exc:  # malformed JSON is a reading failure, not a crash
        result.errors.append(f"unparseable VLM output: {exc}")
        return result
    result.rows, errs = _parse_vlm_grid(payload, N_QUESTIONS)
    result.errors.extend(errs)
    return result


@dataclass
class ReconciledRow:
    question_id: str
    answer: Optional[str]
    needs_review: bool
    reasons: List[str]

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "needs_review": self.needs_review,
            "reasons": self.reasons,
        }


def reconcile(cv: GridReading, vlm: Optional[VlmGridReading]) -> List[ReconciledRow]:
    """
    Combine the CV and VLM readings. Agreement on a clean CV row is accepted;
    anything else (disagreement, CV ambiguity, thin margin, missing VLM row)
    keeps the CV answer as a proposal but is routed to human review.
    """
    vlm_rows = {r.question_id: r for r in (vlm.rows if vlm else [])}
    out: List[ReconciledRow] = []
    for row in cv.rows:
        reasons: List[str] = []
        if row.status == RowStatus.AMBIGUOUS:
            reasons.append(f"CV ambiguous: {row.reason}")
        if row.low_margin:
            reasons.append("CV low margin")
        v = vlm_rows.get(row.question_id)
        if v is None:
            reasons.append("no VLM reading")
        elif v.final_answer != row.chosen:
            reasons.append(f"CV={row.chosen} vs VLM={v.final_answer}")
        answer = row.chosen
        if row.status == RowStatus.AMBIGUOUS and v is not None and v.final_answer:
            answer = v.final_answer
        out.append(ReconciledRow(row.question_id, answer, bool(reasons), reasons))
    if cv.errors:
        for r in out:
            r.needs_review = True
            r.reasons.append("CV grid errors: " + "; ".join(cv.errors))
    return out
