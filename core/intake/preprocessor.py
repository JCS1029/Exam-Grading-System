"""
Step 2 — Computer Vision Image Cleaning.

Sub-operations applied to every rasterised page:
    1. Deskew  — Hough-line angle detection → rotation correction (≤ 0.5° residual)
    2. White-point clamp — crush near-white bleed-through ghosts before contrast boost
    3. CLAHE   — Adaptive contrast enhancement for faint pencil strokes
    4. Border crop — Perspective rectification to remove scanner borders

Usage:
    from core.intake.preprocessor import preprocess_page, preprocess_booklet

    result = preprocess_page("storage/raw/test1/page_001.png", "test1")
    results = preprocess_booklet("test1")
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STORAGE_ROOT: Path = Path(os.getenv("STORAGE_ROOT", "storage"))
RAW_STORAGE: Path = STORAGE_ROOT / "raw"
PREPROCESSED_STORAGE: Path = STORAGE_ROOT / "preprocessed"

# Deskew
MAX_ACCEPTABLE_SKEW_DEG: float = 0.5   # residual angle must be ≤ this
MAX_CORRECTION_DEG: float = 15.0        # refuse to correct absurd angles

# White-point clamp (bleed-through / show-through suppression)
# Ghost strokes sit in the top ~5–10% of 8-bit gray (≈228–245) while paper is
# ≈250–255 and front-side ink/pencil is ≈0–180. Crush that band to 255 *before*
# CLAHE so local histogram equalisation cannot amplify the ghosts.
#
# Tune via .env — higher CLAMP_FLOOR = more forgiving toward faint real pencil.
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


PAPER_WHITE_PERCENTILE: float = _env_float("WHITE_POINT_PAPER_PERCENTILE", 99.0)
BLEED_BAND_FRAC: float = _env_float("WHITE_POINT_BLEED_BAND_FRAC", 0.08)
INK_SAFE_MAX: int = _env_int("WHITE_POINT_INK_SAFE_MAX", 180)
CLAMP_FLOOR: int = _env_int("WHITE_POINT_CLAMP_FLOOR", 238)
MIN_PAPER_WHITE: float = _env_float("WHITE_POINT_MIN_PAPER_WHITE", 230.0)
WHITE_POINT_CLAMP_ENABLED: bool = _env_bool("WHITE_POINT_CLAMP_ENABLED", True)

# CLAHE
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)

# Border crop
MIN_CONTOUR_AREA_RATIO: float = 0.25   # contour must cover ≥ 25% of image area


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PreprocessResult:
    """Result of preprocessing a single page image."""

    input_path: str
    output_path: str
    booklet_id: str
    page_index: int
    skew_detected_deg: float = 0.0
    skew_corrected: bool = False
    residual_skew_deg: float = 0.0
    white_point_clamped: bool = False
    clahe_applied: bool = False
    border_cropped: bool = False
    fixing_score: float = 0.0
    metrics: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# 1. Deskew
# ---------------------------------------------------------------------------
def _detect_skew_angle(gray: np.ndarray) -> float:
    """
    Detect the dominant text-line angle using Hough Line Transform.

    Returns the median angle in degrees. Positive = clockwise tilt.
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines via probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 8,   # at least 1/8 of image width
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    # Compute angles of all detected lines
    angles = []
    for line in lines:
        coords = line.flatten()
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1:
            continue  # skip near-vertical lines
        angle = math.degrees(math.atan2(dy, dx))
        # Only consider near-horizontal lines (within ±45° of horizontal)
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def _rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by *angle_deg* around its centre, filling borders with white."""
    h, w = image.shape[:2]
    centre = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)

    # Compute new bounding dimensions
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w = int(h * sin_val + w * cos_val)
    new_h = int(h * cos_val + w * sin_val)

    # Adjust the rotation matrix for the new dimensions
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    border_color = (255, 255, 255) if len(image.shape) == 3 else 255
    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )
    return rotated


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Straighten a tilted page image.

    Returns:
        (corrected_image, detected_angle, residual_angle)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    detected = _detect_skew_angle(gray)
    logger.debug("Detected skew: %.2f°", detected)

    if abs(detected) <= MAX_ACCEPTABLE_SKEW_DEG:
        # Already straight enough
        return image, detected, detected

    if abs(detected) > MAX_CORRECTION_DEG:
        logger.warning(
            "Detected skew %.2f° exceeds max correction (%.1f°) — skipping",
            detected, MAX_CORRECTION_DEG,
        )
        return image, detected, detected

    # Rotate to correct
    corrected = _rotate_image(image, detected)

    # Measure residual
    gray_corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if len(corrected.shape) == 3 else corrected
    residual = _detect_skew_angle(gray_corrected)

    return corrected, detected, residual


# ---------------------------------------------------------------------------
# 2. White-point clamp (bleed-through suppression)
# ---------------------------------------------------------------------------
def estimate_paper_white(gray: np.ndarray) -> float:
    """
    Estimate the paper background intensity.

    Exam pages are dominated by paper, so a high percentile plus the histogram
    peak in the bright band is a stable white-point.
    """
    p_hi = float(np.percentile(gray, PAPER_WHITE_PERCENTILE))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    lo = int(max(MIN_PAPER_WHITE, min(p_hi - 20.0, 250.0)))
    peak = int(np.argmax(hist[lo:256]) + lo)
    return float(max(p_hi, peak))


def clamp_white_point(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Crush near-white bleed-through ghosts to pure white.

    Identifies the paper white-point, then applies a hard threshold clamp
    (with an identity LUT below the threshold) so anything lighter than the
    bleed band — typically > 215 — becomes 255. Front-side ink/pencil at or
    below ``INK_SAFE_MAX`` (180) is never modified.

    Operates on the L channel in LAB so chroma is preserved. Must run
    *before* CLAHE.

    Aggressiveness is controlled by ``WHITE_POINT_CLAMP_FLOOR`` (higher =
    only nearer-white ghosts removed) and ``WHITE_POINT_BLEED_BAND_FRAC``.
    """
    if not WHITE_POINT_CLAMP_ENABLED:
        return image, {
            "paper_white": 0.0,
            "white_point_threshold": 0.0,
            "clamped_frac": 0.0,
            "applied": False,
        }

    is_color = len(image.shape) == 3
    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        work = l_channel
    else:
        work = image

    paper_white = estimate_paper_white(work)
    stats = {
        "paper_white": round(paper_white, 1),
        "white_point_threshold": 0.0,
        "clamped_frac": 0.0,
        "applied": False,
    }

    if paper_white < MIN_PAPER_WHITE:
        return image, stats

    # Crush the near-white band below paper white, but never drop the
    # threshold into real-ink territory.
    threshold = paper_white - (255.0 * BLEED_BAND_FRAC)
    threshold = max(threshold, float(CLAMP_FLOOR), float(INK_SAFE_MAX))
    if threshold >= paper_white:
        threshold = max(float(INK_SAFE_MAX), paper_white - 8.0)

    stats["white_point_threshold"] = round(float(threshold), 1)

    lut = np.arange(256, dtype=np.uint8)
    thr_i = int(np.clip(np.floor(threshold), 0, 255))
    lut[thr_i:] = 255  # ≥ threshold → pure white

    clamped = cv2.LUT(work, lut)
    changed = int(np.count_nonzero(clamped != work))
    stats["clamped_frac"] = round(changed / float(work.size), 4)
    stats["applied"] = changed > 0

    if is_color:
        lab = cv2.merge([clamped, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), stats
    return clamped, stats


# ---------------------------------------------------------------------------
# 3. CLAHE contrast enhancement
# ---------------------------------------------------------------------------
def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Works on the L channel in LAB colour space to boost faint pencil
    strokes without blowing out the white background.
    """
    if len(image.shape) == 2:
        # Greyscale input
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        return clahe.apply(image)

    # Colour input → convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# 4. Border / margin rectification
# ---------------------------------------------------------------------------
def crop_borders(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Detect and crop scanner borders / dark table edges.

    Finds the largest rectangular contour and applies a perspective transform.
    Returns (cropped_image, was_cropped).
    Falls back gracefully if no clear border is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h, w = gray.shape[:2]
    image_area = h * w

    # Threshold to find dark borders
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Invert so the page content is white and borders are black
    inverted = cv2.bitwise_not(binary)

    # Find contours
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, False

    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < image_area * MIN_CONTOUR_AREA_RATIO:
        # No significant border detected
        return image, False

    # Approximate to a polygon
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        # We found a quadrilateral — apply perspective transform
        pts = approx.reshape(4, 2).astype(np.float32)

        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = _order_points(pts)

        # Compute destination dimensions
        width_a = np.linalg.norm(rect[2] - rect[3])
        width_b = np.linalg.norm(rect[1] - rect[0])
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(rect[1] - rect[2])
        height_b = np.linalg.norm(rect[0] - rect[3])
        max_height = int(max(height_a, height_b))

        if max_width < w * 0.5 or max_height < h * 0.5:
            # Detected rectangle is too small — probably not the page border
            return image, False

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped, True
    else:
        # Not a clear quadrilateral — use bounding rect as fallback
        x, y, bw, bh = cv2.boundingRect(largest)
        if bw > w * 0.5 and bh > h * 0.5:
            cropped = image[y:y + bh, x:x + bw]
            return cropped, True

    return image, False


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]     # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference
    return rect


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def preprocess_page(
    image_path: str | Path,
    booklet_id: str,
    page_index: int = 0,
) -> PreprocessResult:
    """
    Apply the full preprocessing pipeline to a single page image.

    1. Deskew (straighten)
    2. White-point clamp (bleed-through crush)
    3. CLAHE (contrast boost)
    4. Border crop (perspective rectification)

    Saves the result to ``storage/preprocessed/{booklet_id}/page_NNN.png``.
    """
    image_path = Path(image_path).resolve()
    out_dir = PREPROCESSED_STORAGE / booklet_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"page_{page_index + 1:03d}.png"
    out_path = out_dir / out_filename

    result = PreprocessResult(
        input_path=str(image_path),
        output_path=str(out_path),
        booklet_id=booklet_id,
        page_index=page_index,
    )

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        result.errors.append(f"Failed to load image: {image_path}")
        return result

    orig_h, orig_w = image.shape[:2]
    orig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    orig_gray_std = float(np.std(orig_gray))

    # 1. Deskew
    try:
        image, detected, residual = deskew(image)
        result.skew_detected_deg = round(detected, 3)
        result.residual_skew_deg = round(residual, 3)
        result.skew_corrected = abs(detected) > MAX_ACCEPTABLE_SKEW_DEG
        if result.skew_corrected:
            logger.info(
                "Deskewed %s: %.2f° → %.2f° residual",
                image_path.name, detected, residual,
            )
    except Exception as exc:
        result.errors.append(f"Deskew failed: {exc}")
        logger.error("Deskew error on %s: %s", image_path.name, exc)

    # 2. White-point clamp (before CLAHE so ghosts are never boosted)
    wp_stats: dict = {
        "paper_white": 0.0,
        "white_point_threshold": 0.0,
        "clamped_frac": 0.0,
        "applied": False,
    }
    try:
        image, wp_stats = clamp_white_point(image)
        result.white_point_clamped = bool(wp_stats.get("applied"))
        if result.white_point_clamped:
            logger.info(
                "White-point clamp %s: paper=%.1f thr=%.1f crushed=%.2f%%",
                image_path.name,
                wp_stats["paper_white"],
                wp_stats["white_point_threshold"],
                100.0 * wp_stats["clamped_frac"],
            )
    except Exception as exc:
        result.errors.append(f"White-point clamp failed: {exc}")
        logger.error("White-point clamp error on %s: %s", image_path.name, exc)

    # 3. CLAHE
    image_after_clahe_std = orig_gray_std
    try:
        image = apply_clahe(image)
        result.clahe_applied = True
        gray_after_clahe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        image_after_clahe_std = float(np.std(gray_after_clahe))
    except Exception as exc:
        result.errors.append(f"CLAHE failed: {exc}")
        logger.error("CLAHE error on %s: %s", image_path.name, exc)

    # 4. Border crop
    try:
        image, was_cropped = crop_borders(image)
        result.border_cropped = was_cropped
        if was_cropped:
            logger.info("Cropped borders from %s", image_path.name)
    except Exception as exc:
        result.errors.append(f"Border crop failed: {exc}")
        logger.error("Border crop error on %s: %s", image_path.name, exc)

    # Compute Comprehensive Fixing Score (0 - 100)
    # Quantifies how much degradation was present and how much restoration was applied:
    # 1. Deskew severity: tilt angle needing correction
    # 2. Border crop severity: margin/skewed scanner edge cut
    # 3. Contrast & Dynamic Range Deficit: low baseline luminance standard deviation
    # 4. Stroke Faintness / Washout: proportion of strokes with faint intensity ([140, 225] vs confident dark [<140])
    # 5. Non-uniform illumination: luminance spread across quadrants (scanner shadow / uneven light)
    try:
        # 1. Deskew Score (0 - 100)
        deskew_score = min(100.0, (abs(result.skew_detected_deg) / 5.0) * 100.0)

        # 2. Border Crop Score (0 - 100)
        orig_area = orig_h * orig_w
        final_h, final_w = image.shape[:2]
        final_area = final_h * final_w
        area_diff_ratio = abs(orig_area - final_area) / float(orig_area) if orig_area > 0 else 0.0
        crop_score = min(100.0, (area_diff_ratio / 0.15) * 100.0) if result.border_cropped else 0.0

        # 3. Contrast Deficit & CLAHE boost (0 - 100)
        contrast_deficit = max(0.0, (40.0 - orig_gray_std) / 20.0) * 100.0 if orig_gray_std < 40.0 else 0.0
        clahe_gain = max(0.0, image_after_clahe_std - orig_gray_std)
        clahe_boost_score = min(100.0, (clahe_gain / 3.0) * 100.0)
        contrast_score = min(100.0, 0.6 * contrast_deficit + 0.4 * clahe_boost_score)

        # 4. Stroke Faintness / Washout Score (0 - 100)
        dark_strokes = int(np.sum(orig_gray < 140))
        faint_strokes = int(np.sum((orig_gray >= 140) & (orig_gray < 225)))
        total_content = dark_strokes + faint_strokes
        faint_ratio = (faint_strokes / float(total_content)) if total_content > 0 else 0.0
        faintness_score = min(100.0, max(0.0, (faint_ratio - 0.20) / 0.25 * 100.0))

        # 5. Lighting Non-uniformity Score (0 - 100)
        half_h, half_w = orig_h // 2, orig_w // 2
        q1 = float(np.mean(orig_gray[:half_h, :half_w]))
        q2 = float(np.mean(orig_gray[:half_h, half_w:]))
        q3 = float(np.mean(orig_gray[half_h:, :half_w]))
        q4 = float(np.mean(orig_gray[half_h:, half_w:]))
        lighting_spread = max(q1, q2, q3, q4) - min(q1, q2, q3, q4)
        lighting_score = min(100.0, (lighting_spread / 12.0) * 100.0)

        # Composite Fixing Score
        # If severe deskew or border crop happens, they contribute prominently.
        # For scans with faint pencil and shadow gradients, contrast & faintness contribute heavily.
        composite_score = round(
            0.30 * deskew_score
            + 0.20 * crop_score
            + 0.20 * contrast_score
            + 0.15 * faintness_score
            + 0.15 * lighting_score,
            2,
        )
        result.fixing_score = composite_score
        result.metrics = {
            "deskew_score": round(deskew_score, 1),
            "crop_score": round(crop_score, 1),
            "contrast_score": round(contrast_score, 1),
            "faintness_score": round(faintness_score, 1),
            "lighting_score": round(lighting_score, 1),
            "skew_detected_deg": result.skew_detected_deg,
            "orig_contrast_std": round(orig_gray_std, 2),
            "final_contrast_std": round(image_after_clahe_std, 2),
            "faint_ratio": round(faint_ratio, 3),
            "lighting_spread": round(lighting_spread, 2),
            "paper_white": wp_stats.get("paper_white", 0.0),
            "white_point_threshold": wp_stats.get("white_point_threshold", 0.0),
            "clamped_frac": wp_stats.get("clamped_frac", 0.0),
            "white_point_clamped": result.white_point_clamped,
        }
    except Exception as exc:
        logger.warning("Failed to compute fixing score for %s: %s", image_path.name, exc)
        result.fixing_score = 0.0

    # Save result + metrics sidecar (used by the visual benchmark report)
    cv2.imwrite(str(out_path), image)
    metrics_path = out_path.with_suffix(".metrics.json")
    sidecar = {
        "booklet_id": booklet_id,
        "page_index": page_index,
        "fixing_score": result.fixing_score,
        "white_point_clamped": result.white_point_clamped,
        "clahe_applied": result.clahe_applied,
        "border_cropped": result.border_cropped,
        "skew_detected_deg": result.skew_detected_deg,
        "metrics": result.metrics,
    }
    metrics_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    logger.info("Preprocessed → %s (Fixing Score: %.1f)", out_path, result.fixing_score)

    return result


def preprocess_booklet(booklet_id: str) -> List[PreprocessResult]:
    """
    Preprocess all pages of a booklet from ``storage/raw/{booklet_id}/``.

    Returns a list of :class:`PreprocessResult` in page order.
    """
    raw_dir = RAW_STORAGE / booklet_id
    if not raw_dir.is_dir():
        logger.error("Raw booklet directory not found: %s", raw_dir)
        return []

    pages = sorted(raw_dir.glob("page_*.png"))
    if not pages:
        logger.warning("No page images found in %s", raw_dir)
        return []

    results: List[PreprocessResult] = []
    for idx, page_path in enumerate(pages):
        result = preprocess_page(page_path, booklet_id, page_index=idx)
        results.append(result)

    ok_count = sum(1 for r in results if r.ok)
    logger.info(
        "Preprocessed booklet %s: %d/%d pages OK",
        booklet_id, ok_count, len(results),
    )
    return results
