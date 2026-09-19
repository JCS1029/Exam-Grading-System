"""
Step 2 — Computer Vision Image Cleaning.

Three sub-operations applied to every rasterised page:
    1. Deskew  — Hough-line angle detection → rotation correction (≤ 0.5° residual)
    2. CLAHE   — Adaptive contrast enhancement for faint pencil strokes
    3. Border crop — Perspective rectification to remove scanner borders

Usage:
    from core.intake.preprocessor import preprocess_page, preprocess_booklet

    result = preprocess_page("storage/raw/test1/page_001.png", "test1")
    results = preprocess_booklet("test1")
"""

from __future__ import annotations

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
    clahe_applied: bool = False
    border_cropped: bool = False
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
# 2. CLAHE contrast enhancement
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
# 3. Border / margin rectification
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
    2. CLAHE (contrast boost)
    3. Border crop (perspective rectification)

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

    # 2. CLAHE
    try:
        image = apply_clahe(image)
        result.clahe_applied = True
    except Exception as exc:
        result.errors.append(f"CLAHE failed: {exc}")
        logger.error("CLAHE error on %s: %s", image_path.name, exc)

    # 3. Border crop
    try:
        image, was_cropped = crop_borders(image)
        result.border_cropped = was_cropped
        if was_cropped:
            logger.info("Cropped borders from %s", image_path.name)
    except Exception as exc:
        result.errors.append(f"Border crop failed: {exc}")
        logger.error("Border crop error on %s: %s", image_path.name, exc)

    # Save result
    cv2.imwrite(str(out_path), image)
    logger.info("Preprocessed → %s", out_path)

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
