"""
Step 1 — Ingestion & High-Res Rasterisation.

Converts PDF pages into crisp 300 DPI PNG images, or validates / converts
raw image uploads. Each booklet gets its own sub-directory under storage/raw/.

Usage:
    from core.intake.rasterizer import rasterize_file, rasterize_directory

    manifest = rasterize_file("docs/test1.pdf")
    all_manifests = rasterize_directory("docs/")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DPI: int = int(os.getenv("RASTERIZE_DPI", "300"))
RAW_STORAGE: Path = Path(os.getenv("STORAGE_ROOT", "storage")) / "raw"

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
SUPPORTED_PDF_EXTENSION = ".pdf"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PageInfo:
    """Metadata for a single rasterised page."""

    index: int          # 0-based page index within the booklet
    path: str           # absolute path to the saved PNG
    width: int          # pixel width
    height: int         # pixel height


@dataclass
class BookletManifest:
    """Result of rasterising one source file (PDF or image)."""

    booklet_id: str
    source_file: str
    pages: List[PageInfo] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0 and self.page_count > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _booklet_id_from_filename(filepath: str | Path) -> str:
    """Derive a booklet ID from the filename (without extension)."""
    return Path(filepath).stem


def _ensure_output_dir(booklet_id: str) -> Path:
    """Create and return the output directory for a booklet."""
    out_dir = RAW_STORAGE / booklet_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _page_filename(index: int) -> str:
    """Consistent zero-padded page filename."""
    return f"page_{index + 1:03d}.png"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def rasterize_pdf(
    pdf_path: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    booklet_id: Optional[str] = None,
) -> BookletManifest:
    """
    Render every page of *pdf_path* to a 300 DPI PNG.

    Pages are saved to  ``storage/raw/{booklet_id}/page_001.png`` etc.
    Returns a :class:`BookletManifest` with per-page metadata.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    bid = booklet_id or _booklet_id_from_filename(pdf_path)
    out_dir = _ensure_output_dir(bid)
    manifest = BookletManifest(booklet_id=bid, source_file=str(pdf_path))

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        manifest.errors.append(f"Failed to open PDF: {exc}")
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        return manifest

    for page_idx in range(len(doc)):
        try:
            page = doc[page_idx]
            # Render at target DPI
            pix = page.get_pixmap(dpi=dpi)
            out_path = out_dir / _page_filename(page_idx)
            pix.save(str(out_path))

            manifest.pages.append(
                PageInfo(
                    index=page_idx,
                    path=str(out_path),
                    width=pix.width,
                    height=pix.height,
                )
            )
            logger.info(
                "Rasterised %s page %d → %s (%dx%d)",
                pdf_path.name, page_idx + 1, out_path.name, pix.width, pix.height,
            )
        except Exception as exc:
            error_msg = f"Page {page_idx + 1} render failed: {exc}"
            manifest.errors.append(error_msg)
            logger.error("Rasterise error in %s: %s", pdf_path.name, error_msg)

    doc.close()
    return manifest


def rasterize_image(
    image_path: str | Path,
    *,
    booklet_id: Optional[str] = None,
) -> BookletManifest:
    """
    Validate and convert a single image file to PNG under storage/raw/.

    Treats the image as a single-page booklet.
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bid = booklet_id or _booklet_id_from_filename(image_path)
    out_dir = _ensure_output_dir(bid)
    manifest = BookletManifest(booklet_id=bid, source_file=str(image_path))

    try:
        img = Image.open(str(image_path))
        out_path = out_dir / _page_filename(0)
        # Convert to RGB (handles RGBA, palette, greyscale)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(str(out_path), format="PNG")

        manifest.pages.append(
            PageInfo(
                index=0,
                path=str(out_path),
                width=img.width,
                height=img.height,
            )
        )
        logger.info("Saved image %s → %s (%dx%d)", image_path.name, out_path.name, img.width, img.height)
    except Exception as exc:
        error_msg = f"Image processing failed: {exc}"
        manifest.errors.append(error_msg)
        logger.error("Image error for %s: %s", image_path.name, error_msg)

    return manifest


def rasterize_file(
    filepath: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    booklet_id: Optional[str] = None,
) -> BookletManifest:
    """
    Auto-detect file type and rasterise accordingly.

    Accepts PDF files and common image formats.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext == SUPPORTED_PDF_EXTENSION:
        return rasterize_pdf(filepath, dpi=dpi, booklet_id=booklet_id)
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        return rasterize_image(filepath, booklet_id=booklet_id)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {SUPPORTED_PDF_EXTENSION}, {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )


def rasterize_directory(
    directory: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> List[BookletManifest]:
    """
    Rasterise every PDF and image in *directory* (non-recursive).

    Returns a list of :class:`BookletManifest` objects.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    all_extensions = {SUPPORTED_PDF_EXTENSION} | SUPPORTED_IMAGE_EXTENSIONS
    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in all_extensions
    )

    if not files:
        logger.warning("No supported files found in %s", directory)
        return []

    manifests: List[BookletManifest] = []
    for f in files:
        logger.info("Processing %s ...", f.name)
        manifest = rasterize_file(f, dpi=dpi)
        manifests.append(manifest)

    logger.info(
        "Rasterised %d files → %d total pages",
        len(manifests),
        sum(m.page_count for m in manifests),
    )
    return manifests
