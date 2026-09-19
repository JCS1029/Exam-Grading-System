"""
Step 4 — Local Anonymisation & Pseudonym Mapping.

Masks the student identity header on page 1 (cover sheet) with a solid
black rectangle. Generates a cryptographically random pseudonym for each
booklet and stores the real-name ↔ pseudonym mapping in a local SQLite
database that **never leaves the machine**.

Usage:
    from core.intake.anonymizer import anonymize_booklet, PseudonymDB

    db = PseudonymDB()
    result = anonymize_booklet("test1", db)
    print(result.pseudonym)
    db.close()
"""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STORAGE_ROOT: Path = Path(os.getenv("STORAGE_ROOT", "storage"))
PREPROCESSED_STORAGE: Path = STORAGE_ROOT / "preprocessed"
ANONYMIZED_STORAGE: Path = STORAGE_ROOT / "anonymized"
PSEUDONYM_DB_PATH: Path = STORAGE_ROOT / "pseudonym_map.db"

# Identity region: top portion of page 1 where name, ID, seat number live
# Expressed as fraction of page height from the top
IDENTITY_REGION_TOP_FRACTION: float = float(
    os.getenv("IDENTITY_REGION_TOP_FRACTION", "0.15")
)

# Pseudonym format
PSEUDONYM_PREFIX: str = "STUDENT_"
PSEUDONYM_HEX_CHARS: int = 4  # e.g. STUDENT_A7F9


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AnonymizeResult:
    """Result of anonymising a single booklet."""

    booklet_id: str
    pseudonym: str
    pages_processed: int
    identity_masked: bool = False
    output_dir: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0 and self.identity_masked


# ---------------------------------------------------------------------------
# Pseudonym Database
# ---------------------------------------------------------------------------
class PseudonymDB:
    """
    SQLite-backed mapping between pseudonyms and real student identities.

    This database stays local — it is the only place the real name ↔ pseudonym
    link exists, and it must never be transmitted or uploaded.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or PSEUDONYM_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()
        self._used_pseudonyms: set = set()
        self._load_existing_pseudonyms()

    def _init_schema(self) -> None:
        """Create the mapping table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pseudonym_map (
                pseudonym     TEXT PRIMARY KEY,
                original_name TEXT,
                original_id   TEXT,
                booklet_id    TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_booklet_id
            ON pseudonym_map(booklet_id)
        """)
        self.conn.commit()

    def _load_existing_pseudonyms(self) -> None:
        """Load already-assigned pseudonyms to avoid collisions."""
        cursor = self.conn.execute("SELECT pseudonym FROM pseudonym_map")
        for row in cursor:
            self._used_pseudonyms.add(row[0])

    def generate_pseudonym(self) -> str:
        """Generate a unique cryptographically random pseudonym."""
        for _ in range(1000):  # safety limit
            hex_part = secrets.token_hex(PSEUDONYM_HEX_CHARS // 2 + 1)[
                :PSEUDONYM_HEX_CHARS
            ].upper()
            pseudonym = f"{PSEUDONYM_PREFIX}{hex_part}"
            if pseudonym not in self._used_pseudonyms:
                self._used_pseudonyms.add(pseudonym)
                return pseudonym
        raise RuntimeError("Failed to generate a unique pseudonym after 1000 attempts")

    def store_mapping(
        self,
        pseudonym: str,
        booklet_id: str,
        original_name: str = "",
        original_id: str = "",
    ) -> None:
        """Insert a pseudonym → identity mapping."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO pseudonym_map
                (pseudonym, original_name, original_id, booklet_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                pseudonym,
                original_name,
                original_id,
                booklet_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()
        logger.info("Stored mapping: %s ↔ booklet %s", pseudonym, booklet_id)

    def lookup_by_pseudonym(self, pseudonym: str) -> Optional[Dict]:
        """Look up the real identity behind a pseudonym."""
        cursor = self.conn.execute(
            "SELECT pseudonym, original_name, original_id, booklet_id, created_at "
            "FROM pseudonym_map WHERE pseudonym = ?",
            (pseudonym,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "pseudonym": row[0],
                "original_name": row[1],
                "original_id": row[2],
                "booklet_id": row[3],
                "created_at": row[4],
            }
        return None

    def lookup_by_booklet(self, booklet_id: str) -> Optional[Dict]:
        """Look up the pseudonym for a given booklet ID."""
        cursor = self.conn.execute(
            "SELECT pseudonym, original_name, original_id, booklet_id, created_at "
            "FROM pseudonym_map WHERE booklet_id = ?",
            (booklet_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "pseudonym": row[0],
                "original_name": row[1],
                "original_id": row[2],
                "booklet_id": row[3],
                "created_at": row[4],
            }
        return None

    def get_all_mappings(self) -> List[Dict]:
        """Return all pseudonym mappings."""
        cursor = self.conn.execute(
            "SELECT pseudonym, original_name, original_id, booklet_id, created_at "
            "FROM pseudonym_map ORDER BY created_at"
        )
        return [
            {
                "pseudonym": row[0],
                "original_name": row[1],
                "original_id": row[2],
                "booklet_id": row[3],
                "created_at": row[4],
            }
            for row in cursor
        ]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Identity masking
# ---------------------------------------------------------------------------
def _compute_identity_region(
    image: np.ndarray,
    top_fraction: float = IDENTITY_REGION_TOP_FRACTION,
) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box (x, y, w, h) of the identity header region.

    Default: full width, top 15% of the image.
    """
    h, w = image.shape[:2]
    region_height = int(h * top_fraction)
    return (0, 0, w, region_height)


def mask_identity_region(
    image: np.ndarray,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Draw a solid black rectangle over the identity region.

    Parameters
    ----------
    image : np.ndarray
        The page image (BGR or greyscale).
    region : tuple(x, y, w, h), optional
        The bounding box to mask. If None, uses the default top-15% region.

    Returns
    -------
    np.ndarray
        The image with the identity region blacked out.
    """
    masked = image.copy()
    if region is None:
        region = _compute_identity_region(image)

    x, y, w, h = region
    # Solid black fill
    masked[y : y + h, x : x + w] = 0

    return masked


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def anonymize_booklet(
    booklet_id: str,
    pseudonym_db: PseudonymDB,
    *,
    identity_region: Optional[Tuple[int, int, int, int]] = None,
    source_dir: Optional[Path] = None,
) -> AnonymizeResult:
    """
    Anonymise all pages of a preprocessed booklet.

    1. Mask the identity header on page 1 (cover sheet).
    2. Copy all pages to ``storage/anonymized/{booklet_id}/``.
    3. Generate a pseudonym and store the mapping in the database.

    Parameters
    ----------
    booklet_id : str
        The booklet identifier.
    pseudonym_db : PseudonymDB
        The pseudonym database instance.
    identity_region : tuple, optional
        Custom bounding box (x, y, w, h) for the identity region.
        If None, uses the default (top 15% of page).
    source_dir : Path, optional
        Where to read preprocessed pages from. Defaults to
        ``storage/preprocessed/{booklet_id}/``.
    """
    src_dir = source_dir or (PREPROCESSED_STORAGE / booklet_id)
    out_dir = ANONYMIZED_STORAGE / booklet_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = AnonymizeResult(
        booklet_id=booklet_id,
        pseudonym="",
        pages_processed=0,
        output_dir=str(out_dir),
    )

    if not src_dir.is_dir():
        result.errors.append(f"Source directory not found: {src_dir}")
        logger.error("Anonymize: source not found for booklet %s", booklet_id)
        return result

    pages = sorted(src_dir.glob("page_*.png"))
    if not pages:
        result.errors.append(f"No page images found in {src_dir}")
        logger.warning("Anonymize: no pages for booklet %s", booklet_id)
        return result

    # Generate pseudonym
    pseudonym = pseudonym_db.generate_pseudonym()
    result.pseudonym = pseudonym

    for idx, page_path in enumerate(pages):
        try:
            image = cv2.imread(str(page_path))
            if image is None:
                result.errors.append(f"Failed to load {page_path.name}")
                continue

            if idx == 0:
                # Page 1 (cover sheet) — mask identity region
                region = identity_region or _compute_identity_region(image)
                image = mask_identity_region(image, region)
                result.identity_masked = True
                logger.info(
                    "Masked identity region on %s page 1 (region: %s)",
                    booklet_id, region,
                )

            # Save to anonymized directory
            out_path = out_dir / page_path.name
            cv2.imwrite(str(out_path), image)
            result.pages_processed += 1

        except Exception as exc:
            result.errors.append(f"Page {idx + 1} error: {exc}")
            logger.error(
                "Anonymize error on %s page %d: %s",
                booklet_id, idx + 1, exc,
            )

    # Store the mapping (name/ID are empty in v1 — no OCR on the header yet)
    pseudonym_db.store_mapping(
        pseudonym=pseudonym,
        booklet_id=booklet_id,
        original_name="",  # populated when header OCR is added
        original_id="",
    )

    logger.info(
        "Anonymised booklet %s → %s (%d pages)",
        booklet_id, pseudonym, result.pages_processed,
    )
    return result


def anonymize_all_booklets(
    booklet_ids: List[str],
    pseudonym_db: PseudonymDB,
) -> List[AnonymizeResult]:
    """Anonymise multiple booklets in batch."""
    results = []
    for bid in booklet_ids:
        r = anonymize_booklet(bid, pseudonym_db)
        results.append(r)
    return results
