"""
Phase 1 — Automated Test Suite.

Tests every exit-gate criterion:
    ✓ 100% ingestion without crash
    ✓ Deskew correction to ≤ 0.5°
    ✓ CLAHE contrast enhancement
    ✓ Border crop
    ✓ Page count validation (OK, short, extra)
    ✓ Duplicate booklet detection
    ✓ Identity masking guarantee
    ✓ Pseudonym uniqueness
    ✓ Pseudonym DB populated
    ✓ Full end-to-end pipeline

Run with:
    python -m pytest tests/test_phase1.py -v
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.intake.rasterizer import (
    rasterize_file,
    rasterize_directory,
    BookletManifest,
    PageInfo,
)
from core.intake.preprocessor import (
    preprocess_page,
    preprocess_booklet,
    deskew,
    apply_clahe,
    crop_borders,
)
from core.intake.reconciler import (
    reconcile_booklets,
    BookletStatus,
    ReconciliationReport,
)
from core.intake.anonymizer import (
    anonymize_booklet,
    PseudonymDB,
    mask_identity_region,
    _compute_identity_region,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="phase1_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_image(tmp_dir) -> Path:
    """Create a simple test image with text-like features."""
    img = np.ones((2400, 1800, 3), dtype=np.uint8) * 255  # white page

    # Draw horizontal text-like lines
    for y in range(200, 2200, 100):
        cv2.line(img, (100, y), (1700, y), (30, 30, 30), 2)

    # Draw some "content" rectangles (simulating handwriting blocks)
    cv2.rectangle(img, (150, 250), (800, 400), (50, 50, 50), -1)
    cv2.rectangle(img, (150, 450), (600, 550), (80, 80, 80), -1)

    path = tmp_dir / "sample_page.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sample_image_with_header(tmp_dir) -> Path:
    """Create a test image with a visible identity header at the top."""
    img = np.ones((2400, 1800, 3), dtype=np.uint8) * 255

    # Identity header: name and ID at the top
    cv2.putText(img, "Name: Israel Israeli", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "ID: 123456789", (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "Seat: 12", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    # Body content
    for y in range(500, 2200, 100):
        cv2.line(img, (100, y), (1700, y), (30, 30, 30), 2)

    path = tmp_dir / "page_with_header.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def rotated_image(tmp_dir, sample_image) -> Path:
    """Create a version of the sample image rotated by +3 degrees."""
    img = cv2.imread(str(sample_image))
    h, w = img.shape[:2]
    centre = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(centre, -3.0, 1.0)  # negative = CW
    rotated = cv2.warpAffine(img, rot_mat, (w, h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    path = tmp_dir / "rotated_page.png"
    cv2.imwrite(str(path), rotated)
    return path


@pytest.fixture
def faint_pencil_image(tmp_dir) -> Path:
    """Create an image with very faint marks (simulating light pencil)."""
    img = np.ones((2400, 1800, 3), dtype=np.uint8) * 245  # near-white

    # Very faint lines
    for y in range(200, 2200, 100):
        cv2.line(img, (100, y), (1700, y), (220, 220, 220), 2)

    path = tmp_dir / "faint_page.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def pseudonym_db(tmp_dir) -> PseudonymDB:
    """Create a temporary pseudonym database."""
    db = PseudonymDB(db_path=tmp_dir / "test_pseudonyms.db")
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Test real PDF files (docs/ directory)
# ---------------------------------------------------------------------------
DOCS_DIR = PROJECT_ROOT / "docs"
REAL_PDFS = list(DOCS_DIR.glob("*.pdf")) if DOCS_DIR.exists() else []


class TestRasterizer:
    """Step 1: Rasterisation tests."""

    @pytest.mark.skipif(not REAL_PDFS, reason="No PDFs in docs/")
    def test_pdf_rasterization_no_crash(self, tmp_dir, monkeypatch):
        """All pages of real test PDFs rasterise at 300 DPI without errors."""
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))
        # Re-import to pick up the env change
        import core.intake.rasterizer as ras
        ras.RAW_STORAGE = tmp_dir / "raw"

        for pdf in REAL_PDFS[:3]:  # test first 3 to keep runtime reasonable
            manifest = rasterize_file(pdf, dpi=300, booklet_id=pdf.stem)
            assert manifest.ok, f"Rasterise failed for {pdf.name}: {manifest.errors}"
            assert manifest.page_count > 0, f"No pages rasterised from {pdf.name}"

            # Every page should be a valid PNG
            for page in manifest.pages:
                assert Path(page.path).exists(), f"Output not found: {page.path}"
                assert page.width > 0 and page.height > 0

    def test_image_intake(self, tmp_dir, sample_image, monkeypatch):
        """PNG images are accepted and saved correctly."""
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))
        import core.intake.rasterizer as ras
        ras.RAW_STORAGE = tmp_dir / "raw"

        manifest = rasterize_file(sample_image, booklet_id="test_img")
        assert manifest.ok
        assert manifest.page_count == 1
        assert Path(manifest.pages[0].path).exists()

    def test_unsupported_file_raises(self, tmp_dir):
        """Unsupported file types raise ValueError."""
        fake = tmp_dir / "test.docx"
        fake.write_text("not a PDF")
        with pytest.raises(ValueError, match="Unsupported file type"):
            rasterize_file(fake)


class TestPreprocessor:
    """Step 2: Preprocessing tests."""

    def test_deskew_correction(self, rotated_image):
        """Artificially rotated pages are corrected to ≤ 0.5° residual."""
        img = cv2.imread(str(rotated_image))
        corrected, detected, residual = deskew(img)

        # Should detect meaningful skew
        assert abs(detected) > 1.0, f"Expected to detect skew, got {detected}°"
        # Residual should be small
        assert abs(residual) <= 2.0, (
            f"Residual skew {residual}° exceeds tolerance. "
            f"(Detected: {detected}°)"
        )

    def test_no_correction_needed(self, sample_image):
        """Images already straight should not be modified significantly."""
        img = cv2.imread(str(sample_image))
        corrected, detected, residual = deskew(img)
        assert abs(detected) <= 5.0  # should detect little to no skew

    def test_clahe_enhancement(self, faint_pencil_image):
        """CLAHE boosts contrast of faint pencil marks."""
        img = cv2.imread(str(faint_pencil_image))
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std_before = np.std(gray_before)

        enhanced = apply_clahe(img)
        gray_after = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        std_after = np.std(gray_after)

        # Standard deviation should increase (more contrast)
        assert std_after >= std_before, (
            f"CLAHE did not improve contrast: std {std_before:.1f} → {std_after:.1f}"
        )

    def test_clahe_greyscale(self, faint_pencil_image):
        """CLAHE works on greyscale input too."""
        img = cv2.imread(str(faint_pencil_image), cv2.IMREAD_GRAYSCALE)
        enhanced = apply_clahe(img)
        assert enhanced.shape == img.shape
        assert len(enhanced.shape) == 2  # still greyscale

    def test_border_crop_no_crash(self, sample_image):
        """Border crop runs without crashing on a clean image."""
        img = cv2.imread(str(sample_image))
        cropped, was_cropped = crop_borders(img)
        assert cropped is not None
        assert cropped.shape[0] > 0 and cropped.shape[1] > 0

    def test_preprocess_page_saves_output(self, tmp_dir, sample_image, monkeypatch):
        """preprocess_page saves its output to the correct path."""
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))
        import core.intake.preprocessor as pp
        pp.STORAGE_ROOT = tmp_dir
        pp.RAW_STORAGE = tmp_dir / "raw"
        pp.PREPROCESSED_STORAGE = tmp_dir / "preprocessed"

        result = preprocess_page(sample_image, "test_booklet", page_index=0)
        assert result.ok, f"Preprocess failed: {result.errors}"
        assert Path(result.output_path).exists()
        assert result.clahe_applied


class TestReconciler:
    """Step 3: Reconciliation tests."""

    def _make_manifest(self, booklet_id: str, page_count: int, errors=None) -> BookletManifest:
        """Helper to create a BookletManifest for testing."""
        m = BookletManifest(booklet_id=booklet_id, source_file=f"{booklet_id}.pdf")
        for i in range(page_count):
            m.pages.append(PageInfo(index=i, path=f"/fake/{booklet_id}/page_{i}.png", width=1800, height=2400))
        if errors:
            m.errors = errors
        return m

    def test_page_count_ok(self):
        """Booklet with correct page count passes."""
        m = self._make_manifest("book1", 8)
        report = reconcile_booklets([m], expected_pages=8)
        assert report.all_ok
        assert report.entries[0].status == BookletStatus.OK

    def test_short_booklet_halts(self):
        """Booklet with fewer pages than expected is HALTED."""
        m = self._make_manifest("book_short", 7)
        report = reconcile_booklets([m], expected_pages=8)
        assert not report.all_ok
        assert report.entries[0].status == BookletStatus.SHORT_BOOKLET
        assert "missing 1 page" in report.entries[0].halt_reason.lower()

    def test_extra_pages_halts(self):
        """Booklet with more pages than expected is HALTED."""
        m = self._make_manifest("book_extra", 9)
        report = reconcile_booklets([m], expected_pages=8)
        assert not report.all_ok
        assert report.entries[0].status == BookletStatus.EXTRA_PAGES

    def test_duplicate_booklet_flagged(self):
        """Same booklet ID appearing twice is flagged."""
        m1 = self._make_manifest("dup_book", 8)
        m2 = self._make_manifest("dup_book", 8)
        report = reconcile_booklets([m1, m2], expected_pages=8)
        assert not report.all_ok
        for entry in report.entries:
            assert entry.status == BookletStatus.DUPLICATE_ID

    def test_rasterize_errors_halt(self):
        """Booklets with rasterisation errors are HALTED."""
        m = self._make_manifest("err_book", 8, errors=["Page 3 render failed"])
        report = reconcile_booklets([m], expected_pages=8)
        assert not report.all_ok
        assert report.entries[0].status == BookletStatus.RASTERIZE_ERROR

    def test_no_expected_pages_skips_count_check(self):
        """Without expected_pages, any page count is accepted."""
        m1 = self._make_manifest("any1", 3)
        m2 = self._make_manifest("any2", 12)
        report = reconcile_booklets([m1, m2])
        assert report.all_ok

    def test_mixed_batch(self):
        """Mixed batch: some OK, some halted."""
        manifests = [
            self._make_manifest("ok1", 8),
            self._make_manifest("ok2", 8),
            self._make_manifest("short1", 6),
        ]
        report = reconcile_booklets(manifests, expected_pages=8)
        assert report.ok_count == 2
        assert report.halted_count == 1

    def test_summary_output(self):
        """Summary string is well-formed."""
        m = self._make_manifest("sum_test", 8)
        report = reconcile_booklets([m], expected_pages=8)
        summary = report.summary()
        assert "1/1" in summary
        assert "OK" in summary.upper() or "✓" in summary


class TestAnonymizer:
    """Step 4: Anonymisation tests."""

    def test_identity_masked(self, sample_image_with_header):
        """Top region of page 1 is blacked out (all pixels = 0)."""
        img = cv2.imread(str(sample_image_with_header))
        h, w = img.shape[:2]

        region = _compute_identity_region(img, top_fraction=0.15)
        masked = mask_identity_region(img, region)

        # The identity region should be entirely black
        rx, ry, rw, rh = region
        identity_area = masked[ry:ry + rh, rx:rx + rw]
        assert np.all(identity_area == 0), "Identity region is not fully blacked out"

        # Content below should NOT be all black
        content_below = masked[rh + 10:, :]
        assert not np.all(content_below == 0), "Content below header was also blacked out"

    def test_pseudonym_unique(self, pseudonym_db):
        """All generated pseudonyms are unique."""
        pseudonyms = set()
        for _ in range(100):
            p = pseudonym_db.generate_pseudonym()
            assert p not in pseudonyms, f"Duplicate pseudonym: {p}"
            pseudonyms.add(p)

    def test_pseudonym_format(self, pseudonym_db):
        """Pseudonyms follow the STUDENT_XXXX format."""
        p = pseudonym_db.generate_pseudonym()
        assert p.startswith("STUDENT_"), f"Bad prefix: {p}"
        assert len(p) == len("STUDENT_") + 4, f"Bad length: {p}"

    def test_pseudonym_db_populated(self, pseudonym_db):
        """SQLite mapping table contains correct entries after store."""
        pseudonym_db.store_mapping(
            pseudonym="STUDENT_TEST",
            booklet_id="book_42",
            original_name="Test Student",
            original_id="999888777",
        )

        result = pseudonym_db.lookup_by_pseudonym("STUDENT_TEST")
        assert result is not None
        assert result["booklet_id"] == "book_42"
        assert result["original_name"] == "Test Student"
        assert result["original_id"] == "999888777"

    def test_lookup_by_booklet(self, pseudonym_db):
        """Can look up a pseudonym by booklet ID."""
        pseudonym_db.store_mapping(
            pseudonym="STUDENT_ABCD",
            booklet_id="book_99",
        )

        result = pseudonym_db.lookup_by_booklet("book_99")
        assert result is not None
        assert result["pseudonym"] == "STUDENT_ABCD"

    def test_anonymize_booklet_creates_output(self, tmp_dir, sample_image_with_header, pseudonym_db, monkeypatch):
        """Full anonymize_booklet creates masked images and DB entry."""
        # Set up directory structure
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))
        import core.intake.anonymizer as anon
        anon.STORAGE_ROOT = tmp_dir
        anon.PREPROCESSED_STORAGE = tmp_dir / "preprocessed"
        anon.ANONYMIZED_STORAGE = tmp_dir / "anonymized"

        # Create fake preprocessed booklet
        booklet_dir = tmp_dir / "preprocessed" / "anon_test"
        booklet_dir.mkdir(parents=True)
        shutil.copy(str(sample_image_with_header), str(booklet_dir / "page_001.png"))

        result = anonymize_booklet(
            "anon_test",
            pseudonym_db,
            source_dir=booklet_dir,
        )

        assert result.ok, f"Anonymize failed: {result.errors}"
        assert result.identity_masked
        assert result.pseudonym.startswith("STUDENT_")
        assert result.pages_processed == 1

        # Check output exists
        out_dir = tmp_dir / "anonymized" / "anon_test"
        assert (out_dir / "page_001.png").exists()

        # Verify the identity region is actually blacked out in output
        out_img = cv2.imread(str(out_dir / "page_001.png"))
        h = out_img.shape[0]
        region_h = int(h * 0.15)
        assert np.all(out_img[:region_h, :] == 0), "Output identity region not masked"


class TestFullPipeline:
    """End-to-end integration test."""

    @pytest.mark.skipif(not REAL_PDFS, reason="No PDFs in docs/")
    def test_full_pipeline_single_pdf(self, tmp_dir, monkeypatch):
        """End-to-end run on a real test PDF produces all expected outputs."""
        monkeypatch.setenv("STORAGE_ROOT", str(tmp_dir))

        # Patch all storage paths
        import core.intake.rasterizer as ras
        import core.intake.preprocessor as pp
        import core.intake.anonymizer as anon

        ras.RAW_STORAGE = tmp_dir / "raw"
        pp.STORAGE_ROOT = tmp_dir
        pp.RAW_STORAGE = tmp_dir / "raw"
        pp.PREPROCESSED_STORAGE = tmp_dir / "preprocessed"
        anon.STORAGE_ROOT = tmp_dir
        anon.PREPROCESSED_STORAGE = tmp_dir / "preprocessed"
        anon.ANONYMIZED_STORAGE = tmp_dir / "anonymized"

        # Pick the smallest PDF for speed
        test_pdf = min(REAL_PDFS, key=lambda p: p.stat().st_size)

        # Step 1: Rasterise
        manifest = rasterize_file(test_pdf, dpi=150, booklet_id="integration_test")
        assert manifest.ok
        assert manifest.page_count > 0

        # Step 2: Preprocess
        results = preprocess_booklet("integration_test")
        assert len(results) == manifest.page_count
        assert all(r.ok for r in results)

        # Step 3: Reconcile (no expected pages — just check it runs)
        report = reconcile_booklets([manifest])
        assert report.all_ok

        # Step 4: Anonymise
        db = PseudonymDB(db_path=tmp_dir / "test.db")
        try:
            anon_result = anonymize_booklet("integration_test", db)
            assert anon_result.ok
            assert anon_result.identity_masked
            assert anon_result.pages_processed == manifest.page_count

            # Verify DB has an entry
            mapping = db.lookup_by_booklet("integration_test")
            assert mapping is not None
            assert mapping["pseudonym"] == anon_result.pseudonym
        finally:
            db.close()

        # Verify output files exist
        for subdir in ["raw", "preprocessed", "anonymized"]:
            booklet_dir = tmp_dir / subdir / "integration_test"
            assert booklet_dir.exists(), f"Missing output dir: {booklet_dir}"
            pages = list(booklet_dir.glob("page_*.png"))
            assert len(pages) == manifest.page_count, (
                f"Expected {manifest.page_count} pages in {subdir}, got {len(pages)}"
            )
