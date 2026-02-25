"""
PDF text extraction module.

Uses multiple strategies to extract text from pharmaceutical COA PDFs:
1. pdfplumber (primary) - good for structured/tabular PDFs
2. PyMuPDF/fitz (fallback) - good for general text extraction
3. pytesseract OCR (last resort) - for scanned/image-based PDFs

OCR pipeline includes image preprocessing (grayscale, contrast enhancement,
binarization, deskew) for improved accuracy on scanned COA documents.
"""

import io
import logging
from typing import Optional

import pdfplumber

try:
    import fitz  # PyMuPDF

    HAS_FITZ = True
except (ImportError, OSError):
    HAS_FITZ = False

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps

    # Verify pytesseract can actually locate the tesseract binary
    pytesseract.get_tesseract_version()
    HAS_OCR = True
except (ImportError, OSError, Exception):
    HAS_OCR = False

logger = logging.getLogger(__name__)


def _score_extracted_text(text: str, page_count: int) -> float:
    """Heuristic score to compare competing extraction outputs."""
    if not text:
        return 0.0

    safe_pages = max(page_count, 1)
    non_ws = sum(1 for ch in text if not ch.isspace())
    alnum = sum(1 for ch in text if ch.isalnum())
    line_count = text.count("\n") + 1

    # Prefer denser text and better per-page coverage.
    return (
        alnum
        + 0.2 * non_ws
        + 2.0 * line_count
        + 0.5 * (alnum / safe_pages)
    )


def _is_sparse_text(text: str, page_count: int) -> bool:
    """
    Identify suspiciously thin extraction (common for scanned PDFs with tiny
    hidden text layers), where OCR should be attempted.
    """
    if not text:
        return True

    safe_pages = max(page_count, 1)
    chars_per_page = len(text.strip()) / safe_pages
    alnum_per_page = sum(1 for ch in text if ch.isalnum()) / safe_pages
    return chars_per_page < 450 or alnum_per_page < 180


# ---------------------------------------------------------------------------
# Image preprocessing helpers for OCR
# ---------------------------------------------------------------------------

def _preprocess_image_for_ocr(image: "Image.Image") -> "Image.Image":
    """
    Apply a sequence of image preprocessing steps to improve OCR accuracy
    on scanned pharmaceutical COA documents.

    Steps:
        1. Convert to grayscale
        2. Upscale small images to ensure minimum effective DPI
        3. Enhance contrast via autocontrast
        4. Apply slight sharpening
        5. Binarize with adaptive-like thresholding (Otsu via point())
    """
    # 1. Grayscale
    img = image.convert("L")

    # 2. Upscale if the image is small (ensures ~300 DPI equivalent)
    min_width = 2000
    if img.width < min_width:
        scale = min_width / img.width
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )

    # 3. Autocontrast — stretches the histogram to use the full 0-255 range
    img = ImageOps.autocontrast(img, cutoff=1)

    # 4. Sharpen — helps with slightly blurry scans
    img = img.filter(ImageFilter.SHARPEN)

    # 5. Binarize — simple threshold; works well after autocontrast
    threshold = 180
    img = img.point(lambda px: 255 if px > threshold else 0, mode="1")

    # Convert back to L for tesseract compatibility
    img = img.convert("L")

    return img


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------

def extract_with_pdfplumber(pdf_bytes: bytes) -> tuple[Optional[str], int]:
    """
    Extract text using pdfplumber.  Works well with structured PDFs
    that contain tables (common in COA documents).

    Returns (text, page_count).
    """
    try:
        text_parts = []
        page_count = 0
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

                # Also try to extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = _format_table(table)
                        if table_text:
                            text_parts.append(table_text)

        result = "\n\n".join(text_parts).strip()
        return (result if result else None), page_count
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
        return None, 0


def extract_with_pymupdf(pdf_bytes: bytes) -> tuple[Optional[str], int]:
    """
    Extract text using PyMuPDF (fitz).  Good general-purpose extraction.

    Returns (text, page_count).
    """
    if not HAS_FITZ:
        logger.info("PyMuPDF (fitz) not available, skipping")
        return None, 0

    try:
        text_parts = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        for i, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"--- Page {i + 1} ---\n{page_text.strip()}")
        doc.close()

        result = "\n\n".join(text_parts).strip()
        return (result if result else None), page_count
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
        return None, 0


def _render_pages_to_images_fitz(pdf_bytes: bytes, dpi: int = 300) -> list:
    """Render PDF pages to PIL Images using PyMuPDF at the given DPI."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        images.append(Image.open(io.BytesIO(img_bytes)))
    doc.close()
    return images


def _render_pages_to_images_pdfplumber(pdf_bytes: bytes, dpi: int = 300) -> list:
    """
    Render PDF pages to PIL Images using pdfplumber's built-in
    page.to_image().  This does NOT require PyMuPDF.
    """
    images = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_img = page.to_image(resolution=dpi)
            images.append(page_img.original)  # PIL Image
    return images


def extract_with_ocr(
    pdf_bytes: bytes,
    preprocess: bool = True,
) -> tuple[Optional[str], int]:
    """
    Extract text using OCR (pytesseract) for scanned / image-based PDFs.

    Rendering pipeline:
        - Prefers PyMuPDF (high-quality rasterisation).
        - Falls back to pdfplumber's page.to_image() if fitz is unavailable.

    Image preprocessing (when *preprocess* is True):
        grayscale → upscale → autocontrast → sharpen → binarize

    Tesseract is invoked with:
        - ``--psm 6`` (assume a single uniform block of text — good for
          structured documents / forms / COA tables).
        - ``--oem 3`` (default LSTM engine).

    Returns (text, page_count).
    """
    if not HAS_OCR:
        logger.warning("pytesseract or Pillow not installed; OCR unavailable")
        return None, 0

    # --- Render pages to images ------------------------------------------
    page_images: list["Image.Image"] = []
    try:
        if HAS_FITZ:
            page_images = _render_pages_to_images_fitz(pdf_bytes, dpi=300)
        else:
            page_images = _render_pages_to_images_pdfplumber(pdf_bytes, dpi=300)
    except Exception as e:
        logger.warning(f"Failed to render PDF pages to images: {e}")
        return None, 0

    page_count = len(page_images)
    if page_count == 0:
        return None, 0

    # --- OCR each page ---------------------------------------------------
    # Tesseract config: PSM 6 = single uniform block of text,
    #                   OEM 3 = default LSTM engine
    tess_config = "--psm 6 --oem 3"

    text_parts = []
    for i, image in enumerate(page_images):
        try:
            if preprocess:
                image = _preprocess_image_for_ocr(image)

            page_text = pytesseract.image_to_string(
                image,
                lang="eng",
                config=tess_config,
            )

            # Quality gate: reject pages with too few alphanumeric characters
            alnum_count = sum(1 for ch in page_text if ch.isalnum())
            if alnum_count < 10:
                logger.info(f"Page {i + 1}: OCR output too short ({alnum_count} alnum chars), skipping")
                continue

            text_parts.append(f"--- Page {i + 1} (OCR) ---\n{page_text.strip()}")
        except Exception as e:
            logger.warning(f"OCR failed on page {i + 1}: {e}")
            continue

    result = "\n\n".join(text_parts).strip()
    return (result if result else None), page_count


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Main extraction function.  Tries multiple strategies and returns
    the best result.

    Returns:
        dict with keys:
            - 'text': extracted text string
            - 'method': which extraction method was used
            - 'success': whether extraction succeeded
            - 'page_count': number of pages in the PDF
    """
    def _candidate(text: Optional[str], method: str, page_count: int) -> Optional[dict]:
        if not text or not text.strip():
            return None

        score = _score_extracted_text(text, page_count)
        logger.info(
            "Extraction candidate %s: pages=%s, chars=%s, score=%.2f",
            method,
            page_count,
            len(text),
            score,
        )
        return {
            "text": text,
            "method": method,
            "page_count": page_count,
            "score": score,
        }

    candidates: list[dict] = []
    digital_candidates: list[dict] = []
    page_count = 0

    # Strategy 1: pdfplumber (best for table-rich structured PDFs)
    text, pc = extract_with_pdfplumber(pdf_bytes)
    if pc > 0:
        page_count = max(page_count, pc)
    c = _candidate(text, "pdfplumber", pc or page_count)
    if c:
        candidates.append(c)
        digital_candidates.append(c)

    # Strategy 2: PyMuPDF (alternative digital text extractor)
    text, pc = extract_with_pymupdf(pdf_bytes)
    if pc > 0:
        page_count = max(page_count, pc)
    c = _candidate(text, "PyMuPDF", pc or page_count)
    if c:
        candidates.append(c)
        digital_candidates.append(c)

    best_digital = (
        max(digital_candidates, key=lambda item: item["score"])
        if digital_candidates
        else None
    )

    # OCR is expensive, so run it only when digital extraction is missing/sparse.
    should_try_ocr = HAS_OCR and (
        best_digital is None
        or _is_sparse_text(
            best_digital["text"],
            best_digital["page_count"],
        )
    )

    if should_try_ocr:
        logger.info("Digital extraction is weak; attempting OCR fallback")

        # Strategy 3: OCR with preprocessing
        text, pc = extract_with_ocr(pdf_bytes, preprocess=True)
        if pc > 0:
            page_count = max(page_count, pc)
        c = _candidate(text, "OCR (pytesseract)", pc or page_count)
        if c:
            candidates.append(c)

        # Strategy 4: OCR without preprocessing
        text, pc = extract_with_ocr(pdf_bytes, preprocess=False)
        if pc > 0:
            page_count = max(page_count, pc)
        c = _candidate(text, "OCR (pytesseract, no preprocessing)", pc or page_count)
        if c:
            candidates.append(c)

    if candidates:
        best = max(candidates, key=lambda item: item["score"])
        return {
            "text": best["text"],
            "method": best["method"],
            "success": True,
            "page_count": best["page_count"] or page_count,
        }

    return {
        "text": "",
        "method": "none",
        "success": False,
        "page_count": page_count,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_table(table: list) -> str:
    """Format an extracted table as readable text."""
    if not table:
        return ""

    rows = []
    for row in table:
        if row:
            cells = [str(cell).strip() if cell else "" for cell in row]
            rows.append(" | ".join(cells))

    return "\n".join(rows) if rows else ""
