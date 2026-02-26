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
import os
import tempfile
from typing import Optional

import pdfplumber

try:
    import fitz  # PyMuPDF

    HAS_FITZ = True
except (ImportError, OSError):
    HAS_FITZ = False

try:
    import pypdfium2 as pdfium

    HAS_PDFIUM = True
except (ImportError, OSError):
    HAS_PDFIUM = False

try:
    import camelot

    HAS_CAMELOT = True
except (ImportError, OSError):
    HAS_CAMELOT = False

try:
    import tabula

    HAS_TABULA = True
except (ImportError, OSError):
    HAS_TABULA = False

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


def _looks_like_pdf(file_bytes: bytes) -> bool:
    """Best-effort PDF signature check (handles some prefixed garbage bytes)."""
    return b"%PDF-" in file_bytes[:1024]


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


def _render_pages_to_images_pdfium(pdf_bytes: bytes, dpi: int = 300) -> list:
    """Render PDF pages using pypdfium2 (if available)."""
    images = []
    pdf = pdfium.PdfDocument(pdf_bytes)
    scale = dpi / 72
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        images.append(bitmap.to_pil())
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
    render_errors: list[str] = []

    for renderer_name, renderer in _get_pdf_ocr_renderers():
        try:
            page_images = renderer(pdf_bytes, dpi=300)
            if page_images:
                logger.info(
                    "Rendered %s page(s) for OCR via %s",
                    len(page_images),
                    renderer_name,
                )
                break
        except Exception as e:
            msg = f"{renderer_name}: {e}"
            render_errors.append(msg)
            logger.warning("PDF OCR renderer failed (%s)", msg)

    if not page_images:
        if render_errors:
            logger.warning(
                "Failed to render PDF pages for OCR. Attempts: %s",
                " | ".join(render_errors),
            )
        return None, 0

    return _ocr_images(page_images, preprocess=preprocess, method_label="OCR")


def extract_text_from_image_bytes(
    image_bytes: bytes,
    preprocess: bool = True,
) -> tuple[Optional[str], int]:
    """OCR extraction for uploaded image files or image-like payloads."""
    if not HAS_OCR:
        logger.warning("pytesseract or Pillow not installed; OCR unavailable")
        return None, 0

    images: list["Image.Image"] = []
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            frame_count = getattr(img, "n_frames", 1)
            for frame_idx in range(frame_count):
                if frame_count > 1:
                    img.seek(frame_idx)
                images.append(img.convert("RGB").copy())
    except Exception as e:
        logger.warning("Could not parse file bytes as image: %s", e)
        return None, 0

    return _ocr_images(images, preprocess=preprocess, method_label="Image OCR")


def _get_pdf_ocr_renderers():
    """Ordered list of PDF page renderers for OCR."""
    renderers = []
    if HAS_FITZ:
        renderers.append(("PyMuPDF", _render_pages_to_images_fitz))
    if HAS_PDFIUM:
        renderers.append(("pdfium", _render_pages_to_images_pdfium))
    renderers.append(("pdfplumber", _render_pages_to_images_pdfplumber))
    return renderers


def _ocr_images(
    images: list["Image.Image"],
    preprocess: bool = True,
    method_label: str = "OCR",
) -> tuple[Optional[str], int]:
    """Run tesseract OCR on a list of PIL images."""
    page_count = len(images)
    if page_count == 0:
        return None, 0

    tess_config = "--psm 6 --oem 3"
    text_parts = []

    for i, image in enumerate(images):
        try:
            working_image = _preprocess_image_for_ocr(image) if preprocess else image
            page_text = _extract_best_ocr_text(working_image, fallback_config=tess_config)
            alnum_count = sum(1 for ch in page_text if ch.isalnum())
            if alnum_count < 10:
                logger.info(
                    "%s page %s too short (%s alnum), skipping",
                    method_label,
                    i + 1,
                    alnum_count,
                )
                continue

            text_parts.append(f"--- Page {i + 1} ({method_label}) ---\n{page_text.strip()}")
        except Exception as e:
            logger.warning("%s failed on page %s: %s", method_label, i + 1, e)

    result = "\n\n".join(text_parts).strip()
    return (result if result else None), page_count


def _extract_best_ocr_text(image: "Image.Image", fallback_config: str) -> str:
    """
    Run OCR with multiple page-segmentation modes and choose the best result.
    """
    configs = [
        fallback_config,
        "--psm 4 --oem 3",
        "--psm 11 --oem 3",
    ]
    best_text = ""
    best_score = -1.0

    for config in configs:
        try:
            text = pytesseract.image_to_string(image, lang="eng", config=config)
        except Exception:
            continue

        alnum = sum(1 for ch in text if ch.isalnum())
        confidence = _estimate_ocr_confidence(image, config)
        score = alnum + (2.0 * confidence)

        if score > best_score:
            best_score = score
            best_text = text

    return best_text


def _estimate_ocr_confidence(image: "Image.Image", config: str) -> float:
    """Estimate OCR confidence via image_to_data, if available."""
    try:
        output = pytesseract.image_to_data(
            image,
            lang="eng",
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return 0.0

    confs = []
    for raw_conf in output.get("conf", []):
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if conf >= 0:
            confs.append(conf)

    if not confs:
        return 0.0
    return sum(confs) / len(confs)


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
        method = best["method"]
        text = best["text"]
        advanced_tables = _extract_advanced_table_text(
            pdf_bytes,
            existing_text=text,
        )
        if advanced_tables:
            method += " + advanced tables"

        return {
            "text": text,
            "method": method,
            "success": True,
            "page_count": best["page_count"] or page_count,
            "table_supplement": advanced_tables,
        }

    # Last-resort fallback: sometimes users upload files labeled as .pdf that
    # are actually image payloads. Try direct image OCR on raw bytes.
    if HAS_OCR:
        image_text, image_pages = extract_text_from_image_bytes(
            pdf_bytes,
            preprocess=True,
        )
        if image_text and image_text.strip():
            return {
                "text": image_text,
                "method": "Image OCR fallback",
                "success": True,
                "page_count": image_pages or page_count or 1,
                "table_supplement": "",
            }

    return {
        "text": "",
        "method": "none",
        "success": False,
        "page_count": page_count,
        "table_supplement": "",
    }


def extract_text_from_upload(file_bytes: bytes, filename: str = "") -> dict:
    """
    Unified extraction entrypoint for PDF and image uploads.
    """
    name = (filename or "").lower()
    looks_pdf = _looks_like_pdf(file_bytes) or name.endswith(".pdf")

    if looks_pdf:
        return extract_text_from_pdf(file_bytes)

    # Non-PDF uploads are treated as image files and OCR'd directly.
    text, pages = extract_text_from_image_bytes(file_bytes, preprocess=True)
    if text and text.strip():
        return {
            "text": text,
            "method": "Image OCR",
            "success": True,
            "page_count": pages or 1,
            "table_supplement": "",
        }

    text, pages = extract_text_from_image_bytes(file_bytes, preprocess=False)
    if text and text.strip():
        return {
            "text": text,
            "method": "Image OCR (no preprocessing)",
            "success": True,
            "page_count": pages or 1,
            "table_supplement": "",
        }

    return {
        "text": "",
        "method": "none",
        "success": False,
        "page_count": pages or 0,
        "table_supplement": "",
    }


def get_extraction_capabilities() -> dict:
    """Runtime capability flags for UI diagnostics."""
    return {
        "has_ocr": HAS_OCR,
        "has_fitz": HAS_FITZ,
        "has_pdfium": HAS_PDFIUM,
        "has_camelot": HAS_CAMELOT,
        "has_tabula": HAS_TABULA,
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


def _extract_advanced_table_text(pdf_bytes: bytes, existing_text: str = "") -> str:
    """
    Try advanced table extraction (Camelot/Tabula) and return deduplicated
    pipe-delimited tables as text.
    """
    if not (HAS_CAMELOT or HAS_TABULA):
        return ""

    table_texts: list[str] = []
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as tmp:
            tmp.write(pdf_bytes)
            temp_path = tmp.name

        if HAS_CAMELOT:
            table_texts.extend(_extract_tables_with_camelot(temp_path))
        if HAS_TABULA:
            table_texts.extend(_extract_tables_with_tabula(temp_path))
    except Exception as e:
        logger.warning("Advanced table extraction failed: %s", e)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    existing_lines = {
        _normalise_line_for_dedup(line)
        for line in existing_text.splitlines()
        if line.strip()
    }

    deduped: list[str] = []
    seen: set[str] = set()
    for text in table_texts:
        norm = _normalise_line_for_dedup(text)
        if not norm or norm in seen:
            continue
        if norm in existing_lines:
            continue
        seen.add(norm)
        deduped.append(text)

    return "\n\n".join(deduped)


def _extract_tables_with_camelot(pdf_path: str) -> list[str]:
    """Extract tables via camelot (both lattice and stream modes)."""
    results: list[str] = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages="all",
                flavor=flavor,
            )
            for i, table in enumerate(tables):
                if i >= 12:
                    break
                df = table.df
                if df is None or df.empty:
                    continue
                rows = []
                for _, row in df.fillna("").iterrows():
                    cells = [str(cell).strip() for cell in row.tolist()]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    results.append(
                        f"[Camelot-{flavor} table]\n" + "\n".join(rows)
                    )
        except Exception as e:
            logger.info("Camelot (%s) unavailable for this PDF: %s", flavor, e)
    return results


def _extract_tables_with_tabula(pdf_path: str) -> list[str]:
    """Extract tables via tabula-py."""
    results: list[str] = []
    try:
        dataframes = tabula.read_pdf(
            pdf_path,
            pages="all",
            multiple_tables=True,
            guess=True,
            pandas_options={"dtype": str},
        )
    except Exception as e:
        logger.info("Tabula unavailable for this PDF: %s", e)
        return results

    for i, df in enumerate(dataframes):
        if i >= 12:
            break
        if df is None or df.empty:
            continue
        rows = []
        for _, row in df.fillna("").iterrows():
            cells = [str(cell).strip() for cell in row.tolist()]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            results.append("[Tabula table]\n" + "\n".join(rows))

    return results


def _normalise_line_for_dedup(text: str) -> str:
    return " ".join(text.lower().split())
