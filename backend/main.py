import base64
import inspect
import io
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, Field

from modules.doc_generator import extract_template_hints, generate_structured_doc
from modules.pdf_extractor import extract_text_from_upload, get_extraction_capabilities
from modules.translator import translate_text_structured

try:
    import fitz  # PyMuPDF

    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

logger = logging.getLogger(__name__)


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    model: str = "gpt-4.1"
    template_hints: dict[str, Any] | None = None
    table_supplement: str = ""
    custom_glossary: str = ""


class GenerateDocRequest(BaseModel):
    sections: dict[str, Any]
    original_filename: str
    extraction_method: str = ""
    model_used: str = ""
    template_fields: dict[str, Any] = Field(default_factory=dict)
    template_heading_map: dict[str, Any] = Field(default_factory=dict)
    user_template_base64: str | None = None


class ProcessPipelineResponse(BaseModel):
    success: bool
    extraction: dict[str, Any] | None = None
    translation: dict[str, Any] | None = None
    error: str | None = None


def _run_translation_structured(
    text: str,
    api_key: str,
    model: str,
    template_hints: dict[str, Any] | None,
    table_supplement: str,
    custom_glossary: str = "",
) -> dict[str, Any]:
    params = inspect.signature(translate_text_structured).parameters
    kwargs: dict[str, Any] = {
        "text": text,
        "api_key": api_key,
        "model": model,
        "progress_callback": None,
    }
    if "template_hints" in params:
        kwargs["template_hints"] = template_hints
    if "table_supplement" in params:
        kwargs["table_supplement"] = table_supplement
    if "custom_glossary" in params:
        kwargs["custom_glossary"] = custom_glossary
    return translate_text_structured(**kwargs)


def _run_generate_structured_doc(
    sections: dict[str, Any],
    original_filename: str,
    extraction_method: str,
    model_used: str,
    user_template_bytes: bytes | None,
    template_fields: dict[str, Any],
    template_heading_map: dict[str, Any],
) -> bytes:
    params = inspect.signature(generate_structured_doc).parameters
    kwargs: dict[str, Any] = {
        "sections": sections,
        "original_filename": original_filename,
        "extraction_method": extraction_method,
        "model_used": model_used,
        "user_template_bytes": user_template_bytes,
    }
    if "template_fields" in params:
        kwargs["template_fields"] = template_fields
    if "template_heading_map" in params:
        kwargs["template_heading_map"] = template_heading_map
    return generate_structured_doc(**kwargs)


def _decode_template_from_base64(value: str | None) -> bytes | None:
    if not value:
        return None
    try:
        return base64.b64decode(value)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail="Invalid template base64 payload") from exc


def _score_text_quality(text: str, page_count: int) -> float:
    """Heuristic quality score used to compare extraction variants."""
    if not text:
        return 0.0
    safe_pages = max(page_count, 1)
    non_ws = sum(1 for ch in text if not ch.isspace())
    alnum = sum(1 for ch in text if ch.isalnum())
    lines = text.count("\n") + 1
    return alnum + (0.15 * non_ws) + (2.5 * lines) + (0.6 * (alnum / safe_pages))


def _score_extraction_payload(payload: dict[str, Any] | None) -> float:
    if not payload or not payload.get("success"):
        return 0.0
    text = str(payload.get("text") or "")
    page_count = int(payload.get("page_count") or 1)
    return _score_text_quality(text, page_count)


def _is_weak_extraction(payload: dict[str, Any] | None) -> bool:
    """
    Identify low-quality extraction (common for scanned/image-like PDFs with
    hidden sparse text layers).
    """
    if not payload or not payload.get("success"):
        return True
    text = str(payload.get("text") or "").strip()
    if not text:
        return True
    pages = max(int(payload.get("page_count") or 1), 1)
    chars_per_page = len(text) / pages
    alnum_per_page = sum(1 for ch in text if ch.isalnum()) / pages
    # Typical COA pages are dense; anything below this is usually incomplete.
    return chars_per_page < 650 or alnum_per_page < 260


def _filename_looks_like_image(name: str) -> bool:
    n = (name or "").lower()
    return n.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"))


def _looks_like_pdf(file_bytes: bytes, filename: str) -> bool:
    return b"%PDF-" in file_bytes[:1024] or (filename or "").lower().endswith(".pdf")


def _normalise_line_for_merge(text: str) -> str:
    return " ".join(text.lower().split())


def _merge_ocr_text_blocks(blocks: list[str]) -> str:
    """Merge OCR chunks while removing obvious overlap duplicates."""
    merged_lines: list[str] = []
    seen_recent: list[str] = []
    for block in blocks:
        for raw_line in block.splitlines():
            line = raw_line.rstrip()
            norm = _normalise_line_for_merge(line)
            if not norm:
                merged_lines.append("")
                continue
            if norm in seen_recent:
                continue
            merged_lines.append(line)
            seen_recent.append(norm)
            if len(seen_recent) > 48:
                seen_recent = seen_recent[-48:]
    return "\n".join(merged_lines).strip()


def _unique_alnum_gain(primary_text: str, secondary_text: str) -> int:
    """
    Rough measure of unique content contributed by the secondary text.
    """
    primary_lines = {
        _normalise_line_for_merge(line)
        for line in (primary_text or "").splitlines()
        if _normalise_line_for_merge(line)
    }
    gain = 0
    for line in (secondary_text or "").splitlines():
        norm = _normalise_line_for_merge(line)
        if not norm or norm in primary_lines:
            continue
        gain += sum(1 for ch in line if ch.isalnum())
    return gain


def _split_image_for_dense_ocr(
    image_bytes: bytes,
    vertical_parts: int = 4,
    overlap_px: int = 120,
) -> list[bytes]:
    """Split a dense page into overlapping vertical chunks for higher OCR recall."""
    chunks: list[bytes] = []
    with Image.open(io.BytesIO(image_bytes)) as img:
        src = img.convert("RGB")
        w, h = src.size
        if h < 1400:
            return [image_bytes]
        part_h = max(h // vertical_parts, 1)
        for idx in range(vertical_parts):
            top = max(0, idx * part_h - overlap_px)
            bottom = min(h, (idx + 1) * part_h + overlap_px)
            crop = src.crop((0, top, w, bottom))
            buff = io.BytesIO()
            crop.save(buff, format="PNG")
            chunks.append(buff.getvalue())
    return chunks or [image_bytes]


def _prepare_image_for_vision(
    image_bytes: bytes,
    min_width: int = 1800,
) -> bytes:
    """
    Light preprocessing before Vision OCR to improve small-text readability.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        if rgb.width < min_width:
            scale = min_width / max(rgb.width, 1)
            rgb = rgb.resize(
                (int(rgb.width * scale), int(rgb.height * scale)),
                Image.LANCZOS,
            )
        # Keep color image but improve contrast/sharpness for dense scans.
        rgb = ImageOps.autocontrast(rgb, cutoff=1)
        rgb = rgb.filter(ImageFilter.SHARPEN)
        buff = io.BytesIO()
        rgb.save(buff, format="PNG", optimize=True)
        return buff.getvalue()


def _vision_model_candidates(requested: str) -> list[str]:
    candidates = [
        (requested or "").strip(),
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ]
    deduped: list[str] = []
    for model in candidates:
        if model and model not in deduped:
            deduped.append(model)
    return deduped


def _vision_ocr_image_bytes(
    image_bytes: bytes,
    client: OpenAI,
    model: str = "gpt-4o",
) -> str:
    """Run OCR via OpenAI vision on a single image payload."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    system_text = (
        "You are an ultra-accurate OCR engine for pharmaceutical Certificate of Analysis (COA) documents. "
        "These are compliance-critical files. Missing a single limit/value/row is unacceptable. "
        "Extract ALL visible text exactly, including headers, footers, small-print notes, "
        "specification limits, batch fields, methods, acceptance criteria, and table rows. "
        "Do NOT summarize. Do NOT translate. Do NOT infer. Keep original language, punctuation, and numbers."
    )
    user_text = (
        "OCR this page with maximum recall and precision. "
        "Preserve reading order top-to-bottom. Keep one logical line per line. "
        "For tables, preserve each row using pipe delimiters, e.g. col1 | col2 | col3. "
        "Include every detected line, including short codes and tiny text."
    )

    # Try Responses API first with high image detail and explicit output budget.
    if hasattr(client, "responses"):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_text}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_text},
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": "high",
                            },
                        ],
                    },
                ],
                max_output_tokens=6000,
            )
            text = (getattr(response, "output_text", "") or "").strip()
            if text:
                return text
        except Exception as e:
            logger.info("Responses API OCR failed, fallback to chat: %s", e)

    # Fallback for compatibility.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            },
        ],
        max_completion_tokens=6000,
    )
    return (response.choices[0].message.content or "").strip()


def _vision_ocr_recall_pass(
    *,
    image_bytes: bytes,
    draft_text: str,
    client: OpenAI,
    model: str = "gpt-4o",
) -> str:
    """
    Second vision pass: find text missed by the first OCR draft.
    Returns only additional missing lines.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    system_text = (
        "You are auditing OCR completeness for a pharmaceutical COA page. "
        "Return ONLY text lines missing from the provided OCR draft. "
        "Critical: do not omit numeric values, limits, units, or table cells."
    )
    user_text = (
        "Given the image and existing OCR draft, output only missing lines. "
        "No commentary. If nothing is missing, return an empty string.\n\n"
        "=== OCR DRAFT START ===\n"
        f"{draft_text}\n"
        "=== OCR DRAFT END ==="
    )

    if hasattr(client, "responses"):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_text}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_text},
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": "high",
                            },
                        ],
                    },
                ],
                max_output_tokens=2500,
            )
            return (getattr(resp, "output_text", "") or "").strip()
        except Exception as e:
            logger.info("Vision recall pass (responses) failed: %s", e)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
            max_completion_tokens=2500,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.info("Vision recall pass (chat) failed: %s", e)
        return ""


def _render_pdf_pages_to_png_bytes(pdf_bytes: bytes, max_pages: int = 50) -> list[bytes]:
    """Render PDF pages to PNG bytes for vision OCR fallback."""
    if not HAS_FITZ:
        return []
    images: list[bytes] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        scale = 300 / 72
        matrix = fitz.Matrix(scale, scale)
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(_prepare_image_for_vision(pix.tobytes("png")))
    finally:
        doc.close()
    return images


def _extract_with_openai_vision(
    file_bytes: bytes,
    filename: str,
    api_key: str,
    model: str = "gpt-4o",
) -> dict[str, Any] | None:
    """Fallback extraction path for scanned PDFs/images without local OCR stack."""
    name = (filename or "").lower()
    is_pdf = b"%PDF-" in file_bytes[:1024] or name.endswith(".pdf")

    page_images: list[bytes]
    if is_pdf:
        page_images = _render_pdf_pages_to_png_bytes(file_bytes, max_pages=12)
        if not page_images:
            return None
    else:
        page_images = [_prepare_image_for_vision(file_bytes)]

    client = OpenAI(api_key=api_key)
    text_parts: list[str] = []
    for idx, image_bytes in enumerate(page_images, start=1):
        try:
            primary_text = _vision_ocr_image_bytes(
                image_bytes=image_bytes,
                client=client,
                model=model,
            )
            page_text = primary_text
            # High-recall fallback only when primary output is suspiciously short.
            if len(primary_text.strip()) < 1200:
                chunk_texts: list[str] = []
                for chunk_bytes in _split_image_for_dense_ocr(image_bytes):
                    try:
                        t = _vision_ocr_image_bytes(
                            image_bytes=chunk_bytes,
                            client=client,
                            model=model,
                        )
                    except Exception:
                        t = ""
                    if t.strip():
                        chunk_texts.append(t.strip())
                if chunk_texts:
                    page_text = _merge_ocr_text_blocks([primary_text] + chunk_texts)

            # Recall pass only for sparse pages to avoid runaway latency.
            if len(page_text.strip()) < 1800:
                missing_lines = _vision_ocr_recall_pass(
                    image_bytes=image_bytes,
                    draft_text=page_text,
                    client=client,
                    model=model,
                )
                if missing_lines.strip():
                    page_text = _merge_ocr_text_blocks([page_text, missing_lines])
        except Exception as exc:
            logger.warning("Vision OCR page %s failed: %s", idx, exc)
            continue
        if page_text:
            text_parts.append(f"--- Page {idx} (Vision OCR) ---\n{page_text}")

    merged = "\n\n".join(text_parts).strip()
    if not merged:
        return None

    return {
        "text": merged,
        "method": f"OpenAI Vision OCR ({model})",
        "success": True,
        "page_count": len(page_images),
        "table_supplement": "",
    }


def _vision_translate_single_request(
    *,
    file_bytes: bytes,
    filename: str,
    api_key: str,
    model: str,
    template_hints: dict[str, Any] | None = None,
    custom_glossary: str = "",
) -> dict[str, Any]:
    """
    Single backend endpoint path (AI-first):
    1) Vision OCR with high-recall extraction
    2) Structured RU translation

    Note: this is one API call from client perspective, while internally we use
    multiple model calls for reliability.
    """
    # Vision OCR model candidates are separate from translation model choice.
    vision_candidates = _vision_model_candidates("gpt-4o")
    ocr_result: dict[str, Any] | None = None
    ocr_error = ""
    for vision_model in vision_candidates:
        try:
            candidate = _extract_with_openai_vision(
                file_bytes=file_bytes,
                filename=filename,
                api_key=api_key,
                model=vision_model,
            )
            if candidate and candidate.get("success"):
                ocr_result = candidate
                ocr_result["method"] = f"Vision OCR ({vision_model})"
                break
        except Exception as exc:
            ocr_error = str(exc)
            logger.warning("Vision OCR model '%s' failed: %s", vision_model, exc)

    if not ocr_result or not ocr_result.get("success"):
        return {
            "source_text": "",
            "translated_text": "",
            "sections": {},
            "template_fields": {},
            "template_heading_map": {},
            "success": False,
            "error": ocr_error or "Vision OCR failed",
            "model_used": model,
            "chunks_translated": 0,
        }

    source_text = str(ocr_result.get("text") or "").strip()
    if not source_text:
        return {
            "source_text": "",
            "translated_text": "",
            "sections": {},
            "template_fields": {},
            "template_heading_map": {},
            "success": False,
            "error": "Vision OCR returned empty text",
            "model_used": model,
            "chunks_translated": 0,
        }

    translation = _run_translation_structured(
        text=source_text,
        api_key=api_key,
        model=model,
        template_hints=template_hints,
        table_supplement=str(ocr_result.get("table_supplement") or ""),
        custom_glossary=custom_glossary,
    )

    if not translation.get("success"):
        translation["source_text"] = source_text
        return translation

    return {
        **translation,
        "source_text": source_text,
        "method": (
            f"{ocr_result.get('method', 'Vision OCR')} + structured translation "
            f"({translation.get('model_used', model)})"
        ),
    }


def _extract_with_best_strategy(
    *,
    file_bytes: bytes,
    filename: str,
    api_key: str,
    ocr_mode: str = "auto",
    vision_ocr_model: str = "gpt-4o",
) -> dict[str, Any]:
    """
    Unified extraction strategy:
    - always try local extractor first (digital text + local OCR when available)
    - run vision OCR based on mode / weak quality signals
    - choose the higher-quality result
    """
    mode = (ocr_mode or "auto").strip().lower()
    if mode not in {"auto", "vision_only", "local_only"}:
        mode = "auto"

    use_ai_first = bool(api_key.strip()) and mode != "local_only"
    vision_result: dict[str, Any] | None = None
    local_result: dict[str, Any] | None = None

    if use_ai_first:
        vision_result = _extract_with_openai_vision(
            file_bytes=file_bytes,
            filename=filename,
            api_key=api_key.strip(),
            model=vision_ocr_model.strip() or "gpt-4o",
        )
        # Second request path: run local extraction after AI to recover any extras.
        local_result = extract_text_from_upload(file_bytes, filename=filename)

        if vision_result and vision_result.get("success"):
            chosen = dict(vision_result)

            if local_result and local_result.get("success"):
                local_text = str(local_result.get("text") or "")
                vision_text = str(vision_result.get("text") or "")
                gain = _unique_alnum_gain(vision_text, local_text)
                if gain >= 140:
                    merged_text = _merge_ocr_text_blocks([vision_text, local_text])
                    if merged_text:
                        chosen["text"] = merged_text
                        chosen["method"] = (
                            f"{vision_result.get('method')} + local recovery"
                        )
                        chosen["page_count"] = max(
                            int(vision_result.get("page_count") or 1),
                            int(local_result.get("page_count") or 1),
                        )

            chosen["success"] = True
            chosen["table_supplement"] = (
                local_result.get("table_supplement", "")
                if local_result and local_result.get("success")
                else ""
            )
            chosen["quality_score"] = round(_score_extraction_payload(chosen), 2)
            chosen["vision_quality_score"] = round(_score_extraction_payload(vision_result), 2)
            chosen["local_quality_score"] = (
                round(_score_extraction_payload(local_result), 2)
                if local_result
                else 0.0
            )
            return chosen

        # AI-first requested but failed: fallback to local pass result.
        if local_result and local_result.get("success"):
            fallback = dict(local_result)
            fallback["method"] = f"{fallback.get('method', 'local')} (AI-first fallback)"
            fallback["quality_score"] = round(_score_extraction_payload(fallback), 2)
            fallback["vision_quality_score"] = 0.0
            fallback["local_quality_score"] = round(_score_extraction_payload(local_result), 2)
            return fallback

        return {
            "text": "",
            "method": "none",
            "success": False,
            "page_count": 0,
            "table_supplement": "",
            "error": "AI-first OCR failed to extract readable text from this file.",
        }

    # local_only or no API key provided
    local_result = extract_text_from_upload(file_bytes, filename=filename)
    if local_result.get("success"):
        local_result = dict(local_result)
        local_result["quality_score"] = round(_score_extraction_payload(local_result), 2)
        local_result["vision_quality_score"] = 0.0
        local_result["local_quality_score"] = round(_score_extraction_payload(local_result), 2)
        return local_result

    if _is_weak_extraction(local_result):
        return {
            "text": "",
            "method": "none",
            "success": False,
            "page_count": int(local_result.get("page_count") or 0),
            "table_supplement": "",
            "error": (
                "Could not extract readable text. Provide an OpenAI API key and use "
                "Vision-first OCR for scanned/image-like files."
            ),
        }

    return local_result


app = FastAPI(title="COA Translator API", version="3.0.0")

allowed_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    caps = get_extraction_capabilities()
    caps["has_vision_ocr"] = True
    return caps


@app.post("/api/extract")
async def extract(
    file: UploadFile = File(...),
    template: UploadFile | None = File(None),
    api_key: str = Form(""),
    vision_ocr_model: str = Form("gpt-4o"),
    ocr_mode: str = Form("auto"),
) -> JSONResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    extraction = _extract_with_best_strategy(
        file_bytes=file_bytes,
        filename=file.filename or "",
        api_key=api_key,
        ocr_mode=ocr_mode,
        vision_ocr_model=vision_ocr_model,
    )
    template_hints = None

    if template is not None:
        template_bytes = await template.read()
        if template_bytes:
            template_hints = extract_template_hints(template_bytes)

    payload = {**extraction, "template_hints": template_hints}
    status_code = 200 if extraction.get("success") else 422
    return JSONResponse(content=payload, status_code=status_code)


@app.post("/api/translate")
def translate(req: TranslateRequest) -> JSONResponse:
    result = _run_translation_structured(
        text=req.text,
        api_key=req.api_key,
        model=req.model,
        template_hints=req.template_hints,
        table_supplement=req.table_supplement,
        custom_glossary=req.custom_glossary,
    )
    status_code = 200 if result.get("success") else 422
    return JSONResponse(content=result, status_code=status_code)


@app.post("/api/vision-translate")
async def vision_translate(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    model: str = Form("gpt-5-chat-latest"),
    template: UploadFile | None = File(None),
    custom_glossary: str = Form(""),
) -> JSONResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    template_hints = None
    if template is not None:
        template_bytes = await template.read()
        if template_bytes:
            template_hints = extract_template_hints(template_bytes)

    result = _vision_translate_single_request(
        file_bytes=file_bytes,
        filename=file.filename or "",
        api_key=api_key.strip(),
        model=model,
        template_hints=template_hints,
        custom_glossary=custom_glossary,
    )

    # Helpful metadata for UI diagnostics.
    if result.get("success"):
        result["method"] = f"AI-first vision translate ({result.get('model_used')})"
        result["source_file"] = file.filename or ""
    status_code = 200 if result.get("success") else 422
    return JSONResponse(content=result, status_code=status_code)


@app.post("/api/generate-doc")
def generate_doc(req: GenerateDocRequest) -> StreamingResponse:
    template_bytes = _decode_template_from_base64(req.user_template_base64)

    doc_bytes = _run_generate_structured_doc(
        sections=req.sections,
        original_filename=req.original_filename,
        extraction_method=req.extraction_method,
        model_used=req.model_used,
        user_template_bytes=template_bytes,
        template_fields=req.template_fields,
        template_heading_map=req.template_heading_map,
    )

    base_name = Path(req.original_filename).stem or "coa"
    output_filename = f"{base_name}_RU.docx"

    return StreamingResponse(
        io.BytesIO(doc_bytes),
        media_type=(
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{output_filename}"',
        },
    )


@app.post("/api/process", response_model=ProcessPipelineResponse)
async def process(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    model: str = Form("gpt-4.1"),
    template: UploadFile | None = File(None),
    custom_glossary: str = Form(""),
    vision_ocr_model: str = Form("gpt-4o"),
    ocr_mode: str = Form("auto"),
) -> ProcessPipelineResponse:
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return ProcessPipelineResponse(success=False, error="Uploaded file is empty")

        template_bytes = None
        template_hints = None
        if template is not None:
            template_bytes = await template.read()
            if template_bytes:
                template_hints = extract_template_hints(template_bytes)

        prefer_single_vision = bool(api_key.strip()) and ocr_mode in {"vision_only", "auto"}
        if prefer_single_vision:
            translation = _vision_translate_single_request(
                file_bytes=file_bytes,
                filename=file.filename or "",
                api_key=api_key.strip(),
                model=model,
                template_hints=template_hints,
                custom_glossary=custom_glossary,
            )
            extraction = {
                "success": translation.get("success", False),
                "text": "",
                "method": f"AI-first vision translate ({translation.get('model_used', model)})",
                "page_count": 0,
                "table_supplement": "",
            }
        else:
            extraction = _extract_with_best_strategy(
                file_bytes=file_bytes,
                filename=file.filename or "",
                api_key=api_key,
                ocr_mode=ocr_mode,
                vision_ocr_model=vision_ocr_model,
            )
            if not extraction.get("success"):
                return ProcessPipelineResponse(success=False, extraction=extraction, error=extraction.get("error"))

            translation = _run_translation_structured(
                text=extraction["text"],
                api_key=api_key,
                model=model,
                template_hints=template_hints,
                table_supplement=extraction.get("table_supplement", ""),
                custom_glossary=custom_glossary,
            )
        if not translation.get("success"):
            return ProcessPipelineResponse(
                success=False,
                extraction=extraction,
                translation=translation,
                error=translation.get("error"),
            )

        doc_bytes = _run_generate_structured_doc(
            sections=translation.get("sections", {}),
            original_filename=file.filename or "coa.pdf",
            extraction_method=extraction.get("method", "unknown"),
            model_used=translation.get("model_used", model),
            user_template_bytes=template_bytes,
            template_fields=translation.get("template_fields", {}),
            template_heading_map=translation.get("template_heading_map", {}),
        )

        return ProcessPipelineResponse(
            success=True,
            extraction={**extraction, "template_hints": template_hints},
            translation={
                **translation,
                "docx_base64": base64.b64encode(doc_bytes).decode("utf-8"),
            },
            error=None,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Pipeline failure")
        return ProcessPipelineResponse(success=False, error=str(exc))
