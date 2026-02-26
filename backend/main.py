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


def _vision_ocr_image_bytes(
    image_bytes: bytes,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Run OCR via OpenAI vision on a single image payload."""
    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise OCR engine for pharmaceutical COA documents. "
                    "Extract all visible text exactly. "
                    "Do not summarize, translate, or add commentary."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract full text from this page. Preserve line breaks and "
                            "table-like rows using pipe separators when obvious."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _render_pdf_pages_to_png_bytes(pdf_bytes: bytes, max_pages: int = 12) -> list[bytes]:
    """Render PDF pages to PNG bytes for vision OCR fallback."""
    if not HAS_FITZ:
        return []
    images: list[bytes] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        scale = 200 / 72
        matrix = fitz.Matrix(scale, scale)
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images


def _extract_with_openai_vision(
    file_bytes: bytes,
    filename: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> dict[str, Any] | None:
    """Fallback extraction path for scanned PDFs/images without local OCR stack."""
    name = (filename or "").lower()
    is_pdf = b"%PDF-" in file_bytes[:1024] or name.endswith(".pdf")

    page_images: list[bytes]
    if is_pdf:
        page_images = _render_pdf_pages_to_png_bytes(file_bytes)
        if not page_images:
            return None
    else:
        page_images = [file_bytes]

    text_parts: list[str] = []
    for idx, image_bytes in enumerate(page_images, start=1):
        try:
            page_text = _vision_ocr_image_bytes(
                image_bytes=image_bytes,
                api_key=api_key,
                model=model,
            )
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
    vision_ocr_model: str = Form("gpt-4o-mini"),
) -> JSONResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    extraction = extract_text_from_upload(file_bytes, filename=file.filename or "")
    if not extraction.get("success") and api_key.strip():
        vision_result = _extract_with_openai_vision(
            file_bytes=file_bytes,
            filename=file.filename or "",
            api_key=api_key.strip(),
            model=vision_ocr_model.strip() or "gpt-4o-mini",
        )
        if vision_result:
            extraction = vision_result
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

        extraction = extract_text_from_upload(file_bytes, filename=file.filename or "")
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
