"""
Pharmaceutical COA Translator — Streamlit Application

Upload a pharmaceutical Certificate of Analysis (COA) in PDF format,
translate it to Russian using OpenAI with a pharmaceutical glossary,
and download the result as a fixed-structure Word document.
"""

import difflib
import inspect

import streamlit as st

from modules.pdf_extractor import (
    extract_text_from_upload,
    get_extraction_capabilities,
)
from modules.translator import translate_text_structured
from modules.doc_generator import generate_structured_doc, extract_template_hints


def _run_translation_structured(
    text: str,
    api_key: str,
    model: str,
    progress_callback,
    template_hints: dict | None,
    table_supplement: str,
):
    """
    Call translator with only the kwargs supported by the currently loaded
    module version. This avoids runtime crashes on Streamlit Cloud workers
    that may briefly run mixed code during deploy/update.
    """
    params = inspect.signature(translate_text_structured).parameters
    kwargs = {
        "text": text,
        "api_key": api_key,
        "model": model,
        "progress_callback": progress_callback,
    }
    if "template_hints" in params:
        kwargs["template_hints"] = template_hints
    if "table_supplement" in params:
        kwargs["table_supplement"] = table_supplement
    return translate_text_structured(**kwargs)


def _run_generate_structured_doc(
    sections: dict,
    original_filename: str,
    extraction_method: str,
    model_used: str,
    user_template_bytes: bytes | None,
    template_fields: dict,
    template_heading_map: dict,
):
    """
    Backward-compatible call for doc generation across rolling deploys.
    """
    params = inspect.signature(generate_structured_doc).parameters
    kwargs = {
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="COA Translator — EN → RU",
    page_icon="💊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        color: #1E88E5;
        font-size: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">'
    "<h1>Pharmaceutical COA Translator</h1>"
    "<p>English → Russian | Certificate of Analysis</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. Used only for the current session.",
    )

    model_choice = st.selectbox(
        "Translation Model",
        options=[
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "Custom model ID",
        ],
        index=0,
        help=(
            "Choose your model tier. If your org has access to newer models, "
            "you can also enter any custom model ID."
        ),
    )
    custom_model_id = ""
    if model_choice == "Custom model ID":
        custom_model_id = st.text_input(
            "Custom model ID",
            placeholder="e.g. gpt-5",
            help="Exact model ID from your OpenAI account access.",
        )
    selected_model = custom_model_id.strip() or model_choice
    selected_model_valid = not (
        model_choice == "Custom model ID" and not custom_model_id.strip()
    )

    st.divider()

    st.subheader("Word Template (optional)")
    user_template = st.file_uploader(
        "Upload a .docx structure template",
        type=["docx"],
        help=(
            "Upload your own Word template with Jinja2 placeholders "
            "(e.g. {{ product_name }}, {{ test_results }}). "
            "If not provided, the built-in fixed COA structure is used.\n\n"
            "**Available placeholders:** document_title, company_info, "
            "product_name, product_details, batch_info, storage_conditions, "
            "test_results, conclusion, signatures, notes, "
            "original_filename, translation_date"
        ),
    )

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This app translates pharmaceutical Certificate of Analysis (COA) "
        "documents from English to Russian using AI with a specialized "
        "pharmaceutical glossary."
    )
    st.markdown(
        "**Features:**\n"
        "- Multi-method PDF extraction\n"
        "- OCR with image preprocessing\n"
        "- 200+ pharma term glossary\n"
        "- Fixed-structure Word output\n"
        "- Custom template support"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.subheader("1. Upload COA File")

caps = get_extraction_capabilities()
if not caps["has_ocr"]:
    st.warning(
        "OCR engine is not available in this environment. Scanned PDFs/images "
        "will not be readable until Tesseract OCR is installed."
    )
if not (caps.get("has_camelot") or caps.get("has_tabula")):
    st.caption(
        "Advanced table extractors (Camelot/Tabula) are unavailable in this "
        "runtime; baseline extraction will still work."
    )

uploaded_file = st.file_uploader(
    "Upload a Certificate of Analysis (PDF or image)",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    help=(
        "Supports text-based PDFs, scanned/image-based PDFs, and image files "
        "(PNG/JPG/TIFF/BMP/WEBP) up to 50 MB."
    ),
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_size_mb = len(pdf_bytes) / (1024 * 1024)
    file_signature = (uploaded_file.name, len(pdf_bytes))
    template_bytes = user_template.getvalue() if user_template else None
    template_signature = (
        (user_template.name, len(template_bytes))
        if user_template and template_bytes is not None
        else None
    )

    if st.session_state.get("last_template_signature") != template_signature:
        st.session_state["last_template_signature"] = template_signature
        st.session_state.pop("template_hints", None)
        st.session_state.pop("translation_result", None)
        st.session_state.pop("doc_bytes", None)

    template_hints = None
    if template_bytes:
        if "template_hints" not in st.session_state:
            st.session_state["template_hints"] = extract_template_hints(template_bytes)
        template_hints = st.session_state["template_hints"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("File", uploaded_file.name)
    with col2:
        st.metric("Size", f"{file_size_mb:.2f} MB")

    # ------------------------------------------------------------------
    # Step 2: Extract text
    # ------------------------------------------------------------------
    st.subheader("2. Extract Text")

    if (
        "extraction_result" not in st.session_state
        or st.session_state.get("last_file_signature") != file_signature
    ):
        with st.spinner("Extracting text from file..."):
            extraction = extract_text_from_upload(
                pdf_bytes,
                filename=uploaded_file.name,
            )
            st.session_state["extraction_result"] = extraction
            st.session_state["last_file_signature"] = file_signature
            # Clear stale downstream state
            st.session_state.pop("translation_result", None)
            st.session_state.pop("doc_bytes", None)
    else:
        extraction = st.session_state["extraction_result"]

    if extraction["success"]:
        st.success(
            f"Text extracted using **{extraction['method']}** "
            f"({extraction['page_count']} page(s), "
            f"{len(extraction['text']):,} characters)"
        )

        with st.expander("Preview extracted text", expanded=False):
            st.text_area(
                "Extracted text (full)",
                extraction["text"],
                height=420,
                disabled=True,
            )

        if template_hints:
            placeholders_count = len(template_hints.get("placeholders", []))
            headings_count = len(template_hints.get("headings", []))
            st.caption(
                "Template detected: "
                f"{placeholders_count} placeholder(s), "
                f"{headings_count} heading hint(s)."
            )

        # ------------------------------------------------------------------
        # Step 3: Translate
        # ------------------------------------------------------------------
        st.subheader("3. Translate to Russian")

        if not api_key:
            st.warning(
                "Please enter your OpenAI API key in the sidebar to proceed."
            )
        elif not selected_model_valid:
            st.warning("Please provide a custom model ID.")
        else:
            translate_btn = st.button(
                "Translate to Russian",
                type="primary",
                use_container_width=True,
            )

            if translate_btn or st.session_state.get("translation_result"):
                if translate_btn:
                    progress_bar = st.progress(0, text="Translating...")
                    status_text = st.empty()

                    def update_progress(current, total):
                        pct = min(int(current / total * 100), 100)
                        progress_bar.progress(
                            pct,
                            text=f"Translating step {current}/{total}...",
                        )
                        status_text.text(
                            f"Processing step {current} of {total}"
                        )

                    with st.spinner("Translating document (structured)..."):
                        result = _run_translation_structured(
                            text=extraction["text"],
                            api_key=api_key,
                            model=selected_model,
                            progress_callback=update_progress,
                            template_hints=template_hints,
                            table_supplement=extraction.get("table_supplement", ""),
                        )

                    progress_bar.empty()
                    status_text.empty()

                    st.session_state["translation_result"] = result
                    # Clear stale doc
                    st.session_state.pop("doc_bytes", None)
                else:
                    result = st.session_state["translation_result"]

                if result["success"]:
                    st.success(
                        f"Translation complete! Model: **{result['model_used']}**"
                    )

                    with st.expander(
                        "Preview translated text", expanded=False
                    ):
                        st.text_area(
                            "Переведенный текст (полный)",
                            result["translated_text"],
                            height=420,
                            disabled=True,
                        )

                    st.subheader("3.5 Bilingual Review")
                    left_col, right_col = st.columns(2)
                    with left_col:
                        st.text_area(
                            "English source (extracted)",
                            extraction["text"],
                            height=320,
                            disabled=True,
                        )
                    with right_col:
                        st.text_area(
                            "Russian translation",
                            result["translated_text"],
                            height=320,
                            disabled=True,
                        )

                    with st.expander("Line-by-line bilingual diff (preview)"):
                        max_lines = 250
                        en_lines = extraction["text"].splitlines()[:max_lines]
                        ru_lines = result["translated_text"].splitlines()[:max_lines]
                        diff_html = difflib.HtmlDiff(
                            wrapcolumn=70
                        ).make_table(
                            en_lines,
                            ru_lines,
                            fromdesc="English Source",
                            todesc="Russian Translation",
                            context=False,
                            numlines=0,
                        )
                        st.caption(
                            "Showing first 250 lines for performance."
                        )
                        st.markdown(
                            "<style>"
                            ".diff {font-size: 12px; width: 100%;}"
                            ".diff_header {background: #f1f5f9;}"
                            ".diff_add {background: #e8f5e9;}"
                            ".diff_sub {background: #ffebee;}"
                            "</style>"
                            + diff_html,
                            unsafe_allow_html=True,
                        )

                    # ------------------------------------------------------
                    # Step 4: Generate & Download Word doc
                    # ------------------------------------------------------
                    st.subheader("4. Download Word Document")

                    if "doc_bytes" not in st.session_state or translate_btn:
                        with st.spinner("Generating Word document..."):
                            doc_bytes = _run_generate_structured_doc(
                                sections=result.get("sections", {}),
                                original_filename=uploaded_file.name,
                                extraction_method=extraction["method"],
                                model_used=result["model_used"],
                                user_template_bytes=template_bytes,
                                template_fields=result.get("template_fields", {}),
                                template_heading_map=result.get(
                                    "template_heading_map",
                                    {},
                                ),
                            )
                            st.session_state["doc_bytes"] = doc_bytes
                    else:
                        doc_bytes = st.session_state["doc_bytes"]

                    base_name = uploaded_file.name.rsplit(".", 1)[0]
                    output_filename = f"{base_name}_RU.docx"

                    st.download_button(
                        label="Download Translated COA (.docx)",
                        data=doc_bytes,
                        file_name=output_filename,
                        mime=(
                            "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document"
                        ),
                        type="primary",
                        use_container_width=True,
                    )

                    st.info(
                        "The document follows a fixed COA structure with "
                        "predefined sections. We recommend having a "
                        "pharmaceutical specialist review the translation."
                    )

                else:
                    st.error(f"Translation failed: {result['error']}")

                    if "api_key" in str(result["error"]).lower() or "auth" in str(
                        result["error"]
                    ).lower():
                        st.warning(
                            "This looks like an authentication error. "
                            "Please check your OpenAI API key."
                        )

    else:
        st.error(
            "Could not extract text from the uploaded file. "
            "The file may be corrupted, image quality may be too low, "
            "or OCR may be unavailable."
        )
        st.info(
            "Tips:\n"
            "- Ensure the PDF is not password-protected\n"
            "- Try 300 DPI+ scans with strong contrast\n"
            "- Upload as PNG/JPG if scanner exports problematic PDFs\n"
            "- Ensure pytesseract + Tesseract are installed on the server"
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
    "COA Translator v2.0 | Fixed-structure output | "
    "Pharmaceutical glossary with 200+ terms | "
    "Powered by OpenAI"
    "</div>",
    unsafe_allow_html=True,
)
