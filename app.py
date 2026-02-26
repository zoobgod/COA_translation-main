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
    page_icon="🧪",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-deep: #020203;
        --bg-base: #050506;
        --bg-elevated: #0a0a0c;
        --surface: rgba(255, 255, 255, 0.05);
        --surface-hover: rgba(255, 255, 255, 0.08);
        --fg: #EDEDEF;
        --fg-muted: #8A8F98;
        --fg-subtle: rgba(255, 255, 255, 0.6);
        --accent: #5E6AD2;
        --accent-bright: #6872D9;
        --accent-glow: rgba(94, 106, 210, 0.30);
        --border-default: rgba(255, 255, 255, 0.06);
        --border-hover: rgba(255, 255, 255, 0.10);
        --border-accent: rgba(94, 106, 210, 0.30);
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: "Inter", "Geist Sans", system-ui, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at top, #0a0a0f 0%, #050506 50%, #020203 100%);
        color: var(--fg);
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px);
        background-size: 64px 64px;
        opacity: 0.45;
        pointer-events: none;
        z-index: 0;
    }

    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140' viewBox='0 0 140 140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='1.2' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
        opacity: 0.015;
        pointer-events: none;
        mix-blend-mode: screen;
        z-index: 0;
    }

    .linear-ambient {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
    }

    .linear-ambient .blob {
        position: absolute;
        border-radius: 999px;
        filter: blur(120px);
        animation: float 10s ease-in-out infinite;
        opacity: 0.18;
    }

    .linear-ambient .blob-primary {
        width: 900px;
        height: 1400px;
        top: -450px;
        left: 50%;
        transform: translateX(-50%);
        background: radial-gradient(circle, rgba(94, 106, 210, 0.8) 0%, rgba(94, 106, 210, 0) 68%);
    }

    .linear-ambient .blob-left {
        width: 650px;
        height: 900px;
        left: -260px;
        top: 120px;
        background: radial-gradient(circle, rgba(146, 99, 237, 0.55) 0%, rgba(146, 99, 237, 0) 70%);
        animation-duration: 12s;
    }

    .linear-ambient .blob-right {
        width: 560px;
        height: 780px;
        right: -180px;
        top: 260px;
        background: radial-gradient(circle, rgba(94, 106, 210, 0.60) 0%, rgba(94, 106, 210, 0) 72%);
        animation-duration: 11s;
    }

    .linear-ambient .blob-bottom {
        width: 720px;
        height: 420px;
        left: 30%;
        bottom: -160px;
        background: radial-gradient(circle, rgba(94, 106, 210, 0.38) 0%, rgba(94, 106, 210, 0) 74%);
        animation: pulseGlow 8s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }

    @keyframes pulseGlow {
        0%, 100% { opacity: 0.12; transform: translateY(0px); }
        50% { opacity: 0.2; transform: translateY(-14px); }
    }

    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stSidebar"] {
        position: relative;
        z-index: 1;
    }

    .block-container {
        max-width: 1160px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero-shell {
        border: 1px solid var(--border-default);
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        box-shadow:
            0 0 0 1px rgba(255, 255, 255, 0.05),
            0 20px 60px rgba(0, 0, 0, 0.45),
            0 0 90px rgba(94, 106, 210, 0.10);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(8px);
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: 999px;
        border: 1px solid var(--border-accent);
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #c9d0ff;
        background: rgba(94, 106, 210, 0.12);
        margin-bottom: 0.8rem;
    }

    .hero-title {
        margin: 0;
        font-size: clamp(2rem, 3vw, 3.4rem);
        line-height: 1.05;
        letter-spacing: -0.03em;
        font-weight: 650;
        background: linear-gradient(to bottom, #ffffff, rgba(255, 255, 255, 0.72));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-title .accent {
        background: linear-gradient(90deg, #5E6AD2 0%, #8C96EC 45%, #5E6AD2 100%);
        background-size: 200% auto;
        animation: shimmer 5s linear infinite;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    .hero-subtitle {
        margin-top: 0.55rem;
        margin-bottom: 0;
        color: var(--fg-muted);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 74ch;
    }

    .step-head {
        margin-top: 1.1rem;
        margin-bottom: 0.55rem;
    }

    .step-pill {
        display: inline-flex;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        border: 1px solid var(--border-default);
        background: rgba(255, 255, 255, 0.03);
        color: var(--fg-subtle);
        font-size: 0.74rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .step-title {
        margin-top: 0.55rem;
        margin-bottom: 0;
        color: var(--fg);
        font-size: clamp(1.18rem, 2vw, 1.7rem);
        line-height: 1.24;
        letter-spacing: -0.02em;
    }

    .step-subtitle {
        margin-top: 0.25rem;
        margin-bottom: 0;
        color: var(--fg-muted);
        font-size: 0.94rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 10, 12, 0.96), rgba(5, 5, 6, 0.95));
        border-right: 1px solid var(--border-default);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: var(--fg);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed var(--border-accent);
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
        transition: border-color 220ms ease, background 220ms ease, box-shadow 220ms ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(94, 106, 210, 0.55);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.03));
        box-shadow:
            0 0 0 1px rgba(255, 255, 255, 0.06),
            0 8px 40px rgba(0, 0, 0, 0.45),
            0 0 60px rgba(94, 106, 210, 0.16);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        background: rgba(15, 15, 18, 0.96) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        color: var(--fg) !important;
        border-radius: 10px !important;
        transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(94, 106, 210, 0.28) !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.46) !important;
    }

    [data-testid="stMetric"] {
        border-radius: 16px;
        border: 1px solid var(--border-default);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
        padding: 0.9rem 1rem;
        box-shadow:
            0 0 0 1px rgba(255, 255, 255, 0.05),
            0 10px 30px rgba(0, 0, 0, 0.35);
    }

    [data-testid="stMetric"] label p,
    [data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: var(--fg-muted);
        letter-spacing: 0.03em;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--fg);
        font-weight: 560;
    }

    .stButton > button,
    .stDownloadButton > button {
        border: 1px solid rgba(94, 106, 210, 0.45) !important;
        border-radius: 10px !important;
        background: linear-gradient(180deg, #6872D9, #5E6AD2) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        box-shadow:
            0 0 0 1px rgba(94, 106, 210, 0.50),
            0 8px 24px rgba(94, 106, 210, 0.33),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.28);
        transition: transform 220ms cubic-bezier(0.16, 1, 0.3, 1), box-shadow 220ms ease, filter 220ms ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-3px);
        filter: brightness(1.03);
        box-shadow:
            0 0 0 1px rgba(104, 114, 217, 0.58),
            0 14px 42px rgba(94, 106, 210, 0.38),
            0 0 80px rgba(94, 106, 210, 0.16),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.34);
    }

    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: scale(0.98);
        box-shadow:
            0 0 0 1px rgba(94, 106, 210, 0.35),
            0 4px 14px rgba(94, 106, 210, 0.24),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.20);
    }

    .stButton > button:focus,
    .stDownloadButton > button:focus {
        outline: none !important;
        box-shadow:
            0 0 0 2px rgba(94, 106, 210, 0.36),
            0 0 0 6px rgba(5, 5, 6, 0.92),
            0 8px 24px rgba(94, 106, 210, 0.25) !important;
    }

    .stExpander,
    [data-testid="stExpander"] {
        border-radius: 14px !important;
        border: 1px solid var(--border-default) !important;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02)) !important;
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.30);
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--border-default);
        background: rgba(255, 255, 255, 0.04);
        color: var(--fg);
    }

    [data-testid="stProgressBar"] > div > div > div > div {
        background: linear-gradient(90deg, #5E6AD2, #8C96EC) !important;
    }

    div[data-testid="stHorizontalBlock"] > div {
        gap: 1rem !important;
    }

    .block-divider {
        margin: 1.5rem 0;
        height: 1px;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.0), rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.0));
    }

    .diff {
        font-size: 12px;
        width: 100%;
        border-collapse: collapse;
        background: rgba(10, 10, 12, 0.88);
        color: #d9dce4;
    }

    .diff_header {
        background: rgba(94, 106, 210, 0.26);
        color: #eef1ff;
    }

    .diff_next {
        background: rgba(255, 255, 255, 0.06);
    }

    .diff_add {
        background: rgba(73, 187, 133, 0.18);
    }

    .diff_chg {
        background: rgba(245, 171, 70, 0.20);
    }

    .diff_sub {
        background: rgba(227, 95, 111, 0.22);
    }

    .footer-note {
        text-align: center;
        color: var(--fg-muted);
        font-size: 0.84rem;
        padding-top: 0.4rem;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 1.1rem;
        }

        .hero-shell {
            padding: 1.1rem;
        }

        .hero-title {
            font-size: clamp(1.65rem, 9vw, 2.5rem);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .linear-ambient .blob,
        .hero-title .accent,
        .stButton > button,
        .stDownloadButton > button {
            animation: none !important;
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="linear-ambient" aria-hidden="true">
        <div class="blob blob-primary"></div>
        <div class="blob blob-left"></div>
        <div class="blob blob-right"></div>
        <div class="blob blob-bottom"></div>
    </div>
    <section class="hero-shell">
        <div class="hero-kicker">Pharmacopeia Workflow</div>
        <h1 class="hero-title">
            Pharmaceutical <span class="accent">COA Translator</span>
        </h1>
        <p class="hero-subtitle">
            Extract, review, and convert English Certificates of Analysis to
            polished Russian output while preserving technical structure.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


def render_step_header(step: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="step-head">
            <span class="step-pill">{step}</span>
            <h2 class="step-title">{title}</h2>
            <p class="step-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="hero-kicker">Workspace Settings</div>
        <p class="step-subtitle" style="margin-bottom: 0.8rem;">
            Configure model access and optional output template behavior.
        </p>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown("**Word Template (optional)**")
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
render_step_header(
    "Step 1",
    "Upload COA File",
    "Import a certificate in PDF or image format for extraction.",
)

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
    st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
    render_step_header(
        "Step 2",
        "Extract Full Source Text",
        "Run multi-method extraction and OCR to recover complete COA content.",
    )

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
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        render_step_header(
            "Step 3",
            "Translate to Russian",
            "Generate a high-fidelity technical translation for professional review.",
        )

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

                    st.markdown(
                        '<div class="block-divider"></div>',
                        unsafe_allow_html=True,
                    )
                    render_step_header(
                        "Step 3.5",
                        "Bilingual Review",
                        "Compare English extraction and Russian translation before export.",
                    )
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
                            diff_html,
                            unsafe_allow_html=True,
                        )

                    # ------------------------------------------------------
                    # Step 4: Generate & Download Word doc
                    # ------------------------------------------------------
                    st.markdown(
                        '<div class="block-divider"></div>',
                        unsafe_allow_html=True,
                    )
                    render_step_header(
                        "Step 4",
                        "Download Word Document",
                        "Export clean translated COA content in .docx format.",
                    )

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
st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
st.markdown(
    "<div class='footer-note'>"
    "COA Translator | EN → RU workflow with OCR, table support, and structured DOCX export"
    "</div>",
    unsafe_allow_html=True,
)
