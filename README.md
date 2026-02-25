# COA Translation — Pharmaceutical Certificate of Analysis Translator

Streamlit application that translates pharmaceutical Certificate of Analysis (COA) documents from English to Russian using OpenAI, with a specialized pharmaceutical and medicinal glossary.

## Features

- **PDF Upload** — Upload COA documents in PDF format
- **Multi-method text extraction** — Uses pdfplumber (primary), PyMuPDF (fallback), and OCR via pytesseract (for scanned documents)
- **Optimized OCR pipeline** — Image preprocessing (grayscale, autocontrast, sharpening, binarization) with tuned Tesseract settings (PSM 6 for structured documents) and pdfplumber-based rendering fallback when PyMuPDF is unavailable
- **AI Translation with pharma glossary** — Translates via OpenAI models with a 200+ term pharmaceutical glossary enforcing standard Russian pharmaceutical terminology
- **Structured JSON translation** — OpenAI outputs a structured JSON response mapping content to predefined COA sections, ensuring consistent document layout
- **Fixed-structure Word output** — Every output document follows the same 10-section predefined structure regardless of the original PDF layout
- **Custom template support** — Optionally upload your own `.docx` template with Jinja2 placeholders for custom formatting
- **Download** — One-click download of the translated document

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key
- (Optional) Tesseract OCR installed on the system for scanned PDF support

### Installation

```bash
pip install -r requirements.txt
```

For OCR support (scanned PDFs), install Tesseract:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### Generate the Word template (optional)

A pre-prepared Word template can be generated for docxtpl-based rendering:

```bash
python -m modules.create_template
```

If the template is not provided by the user, the app generates documents using the built-in fixed structure.

## Running the App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

## Usage

1. Enter your OpenAI API key in the sidebar
2. Select a translation model (gpt-4o recommended for best quality)
3. (Optional) Upload a custom `.docx` structure template in the sidebar
4. Upload a COA PDF file
5. Review the extracted text
6. Click **Translate to Russian**
7. Download the translated `.docx` file

## Fixed COA Document Structure

Every output document contains these sections in order:

| # | Section Key | Russian Label |
|---|-------------|---------------|
| 1 | `document_title` | Наименование документа |
| 2 | `company_info` | Информация о компании |
| 3 | `product_name` | Наименование продукта |
| 4 | `product_details` | Сведения о продукте |
| 5 | `batch_info` | Информация о серии |
| 6 | `storage_conditions` | Условия хранения |
| 7 | `test_results` | Результаты испытаний |
| 8 | `conclusion` | Заключение |
| 9 | `signatures` | Подписи |
| 10 | `notes` | Примечания |

The `test_results` section is rendered as a formatted table; all others are text paragraphs.

## Custom Templates

Upload a `.docx` file containing Jinja2 placeholders (e.g. `{{ product_name }}`, `{{ test_results }}`). Available placeholders:

- All 10 section keys from the table above
- `original_filename` — source PDF filename
- `translation_date` — date of translation
- `model_used` — OpenAI model used
- `extraction_method` — PDF extraction method used

## Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit theme and server config
├── modules/
│   ├── __init__.py
│   ├── glossary.py           # Pharmaceutical EN→RU glossary (200+ terms)
│   ├── coa_structure.py      # Fixed COA section definitions
│   ├── pdf_extractor.py      # PDF text extraction (pdfplumber, PyMuPDF, OCR)
│   ├── translator.py         # OpenAI translation (plain + structured JSON)
│   ├── doc_generator.py      # Word document generation (fixed structure)
│   └── create_template.py    # Script to generate a sample docxtpl template
└── templates/
    └── coa_template.docx     # Sample template (created by create_template.py)
```

## OCR Pipeline

The OCR system is designed for scanned pharmaceutical COA documents:

1. **Page rendering** — PyMuPDF at 300 DPI (preferred), or pdfplumber `page.to_image()` as fallback
2. **Image preprocessing** — Grayscale → upscale (if small) → autocontrast → sharpen → binarize
3. **Tesseract OCR** — PSM 6 (single uniform block, good for forms/tables), OEM 3 (LSTM engine)
4. **Quality gate** — Pages with fewer than 10 alphanumeric characters are rejected
5. **Dual-pass** — If preprocessed OCR fails, a second pass without preprocessing is attempted

## Glossary

The pharmaceutical glossary (`modules/glossary.py`) includes 246 standard translations for:

- Document headers and metadata fields
- Physical and chemical test parameters
- Analytical methods (HPLC, GC, MS, etc.)
- Microbiological tests
- Dosage forms
- Units and measurements
- Regulatory references (USP, EP, BP, etc.)
- Common COA result phrases

The glossary terms are injected into the OpenAI system prompt to ensure consistent, accurate pharmaceutical translations.
