# COA Translation Platform (Python API + TypeScript UI)

This project now runs as a modern web app stack:

- `backend/` FastAPI API (Python) reusing existing extraction/translation/doc generation logic
- `frontend/` React + TypeScript + Tailwind UI (Linear-style dark interface)
- `api/index.py` Vercel Python entrypoint

Core logic remains unchanged and still uses:

- Multi-method PDF/image extraction with OCR and optional Camelot/Tabula table recovery
- OpenAI structured pharmaceutical translation with glossary enforcement
- Optional user glossary upload merged into translation prompts
- Template-aware `.docx` output generation

## Architecture

```text
frontend (React + TS + Tailwind)
  -> /api/extract
  -> /api/translate
  -> /api/generate-doc
backend (FastAPI)
  -> modules/pdf_extractor.py
  -> modules/translator.py
  -> modules/doc_generator.py
```

## API Endpoints

- `GET /api/health`
- `GET /api/capabilities`
- `POST /api/extract` (multipart: `file`, optional `template`)
- `POST /api/translate` (JSON)
- `POST /api/generate-doc` (JSON, streams `.docx`)
- `POST /api/process` (single-call pipeline, multipart)

## Local Development

### 1) Python backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python backend/run.py
```

Backend runs on `http://localhost:8000`.

`requirements.txt` is the Vercel-safe profile (smaller dependency set).

If you need full local OCR/table stack (Tesseract + Camelot/Tabula), install:

```bash
pip install -r requirements.full.txt
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` and proxies `/api/*` to `localhost:8000`.

## Deploying on Vercel

This repo includes `vercel.json` configured for:

- Python serverless API from `api/index.py`
- Static frontend build from `frontend/`

### Environment Variables

- `CORS_ALLOW_ORIGINS` (optional, comma-separated). Default is `*`.
- Frontend optional: `VITE_API_BASE_URL` (if API is hosted elsewhere).

### Important runtime note

OCR + PDF processing dependencies (PyMuPDF, OCR stack, Camelot/Tabula) can be heavy for serverless limits. If Vercel function size or execution time becomes a blocker, deploy `backend/` on a dedicated Python host (Railway/Render/Fly.io) and keep `frontend/` on Vercel.

This repo already uses a reduced dependency profile on Vercel to fit Lambda limits.  
When local OCR is unavailable in runtime, the backend can fallback to OpenAI vision OCR during extraction if an API key is provided.

## OCR / Extraction Notes

For scanned documents, server environment must have Tesseract installed.

`packages.txt` contains apt dependencies used previously in Streamlit deployments:

- `tesseract-ocr`
- `tesseract-ocr-eng`
- `ghostscript`
- `default-jre-headless`

## Project Structure

```text
.
├── api/
│   └── index.py                  # Vercel Python entrypoint
├── backend/
│   ├── __init__.py
│   ├── main.py                   # FastAPI routes
│   └── run.py                    # Local dev runner
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # New UI workflow
│   │   ├── index.css             # Tailwind + custom design system
│   │   └── main.tsx
│   ├── package.json
│   ├── tailwind.config.ts
│   └── vite.config.ts
├── modules/                      # Existing core logic (unchanged)
│   ├── pdf_extractor.py
│   ├── translator.py
│   ├── doc_generator.py
│   └── ...
├── templates/
├── requirements.txt
├── requirements.full.txt
└── vercel.json
```
