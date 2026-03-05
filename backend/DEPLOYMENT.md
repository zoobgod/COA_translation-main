# Backend Deployment Profiles

## 1) Vercel-safe (lightweight)
- Uses `requirements.txt`
- Relies on digital extraction + Vision OCR fallback (`/api/extract` with API key)
- Best for serverless constraints

## 2) Full OCR quality (recommended for production parsing)
- Uses `requirements.full.txt`
- Requires system packages: Tesseract, Ghostscript, Java runtime
- Supports stronger local OCR/table extraction behavior

### Docker run

```bash
docker build -f Dockerfile.ocr-backend -t coa-ocr-backend .
docker run --rm -p 8000:8000 coa-ocr-backend
```

Set frontend `VITE_API_BASE_URL` to your backend URL (e.g. Render/Railway/Fly).
