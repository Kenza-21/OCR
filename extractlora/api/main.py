"""FastAPI demo service: upload a document image, get back validated JSON.

Run with:
    uvicorn extractlora.api.main:app --reload
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from extractlora.extraction.config import DEFAULT_ADAPTER_DIR
from extractlora.extraction.pipeline import extract_invoice
from extractlora.ocr.engine import extract_text

app = FastAPI(title="ExtractLoRA", description="OCR -> LoRA -> JSON structure")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile):
    suffix = Path(file.filename or "document.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        ocr_text = extract_text(tmp_path)
        adapter_dir = DEFAULT_ADAPTER_DIR if Path(DEFAULT_ADAPTER_DIR).exists() else None
        data = extract_invoice(ocr_text, adapter_dir=adapter_dir)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"ocr_text": ocr_text, "extraction": data}
