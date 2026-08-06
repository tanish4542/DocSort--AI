"""
FastAPI service for domain prediction (text or uploaded PDF/DOCX/TXT).
Run: uvicorn api.main:app --app-dir .. --reload
"""
from __future__ import annotations

import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdcs.predictor import DomainClassifier  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clf = DomainClassifier.load()
    yield


app = FastAPI(title="Multi-Domain Document Classifier", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_text")
def predict_text(body: TextIn):
    r = app.state.clf.predict_domain(body.text)
    return {
        "predicted_domain": r.predicted_domain,
        "confidence_percent": r.confidence,
        "probabilities_percent": r.probabilities,
        "top_tfidf_terms": r.top_tfidf_terms,
        "matched_manual_keywords": r.matched_manual_keywords,
        "keyword_domain_scores_percent": r.keyword_domain_scores,
    }


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload").suffix.lower() or ".txt"
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        r = app.state.clf.predict_document(path)
        return {
            "filename": file.filename,
            "predicted_domain": r.predicted_domain,
            "confidence_percent": r.confidence,
            "probabilities_percent": r.probabilities,
            "top_tfidf_terms": r.top_tfidf_terms,
            "matched_manual_keywords": r.matched_manual_keywords,
            "keyword_domain_scores_percent": r.keyword_domain_scores,
        }
    finally:
        Path(path).unlink(missing_ok=True)
