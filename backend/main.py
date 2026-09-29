"""
backend/main.py

FastAPI backend service for DocSort AI.
Integrated with the Production Combined TF-IDF + Structural Features + LinearSVC model.

Supported endpoints:
- GET /model-info : Metadata confirming active production model
- GET /categories : List of 6 taxonomy categories + Anonymous
- POST /predict : Single document upload & classification (Automatic Sorting)
- POST /predict-bulk : Batch document upload & classification (Automatic Sorting)
- POST /confirm-sort : Preservation of API compatibility for legacy requests
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from docx import Document
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader

# Add current backend folder to sys.path so research_inference is discoverable
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from research_inference import (
        CLASSES,
        PROVISIONAL_MARGIN_THRESHOLD,
        get_model_metadata,
        predict_text,
    )
except ImportError:
    from .research_inference import (
        CLASSES,
        PROVISIONAL_MARGIN_THRESHOLD,
        get_model_metadata,
        predict_text,
    )

# ---------------------------------------------------
# FASTAPI SETUP
# ---------------------------------------------------

app = FastAPI(
    title="DocSort AI — Document Classification API",
    description="Multi-domain document classification using the Production Combined TF-IDF + Structural Features + LinearSVC model.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# CATEGORIES & LOCAL STORAGE DIRECTORIES
# ---------------------------------------------------

CATEGORIES: List[str] = [
    "Technology & Computing",
    "Science & Academics",
    "Medical Health",
    "Business & Finance",
    "Entertainment",
    "Sports",
]
TAXONOMY_CLASSES: List[str] = CATEGORIES
ANONYMOUS_CATEGORY = "Anonymous"

BASE_DIR = os.path.expanduser("~/Desktop/SortedDocuments")
BASE_SORTED_DIR = BASE_DIR

# Ensure target category directories exist
for category in CATEGORIES:
    os.makedirs(os.path.join(BASE_DIR, category), exist_ok=True)

# Ensure Anonymous directory exists
ANONYMOUS_DIR = os.path.join(BASE_DIR, ANONYMOUS_CATEGORY)
os.makedirs(ANONYMOUS_DIR, exist_ok=True)

# Temporary staging folder during extraction
STAGING_DIR = os.path.join(BASE_DIR, "_staging")
PENDING_DIR = STAGING_DIR
os.makedirs(STAGING_DIR, exist_ok=True)

PENDING_FILES: Dict[str, dict] = {}


class ConfirmSortBody(BaseModel):
    pending_id: str = Field(..., min_length=4)
    chosen_domain: str = Field(..., min_length=1)


# ---------------------------------------------------
# DOCUMENT TEXT EXTRACTION
# ---------------------------------------------------

def extract_pdf(file_path: str) -> str:
    """Extract readable text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text


def extract_docx(file_path: str) -> str:
    """Extract readable text from a Word DOCX file."""
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text:
                text += para.text + " "
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text


def extract_txt(file_path: str) -> str:
    """Extract text from a plain TXT file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Dispatches to the correct extractor based on file extension."""
    low = filename.lower()
    if low.endswith(".pdf"):
        return extract_pdf(file_path)
    elif low.endswith(".docx"):
        return extract_docx(file_path)
    elif low.endswith(".txt"):
        return extract_txt(file_path)
    return ""


# ---------------------------------------------------
# DOCUMENT VALIDATION
# ---------------------------------------------------

BLANK_DOCUMENT_ERROR_MESSAGE = (
    "This document appears to be blank or contains no readable text. "
    "Please upload a document with readable content."
)


def validate_document_content(raw_text: str | None) -> tuple[bool, str]:
    """Validates extracted document text before sending to ML model."""
    if not raw_text:
        return False, BLANK_DOCUMENT_ERROR_MESSAGE

    cleaned = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f\ufeff\u00ad]",
        "",
        raw_text,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return False, BLANK_DOCUMENT_ERROR_MESSAGE

    alnum_chars = len(re.findall(r"[a-zA-Z0-9]", cleaned))
    words = [w for w in cleaned.split() if re.search(r"[a-zA-Z0-9]", w)]

    if alnum_chars < 3 or len(words) == 0:
        return False, BLANK_DOCUMENT_ERROR_MESSAGE

    return True, cleaned


# ---------------------------------------------------
# SHARED INFERENCE & AUTOMATIC SORTING PIPELINE
# ---------------------------------------------------

async def process_uploaded_file(file: UploadFile) -> dict:
    client_name = os.path.basename(file.filename or "upload")
    low = client_name.lower()

    if not (low.endswith(".pdf") or low.endswith(".docx") or low.endswith(".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF, DOCX, or TXT file."
        )

    timestamp = int(time.time() * 1000)
    temp_filename = f"{timestamp}_{client_name}"
    temp_path = os.path.join(STAGING_DIR, temp_filename)

    try:
        # 1. Save uploaded file to staging
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract raw text
        raw_text = extract_text_from_file(temp_path, client_name)

        # 3. Validate content
        is_valid, validation_result = validate_document_content(raw_text)
        if not is_valid:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise HTTPException(
                status_code=400,
                detail=BLANK_DOCUMENT_ERROR_MESSAGE
            )

        # 4. Predict with Combined TF-IDF + Structural Features + LinearSVC model
        pred_res = predict_text(validation_result)

        predicted_class = pred_res["predicted_class"]
        decision_margin = pred_res["decision_margin"]
        is_anonymous = pred_res["is_anonymous"]
        final_destination = pred_res["final_destination"]
        model_name = pred_res["model"]
        decision_scores = pred_res["decision_scores"]
        top_keywords = pred_res["top_keywords"]
        top_features = pred_res["top_features"]
        text_length = pred_res["text_length"]

        # 5. Automatic Folder Sorting
        if is_anonymous or final_destination == ANONYMOUS_CATEGORY:
            dest_dir = ANONYMOUS_DIR
        else:
            dest_dir = os.path.join(BASE_DIR, final_destination)

        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, temp_filename)
        shutil.move(temp_path, dest_path)

        print(
            f"[Automatic Sort] {client_name} -> Predicted: {predicted_class} | "
            f"Margin: {decision_margin:.4f} | Final Destination: {final_destination} -> {dest_path}"
        )

        return {
            "filename": client_name,
            "prediction": final_destination,
            "predicted_class": predicted_class,
            "decision_margin": decision_margin,
            "is_anonymous": is_anonymous,
            "final_destination": final_destination,
            "model": model_name,
            "status": "Automatically Placed in Anonymous" if is_anonymous else "Automatically Sorted",
            "decision_scores": decision_scores,
            "top_keywords": top_keywords,
            "top_features": top_features,
            "stored_in": dest_path,
            "text_length": text_length,
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------

@app.get("/model-info")
def model_info():
    """Returns metadata confirming active production model configuration."""
    return get_model_metadata()


@app.get("/categories")
def get_categories():
    """Returns the 6 final taxonomy categories plus Anonymous."""
    return {
        "categories": CATEGORIES,
        "anonymous": ANONYMOUS_CATEGORY,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Single file classification endpoint with automatic sorting."""
    return await process_uploaded_file(file)


@app.post("/predict-bulk")
async def predict_bulk(files: list[UploadFile] = File(...)):
    """Bulk file classification endpoint with automatic sorting."""
    results = []
    for file in files:
        res = await process_uploaded_file(file)
        results.append(res)
    return {"results": results}


@app.post("/confirm-sort")
async def confirm_sort(body: ConfirmSortBody):
    """Preserves API compatibility for legacy requests."""
    return {
        "message": "Automatic sorting is active. Manual confirmation is no longer required.",
        "pending_id": body.pending_id,
        "chosen_domain": body.chosen_domain,
    }