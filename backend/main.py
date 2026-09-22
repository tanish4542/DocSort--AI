"""
backend/main.py

FastAPI backend service for DocSort AI.
Integrated with the FINAL Research LinearSVC (C=0.5) model and 100,000-feature TF-IDF vectorizer.

Supported endpoints:
- GET /model-info : Metadata confirming active research model
- POST /predict : Single document upload & classification
- POST /predict-bulk : Batch document upload & classification
- POST /confirm-sort : Confirmation and folder routing for ambiguous uploads
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
        get_model_metadata,
        predict_text,
    )
except ImportError:
    from .research_inference import (
        CLASSES,
        get_model_metadata,
        predict_text,
    )

# ---------------------------------------------------
# FASTAPI SETUP
# ---------------------------------------------------

app = FastAPI(
    title="DocSort AI — Research Document Classification API",
    description="Multi-domain document classification using the finalized Research LinearSVC model.",
    version="2.0.0",
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

CATEGORIES: List[str] = list(CLASSES)

BASE_DIR = os.path.expanduser("~/Desktop/SortedDocuments")

# Ensure target category directories exist
for category in CATEGORIES:
    os.makedirs(os.path.join(BASE_DIR, category), exist_ok=True)

# Ensure staging directory for ambiguous predictions exists
PENDING_DIR = os.path.join(BASE_DIR, "_pending")
os.makedirs(PENDING_DIR, exist_ok=True)

# Staged files in-memory registry: pending_id -> metadata
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
# DOCUMENT VALIDATION BEFORE ML INFERENCE
# ---------------------------------------------------

BLANK_DOCUMENT_ERROR_MESSAGE = (
    "This document appears to be blank or contains no readable text. "
    "Please upload a document with readable content."
)


def validate_document_content(raw_text: str | None) -> tuple[bool, str]:
    r"""
    Validates extracted document text before sending to ML model.
    Steps:
    1. Check if raw text is None or empty.
    2. Remove invisible, control, and zero-width characters:
       - Control chars: [\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]
       - Zero-width & directional marks: [\u200b-\u200f\u2028-\u202f\ufeff\u00ad]
    3. Normalize whitespace (collapse \s+ to single space, strip).
    4. Check for meaningful content:
       Requires at least one word token with alphanumeric content,
       and total alphanumeric characters >= 3.
       This rejects blank documents, whitespace-only, and punctuation/control noise,
       while accepting legitimate short documents (e.g. 'Total: $15.00 Tax Paid', 'Biology Notes').
    Returns:
        (is_valid, cleaned_text_or_error_message)
    """
    if not raw_text:
        return False, BLANK_DOCUMENT_ERROR_MESSAGE

    # 1. Remove control, invisible, and zero-width characters
    cleaned = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f\ufeff\u00ad]",
        "",
        raw_text,
    )

    # 2. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return False, BLANK_DOCUMENT_ERROR_MESSAGE

    # 3. Meaningful content check
    alnum_chars = len(re.findall(r"[a-zA-Z0-9]", cleaned))
    words = [w for w in cleaned.split() if re.search(r"[a-zA-Z0-9]", w)]

    if alnum_chars < 3 or len(words) == 0:
        return False, BLANK_DOCUMENT_ERROR_MESSAGE

    return True, cleaned


# ---------------------------------------------------
# SHARED FILE INFERENCE PIPELINE
# ---------------------------------------------------

async def process_uploaded_file(file: UploadFile) -> dict:
    client_name = os.path.basename(file.filename or "upload")
    low = client_name.lower()

    if not (low.endswith(".pdf") or low.endswith(".docx") or low.endswith(".txt")):
        return {
            "filename": client_name,
            "error": "Unsupported file type. Please upload a PDF, DOCX, or TXT file.",
        }

    timestamp = int(time.time() * 1000)
    temp_filename = f"{timestamp}_{client_name}"
    temp_path = os.path.join(PENDING_DIR, temp_filename)

    try:
        # 1. Save uploaded file to temp path
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract raw text
        raw_text = extract_text_from_file(temp_path, client_name)

        # 3. Robust document-content validation before ML inference
        is_valid, validation_result = validate_document_content(raw_text)
        if not is_valid:
            # Reject immediately: do NOT call research model, do NOT auto-sort
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            print(f"[Blank/Empty Rejected] {client_name}: Document contains no readable text.")
            return {
                "filename": client_name,
                "error": BLANK_DOCUMENT_ERROR_MESSAGE,
                "is_blank": True,
                "prediction": None,
                "predicted_class": None,
                "confidence": None,
                "decision_margin": None,
                "uncertainty_level": "Blank document",
                "decision_scores": {},
                "top_keywords": [],
                "top_two_domains": [],
                "requires_manual_choice": False,
                "pending_id": None,
                "stored_in": None,
                "text_length": 0,
            }

        # 4. Predict with finalized Research LinearSVC model
        pred_res = predict_text(validation_result)

        prediction = pred_res["predicted_class"]
        decision_margin = pred_res["decision_margin"]
        uncertainty_level = pred_res["uncertainty_level"]
        decision_scores = pred_res["decision_scores"]
        top_keywords = pred_res["top_keywords"]
        top_two = pred_res["top_two_domains"]
        text_length = pred_res["text_length"]

        # Operational uncertainty heuristic:
        # Ambiguous if decision margin < 0.5 -> require manual user confirmation
        requires_manual = (uncertainty_level == "Ambiguous")

        if requires_manual:
            pending_id = str(uuid.uuid4())
            pending_path = os.path.join(
                PENDING_DIR,
                f"{pending_id}__{temp_filename}"
            )
            shutil.move(temp_path, pending_path)

            allowed = tuple(d["domain"] for d in top_two) if len(top_two) >= 2 else (prediction, prediction)

            PENDING_FILES[pending_id] = {
                "path": pending_path,
                "allowed_domains": allowed,
                "final_basename": temp_filename,
                "client_filename": client_name,
            }

            print(
                f"[Ambiguous] Prediction: {prediction} | Margin: {decision_margin:.4f} "
                f"| Staging for review (pending_id={pending_id})"
            )

            return {
                "filename": client_name,
                "prediction": prediction,
                "predicted_class": prediction,
                "final_prediction": prediction,
                "selected_model": pred_res.get("selected_model", "LinearSVC"),
                "selection_reason": pred_res.get(
                    "selection_reason",
                    "LinearSVC achieved the highest test Macro F1 (94.49%) among the evaluated individual classifiers.",
                ),
                "model_comparison": pred_res.get("model_comparison", []),
                "model_performance": pred_res.get("model_performance", {}),
                "ensemble_experiments": pred_res.get("ensemble_experiments", {}),
                "decision_margin": decision_margin,
                "uncertainty_level": uncertainty_level,
                "decision_scores": decision_scores,
                "confidence": decision_margin,
                "top_keywords": top_keywords,
                "ranking": decision_scores,
                "top_two_domains": top_two,
                "requires_manual_choice": True,
                "pending_id": pending_id,
                "stored_in": None,
                "text_length": text_length,
            }

        # Auto-sort path: Moderate or High confidence
        dest_dir = os.path.join(BASE_DIR, prediction)
        os.makedirs(dest_dir, exist_ok=True)
        destination_path = os.path.join(dest_dir, temp_filename)

        shutil.move(temp_path, destination_path)

        print(
            f"[Auto-Sort] Prediction: {prediction} | Margin: {decision_margin:.4f} "
            f"({uncertainty_level}) -> {destination_path}"
        )

        return {
            "filename": client_name,
            "prediction": prediction,
            "predicted_class": prediction,
            "final_prediction": prediction,
            "selected_model": pred_res.get("selected_model", "LinearSVC"),
            "selection_reason": pred_res.get(
                "selection_reason",
                "LinearSVC achieved the highest test Macro F1 (94.49%) among the evaluated individual classifiers.",
            ),
            "model_comparison": pred_res.get("model_comparison", []),
            "model_performance": pred_res.get("model_performance", {}),
            "ensemble_experiments": pred_res.get("ensemble_experiments", {}),
            "decision_margin": decision_margin,
            "uncertainty_level": uncertainty_level,
            "decision_scores": decision_scores,
            "confidence": decision_margin,
            "top_keywords": top_keywords,
            "ranking": decision_scores,
            "top_two_domains": top_two,
            "requires_manual_choice": False,
            "pending_id": None,
            "stored_in": destination_path,
            "text_length": text_length,
        }

    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

        print(f"Inference error for {client_name}: {e}")
        return {
            "filename": client_name,
            "error": str(e),
        }


# ---------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------

@app.get("/model-info")
def model_info():
    """Returns metadata confirming active research model and configuration."""
    return get_model_metadata()


@app.get("/categories")
def get_categories():
    """Returns the exact 6 research categories exposed by the classifier."""
    return {"categories": CATEGORIES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Single file classification endpoint."""
    return await process_uploaded_file(file)


@app.post("/predict-bulk")
async def predict_bulk(files: list[UploadFile] = File(...)):
    """Bulk file classification endpoint."""
    results = []
    for file in files:
        res = await process_uploaded_file(file)
        results.append(res)
    return {"results": results}


@app.post("/confirm-sort")
async def confirm_sort(body: ConfirmSortBody):
    """Sorts a staged ambiguous file into the user's selected category."""
    entry = PENDING_FILES.get(body.pending_id)

    if not entry:
        return {"error": "Invalid or expired pending_id. Please re-upload the document."}

    chosen = body.chosen_domain.strip()

    if chosen not in CATEGORIES:
        return {"error": f"Invalid category '{chosen}'. Must be one of: {CATEGORIES}"}

    allowed = entry["allowed_domains"]
    if chosen not in allowed:
        return {"error": f"Chosen category must be one of candidate domains: {', '.join(allowed)}"}

    src = entry["path"]
    if not os.path.isfile(src):
        PENDING_FILES.pop(body.pending_id, None)
        return {"error": "Pending file missing on disk. Please upload again."}

    dest_dir = os.path.join(BASE_DIR, chosen)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, entry["final_basename"])

    shutil.move(src, dest_path)
    PENDING_FILES.pop(body.pending_id, None)

    print(f"[User Confirmed] Choice: {chosen} -> {dest_path}")

    return {
        "filename": entry["client_filename"],
        "prediction": chosen,
        "predicted_class": chosen,
        "stored_in": dest_path,
        "confidence": 100.0,
        "decision_margin": 10.0,
        "uncertainty_level": "User confirmed",
        "requires_manual_choice": False,
        "pending_id": None,
        "resolved_by_user": True,
    }