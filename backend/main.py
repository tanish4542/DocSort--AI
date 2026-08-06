from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
sys.path.append("multi_domain_doc_classifier/src")

import re
import nltk
import os
import shutil
import time
import uuid

from mdcs.predictor import predict_document
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from docx import Document

# ---------------------------------------------------
# DOWNLOAD NLTK STOPWORDS
# ---------------------------------------------------

nltk.download('stopwords')

# ---------------------------------------------------
# FASTAPI SETUP
# ---------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# STOPWORDS
# ---------------------------------------------------

stop_words = set(stopwords.words("english"))

# ---------------------------------------------------
# CREATE CATEGORY FOLDERS
# ---------------------------------------------------

CATEGORIES = [
    "Medical",
    "Finance",
    "Technology",
    "Sports",
    "Education"
]

BASE_DIR = os.path.expanduser("~/Desktop/SortedDocuments")

for category in CATEGORIES:
    os.makedirs(os.path.join(BASE_DIR, category), exist_ok=True)

PENDING_DIR = os.path.join(BASE_DIR, "_pending")
os.makedirs(PENDING_DIR, exist_ok=True)

# pending_id -> metadata
PENDING_FILES: dict[str, dict] = {}


class ConfirmSortBody(BaseModel):
    pending_id: str = Field(..., min_length=4)
    chosen_domain: str = Field(..., min_length=1)

# ---------------------------------------------------
# CLEAN TEXT FUNCTION
# ---------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)

# ---------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------

def extract_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    return text

# ---------------------------------------------------
# DOCX TEXT EXTRACTION
# ---------------------------------------------------

def extract_docx(file_path):
    doc = Document(file_path)
    text = ""

    for para in doc.paragraphs:
        text += para.text + " "

    return text

# ---------------------------------------------------
# TXT EXTRACTION
# ---------------------------------------------------

def extract_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------
# SHARED FILE PROCESSOR
# ---------------------------------------------------

async def process_uploaded_file(file: UploadFile):

    client_name = os.path.basename(file.filename or "upload")
    low = client_name.lower()

    if not (
        low.endswith(".pdf")
        or low.endswith(".docx")
        or low.endswith(".txt")
    ):
        return {
            "filename": client_name,
            "error": "Unsupported file type"
        }

    timestamp = int(time.time() * 1000)
    temp_filename = f"{timestamp}_{client_name}"
    temp_path = temp_filename

    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Predict
        result = predict_document(temp_path)

        prediction = result["predicted_domain"]
        confidence = float(result["confidence_percent"])
        top_keywords = [x[0] for x in result["top_tfidf_terms"][:10]]
        ranking = result["probabilities_percent"]

        top_two = result.get("top_two_domains") or []

        if not top_two and isinstance(ranking, dict):
            sr = sorted(ranking.items(), key=lambda kv: -kv[1])
            top_two = [
                {"domain": d, "confidence": float(c)}
                for d, c in sr[:2]
            ]

        requires_manual = confidence < 100.0

        # Manual confirmation case
        if requires_manual:
            pending_id = str(uuid.uuid4())
            pending_path = os.path.join(
                PENDING_DIR,
                f"{pending_id}__{temp_filename}"
            )

            shutil.move(temp_path, pending_path)

            uniq = []

            for row in top_two:
                d = str(row.get("domain", "")).strip()
                if d and d not in uniq:
                    uniq.append(d)

            allowed = tuple(uniq[:2])

            if len(allowed) < 2:
                allowed = (prediction, prediction)

            PENDING_FILES[pending_id] = {
                "path": pending_path,
                "allowed_domains": allowed,
                "final_basename": temp_filename,
                "client_filename": client_name,
            }

            print(
                "Prediction (pending user choice):",
                prediction,
                "confidence:",
                confidence
            )

            return {
                "filename": client_name,
                "prediction": prediction,
                "stored_in": None,
                "confidence": confidence,
                "top_keywords": top_keywords,
                "ranking": ranking,
                "top_two_domains": top_two,
                "requires_manual_choice": True,
                "pending_id": pending_id,
            }

        # Auto sort case
        destination_path = os.path.join(
            BASE_DIR,
            prediction,
            temp_filename
        )

        print(
            "Prediction:",
            prediction,
            "Moving file to:",
            destination_path
        )

        shutil.move(temp_path, destination_path)

        return {
            "filename": client_name,
            "prediction": prediction,
            "stored_in": destination_path,
            "confidence": confidence,
            "top_keywords": top_keywords,
            "ranking": ranking,
            "top_two_domains": top_two,
            "requires_manual_choice": False,
            "pending_id": None,
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "filename": client_name,
            "error": str(e)
        }

# ---------------------------------------------------
# SINGLE FILE PREDICTION
# ---------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return await process_uploaded_file(file)

# ---------------------------------------------------
# BULK FILE PREDICTION
# ---------------------------------------------------

@app.post("/predict-bulk")
async def predict_bulk(files: list[UploadFile] = File(...)):

    results = []

    for file in files:
        result = await process_uploaded_file(file)
        results.append(result)

    return {
        "results": results
    }

# ---------------------------------------------------
# CONFIRM SORT
# ---------------------------------------------------

@app.post("/confirm-sort")
async def confirm_sort(body: ConfirmSortBody):

    entry = PENDING_FILES.get(body.pending_id)

    if not entry:
        return {
            "error": "Invalid or expired pending_id. Upload again."
        }

    chosen = body.chosen_domain.strip()

    if chosen not in CATEGORIES:
        return {
            "error": "Invalid folder name."
        }

    allowed = entry["allowed_domains"]

    if chosen not in allowed:
        return {
            "error": f"Choose one of: {', '.join(allowed)}"
        }

    src = entry["path"]

    if not os.path.isfile(src):
        PENDING_FILES.pop(body.pending_id, None)

        return {
            "error": "Pending file missing. Upload again."
        }

    dest_dir = os.path.join(BASE_DIR, chosen)
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(
        dest_dir,
        entry["final_basename"]
    )

    shutil.move(src, dest_path)

    PENDING_FILES.pop(body.pending_id, None)

    print("User chose:", chosen, "→", dest_path)

    return {
        "filename": entry["client_filename"],
        "prediction": chosen,
        "stored_in": dest_path,
        "confidence": 100.0,
        "requires_manual_choice": False,
        "pending_id": None,
        "resolved_by_user": True,
    }