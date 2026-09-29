#!/usr/bin/env python3
"""
generate_document_manifest.py

Automatically scans real-world test assets, extracts text streams using DocSort extraction logic,
and creates research/real_world_eval/document_manifest.csv.
"""

import os
from pathlib import Path
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "research" / "real_world_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = OUTPUT_DIR / "document_manifest.csv"

# Scan directories for candidate documents
SEARCH_PATHS = [
    ROOT_DIR / "data" / "real_world_eval",
    ROOT_DIR / "test_assets",
    ROOT_DIR / "test_assets" / "controlled_pdfs",
]

candidate_files = []
seen_paths = set()

for s_path in SEARCH_PATHS:
    if not s_path.exists():
        continue
    for f_path in s_path.glob("*"):
        if f_path.is_file() and f_path.suffix.lower() in [".pdf", ".docx", ".txt"]:
            abs_path = f_path.resolve()
            if abs_path not in seen_paths:
                seen_paths.add(abs_path)
                candidate_files.append(f_path)

records = []

for f in sorted(candidate_files, key=lambda x: str(x)):
    rel_path = str(f.relative_to(ROOT_DIR))
    filename = f.name
    ext = f.suffix.lower().replace(".", "").upper()

    text = ""
    try:
        if ext == "PDF":
            reader = PdfReader(str(f))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif ext == "DOCX":
            doc = Document(str(f))
            for para in doc.paragraphs:
                if para.text:
                    text += para.text + " "
        elif ext == "TXT":
            with open(f, "r", encoding="utf-8", errors="ignore") as tf:
                text = tf.read()
    except Exception as e:
        text = ""

    clean_text = " ".join(text.split())
    text_len = len(clean_text)

    # Classification rules based on document content & provenances
    provisional_label = "OOD"
    reason = "Out of domain document"
    needs_manual_review = False

    # 1. Controlled PDFs
    if "controlled_pdfs" in rel_path:
        if "01_Technology_Computing" in filename:
            provisional_label = "Technology & Computing"
            reason = "Controlled research benchmark sample for Technology & Computing"
        elif "02_Medical_Health" in filename:
            provisional_label = "Medical Health"
            reason = "Controlled research benchmark sample for Medical Health"
        elif "03_Business_Finance" in filename:
            provisional_label = "Business and Finance"
            reason = "Controlled research benchmark sample for Business and Finance"
        elif "04_Entertainment" in filename:
            provisional_label = "Entertainment"
            reason = "Controlled research benchmark sample for Entertainment"
        elif "05_Sports" in filename:
            provisional_label = "Sports"
            reason = "Controlled research benchmark sample for Sports"
        elif "06_Science" in filename:
            provisional_label = "Science"
            reason = "Controlled research benchmark sample for Science"
        needs_manual_review = False

    # 2. Known OOD Documents
    elif filename in ["MANVI_BIODATA.pdf"]:
        provisional_label = "OOD"
        reason = "Personal marriage biodata profile document"
        needs_manual_review = False

    elif filename in ["TANISH_ARORA_CV_OLD.pdf"]:
        provisional_label = "OOD"
        reason = "Professional resume / CV document"
        needs_manual_review = False

    elif "Air India Web Booking" in filename:
        provisional_label = "OOD"
        reason = "Airline e-ticket booking confirmation"
        needs_manual_review = False

    elif "ID card" in filename:
        provisional_label = "OOD"
        reason = "Student identity card"
        needs_manual_review = False

    elif "Receipt_" in filename:
        provisional_label = "OOD"
        reason = "Retail payment transaction receipt"
        needs_manual_review = False

    elif filename in ["ood_miscellaneous_doc.pdf"]:
        provisional_label = "OOD"
        reason = "Administrative building access protocol"
        needs_manual_review = False

    # 3. Clear Domain Documents
    elif filename == "back-of-the-napkin.pdf":
        provisional_label = "Business and Finance"
        reason = "Business problem-solving and corporate executive strategy publication"
        needs_manual_review = False

    # 4. Ambiguous Candidate Documents
    elif filename == "Speaker1_What_to_Speak_Only.docx":
        provisional_label = "AMBIGUOUS"
        reason = "Dual-domain speech script: Machine Learning AI technology applied to Medical Disease Diagnosis"
        needs_manual_review = True

    elif filename == "LAB MANUAL.pdf":
        provisional_label = "AMBIGUOUS"
        reason = "Academic course lab manual containing Deep Learning & Computer Science neural network lab exercises"
        needs_manual_review = True

    elif filename == "Major Project Synopsis.pdf":
        provisional_label = "AMBIGUOUS"
        reason = "Academic project proposal formatting combined with Computer Science & AI software architecture content"
        needs_manual_review = True

    elif filename == "Syllabus.pdf":
        provisional_label = "AMBIGUOUS"
        reason = "Academic course syllabus covering Microcontrollers, assembly programming, and hardware architecture"
        needs_manual_review = True

    records.append({
        "filename": filename,
        "filepath": rel_path,
        "file_type": ext,
        "text_length": text_len,
        "provisional_label": provisional_label,
        "reason": reason,
        "needs_manual_review": needs_manual_review,
    })

df = pd.DataFrame(records)
df.to_csv(MANIFEST_CSV, index=False)
print(f"Successfully generated document manifest CSV at: {MANIFEST_CSV}")
print(f"Total documents inventoried: {len(df)}")
print(f"Documents requiring manual review: {len(df[df['needs_manual_review'] == True])}")
