"""
test_docsort_fixes.py

Automated test suite verifying the DocSort AI application behavior:
1. Blank/empty document validation (PDF, TXT, DOCX, whitespace, control noise) -> HTTP 400 rejection
2. Machine Learning wording & 6 categories consistency
3. All six research categories classification via production model:
   - Technology & Computing
   - Science & Academics
   - Medical Health
   - Business & Finance
   - Entertainment
   - Sports
"""

import io
import pytest
from fastapi.testclient import TestClient
from docx import Document
from PyPDF2 import PdfWriter

from backend.main import app, CATEGORIES, validate_document_content, BLANK_DOCUMENT_ERROR_MESSAGE

client = TestClient(app)

EXPECTED_CATEGORIES = [
    "Technology & Computing",
    "Science & Academics",
    "Medical Health",
    "Business & Finance",
    "Entertainment",
    "Sports",
]


# =====================================================================
# 1. BLANK / EMPTY DOCUMENT VALIDATION TESTS
# =====================================================================

def test_unit_validate_document_content():
    """Direct unit tests for validation logic."""
    assert validate_document_content(None)[0] is False
    assert validate_document_content("")[0] is False
    assert validate_document_content("    \t\n\r  ")[0] is False
    assert validate_document_content("\x00\x01\x1f\x7f\u200b\u200c\ufeff")[0] is False
    assert validate_document_content("   \u200b\u200e\ufeff   ")[0] is False
    assert validate_document_content("--- ... ,,, !!!")[0] is False
    assert validate_document_content("a")[0] is False

    ok_receipt, text_receipt = validate_document_content("Receipt #104 Total: $24.50")
    assert ok_receipt is True
    assert "Receipt #104 Total: $24.50" in text_receipt

    ok_note, text_note = validate_document_content("Biology Notes - Genetics")
    assert ok_note is True
    assert "Biology Notes - Genetics" in text_note


def test_blank_txt_upload():
    """Empty TXT should be rejected with HTTP 400."""
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "blank" in response.json().get("detail", "").lower()


def test_whitespace_only_txt_upload():
    """Whitespace-only TXT should be rejected with HTTP 400."""
    files = {"file": ("spaces.txt", b"   \n\t  \r\n   ", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "blank" in response.json().get("detail", "").lower()


def test_control_chars_txt_upload():
    """TXT containing only invisible/control chars should be rejected with HTTP 400."""
    noise_bytes = "\x00\x05\x1b\x7f\u200b\u200c\ufeff".encode("utf-8")
    files = {"file": ("noise.txt", noise_bytes, "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "blank" in response.json().get("detail", "").lower()


def test_blank_pdf_upload():
    """Empty PDF with blank page should be rejected with HTTP 400."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    pdf_bytes = io.BytesIO()
    writer.write(pdf_bytes)
    pdf_bytes.seek(0)

    files = {"file": ("blank.pdf", pdf_bytes.read(), "application/pdf")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "blank" in response.json().get("detail", "").lower()


def test_blank_docx_upload():
    """Empty DOCX should be rejected with HTTP 400."""
    doc = Document()
    docx_bytes = io.BytesIO()
    doc.save(docx_bytes)
    docx_bytes.seek(0)

    files = {"file": ("blank.docx", docx_bytes.read(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "blank" in response.json().get("detail", "").lower()


def test_valid_short_document():
    """Legitimate short document should be allowed and processed."""
    content = b"Biology Lab Report: Microscopic cellular examination of plant specimens in academic course laboratory."
    files = {"file": ("lab_note.txt", content, "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_class") in CATEGORIES


# =====================================================================
# 2. SIX-CATEGORY PREDICTION TESTS (RESEARCH TAXONOMY)
# =====================================================================

def test_technology_document():
    text = (
        "Modern cloud computing architectures rely on distributed microservices, "
        "Kubernetes orchestration, API gateways, serverless computing, and GPU hardware acceleration. "
        "Software engineers write backend algorithms in Python and Rust, deploying containers to data centers."
    )
    files = {"file": ("tech.txt", text.encode("utf-8"), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_class") == "Technology & Computing"


def test_medical_document():
    text = (
        "The clinical trial evaluated patient response to the pharmaceutical therapeutic regimen. "
        "Physicians recorded vital symptoms, blood pressure, diagnosis criteria, and chemotherapy dosage. "
        "Hospital healthcare protocols required continuous oncology monitoring for adverse side effects."
    )
    files = {"file": ("med.txt", text.encode("utf-8"), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_class") == "Medical Health"


def test_finance_document():
    text = (
        "The corporation reported quarterly revenue growth, earnings per share, and EBITDA margins. "
        "Wall Street investment analysts tracked equity markets, federal reserve interest rates, "
        "and shareholder dividends across commercial banking and mutual fund portfolios."
    )
    files = {"file": ("finance.txt", text.encode("utf-8"), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_class") == "Business & Finance"


def test_entertainment_document():
    text = (
        "The Hollywood feature film premiered at the Venice International Film Festival to standing ovations. "
        "The lead actress and director discussed the movie cinematography, screenplay soundtrack, and studio release. "
        "Box office ticket sales topped nationwide theatre charts as fans praised the cinematic drama."
    )
    files = {"file": ("movie.txt", text.encode("utf-8"), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_class") == "Entertainment"


def test_sports_document():
    text = (
        "The championship tournament match ended with a dramatic stoppage time goal by the striker. "
        "League coaches praised the team defense, athletic conditioning, and playoff victory. "
        "Fans celebrated across the stadium as the soccer squad advanced to the tournament finals."
    )
    files = {"file": ("sports.txt", text.encode("utf-8"), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_class") == "Sports"


def test_science_document():
    text = (
        "Astronomers published empirical research detailing deep space observations from the space telescope. "
        "Astrophysicists confirmed gravitational lensing phenomena around distant galaxies and planetary nebulae. "
        "The scientific laboratory verified quantum particle physics data against mathematical simulations."
    )
    files = {"file": ("science.txt", text.encode("utf-8"), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_class") == "Science & Academics"


# =====================================================================
# 3. CONSISTENCY & MODEL METADATA TESTS
# =====================================================================

def test_model_info_and_categories_endpoints():
    """Verify backend exposes exact 6 categories and production model."""
    res = client.get("/model-info")
    assert res.status_code == 200
    info = res.json()
    assert info.get("model") == "TF-IDF + Structural Features + LinearSVC"
    assert set(info.get("classes")) == set(EXPECTED_CATEGORIES)

    res_cat = client.get("/categories")
    assert res_cat.status_code == 200
    cats = res_cat.json().get("categories", [])
    assert cats == EXPECTED_CATEGORIES
