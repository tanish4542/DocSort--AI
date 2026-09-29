"""
test_ml2_model_comparison.py

Production model comparison and inference test suite:
1. Evaluation of controlled domain PDFs across the 6 final taxonomy categories:
   - 01_Technology_Computing.pdf -> Technology & Computing
   - 02_Medical_Health.pdf -> Medical Health
   - 03_Business_Finance.pdf -> Business & Finance
   - 04_Entertainment.pdf -> Entertainment
   - 05_Sports.pdf -> Sports
   - 06_Science.pdf -> Science & Academics
2. Verification of production model fields:
   - predicted_class
   - decision_margin
   - is_anonymous
   - final_destination
   - model
3. Verification of automatic sorting without manual review workflows.
4. Blank document rejection (HTTP 400).
5. Bulk prediction support.
"""

import io
from pathlib import Path
import fitz  # PyMuPDF
import pytest
from docx import Document
from fastapi.testclient import TestClient

from backend.main import app, CATEGORIES

client = TestClient(app)

CONTROLLED_DIR = Path("test_assets/controlled_pdfs")
CONTROLLED_DIR.mkdir(parents=True, exist_ok=True)

CONTROLLED_DOCS = [
    {
        "filename": "01_Technology_Computing.pdf",
        "expected_category": "Technology & Computing",
        "title": "Cloud Computing Architectures, Distributed Microservices, and ML Systems",
        "content": (
            "Modern enterprise infrastructure utilizes Kubernetes container orchestration, Docker virtualization, "
            "serverless compute functions, and API gateway routing. Software developers deploy low-latency backend "
            "architectures programmed in Python, Rust, and Go, leveraging GPU hardware accelerators and high-throughput "
            "network fabrics across global data centers. Database systems utilize distributed key-value storage, "
            "Apache Kafka event streams, and automated continuous integration pipelines."
        ),
    },
    {
        "filename": "02_Medical_Health.pdf",
        "expected_category": "Medical Health",
        "title": "Clinical Therapeutics, Oncology Protocols, and Patient Care",
        "content": (
            "The clinical trial evaluated patient response to novel chemotherapy dosage and pharmaceutical "
            "therapeutic regimens. Attending physicians and nurses monitored patient vital signs, diagnostic blood pressure, "
            "blood glucose, and oncology symptoms to prevent adverse drug reactions. Medical healthcare protocols mandated "
            "comprehensive pathology reviews, outpatient clinical therapy schedules, and cardiology diagnostic evaluations."
        ),
    },
    {
        "filename": "03_Business_Finance.pdf",
        "expected_category": "Business & Finance",
        "title": "Quarterly Corporate Earnings, Equity Valuation, and Asset Management",
        "content": (
            "The public corporation announced its quarterly fiscal results, recording record revenue growth, operating "
            "cash flow, and EBITDA profit margins. Wall Street investment analysts and equity portfolio managers reviewed "
            "capital expenditure allocations, shareholder dividend payments, balance sheet leverage, and commercial banking "
            "credit facilities amidst macroeconomic interest rate fluctuations by the Federal Reserve."
        ),
    },
    {
        "filename": "04_Entertainment.pdf",
        "expected_category": "Entertainment",
        "title": "Cinema Premiere, Box Office Performance, and Film Festival Reviews",
        "content": (
            "The Hollywood theatrical feature film premiered at the international film festival, earning praise from "
            "movie critics, directors, and audiences. The lead actress and actor delivered stellar acting performances "
            "supported by an original orchestral soundtrack, cinematography, and screenplay script. The entertainment "
            "studio reported box office ticket sales surpassing weekend theatre projections across multiplex cinemas."
        ),
    },
    {
        "filename": "05_Sports.pdf",
        "expected_category": "Sports",
        "title": "Championship Tournament Match, Team Performance, and League Standings",
        "content": (
            "The soccer championship finals concluded with a dramatic stoppage time winning goal by the forward striker. "
            "League coaches praised the roster's defensive formation, tactical speed, and athletic conditioning throughout "
            "the playoff tournament. Thousands of cheering fans in the stadium celebrated the squad hoisting the golden victory "
            "trophy as sports journalists covered the post-game press conference."
        ),
    },
    {
        "filename": "06_Science.pdf",
        "expected_category": "Science & Academics",
        "title": "Astrophysical Observations, Quantum Mechanics, and Empirical Laboratory Research",
        "content": (
            "Astronomers and astrophysicists published groundbreaking observational data from space telescopes observing "
            "gravitational lensing, cosmic microwave radiation, and stellar nebulae in distant galaxies. Physics laboratory "
            "researchers verified quantum particle behavior using laser spectroscopy and mathematical simulation models, "
            "establishing empirical scientific data regarding subatomic particle interactions."
        ),
    },
]


def ensure_controlled_pdfs():
    """Generate the six controlled domain PDFs if not present."""
    for item in CONTROLLED_DOCS:
        filepath = CONTROLLED_DIR / item["filename"]
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 70), item["title"], fontsize=14)
        rect = fitz.Rect(50, 100, 550, 400)
        page.insert_textbox(rect, item["content"], fontsize=11)
        doc.save(str(filepath))
        doc.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    ensure_controlled_pdfs()


# =====================================================================
# 1. SIX CONTROLLED PDF PREDICTION TESTS
# =====================================================================

@pytest.mark.parametrize("doc_info", CONTROLLED_DOCS)
def test_controlled_pdf_prediction(doc_info):
    pdf_path = CONTROLLED_DIR / doc_info["filename"]
    assert pdf_path.exists(), f"File {pdf_path} must exist"

    with open(pdf_path, "rb") as f:
        response = client.post("/predict", files={"file": (doc_info["filename"], f, "application/pdf")})

    assert response.status_code == 200, f"Error {response.status_code}: {response.text}"
    data = response.json()

    # 1. Final prediction must match the expected research category
    assert data.get("predicted_class") == doc_info["expected_category"]
    assert data.get("final_destination") == doc_info["expected_category"]
    assert data.get("model") == "TF-IDF + Structural Features + LinearSVC"
    assert data.get("decision_margin") is not None
    assert len(data.get("decision_scores", {})) == 6
    assert len(data.get("top_keywords", [])) > 0


# =====================================================================
# 2. BLANK DOCUMENT VALIDATION TESTS
# =====================================================================

def test_blank_pdf_rejected():
    """Verify empty PDF is rejected with HTTP 400."""
    blank_doc = fitz.open()
    blank_doc.new_page()
    pdf_bytes = blank_doc.tobytes()
    blank_doc.close()

    response = client.post("/predict", files={"file": ("blank.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 400
    assert "blank" in response.json().get("detail", "").lower()


def test_blank_txt_rejected():
    response = client.post("/predict", files={"file": ("empty.txt", b"   \n\t   ", "text/plain")})
    assert response.status_code == 400
    assert "blank" in response.json().get("detail", "").lower()


def test_blank_docx_rejected():
    doc = Document()
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    response = client.post(
        "/predict",
        files={"file": ("empty.docx", bio.read(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 400
    assert "blank" in response.json().get("detail", "").lower()


# =====================================================================
# 3. DOCX AND TXT VALID PREDICTIONS
# =====================================================================

def test_docx_valid_upload():
    doc = Document()
    doc.add_heading("Cardiology Clinical Protocols", level=1)
    doc.add_paragraph("The patient cardiology examination revealed sinus rhythm without acute ischemic symptoms.")
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    response = client.post(
        "/predict",
        files={"file": ("cardio.docx", bio.read(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_class") == "Medical Health"
    assert data.get("model") == "TF-IDF + Structural Features + LinearSVC"


def test_txt_valid_upload():
    text = "Wall Street quarterly dividend yields and equity bond portfolio asset management."
    response = client.post("/predict", files={"file": ("finance.txt", text.encode("utf-8"), "text/plain")})
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_class") == "Business & Finance"
    assert data.get("model") == "TF-IDF + Structural Features + LinearSVC"


# =====================================================================
# 4. BULK PREDICTION TESTS
# =====================================================================

def test_bulk_predict_multi_models():
    ensure_controlled_pdfs()
    p1 = CONTROLLED_DIR / "01_Technology_Computing.pdf"
    p2 = CONTROLLED_DIR / "05_Sports.pdf"

    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        files = [
            ("files", ("01_Technology_Computing.pdf", f1.read(), "application/pdf")),
            ("files", ("05_Sports.pdf", f2.read(), "application/pdf")),
        ]
        response = client.post("/predict-bulk", files=files)

    assert response.status_code == 200
    data = response.json()
    items = data.get("results", [])
    assert len(items) == 2

    for item in items:
        assert item.get("model") == "TF-IDF + Structural Features + LinearSVC"
        assert item.get("final_destination") == item.get("predicted_class")


# =====================================================================
# 5. OUT-OF-DOMAIN & LOW-MARGIN UNCERTAINTY HANDLING TESTS
# =====================================================================

def test_sister_biodata_manvi_low_margin_anonymous():
    """
    MANVI_BIODATA.pdf:
    - LinearSVC margin is < 0.25.
    - Must automatically route to Anonymous.
    """
    biodata_path = Path("test_assets/MANVI_BIODATA.pdf")
    if biodata_path.exists():
        with open(biodata_path, "rb") as f:
            res = client.post("/predict", files={"file": ("MANVI_BIODATA.pdf", f, "application/pdf")})

        assert res.status_code == 200
        data = res.json()
        assert data.get("is_anonymous") is True
        assert data.get("final_destination") == "Anonymous"
        assert data.get("decision_margin") < 0.25
        assert "Anonymous" in data.get("stored_in", "")
