"""
test_ml2_model_comparison.py

Comprehensive test suite verifying ML-2 research model comparison implementation:
1. Generation and evaluation of the 6 controlled domain PDFs:
   - 01_Technology_Computing.pdf
   - 02_Medical_Health.pdf
   - 03_Business_Finance.pdf
   - 04_Entertainment.pdf
   - 05_Sports.pdf
   - 06_Science.pdf
2. Verification of all 3 individual models:
   - Multinomial Naive Bayes (Test Acc: 91.16%, Macro F1: 91.17%)
   - Logistic Regression (Test Acc: 94.06%, Macro F1: 94.05%)
   - LinearSVC (Test Acc: 94.50%, Macro F1: 94.49%) ★ SELECTED
3. Verification that LinearSVC is the final prediction and sorting model (NO voting).
4. Verification of per-document scores (Probability for NB/LR, Decision Score for LinearSVC).
5. Verification of Ensemble Experiments benchmarks (Soft Voting 94.15%, Stacking 94.47%).
6. Verification of Blank document rejection (PDF, TXT, DOCX).
7. Verification of Bulk prediction.
"""

import io
import os
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
        "expected_category": "Business and Finance",
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
        "expected_category": "Science",
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
        # Title
        page.insert_text((50, 70), item["title"], fontsize=14)
        # Content
        rect = fitz.Rect(50, 100, 550, 400)
        page.insert_textbox(rect, item["content"], fontsize=11)
        doc.save(str(filepath))
        doc.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    ensure_controlled_pdfs()


# =====================================================================
# 1. SIX CONTROLLED PDF PREDICTION & MODEL COMPARISON TESTS
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
    assert data.get("prediction") == doc_info["expected_category"]
    assert data.get("final_prediction") == doc_info["expected_category"]

    # 2. Selected model must be LinearSVC
    assert data.get("selected_model") == "LinearSVC"
    assert "LinearSVC achieved the highest test Macro F1" in data.get("selection_reason", "")

    # 3. Model comparison must contain all 3 individual models
    comparison = data.get("model_comparison", [])
    assert len(comparison) == 3, f"Expected 3 models in comparison, got {len(comparison)}"

    model_names = [m["model"] for m in comparison]
    assert "Multinomial Naive Bayes" in model_names
    assert "Logistic Regression" in model_names
    assert "LinearSVC" in model_names

    # Check each model in comparison
    for m in comparison:
        assert m["prediction"] in CATEGORIES
        assert "test_accuracy" in m
        assert "test_macro_f1" in m
        assert "score_type" in m

        if m["model"] == "Multinomial Naive Bayes":
            assert m["test_accuracy"] == 0.9116
            assert m["test_macro_f1"] == 0.9117
            assert m["score_type"] == "Probability"
            assert m["is_selected"] is False

        elif m["model"] == "Logistic Regression":
            assert m["test_accuracy"] == 0.9406
            assert m["test_macro_f1"] == 0.9405
            assert m["score_type"] == "Probability"
            assert m["is_selected"] is False

        elif m["model"] == "LinearSVC":
            assert m["test_accuracy"] == 0.9450
            assert m["test_macro_f1"] == 0.9449
            assert m["score_type"] == "Decision Score"
            assert m["is_selected"] is True
            # LinearSVC prediction must equal the final prediction
            assert m["prediction"] == data["final_prediction"]

    # 4. Model performance dictionary
    perf = data.get("model_performance", {})
    assert perf["Multinomial Naive Bayes"]["accuracy"] == 0.9116
    assert perf["Multinomial Naive Bayes"]["macro_f1"] == 0.9117
    assert perf["Logistic Regression"]["accuracy"] == 0.9406
    assert perf["Logistic Regression"]["macro_f1"] == 0.9405
    assert perf["LinearSVC"]["accuracy"] == 0.9450
    assert perf["LinearSVC"]["macro_f1"] == 0.9449

    # 5. Ensemble experiments section
    ensembles = data.get("ensemble_experiments", {})
    assert ensembles["Soft Voting"]["macro_f1"] == 0.9415
    assert ensembles["Stacking"]["macro_f1"] == 0.9447
    assert ensembles["LinearSVC"]["macro_f1"] == 0.9449

    # 6. Existing decision margin and XAI functionality intact
    assert data.get("decision_margin") is not None
    assert isinstance(data.get("decision_margin"), (int, float))
    assert data.get("uncertainty_level") in ["High confidence", "Moderate confidence", "Ambiguous"]
    assert len(data.get("decision_scores", {})) == 6
    assert len(data.get("top_keywords", [])) > 0


# =====================================================================
# 2. BLANK DOCUMENT VALIDATION TESTS
# =====================================================================

def test_blank_pdf_rejected():
    """Verify empty PDF is rejected without classification."""
    blank_doc = fitz.open()
    blank_doc.new_page()  # empty page with no text
    pdf_bytes = blank_doc.tobytes()
    blank_doc.close()

    response = client.post("/predict", files={"file": ("blank.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 200
    data = response.json()
    assert data.get("is_blank") is True
    assert data.get("prediction") is None
    assert data.get("stored_in") is None
    assert "blank or contains no readable text" in data.get("error", "")


def test_blank_txt_rejected():
    response = client.post("/predict", files={"file": ("empty.txt", b"   \n\t   ", "text/plain")})
    assert response.status_code == 200
    data = response.json()
    assert data.get("is_blank") is True
    assert data.get("prediction") is None


def test_blank_docx_rejected():
    doc = Document()
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    response = client.post(
        "/predict",
        files={"file": ("empty.docx", bio.read(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("is_blank") is True
    assert data.get("prediction") is None


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
    assert data.get("prediction") == "Medical Health"
    assert len(data.get("model_comparison", [])) == 3


def test_txt_valid_upload():
    text = "Wall Street quarterly dividend yields and equity bond portfolio asset management."
    response = client.post("/predict", files={"file": ("finance.txt", text.encode("utf-8"), "text/plain")})
    assert response.status_code == 200
    data = response.json()
    assert data.get("prediction") == "Business and Finance"
    assert len(data.get("model_comparison", [])) == 3


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
        assert item.get("selected_model") == "LinearSVC"
        assert len(item.get("model_comparison", [])) == 3
        assert item.get("final_prediction") == item.get("prediction")
        assert "test_macro_f1" in item["model_comparison"][0]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
