import os
import pytest
from fastapi.testclient import TestClient
import joblib

from backend.main import app, BASE_SORTED_DIR, TAXONOMY_CLASSES
from backend.research_inference import (
    load_production_model,
    predict_document,
    extract_structural_features,
    STRUCTURAL_MODEL_DIR,
)

client = TestClient(app)

# 10. Model loads successfully from structural model artifacts
def test_model_loads_successfully_from_structural_model_artifacts():
    artifacts = load_production_model()
    assert "vectorizer" in artifacts
    assert "scaler" in artifacts
    assert "clf" in artifacts
    assert os.path.exists(os.path.join(STRUCTURAL_MODEL_DIR, "tfidf_vectorizer.joblib"))
    assert os.path.exists(os.path.join(STRUCTURAL_MODEL_DIR, "scaler.joblib"))
    assert os.path.exists(os.path.join(STRUCTURAL_MODEL_DIR, "linearsvc_combined.joblib"))

# 9. Six categories returned correctly
def test_six_categories_returned_correctly():
    response = client.get("/categories")
    assert response.status_code == 200
    categories = response.json()["categories"]
    expected = [
        "Technology & Computing",
        "Science & Academics",
        "Medical Health",
        "Business & Finance",
        "Entertainment",
        "Sports",
    ]
    assert categories == expected

# 1. Lab Manual -> Science & Academics
def test_lab_manual_prediction():
    lab_manual_path = "research/real_world_eval/LAB MANUAL.pdf"
    if os.path.exists(lab_manual_path):
        with open(lab_manual_path, "rb") as f:
            response = client.post("/predict", files={"file": ("LAB MANUAL.pdf", f, "application/pdf")})
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "Science & Academics"
        assert data["final_destination"] == "Science & Academics"
        assert not data["is_anonymous"]

# 2. Syllabus -> Science & Academics
def test_syllabus_prediction():
    syllabus_path = "research/real_world_eval/Syllabus.pdf"
    if os.path.exists(syllabus_path):
        with open(syllabus_path, "rb") as f:
            response = client.post("/predict", files={"file": ("Syllabus.pdf", f, "application/pdf")})
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "Science & Academics"
        assert data["final_destination"] == "Science & Academics"
        assert not data["is_anonymous"]

# 3. Major Project Synopsis -> Technology & Computing
def test_major_project_synopsis_prediction():
    synopsis_path = "research/real_world_eval/Major Project Synopsis.pdf"
    if os.path.exists(synopsis_path):
        with open(synopsis_path, "rb") as f:
            response = client.post("/predict", files={"file": ("Major Project Synopsis.pdf", f, "application/pdf")})
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "Technology & Computing"
        assert data["final_destination"] == "Technology & Computing"
        assert not data["is_anonymous"]

# 4. Medical presentation -> Medical Health
def test_medical_presentation_prediction():
    medical_path = "research/real_world_eval/Speaker1_What_to_Speak_Only.docx"
    if os.path.exists(medical_path):
        with open(medical_path, "rb") as f:
            response = client.post("/predict", files={"file": ("Speaker1_What_to_Speak_Only.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "Medical Health"
        assert data["final_destination"] == "Medical Health"
        assert not data["is_anonymous"]

# 5. Low-margin/OOD document -> Anonymous
def test_low_margin_ood_document_anonymous():
    ood_path = "research/real_world_eval/MANVI_BIODATA.pdf"
    if os.path.exists(ood_path):
        with open(ood_path, "rb") as f:
            response = client.post("/predict", files={"file": ("MANVI_BIODATA.pdf", f, "application/pdf")})
        assert response.status_code == 200
        data = response.json()
        assert data["is_anonymous"] is True
        assert data["final_destination"] == "Anonymous"
        assert data["decision_margin"] < 0.25

# 6. Blank document -> rejected
def test_blank_document_rejected():
    response = client.post("/predict", files={"file": ("empty.txt", b"   \n\n  \t ", "text/plain")})
    assert response.status_code == 400
    assert "blank" in response.json()["detail"].lower()

# 7. Known document -> correct category folder
def test_known_document_sorted_to_category_folder():
    synopsis_path = "research/real_world_eval/Major Project Synopsis.pdf"
    if os.path.exists(synopsis_path):
        with open(synopsis_path, "rb") as f:
            response = client.post("/predict", files={"file": ("Major Project Synopsis.pdf", f, "application/pdf")})
        assert response.status_code == 200
        data = response.json()
        expected_dir = os.path.expanduser(f"~/Desktop/SortedDocuments/{data['predicted_class']}")
        assert data["stored_in"].startswith(expected_dir)
        assert os.path.exists(data["stored_in"])

# 8. Anonymous document -> Anonymous folder
def test_anonymous_document_sorted_to_anonymous_folder():
    ood_path = "research/real_world_eval/MANVI_BIODATA.pdf"
    if os.path.exists(ood_path):
        with open(ood_path, "rb") as f:
            response = client.post("/predict", files={"file": ("MANVI_BIODATA.pdf", f, "application/pdf")})
        assert response.status_code == 200
        data = response.json()
        expected_dir = os.path.expanduser("~/Desktop/SortedDocuments/Anonymous")
        assert data["stored_in"].startswith(expected_dir)
        assert os.path.exists(data["stored_in"])
        assert "Anonymous" in data["stored_in"]
