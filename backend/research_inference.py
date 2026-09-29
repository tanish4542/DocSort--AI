"""
backend/research_inference.py

Production inference module for DocSort AI using the Structural Feature + TF-IDF LinearSVC model.

Artifacts Loaded:
- research/taxonomy_v2_structural/models/tfidf_vectorizer.joblib
- research/taxonomy_v2_structural/models/scaler.joblib
- research/taxonomy_v2_structural/models/linearsvc_combined.joblib

6 Final Taxonomy Classes:
    1. Technology & Computing
    2. Science & Academics
    3. Medical Health
    4. Business & Finance
    5. Entertainment
    6. Sports

Decision Rule:
    decision_margin = top_score - second_highest_score
    if decision_margin < 0.25:
        final_destination = "Anonymous"
        is_anonymous = True
    else:
        final_destination = predicted_class
        is_anonymous = False
"""

from __future__ import annotations

import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger("docsort.research_inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Resolve paths to structural research model artifacts
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
STRUCTURAL_MODELS_DIR = PROJECT_ROOT / "research" / "taxonomy_v2_structural" / "models"

VEC_PATH = STRUCTURAL_MODELS_DIR / "tfidf_vectorizer.joblib"
SCALER_PATH = STRUCTURAL_MODELS_DIR / "scaler.joblib"
SVC_PATH = STRUCTURAL_MODELS_DIR / "linearsvc_combined.joblib"

# Provisional decision margin threshold
PROVISIONAL_MARGIN_THRESHOLD = 0.25
STRUCT_WEIGHT = 2.0

# Keywords for Structural Feature Extraction
ACADEMIC_KEYWORDS = [
    "syllabus", "lab manual", "course syllabus", "course objectives",
    "learning objectives", "learning outcomes", "course outcomes",
    "prerequisites", "credits", "semester", "module", "unit",
    "experiment", "experiments", "assessment", "grading", "marks",
    "faculty", "department", "university", "college", "laboratory",
    "lab", "references", "textbook"
]

TECH_KEYWORDS = [
    "software", "program", "programming", "code", "compiler",
    "api", "framework", "library", "repository", "github",
    "deployment", "installation", "version", "implementation",
    "database", "algorithm", "hardware"
]

STRUCTURAL_MODEL_DIR = str(STRUCTURAL_MODELS_DIR)

def load_production_model():
    return {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "clf": linear_svc,
        "model_dir": STRUCTURAL_MODEL_DIR,
    }

# Validation: ensure model files exist
for artifact_path, name in [
    (VEC_PATH, "TF-IDF Vectorizer"),
    (SCALER_PATH, "Structural Scaler"),
    (SVC_PATH, "Combined LinearSVC Classifier"),
]:
    if not artifact_path.is_file():
        raise RuntimeError(
            f"Critical startup error: {name} artifact not found at {artifact_path}."
        )

# Load production structural model artifacts
try:
    vectorizer = joblib.load(VEC_PATH)
    scaler = joblib.load(SCALER_PATH)
    linear_svc = joblib.load(SVC_PATH)

    CLASSES = list(linear_svc.classes_)
    FEATURE_NAMES = np.array(vectorizer.get_feature_names_out())
    VOCAB_SIZE = len(vectorizer.vocabulary_)

    logger.info(f"Loaded structural model artifacts from {STRUCTURAL_MODELS_DIR}")
    logger.info(f"Classes ({len(CLASSES)}): {CLASSES}")
    logger.info(f"TF-IDF vocabulary size: {VOCAB_SIZE:,} features")
except Exception as e:
    raise RuntimeError(f"Failed to load structural model artifacts: {e}") from e


def extract_structural_features(texts: List[str]) -> np.ndarray:
    """Extracts explicit structural feature vector for each text."""
    features = []
    code_line_pattern = re.compile(
        r"^\s*(def |class |import |from |#include|int |void |return |if \(|for \(|var |let |const |;\s*$|\{\s*$|\}\s*$)",
        re.MULTILINE
    )

    for txt in texts:
        t_str = str(txt)
        t_lower = t_str.lower()
        words = t_lower.split()
        num_words = max(len(words), 1)

        acad_cnt = sum(t_lower.count(kw) for kw in ACADEMIC_KEYWORDS)
        tech_cnt = sum(t_lower.count(kw) for kw in TECH_KEYWORDS)

        doc_len_log = np.log1p(len(t_str))
        acad_density = acad_cnt / num_words
        tech_density = tech_cnt / num_words
        struct_ratio = (acad_cnt - tech_cnt) / (acad_cnt + tech_cnt + 1.0)

        lines = [line.strip() for line in t_str.splitlines() if line.strip()]
        num_lines = max(len(lines), 1)
        code_lines_cnt = sum(1 for line in lines if code_line_pattern.match(line))
        code_line_density = code_lines_cnt / num_lines

        has_syllabus_hdr = 1.0 if any(k in t_lower for k in ["syllabus", "course outline", "course overview", "credit hours", "prerequisites:"]) else 0.0
        has_lab_manual_hdr = 1.0 if any(k in t_lower for k in ["lab manual", "laboratory manual", "lab experiment", "experimental procedure", "viva voce"]) else 0.0

        feat_vector = [
            float(acad_cnt),
            float(tech_cnt),
            float(doc_len_log),
            float(acad_density),
            float(tech_density),
            float(struct_ratio),
            float(code_line_density),
            float(has_syllabus_hdr),
            float(has_lab_manual_hdr),
        ]
        features.append(feat_vector)

    return np.array(features, dtype=np.float32)


def get_top_features_for_doc(
    X_tfidf_sparse,
    top_class_idx: int,
    top_k: int = 12
) -> List[Tuple[str, float]]:
    """Calculates TF-IDF lexical feature contributions for the predicted class."""
    cx = X_tfidf_sparse.tocoo()
    if cx.nnz == 0:
        return []

    col_indices = cx.col
    tfidf_values = cx.data
    class_weights = linear_svc.coef_[top_class_idx, col_indices]

    contributions = tfidf_values * class_weights
    ranked_order = np.argsort(contributions)[::-1]

    top_features: List[Tuple[str, float]] = []
    for idx in ranked_order:
        contrib = float(contributions[idx])
        if contrib <= 0 and len(top_features) >= 3:
            break
        word = str(FEATURE_NAMES[col_indices[idx]])
        top_features.append((word, round(contrib, 4)))
        if len(top_features) >= top_k:
            break

    return top_features


def predict_text(text: str) -> Dict[str, Any]:
    """
    Performs production inference on a raw extracted document string using the
    Combined TF-IDF + Structural Features + LinearSVC model.
    """
    cleaned_str = str(text).strip() if text else ""
    text_length = len(cleaned_str)

    if not cleaned_str:
        return {
            "predicted_class": "Anonymous",
            "runner_up_class": "Anonymous",
            "decision_margin": 0.0,
            "is_anonymous": True,
            "final_destination": "Anonymous",
            "model": "TF-IDF + Structural Features + LinearSVC",
            "top_score": 0.0,
            "runner_up_score": 0.0,
            "decision_scores": {c: 0.0 for c in CLASSES},
            "top_keywords": [],
            "top_features": [],
            "text_length": 0,
        }

    # 1. Transform TF-IDF
    X_tfidf = vectorizer.transform([cleaned_str])

    # 2. Extract & Scale Structural Features
    struct_raw = extract_structural_features([cleaned_str])
    struct_scaled = scaler.transform(struct_raw)

    # 3. Concatenate TF-IDF + Scaled Structural Features (with 2.0 weighting matching training)
    X_comb = hstack([X_tfidf, csr_matrix(struct_scaled * STRUCT_WEIGHT)]).tocsr()

    # 4. Predict LinearSVC Decision Function
    svc_scores = linear_svc.decision_function(X_comb)[0]
    order = np.argsort(svc_scores)[::-1]

    top_idx = int(order[0])
    runner_up_idx = int(order[1])

    top_class = str(CLASSES[top_idx])
    runner_up_class = str(CLASSES[runner_up_idx])

    top_score = float(svc_scores[top_idx])
    runner_up_score = float(svc_scores[runner_up_idx])

    decision_margin = round(top_score - runner_up_score, 4)

    # Anonymous decision threshold rule
    if decision_margin < PROVISIONAL_MARGIN_THRESHOLD:
        final_destination = "Anonymous"
        is_anonymous = True
    else:
        final_destination = top_class
        is_anonymous = False

    decision_scores = {
        str(CLASSES[i]): round(float(svc_scores[i]), 4)
        for i in order
    }

    top_features = get_top_features_for_doc(X_tfidf, top_idx, top_k=14)
    top_keywords = [feat[0] for feat in top_features]

    return {
        "predicted_class": top_class,
        "runner_up_class": runner_up_class,
        "decision_margin": decision_margin,
        "is_anonymous": is_anonymous,
        "final_destination": final_destination,
        "model": "TF-IDF + Structural Features + LinearSVC",
        "top_score": round(top_score, 4),
        "runner_up_score": round(runner_up_score, 4),
        "decision_scores": decision_scores,
        "top_keywords": top_keywords,
        "top_features": top_features,
        "text_length": text_length,
    }


def get_model_metadata() -> Dict[str, Any]:
    """Returns metadata for the active production model."""
    return {
        "model": "TF-IDF + Structural Features + LinearSVC",
        "classes": CLASSES,
        "provisional_margin_threshold": PROVISIONAL_MARGIN_THRESHOLD,
        "vectorizer_features": VOCAB_SIZE,
        "source": "research/taxonomy_v2_structural/models/",
        "performance_metrics": {
            "augmented_test_macro_f1": 0.9462,
            "base_v2_test_macro_f1": 0.9606,
            "v1_baseline_macro_f1": 0.9312,
        },
    }

predict_document = predict_text
