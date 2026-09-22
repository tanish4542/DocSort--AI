"""
backend/research_inference.py

Dedicated inference module for the final DocSort AI research model:
- LinearSVC(C=0.5)
- TF-IDF Vectorizer (100,000 unigram/bigram features)
- 6 Balanced Classes:
    1. Business and Finance
    2. Entertainment
    3. Medical Health
    4. Science
    5. Sports
    6. Technology & Computing

Rules:
- Models are loaded strictly once at startup using joblib.
- Production inference uses vectorizer.transform() ONLY (no fitting).
- Decision margins and uncertainty levels are calculated from LinearSVC decision scores.
- Feature attribution is derived mathematically from document TF-IDF values and model coefficients:
    contribution_j = x_j * weight[c, j]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

logger = logging.getLogger("docsort.research_inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Resolve paths to research models
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
RESEARCH_MODELS_DIR = PROJECT_ROOT / "research" / "models"

VEC_PATH = RESEARCH_MODELS_DIR / "tfidf_vectorizer.joblib"
SVC_PATH = RESEARCH_MODELS_DIR / "linear_svc.joblib"
NB_PATH = RESEARCH_MODELS_DIR / "multinomial_nb.joblib"
LR_PATH = RESEARCH_MODELS_DIR / "logistic_regression.joblib"

# Operational uncertainty thresholds for LinearSVC (Not calibrated probabilities)
# margin >= 1.0  -> High confidence
# 0.5 <= margin < 1.0 -> Moderate confidence
# margin < 0.5  -> Ambiguous (manual review recommended)
THRESHOLD_HIGH = 1.0
THRESHOLD_MODERATE = 0.5

# Validation: ensure files exist
for artifact_path, name in [
    (VEC_PATH, "TF-IDF vectorizer"),
    (SVC_PATH, "LinearSVC model"),
    (NB_PATH, "MultinomialNB model"),
    (LR_PATH, "LogisticRegression model"),
]:
    if not artifact_path.is_file():
        raise RuntimeError(
            f"Critical startup error: {name} artifact not found at {artifact_path}. "
            f"Do not fall back to legacy models."
        )

# Fixed test-set performance metrics from research experiment
MODEL_PERFORMANCE: Dict[str, Dict[str, Any]] = {
    "Multinomial Naive Bayes": {
        "accuracy": 0.9116,
        "macro_f1": 0.9117,
        "model_type": "Probabilistic Classifier",
    },
    "Logistic Regression": {
        "accuracy": 0.9406,
        "macro_f1": 0.9405,
        "model_type": "Linear Classifier",
    },
    "LinearSVC": {
        "accuracy": 0.9450,
        "macro_f1": 0.9449,
        "model_type": "Support Vector Classifier (Selected)",
    },
}

ENSEMBLE_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "Soft Voting": {
        "macro_f1": 0.9415,
        "accuracy": 0.9416,
        "notes": "Probability-averaged voting ensemble combining base classifiers.",
    },
    "Stacking": {
        "macro_f1": 0.9447,
        "accuracy": 0.9447,
        "notes": "Meta-classifier trained on base model prediction representations.",
    },
    "LinearSVC": {
        "macro_f1": 0.9449,
        "accuracy": 0.9450,
        "notes": "Selected final model (highest Macro F1 among all evaluated models).",
    },
}

# Load model artifacts once at startup
try:
    vectorizer = joblib.load(VEC_PATH)
    linear_svc = joblib.load(SVC_PATH)
    multinomial_nb = joblib.load(NB_PATH)
    logistic_regression = joblib.load(LR_PATH)

    CLASSES = list(linear_svc.classes_)
    FEATURE_NAMES = np.array(vectorizer.get_feature_names_out())
    VOCAB_SIZE = len(vectorizer.vocabulary_)

    print("=" * 70)
    print("DOCSORT AI — 3-MODEL RESEARCH INFERENCE MODULE INITIALIZED")
    print("=" * 70)
    print(f"Research models loaded successfully from: {RESEARCH_MODELS_DIR}")
    print(f"Models: MultinomialNB, LogisticRegression, LinearSVC (Selected)")
    print(f"Classes ({len(CLASSES)}): {CLASSES}")
    print(f"TF-IDF vocabulary size: {VOCAB_SIZE:,} features")
    print("=" * 70)
    logger.info("All 3 research classifiers and TF-IDF vectorizer ready for inference.")

except Exception as e:
    raise RuntimeError(
        f"Failed to load research model artifacts: {e}. Aborting startup."
    ) from e


def compute_uncertainty_level(margin: float) -> str:
    """
    Assign an operational uncertainty label based on the decision margin between
    the highest and second-highest decision scores.
    NOTE: These are operational heuristics for application routing, NOT calibrated probabilities.
    """
    if margin >= THRESHOLD_HIGH:
        return "High confidence"
    elif margin >= THRESHOLD_MODERATE:
        return "Moderate confidence"
    else:
        return "Ambiguous"


def get_top_features_for_doc(
    X_sparse,
    top_class_idx: int,
    top_k: int = 12
) -> List[Tuple[str, float]]:
    """
    Calculates the exact lexical feature contributions for the predicted class:
        contribution_j = x_j * weight[predicted_class, j]
    Only considers non-zero TF-IDF features present in this document.
    """
    cx = X_sparse.tocoo()
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
            # Prefer positive drivers once we have at least 3 features
            break
        word = str(FEATURE_NAMES[col_indices[idx]])
        top_features.append((word, round(contrib, 4)))
        if len(top_features) >= top_k:
            break

    return top_features


def predict_text(text: str) -> Dict[str, Any]:
    """
    Performs research inference on a raw extracted document string across ALL 3
    individual classifiers (MultinomialNB, LogisticRegression, LinearSVC).

    LinearSVC remains the actual final prediction and auto-sorting decision.
    """
    cleaned_str = str(text).strip() if text else ""
    text_length = len(cleaned_str)

    if not cleaned_str:
        # Fallback for empty/unextractable text
        return {
            "final_prediction": "Unknown",
            "selected_model": "LinearSVC",
            "selection_reason": "LinearSVC achieved the highest test Macro F1 (94.49%) among the evaluated individual classifiers.",
            "model_comparison": [],
            "model_performance": MODEL_PERFORMANCE,
            "ensemble_experiments": ENSEMBLE_EXPERIMENTS,
            "predicted_class": "Unknown",
            "runner_up_class": "Unknown",
            "decision_margin": 0.0,
            "uncertainty_level": "Ambiguous",
            "top_score": 0.0,
            "runner_up_score": 0.0,
            "decision_scores": {c: 0.0 for c in CLASSES},
            "top_keywords": [],
            "top_features": [],
            "top_two_domains": [],
            "text_length": 0,
        }

    # 1. Transform raw text through pre-fitted research TF-IDF vectorizer (100k features)
    X = vectorizer.transform([cleaned_str])

    # 2. Classifier 1: Multinomial Naive Bayes (Probabilistic)
    nb_probas = multinomial_nb.predict_proba(X)[0]
    nb_top_idx = int(np.argmax(nb_probas))
    nb_pred = str(multinomial_nb.classes_[nb_top_idx])
    nb_score = round(float(nb_probas[nb_top_idx]), 4)

    # 3. Classifier 2: Logistic Regression (Linear Probabilistic)
    lr_probas = logistic_regression.predict_proba(X)[0]
    lr_top_idx = int(np.argmax(lr_probas))
    lr_pred = str(logistic_regression.classes_[lr_top_idx])
    lr_score = round(float(lr_probas[lr_top_idx]), 4)

    # 4. Classifier 3: LinearSVC (Maximum-Margin Hyperplane — SELECTED FINAL MODEL)
    svc_scores = linear_svc.decision_function(X)[0]
    order = np.argsort(svc_scores)[::-1]
    top_idx = int(order[0])
    runner_up_idx = int(order[1])

    top_class = str(CLASSES[top_idx])
    runner_up_class = str(CLASSES[runner_up_idx])

    top_score = float(svc_scores[top_idx])
    runner_up_score = float(svc_scores[runner_up_idx])

    decision_margin = round(top_score - runner_up_score, 4)
    uncertainty_level = compute_uncertainty_level(decision_margin)

    # All LinearSVC class decision scores (sorted descending)
    decision_scores = {
        str(CLASSES[i]): round(float(svc_scores[i]), 4)
        for i in order
    }

    # Exact linear feature attribution for LinearSVC top class
    top_features = get_top_features_for_doc(X, top_idx, top_k=14)
    top_keywords = [feat[0] for feat in top_features]

    # Model comparison table data for UI
    model_comparison = [
        {
            "model": "Multinomial Naive Bayes",
            "prediction": nb_pred,
            "score": nb_score,
            "score_type": "Probability",
            "score_display": f"{round(nb_score * 100, 1)}% probability",
            "test_accuracy": 0.9116,
            "test_macro_f1": 0.9117,
            "is_selected": False,
        },
        {
            "model": "Logistic Regression",
            "prediction": lr_pred,
            "score": lr_score,
            "score_type": "Probability",
            "score_display": f"{round(lr_score * 100, 1)}% probability",
            "test_accuracy": 0.9406,
            "test_macro_f1": 0.9405,
            "is_selected": False,
        },
        {
            "model": "LinearSVC",
            "prediction": top_class,
            "score": round(top_score, 4),
            "score_type": "Decision Score",
            "score_display": f"{round(top_score, 2)} decision score (margin: {decision_margin:.2f})",
            "decision_margin": decision_margin,
            "uncertainty_level": uncertainty_level,
            "test_accuracy": 0.9450,
            "test_macro_f1": 0.9449,
            "is_selected": True,
        },
    ]

    return {
        "final_prediction": top_class,
        "selected_model": "LinearSVC",
        "selection_reason": "LinearSVC achieved the highest test Macro F1 (94.49%) among the evaluated individual classifiers.",
        "model_comparison": model_comparison,
        "model_performance": MODEL_PERFORMANCE,
        "ensemble_experiments": ENSEMBLE_EXPERIMENTS,
        "predicted_class": top_class,
        "runner_up_class": runner_up_class,
        "decision_margin": decision_margin,
        "uncertainty_level": uncertainty_level,
        "top_score": round(top_score, 4),
        "runner_up_score": round(runner_up_score, 4),
        "decision_scores": decision_scores,
        "top_keywords": top_keywords,
        "top_features": top_features,
        "top_two_domains": [
            {"domain": top_class, "score": round(top_score, 4)},
            {"domain": runner_up_class, "score": round(runner_up_score, 4)},
        ],
        "text_length": text_length,
    }


def get_model_metadata() -> Dict[str, Any]:
    """
    Returns architecture and training metadata for the active research model.
    Used by the GET /model-info health endpoint.
    """
    return {
        "model": "LinearSVC",
        "C": float(getattr(linear_svc, "C", 0.5)),
        "loss": str(getattr(linear_svc, "loss", "squared_hinge")),
        "classes": CLASSES,
        "vectorizer_features": VOCAB_SIZE,
        "source": "research/models/",
        "research_model": True,
        "performance_metrics": {
            "test_accuracy": 0.9450,
            "test_macro_f1": 0.9449,
            "test_support": 3870,
        },
        "models_evaluated": MODEL_PERFORMANCE,
        "ensemble_experiments": ENSEMBLE_EXPERIMENTS,
    }
