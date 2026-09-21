#!/usr/bin/env python3
"""
run_model_experiments.py

Research Experiment Pipeline for DocSort AI:
"Does an ensemble of complementary machine learning classifiers improve the
performance and reliability of multi-domain document classification compared
with individual classifiers?"

Models:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Classifier (LinearSVC + CalibratedClassifierCV)
4. Soft-Voting Ensemble (MultinomialNB + LogisticRegression + Calibrated LinearSVC)

Experimental Rules:
- NO DATA LEAKAGE: TF-IDF fitted strictly on TRAIN data only.
- Validation set used strictly for hyperparameter selection and ensemble weight tuning.
- Test set evaluated exactly ONCE at the end on finalized models.
- Explainable AI feature-attribution generated from real model weights.
- All models, metrics, and figures saved under 'research/'.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

RANDOM_STATE = 42

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "research" / "splits"
RESEARCH_DIR = BASE_DIR / "research"
MODELS_DIR = RESEARCH_DIR / "models"
RESULTS_DIR = RESEARCH_DIR / "results"
EXPLANATIONS_DIR = RESEARCH_DIR / "explanations"

TARGET_CLASSES = [
    "Business and Finance",
    "Entertainment",
    "Medical Health",
    "Science",
    "Sports",
    "Technology & Computing",
]


def setup_directories() -> None:
    for d in (MODELS_DIR, RESULTS_DIR, EXPLANATIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Weighted Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_confusion_matrix_plot(
    cm: np.ndarray, labels: list[str], title: str, save_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    start_time_all = time.time()
    setup_directories()

    print("=" * 80)
    print("DOCSORT AI — RESEARCH EXPERIMENT: MULTI-MODEL & ENSEMBLE BENCHMARK")
    print("=" * 80)

    # 1. LOAD SPLITS
    print("\n1. Loading Data Splits...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"  - Train:      {len(train_df):,} documents")
    print(f"  - Validation: {len(val_df):,} documents")
    print(f"  - Test:       {len(test_df):,} documents")

    classes = sorted(train_df["label"].unique().tolist())
    print(f"  - Classes ({len(classes)}): {classes}")

    # 2. FIT TF-IDF STRICTLY ON TRAIN ONLY
    print("\n2. Fitting TF-IDF Vectorizer (Strictly on Train Split)...")
    tfidf_config = {
        "lowercase": True,
        "strip_accents": "unicode",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
        "max_features": 100000,
    }
    vec = TfidfVectorizer(**tfidf_config)

    t0 = time.time()
    X_train = vec.fit_transform(train_df["text"])
    t_tfidf = time.time() - t0
    feature_count = len(vec.vocabulary_)
    print(f"  - TF-IDF fitted in {t_tfidf:.2f}s")
    print(f"  - Total Vocabulary Size: {feature_count:,} features")

    # Transform Validation and Test (Transform only!)
    X_val = vec.transform(val_df["text"])
    X_test = vec.transform(test_df["text"])
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_val shape:   {X_val.shape}")
    print(f"  - X_test shape:  {X_test.shape}")

    # Save vectorizer
    joblib.dump(vec, MODELS_DIR / "tfidf_vectorizer.joblib")
    print(f"  - Saved vectorizer to: {MODELS_DIR / 'tfidf_vectorizer.joblib'}")

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    training_times: dict[str, float] = {}

    # 3. MODEL 1: MULTINOMIAL NAIVE BAYES
    print("\n3. Training & Tuning Model 1: Multinomial Naive Bayes...")
    nb_alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    best_nb_alpha = 0.1
    best_nb_val_f1 = -1.0
    best_nb_model = None

    for alpha in nb_alphas:
        nb = MultinomialNB(alpha=alpha)
        t_start = time.time()
        nb.fit(X_train, y_train)
        fit_t = time.time() - t_start
        val_preds = nb.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="macro")
        print(f"  [NB alpha={alpha:<4}] Validation Macro F1 = {val_f1:.4f} (fit: {fit_t:.3f}s)")
        if val_f1 > best_nb_val_f1:
            best_nb_val_f1 = val_f1
            best_nb_alpha = alpha
            best_nb_model = nb
            training_times["MultinomialNB"] = fit_t

    print(f"  => Best NB Alpha selected on Validation: {best_nb_alpha} (Macro F1 = {best_nb_val_f1:.4f})")
    joblib.dump(best_nb_model, MODELS_DIR / "multinomial_nb.joblib")

    # 4. MODEL 2: LOGISTIC REGRESSION
    print("\n4. Training & Tuning Model 2: Logistic Regression...")
    lr_c_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_lr_c = 1.0
    best_lr_val_f1 = -1.0
    best_lr_model = None

    for c_val in lr_c_values:
        lr = LogisticRegression(C=c_val, max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1)
        t_start = time.time()
        lr.fit(X_train, y_train)
        fit_t = time.time() - t_start
        val_preds = lr.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="macro")
        print(f"  [LR C={c_val:<4}] Validation Macro F1 = {val_f1:.4f} (fit: {fit_t:.2f}s)")
        if val_f1 > best_lr_val_f1:
            best_lr_val_f1 = val_f1
            best_lr_c = c_val
            best_lr_model = lr
            training_times["LogisticRegression"] = fit_t

    print(f"  => Best LR C selected on Validation: {best_lr_c} (Macro F1 = {best_lr_val_f1:.4f})")
    joblib.dump(best_lr_model, MODELS_DIR / "logistic_regression.joblib")

    # 5. MODEL 3: LINEAR SVC & CALIBRATED LINEAR SVC
    print("\n5. Training & Tuning Model 3: Linear SVC...")
    svc_c_values = [0.05, 0.1, 0.5, 1.0, 2.0]
    best_svc_c = 0.5
    best_svc_val_f1 = -1.0
    best_svc_model = None

    for c_val in svc_c_values:
        svc = LinearSVC(C=c_val, random_state=RANDOM_STATE)
        t_start = time.time()
        svc.fit(X_train, y_train)
        fit_t = time.time() - t_start
        val_preds = svc.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="macro")
        print(f"  [LinearSVC C={c_val:<4}] Validation Macro F1 = {val_f1:.4f} (fit: {fit_t:.2f}s)")
        if val_f1 > best_svc_val_f1:
            best_svc_val_f1 = val_f1
            best_svc_c = c_val
            best_svc_model = svc
            training_times["LinearSVC"] = fit_t

    print(f"  => Best LinearSVC C selected on Validation: {best_svc_c} (Macro F1 = {best_svc_val_f1:.4f})")
    joblib.dump(best_svc_model, MODELS_DIR / "linear_svc.joblib")

    # Calibrate LinearSVC for probability output (TRAIN SPLIT ONLY via 5-fold internal CV)
    print("  - Calibrating LinearSVC with 5-fold internal CV on Train data only...")
    t_start = time.time()
    calibrated_svc = CalibratedClassifierCV(
        estimator=LinearSVC(C=best_svc_c, random_state=RANDOM_STATE),
        cv=5,
    )
    calibrated_svc.fit(X_train, y_train)
    training_times["CalibratedLinearSVC"] = time.time() - t_start
    print(f"  - Calibrated LinearSVC fit in {training_times['CalibratedLinearSVC']:.2f}s")
    joblib.dump(calibrated_svc, MODELS_DIR / "calibrated_linear_svc.joblib")

    # 6. MODEL 4: PROPOSED SOFT-VOTING ENSEMBLE
    print("\n6. Building Model 4: Soft-Voting Ensemble (NB + LR + Calibrated LinearSVC)...")
    prob_val_nb = best_nb_model.predict_proba(X_val)
    prob_val_lr = best_lr_model.predict_proba(X_val)
    prob_val_svc = calibrated_svc.predict_proba(X_val)

    # Class order alignment check
    assert list(best_nb_model.classes_) == classes
    assert list(best_lr_model.classes_) == classes
    assert list(calibrated_svc.classes_) == classes

    # Baseline: Equal weights
    prob_val_ens_equal = (prob_val_nb + prob_val_lr + prob_val_svc) / 3.0
    val_pred_ens_equal = [classes[i] for i in np.argmax(prob_val_ens_equal, axis=1)]
    f1_ens_equal = f1_score(y_val, val_pred_ens_equal, average="macro")
    print(f"  - Equal Weights [1/3, 1/3, 1/3] Validation Macro F1 = {f1_ens_equal:.4f}")

    # Validation Grid Search for Optimal Weights
    best_weights = (1 / 3, 1 / 3, 1 / 3)
    best_val_ens_f1 = f1_ens_equal
    weight_candidates = []
    for w1 in np.linspace(0.1, 0.8, 8):
        for w2 in np.linspace(0.1, 0.8, 8):
            for w3 in np.linspace(0.1, 0.8, 8):
                total = w1 + w2 + w3
                weight_candidates.append((w1 / total, w2 / total, w3 / total))

    for w1, w2, w3 in set(weight_candidates):
        p_blend = w1 * prob_val_nb + w2 * prob_val_lr + w3 * prob_val_svc
        preds = [classes[i] for i in np.argmax(p_blend, axis=1)]
        f1_blend = f1_score(y_val, preds, average="macro")
        if f1_blend > best_val_ens_f1:
            best_val_ens_f1 = f1_blend
            best_weights = (round(w1, 3), round(w2, 3), round(w3, 3))

    print(f"  => Optimal Weights on Validation: NB={best_weights[0]}, LR={best_weights[1]}, SVC={best_weights[2]} (Macro F1 = {best_val_ens_f1:.4f})")

    # 7. EVALUATION ON VALIDATION SET
    print("\n7. Comprehensive Evaluation on Validation Set...")
    val_models = {
        "MultinomialNB": best_nb_model,
        "LogisticRegression": best_lr_model,
        "LinearSVC": best_svc_model,
        "CalibratedLinearSVC": calibrated_svc,
    }

    val_predictions = {}
    for name, model in val_models.items():
        val_predictions[name] = model.predict(X_val)

    # Ensemble predictions on Validation
    val_prob_ens = (
        best_weights[0] * prob_val_nb
        + best_weights[1] * prob_val_lr
        + best_weights[2] * prob_val_svc
    )
    val_predictions["SoftVotingEnsemble"] = np.array(
        [classes[i] for i in np.argmax(val_prob_ens, axis=1)]
    )

    val_metrics_rows = []
    for name, preds in val_predictions.items():
        m = calculate_metrics(y_val, preds, classes)
        m["Model"] = name
        m["Training Time (s)"] = round(training_times.get(name, sum(training_times.values())), 3)
        val_metrics_rows.append(m)

        # Save classification report
        rep = classification_report(y_val, preds, digits=4)
        with open(RESULTS_DIR / f"val_classification_report_{name}.txt", "w") as f:
            f.write(f"Validation Classification Report — {name}\n")
            f.write("=" * 60 + "\n")
            f.write(rep)

        # Confusion Matrix
        cm = confusion_matrix(y_val, preds, labels=classes)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            RESULTS_DIR / f"val_confusion_matrix_{name}.csv"
        )
        save_confusion_matrix_plot(
            cm,
            classes,
            f"Validation Confusion Matrix — {name}",
            RESULTS_DIR / f"val_confusion_matrix_{name}.png",
        )

    val_metrics_df = pd.DataFrame(val_metrics_rows)[
        [
            "Model",
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1",
            "Training Time (s)",
        ]
    ]
    val_metrics_df.to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)
    print("\nValidation Metrics Table:")
    print(val_metrics_df.to_string(index=False))

    # 8. FINAL TEST SET EVALUATION (EVALUATED EXACTLY ONCE!)
    print("\n" + "=" * 80)
    print("8. FINAL UNBIASED TEST EVALUATION (One-time run on Test Split)")
    print("=" * 80)
    test_prob_nb = best_nb_model.predict_proba(X_test)
    test_prob_lr = best_lr_model.predict_proba(X_test)
    test_prob_svc = calibrated_svc.predict_proba(X_test)

    test_predictions = {
        "MultinomialNB": best_nb_model.predict(X_test),
        "LogisticRegression": best_lr_model.predict(X_test),
        "LinearSVC": best_svc_model.predict(X_test),
        "CalibratedLinearSVC": calibrated_svc.predict(X_test),
    }

    test_prob_ens = (
        best_weights[0] * test_prob_nb
        + best_weights[1] * test_prob_lr
        + best_weights[2] * test_prob_svc
    )
    test_predictions["SoftVotingEnsemble"] = np.array(
        [classes[i] for i in np.argmax(test_prob_ens, axis=1)]
    )

    test_metrics_rows = []
    test_cm_dict = {}
    test_reports = {}

    for name, preds in test_predictions.items():
        m = calculate_metrics(y_test, preds, classes)
        m["Model"] = name
        test_metrics_rows.append(m)

        # Classification report
        rep = classification_report(y_test, preds, digits=4)
        test_reports[name] = rep
        with open(RESULTS_DIR / f"test_classification_report_{name}.txt", "w") as f:
            f.write(f"Test Classification Report — {name}\n")
            f.write("=" * 60 + "\n")
            f.write(rep)

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds, labels=classes)
        test_cm_dict[name] = cm
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            RESULTS_DIR / f"test_confusion_matrix_{name}.csv"
        )
        save_confusion_matrix_plot(
            cm,
            classes,
            f"Test Confusion Matrix — {name}",
            RESULTS_DIR / f"test_confusion_matrix_{name}.png",
        )

    test_metrics_df = pd.DataFrame(test_metrics_rows)[
        [
            "Model",
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1",
        ]
    ]
    test_metrics_df.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    print("\nFINAL TEST METRICS TABLE:")
    print(test_metrics_df.to_string(index=False))

    # 9. EXPLAINABLE AI MODULE (FEATURE-WEIGHT ATTRIBUTION)
    print("\n" + "=" * 80)
    print("9. EXPLAINABLE AI MODULE: Mathematical Feature Attribution")
    print("=" * 80)
    # Feature attribution for Logistic Regression and LinearSVC using model weights
    feature_names = np.array(vec.get_feature_names_out())
    lr_coefs = best_lr_model.coef_  # shape: (n_classes, n_features)
    svc_coefs = best_svc_model.coef_  # shape: (n_classes, n_features)

    # Pick an illustrative document from the test set for each domain
    explanation_records = []
    sample_indices = []
    for c in classes:
        idx_match = np.where(y_test == c)[0]
        if len(idx_match) > 0:
            sample_indices.append(int(idx_match[0]))

    for doc_idx in sample_indices:
        doc_text = test_df.iloc[doc_idx]["text"]
        true_label = test_df.iloc[doc_idx]["label"]
        doc_url = test_df.iloc[doc_idx]["url"]
        doc_vec = X_test[doc_idx]  # 1 x 100000 sparse

        # Model predictions
        lr_probs = test_prob_lr[doc_idx]
        pred_lr = classes[int(np.argmax(lr_probs))]
        ens_probs = test_prob_ens[doc_idx]
        pred_ens = classes[int(np.argmax(ens_probs))]

        # Rank classes
        sorted_class_idx = np.argsort(lr_probs)[::-1]
        top1_c_idx = sorted_class_idx[0]
        top2_c_idx = sorted_class_idx[1]
        top1_name = classes[top1_c_idx]
        top2_name = classes[top2_c_idx]

        # Calculate element-wise contribution: w_c,j * tfidf_j
        doc_indices = doc_vec.indices
        doc_values = doc_vec.data

        # Top 1 class evidence
        top1_weights = lr_coefs[top1_c_idx, doc_indices]
        top1_contributions = top1_weights * doc_values
        top1_sorted = np.argsort(top1_contributions)[::-1]

        top_supporting = [
            (feature_names[doc_indices[i]], round(float(top1_contributions[i]), 4))
            for i in top1_sorted[:10]
            if top1_contributions[i] > 0
        ]

        # Top 2 (competing) class evidence
        top2_weights = lr_coefs[top2_c_idx, doc_indices]
        top2_contributions = top2_weights * doc_values
        top2_sorted = np.argsort(top2_contributions)[::-1]

        competing_supporting = [
            (feature_names[doc_indices[i]], round(float(top2_contributions[i]), 4))
            for i in top2_sorted[:10]
            if top2_contributions[i] > 0
        ]

        explanation_records.append({
            "doc_index": doc_idx,
            "url": doc_url,
            "true_label": true_label,
            "predicted_label_lr": pred_lr,
            "lr_confidence": round(float(lr_probs[top1_c_idx]) * 100, 2),
            "predicted_label_ensemble": pred_ens,
            "ensemble_confidence": round(float(ens_probs[top1_c_idx]) * 100, 2),
            "top_supporting_features": top_supporting,
            "competing_class": top2_name,
            "competing_confidence": round(float(lr_probs[top2_c_idx]) * 100, 2),
            "competing_supporting_features": competing_supporting,
            "text_snippet": doc_text[:350] + "...",
        })

    with open(EXPLANATIONS_DIR / "sample_document_explanations.json", "w") as f:
        json.dump(explanation_records, f, indent=2)

    with open(EXPLANATIONS_DIR / "explanation_summary.txt", "w") as f:
        f.write("DOCSORT AI — EXPLAINABLE AI FEATURE-WEIGHT ATTRIBUTION\n")
        f.write("=" * 70 + "\n\n")
        for rec in explanation_records:
            f.write(f"Sample Document #{rec['doc_index']}\n")
            f.write(f"URL:             {rec['url']}\n")
            f.write(f"True Label:      {rec['true_label']}\n")
            f.write(f"Predicted LR:    {rec['predicted_label_lr']} ({rec['lr_confidence']}%\n")
            f.write(f"Predicted Ens:   {rec['predicted_label_ensemble']} ({rec['ensemble_confidence']}%)\n")
            f.write(f"Top Supporting Features (w_ij * x_j):\n")
            for term, score in rec["top_supporting_features"]:
                f.write(f"    * {term:25s} : +{score:.4f}\n")
            f.write(f"Competing Class: {rec['competing_class']} ({rec['competing_confidence']}%)\n")
            f.write(f"Competing Supporting Features:\n")
            for term, score in rec["competing_supporting_features"]:
                f.write(f"    * {term:25s} : +{score:.4f}\n")
            f.write(f"Text Snippet:\n    \"{rec['text_snippet']}\"\n")
            f.write("-" * 70 + "\n\n")

    print(f"  - Generated explanations for {len(explanation_records)} domain test cases.")
    print(f"  - Saved to: {EXPLANATIONS_DIR / 'explanation_summary.txt'}")

    # 10. SAVE EXPERIMENT CONFIGURATION
    print("\n10. Saving Complete Experiment Configuration...")
    config = {
        "research_question": "Does an ensemble of complementary machine learning classifiers improve the performance and reliability of multi-domain document classification compared with individual classifiers?",
        "dataset": "mdonigian/iab-news-classification (balanced subset)",
        "classes": classes,
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "random_seed": RANDOM_STATE,
        "tfidf_config": tfidf_config,
        "vocabulary_size": feature_count,
        "models": {
            "MultinomialNB": {"best_alpha": best_nb_alpha},
            "LogisticRegression": {"best_C": best_lr_c, "max_iter": 2000},
            "LinearSVC": {"best_C": best_svc_c},
            "CalibratedLinearSVC": {"estimator_C": best_svc_c, "cv": 5},
            "SoftVotingEnsemble": {
                "components": ["MultinomialNB", "LogisticRegression", "CalibratedLinearSVC"],
                "weights": {
                    "MultinomialNB": best_weights[0],
                    "LogisticRegression": best_weights[1],
                    "CalibratedLinearSVC": best_weights[2],
                },
            },
        },
        "training_times_seconds": training_times,
        "validation_metrics": val_metrics_df.to_dict(orient="records"),
        "test_metrics": test_metrics_df.to_dict(orient="records"),
        "leakage_safeguards": {
            "tfidf_fitted_on_train_only": True,
            "test_set_evaluated_once": True,
            "no_synthetic_oversampling": True,
        },
    }

    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    total_time = time.time() - start_time_all
    print(f"\nExperiment execution fully completed in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
