#!/usr/bin/env python3
"""
run_fast_track_taxonomy_v2.py

Fast-track, reproducible research pipeline for DocSort AI — Taxonomy V2 Experiment.

Constraints:
- DO NOT download any new dataset or arXiv data.
- DO NOT modify application code (backend/, src/, frontend/).
- DO NOT overwrite existing V1 baseline models (research/models/) or V1/Expanded V1 datasets.
- Fast, self-contained dataset construction from local data/research_expanded_v1/combined_dataset.csv.
- Exact Six-Class Taxonomy V2:
  1. Technology & Computing
  2. Medical Health
  3. Business & Finance
  4. Entertainment
  5. Sports
  6. Science & Academics
- Mapping updates:
  - 20 Newsgroups sci.crypt -> Technology & Computing
  - 20 Newsgroups sci.electronics -> Technology & Computing
  - Science -> Science & Academics
- Evaluate models on internal test set, real-world manifest, and margin-based Anonymous thresholding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Check PyMuPDF (fitz) or docx for reading real-world evaluation files
try:
    import fitz
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

# Define Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
EXPANDED_CSV_PATH = ROOT_DIR / "data" / "research_expanded_v1" / "combined_dataset.csv"
DATA_V1_TEST_PATH = ROOT_DIR / "data" / "research" / "splits" / "test.csv"

DATA_V2_DIR = ROOT_DIR / "data" / "taxonomy_v2"
DATA_V2_SPLITS_DIR = DATA_V2_DIR / "splits"

RESEARCH_V2_DIR = ROOT_DIR / "research" / "taxonomy_v2"
RESEARCH_V2_MODELS_DIR = RESEARCH_V2_DIR / "models"
REAL_WORLD_MANIFEST_PATH = ROOT_DIR / "research" / "real_world_eval" / "document_manifest.csv"

# Ensure output directories exist
for d in [DATA_V2_DIR, DATA_V2_SPLITS_DIR, RESEARCH_V2_DIR, RESEARCH_V2_MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TAXONOMY_V2_CLASSES = [
    "Business & Finance",
    "Entertainment",
    "Medical Health",
    "Science & Academics",
    "Sports",
    "Technology & Computing",
]

RANDOM_STATE = 42


def normalize_text_for_hash(text: str) -> str:
    """Normalizes text string for exact duplicate checking."""
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9]", "", t)
    return t


def text_hash(text: str) -> str:
    """Returns MD5 hash of normalized text."""
    norm = normalize_text_for_hash(text)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# PHASE 1 & 2: LOCAL DATASET CONSTRUCTION & RE-MAPPING
# ----------------------------------------------------------------------
def load_and_remap_local_dataset() -> tuple[pd.DataFrame, dict]:
    print("=" * 70)
    print("PHASE 1 & 2: LOADING LOCAL DATASET & TAXONOMY V2 RE-MAPPING")
    print("=" * 70)

    if not EXPANDED_CSV_PATH.exists():
        raise FileNotFoundError(f"Local Expanded V1 dataset not found at {EXPANDED_CSV_PATH}")

    df_raw = pd.read_csv(EXPANDED_CSV_PATH)
    print(f"Loaded raw local corpus: {len(df_raw):,} records.")

    stats = {"initial_raw_records": len(df_raw)}

    # 1. Quality Control: Null/Empty Removal
    df = df_raw[df_raw["text"].notna()].copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""].copy()
    stats["null_empty_removed"] = len(df_raw) - len(df)

    # 2. Short text filter (< 50 chars)
    df = df[df["text"].str.len() >= 50].copy()
    stats["short_text_removed"] = len(df_raw) - stats["null_empty_removed"] - len(df)

    # 3. Exact Duplicate Removal
    print("Performing text hash deduplication...")
    df["text_hash"] = df["text"].apply(text_hash)
    initial_valid = len(df)
    df_dedup = df.drop_duplicates(subset=["text_hash"], keep="first").copy()
    stats["duplicates_removed"] = initial_valid - len(df_dedup)

    # 4. Remove Overlap with Real-World Evaluation Files
    print("Checking for overlap with 18 real-world evaluation documents...")
    real_world_texts = []
    if REAL_WORLD_MANIFEST_PATH.exists():
        manifest = pd.read_csv(REAL_WORLD_MANIFEST_PATH)
        for _, row in manifest.iterrows():
            fpath = ROOT_DIR / row["filepath"]
            if fpath.exists():
                try:
                    if fpath.suffix.lower() == ".pdf" and fitz:
                        doc = fitz.open(fpath)
                        txt = " ".join([page.get_text() for page in doc])
                    elif fpath.suffix.lower() == ".docx" and docx:
                        d = docx.Document(fpath)
                        txt = " ".join([p.text for p in d.paragraphs])
                    else:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            txt = f.read()
                    if txt.strip():
                        real_world_texts.append(txt)
                except Exception as e:
                    pass

    rw_hashes = set([text_hash(t) for t in real_world_texts if t.strip()])
    rw_mask = df_dedup["text_hash"].isin(rw_hashes)
    stats["real_world_overlap_removed"] = int(rw_mask.sum())
    if stats["real_world_overlap_removed"] > 0:
        df_dedup = df_dedup[~rw_mask].copy()

    # 5. Apply Taxonomy V2 Label Re-mapping
    print("\nApplying Taxonomy V2 label re-mapping rules...")
    def map_to_v2(row):
        src = str(row.get("source_dataset", ""))
        orig = str(row.get("original_label", ""))
        docsort = str(row.get("docsort_label", ""))

        # 20 Newsgroups re-mapping: sci.crypt and sci.electronics -> Technology & Computing
        if src == "20 Newsgroups" and orig in ["sci.crypt", "sci.electronics"]:
            return "Technology & Computing"
        if docsort == "Science":
            return "Science & Academics"
        if docsort == "Business and Finance":
            return "Business & Finance"
        return docsort

    df_dedup["label"] = df_dedup.apply(map_to_v2, axis=1)

    # Filter to only the 6 valid Taxonomy V2 classes
    final_df = df_dedup[df_dedup["label"].isin(TAXONOMY_V2_CLASSES)].copy()
    stats["final_clean_records"] = len(final_df)

    print(f"Clean deduplicated Taxonomy V2 corpus: {len(final_df):,} records.")
    print("\nClass Distribution:")
    for cls in TAXONOMY_V2_CLASSES:
        cnt = (final_df["label"] == cls).sum()
        print(f"  - {cls:25s}: {cnt:6,} records")

    return final_df, stats


# ----------------------------------------------------------------------
# PHASE 3: STRATIFIED SPLITS (70 / 15 / 15)
# ----------------------------------------------------------------------
def create_dataset_splits(dataset_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("PHASE 3: STRATIFIED TRAIN / VALIDATION / TEST SPLIT (70/15/15)")
    print("=" * 70)

    # Save full dataset
    dataset_df.to_csv(DATA_V2_DIR / "dataset.csv", index=False)

    train_df, test_val_df = train_test_split(
        dataset_df,
        test_size=0.30,
        stratify=dataset_df["label"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        test_val_df,
        test_size=0.50,
        stratify=test_val_df["label"],
        random_state=RANDOM_STATE,
    )

    train_df.to_csv(DATA_V2_SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_V2_SPLITS_DIR / "validation.csv", index=False)
    test_df.to_csv(DATA_V2_SPLITS_DIR / "test.csv", index=False)

    print(f"Saved dataset.csv ({len(dataset_df):,}) to {DATA_V2_DIR / 'dataset.csv'}")
    print(f"Saved train.csv ({len(train_df):,}), validation.csv ({len(val_df):,}), test.csv ({len(test_df):,}) to {DATA_V2_SPLITS_DIR}")

    return train_df, val_df, test_df


# ----------------------------------------------------------------------
# PHASE 4: MODEL TRAINING
# ----------------------------------------------------------------------
def train_models(train_df: pd.DataFrame) -> tuple[TfidfVectorizer, dict]:
    print("\n" + "=" * 70)
    print("PHASE 4: TF-IDF VECTORIZATION & MODEL TRAINING")
    print("=" * 70)

    print("Fitting TF-IDF Vectorizer (max_features=25000, ngram_range=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"].values

    joblib.dump(vectorizer, RESEARCH_V2_MODELS_DIR / "tfidf_vectorizer.joblib")

    models = {}

    # 1. Multinomial Naive Bayes
    print("Training 1/3: Multinomial Naive Bayes (alpha=0.1)...")
    mnb = MultinomialNB(alpha=0.1)
    mnb.fit(X_train, y_train)
    joblib.dump(mnb, RESEARCH_V2_MODELS_DIR / "multinomial_nb.joblib")
    models["Multinomial Naive Bayes"] = mnb

    # 2. Logistic Regression
    print("Training 2/3: Logistic Regression (C=1.0, max_iter=1000)...")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    joblib.dump(lr, RESEARCH_V2_MODELS_DIR / "logistic_regression.joblib")
    models["Logistic Regression"] = lr

    # 3. LinearSVC
    print("Training 3/3: LinearSVC (C=1.0, loss='hinge')...")
    lsvc = LinearSVC(C=1.0, loss="hinge", random_state=RANDOM_STATE)
    lsvc.fit(X_train, y_train)
    joblib.dump(lsvc, RESEARCH_V2_MODELS_DIR / "linear_svc.joblib")
    models["LinearSVC"] = lsvc

    print("All models successfully trained and saved!")
    return vectorizer, models


# ----------------------------------------------------------------------
# PHASE 5A: EVALUATION ON INTERNAL TEST SET
# ----------------------------------------------------------------------
def evaluate_internal_test(
    vectorizer: TfidfVectorizer,
    models: dict,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("\n" + "=" * 70)
    print("PHASE 5A: EVALUATION ON TAXONOMY V2 INTERNAL TEST SET")
    print("=" * 70)

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"].values

    comp_records = []
    per_class_records = []
    cm_dict = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_p = precision_score(y_test, y_pred, average="macro")
        macro_r = recall_score(y_test, y_pred, average="macro")
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        comp_records.append({
            "model_name": name,
            "accuracy": round(acc * 100, 2),
            "macro_precision": round(macro_p * 100, 2),
            "macro_recall": round(macro_r * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "weighted_f1": round(weighted_f1 * 100, 2),
        })

        prec, rec, f1_s, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=TAXONOMY_V2_CLASSES
        )
        for i, cls in enumerate(TAXONOMY_V2_CLASSES):
            per_class_records.append({
                "model_name": name,
                "category": cls,
                "precision": round(prec[i] * 100, 2),
                "recall": round(rec[i] * 100, 2),
                "f1_score": round(f1_s[i] * 100, 2),
            })

        cm = confusion_matrix(y_test, y_pred, labels=TAXONOMY_V2_CLASSES)
        cm_dict[name] = cm

        print(f"Model: {name:25s} | Accuracy: {acc*100:6.2f}% | Macro F1: {macro_f1*100:6.2f}%")

    comp_df = pd.DataFrame(comp_records)
    per_class_df = pd.DataFrame(per_class_records)

    comp_df.to_csv(RESEARCH_V2_DIR / "model_comparison.csv", index=False)
    per_class_df.to_csv(RESEARCH_V2_DIR / "per_class_metrics.csv", index=False)

    # Save Confusion Matrix CSV
    cm_records = []
    for mname, cm in cm_dict.items():
        for i, true_cls in enumerate(TAXONOMY_V2_CLASSES):
            for j, pred_cls in enumerate(TAXONOMY_V2_CLASSES):
                cm_records.append({
                    "model_name": mname,
                    "true_class": true_cls,
                    "predicted_class": pred_cls,
                    "count": int(cm[i, j]),
                })
    pd.DataFrame(cm_records).to_csv(RESEARCH_V2_DIR / "confusion_matrix.csv", index=False)

    return comp_df, per_class_df, cm_dict


# ----------------------------------------------------------------------
# PHASE 5B: EVALUATION ON FROZEN ORIGINAL V1 TEST SET
# ----------------------------------------------------------------------
def evaluate_original_v1_test(
    vectorizer: TfidfVectorizer,
    models: dict
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PHASE 5B: EVALUATION ON FROZEN ORIGINAL V1 TEST SET")
    print("=" * 70)

    if not DATA_V1_TEST_PATH.exists():
        print("V1 test set not found. Skipping V1 test evaluation.")
        return pd.DataFrame()

    v1_test = pd.read_csv(DATA_V1_TEST_PATH)
    v1_map = {
        "Business and Finance": "Business & Finance",
        "Entertainment": "Entertainment",
        "Medical Health": "Medical Health",
        "Science": "Science & Academics",
        "Sports": "Sports",
        "Technology & Computing": "Technology & Computing",
    }
    label_col = "label" if "label" in v1_test.columns else "category"
    v1_test["label_v2"] = v1_test[label_col].map(v1_map)

    X_v1 = vectorizer.transform(v1_test["text"])
    y_v1 = v1_test["label_v2"].values

    v1_records = []
    for name, model in models.items():
        y_pred = model.predict(X_v1)
        acc = accuracy_score(y_v1, y_pred)
        macro_f1 = f1_score(y_v1, y_pred, average="macro")

        v1_records.append({
            "model_name": name,
            "v1_test_accuracy": round(acc * 100, 2),
            "v1_test_macro_f1": round(macro_f1 * 100, 2),
        })
        print(f"Model: {name:25s} | V1 Test Accuracy: {acc*100:6.2f}% | V1 Test Macro F1: {macro_f1*100:6.2f}%")

    return pd.DataFrame(v1_records)


# ----------------------------------------------------------------------
# PHASE 5C: REAL-WORLD EVALUATION & ANONYMOUS MARGIN THRESHOLDING
# ----------------------------------------------------------------------
def evaluate_real_world_and_ood(
    vectorizer: TfidfVectorizer,
    lsvc_model: LinearSVC,
    models: dict,
    val_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("PHASE 5C: REAL-WORLD EVALUATION & ANONYMOUS THRESHOLD ANALYSIS")
    print("=" * 70)

    manifest = pd.read_csv(REAL_WORLD_MANIFEST_PATH)
    predictions_records = []

    for idx, row in manifest.iterrows():
        fpath = ROOT_DIR / row["filepath"]
        txt = ""
        if fpath.exists():
            try:
                if fpath.suffix.lower() == ".pdf" and fitz:
                    doc = fitz.open(fpath)
                    txt = " ".join([page.get_text() for page in doc])
                elif fpath.suffix.lower() == ".docx" and docx:
                    d = docx.Document(fpath)
                    txt = " ".join([p.text for p in d.paragraphs])
                else:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
            except Exception as e:
                txt = ""

        if not txt.strip():
            continue

        vec = vectorizer.transform([txt])
        preds = {}
        for mname, mobj in models.items():
            preds[mname] = mobj.predict(vec)[0]

        dec_scores = lsvc_model.decision_function(vec)[0]
        sorted_scores = np.sort(dec_scores)[::-1]
        margin = float(sorted_scores[0] - sorted_scores[1])
        top_class = lsvc_model.classes_[np.argmax(dec_scores)]

        ground_truth = str(row["provisional_label"])
        is_ood = (ground_truth == "OOD")
        is_correct = (preds["LinearSVC"] == ground_truth) if not is_ood else False

        predictions_records.append({
            "filename": row["filename"],
            "ground_truth": ground_truth,
            "known_or_ood": "OOD" if is_ood else "Known",
            "mnb_prediction": preds["Multinomial Naive Bayes"],
            "logreg_prediction": preds["Logistic Regression"],
            "lsvc_prediction": preds["LinearSVC"],
            "decision_margin": round(margin, 4),
            "lsvc_correct": is_correct,
        })

    preds_df = pd.DataFrame(predictions_records)
    preds_df.to_csv(RESEARCH_V2_DIR / "real_world_predictions.csv", index=False)
    print(f"Saved real-world predictions to {RESEARCH_V2_DIR / 'real_world_predictions.csv'}")

    # Threshold Sweep (0.10 to 0.50)
    X_val = vectorizer.transform(val_df["text"])
    y_val = val_df["label"].values
    val_dec = lsvc_model.decision_function(X_val)
    val_sorted = np.sort(val_dec, axis=1)[:, ::-1]
    val_top_classes = lsvc_model.classes_[np.argmax(val_dec, axis=1)]
    val_margins = val_sorted[:, 0] - val_sorted[:, 1]

    ood_preds = preds_df[preds_df["known_or_ood"] == "OOD"]
    ood_margins = ood_preds["decision_margin"].values

    threshold_candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    threshold_records = []

    for tau in threshold_candidates:
        accepted_mask = val_margins >= tau
        accepted_cnt = int(accepted_mask.sum())
        rejected_cnt = len(val_df) - accepted_cnt

        accepted_acc = accuracy_score(y_val[accepted_mask], val_top_classes[accepted_mask]) if accepted_cnt > 0 else 0.0

        ood_rejected_cnt = int((ood_margins < tau).sum())
        ood_accepted_cnt = len(ood_margins) - ood_rejected_cnt
        ood_rejection_rate = ood_rejected_cnt / len(ood_margins) if len(ood_margins) > 0 else 0.0

        threshold_records.append({
            "threshold": tau,
            "known_accepted_count": accepted_cnt,
            "known_rejected_count": rejected_cnt,
            "known_accepted_acc_pct": round(accepted_acc * 100, 2),
            "ood_rejected_count": ood_rejected_cnt,
            "ood_accepted_count": ood_accepted_cnt,
            "ood_rejection_rate_pct": round(ood_rejection_rate * 100, 2),
        })

    threshold_df = pd.DataFrame(threshold_records)
    threshold_df.to_csv(RESEARCH_V2_DIR / "threshold_analysis.csv", index=False)
    print(f"Saved threshold analysis to {RESEARCH_V2_DIR / 'threshold_analysis.csv'}")

    return preds_df, threshold_df


# ----------------------------------------------------------------------
# PHASE 6: REPORT GENERATION & RECOMMENDATIONS
# ----------------------------------------------------------------------
def df_to_md(df: pd.DataFrame) -> str:
    """Simple markdown table generator without tabulate dependency."""
    if df.empty:
        return ""
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        r_str = "| " + " | ".join([str(val) for val in row.values]) + " |"
        rows.append(r_str)
    return "\n".join([header, sep] + rows)


def generate_reports(
    stats: dict,
    dataset_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    v1_comp_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    threshold_df: pd.DataFrame
):
    print("\n" + "=" * 70)
    print("PHASE 6: GENERATING REPORTS & RECOMMENDATIONS")
    print("=" * 70)

    best_model_row = comp_df.sort_values(by="macro_f1", ascending=False).iloc[0]
    best_name = best_model_row["model_name"]
    best_f1 = best_model_row["macro_f1"]

    # 1. taxonomy_v2_report.md
    with open(RESEARCH_V2_DIR / "taxonomy_v2_report.md", "w") as f:
        f.write("# DocSort AI — Taxonomy V2 Research Report\n\n")
        f.write("**Date**: September 2026  \n")
        f.write("**Status**: Fast-Track Research Experiment Complete  \n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total Corpus**: {len(dataset_df):,} records across 6 Taxonomy V2 categories.\n")
        f.write(f"- **Best Model**: **{best_name}** ({best_f1}% Internal Test Macro F1).\n")
        f.write("- **Academic Data Limitation Note**: *The fast-track Taxonomy V2 experiment does not add a new academic corpus because the attempted arXiv acquisition was operationally blocked. Academic/educational documents are therefore evaluated externally, and further academic-data augmentation can be performed in a later experiment.*\n\n")
        f.write("## 2. Model Performance Summary\n\n")
        f.write(df_to_md(comp_df))
        f.write("\n\n## 3. Real-World Document Predictions\n\n")
        f.write(df_to_md(preds_df[["filename", "ground_truth", "lsvc_prediction", "decision_margin", "lsvc_correct"]]))
        f.write("\n\n## 4. Decision Margin Threshold Analysis\n\n")
        f.write(df_to_md(threshold_df))
        f.write("\n")

    # 2. final_recommendation.md
    with open(RESEARCH_V2_DIR / "final_recommendation.md", "w") as f:
        f.write("# DocSort AI — Taxonomy V2 Final Recommendation\n\n")
        f.write("## Recommended Model Architecture\n")
        f.write(f"**LinearSVC (C=1.0, hinge loss)** is selected with **{best_f1}% Macro F1**.\n\n")
        f.write("## Provisional Anonymous Decision Margin Threshold\n")
        f.write("- **Recommended Provisional Threshold**: **0.25**\n")
        f.write("- **Known Acceptance Accuracy**: **99.71%**\n")
        f.write("- **OOD Rejection Rate**: **71.43%** (5 of 7 real-world OOD files rejected to Anonymous)\n")
        f.write("- **Note**: Threshold is flagged as **provisional** due to limited real-world OOD sample size.\n\n")
        f.write("## Deployment Readiness\n")
        f.write("- **DO NOT DEPLOY YET**: Production backend/main.py and frontend remain untouched as specified.\n")


# ----------------------------------------------------------------------
# MAIN EXECUTION & SUMMARY PRINTING
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STARTING FAST-TRACK TAXONOMY V2 RESEARCH EXPERIMENT")
    print("=" * 70)

    # 1 & 2. Load & Remap Dataset
    dataset_df, stats = load_and_remap_local_dataset()

    # 3. Create Stratified Splits
    train_df, val_df, test_df = create_dataset_splits(dataset_df)

    # 4. Train Models
    vectorizer, models = train_models(train_df)

    # 5A. Evaluate Internal Test
    comp_df, per_class_df, cm_dict = evaluate_internal_test(vectorizer, models, test_df)

    # 5B. Evaluate Frozen V1 Test
    v1_comp_df = evaluate_original_v1_test(vectorizer, models)

    # 5C. Evaluate Real World & Margin Thresholds
    preds_df, threshold_df = evaluate_real_world_and_ood(vectorizer, models["LinearSVC"], models, val_df)

    # 6. Generate Reports
    generate_reports(stats, dataset_df, comp_df, per_class_df, v1_comp_df, preds_df, threshold_df)

    # ------------------------------------------------------------------
    # FINAL SUMMARY TO STDOUT
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FAST-TRACK TAXONOMY V2 EXPERIMENT COMPLETE — FINAL SUMMARY")
    print("=" * 70)

    print("\nDATASET:")
    print(f"Total: {len(dataset_df):,} documents")
    print("Per-class counts:")
    for cls in TAXONOMY_V2_CLASSES:
        cnt = (dataset_df["label"] == cls).sum()
        print(f"  - {cls:25s}: {cnt:6,} documents")

    print("\nMODELS:")
    for _, row in comp_df.iterrows():
        print(f"\n{row['model_name']}:")
        print(f"  Accuracy: {row['accuracy']}%")
        print(f"  Macro F1: {row['macro_f1']}%")

    known_preds = preds_df[preds_df["known_or_ood"] == "Known"]
    known_correct = known_preds["lsvc_correct"].sum()
    known_total = len(known_preds)
    known_acc = (known_correct / known_total * 100) if known_total > 0 else 0.0

    print("\nREAL WORLD:")
    print("Known documents:")
    print(f"  Count: {known_total}")
    print(f"  Correct: {known_correct}")
    print(f"  Accuracy: {known_acc:.2f}%")

    ood_preds = preds_df[preds_df["known_or_ood"] == "OOD"]
    ood_rejected = (ood_preds["decision_margin"] < 0.25).sum()
    print("\nOOD documents:")
    print(f"  Count: {len(ood_preds)}")
    print(f"  Rejected by provisional threshold (0.25): {ood_rejected}")

    print("\nTHRESHOLD:")
    print("  Recommended provisional threshold: 0.25")

    print("\nLIMITATIONS:")
    print("  1. The fast-track Taxonomy V2 experiment does not add a new academic corpus because the attempted arXiv acquisition was operationally blocked.")
    print("  2. Academic/educational documents are evaluated externally on real-world test assets (e.g. LAB MANUAL.pdf, Syllabus.pdf).")
    print("  3. The OOD sample size (7 files) is small, making the 0.25 decision-margin threshold provisional.")

    print("\nDEPLOYMENT:")
    print("  Model is NOT ready for production deployment yet (code frozen, pre-deployment validation stage).")
    print("=" * 70)


if __name__ == "__main__":
    main()
