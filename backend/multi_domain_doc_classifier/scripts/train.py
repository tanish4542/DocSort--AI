#!/usr/bin/env python3
"""
Train MultinomialNB on TF-IDF + manual keyword features; save model.pkl & vectorizer.pkl.

Train/test split is performed BEFORE oversampling so duplicated bootstrap rows from
balancing cannot leak into both splits (which would inflate accuracy).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdcs.balance import hybrid_balance  # noqa: E402
from mdcs.config import (  # noqa: E402
    BALANCE_MAX_PER_CLASS,
    DOMAINS,
    MODELS_DIR,
    RANDOM_STATE,
)
from mdcs.dataset_loader import build_master_dataframe  # noqa: E402
from mdcs.features import build_tfidf_vectorizer, hstack_tfidf_keywords, keyword_match_matrix  # noqa: E402
from mdcs.predictor import DomainClassifier  # noqa: E402
from mdcs.preprocess import preprocess_pipeline  # noqa: E402


def preprocess_frame(df):
    texts_raw = df["text"].astype(str).tolist()
    cleaned = []
    kept_idx = []
    for i, t in enumerate(texts_raw):
        c = preprocess_pipeline(t, min_chars=40)
        if c:
            cleaned.append(c)
            kept_idx.append(i)
    sub = df.iloc[kept_idx].reset_index(drop=True)
    sub["text_clean"] = cleaned
    sub["text_raw"] = sub["text"].astype(str)
    return sub


def featurize_train_test(df_train, df_test):
    vectorizer = build_tfidf_vectorizer(ngram_max=2, max_features=55_000, min_df=2, max_df=0.92)
    X_tfidf_tr = vectorizer.fit_transform(df_train["text_clean"])
    kw_tr = keyword_match_matrix(df_train["text_raw"], list(DOMAINS))
    X_tr = hstack_tfidf_keywords(X_tfidf_tr, kw_tr)

    X_tfidf_te = vectorizer.transform(df_test["text_clean"])
    kw_te = keyword_match_matrix(df_test["text_raw"], list(DOMAINS))
    X_te = hstack_tfidf_keywords(X_tfidf_te, kw_te)
    return vectorizer, X_tr, X_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Smaller medical/finance samples for dev")
    args = ap.parse_args()

    if args.quick:
        import mdcs.dataset_loader as dl

        dl.MAX_ABSTRACTS_MEDICAL = 2_000
        dl.MAX_ROWS_FINANCE = 4_000
        dl.MEDICAL_MAX_ROWS = 400_000

    master, reports = build_master_dataframe()
    processed_dir = MODELS_DIR.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "dataset_analysis.json", "w") as f:
        json.dump(reports, f, indent=2)

    prep = preprocess_frame(master)
    master.to_csv(processed_dir / "standardized_text_label.csv", index=False)
    prep[["text_raw", "label"]].rename(columns={"text_raw": "text"}).to_csv(
        processed_dir / "preprocessed_filtered_text_label.csv", index=False
    )

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        prep,
        prep["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=prep["label"],
    )

    X_train_bal, bal_meta = hybrid_balance(
        X_train_df.reset_index(drop=True),
        max_per_class=BALANCE_MAX_PER_CLASS,
        random_state=RANDOM_STATE,
    )
    with open(processed_dir / "balancing_meta.json", "w") as f:
        json.dump(bal_meta, f, indent=2)
    X_train_bal[["text_raw", "label"]].rename(columns={"text_raw": "text"}).to_csv(
        processed_dir / "train_balanced_text_label.csv", index=False
    )

    vec, Xtr, Xte = featurize_train_test(X_train_bal, X_test_df.reset_index(drop=True))

    grid = GridSearchCV(
        MultinomialNB(),
        param_grid={"alpha": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]},
        scoring="f1_macro",
        cv=4,
        n_jobs=1,
    )
    y_tr = np.asarray(X_train_bal["label"].tolist())
    grid.fit(Xtr, y_tr)
    model = grid.best_estimator_
    print("Best NB alpha:", grid.best_params_, "CV f1_macro:", round(grid.best_score_, 4))

    y_te = np.asarray(y_test.tolist())
    y_pred = model.predict(Xte)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")
    print("Holdout accuracy:", round(acc, 4), "f1_macro:", round(f1, 4))
    print(classification_report(y_te, y_pred, digits=3))
    cm = confusion_matrix(y_te, y_pred, labels=list(DOMAINS))
    print("Confusion matrix [rows=true, cols=pred]:\n", cm)

    clf = DomainClassifier(vec, model, domains=None, min_doc_chars=40)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf.save(MODELS_DIR)

    with open(processed_dir / "evaluation_report.json", "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "f1_macro": f1,
                "confusion_matrix": cm.tolist(),
                "labels": list(DOMAINS),
                "best_params": grid.best_params_,
            },
            f,
            indent=2,
        )

    print("\n--- Optional baselines (same features; subsampled for speed) ---")
    idx = np.random.RandomState(RANDOM_STATE).choice(Xtr.shape[0], size=min(8000, Xtr.shape[0]), replace=False)
    Xs = Xtr[idx]
    ys = y_tr[idx]

    lr = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    lr.fit(Xs, ys)
    print("LR f1_macro:", round(f1_score(y_te, lr.predict(Xte), average="macro"), 4))

    svm = LinearSVC()
    svm.fit(Xs, ys)
    print("LinearSVC accuracy:", round(accuracy_score(y_te, svm.predict(Xte)), 4))

    rf = RandomForestClassifier(n_estimators=80, max_depth=24, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(Xs, ys)
    print("RF f1_macro:", round(f1_score(y_te, rf.predict(Xte), average="macro"), 4))


if __name__ == "__main__":
    main()
