#!/usr/bin/env python3
"""
split_research_dataset.py

Performs a leakage-safe, stratified 70 / 15 / 15 Train / Validation / Test split
on 'data/research/docsort_research_dataset.csv' for the DocSort AI research project.

Specifications:
- Total Dataset: 25,800 documents across 6 balanced classes (4,300 per class)
- Train: 70% = 18,060 documents (3,010 per class)
- Validation: 15% = 3,870 documents (645 per class)
- Test: 15% = 3,870 documents (645 per class)
- Random Seed: 42
- Stratification: Enabled on both split stages
- Overlap Checks: Zero URL or text leakage between splits
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_CLASSES = [
    "Business and Finance",
    "Medical Health",
    "Sports",
    "Technology & Computing",
    "Science",
    "Entertainment",
]

RANDOM_STATE = 42
INPUT_PATH = Path("data/research/docsort_research_dataset.csv")
SPLITS_DIR = Path("data/research/splits")


def main() -> bool:
    print("=" * 70)
    print("STEP 1 — LOAD DATASET & PRE-SPLIT VERIFICATION")
    print("=" * 70)
    if not INPUT_PATH.exists():
        print(f"ERROR: Input dataset '{INPUT_PATH}' does not exist!")
        return False

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded dataset: {INPUT_PATH} ({len(df):,} rows)")

    # Pre-split validation checks
    assert len(df) == 25800, f"Pre-check Failed: Expected 25,800 rows, got {len(df)}"
    print("Pre-check 1 PASSED: Exactly 25,800 rows")

    unique_labels = sorted(df["label"].unique().tolist())
    assert unique_labels == sorted(TARGET_CLASSES), f"Pre-check Failed: Unexpected labels {unique_labels}"
    print(f"Pre-check 2 PASSED: Exactly 6 target labels: {unique_labels}")

    label_counts = df["label"].value_counts().to_dict()
    for cat in TARGET_CLASSES:
        cnt = label_counts.get(cat, 0)
        assert cnt == 4300, f"Pre-check Failed: Class '{cat}' has {cnt} rows (expected 4,300)"
    print("Pre-check 3 PASSED: Exactly 4,300 documents per label")

    null_count = int(df["text"].isna().sum())
    assert null_count == 0, f"Pre-check Failed: Found {null_count} null text values"
    print("Pre-check 4 PASSED: Zero null text values")

    empty_count = int((df["text"].astype(str).str.strip() == "").sum())
    assert empty_count == 0, f"Pre-check Failed: Found {empty_count} empty text values"
    print("Pre-check 5 PASSED: Zero empty text values")

    dup_urls = int(df["url"].duplicated().sum())
    assert dup_urls == 0, f"Pre-check Failed: Found {dup_urls} duplicate URLs"
    print("Pre-check 6 PASSED: Zero duplicate URLs")

    dup_texts = int(df["text"].duplicated().sum())
    assert dup_texts == 0, f"Pre-check Failed: Found {dup_texts} duplicate texts"
    print("Pre-check 7 PASSED: Zero duplicate article text")

    print("\n" + "=" * 70)
    print("STEP 2 — STRATIFIED SPLIT (70% TRAIN / 15% VALIDATION / 15% TEST)")
    print("=" * 70)
    # Stage 1: Split 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Stage 2: Split 30% Temp equally into 15% Val and 15% Test (0.50 of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train split size:      {len(train_df):,} rows (70%)")
    print(f"Validation split size: {len(val_df):,} rows (15%)")
    print(f"Test split size:       {len(test_df):,} rows (15%)")
    print(f"Total split sum:       {len(train_df) + len(val_df) + len(test_df):,} rows")

    print("\nClass distribution in Train split:")
    print(train_df["label"].value_counts())
    print("\nClass distribution in Validation split:")
    print(val_df["label"].value_counts())
    print("\nClass distribution in Test split:")
    print(test_df["label"].value_counts())

    print("\n" + "=" * 70)
    print("STEP 3 — LEAKAGE CHECKS (URL & TEXT OVERLAP)")
    print("=" * 70)
    train_urls = set(train_df["url"])
    val_urls = set(val_df["url"])
    test_urls = set(test_df["url"])

    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    # Overlap tests
    url_overlap_tr_val = train_urls.intersection(val_urls)
    url_overlap_tr_te = train_urls.intersection(test_urls)
    url_overlap_val_te = val_urls.intersection(test_urls)

    text_overlap_tr_val = train_texts.intersection(val_texts)
    text_overlap_tr_te = train_texts.intersection(test_texts)
    text_overlap_val_te = val_texts.intersection(test_texts)

    print(f"URL Overlap Train vs Validation: {len(url_overlap_tr_val)}")
    print(f"URL Overlap Train vs Test:       {len(url_overlap_tr_te)}")
    print(f"URL Overlap Validation vs Test:  {len(url_overlap_val_te)}")

    print(f"Text Overlap Train vs Validation: {len(text_overlap_tr_val)}")
    print(f"Text Overlap Train vs Test:       {len(text_overlap_tr_te)}")
    print(f"Text Overlap Validation vs Test:  {len(text_overlap_val_te)}")

    assert len(url_overlap_tr_val) == 0, f"Leakage detected: {len(url_overlap_tr_val)} URLs between Train & Val"
    assert len(url_overlap_tr_te) == 0, f"Leakage detected: {len(url_overlap_tr_te)} URLs between Train & Test"
    assert len(url_overlap_val_te) == 0, f"Leakage detected: {len(url_overlap_val_te)} URLs between Val & Test"

    assert len(text_overlap_tr_val) == 0, f"Leakage detected: {len(text_overlap_tr_val)} Texts between Train & Val"
    assert len(text_overlap_tr_te) == 0, f"Leakage detected: {len(text_overlap_tr_te)} Texts between Train & Test"
    assert len(text_overlap_val_te) == 0, f"Leakage detected: {len(text_overlap_val_te)} Texts between Val & Test"

    print("ALL LEAKAGE CHECKS PASSED: Exactly 0 overlapping URLs and 0 overlapping texts.")

    # Verify all classes in all splits
    for name, s_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        present = sorted(s_df["label"].unique().tolist())
        assert present == sorted(TARGET_CLASSES), f"Missing classes in {name}: {set(TARGET_CLASSES) - set(present)}"
    print("All 6 target categories are present in all 3 splits.")

    print("\n" + "=" * 70)
    print("STEP 4 — SAVE SPLITS")
    print("=" * 70)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"
    test_path = SPLITS_DIR / "test.csv"

    columns = ["url", "domain", "text", "label"]
    train_df[columns].to_csv(train_path, index=False)
    val_df[columns].to_csv(val_path, index=False)
    test_df[columns].to_csv(test_path, index=False)

    print(f"Saved: {train_path} ({len(train_df):,} rows)")
    print(f"Saved: {val_path} ({len(val_df):,} rows)")
    print(f"Saved: {test_path} ({len(test_df):,} rows)")

    print("\n" + "=" * 70)
    print("STEP 5 — GENERATE SPLIT REPORT")
    print("=" * 70)
    report_path = SPLITS_DIR / "split_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DOCSORT AI — STRATIFIED RESEARCH SPLIT REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. DATASET OVERVIEW\n")
        f.write(f"  - Source File: {INPUT_PATH}\n")
        f.write(f"  - Original Dataset Size: {len(df):,} documents\n")
        f.write(f"  - Total Classes: {len(TARGET_CLASSES)}\n")
        f.write(f"  - Documents per Class in Source: {len(df) // len(TARGET_CLASSES):,}\n\n")
        f.write("2. SPLIT CONFIGURATION\n")
        f.write("  - Methodology: Stratified Two-Stage Split\n")
        f.write("  - Proportions: 70% Train / 15% Validation / 15% Test\n")
        f.write(f"  - Random Seed: {RANDOM_STATE}\n")
        f.write("  - Stratification: Applied in both stages to preserve exact class balance\n\n")
        f.write("3. SPLIT SIZES\n")
        f.write(f"  - Train:      {len(train_df):,} documents (70.0%)\n")
        f.write(f"  - Validation: {len(val_df):,} documents (15.0%)\n")
        f.write(f"  - Test:       {len(test_df):,} documents (15.0%)\n")
        f.write(f"  - Total:      {len(train_df) + len(val_df) + len(test_df):,} documents (100.0%)\n\n")
        f.write("4. CLASS DISTRIBUTION PER SPLIT\n")
        f.write(f"{'Category':26s} | {'Train (70%)':12s} | {'Val (15%)':10s} | {'Test (15%)':10s} | {'Total':8s}\n")
        f.write("-" * 72 + "\n")
        for cat in TARGET_CLASSES:
            tr_c = int((train_df["label"] == cat).sum())
            vl_c = int((val_df["label"] == cat).sum())
            ts_c = int((test_df["label"] == cat).sum())
            tot_c = tr_c + vl_c + ts_c
            f.write(f"{cat:26s} | {tr_c:12,d} | {vl_c:10,d} | {ts_c:10,d} | {tot_c:8,d}\n")
        f.write("-" * 72 + "\n")
        f.write(f"{'TOTAL':26s} | {len(train_df):12,d} | {len(val_df):10,d} | {len(test_df):10,d} | {len(df):8,d}\n\n")
        f.write("5. DATA LEAKAGE & OVERLAP VERIFICATION\n")
        f.write(f"  - URL Overlap (Train vs Validation):  {len(url_overlap_tr_val)} (ZERO LEAKAGE)\n")
        f.write(f"  - URL Overlap (Train vs Test):        {len(url_overlap_tr_te)} (ZERO LEAKAGE)\n")
        f.write(f"  - URL Overlap (Validation vs Test):   {len(url_overlap_val_te)} (ZERO LEAKAGE)\n")
        f.write(f"  - Text Overlap (Train vs Validation): {len(text_overlap_tr_val)} (ZERO LEAKAGE)\n")
        f.write(f"  - Text Overlap (Train vs Test):       {len(text_overlap_tr_te)} (ZERO LEAKAGE)\n")
        f.write(f"  - Text Overlap (Validation vs Test):  {len(text_overlap_val_te)} (ZERO LEAKAGE)\n\n")
        f.write("6. METHODOLOGICAL & RESEARCH ASSURANCES\n")
        f.write("  - Stratified split used across all stages.\n")
        f.write("  - Zero TF-IDF vectorizer fitting performed.\n")
        f.write("  - Zero model training performed.\n")
        f.write("  - Zero feature selection or data augmentation performed.\n")
        f.write("  - Existing DocSort models (model.pkl, vectorizer.pkl) remain untouched.\n")
        f.write("  - Existing FastAPI backend and React frontend remain untouched.\n")

    print(f"Saved split report: {report_path}")

    print("\n" + "=" * 70)
    print("STEP 6 — RELOAD & FINAL PROGRAMMATIC VERIFICATION")
    print("=" * 70)
    re_train = pd.read_csv(train_path)
    re_val = pd.read_csv(val_path)
    re_test = pd.read_csv(test_path)

    # 1. Total row count assertions
    assert len(re_train) == 18060, f"Verification Failed: Train rows = {len(re_train)} (expected 18,060)"
    assert len(re_val) == 3870, f"Verification Failed: Val rows = {len(re_val)} (expected 3,870)"
    assert len(re_test) == 3870, f"Verification Failed: Test rows = {len(re_test)} (expected 3,870)"
    print(f"Verification 1 PASSED: Train={len(re_train):,}, Validation={len(re_val):,}, Test={len(re_test):,}")

    # 2. Sum assertion
    total_reloaded = len(re_train) + len(re_val) + len(re_test)
    assert total_reloaded == 25800, f"Verification Failed: Total = {total_reloaded} (expected 25,800)"
    print(f"Verification 2 PASSED: Train + Validation + Test = {total_reloaded:,} (exactly 25,800)")

    # 3. Per-class distribution assertions
    train_counts = re_train["label"].value_counts().to_dict()
    val_counts = re_val["label"].value_counts().to_dict()
    test_counts = re_test["label"].value_counts().to_dict()

    for cat in TARGET_CLASSES:
        assert train_counts.get(cat, 0) == 3010, f"Verification Failed: Train class '{cat}' count = {train_counts.get(cat)}"
        assert val_counts.get(cat, 0) == 645, f"Verification Failed: Val class '{cat}' count = {val_counts.get(cat)}"
        assert test_counts.get(cat, 0) == 645, f"Verification Failed: Test class '{cat}' count = {test_counts.get(cat)}"

    print("Verification 3 PASSED: Exactly 3,010 per class in Train, 645 per class in Validation, 645 per class in Test")

    # 4. Null & Column checks
    for name, d in [("Train", re_train), ("Validation", re_val), ("Test", re_test)]:
        assert d.columns.tolist() == columns, f"Column mismatch in {name}: {d.columns.tolist()}"
        assert int(d["text"].isna().sum()) == 0, f"Null text in {name}"
        assert int(d["url"].duplicated().sum()) == 0, f"Duplicate URL in {name}"
        assert int(d["text"].duplicated().sum()) == 0, f"Duplicate text in {name}"

    print("Verification 4 PASSED: Columns, null values, and internal uniqueness confirmed across all reloaded splits")
    print("\nALL SPLIT & LEAKAGE CHECKS SUCCESSFULLY COMPLETED!")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
