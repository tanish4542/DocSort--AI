#!/usr/bin/env python3
"""
prepare_research_dataset.py

Prepares a clean, balanced 6-category research dataset from the Hugging Face
'mdonigian/iab-news-classification' dataset for the DocSort AI research project.

Target categories (6):
  1. Business and Finance
  2. Medical Health
  3. Sports
  4. Technology & Computing
  5. Science
  6. Entertainment

Target: 4,300 documents per category = 25,800 documents.
Random State: 42
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
from datasets import load_dataset

TARGET_CATEGORIES = [
    "Business and Finance",
    "Medical Health",
    "Sports",
    "Technology & Computing",
    "Science",
    "Entertainment",
]

TARGET_PER_CLASS = 4300
RANDOM_STATE = 42


def main() -> bool:
    print("=" * 70)
    print("STEP 1 — LOAD DATA")
    print("=" * 70)
    print("Loading dataset 'mdonigian/iab-news-classification' (split='train')...")
    dataset = load_dataset("mdonigian/iab-news-classification", split="train")
    df = dataset.to_pandas()

    total_original_rows = len(df)
    print(f"Total original rows: {total_original_rows:,}")
    print(f"Column names: {df.columns.tolist()}")

    print("\n" + "=" * 70)
    print("STEP 2 — SELECT COLUMNS")
    print("=" * 70)
    columns_to_keep = ["url", "domain", "maintext", "iab_category"]
    df = df[columns_to_keep].copy()
    print(f"Retained columns: {columns_to_keep}")

    print("\n" + "=" * 70)
    print("STEP 3 — SELECT SIX CATEGORIES")
    print("=" * 70)
    df_selected = df[df["iab_category"].isin(TARGET_CATEGORIES)].copy()
    print(f"Selected categories ({len(TARGET_CATEGORIES)}): {TARGET_CATEGORIES}")
    counts_before_cleaning = df_selected["iab_category"].value_counts()
    print("\nCategory counts before cleaning:")
    print(counts_before_cleaning)
    rows_before_cleaning = len(df_selected)
    print(f"Total selected rows before cleaning: {rows_before_cleaning:,}")

    print("\n" + "=" * 70)
    print("STEP 4 — DATA QUALITY CLEANING")
    print("=" * 70)
    # 1. Remove rows where maintext is missing/null
    df_step1 = df_selected[df_selected["maintext"].notna()].copy()
    null_removed = len(df_selected) - len(df_step1)
    print(f"1. Removed null/missing maintext: {null_removed:,} removed (Remaining: {len(df_step1):,})")

    # 2. Convert maintext to string
    df_step1["maintext"] = df_step1["maintext"].astype(str)

    # 3. Remove documents whose text length is less than 100 characters
    df_step3 = df_step1[df_step1["maintext"].str.len() >= 100].copy()
    short_removed = len(df_step1) - len(df_step3)
    print(f"2. Removed documents < 100 characters: {short_removed:,} removed (Remaining: {len(df_step3):,})")

    # 4. Remove duplicate URLs
    df_step4 = df_step3.drop_duplicates(subset=["url"], keep="first").copy()
    dup_urls_removed = len(df_step3) - len(df_step4)
    print(f"3. Removed duplicate URLs: {dup_urls_removed:,} removed (Remaining: {len(df_step4):,})")

    # 5. Remove exact duplicate article text
    df_step5 = df_step4.drop_duplicates(subset=["maintext"], keep="first").copy()
    dup_text_removed = len(df_step4) - len(df_step5)
    print(f"4. Removed exact duplicate article text: {dup_text_removed:,} removed (Remaining: {len(df_step5):,})")

    total_removed = rows_before_cleaning - len(df_step5)
    pct_removed = (total_removed / rows_before_cleaning) * 100 if rows_before_cleaning > 0 else 0.0
    print(f"\nTotal records removed during cleaning: {total_removed:,} ({pct_removed:.2f}%)")
    print(f"Clean unique records remaining: {len(df_step5):,}")

    counts_after_cleaning = df_step5["iab_category"].value_counts()
    print("\nCategory distribution after cleaning:")
    print(counts_after_cleaning)

    print("\n" + "=" * 70)
    print(f"STEP 5 — CHECK WHETHER EACH CLASS HAS >= {TARGET_PER_CLASS:,}")
    print("=" * 70)
    for cat in TARGET_CATEGORIES:
        cnt = counts_after_cleaning.get(cat, 0)
        assert cnt >= TARGET_PER_CLASS, f"Class '{cat}' has only {cnt} records (< {TARGET_PER_CLASS})"
        print(f"PASSED: '{cat}' has {cnt:,} unique records (>= {TARGET_PER_CLASS:,})")

    print("\n" + "=" * 70)
    print(f"STEP 6 — BALANCED SAMPLING ({TARGET_PER_CLASS:,} PER CLASS)")
    print("=" * 70)
    samples = []
    for cat in TARGET_CATEGORIES:
        cat_df = df_step5[df_step5["iab_category"] == cat].sample(
            n=TARGET_PER_CLASS, random_state=RANDOM_STATE
        )
        samples.append(cat_df)

    final_df = pd.concat(samples, ignore_index=True)
    # Shuffle full dataset
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampled exactly {TARGET_PER_CLASS:,} per category across all {len(TARGET_CATEGORIES)} categories.")
    print(f"Total sampled documents: {len(final_df):,}")

    print("\n" + "=" * 70)
    print("STEP 7 — OUTPUT FORMATTING & STORAGE")
    print("=" * 70)
    final_df = final_df.rename(columns={"maintext": "text", "iab_category": "label"})
    final_df = final_df[["url", "domain", "text", "label"]]

    output_dir = Path("data/research")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "docsort_research_dataset.csv"
    final_df.to_csv(csv_path, index=False)
    print(f"Saved dataset CSV to: {csv_path}")

    print("\n" + "=" * 70)
    print("STEP 8 — DATASET QUALITY METRICS & REPORT")
    print("=" * 70)
    final_df["text_len"] = final_df["text"].str.len()

    stats = []
    print("\nPer-Category Text Length Statistics:")
    for cat in TARGET_CATEGORIES:
        sub = final_df[final_df["label"] == cat]
        min_l = int(sub["text_len"].min())
        max_l = int(sub["text_len"].max())
        mean_l = float(sub["text_len"].mean())
        median_l = float(sub["text_len"].median())
        stats.append({
            "category": cat,
            "count": len(sub),
            "min_len": min_l,
            "max_len": max_l,
            "mean_len": round(mean_l, 1),
            "median_len": round(median_l, 1),
        })
        print(f"  - {cat:25s}: Count={len(sub):,}, Min={min_l:,}, Max={max_l:,}, Mean={mean_l:,.1f}, Median={median_l:,.1f}")

    unique_urls = int(final_df["url"].nunique())
    unique_texts = int(final_df["text"].nunique())
    null_texts = int(final_df["text"].isna().sum())
    empty_texts = int((final_df["text"].str.strip() == "").sum())

    report_path = output_dir / "dataset_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DOCSORT AI — RESEARCH DATASET PREPARATION REPORT\n")
        f.write("STATUS: SUCCESS (VALIDATED & BALANCED)\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. DATASET SOURCE\n")
        f.write("  - Dataset Source: Hugging Face\n")
        f.write("  - Dataset Name: mdonigian/iab-news-classification\n")
        f.write("  - Split: train\n")
        f.write(f"  - Original dataset: {total_original_rows:,}\n\n")
        f.write("2. TARGET CATEGORIES (6)\n")
        for c in TARGET_CATEGORIES:
            f.write(f"  - {c}\n")
        f.write(f"\n3. RAW SUBSET COUNTS\n")
        f.write(f"  - Six-category raw subset: {rows_before_cleaning:,}\n")
        for c in TARGET_CATEGORIES:
            f.write(f"      * {c}: {counts_before_cleaning.get(c, 0):,}\n")
        f.write("\n4. DATA CLEANING SUMMARY\n")
        f.write(f"  - Missing/null text removed: {null_removed:,}\n")
        f.write(f"  - Short documents (<100 chars) removed: {short_removed:,}\n")
        f.write(f"  - Duplicate URLs removed: {dup_urls_removed:,}\n")
        f.write(f"  - Exact duplicate texts removed: {dup_text_removed:,}\n")
        f.write(f"  - Clean unique documents: {len(df_step5):,}\n")
        f.write(f"  - Percentage of selected records removed: {pct_removed:.2f}%\n\n")
        f.write("5. BALANCED SAMPLING SPECIFICATIONS\n")
        f.write(f"  - Final documents: {len(final_df):,}\n")
        f.write(f"  - Documents per class: {TARGET_PER_CLASS:,}\n")
        f.write(f"  - Random seed: {RANDOM_STATE}\n")
        f.write("  - Oversampling: None (No oversampling)\n")
        f.write("  - Synthetic data: None (No synthetic data)\n")
        f.write("  - Cross-dataset supplementation: None (No cross-dataset supplementation)\n")
        f.write(f"  - Final columns: {['url', 'domain', 'text', 'label']}\n\n")
        f.write("6. FINAL CLASS DISTRIBUTION\n")
        for c in TARGET_CATEGORIES:
            f.write(f"  - {c}: {len(final_df[final_df['label'] == c]):,}\n")
        f.write("\n7. PER-CATEGORY TEXT LENGTH METRICS (CHARACTERS)\n")
        for s in stats:
            f.write(f"  - {s['category']}:\n")
            f.write(f"      Count: {s['count']:,}\n")
            f.write(f"      Min Length: {s['min_len']:,}\n")
            f.write(f"      Max Length: {s['max_len']:,}\n")
            f.write(f"      Mean Length: {s['mean_len']:,}\n")
            f.write(f"      Median Length: {s['median_len']:,}\n")
        f.write("\n8. DEDUPLICATION & INTEGRITY VALIDATION\n")
        f.write(f"  - Total unique URLs: {unique_urls:,} (Expected: {len(final_df):,})\n")
        f.write(f"  - Total unique article texts: {unique_texts:,} (Expected: {len(final_df):,})\n")
        f.write(f"  - Null text values: {null_texts}\n")
        f.write(f"  - Empty text values: {empty_texts}\n\n")
        f.write("9. INTEGRITY ASSURANCE\n")
        f.write("  - No model training performed.\n")
        f.write("  - No train/val/test splits created.\n")
        f.write("  - No existing DocSort model or pipeline modified.\n")

    print(f"Saved dataset report to: {report_path}")

    # STEP 9 — PROGRAMMATIC VALIDATION CHECKS
    print("\n" + "=" * 70)
    print("STEP 9 — VERIFICATION CHECKS")
    print("=" * 70)
    check_df = pd.read_csv(csv_path)
    total_rows = len(check_df)
    unique_labels = check_df["label"].unique().tolist()
    label_counts = check_df["label"].value_counts().to_dict()
    null_texts_check = int(check_df["text"].isna().sum())
    empty_texts_check = int((check_df["text"].astype(str).str.strip() == "").sum())
    dup_urls_check = int(check_df["url"].duplicated().sum())
    dup_texts_check = int(check_df["text"].duplicated().sum())

    assert total_rows == 25800, f"Check 1 Failed: Expected 25,800 rows, got {total_rows}"
    print(f"Check 1 PASSED: Exactly {total_rows:,} rows")

    assert len(unique_labels) == 6, f"Check 2 Failed: Expected 6 unique labels, got {len(unique_labels)}"
    print(f"Check 2 PASSED: Exactly {len(unique_labels)} unique labels: {unique_labels}")

    for lab in TARGET_CATEGORIES:
        cnt = label_counts.get(lab, 0)
        assert cnt == 4300, f"Check 3 Failed: Label '{lab}' has {cnt} rows (expected 4,300)"
    print(f"Check 3 PASSED: Exactly 4,300 rows per label across all 6 categories")

    assert null_texts_check == 0, f"Check 4 Failed: Found {null_texts_check} null text values"
    print(f"Check 4 PASSED: Zero null text values")

    assert empty_texts_check == 0, f"Check 5 Failed: Found {empty_texts_check} empty text values"
    print(f"Check 5 PASSED: Zero empty text values")

    assert dup_urls_check == 0, f"Check 6 Failed: Found {dup_urls_check} duplicate URLs"
    print(f"Check 6 PASSED: Zero duplicate URLs")

    assert dup_texts_check == 0, f"Check 7 Failed: Found {dup_texts_check} duplicate article texts"
    print(f"Check 7 PASSED: Zero duplicate article text")

    print("\nCheck 8: Text length summary per class:")
    for s in stats:
        print(f"  {s['category']:24s} | Min: {s['min_len']:6,d} | Mean: {s['mean_len']:7.1f} | Median: {s['median_len']:7.1f} | Max: {s['max_len']:7,d}")

    print("\nALL 8 VALIDATION CHECKS SUCCESSFULLY PASSED!")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
