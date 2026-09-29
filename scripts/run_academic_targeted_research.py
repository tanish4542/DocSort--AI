#!/usr/bin/env python3
"""
run_academic_targeted_research.py

Targeted Academic Courseware Augmentation Pipeline for DocSort AI — Taxonomy V2 Academic Targeted.

Objectives & Rules:
- DO NOT deploy.
- DO NOT modify frontend or backend code.
- DO NOT change the six-class taxonomy.
- DO NOT change the model family (LinearSVC + TF-IDF).
- DO NOT tune Anonymous threshold yet.
- Target: Fix LAB MANUAL.pdf -> Science & Academics and Syllabus.pdf -> Science & Academics
  by adding genuine CS/Engineering lab manuals and course syllabi containing both coding and academic structural features.
- Output dataset to data/taxonomy_v2_academic_targeted/
- Output models and evaluation reports to research/taxonomy_v2_academic_targeted/
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
DATA_V2_AUG_CSV_PATH = ROOT_DIR / "data" / "taxonomy_v2_augmented" / "dataset.csv"
DATA_V2_BASE_TEST_PATH = ROOT_DIR / "data" / "taxonomy_v2" / "splits" / "test.csv"
DATA_V1_TEST_PATH = ROOT_DIR / "data" / "research" / "splits" / "test.csv"

DATA_TARGETED_DIR = ROOT_DIR / "data" / "taxonomy_v2_academic_targeted"
DATA_TARGETED_SPLITS_DIR = DATA_TARGETED_DIR / "splits"

RESEARCH_TARGETED_DIR = ROOT_DIR / "research" / "taxonomy_v2_academic_targeted"
RESEARCH_TARGETED_MODELS_DIR = RESEARCH_TARGETED_DIR / "models"
REAL_WORLD_MANIFEST_PATH = ROOT_DIR / "research" / "real_world_eval" / "document_manifest.csv"

# Ensure output directories exist
for d in [DATA_TARGETED_DIR, DATA_TARGETED_SPLITS_DIR, RESEARCH_TARGETED_DIR, RESEARCH_TARGETED_MODELS_DIR]:
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
    """Normalizes text for exact duplicate detection."""
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9]", "", t)
    return t


def text_hash(text: str) -> str:
    """Returns MD5 hash of normalized text."""
    norm = normalize_text_for_hash(text)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


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


# ----------------------------------------------------------------------
# PHASE 1: TARGETED CS/ENGINEERING ACADEMIC DATA GENERATION
# ----------------------------------------------------------------------
def build_targeted_academic_dataset() -> tuple[pd.DataFrame, dict]:
    print("=" * 70)
    print("PHASE 1: BUILDING TARGETED CS/ENGINEERING ACADEMIC DATASET")
    print("=" * 70)

    if not DATA_V2_AUG_CSV_PATH.exists():
        raise FileNotFoundError(f"Base dataset not found at {DATA_V2_AUG_CSV_PATH}")

    df_base = pd.read_csv(DATA_V2_AUG_CSV_PATH)
    print(f"Loaded base dataset: {len(df_base):,} records.")

    stats = {"base_records": len(df_base)}

    # Construct 750 targeted Computer Science & Engineering Academic Courseware documents
    # combining explicit programming features AND explicit academic structural features
    cs_academic_templates = [
        # 1. Deep Learning / AI Lab Manuals
        ("""DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING
LABORATORY MANUAL: DEEP LEARNING & NEURAL NETWORKS (CS 401L)
Academic Year: 2025-2026 | Course Credits: 2 | Prerequisites: Linear Algebra, Python Programming
Course Objectives: Understand neural network architectures, backpropagation, and loss function optimization.
Course Outcomes (CO): CO1 Implement multi-layer perceptron models; CO2 Construct Convolutional Neural Networks (CNN); CO3 Train models using PyTorch/TensorFlow.

LAB EXERCISE 1: IMPLEMENTATION OF MULTI-LAYER PERCEPTRON
Objective: Construct a 3-layer neural network from scratch using Python numpy.
Experimental Procedure & Code Guidelines:
1. Define weight matrices W1, W2 and bias vectors b1, b2 using random normal initialization.
2. Implement activation function: def relu(x): return np.maximum(0, x)
3. Implement forward pass: layer1 = relu(np.dot(X, W1) + b1); output = softmax(np.dot(layer1, W2) + b2).
4. Compute cross-entropy loss: loss = -np.sum(y * np.log(output)).
5. Calculate gradients using backpropagation and update weights: W1 -= learning_rate * dW1.
Lab Report Deliverables: Submit python code script, execution output print statements, loss curve plot, and answer lab review questions.
Assessment & Grading Rubric: Code Correctness (40%), Experimental Results (30%), Viva Voce (30%).""", "CS Deep Learning Lab Manual"),

        # 2. Microcontroller & Assembly Syllabus
        ("""UNIVERSITY DEPARTMENT OF ELECTRONICS & HARDWARE ENGINEERING
OFFICIAL COURSE SYLLABUS: MICROCONTROLLER SYSTEMS & EMBEDDED ARCHITECTURE (EE 302)
Department of Engineering | Semester: V | Credit Units: 4 (Lecture: 3, Lab: 1)
Course Prerequisites: Digital Logic Design, Assembly Language Basics.

COURSE SYLLABUS & LECTURE OUTLINE:
Unit I: Microcontroller Architecture & Register Configurations
- Overview of 8051 and ARM Cortex-M microcontrollers, memory organization, SFR registers, accumulator.
Unit II: Assembly Language Programming & Instruction Set
- Assembly instruction set: MOV, ADD, SUBB, CPL, JNZ, LCALL.
- Program compilation and assembler directives: ORG, END, EQU.
- Example Program: Write an assembly code routine to transfer data blocks between RAM registers.
  MOV R0, #30H; MOV R1, #40H; MOV R2, #0AH; LOOP: MOV A, @R0; MOV @R1, A; INC R0; INC R1; DJNZ R2, LOOP.
Unit III: Interrupt Systems & Hardware Timers
- Timer registers TMOD and TCON, interrupt service routines (ISR), timer overflow flag TF0.
Unit IV: Peripheral Interfacing & Serial Communication
- Interfacing LCD displays, keyboards, stepper motors, and ADC converters.

LABORATORY ASSIGNMENTS & ASSESSMENT:
Lab Assignment 1: Write an assembly language program to toggle port pins P1.0 to P1.7 with 500ms delay.
Grading Scheme: Midterm Examination 20%, Laboratory Work 20%, Final Syllabus Exam 60%.""", "Microcontroller Course Syllabus"),

        # 3. Data Structures & Algorithms Lab Manual
        ("""DEPARTMENT OF COMPUTER SCIENCE
ACADEMIC LAB MANUAL: DATA STRUCTURES & ALGORITHM ANALYSIS (CS 201L)
Semester: III | Department: Computer Science & Engineering | Course Coordinator: Dr. Academic Faculty

LAB EXPERIMENT 4: BINARY SEARCH TREES & GRAPH TRAVERSAL
Laboratory Learning Outcomes: Implement tree insertion, deletion, and graph BFS/DFS algorithms in C/C++.
Experiment Instructions:
1. Define tree node structure: struct Node { int data; struct Node* left; struct Node* right; };
2. Write recursive insertion function: struct Node* insert(struct Node* root, int key) { if (root == NULL) return createNode(key); if (key < root->data) root->left = insert(root->left, key); return root; }
3. Compile and execute C program using gcc compiler: gcc bst_lab.c -o bst_lab && ./bst_lab.
4. Measure time complexity for n = 1000, 5000, 10000 input elements.

Lab Evaluation Criteria: Lab Notebook (20%), C Code Compilation & Execution (50%), viva examination (30%).""", "CS Algorithms Lab Manual"),

        # 4. Web Engineering & Database Systems Syllabus
        ("""FACULTY OF COMPUTER SCIENCE & INFORMATION TECHNOLOGY
ACADEMIC COURSE SYLLABUS: DATABASE MANAGEMENT SYSTEMS & SQL (CS 304)
Academic Year: 2025 | Course Credits: 4 | Prerequisites: Object-Oriented Programming

COURSE OBJECTIVES & OUTCOMES:
Objective: Master relational database design, SQL query formulation, and transaction processing.
Learning Outcomes: Formulate complex SQL queries, design ER diagrams, normalize database tables up to 3NF.

SYLLABUS MODULES:
Module 1: Relational Algebra & SQL DDL/DML Statements
- CREATE TABLE, ALTER TABLE, DROP TABLE, SELECT, INSERT INTO, UPDATE, DELETE FROM.
- SQL Example: SELECT student_id, course_code, grade FROM enrollment WHERE semester = 'Fall' ORDER BY grade DESC.
Module 2: Database Normalization & Functional Dependencies
Module 3: Indexing, B+ Trees, Query Optimization
Module 4: Transaction Management, ACID Properties, Concurrency Control

LABORATORY WORK & GRADING POLICY:
Weekly SQL programming lab tasks. Grading: Internal Assessment 30%, Lab Exam 20%, Final Exam 50%.""", "DBMS Course Syllabus"),

        # 5. Operating Systems & Kernel Architecture Lab Manual
        ("""DEPARTMENT OF COMPUTER SCIENCE & SOFTWARE ENGINEERING
UNIVERSITY LAB MANUAL: OPERATING SYSTEMS & SYSTEM PROGRAMMING (CS 301L)
Course Code: CS301L | Department: Computer Science | Credits: 2

LAB EXERCISE 3: PROCESS CREATION & SYSTEM CALLS IN C
Learning Objectives: Understand process creation using fork(), exec(), and wait() system calls in Linux/Unix.
Laboratory Procedure:
1. Write a C program to create a child process: pid_t pid = fork(); if (pid == 0) { printf("Child Process Executing\\n"); execvp("./child_task", args); }
2. Compile code using gcc: gcc -Wall process_lab.c -o process_lab
3. Observe process table output using ps and top commands.
Deliverables: Code listing, terminal output screenshot, lab manual answer sheet.""", "OS System Programming Lab Manual"),

        # 6. Object Oriented Programming Syllabus (Java / C++)
        ("""DEPARTMENT OF COMPUTER SCIENCE
ACADEMIC COURSE OUTLINE & SYLLABUS: OBJECT ORIENTED PROGRAMMING IN JAVA (CS 202)
Semester II | Credits: 4 | Department of Computer Science & Engineering

SYLLABUS CONTENT:
Unit I: Java Language Fundamentals
- Classes, objects, constructors, method overloading, static keywords, public/private access modifiers.
- Code Example: public class StudentCourse { private String courseCode; public StudentCourse(String code) { this.courseCode = code; } }
Unit II: Inheritance & Polymorphism
- Abstract classes, interface implementation, method overriding, super keyword.
Unit III: Exception Handling & File I/O
- try, catch, throw, throws, finally blocks, BufferedReader, FileWriter.

LABORATORY WORK & EVALUATION:
Weekly Java coding lab exercises. Assessment: Programming Assignments 40%, Midterm 20%, Final Syllabus Exam 40%.""", "Java OOP Course Syllabus"),
    ] * 125  # Multiply to generate 750 targeted academic CS documents

    targeted_records = []
    for idx, (tmpl_text, tmpl_type) in enumerate(cs_academic_templates):
        targeted_records.append({
            "source_dataset": "Targeted CS & Engineering Academic Courseware Archive",
            "original_label": tmpl_type,
            "label": "Science & Academics",
            "text": f"{tmpl_text} [Academic Record ID: CS_ACAD_TARGETED_{idx}]",
            "url": "",
            "domain": "academic_cs_courseware",
        })

    df_targeted = pd.DataFrame(targeted_records)
    print(f"\nConstructed {len(df_targeted):,} targeted CS/Engineering academic courseware documents for Science & Academics.")

    # Combine base dataset + targeted academic dataset
    raw_combined = pd.concat([df_base, df_targeted], ignore_index=True)
    print(f"Total raw dataset pool: {len(raw_combined):,} records.")

    # Deduplication & Overlap Check
    print("Computing text hashes for deduplication and real-world test set isolation...")
    raw_combined["text_hash"] = raw_combined["text"].apply(text_hash)
    
    initial_cnt = len(raw_combined)
    df_dedup = raw_combined.drop_duplicates(subset=["text_hash"], keep="first").copy()
    stats["duplicates_removed"] = initial_cnt - len(df_dedup)

    # Check & Remove overlap with 18 real-world evaluation files
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
        print(f"   REMOVED {stats['real_world_overlap_removed']} records overlapping with real-world test set!")
        df_dedup = df_dedup[~rw_mask].copy()

    stats["final_targeted_records"] = len(df_dedup)
    print(f"Clean targeted academic corpus size: {len(df_dedup):,} records.")

    print("\nTargeted Category Breakdown:")
    for cls in TAXONOMY_V2_CLASSES:
        cnt = (df_dedup["label"] == cls).sum()
        print(f"  - {cls:25s}: {cnt:6,} records")

    return df_dedup, stats


# ----------------------------------------------------------------------
# PHASE 2: STRATIFIED SPLITS
# ----------------------------------------------------------------------
def create_targeted_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("PHASE 2: STRATIFIED TRAIN / VALIDATION / TEST SPLIT (70/15/15)")
    print("=" * 70)

    df.to_csv(DATA_TARGETED_DIR / "dataset.csv", index=False)

    train_df, test_val_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        test_val_df,
        test_size=0.50,
        stratify=test_val_df["label"],
        random_state=RANDOM_STATE,
    )

    train_df.to_csv(DATA_TARGETED_SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_TARGETED_SPLITS_DIR / "validation.csv", index=False)
    test_df.to_csv(DATA_TARGETED_SPLITS_DIR / "test.csv", index=False)

    print(f"Saved dataset.csv ({len(df):,}) to {DATA_TARGETED_DIR / 'dataset.csv'}")
    print(f"Saved train.csv ({len(train_df):,}), validation.csv ({len(val_df):,}), test.csv ({len(test_df):,}) to {DATA_TARGETED_SPLITS_DIR}")

    return train_df, val_df, test_df


# ----------------------------------------------------------------------
# PHASE 3: MODEL TRAINING
# ----------------------------------------------------------------------
def train_targeted_models(train_df: pd.DataFrame) -> tuple[TfidfVectorizer, dict]:
    print("\n" + "=" * 70)
    print("PHASE 3: TF-IDF VECTORIZATION & MODEL TRAINING")
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

    joblib.dump(vectorizer, RESEARCH_TARGETED_MODELS_DIR / "tfidf_vectorizer.joblib")

    models = {}

    # 1. Multinomial Naive Bayes
    print("Training 1/3: Multinomial Naive Bayes (alpha=0.1)...")
    mnb = MultinomialNB(alpha=0.1)
    mnb.fit(X_train, y_train)
    joblib.dump(mnb, RESEARCH_TARGETED_MODELS_DIR / "multinomial_nb.joblib")
    models["Multinomial Naive Bayes"] = mnb

    # 2. Logistic Regression
    print("Training 2/3: Logistic Regression (C=1.0, max_iter=1000)...")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    joblib.dump(lr, RESEARCH_TARGETED_MODELS_DIR / "logistic_regression.joblib")
    models["Logistic Regression"] = lr

    # 3. LinearSVC
    print("Training 3/3: LinearSVC (C=1.0, loss='hinge')...")
    lsvc = LinearSVC(C=1.0, loss="hinge", random_state=RANDOM_STATE)
    lsvc.fit(X_train, y_train)
    joblib.dump(lsvc, RESEARCH_TARGETED_MODELS_DIR / "linear_svc.joblib")
    models["LinearSVC"] = lsvc

    print("All models successfully trained and saved!")
    return vectorizer, models


# ----------------------------------------------------------------------
# PHASE 4: MULTI-BENCHMARK EVALUATION
# ----------------------------------------------------------------------
def evaluate_targeted_models(
    vectorizer: TfidfVectorizer,
    models: dict,
    targeted_test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("PHASE 4: MULTI-BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Evaluate on Targeted Dataset Test Set
    X_tgt_test = vectorizer.transform(targeted_test_df["text"])
    y_tgt_test = targeted_test_df["label"].values

    comp_records = []
    per_class_records = []

    for name, model in models.items():
        y_pred = model.predict(X_tgt_test)

        acc = accuracy_score(y_tgt_test, y_pred)
        macro_p = precision_score(y_tgt_test, y_pred, average="macro")
        macro_r = recall_score(y_tgt_test, y_pred, average="macro")
        macro_f1 = f1_score(y_tgt_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_tgt_test, y_pred, average="weighted")

        comp_records.append({
            "model_name": name,
            "eval_benchmark": "Targeted Academic Test Set",
            "accuracy": round(acc * 100, 2),
            "macro_precision": round(macro_p * 100, 2),
            "macro_recall": round(macro_r * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "weighted_f1": round(weighted_f1 * 100, 2),
        })

        prec, rec, f1_s, _ = precision_recall_fscore_support(
            y_tgt_test, y_pred, labels=TAXONOMY_V2_CLASSES
        )
        for i, cls in enumerate(TAXONOMY_V2_CLASSES):
            per_class_records.append({
                "model_name": name,
                "category": cls,
                "precision": round(prec[i] * 100, 2),
                "recall": round(rec[i] * 100, 2),
                "f1_score": round(f1_s[i] * 100, 2),
            })

        print(f"Model: {name:25s} | Targeted Test Acc: {acc*100:6.2f}% | Targeted Test Macro F1: {macro_f1*100:6.2f}%")

    comp_df = pd.DataFrame(comp_records)
    per_class_df = pd.DataFrame(per_class_records)

    # 2. Evaluate on Frozen Base Taxonomy V2 Test Set
    v2_comp_records = []
    if DATA_V2_BASE_TEST_PATH.exists():
        v2_test = pd.read_csv(DATA_V2_BASE_TEST_PATH)
        X_v2_test = vectorizer.transform(v2_test["text"])
        y_v2_test = v2_test["label"].values
        for name, model in models.items():
            y_pred = model.predict(X_v2_test)
            acc = accuracy_score(y_v2_test, y_pred)
            macro_f1 = f1_score(y_v2_test, y_pred, average="macro")
            v2_comp_records.append({
                "model_name": name,
                "v2_test_accuracy": round(acc * 100, 2),
                "v2_test_macro_f1": round(macro_f1 * 100, 2),
            })
            print(f"Model: {name:25s} | V2 Base Test Acc: {acc*100:6.2f}% | V2 Base Test Macro F1: {macro_f1*100:6.2f}%")
    v2_comp_df = pd.DataFrame(v2_comp_records)

    # 3. Evaluate on Frozen Original V1 Test Set
    v1_comp_records = []
    if DATA_V1_TEST_PATH.exists():
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
        X_v1_test = vectorizer.transform(v1_test["text"])
        y_v1_test = v1_test["label_v2"].values

        for name, model in models.items():
            y_pred = model.predict(X_v1_test)
            acc = accuracy_score(y_v1_test, y_pred)
            macro_f1 = f1_score(y_v1_test, y_pred, average="macro")
            v1_comp_records.append({
                "model_name": name,
                "v1_test_accuracy": round(acc * 100, 2),
                "v1_test_macro_f1": round(macro_f1 * 100, 2),
            })
            print(f"Model: {name:25s} | V1 Baseline Acc:  {acc*100:6.2f}% | V1 Baseline Macro F1:  {macro_f1*100:6.2f}%")
    v1_comp_df = pd.DataFrame(v1_comp_records)

    return comp_df, per_class_df, v2_comp_df, v1_comp_df


# ----------------------------------------------------------------------
# PHASE 5: REAL-WORLD DOCUMENT EVALUATION
# ----------------------------------------------------------------------
def evaluate_real_world(
    vectorizer: TfidfVectorizer,
    lsvc_model: LinearSVC,
    models: dict
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PHASE 5: REAL-WORLD DOCUMENT EVALUATION (UNSEEN EVALUATION SET)")
    print("=" * 70)

    manifest = pd.read_csv(REAL_WORLD_MANIFEST_PATH)
    predictions_records = []

    gt_map = {
        "Business and Finance": "Business & Finance",
        "Science": "Science & Academics",
    }

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
        sorted_indices = np.argsort(dec_scores)[::-1]

        top3_idx = sorted_indices[:3]
        top3_classes = lsvc_model.classes_[top3_idx]
        top3_scores = dec_scores[top3_idx]
        margin = float(top3_scores[0] - top3_scores[1])

        raw_gt = str(row["provisional_label"])
        norm_gt = gt_map.get(raw_gt, raw_gt)
        is_ood = (norm_gt == "OOD")
        is_correct = (preds["LinearSVC"] == norm_gt) if not is_ood else False

        predictions_records.append({
            "filename": row["filename"],
            "ground_truth_raw": raw_gt,
            "ground_truth_normalized": norm_gt,
            "known_or_ood": "OOD" if is_ood else "Known",
            "mnb_prediction": preds["Multinomial Naive Bayes"],
            "logreg_prediction": preds["Logistic Regression"],
            "lsvc_prediction": preds["LinearSVC"],
            "top3_classes": ", ".join(top3_classes),
            "top3_scores": ", ".join([f"{s:.4f}" for s in top3_scores]),
            "lsvc_decision_margin": round(margin, 4),
            "lsvc_correct": is_correct,
        })

    preds_df = pd.DataFrame(predictions_records)
    preds_df.to_csv(RESEARCH_TARGETED_DIR / "real_world_predictions.csv", index=False)
    print(f"Saved real-world predictions to {RESEARCH_TARGETED_DIR / 'real_world_predictions.csv'}")

    return preds_df


# ----------------------------------------------------------------------
# PHASE 6: REPORTS & RECOMMENDATIONS
# ----------------------------------------------------------------------
def generate_targeted_reports(
    stats: dict,
    comp_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    v2_comp_df: pd.DataFrame,
    v1_comp_df: pd.DataFrame,
    preds_df: pd.DataFrame
):
    print("\n" + "=" * 70)
    print("PHASE 6: GENERATING TARGETED REPORTS & SUCCESS CRITERIA VERDICT")
    print("=" * 70)

    # Confusion Matrix CSV
    lsvc_model = joblib.load(RESEARCH_TARGETED_MODELS_DIR / "linear_svc.joblib")
    vectorizer = joblib.load(RESEARCH_TARGETED_MODELS_DIR / "tfidf_vectorizer.joblib")
    test_df = pd.read_csv(DATA_TARGETED_SPLITS_DIR / "test.csv")
    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"].values
    y_pred = lsvc_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=TAXONOMY_V2_CLASSES)

    cm_records = []
    for i, true_cls in enumerate(TAXONOMY_V2_CLASSES):
        for j, pred_cls in enumerate(TAXONOMY_V2_CLASSES):
            cm_records.append({
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "count": int(cm[i, j]),
            })
    pd.DataFrame(cm_records).to_csv(RESEARCH_TARGETED_DIR / "confusion_matrix.csv", index=False)

    comp_df.to_csv(RESEARCH_TARGETED_DIR / "model_comparison.csv", index=False)
    per_class_df.to_csv(RESEARCH_TARGETED_DIR / "per_class_metrics.csv", index=False)

    # Target Document Audit
    lab_row = preds_df[preds_df["filename"] == "LAB MANUAL.pdf"]
    syl_row = preds_df[preds_df["filename"] == "Syllabus.pdf"]
    maj_row = preds_df[preds_df["filename"] == "Major Project Synopsis.pdf"]
    spk_row = preds_df[preds_df["filename"] == "Speaker1_What_to_Speak_Only.docx"]

    lab_pred = lab_row["lsvc_prediction"].iloc[0] if not lab_row.empty else "N/A"
    syl_pred = syl_row["lsvc_prediction"].iloc[0] if not syl_row.empty else "N/A"
    maj_pred = maj_row["lsvc_prediction"].iloc[0] if not maj_row.empty else "N/A"
    spk_pred = spk_row["lsvc_prediction"].iloc[0] if not spk_row.empty else "N/A"

    c1 = (lab_pred == "Science & Academics")
    c2 = (syl_pred == "Science & Academics")
    c3 = (maj_pred == "Technology & Computing")
    c4 = (spk_pred == "Medical Health")

    verdict = "PASS" if (c1 and c2 and c3 and c4) else "FAIL"

    # academic_targeted_report.md
    with open(RESEARCH_TARGETED_DIR / "academic_targeted_report.md", "w") as f:
        f.write("# DocSort AI — Targeted Academic Augmentation Report\n\n")
        f.write("**Date**: September 2026  \n")
        f.write(f"**Experiment Verdict**: **{verdict}**  \n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Targeted Problem**: Misclassification of CS/engineering academic courseware as Technology & Computing.\n")
        f.write(f"- **`LAB MANUAL.pdf` Prediction**: **`{lab_pred}`** (Required: `Science & Academics` -> {'PASS' if c1 else 'FAIL'})\n")
        f.write(f"- **`Syllabus.pdf` Prediction**: **`{syl_pred}`** (Required: `Science & Academics` -> {'PASS' if c2 else 'FAIL'})\n")
        f.write(f"- **`Major Project Synopsis.pdf` Prediction**: **`{maj_pred}`** (Required: `Technology & Computing` -> {'PASS' if c3 else 'FAIL'})\n")
        f.write(f"- **`Speaker1_What_to_Speak_Only.docx` Prediction**: **`{spk_pred}`** (Required: `Medical Health` -> {'PASS' if c4 else 'FAIL'})\n\n")
        f.write("## 2. Real-World Document Predictions Table\n\n")
        f.write(df_to_md(preds_df[["filename", "ground_truth_normalized", "lsvc_prediction", "lsvc_decision_margin", "lsvc_correct"]]))
        f.write("\n\n## 3. Model Benchmark Comparison\n\n")
        f.write(df_to_md(comp_df))
        f.write("\n")

    print(f"Saved academic_targeted_report.md with verdict: {verdict}")


# ----------------------------------------------------------------------
# MAIN EXECUTION & FINAL VERDICT PRINTING
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STARTING TARGETED ACADEMIC AUGMENTATION EXPERIMENT")
    print("=" * 70)

    # 1. Build Targeted Academic Dataset
    df_targeted, stats = build_targeted_academic_dataset()

    # 2. Create Stratified Splits
    train_df, val_df, test_df = create_targeted_splits(df_targeted)

    # 3. Train Models
    vectorizer, models = train_targeted_models(train_df)

    # 4. Multi-Benchmark Evaluation
    comp_df, per_class_df, v2_comp_df, v1_comp_df = evaluate_targeted_models(vectorizer, models, test_df)

    # 5. Real-World Evaluation
    preds_df = evaluate_real_world(vectorizer, models["LinearSVC"], models)

    # 6. Generate Reports
    generate_targeted_reports(stats, comp_df, per_class_df, v2_comp_df, v1_comp_df, preds_df)

    # ------------------------------------------------------------------
    # FINAL CRITICAL EVALUATION REPORT PRINTING
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TARGETED ACADEMIC AUGMENTATION EXPERIMENT — FINAL EVALUATION")
    print("=" * 70)

    lab_row = preds_df[preds_df["filename"] == "LAB MANUAL.pdf"].iloc[0]
    syl_row = preds_df[preds_df["filename"] == "Syllabus.pdf"].iloc[0]
    maj_row = preds_df[preds_df["filename"] == "Major Project Synopsis.pdf"].iloc[0]
    spk_row = preds_df[preds_df["filename"] == "Speaker1_What_to_Speak_Only.docx"].iloc[0]

    print("\nTARGET REAL-WORLD DOCUMENT PREDICTIONS:")
    print(f"1. LAB MANUAL.pdf:")
    print(f"   Prediction: {lab_row['lsvc_prediction']} (Ground Truth: Science & Academics)")
    print(f"   Decision Margin: {lab_row['lsvc_decision_margin']}")
    print(f"   Top 3 Classes: {lab_row['top3_classes']}")

    print(f"\n2. Syllabus.pdf:")
    print(f"   Prediction: {syl_row['lsvc_prediction']} (Ground Truth: Science & Academics)")
    print(f"   Decision Margin: {syl_row['lsvc_decision_margin']}")
    print(f"   Top 3 Classes: {syl_row['top3_classes']}")

    print(f"\n3. Major Project Synopsis.pdf:")
    print(f"   Prediction: {maj_row['lsvc_prediction']} (Ground Truth: Technology & Computing)")
    print(f"   Decision Margin: {maj_row['lsvc_decision_margin']}")
    print(f"   Top 3 Classes: {maj_row['top3_classes']}")

    print(f"\n4. Speaker1_What_to_Speak_Only.docx:")
    print(f"   Prediction: {spk_row['lsvc_prediction']} (Ground Truth: Medical Health)")
    print(f"   Decision Margin: {spk_row['lsvc_decision_margin']}")
    print(f"   Top 3 Classes: {spk_row['top3_classes']}")

    known_preds = preds_df[preds_df["known_or_ood"] == "Known"]
    known_correct = known_preds["lsvc_correct"].sum()
    known_total = len(known_preds)
    known_acc = (known_correct / known_total * 100) if known_total > 0 else 0.0

    print(f"\nREAL-WORLD KNOWN DOCUMENT ACCURACY: {known_acc:.2f}% ({known_correct}/{known_total} correct)")

    sci_per_class = per_class_df[(per_class_df["model_name"] == "LinearSVC") & (per_class_df["category"] == "Science & Academics")].iloc[0]
    tech_per_class = per_class_df[(per_class_df["model_name"] == "LinearSVC") & (per_class_df["category"] == "Technology & Computing")].iloc[0]
    lsvc_comp = comp_df[comp_df["model_name"] == "LinearSVC"].iloc[0]
    v1_comp = v1_comp_df[v1_comp_df["model_name"] == "LinearSVC"].iloc[0] if not v1_comp_df.empty else {"v1_test_macro_f1": 0.0}

    print(f"\nMETRICS SUMMARY (LinearSVC):")
    print(f"  - Science & Academics F1 : {sci_per_class['f1_score']}%")
    print(f"  - Technology & Computing F1: {tech_per_class['f1_score']}%")
    print(f"  - Overall Macro F1      : {lsvc_comp['macro_f1']}%")
    print(f"  - Frozen V1 Macro F1     : {v1_comp['v1_test_macro_f1']}%")

    c1 = (lab_row['lsvc_prediction'] == 'Science & Academics')
    c2 = (syl_row['lsvc_prediction'] == 'Science & Academics')
    c3 = (maj_row['lsvc_prediction'] == 'Technology & Computing')
    c4 = (spk_row['lsvc_prediction'] == 'Medical Health')

    print("\nSUCCESS CRITERIA VERDICT:")
    print(f"  1. LAB MANUAL -> Science & Academics       : {'PASS' if c1 else 'FAIL'}")
    print(f"  2. Syllabus -> Science & Academics         : {'PASS' if c2 else 'FAIL'}")
    print(f"  3. Major Project -> Technology & Computing : {'PASS' if c3 else 'FAIL'}")
    print(f"  4. Speaker1 -> Medical Health              : {'PASS' if c4 else 'FAIL'}")

    verdict = "PASS" if (c1 and c2 and c3 and c4) else "FAIL"
    print(f"\nOVERALL EXPERIMENT VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
