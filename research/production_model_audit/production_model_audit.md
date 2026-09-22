# DocSort AI V2: Comprehensive Production Model Audit & System Design Report

**Status:** Completed Technical Audit & Production Design  
**Date:** September 2026  
**Scope:** Architecture Audit, Research-to-Production Distribution Analysis, Production Taxonomy, OOD Engine, Dataset Specification, and Implementation Roadmap for DocSort AI V2.

---

## A. Executive Summary

DocSort AI recently completed its ML-2 research benchmark, proving that a **LinearSVC ($C=0.5$)** classifier utilizing a **100,000-feature TF-IDF** representation outperforms individual Naive Bayes, Logistic Regression, Calibrated Classifiers, Soft Voting, Hard Voting, and 5-Fold Stacking ensembles, achieving **94.50% Accuracy** and **0.9449 Macro F1** on a balanced test set of 3,870 news documents.

However, real-world testing with authentic user files (`PDFPrint.pdf`, `ID card 25 Mar 2025.pdf`, `COURSERA SDV.pdf`, `morphological_Image_Processing.pdf`) revealed a **critical distribution mismatch**:
1. **The Research Task** was *News Article Topic Categorization* across 6 web-scraped journalism domains (`Business and Finance`, `Medical Health`, `Sports`, `Technology & Computing`, `Science`, `Entertainment`).
2. **The Production Task** is *Desktop Document Organization* for arbitrary personal, academic, and workplace files (tuition receipts, lecture slides, coding exercises, ID cards, syllabi, lab test reports, resumes).
3. **The Unconstrained Argmax Failure:** When presented with out-of-distribution desktop files, the LinearSVC model universally rejected the documents (all hyperplane decision scores were negative, e.g. $\max f_c(x) = -0.0167$), but naive relative margin calculation forced a "least-negative" winner (`Technology & Computing`), resulting in false automated sorting.

**Core Recommendation:** To build a reliable **DocSort AI V2**, we must:
- Freeze and preserve all ML-2 research models and datasets without alteration.
- Transition from a topic-based news taxonomy to a **functional 6+1 desktop taxonomy** (`Academic & Educational`, `Financial & Receipts`, `Administrative & Identification`, `Technology & Technical Docs`, `Medical & Healthcare`, `Career & Professional`, and `Unassigned / Needs Review`).
- Implement a **Dual-Gated Decision Engine** combining absolute hyperplane support ($\max f_c(x) \ge \tau_{\text{support}}$) with relative decision margins ($f_{\text{top1}} - f_{\text{top2}} \ge \tau_{\text{margin}}$).
- Curate a dedicated 18,000-document multi-source production dataset (`data/production_v2/`).
- Adopt **False Automatic Sorting Rate (FASR < 1.5%)** as the primary production safety metric.

---

## B. Current Application Audit

We inspected the end-to-end DocSort AI codebase:

### 1. Backend Architecture (`backend/main.py` & `backend/research_inference.py`)
- **Framework:** FastAPI with Uvicorn ASGI server running on port `8001`.
- **Endpoints:**
  - `GET /model-info`: Health and verification endpoint returning model architecture (`LinearSVC`), parameter $C=0.5$, active classes, and vocabulary size ($100,000$).
  - `POST /predict`: Single multipart document upload (`file`), saving to a temporary path, extracting text, predicting, and sorting.
  - `POST /predict-bulk`: Multi-file upload (`files`) processing batch inputs sequentially through the same inference pipeline.
  - `POST /confirm-sort`: Resolves ambiguous files staged in `~/Desktop/SortedDocuments/_pending/` based on user selection.
- **Inference Lifecycle:**
  - Models (`research/models/tfidf_vectorizer.joblib` and `linear_svc.joblib`) are loaded strictly once at backend startup using `joblib`.
  - Feature extraction uses `vectorizer.transform()` (zero runtime fitting).
  - Decision scores are computed via `model.decision_function(X)`.
  - Operational heuristic flags `requires_manual_choice = True` if `decision_margin < 0.50`.

### 2. Document Extraction System
- **Extractors Used:**
  - PDF: `PyPDF2.PdfReader` iterating through pages and appending text.
  - Word: `python-docx.Document` reading paragraph strings.
  - Text: Standard Python file read (`open(..., "r", encoding="utf-8", errors="ignore")`).
- **Deficiencies Identified:**
  - **Zero OCR Capability:** If a PDF is a flat image scan (such as a scanned Aadhaar/passport, doctor prescription, or paper receipt), `PyPDF2` returns an empty string (`""`).
  - **No Quality Gate:** Documents with 15–50 characters of garbage or corrupted OCR ligatures (such as the barcode text in `ID card 25 Mar 2025.pdf`) are passed directly into the TF-IDF vectorizer, producing unreliable predictions.
  - **Formatting Loss:** Tables, two-column layouts, and key-value forms lose structural alignment when flattened into a raw string.

### 3. Frontend & Desktop Sorting (`src/` & `~/Desktop/SortedDocuments/`)
- **React Frontend:** Modern dark/light theme dashboard with state management in `AppContext.jsx`.
- **Workflow:** Real-time upload, progress indication, result dashboard with Decision Margin meter, uncertainty badges (`High confidence`, `Moderate confidence`, `Ambiguous — manual review recommended`), and an expandable **Model Decision Scores** breakdown table.
- **Storage Subsystem:** Moves files on the host filesystem to:
  `~/Desktop/SortedDocuments/<Category>/`
  or stages ambiguous files in:
  `~/Desktop/SortedDocuments/_pending/<uuid>__<filename>`.

---

## C. Research-vs-Production Mismatch

| Feature | Research Environment (ML-2 Experiment) | Production Environment (DocSort App) |
| :--- | :--- | :--- |
| **Corpus Origin** | `mdonigian/iab-news-classification` (Hugging Face) | Real user desktop filesystem uploads |
| **Text Genre** | Online journalism, news stories, op-eds | Invoices, receipts, syllabi, lecture slides, code, ID cards, lab reports |
| **Taxonomy Basis** | Topic-based news beats (Sports, Entertainment, etc.) | Functional utility (Where should this document live?) |
| **Document Structure** | Well-formed prose paragraphs, journalistic grammar | Key-value pairs, tables, bulleted slides, code blocks, form fields |
| **Vocabulary** | Press releases, quotations, reporter commentary | Administrative headers, line items, mathematical symbols, syntax |
| **Length Distribution** | Homogeneous (Mean: 3,400 chars, Median: 3,000 chars) | Extreme variance (50 chars for ID cards to 80,000 chars for manuals) |
| **Class Overlap** | Controlled news categories with minimal bleed | Massive contextual bleed (e.g. CS syllabus blends Tech + Academic) |
| **OOD Prevalence** | 0% (Closed-world test set from same distribution) | 20%–40% (Arbitrary files outside any trained category) |

---

## D. Real-World Test Interpretation

We performed forensic lexical and mathematical analyses on the live user test documents:

### Case 1: `PDFPrint.pdf` (Tuition Payment Receipt)
- **Actual Content:** College online payment e-receipt of ₹2,000 for "TRAINING & PLACEMENT REGISTRATION FEES". Student enrolled in "ARTIFICIAL INTELLIGENCE & MACHINE LEARNING" under "COMPUTER SCIENCE & ENGINEERING".
- **Model Output:** Predicted `Technology & Computing` with margin **0.6940** (Moderate confidence). Auto-sorted into Technology folder!
- **Mathematical Cause:**
  - All 6 hyperplane scores were **negative**: Technology ($-0.0167$), Sports ($-0.7107$), Business ($-0.7224$), Entertainment ($-0.8714$), Medical ($-0.9225$), Science ($-0.9728$).
  - The model has no "Receipts" or "Administrative" class.
  - The words `computer`, `intelligence`, `artificial`, `machine learning` provided minor positive weight ($+0.02$ to $+0.12$), making Technology slightly "less negative" than the other categories.
  - Relative margin calculation ($(-0.0167) - (-0.7107) = 0.6940 > 0.50$) triggered an automatic sort.

### Case 2: `ID card 25 Mar 2025.pdf` (Student Identity Card)
- **Actual Content:** Identity card with photo, roll number (`4NI23CI116`), student name, branch, and validity period (`2023-2027`).
- **Extracted Text Length:** Only **130 characters** with barcode scanner noise (`'  I 11111111 I 11111111 ll Ill...`).
- **Model Output:** Predicted `Technology & Computing` with margin **0.0006** (Ambiguous). Staged in `_pending`.
- **Mathematical Cause:** All hyperplane distances were $-0.42$ to $-0.88$. Technology ($-0.4244$) edged Science ($-0.4250$) by $0.0006$.

### Case 3: `COURSERA SDV.pdf` (Online Course Certificate)
- **Actual Content:** Professional credential for completing "SDV 101: Software Defined Vehicles".
- **Model Output:** Predicted `Technology & Computing` with margin **0.1302** (Ambiguous). Staged in `_pending`.
- **Mathematical Cause:** All hyperplane scores were negative ($-0.4277$ to $-0.9196$). Because the credential was for a software course, it bled between Tech and Business.

### Case 4: `morphological_Image_Processing.pdf` (Computer Vision Slides)
- **Actual Content:** Academic slides explaining morphological dilation, erosion, and set theory.
- **Model Output:** Predicted `Technology & Computing` with margin **0.2635** (Ambiguous). Staged in `_pending`.
- **Mathematical Cause:** The document straddles computer science algorithms and mathematical science. With no `Academic` category, neither hyperplane reached strong positive support.

---

## E. Proposed Production Taxonomy

To solve the distribution mismatch, we establish a **Functional 6+1 Taxonomy** (detailed in [`proposed_taxonomy.md`](file:///Users/tanish4542/CODE%20AND%20PROJECTS/DocSort-AI/research/production_model_audit/proposed_taxonomy.md)):

```
1. Academic & Educational        -> Syllabi, lecture slides, assignments, research preprints, certificates
2. Financial & Receipts          -> Invoices, fee receipts, bank statements, tax forms, pay slips
3. Administrative & ID           -> ID cards, passports, NDAs, lease agreements, registration circulars
4. Technology & Technical Docs   -> Code listings, API documentation, developer manuals, architecture specs
5. Medical & Healthcare          -> Lab blood tests, prescriptions, clinic notes, discharge summaries
6. Career & Professional         -> Resumes, CVs, job offer letters, recommendation letters
7. Unassigned / Needs Review     -> Staged in _pending (OOD reject, corrupt text, ambiguous margin)
```

---

## F. Unknown / Out-of-Distribution (OOD) Analysis

As proven in [`ood_strategy_analysis.md`](file:///Users/tanish4542/CODE%20AND%20PROJECTS/DocSort-AI/research/production_model_audit/ood_strategy_analysis.md), unconstrained argmax on linear classifiers is fundamentally unsafe for open-world desktop sorting.

### The Recommended Dual-Gated Decision Engine
A document is classified automatically if and only if **both gates** pass:

$$\text{Auto-Sort} \iff \begin{cases}
\max_{c} f_c(x) \ge \tau_{\text{support}} & \text{(Gate 1: Absolute Membership Support, } \tau_{\text{support}} = 0.15\text{)} \\
f_{\text{top1}}(x) - f_{\text{top2}}(x) \ge \tau_{\text{margin}} & \text{(Gate 2: Relative Decision Margin, } \tau_{\text{margin}} = 0.55\text{)}
\end{cases}$$

- If **Gate 1 fails**: Document falls outside all learned category hyperplanes. It is flagged as `"Low Class Support — Out-of-Distribution"` and staged in `_pending`.
- If **Gate 1 passes but Gate 2 fails**: Document has positive support but competing categories are too close. Flagged as `"Ambiguous Decision Margin"` and staged in `_pending`.
- If **both gates pass**: Document is auto-sorted to `~/Desktop/SortedDocuments/<Category>/`.

This dual-gated rule has **zero latency overhead ($O(1)$)**, requires no extra dependencies, and would have prevented 100% of the erroneous auto-sorts in real-world testing.

---

## G. Production Dataset Requirements

Detailed in [`production_dataset_spec.md`](file:///Users/tanish4542/CODE%20AND%20PROJECTS/DocSort-AI/research/production_model_audit/production_dataset_spec.md):
- **Location:** `data/production_v2/` (Completely separate from frozen research data).
- **Scale:** 18,000 balanced documents (3,000 per class across 6 classes).
- **Format Mix:** 60% PDF (native & OCR), 25% DOCX, 15% TXT.
- **Approved Open Sourcing:**
  - *Academic*: ArXiv, OpenStax, MIT OpenCourseWare.
  - *Financial*: SROIE, CORD, SEC EDGAR, synthetic billing templates.
  - *Administrative*: RVL-CDIP forms/letters, CUAD contracts, open government circulars.
  - *Technology*: GitHub docs, Linux man pages, open-source codebases.
  - *Medical*: MTSamples transcriptions, PubMed Central case reports.
  - *Career*: Kaggle curated resumes, open offer templates.
- **Deduplication:** SHA-256 exact hash + MinHash LSH (0.85 Jaccard threshold).
- **Splitting:** Leakage-safe stratified 70% Train (12,600 docs), 15% Validation (2,700 docs), 15% Test (2,700 docs).

---

## H. Extraction & OCR Requirements

### 1. Document Extraction Quality Gate
Before any feature extraction occurs, the application must evaluate the raw text:
```python
def validate_extracted_text(text: str) -> tuple[bool, str]:
    if not text or len(text.strip()) < 80:
        return False, "Insufficient text extracted (<80 characters). Scanned document or image PDF detected."
    # Check printable character density
    printable = sum(1 for c in text if c.isprintable() and not c.isspace())
    if printable / max(1, len(text)) < 0.60:
        return False, "Extracted text contains excessive binary noise or unreadable encoding."
    return True, ""
```

### 2. OCR Architecture Strategy
- If `validate_extracted_text` fails on a PDF:
  - Do not pass corrupted text to the classifier.
  - **Graceful Fallback:** If `pytesseract` or `easyocr` is available, trigger an OCR pass on the first 3 pages of the PDF.
  - If OCR is disabled or fails, immediately route the document to `_pending` with status: `"Requires Manual Review (Scanned Image PDF)"`.

---

## I. Model Strategy for V2

Our ML-2 research experiment conclusively proved:
1. Standalone **LinearSVC ($C=0.5$)** achieved the highest accuracy ($94.50\%$) and macro F1 ($0.9449$).
2. Soft Voting, Hard Voting, and 5-Fold Stacking ensembles were **empirically inferior** due to probability distortion from high-dimensional text.

### V2 Architecture Recommendation
- **Feature Pipeline:** `TfidfVectorizer(ngram_range=(1,2), max_features=100000, sublinear_tf=True, min_df=2, max_df=0.95)`.
- **Classifier:** `LinearSVC(C=0.5, loss='squared_hinge')`.
- **Decision Engine:** Dual-Gated Decision Margin with threshold tuning on the V2 validation split.
- **Alternative Evaluated:** `CalibratedClassifierCV(LinearSVC)` provides calibrated probabilities but increases training time $5\times$ and slightly reduced Macro F1 in research testing ($0.9423$ vs $0.9449$). Dual-gated hyperplane thresholding achieves superior OOD safety without calibration overhead.

---

## J. Production Evaluation Strategy

Detailed in [`evaluation_strategy.md`](file:///Users/tanish4542/CODE%20AND%20PROJECTS/DocSort-AI/research/production_model_audit/evaluation_strategy.md):
- **Benchmark A (In-Domain Test Split, N=2,700):** Evaluates core category separation. Target: Macro F1 $\ge 0.930$.
- **Benchmark B (Real-World Desktop Wild Corpus, N=200):** 150 real desktop files + 50 deliberate OOD files (recipes, memes, novels, blank scans).
- **Core Production Metric:**
  $$\text{False Automatic Sorting Rate (FASR)} = \frac{\text{Auto-Sorted Errors}}{\text{Total Auto-Sorted}} \times 100\% < \mathbf{1.5\%}$$
- **Manual Review Rate (MRR):** Target $15\% - 25\%$ staged in `_pending`.

---

## K. Explainable AI (XAI) Requirements

DocSort V2 must retain linear feature attribution without misleading users:
1. **Mathematical Grounding:** Attribute tokens via exact linear contribution:
   $$\text{contribution}_j = x_j \cdot w_{\hat{c}, j}$$
2. **Responsible Language:** Use phrasing such as:
   - *"These lexical features provided positive support for the model's decision."*
   - Never say: *"This word caused the classification."*
3. **Hyperplane Distance Transparency:** The UI breakdown table must clearly display signed decision scores with the explicit notice:
   *"Signed distances to class-separating hyperplanes, not probabilities."*

---

## L. Recommended V2 End-to-End Architecture

```
                                    USER DOCUMENT UPLOAD (PDF / DOCX / TXT)
                                                      │
                                                      ▼
                                       Document Extraction Pipeline
                                                      │
                            ┌─────────────────────────┴─────────────────────────┐
                            ▼                                                   ▼
                Text Length < 80 chars                               Text Length >= 80 chars
                            │                                                   │
                            ▼                                                   ▼
                OCR Fallback / Scanned PDF                            100k TF-IDF Vectorizer
                            │                                         (sublinear_tf=True)
                            ▼                                                   │
                Text Extraction Valid?                                          ▼
                   ├── NO ──► [Stage in _pending: Scanned/Unreadable]     LinearSVC (C=0.5)
                   └── YES                                                      │
                            │                                                   ▼
                            └───────────────────────────────────────► Decision Function: f_c(x)
                                                                                │
                                                      ┌─────────────────────────┴─────────────────────────┐
                                                      ▼                                                   ▼
                                         Gate 1: max f_c(x) < 0.15 ?                         Gate 2: f_top1 - f_top2 < 0.55 ?
                                                      │                                                   │
                                                     YES                                                 YES
                                                      │                                                   │
                                                      ▼                                                   ▼
                                         [Stage in _pending: OOD/Unknown]                    [Stage in _pending: Ambiguous Margin]
                                                      │                                                   │
                                                      NO                                                  NO
                                                      │                                                   │
                                                      └─────────────────────────┬─────────────────────────┘
                                                                                │
                                                                                ▼
                                                                  [AUTO-SORT: High Confidence]
                                                                ~/Desktop/SortedDocuments/<Cat>/
```

---

## M. What Should NOT Be Changed

To preserve scientific reproducibility and audit integrity:
1. **DO NOT modify or retrain:**
   - `research/models/linear_svc.joblib`
   - `research/models/tfidf_vectorizer.joblib`
   - Any model in `research/models/` or `research/ensemble_experiments/`
2. **DO NOT modify research reports or metrics:**
   - `research/results/`
   - `research/xai/`
   - `research/ensemble_experiments/reports/`
3. **DO NOT modify the research dataset:**
   - `data/research/docsort_research_dataset.csv`
   - `data/research/splits/train.csv`, `validation.csv`, `test.csv`
4. **DO NOT alter legacy models:**
   - `backend/multi_domain_doc_classifier/models/model.pkl`
   - `backend/multi_domain_doc_classifier/models/vectorizer.pkl`

---

## N. Exact Next Implementation Steps

When the user is ready to begin V2 development, execution should follow this strict phased sequence:

```
Phase 1: Sourcing & Assembly
├── 1.1 Create data/production_v2/raw/
├── 1.2 Download and curate 3,000 documents per class across 6 production categories
└── 1.3 Collect 200 real desktop files for the Real-World Wild Benchmark

Phase 2: Preprocessing & Leakage-Safe Splitting
├── 2.1 Deduplicate via SHA-256 and MinHash LSH (0.85 Jaccard threshold)
├── 2.2 Stratified 70/15/15 split (data/production_v2/splits/)
└── 2.3 Verify zero text overlap and zero URL overlap

Phase 3: Model Training & Threshold Optimization
├── 3.1 Fit 100k TF-IDF vectorizer strictly on Train split
├── 3.2 Train LinearSVC (C=0.5)
├── 3.3 Optimize dual-gate thresholds (tau_support, tau_margin) on Validation split for FASR < 1.5%
└── 3.4 Save artifacts to models/production_v2/

Phase 4: Two-Tier Evaluation
├── 4.1 Run In-Domain Test Benchmark (Benchmark A)
├── 4.2 Run Real-World Wild Benchmark (Benchmark B)
└── 4.3 Validate that FASR < 1.5% and OOD Rejection Recall >= 96%

Phase 5: Application Integration
├── 5.1 Update backend extraction with quality gate and OCR fallback
├── 5.2 Integrate dual-gated production_inference_v2.py
├── 5.3 Update React frontend taxonomy and decision margin visuals
└── 5.4 Live end-to-end user acceptance testing
```
