# Production Dataset Specification for DocSort AI V2

This document specifies the architecture, sourcing, curation, and quality assurance standards for the DocSort AI V2 Production Dataset.

---

## 1. Research Dataset vs. Production Dataset: Fundamental Separation

> [!IMPORTANT]
> The existing research dataset (`data/research/docsort_research_dataset.csv` and splits under `data/research/splits/`) is part of the completed ML-2 research experiment. It must remain **strictly read-only and permanently frozen**.
>
> The Production Dataset V2 will be created in a completely new, dedicated directory:
> `data/production_v2/`

| Dimension | Research Dataset (Frozen) | Production Dataset V2 (Proposed) |
| :--- | :--- | :--- |
| **Origin** | Single web-scrape news corpus (`mdonigian/iab-news-classification`) | Multi-source corpus of real desktop, workplace, and academic documents |
| **Domain** | Journalism / News Reporting | Functional Desktop Document Organization |
| **Document Formats** | Web text only | PDF (native & scanned), DOCX, TXT |
| **Target Audience** | Academic research paper benchmarks | End-users managing desktop filesystem directories |
| **Class Count** | 6 news topics | 6 functional categories + 1 OOD evaluation set |
| **Size** | 25,800 documents (4,300 / class) | 15,000–18,000 documents (2,500–3,000 / class) |
| **Storage Location** | `data/research/` | `data/production_v2/` |

---

## 2. Target Categories & Scale Specifications

| Category ID | Category Name | Target Training Documents | Target Val Documents | Target Test Documents | Total Target Docs |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `CAT_1` | **Academic & Educational** | 2,100 | 450 | 450 | **3,000** |
| `CAT_2` | **Financial & Receipts** | 2,100 | 450 | 450 | **3,000** |
| `CAT_3` | **Administrative & Identification** | 2,100 | 450 | 450 | **3,000** |
| `CAT_4` | **Technology & Technical Docs** | 2,100 | 450 | 450 | **3,000** |
| `CAT_5` | **Medical & Healthcare** | 2,100 | 450 | 450 | **3,000** |
| `CAT_6` | **Career & Professional** | 2,100 | 450 | 450 | **3,000** |
| **Total** | **6 Balanced Classes** | **12,600** (70%) | **2,700** (15%) | **2,700** (15%) | **18,000** |

---

## 3. Approved Corpus Sources by Category

To prevent single-source bias, each production category must draw from at least **three distinct open datasets**:

### 1. Academic & Educational
- **ArXiv Open Access Preprints**: Academic papers in physics, computer science, mathematics, quantitative biology.
- **OpenStax Educational Textbooks**: Chapter excerpts across college biology, sociology, economics, physics.
- **MIT OpenCourseWare / University Repositories**: Public course syllabi, lecture slide transcriptions, assignment problem sets.

### 2. Financial & Receipts
- **SROIE (Scanned Receipts OCR and Information Extraction)**: Real-world retail and service transaction receipts.
- **CORD (Consolidated Receipt Dataset)**: Structured receipt scans and line-item payment records.
- **SEC EDGAR Filings**: 10-Q/10-K financial reports, balance sheets, and audit statements.
- **Synthetic Invoicing Engines**: Programmatically generated invoices with dynamic totals, tax IDs, and bill-to addresses.

### 3. Administrative & Identification
- **RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)**: Subsets for `form`, `letter`, and `memo`.
- **CUAD (Contract Understanding Atticus Dataset)**: Non-disclosure agreements, commercial contracts, lease agreements.
- **Public Government Circulars & Forms**: Open institutional notices, application templates, registration frameworks.

### 4. Technology & Technical Docs
- **GitHub Documentation / ReadTheDocs**: Developer API references, installation guides, framework documentation.
- **Linux Manual Pages (man pages)**: Systems programming, network administration, CLI utilities.
- **Open-Source Code Repositories**: Python, JavaScript, C++ annotated source code files and exercise modules.

### 5. Medical & Healthcare
- **MTSamples**: De-identified medical transcription reports (consultations, discharge summaries, progress notes).
- **PubMed Central Open Access**: Clinical case reports and pathology diagnostic summaries.
- **De-identified Laboratory Panels**: Standard reference range test reports (CBC, metabolic panels).

### 6. Career & Professional
- **Kaggle Resume Corpus**: Curated, de-identified resumes across engineering, finance, healthcare, and education.
- **Open Job Offer & Appointment Templates**: Corporate offer letters, compensation summaries, recommendation letters.
- **Performance Evaluation Templates**: Standard corporate appraisal forms and self-assessment summaries.

---

## 4. Multi-Format Representation & Preprocessing Standards

### 4.1 Document Format Distribution
The training and evaluation sets must intentionally represent the three formats supported by DocSort:
- **PDF (60%)**: 40% digital vector PDFs (native text), 20% OCR-extracted scanned PDFs.
- **DOCX (25%)**: Word documents with headings, paragraphs, and lists.
- **TXT (15%)**: Plain ASCII / UTF-8 text files and source code.

### 4.2 Document Length & Truncation Thresholds
- **Minimum Length**: Documents with fewer than **100 characters** of extractable text are rejected as non-informative (or routed to OCR fallback).
- **Maximum Length Handling**:
  - Rather than hard-truncating long documents (which discards crucial terms at the end of 50-page contracts or syllabi), use **sublinear term frequency scaling** (`sublinear_tf=True` in TF-IDF):
    $$\text{TF}_{\text{sublinear}} = 1 + \log(\text{TF})$$
  - Cap maximum analyzed characters at **50,000 characters** (~8,000 words) for memory and throughput efficiency.

### 4.3 Data Cleaning & Deduplication
1. **Cryptographic Deduplication**: Compute SHA-256 hash of extracted text. Remove exact duplicates within and across classes.
2. **Fuzzy Near-Duplicate Removal**: Apply MinHash LSH with a Jaccard threshold of **0.85** on character 5-grams. This eliminates nearly-identical template copies while retaining distinct instances.
3. **De-identification**: Strip personal phone numbers, physical street addresses, and social security/national identity numbers using regex scrubbing.

---

## 5. Leakage-Safe Stratified Splitting Protocol

```
Full Curated Corpus (18,000 unique docs)
    │
    ▼
Deduplication & Quality Filter (Strict SHA-256 + MinHash)
    │
    ▼
Stratified Split (random_state=42)
    ├── 70% TRAIN (12,600 docs) ──► Fit TF-IDF Vectorizer ONLY HERE
    │
    ├── 15% VALIDATION (2,700 docs) ──► Transform ONLY (Tune thresholds)
    │
    └── 15% TEST (2,700 docs) ──────► Transform ONLY (Evaluate ONCE)
```

**Golden Rules**:
- The TF-IDF vectorizer must be fitted **strictly on the Train split**.
- No text transformation, normalizer tuning, or threshold selection may use Validation or Test data.
- Zero URL and zero text overlap across splits.

---

## 6. Real-World Evaluation Benchmark (Out-of-Dataset)

In addition to the 15% held-out test split, a dedicated **Real-World Desktop Benchmark** (`data/production_v2/desktop_wild_test/`) of **200 real documents** collected from actual users will be maintained:
- Includes 150 target-class documents (messy receipts, syllabi, slides, resumes).
- Includes 50 true out-of-domain documents (coupons, memes, scanned blank pages, food recipes, novels) to evaluate the dual-gated OOD rejection mechanism.
