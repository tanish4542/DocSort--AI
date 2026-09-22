# Production V2 Dataset Compatibility Report

**Project:** DocSort AI — Production Document Classification Engine (V2)  
**Date:** September 2026  
**Status:** Pre-Implementation Feasibility Study & Dataset Audit  
**Author:** Antigravity AI Engineering Team  

---

## 1. Objective

This report investigates whether publicly available, open-licensed datasets can legitimately support the training, validation, and evaluation of **DocSort AI V2**, a functional desktop document classification and automated sorting engine.

### The Separation of Concerns
1. **The Frozen Research Benchmark (ML-2 Experiment)**:
   - **Corpus**: `data/research/docsort_research_dataset.csv` (25,800 news articles from `mdonigian/iab-news-classification`).
   - **Model**: `research/models/linear_svc.joblib` ($C=0.5$, 100,000 TF-IDF features).
   - **Metrics**: $94.50\%$ test accuracy, $0.9449$ macro F1 on 6 news categories.
   - **Status**: **Permanently frozen** as part of an IEEE conference paper contribution. It must never be retrained, overwritten, or modified.
2. **The Production Document Sorting Problem**:
   - **Goal**: Classify real desktop documents (receipts, syllabi, coding assignments, ID cards, slide decks, lab reports, resumes) by **functional organizational destination**, not topical news beats.
   - **Failure of News Baseline**: In live desktop testing, a tuition payment receipt (`PDFPrint.pdf`) was misclassified into `Technology & Computing` because all six news hyperplanes were negative ($\max f_c(x) = -0.0167$), but unconstrained relative margin forced a "least-negative" auto-sort.
   - **Objective of V2**: Build a dedicated production model trained on real document formats with an explicit rejection/OOD safety mechanism.

---

## 2. Target Production Taxonomy

DocSort V2 operates on a **6-class functional taxonomy** plus a safety rejection queue:

```
DocSort V2 Operational Taxonomy
├── 1. Academic & Educational        (Trainable Class)
├── 2. Financial & Receipts          (Trainable Class)
├── 3. Administrative & ID           (Trainable Class)
├── 4. Technology & Technical Docs   (Trainable Class)
├── 5. Medical & Healthcare          (Trainable Class)
├── 6. Career & Professional         (Trainable Class)
└── 7. Unassigned / Needs Review     (OOD Rejection Queue - Staged in _pending)
```

> [!IMPORTANT]
> **Functional Utility vs. Topic Keywords**:
> - A resume for a software engineer belongs to **`Career & Professional`**, NOT `Technology & Computing`.
> - A college fee payment receipt belongs to **`Financial & Receipts`**, NOT `Academic & Educational`.
> - A computer science course syllabus belongs to **`Academic & Educational`**, NOT `Technology & Computing`.
> - A medical journal research paper belongs to **`Academic & Educational`** (scholarly paper format) or an evaluation benchmark, NOT clinical `Medical & Healthcare`.

---

## 3. Dataset Compatibility Matrix Summary

We investigated 12 publicly available datasets without downloading unnecessary bulk archives. The complete tabular data is recorded in [`dataset_compatibility_matrix.csv`](file:///Users/tanish4542/CODE%20AND%20PROJECTS/DocSort-AI/research/production_model_audit/dataset_compatibility_matrix.csv).

| Dataset | Native Format | Usable Docs | Text Available? | OCR Needed? | Label Type | Primary Production Role | Confidence |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **RVL-CDIP** | Grayscale TIFF/JPEG | ~150,000 | No | **Yes** | Human / Source metadata | Benchmark & Selective Pre-OCR Sampling | Medium |
| **SROIE (2019)** | JPG + TXT / JSON | 973 | **Yes** (Transcribed) | No | Human Verified OCR | Primary Train & Test (`Financial & Receipts`) | High |
| **CORD (v2)** | JPG + JSON | 1,000 | **Yes** (JSON tokens) | No | Human Annotated | Multilingual / Cross-Lingual Benchmark Only | Medium |
| **MTSamples** | CSV (Plain Text) | 4,966 | **Yes** (Native text) | No | Source-derived (Transcriptionist) | Primary Train & Val (`Medical & Healthcare`) | High |
| **PubMed Central OA**| JATS XML / PDF / TXT | Millions | **Yes** (API / S3) | No | Publisher metadata | Academic Training & Medical Benchmark | High |
| **OpenStax** | PDF / HTML / CNXML | ~1,200 | **Yes** (Native text) | No | Expert Authors / Peer Reviewed | Primary Train & Val (`Academic & Educational`) | High |
| **arXiv Preprints** | PDF / TeX / JSON | Millions | **Yes** (API / S3) | No | Author / Moderator metadata | Training Sample (`Academic & Educational`) | High |
| **GitHub Tech Docs** | Markdown / RST | Millions | **Yes** (Native text) | No | Source-derived (Repository docs) | Primary Train & Val (`Technology & Tech Docs`) | High |
| **Resume Dataset** | CSV (Plain Text) | ~1,000–2,400 | **Yes** (Native text) | No | Human Curated / De-identified | Primary Train & Val (`Career & Professional`) | High |
| **CareerCorpus** | Excel / Text | 302 | **Yes** (Native text) | No | Dual Human Expert Verified | Gold-Standard Test / Benchmark Only | High |
| **CUAD (Atticus)** | PDF / TXT / CSV | 510 contracts | **Yes** (Plain TXT) | No | Legal Expert Annotated (Stanford) | Primary Train & Test (`Administrative & ID`) | High |
| **superdoc/docx** | Word DOCX (Parquet) | ~250,000 (EN) | **Yes** (Docling text) | No | **Automatic (XLM-RoBERTa)** | Multi-Class Augmented Training (Filtered) | High |

---

## 4. Detailed Dataset Assessment

### 4.1 RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)
- **Source & URL**: [aharley/rvl_cdip on Hugging Face](https://huggingface.co/datasets/aharley/rvl_cdip) / Carnegie Mellon University.
- **License**: CC0 / Public Domain (derived from the Legacy Tobacco Documents Library).
- **Size & Count**: 400,000 grayscale images (38 GB archive); 25,000 images per class across 16 categories.
- **Taxonomy Alignment**:
  - `invoice`, `budget` $\to$ **`Financial & Receipts`**
  - `form`, `questionnaire`, `memo` $\to$ **`Administrative & Identification`**
  - `scientific publication`, `scientific report`, `presentation` $\to$ **`Academic & Educational`**
  - `specification` $\to$ **`Technology & Technical Docs`**
  - `resume` $\to$ **`Career & Professional`**
  - `advertisement`, `handwritten`, `news article` $\to$ **`Unassigned / OOD Evaluation`**
- **OCR Status**: **Image-only**. Native text is not stored in the dataset. Running OCR across 400,000 images would require weeks of compute and introduces OCR transcription noise.
- **Verdict**: **Do NOT download in bulk.** Suitable as an offline visual/OCR benchmark or by sampling a pre-OCR'd research subset (e.g., 500 documents per target class).

---

### 4.2 SROIE (ICDAR 2019 Scanned Receipts OCR and Information Extraction)
- **Source & URL**: [jsdnrs/ICDAR2019-SROIE on Hugging Face](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE).
- **License**: CC-BY-4.0.
- **Size & Count**: 1,000 receipt images (626 train, 347 test); ~2 GB with images.
- **Format & Text Availability**: Images (JPG) accompanied by plain-text OCR annotations (`.txt` files containing bounding boxes and transcribed string tokens) and key information labels (`company`, `date`, `address`, `total`).
- **Taxonomy Alignment**: 100% maps to **`Financial & Receipts`**.
- **OCR Requirement**: **None for text modeling**. The pre-extracted text annotations provide verified string transcriptions of retail receipts.
- **Limitations**: Geographically centered in Southeast Asia (stores, addresses, Malaysian Ringgit currency). Receipts are short and token-sparse.
- **Verdict**: **Definitely use.** Essential anchor dataset for real retail receipts in `Financial & Receipts`.

---

### 4.3 CORD (Consolidated Receipt Dataset for Post-OCR Parsing)
- **Source & URL**: [naver-clova-ix/cord-v2 on Hugging Face](https://huggingface.co/datasets/naver-clova-ix/cord-v2).
- **License**: CC-BY-4.0.
- **Size & Count**: 1,000 structured receipts in official benchmark split (800 train, 100 val, 100 test); ~11,000 in full collection.
- **Format & Text Availability**: Images accompanied by rich JSON annotations containing word tokens, bounding boxes, and hierarchical parse trees (`menu`, `total`, `subtotal`).
- **Taxonomy Alignment**: Formally maps to `Financial & Receipts`.
- **Major Limitation**: **Language is Bahasa Indonesia (Indonesian)** with Indonesian currency (Rupiah/Rp), store names, and food items.
- **Verdict**: **Do NOT use for primary English training.** Training on Indonesian receipts would induce severe cross-lingual bias and false feature weighting. Retain solely as a cross-lingual OOD evaluation benchmark.

---

### 4.4 MTSamples (Medical Transcription Samples)
- **Source & URL**: [tboyle10/medical-transcriptions on Kaggle](https://www.kaggle.com/datasets/tboyle10/medical-transcriptions) / [mtsamples.com](https://www.mtsamples.com).
- **License**: CC0: Public Domain.
- **Size & Count**: 4,966 non-empty clinical transcription reports (~14 MB CSV).
- **Format & Text Availability**: Clean tabular plain text. Columns: `description`, `medical_specialty`, `sample_name`, `transcription`, `keywords`.
- **Taxonomy Alignment**: 100% maps to **`Medical & Healthcare`**. Covers 40 authentic clinical specialties (Surgery, History & Physical, Consultations, Cardiology, Radiology, Discharge Summaries).
- **OCR Requirement**: Zero.
- **Verdict**: **Definitely use.** The single best open-access source for authentic outpatient notes, clinical histories, and healthcare provider documents.

---

### 4.5 PubMed Central (PMC) Open Access Subset
- **Source & URL**: National Library of Medicine / [AWS Registry of Open Data](https://registry.opendata.aws/pmc/).
- **License**: Variable open licenses per article (CC0, CC-BY, CC-BY-NC, CC-BY-ND).
- **Size & Count**: Millions of full-text papers (>4 million articles; Terabytes).
- **Functional Misconception Warning**: A PubMed Central article is a peer-reviewed **scientific research paper** (IMRaD structure: Abstract, Introduction, Materials & Methods, Results, Discussion, References). In a desktop sorter, users file research papers under **`Academic & Educational`**, not clinical patient records!
- **Verdict**: **Use selectively for Academic & Educational (or scientific benchmarking).** Do NOT use as the primary training data for `Medical & Healthcare`, as it replicates the exact news topic-bleed bug.

---

### 4.6 OpenStax Textbooks
- **Source & URL**: [OpenStax / Rice University](https://openstax.org).
- **License**: CC BY 4.0 / CC BY-NC-SA 4.0.
- **Size & Count**: 50+ college-level textbooks (~1,200 to 1,500 distinct chapter units).
- **Format & Text Availability**: Digital native PDFs, web HTML, and structured CNXML. Clean extractable text without OCR.
- **Taxonomy Alignment**: 100% maps to **`Academic & Educational`** across all subjects (Calculus, Physics, Anatomy, Sociology, Macroeconomics, US History, Business Law).
- **Verdict**: **Definitely use.** High-quality, authoritative educational text representing textbook chapters, learning outcomes, and academic instruction.

---

### 4.7 arXiv Open-Access Scholarly Papers
- **Source & URL**: [arXiv.org](https://arxiv.org) / Kaggle `arxiv-metadata-oai-snapshot`.
- **License**: Varied per paper; metadata is CC0; ~25% of full-text submissions explicitly license under CC-BY or CC-BY-SA.
- **Size & Count**: 2.5+ million papers (metadata ~5 GB; full text multi-terabytes).
- **Taxonomy Alignment**: Maps to **`Academic & Educational`** (scholarly preprints) or held-out scientific benchmarks.
- **Verdict**: **Use a small licensed sample (1,000–2,000 CC-BY papers)** to represent graduate-level academic publications. Do not download the multi-terabyte bulk archive.

---

### 4.8 GitHub Technical Documentation Corpus
- **Source & URL**: [The Stack (BigCode / Hugging Face)](https://huggingface.co/datasets/bigcode/the-stack) / ReadTheDocs.
- **License**: Permissive Open Source (MIT, Apache 2.0, BSD, ISC).
- **Size & Count**: Millions of documentation files; sampling 3,000–5,000 documents is straightforward.
- **Format & Text Availability**: Plain text, Markdown (`.md`), and reStructuredText (`.rst`).
- **Taxonomy Alignment**: 100% maps to **`Technology & Technical Docs`** (installation guides, developer READMEs, API specifications, CLI manual guides).
- **Verdict**: **Definitely use.** Authentic software engineering, systems administration, and programming documentation.

---

### 4.9 Resume Classification Dataset for NLP
- **Source & URL**: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) / [UpdatedResumeDataSet](https://huggingface.co/datasets/DevashishBhake/resume_section_classification).
- **License**: CC0 / Community Shared.
- **Size & Count**: ~1,000 to 2,400 de-identified resumes in CSV format.
- **Taxonomy Alignment**: All occupational labels (e.g., Data Science, HR, Sales, Civil Engineering) **collapse into a single functional class: `Career & Professional`**.
- **OCR Requirement**: Zero.
- **Verdict**: **Definitely use.** Directly models candidate profiles, work histories, and career summaries.

---

### 4.10 CareerCorpus (Mendeley Data)
- **Source & URL**: [Mendeley Data: CareerCorpus](https://data.mendeley.com/datasets/p95m24c3n8/1).
- **License**: CC-BY-4.0.
- **Size & Count**: Exactly 302 resumes across 6 occupational fields.
- **Annotation Quality**: **Gold Standard**: Dual human expert annotations with PII anonymization.
- **Limitations**: Too small (302 documents) for standalone model training.
- **Verdict**: **Reserve exclusively for held-out evaluation & testing.** Serves as a high-integrity benchmark for `Career & Professional`.

---

### 4.11 CUAD (Contract Understanding Atticus Dataset)
- **Source & URL**: [theatticusproject/cuad on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad).
- **License**: CC BY 4.0.
- **Size & Count**: 510 commercial contracts, non-disclosure agreements, and licensing pacts.
- **Format & Text Availability**: Both native PDF and pre-extracted plain TXT provided.
- **Taxonomy Alignment**: Maps to **`Administrative & Identification`** (formal contracts, NDAs, corporate agreements).
- **Annotation Quality**: Expert legal annotation (supervised by Stanford Law).
- **Verdict**: **Definitely use.** Represents formal institutional agreements and legal contracts with zero OCR required.

---

### 4.12 superdoc / docx-corpus
- **Source & URL**: [superdoc-dev/docx-corpus on Hugging Face](https://huggingface.co/datasets/superdoc-dev/docx-corpus) / [GitHub](https://github.com/superdoc/docx-corpus).
- **License**: ODC-BY (Open Data Commons Attribution).
- **Size & Count**: 736,000+ `.docx` files collected from Common Crawl (~250,000 English documents).
- **Taxonomy Alignment**:
  - `educational` $\to$ **`Academic & Educational`**
  - `technical` $\to$ **`Technology & Technical Docs`**
  - `forms`, `legal`, `policies`, `administrative` $\to$ **`Administrative & Identification`**
  - `reports` (finance subset) $\to$ **`Financial & Receipts`**
- **CRITICAL LABEL QUALITY FINDING**: Labels in `docx-corpus` were **automatically generated using XLM-RoBERTa pseudo-labeling** and pattern heuristics. **They are NOT human-verified ground truth.**
- **Verdict**: **Use as an augmented multi-class corpus with strict filtering.** Filter to high-confidence English documents ($>500$ words, confidence $>0.85$). Do not rely on it without quality checks.

---

## 5. Label Quality Assessment

To ensure scientific honesty and prevent model degradation, we classify candidate sources by label provenance:

```
                            LABEL PROVENANCE HIERARCHY
                                        │
    ┌───────────────────────────────────┼───────────────────────────────────┐
    ▼                                   ▼                                   ▼
LEVEL 1: GOLD EXPERT             LEVEL 2: SOURCE-DERIVED             LEVEL 3: AUTOMATIC / WEAK
Human / Expert Annotated         Natural Document Provenance         Model-Generated Pseudo-Labels
├── CUAD (Stanford Law)          ├── SROIE (Store Receipts)          └── docx-corpus (XLM-RoBERTa)
├── CareerCorpus (Dual Expert)   ├── MTSamples (Medical Transcripts)
├── OpenStax (Peer Review)       ├── GitHub Docs (Repo Documentation)
└── SROIE OCR (Verified)         ├── OpenStax (Published Textbooks)
                                 └── arXiv / PMC (Journal Metadata)
```

**Quality Constraint for V2**:
- All evaluation and test splits must use **Level 1 or Level 2** labels only.
- Level 3 (pseudo-labeled) data from `docx-corpus` may only be used to augment training data after automated confidence filtering and manual sanity audits.

---

## 6. Format and OCR Assessment

A core defect of the legacy system was assuming that all PDFs contain clean ASCII text:

```
Upload Stream
    │
    ▼
Check Text Length: len(text.strip())
    ├── < 80 characters ────────► [REJECT / FLAG: Scanned Image or Corrupted PDF]
    │                             (Do not vectorize empty noise)
    └── >= 80 characters ───────► Calculate Printable Character Ratio: printable / len(text)
                                       ├── < 0.60 ──► [FLAG: Binary Garbage / Corrupted Font]
                                       └── >= 0.60 ─► [PROCEED to TF-IDF Vectorizer]
```

### OCR Feasibility in V2:
- **Offline Dataset Preparation**: SROIE, CORD, and CUAD provide pre-transcribed text, so **no offline OCR compute is needed** to prepare the V2 training corpus.
- **Runtime Application Pipeline**: For scanned PDFs (such as `ID card 25 Mar 2025.pdf`), the application should incorporate a lightweight OCR fallback (Tesseract on page 1) when extracted text is $<80$ characters. If OCR is unavailable, the document must be safely staged in `_pending` with status `"Scanned Image — Manual Review Required"`.

---

## 7. Licensing Assessment

All recommended sources permit academic research and open development:
- **CC0 / Public Domain**: MTSamples, RVL-CDIP, Kaggle Resume Dataset.
- **CC-BY-4.0**: SROIE, CORD, CareerCorpus, CUAD, OpenStax. (Requires author attribution in the research report/paper).
- **ODC-BY**: `superdoc/docx-corpus`. (Permits commercial and non-commercial reuse with attribution).
- **Permissive Open Source**: GitHub technical documentation (MIT, Apache 2.0, BSD).
- **Variable**: arXiv and PubMed Central require filtering to records with explicit Creative Commons licenses before inclusion.

---

## 8. Cross-Dataset Overlap, Syndication, and Leakage Risks

When combining heterogeneous public datasets, data leakage can occur through three vectors:

1. **Template Boilerplate Leakage**:
   - Resumes from Kaggle share identical template formatting (e.g., "Objective", "Declaration", "Languages").
   - GitHub READMEs share identical badge and build boilerplate (e.g., "Build Status", "License: MIT", "Contributors").
   - *Mitigation*: Strip common web markdown badges and standard header boilerplate during text cleaning.
2. **Exact & Near-Duplicate Contamination**:
   - Web-scraped repositories and Kaggle forks frequently duplicate identical resumes or receipt images.
   - *Mitigation*: Apply **SHA-256 exact hashing** on normalized lowercase text, followed by **MinHash LSH deduplication** at a Jaccard threshold of **0.85**.
3. **Cross-Split Leakage**:
   - Deduplication and normalization must be executed **prior to splitting**.
   - The TF-IDF vectorizer must be fitted **strictly on the Train split**.
   - Documents originating from the same organization, author, or GitHub repository must be grouped into the same split (`GroupKFold` / group stratification) to prevent intra-domain memorization.

---

## 9. Recommended Production V2 Corpus

### Reality Check: Why 18,000 Documents Is Unrealistic for a High-Quality V2
Our initial design report tentatively suggested 18,000 documents (3,000/class). However, our empirical audit reveals that:
- Clean, human-verified receipts (SROIE) number **under 1,000 documents**.
- Clean, de-identified resumes (Kaggle/CareerCorpus) number **~1,200 to 2,000 documents**.
- Formal commercial contracts (CUAD) number **510 documents**.

Forcing 3,000 documents per class would require oversampling, synthetic hallucination, or relying heavily on noisy pseudo-labels from `docx-corpus`.

### The Defensible V2 Corpus: 1,500 Documents Per Class (9,000 Total)
We recommend a balanced, high-integrity target of **1,500 clean documents per class** (Total: **9,000 documents** across 6 classes):

| Category | Primary Sources | Document Count | Format Mix |
| :--- | :--- | :---: | :--- |
| **1. Academic & Educational** | OpenStax (600) + arXiv CC-BY (600) + docx-corpus educational (300) | **1,500** | PDF, DOCX, TXT |
| **2. Financial & Receipts** | SROIE receipts (800) + SEC EDGAR invoices/statements (400) + docx-corpus finance (300) | **1,500** | TXT (OCR), PDF, DOCX |
| **3. Administrative & ID** | CUAD contracts (500) + docx-corpus forms/policies (800) + government circulars (200) | **1,500** | PDF, TXT, DOCX |
| **4. Technology & Tech Docs** | GitHub markdown docs (900) + docx-corpus technical (400) + Linux man pages (200) | **1,500** | MD/TXT, DOCX |
| **5. Medical & Healthcare** | MTSamples clinical transcriptions (1,200) + clinical case notes (300) | **1,500** | CSV/TXT, PDF |
| **6. Career & Professional** | Kaggle Resume dataset (1,200) + CareerCorpus (300) | **1,500** | CSV/TXT, PDF |
| **Total** | **6 Balanced Functional Classes** | **9,000** | **Multi-Format** |

---

## 10. Recommended Pilot Study (500 Documents Per Class)

Before curating 9,000 documents, we strongly recommend building an **Initial Pilot Corpus of 3,000 documents (500 per class)**:

- **Academic & Educational (500)**: 250 OpenStax textbook chapters + 250 arXiv preprints.
- **Financial & Receipts (500)**: 500 SROIE receipt text transcriptions.
- **Administrative & ID (500)**: 400 CUAD commercial contracts + 100 government circulars.
- **Technology & Tech Docs (500)**: 500 GitHub repository technical documentation guides.
- **Medical & Healthcare (500)**: 500 MTSamples clinical transcription reports.
- **Career & Professional (500)**: 500 Kaggle de-identified resumes.

### Purpose of the Pilot:
1. Verify that a lightweight **LinearSVC** trained on 500 functional documents per class can successfully separate receipts from technical code and academic syllabi.
2. Test the live test documents (`PDFPrint.pdf`, `Python Basic Programs.pdf`, `morphological_Image_Processing.pdf`) against the pilot model to confirm that the tuition receipt routes to `Financial & Receipts` instead of `Technology`.

---

## 11. Recommended Train / Validation / Test Strategy

For the production dataset, we recommend a stratified split with group-level integrity:
- **70% TRAIN (6,300 docs in full / 2,100 in pilot)**: Used strictly for vocabulary fitting and LinearSVC hyperparameter learning.
- **15% VALIDATION (1,350 docs in full / 450 in pilot)**: Used for threshold calibration ($\tau_{\text{support}}$ and $\tau_{\text{margin}}$ optimization).
- **15% TEST (1,350 docs in full / 450 in pilot)**: Held-out in-domain evaluation.

### Separate Real-World "Wild" Benchmark (200 Documents)
Completely independent from the 9,000-document training split, a **Wild Benchmark** of 200 real desktop files will be maintained:
- 150 real workplace/student documents across all 6 classes.
- 50 deliberate OOD negative controls (food recipes, boarding passes, scanned art, corrupted PDFs).
- Used solely to evaluate **False Automatic Sorting Rate (FASR < 1.5%)** and **OOD Rejection Recall ($\ge 96\%$)**.

---

## 12. OOD / Rejection Strategy

Desktop document organization is an **open-world problem**. A user will inevitably upload files that do not belong to any of the 6 functional categories (e.g., a scanned restaurant menu, an airline boarding pass, a creative writing novel, or an architectural CAD export).

Forcing such documents into a closed 6-class `argmax` guarantees false classification.

### The Dual-Gated Decision Rule:
$$\text{Decision} = \begin{cases}
\text{Auto-Sort into Class } c^* & \text{if } f_{c^*}(x) \ge \tau_{\text{support}} \text{ AND } f_{c^*}(x) - f_{\text{runner-up}}(x) \ge \tau_{\text{margin}} \\
\text{Stage in } \texttt{\_pending} \text{ (OOD)} & \text{if } \max_c f_c(x) < \tau_{\text{support}} \\
\text{Stage in } \texttt{\_pending} \text{ (Ambiguous)} & \text{if } f_{\text{top1}}(x) - f_{\text{top2}}(x) < \tau_{\text{margin}}
\end{cases}$$

Where:
- $\tau_{\text{support}} \approx +0.15$ (guarantees that at least one hyperplane actively claims the document).
- $\tau_{\text{margin}} \approx +0.55$ (guarantees clear separation from the second-best class).

---

## 13. Final Source Recommendations

| Recommendation Tier | Datasets | Rationale |
| :--- | :--- | :--- |
| **DEFINITELY USE (Tier 1 Core)** | **SROIE**, **MTSamples**, **OpenStax**, **GitHub Docs**, **Resume Dataset**, **CUAD** | High-quality text-native documents, clean permissive licensing, zero OCR required, directly represent functional desktop document types. |
| **POSSIBLY USE (Tier 2 Augmentation)** | **superdoc/docx-corpus**, **arXiv (CC-BY)** | Valuable for DOCX format diversity and academic papers; requires automated filtering to remove noisy pseudo-labels and non-English text. |
| **BENCHMARK / EVALUATION ONLY** | **CareerCorpus**, **CORD (v2)**, **RVL-CDIP (Pre-OCR Subset)** | *CareerCorpus* has gold-standard dual-expert labels (ideal test set). *CORD* evaluates cross-lingual receipt handling. *RVL-CDIP* evaluates visual/OCR robustness. |
| **AVOID FOR PRIMARY TRAINING** | **PubMed Central (Bulk)**, **CORD (Raw)** | Bulk PMC causes severe topical bleed (research papers $\ne$ clinical records). Raw CORD introduces Indonesian language bias into an English classifier. |

---

## 14. Summary & Next Action

This audit confirms that a **Production V2 Document Classifier is entirely feasible** using existing open-access datasets, without requiring expensive labeling campaigns or heavy deep-learning infrastructure.

**CRITICAL SAFEGUARDS CONFIRMED**:
- The ML-2 research model, dataset, results, and XAI code remain 100% frozen and untouched.
- No production files or models were modified.
- No bulk downloads were executed.

**Next Immediate Step**: When approved by the user, build the **3,000-document Pilot Corpus (500 docs/class)** across the 6 Tier-1 datasets to validate the functional taxonomy and dual-gated margin engine.
