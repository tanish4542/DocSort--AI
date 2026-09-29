# DocSort AI — Next Training Data Expansion Plan (Taxonomy V2)

**Date**: September 2026  
**Status**: Data Architecture & Augmentation Roadmap (Pre-Training Phase)  
**Objective**: Define the precise, minimal set of real-world document formats and open dataset sources required to address training gaps without expanding the six-class taxonomy.

---

## 1. Priority 1: Science & Academics Augmentation (Highest Priority)

### Target Document Formats Needed
- **University Course Syllabi**: Course descriptions, credit hours, prerequisites, grading rubrics, lecture schedules.
- **Academic Lab Manuals**: Experiment protocols, lab safety instructions, programming lab assignments, neural network/circuit exercises.
- **Courseware & Textbook Material**: Chapter excerpts, lecture slides, academic homework assignments.
- **Peer-Reviewed STEM Papers & Abstracts**: Academic papers from physics, biology, chemistry, and computer science.

### Recommended Open Data Sources
1. **arXiv STEM Abstracts Dataset** (`ccdv/arxiv-summarization` / `arxiv_dataset`): 3,000–5,000 academic research abstracts.
2. **Open Courseware & University Syllabi Repositories**: 500–1,000 open courseware extracts (MIT OCW, university public syllabi).

---

## 2. Priority 2: Technology & Computing Augmentation

### Target Document Formats Needed
- **Technical Software Documentation**: API references, system architecture manuals, developer documentation, installation guides.
- **Source Code Snippets & Codebases**: Python, C++, Java, and Assembly technical documentation.
- **Software Project Proposals**: Technical project synopses, system design documents.

### Recommended Open Data Sources
1. **GitHub Technical Documentation & Readmes**: 2,000–3,000 open-source technical documentations and developer guides.
2. **StackOverflow / Technical Documentation Datasets**: Structured software architecture descriptions.

---

## 3. Priority 3: Business & Finance Augmentation

### Target Document Formats Needed
- **Invoices & Payment Receipts**: Itemized billing statements, transaction receipts, purchase orders.
- **Corporate & Financial Reports**: Quarterly financial summaries, executive strategy publications, business proposals.

### Recommended Open Data Sources
1. **CORD / SROIE / Receipts Datasets**: 1,500–2,000 anonymized OCR text receipts and invoices.
2. **Edgar SEC Corporate Filings**: Executive business summary reports.

---

## 4. Secondary Priorities: Medical Health, Sports & Entertainment

- **Medical Health**: Current clinical transcription pool (`MTSamples`) is strong. Supplement with 500 patient care guides & medical diagnosis articles.
- **Sports & Entertainment**: Current news coverage (`IAB News`, `BBC`) is highly effective (100% real-world accuracy). Retain current corpus composition.

---

## 5. Execution Rules for Next Iteration

1. **Strict Six-Class Balance**: Maintain exactly equal document counts across all 6 classes (e.g. 7,000 documents per class).
2. **No Seventh "Anonymous" Class**: Keep Anonymous as an operational margin threshold rule.
3. **No Retraining or Downloads in Current Phase**: All data acquisition and retraining will take place in the subsequent iteration following validation.
