# DocSort AI — Taxonomy V2 Training Data Expansion Plan

**Date**: September 2026  
**Status**: Data Architecture & Provenance Plan (Pre-Training Phase)  
**Objective**: Plan the augmentation and re-balancing of the DocSort AI dataset under **Taxonomy V2**, addressing the critical gap in genuine academic and educational material while preserving dataset balance and provenance compliance.

---

## 1. Assessment of Current Training Data & Necessity of Additional Data

### Is Additional Data Actually Necessary?
**YES, absolutely.**

1. **Category Deficit**: Re-mapping `sci.crypt` (644 docs) and `sci.electronics` (638 docs) from `Science` to `Technology & Computing` removes 1,282 documents from `Science & Academics`, leaving it at only ~5,088 documents while `Technology & Computing` expands to ~7,652 documents.
2. **Domain Representation Deficit**: The current `Science & Academics` pool consists almost exclusively of popular space news (`IAB Science`, `sci.space`) and general medical news. It contains **zero** academic courseware, university syllabi, lab manuals, or peer-reviewed academic abstracts.
3. **Real-World Model Performance**: Without genuine academic syntax (course objectives, grading schemes, lab experiments, academic research citations), the model misclassifies real-world academic files like `LAB MANUAL.pdf` or `Syllabus.pdf` into `Technology & Computing` if they contain any programming code.

---

## 2. Dataset Provenance, Sources & Available Volume

To construct a robust `Science & Academics` corpus without violating license constraints, we identify two primary open dataset sources:

### A. Candidate Academic Dataset Sources

1. **arXiv Dataset (`ccdv/arxiv-summarization` / `arxiv_dataset`)**
   - **Provenance & Source**: Open Access research paper abstracts and introductions across STEM disciplines (Physics, Computer Science, Mathematics, Biology).
   - **License**: Creative Commons CC0 / Public Domain / arXiv Open Access.
   - **Available Documents**: >100,000 documents.
   - **Proposed Selection**: 2,500 - 3,500 academic STEM abstracts/papers.

2. **Open Educational & Scientific Abstracts (`scientific_papers` / PubMed / BioMed)**
   - **Provenance & Source**: Peer-reviewed scientific literature and educational course synopses.
   - **License**: CC-BY 4.0 / Public Domain.
   - **Available Documents**: >50,000 documents.
   - **Proposed Selection**: 1,500 - 2,500 educational science papers.

3. **Academic Courseware & Syllabi Synthetic Baseline (Optional Supplement)**
   - Open university course syllabi and lab manual templates (MIT OpenCourseWare text extracts).
   - **Proposed Selection**: 500 - 1,000 structured academic syllabus/lab manual extracts.

---

## 3. Retained vs. Re-Mapped Data Inventory

| Existing Category | Retained Docs | Re-mapped Docs | New Source Addition | Target Balanced Class Size |
| :--- | :---: | :---: | :---: | :---: |
| **Technology & Computing** | 6,370 | +1,282 (`sci.crypt`, `sci.electronics`) | 0 (Sub-sampled to target) | **6,500** |
| **Science & Academics** | 5,088 | -1,282 (Re-mapped to Tech) | **+2,694** (arXiv + STEM Papers) | **6,500** |
| **Medical Health** | 6,370 | 0 | 0 | **6,500** |
| **Business & Finance** | 6,370 | 0 | 0 | **6,500** |
| **Entertainment** | 6,370 | 0 | 0 | **6,500** |
| **Sports** | 6,370 | 0 | 0 | **6,500** |
| **TOTAL CORPUS** | **36,938** | — | — | **39,000 (6 x 6,500)** |

---

## 4. Execution Rules & Constraints

1. **Strict Six-Class Balance**: The final expanded corpus will feature exactly 6,500 documents per category across all 6 classes (totaling 39,000 documents).
2. **No Seventh "Anonymous" Class**: The ML model will be trained strictly on the 6 defined classes. Anonymous handling will be governed by post-hoc decision margin thresholding on validation/OOD data.
3. **Freeze Original V1 & Expanded V1**: This new dataset will be created as `data/taxonomy_v2/` without altering or deleting `data/research/` or `data/research_expanded_v1/`.
4. **No Downloading Yet**: In accordance with instructions, dataset fetching and feature matrix generation will be executed in the subsequent training phase.
