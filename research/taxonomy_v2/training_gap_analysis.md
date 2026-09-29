# DocSort AI — Training Data Gap Analysis (Taxonomy V2)

**Date**: September 2026  
**Status**: Data Composition & Gap Analysis Complete  
**Objective**: Audit the source distribution of `data/taxonomy_v2/dataset.csv` against real-world failure patterns to identify missing document structures and domain vocabulary.

---

## 1. Source Composition of current Taxonomy V2 Corpus

| Category | Total Docs | 20 Newsgroups | BBC News | IAB News | MTSamples | Reuters-21578 | Dominant Source Format |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Business & Finance** | 6,370 | 0 | 236 | 2,972 | 0 | 3,162 | Financial Newswire / Web News |
| **Entertainment** | 6,370 | 0 | 369 | 6,001 | 0 | 0 | Entertainment Web News |
| **Medical Health** | 6,370 | 751 | 0 | 3,684 | 1,935 | 0 | Medical Transcriptions & News |
| **Science & Academics** | 4,570 | 908 (`sci.space`) | 0 | 3,662 | 0 | 0 | Popular Space / Astronomy News |
| **Sports** | 6,370 | 922 | 238 | 5,210 | 0 | 0 | Sports Web News & Usenet |
| **Technology & Computing** | 8,170 | 5,013 (`comp.*`, `sci.crypt`, `sci.electronics`) | 247 | 2,910 | 0 | 0 | Usenet Technical Posts & Tech News |

---

## 2. Identified Training Data Gaps

### Gap 1: Total Absence of Academic & Educational Courseware in `Science & Academics`
- **Current Corpus**: 100% of `Science & Academics` training data comes from space news articles (`IAB News Science`) and Usenet astrophysics discussions (`sci.space`).
- **Missing Document Types**:
  1. University course syllabi (`Syllabus.pdf`)
  2. Academic lab manuals (`LAB MANUAL.pdf`)
  3. Lecture notes, courseware, and homework assignments
  4. Academic STEM research papers & textbook extracts
- **Impact**: Real-world academic documents containing code snippets (e.g. neural network lab exercises or microcontroller assembly programming) lack academic structure signals in the model feature weights, causing them to be misclassified as `Technology & Computing`.

### Gap 2: Lack of Structured Financial & Transaction Documents in `Business & Finance`
- **Current Corpus**: Composed entirely of commercial newswire reports (`Reuters-21578`) and corporate earnings news (`IAB News`).
- **Missing Document Types**:
  1. Transaction receipts (`Receipt_13Jul2025_130649.pdf`)
  2. Invoices, purchase orders, and billing statements
  3. Business strategy publications & management framework documents (`back-of-the-napkin.pdf`)
- **Impact**: Retail receipts lack newswire financial terms, leading to low decision margins or misclassification.

### Gap 3: Missing Structural Diversity in `Technology & Computing`
- **Current Corpus**: Heavy concentration of Usenet forum posts (`20 Newsgroups comp.*`) and short tech news summaries (`IAB News`).
- **Missing Document Types**:
  1. Formal technical software manuals and system specifications
  2. Source code files and developer API documentation
  3. Technical project synopses and software architecture proposals
- **Impact**: Resumes and CVs (`TANISH_ARORA_CV_OLD.pdf`) listing technical skills trigger high `Technology & Computing` margins because the model interprets dense keyword lists as computing text.
