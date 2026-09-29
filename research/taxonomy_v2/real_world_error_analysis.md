# DocSort AI — Real-World Error Analysis & Margin Audit (Taxonomy V2)

**Date**: September 2026  
**Status**: Completed Error & Margin Analysis  
**Objective**: Diagnose real-world classification failures, analyze decision margin distributions, and evaluate OOD rejection threshold mechanics following the Fast-Track Taxonomy V2 experiment.

---

## 1. Itemized Real-World Document Evaluation Table

The 18 real-world evaluation documents were evaluated against the frozen LinearSVC model trained on Taxonomy V2. Ground-truth labels from `document_manifest.csv` were normalized to match Taxonomy V2 category names (`Business and Finance` $\rightarrow$ `Business & Finance`, `Science` $\rightarrow$ `Science & Academics`).

| # | Filename | Ground Truth (Normalized) | Known / OOD | LinearSVC Prediction | Decision Margin | Correct? | Error / Classification Notes |
| :-: | :--- | :--- | :-: | :--- | :-: | :-: | :--- |
| 1 | `Air India Web Booking eTicket (9WSBWJ) - TANISH.pdf` | `OOD` | OOD | `Technology & Computing` | 1.0296 | — | Airline booking confirmation containing technical system tags |
| 2 | `ID card 25 Mar 2025.pdf` | `OOD` | OOD | `Technology & Computing` | 0.6580 | — | Student identity card with short text |
| 3 | `LAB MANUAL.pdf` 🔴 | `Science & Academics` | Known | `Technology & Computing` | **2.3452** | **FALSE** | **MISCLASSIFIED**: Neural network lab exercises contain Python code, causing Tech override |
| 4 | `MANVI_BIODATA.pdf` | `OOD` | OOD | `Medical Health` | 0.6038 | — | Marriage biodata document |
| 5 | `Major Project Synopsis.pdf` 🟢 | `Technology & Computing` | Known | `Technology & Computing` | **2.9271** | **TRUE** | Computer science project proposal on AI software architecture |
| 6 | `Receipt_13Jul2025_130649.pdf` | `OOD` | OOD | `Business & Finance` | 0.2056 | — | Retail payment transaction receipt |
| 7 | `Speaker1_What_to_Speak_Only.docx` 🟢 | `Medical Health` | Known | `Medical Health` | **1.4513** | **TRUE** | Clinical speech script on AI in Disease Diagnosis |
| 8 | `Syllabus.pdf` 🔴 | `Science & Academics` | Known | `Technology & Computing` | **3.6463** | **FALSE** | **MISCLASSIFIED**: Microcontroller syllabus contains assembly code, causing Tech override |
| 9 | `TANISH_ARORA_CV_OLD.pdf` | `OOD` | OOD | `Technology & Computing` | 3.2583 | — | Professional software engineering CV containing coding projects |
| 10 | `back-of-the-napkin.pdf` 🔴 | `Business & Finance` | Known | `Science & Academics` | **0.4363** | **FALSE** | **MISCLASSIFIED**: Business strategy paper misclassified due to abstract strategic language |
| 11 | `01_Technology_Computing.pdf` 🟢 | `Technology & Computing` | Known | `Technology & Computing` | **1.8528** | **TRUE** | Controlled benchmark sample |
| 12 | `02_Medical_Health.pdf` 🟢 | `Medical Health` | Known | `Medical Health` | **3.4285** | **TRUE** | Controlled benchmark sample |
| 13 | `03_Business_Finance.pdf` 🟢 | `Business & Finance` | Known | `Business & Finance` | **3.6235** | **TRUE** | Controlled benchmark sample |
| 14 | `04_Entertainment.pdf` 🟢 | `Entertainment` | Known | `Entertainment` | **5.2417** | **TRUE** | Controlled benchmark sample |
| 15 | `05_Sports.pdf` 🟢 | `Sports` | Known | `Sports` | **5.3874** | **TRUE** | Controlled benchmark sample |
| 16 | `06_Science.pdf` 🟢 | `Science & Academics` | Known | `Science & Academics` | **3.1773** | **TRUE** | Controlled benchmark sample |
| 17 | `ood_miscellaneous_doc.pdf` | `OOD` | OOD | `Technology & Computing` | 0.2118 | — | Building access protocol |

---

## 2. Summary Statistics on Known Documents

- **Total Known Documents**: 11
- **Correctly Classified**: 8
- **Incorrectly Classified**: 3
- **Known Document Accuracy**: **72.73%** (8 / 11)

---

## 3. Error Breakdown & Common Failure Patterns

### Failure Pattern 1: Code Keywords Override Academic Category Signal (Primary Failure Pattern)
- **Affected Files**: `LAB MANUAL.pdf` and `Syllabus.pdf` (both Ground Truth: `Science & Academics`).
- **Predicted Category**: `Technology & Computing` (High margins: 2.3452 and 3.6463).
- **Root Cause**: The training set (`data/taxonomy_v2/dataset.csv`) contains news articles (`IAB News`, `BBC`) and Usenet posts (`20 Newsgroups`). Whenever a document contains programming syntax (`python`, `import`, `function`, `assembly`, `register`, `neural network`), the model assigns high weight to `Technology & Computing` because the training set contains **zero university lab manuals or syllabi** under `Science & Academics`.

### Failure Pattern 2: Abstract Business Strategy Terminology Bleed
- **Affected File**: `back-of-the-napkin.pdf` (Ground Truth: `Business & Finance`).
- **Predicted Category**: `Science & Academics` (Low margin: 0.4363).
- **Root Cause**: Theoretical business strategy and management methodology text lacks explicit stock/market newswire keywords (`earn`, `shares`, `revenue`, `acquisition`), causing low decision confidence.

---

## 4. Decision Margin Analysis Across Document Groups

Decision margins ($\text{top\_score} - \text{second\_highest\_score}$) were computed for all 18 documents:

| Document Group | Count ($N$) | Mean Margin | Median Margin | Min Margin | Max Margin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Correct Known Documents** | 8 | **3.3862** | **3.3029** | 1.4513 | 5.3874 |
| **Incorrect Known Documents** | 3 | **2.1426** | **2.3452** | 0.4363 | 3.6463 |
| **Out-Of-Domain (OOD) Documents** | 7 | **0.9387** | **0.6038** | 0.2056 | 3.2583 |

### Key Findings from Margin Analysis
1. **Separation**: Correct Known documents exhibit significantly higher average margins (mean 3.39) compared to OOD documents (mean 0.94).
2. **Overconfidence on Code**: `Syllabus.pdf` scored a high margin (3.6463) under `Technology & Computing` because assembly programming keywords strongly triggered the computing feature weights.
3. **Resumes & CVs as High-Margin OOD**: `TANISH_ARORA_CV_OLD.pdf` scored margin 3.2583 because the software engineering CV contains dense technical project listings.

---

## 5. Evaluation of the Provisional 0.25 Anonymous Threshold

- **Provisional Threshold**: $\gamma = 0.25$
- **OOD Rejection Performance**:
  - `Receipt_13Jul2025_130649.pdf` (Margin: 0.2056) $\rightarrow$ **REJECTED to Anonymous** (Correct)
  - `ood_miscellaneous_doc.pdf` (Margin: 0.2118) $\rightarrow$ **REJECTED to Anonymous** (Correct)
  - 5 other OOD documents (margins 0.60 - 3.26) $\rightarrow$ Accepted into known classes.

### Conclusion on Threshold Defensibility
The 0.25 threshold is **PROVISIONAL ONLY**. The 18-document evaluation sample (7 OOD files) is far too small to statistically establish a production margin threshold. A formal threshold validation requires a dedicated validation set of 200+ OOD documents.
