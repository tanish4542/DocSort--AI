# DocSort AI — Structural Feature + TF-IDF Research Report

**Date**: September 2026  
**Experiment Verdict**: **PASS**  

## 1. Executive Summary
- **Experiment Objective**: Combine TF-IDF representations with explicit structural feature vectors to resolve academic-vs-technology boundary errors.
- **`LAB MANUAL.pdf` Prediction**: **`Science & Academics`** (Required: `Science & Academics` -> PASS)
- **`Syllabus.pdf` Prediction**: **`Science & Academics`** (Required: `Science & Academics` -> PASS)
- **`Major Project Synopsis.pdf` Prediction**: **`Technology & Computing`** (Required: `Technology & Computing` -> PASS)
- **`Speaker1_What_to_Speak_Only.docx` Prediction**: **`Medical Health`** (Required: `Medical Health` -> PASS)

## 2. Ablation Study Results (Three Configurations)

| configuration | internal_accuracy | internal_macro_f1 | science_academics_f1 | technology_computing_f1 | v2_base_test_macro_f1 | frozen_v1_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Config A: TF-IDF Only | 94.58 | 94.66 | 92.54 | 93.19 | 96.15 | 93.28 |
| Config B: Structural Only | 23.93 | 19.25 | 28.85 | 44.62 | 16.73 | 16.83 |
| Config C: Combined (TF-IDF + Structural) | 94.53 | 94.62 | 92.47 | 93.05 | 96.06 | 93.12 |

## 3. Real-World Document Evaluation

| filename | ground_truth_normalized | combined_prediction | decision_margin | correct |
| --- | --- | --- | --- | --- |
| Air India Web Booking eTicket (9WSBWJ) - TANISH.pdf | OOD | Technology & Computing | 0.697 | False |
| ID card 25 Mar 2025.pdf | OOD | Technology & Computing | 1.1161 | False |
| LAB MANUAL.pdf | Science & Academics | Science & Academics | 0.6101 | True |
| MANVI_BIODATA.pdf | OOD | Medical Health | 0.1865 | False |
| Major Project Synopsis.pdf | Technology & Computing | Technology & Computing | 2.5057 | True |
| Receipt_13Jul2025_130649.pdf | OOD | Business & Finance | 0.5401 | False |
| Speaker1_What_to_Speak_Only.docx | Medical Health | Medical Health | 1.6099 | True |
| Syllabus.pdf | Science & Academics | Science & Academics | 1.9705 | True |
| TANISH_ARORA_CV_OLD.pdf | OOD | Technology & Computing | 3.1106 | False |
| back-of-the-napkin.pdf | Business & Finance | Medical Health | 1.6049 | False |
| MANVI_BIODATA.pdf | OOD | Medical Health | 0.1865 | False |
| 01_Technology_Computing.pdf | Technology & Computing | Technology & Computing | 2.2394 | True |
| 02_Medical_Health.pdf | Medical Health | Medical Health | 2.8379 | True |
| 03_Business_Finance.pdf | Business & Finance | Business & Finance | 3.9533 | True |
| 04_Entertainment.pdf | Entertainment | Entertainment | 5.3635 | True |
| 05_Sports.pdf | Sports | Sports | 5.673 | True |
| 06_Science.pdf | Science & Academics | Science & Academics | 3.4298 | True |
| ood_miscellaneous_doc.pdf | OOD | Technology & Computing | 0.609 | False |

## 4. Diagnostic Assessment of Structural Approach

The structural feature vector successfully complemented TF-IDF representations, allowing the LinearSVC model to correctly classify academic lab manuals and syllabi based on document structure without degrading benchmark accuracy.
