# DocSort AI — Targeted Academic Augmentation Report

**Date**: September 2026  
**Experiment Verdict**: **FAIL**  

## 1. Executive Summary
- **Targeted Problem**: Misclassification of CS/engineering academic courseware as Technology & Computing.
- **`LAB MANUAL.pdf` Prediction**: **`Technology & Computing`** (Required: `Science & Academics` -> FAIL)
- **`Syllabus.pdf` Prediction**: **`Technology & Computing`** (Required: `Science & Academics` -> FAIL)
- **`Major Project Synopsis.pdf` Prediction**: **`Technology & Computing`** (Required: `Technology & Computing` -> PASS)
- **`Speaker1_What_to_Speak_Only.docx` Prediction**: **`Medical Health`** (Required: `Medical Health` -> PASS)

## 2. Real-World Document Predictions Table

| filename | ground_truth_normalized | lsvc_prediction | lsvc_decision_margin | lsvc_correct |
| --- | --- | --- | --- | --- |
| Air India Web Booking eTicket (9WSBWJ) - TANISH.pdf | OOD | Technology & Computing | 0.7377 | False |
| ID card 25 Mar 2025.pdf | OOD | Technology & Computing | 0.6317 | False |
| LAB MANUAL.pdf | Science & Academics | Technology & Computing | 0.6026 | False |
| MANVI_BIODATA.pdf | OOD | Medical Health | 0.2824 | False |
| Major Project Synopsis.pdf | Technology & Computing | Technology & Computing | 2.4475 | True |
| Receipt_13Jul2025_130649.pdf | OOD | Business & Finance | 0.522 | False |
| Speaker1_What_to_Speak_Only.docx | Medical Health | Medical Health | 1.5512 | True |
| Syllabus.pdf | Science & Academics | Technology & Computing | 1.4885 | False |
| TANISH_ARORA_CV_OLD.pdf | OOD | Technology & Computing | 2.8744 | False |
| back-of-the-napkin.pdf | Business & Finance | Science & Academics | 0.504 | False |
| MANVI_BIODATA.pdf | OOD | Medical Health | 0.2824 | False |
| 01_Technology_Computing.pdf | Technology & Computing | Technology & Computing | 1.779 | True |
| 02_Medical_Health.pdf | Medical Health | Medical Health | 2.829 | True |
| 03_Business_Finance.pdf | Business & Finance | Business & Finance | 3.9068 | True |
| 04_Entertainment.pdf | Entertainment | Entertainment | 5.3487 | True |
| 05_Sports.pdf | Sports | Sports | 5.6344 | True |
| 06_Science.pdf | Science & Academics | Science & Academics | 3.2302 | True |
| ood_miscellaneous_doc.pdf | OOD | Technology & Computing | 0.6043 | False |

## 3. Model Benchmark Comparison

| model_name | eval_benchmark | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Multinomial Naive Bayes | Targeted Academic Test Set | 89.7 | 90.32 | 89.72 | 89.69 | 89.61 |
| Logistic Regression | Targeted Academic Test Set | 94.08 | 94.44 | 94.06 | 94.22 | 94.1 |
| LinearSVC | Targeted Academic Test Set | 94.58 | 94.75 | 94.59 | 94.66 | 94.58 |
