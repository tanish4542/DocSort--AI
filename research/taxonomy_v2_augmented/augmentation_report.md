# DocSort AI — Taxonomy V2 Augmented Research Report

**Date**: September 2026  
**Status**: Targeted Data Augmentation Experiment Complete  

## 1. Executive Summary
- **Final Augmented Corpus**: 40,820 records across 6 Taxonomy V2 categories.
- **Best Model**: **LinearSVC** (94.38% Augmented Internal Test Macro F1).
- **Targeted Augmentation**: Added PubMed QA academic scientific contexts + University Syllabi/Lab Manual templates to `Science & Academics` and structured invoices/receipts to `Business & Finance`.

## 2. Added Augmentation Data Summary

| dataset_name | target_category | added_count | provenance_license | justification |
| --- | --- | --- | --- | --- |
| PubMed QA Academic Research Contexts | Science & Academics | 1000 | NLM Open Access / Educational Scientific Research | Peer-reviewed scientific methodology and experimental findings. |
| University Syllabi & Lab Manual Archive | Science & Academics | 800 | Open Courseware Templates / University Syllabi Standard Formats | Provides explicit course structure, grading rubrics, and lab assignment signals. |
| Structured Invoices & Receipts Archive | Business & Finance | 800 | Open Financial Receipts / Commercial Transaction Standards | Adds transactional billing, invoice, and payment receipt structure signals. |

## 3. Benchmark Performance Summary

| model_name | eval_benchmark | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Multinomial Naive Bayes | Augmented Internal Test Set | 91.03 | 91.42 | 90.89 | 91.01 | 91.01 |
| Logistic Regression | Augmented Internal Test Set | 94.04 | 94.27 | 93.99 | 94.11 | 94.05 |
| LinearSVC | Augmented Internal Test Set | 94.33 | 94.47 | 94.31 | 94.38 | 94.34 |

## 4. Real-World Document Predictions & Generalization

| filename | ground_truth_normalized | lsvc_prediction | lsvc_decision_margin | lsvc_correct |
| --- | --- | --- | --- | --- |
| Air India Web Booking eTicket (9WSBWJ) - TANISH.pdf | OOD | Technology & Computing | 0.5093 | False |
| ID card 25 Mar 2025.pdf | OOD | Technology & Computing | 1.0172 | False |
| LAB MANUAL.pdf | Science & Academics | Technology & Computing | 1.2099 | False |
| MANVI_BIODATA.pdf | OOD | Medical Health | 0.3126 | False |
| Major Project Synopsis.pdf | Technology & Computing | Technology & Computing | 3.1 | True |
| Receipt_13Jul2025_130649.pdf | OOD | Business & Finance | 0.5868 | False |
| Speaker1_What_to_Speak_Only.docx | Medical Health | Medical Health | 0.9406 | True |
| Syllabus.pdf | Science & Academics | Technology & Computing | 2.0217 | False |
| TANISH_ARORA_CV_OLD.pdf | OOD | Technology & Computing | 3.1643 | False |
| back-of-the-napkin.pdf | Business & Finance | Science & Academics | 0.3585 | False |
| MANVI_BIODATA.pdf | OOD | Medical Health | 0.3126 | False |
| 01_Technology_Computing.pdf | Technology & Computing | Technology & Computing | 1.7016 | True |
| 02_Medical_Health.pdf | Medical Health | Medical Health | 2.7517 | True |
| 03_Business_Finance.pdf | Business & Finance | Business & Finance | 3.6582 | True |
| 04_Entertainment.pdf | Entertainment | Entertainment | 5.3931 | True |
| 05_Sports.pdf | Sports | Sports | 5.6723 | True |
| 06_Science.pdf | Science & Academics | Science & Academics | 3.1474 | True |
| ood_miscellaneous_doc.pdf | OOD | Technology & Computing | 0.4695 | False |
