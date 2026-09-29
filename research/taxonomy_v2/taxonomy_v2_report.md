# DocSort AI — Taxonomy V2 Research Report

**Date**: September 2026  
**Status**: Fast-Track Research Experiment Complete  

## 1. Executive Summary
- **Total Corpus**: 38,220 records across 6 Taxonomy V2 categories.
- **Best Model**: **LinearSVC** (94.12% Internal Test Macro F1).
- **Academic Data Limitation Note**: *The fast-track Taxonomy V2 experiment does not add a new academic corpus because the attempted arXiv acquisition was operationally blocked. Academic/educational documents are therefore evaluated externally, and further academic-data augmentation can be performed in a later experiment.*

## 2. Model Performance Summary

| model_name | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| Multinomial Naive Bayes | 91.45 | 91.65 | 91.17 | 91.32 | 91.48 |
| Logistic Regression | 93.96 | 94.09 | 93.74 | 93.9 | 93.98 |
| LinearSVC | 94.24 | 94.15 | 94.09 | 94.12 | 94.25 |

## 3. Real-World Document Predictions

| filename | ground_truth | lsvc_prediction | decision_margin | lsvc_correct |
| --- | --- | --- | --- | --- |
| Air India Web Booking eTicket (9WSBWJ) - TANISH.pdf | OOD | Technology & Computing | 1.0296 | False |
| ID card 25 Mar 2025.pdf | OOD | Technology & Computing | 0.658 | False |
| LAB MANUAL.pdf | Science & Academics | Technology & Computing | 2.3452 | False |
| MANVI_BIODATA.pdf | OOD | Medical Health | 0.6038 | False |
| Major Project Synopsis.pdf | Technology & Computing | Technology & Computing | 2.9271 | True |
| Receipt_13Jul2025_130649.pdf | OOD | Business & Finance | 0.2056 | False |
| Speaker1_What_to_Speak_Only.docx | Medical Health | Medical Health | 1.4513 | True |
| Syllabus.pdf | Science & Academics | Technology & Computing | 3.6463 | False |
| TANISH_ARORA_CV_OLD.pdf | OOD | Technology & Computing | 3.2583 | False |
| back-of-the-napkin.pdf | Business and Finance | Science & Academics | 0.4363 | False |
| MANVI_BIODATA.pdf | OOD | Medical Health | 0.6038 | False |
| 01_Technology_Computing.pdf | Technology & Computing | Technology & Computing | 1.8528 | True |
| 02_Medical_Health.pdf | Medical Health | Medical Health | 3.4285 | True |
| 03_Business_Finance.pdf | Business and Finance | Business & Finance | 3.6235 | False |
| 04_Entertainment.pdf | Entertainment | Entertainment | 5.2417 | True |
| 05_Sports.pdf | Sports | Sports | 5.3874 | True |
| 06_Science.pdf | Science | Science & Academics | 3.1773 | False |
| ood_miscellaneous_doc.pdf | OOD | Technology & Computing | 0.2118 | False |

## 4. Decision Margin Threshold Analysis

| threshold | known_accepted_count | known_rejected_count | known_accepted_acc_pct | ood_rejected_count | ood_accepted_count | ood_rejection_rate_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 5664.0 | 69.0 | 95.69 | 0.0 | 7.0 | 0.0 |
| 0.15 | 5630.0 | 103.0 | 95.88 | 0.0 | 7.0 | 0.0 |
| 0.2 | 5588.0 | 145.0 | 96.31 | 0.0 | 7.0 | 0.0 |
| 0.25 | 5554.0 | 179.0 | 96.53 | 2.0 | 5.0 | 28.57 |
| 0.3 | 5521.0 | 212.0 | 96.67 | 2.0 | 5.0 | 28.57 |
| 0.35 | 5485.0 | 248.0 | 96.96 | 2.0 | 5.0 | 28.57 |
| 0.4 | 5457.0 | 276.0 | 97.14 | 2.0 | 5.0 | 28.57 |
| 0.45 | 5429.0 | 304.0 | 97.29 | 2.0 | 5.0 | 28.57 |
| 0.5 | 5396.0 | 337.0 | 97.39 | 2.0 | 5.0 | 28.57 |
