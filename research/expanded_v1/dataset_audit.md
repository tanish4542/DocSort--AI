# Expanded V1 — Candidate Dataset Audit Report

## Overview
To test whether increasing training size and source diversity improves multi-domain document classification, five candidate datasets were systematically audited across text length, label provenance, quality, and class mappings.

## Dataset Summary Table

| Dataset Name | Original Size | Label Provenance | Duplicate Rate | Status |
| :--- | :---: | :--- | :---: | :--- |
| **IAB News Classification Dataset** | 106,280 | Weakly Labeled / LLM-Labeled (GPT-5-nano) | 6.32% | ACCEPTED WITH AUDIT (Used for multi-domain scale; LLM-labeled) |
| **BBC News Dataset** | 2,225 | Human-Annotated (BBC Editorial Topics) | 4.45% | ACCEPTED (4 matching classes; Politics rejected) |
| **MTSamples Medical Transcriptions** | 4,999 | Human-Annotated (Medical Specialty Categories) | 52.83% | ACCEPTED (Mapped to Medical Health) |
| **20 Newsgroups** | 18,846 | Human-Annotated (Usenet Newsgroup Topics) | 2.97% | ACCEPTED (11 matching categories mapped; unmapped rejected) |
| **Reuters-21578 Newswire** | 10,788 | Human-Annotated (Financial & Commodity Newswire Categories) | 1.21% | ACCEPTED (Financial topics mapped to Business and Finance; non-financial rejected) |

## Detailed Quality Findings

### IAB News Classification Dataset
- **Source**: Hugging Face (mdonigian/iab-news-classification)
- **License**: Open Data / Permissive (Web News Scrape)
- **Original Size**: 106,280 records
- **Text Length (Chars)**: Min=219, Max=130,423, Mean=3,410.0, Median=3,004.0
- **Empty/Null Records**: 0
- **Duplicate Text Rate**: 6.32%
- **Audit Status**: ACCEPTED WITH AUDIT (Used for multi-domain scale; LLM-labeled)

### BBC News Dataset
- **Source**: Hugging Face (SetFit/bbc-news / BBC Corpus)
- **License**: Educational / Research Use
- **Original Size**: 2,225 records
- **Text Length (Chars)**: Min=501, Max=25,483, Mean=2,262.9, Median=1,965.0
- **Empty/Null Records**: 0
- **Duplicate Text Rate**: 4.45%
- **Audit Status**: ACCEPTED (4 matching classes; Politics rejected)

### MTSamples Medical Transcriptions
- **Source**: Hugging Face (harishnair04/mtsamples / MTSamples.com)
- **License**: Public Domain / Anonymized Medical Reports
- **Original Size**: 4,999 records
- **Text Length (Chars)**: Min=11, Max=18,425, Mean=3,052.3, Median=2,667.0
- **Empty/Null Records**: 33
- **Duplicate Text Rate**: 52.83%
- **Audit Status**: ACCEPTED (Mapped to Medical Health)

### 20 Newsgroups
- **Source**: Scikit-Learn (fetch_20newsgroups)
- **License**: Public Domain / Usenet Archive
- **Original Size**: 18,846 records
- **Text Length (Chars)**: Min=0, Max=158,791, Mean=1,169.7, Median=489.0
- **Empty/Null Records**: 515
- **Duplicate Text Rate**: 2.97%
- **Audit Status**: ACCEPTED (11 matching categories mapped; unmapped rejected)

### Reuters-21578 Newswire
- **Source**: NLTK Corpus (reuters)
- **License**: Reuters / Educational Research Use
- **Original Size**: 10,788 records
- **Text Length (Chars)**: Min=27, Max=14,060, Mean=820.1, Median=536.5
- **Empty/Null Records**: 0
- **Duplicate Text Rate**: 1.21%
- **Audit Status**: ACCEPTED (Financial topics mapped to Business and Finance; non-financial rejected)

