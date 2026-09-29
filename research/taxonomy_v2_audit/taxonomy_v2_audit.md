# DocSort AI — Taxonomy V2 Audit Report

**Date**: September 2026  
**Status**: Taxonomy Audit Complete (Pre-Training Phase)  
**Objective**: Audit the transition from the previous six-class taxonomy (`Science`) to the refined **Taxonomy V2** (`Science & Academics`), inspecting source mappings, class boundaries, data distributions, and potential category bleed between `Technology & Computing` and `Science & Academics`.

---

## 1. Executive Summary

Teacher requirement defines the final **Six-Class Taxonomy V2**:

1. **Technology & Computing**: Computing, programming, software engineering, AI/ML, computer systems, networking, hardware, technical software documentation, and coding.
2. **Science & Academics**: Scientific documents **AND** academic/educational documents (university syllabi, lab manuals, academic courseware, textbook materials, research papers).
3. **Medical Health**: Medicine, healthcare, clinical topics, diagnosis, patient care, pharmaceutical research.
4. **Business & Finance**: Corporate finance, banking, markets, economics, asset management.
5. **Entertainment**: Movies, television, music, celebrities, arts, and culture.
6. **Sports**: Athletics, matches, leagues, teams, and sports coverage.

---

## 2. Taxonomy V2 Source-by-Source Mapping Table

| Source Dataset | Original / Source Label | Old DocSort V1 Label | Taxonomy V2 Label | Documents | Mapping Justification & Audit Rule |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **IAB News** | `Business and Finance` | `Business and Finance` | `Business & Finance` | 2,972 | Corporate financial news, equity markets |
| **IAB News** | `Entertainment` | `Entertainment` | `Entertainment` | 6,001 | Media, cinema, arts, and celebrity news |
| **IAB News** | `Medical Health` | `Medical Health` | `Medical Health` | 3,684 | Health, clinical medicine, disease coverage |
| **IAB News** | `Science` | `Science` | `Science & Academics` | 3,662 | Popular science, space, astronomy news |
| **IAB News** | `Sports` | `Sports` | `Sports` | 5,210 | Sports coverage, matches, leagues |
| **IAB News** | `Technology & Computing` | `Technology & Computing` | `Technology & Computing` | 2,910 | Software, AI/ML, cloud, and computing news |
| **BBC News** | `business` | `Business and Finance` | `Business & Finance` | 236 | Financial news articles |
| **BBC News** | `entertainment` | `Entertainment` | `Entertainment` | 369 | Arts and entertainment coverage |
| **BBC News** | `sport` | `Sports` | `Sports` | 238 | Sports reporting |
| **BBC News** | `tech` | `Technology & Computing` | `Technology & Computing` | 247 | Technology & computing news |
| **BBC News** | `politics` | *REJECTED* | *REJECTED* | — | Outside six-class taxonomy |
| **MTSamples** | All Specialties | `Medical Health` | `Medical Health` | 1,935 | Anonymized medical transcription reports |
| **Reuters-21578** | `earn`, `acq`, `money-fx`, `interest`, `trade` | `Business and Finance` | `Business & Finance` | 3,162 | Financial & commercial newswire stories |
| **20 Newsgroups** | `comp.graphics` | `Technology & Computing` | `Technology & Computing` | 643 | Computer graphics & software algorithms |
| **20 Newsgroups** | `comp.os.ms-windows.misc` | `Technology & Computing` | `Technology & Computing` | 647 | Operating systems & software |
| **20 Newsgroups** | `comp.sys.ibm.pc.hardware` | `Technology & Computing` | `Technology & Computing` | 643 | Computer hardware & systems |
| **20 Newsgroups** | `comp.sys.mac.hardware` | `Technology & Computing` | `Technology & Computing` | 643 | Apple hardware & computing systems |
| **20 Newsgroups** | `comp.windows.x` | `Technology & Computing` | `Technology & Computing` | 637 | Windowing systems & X11 software |
| **20 Newsgroups** | `sci.crypt` ⚠️ | `Science` *(Questionable)* | **`Technology & Computing`** | 644 | **RE-MAPPED**: Cryptography, encryption algorithms, & computer security are software/computing. |
| **20 Newsgroups** | `sci.electronics` ⚠️ | `Science` *(Questionable)* | **`Technology & Computing`** | 638 | **RE-MAPPED**: Circuit design & electronic hardware are computer systems/hardware engineering. |
| **20 Newsgroups** | `sci.space` | `Science` | `Science & Academics` | 643 | Astronomy, space science, & physics |
| **20 Newsgroups** | `sci.med` | `Medical Health` | `Medical Health` | 643 | Medical discussions & health Q&A |
| **20 Newsgroups** | Other 9 categories | *REJECTED* | *REJECTED* | — | Religion, politics, autos, motorcycles rejected |

---

## 3. Technology & Computing vs. Science & Academics Boundary Audit

### Critical Findings & Mis-Mappings Corrected
1. **`sci.crypt` (Cryptography & Encryption)**: In V1/Expanded V1, `sci.crypt` was mapped to `Science`. Under Taxonomy V2, encryption algorithms, RSA security, and cryptographic protocols are **pure computer science / software engineering**, so they must be re-mapped to **`Technology & Computing`**.
2. **`sci.electronics` (Circuit Design & Hardware)**: In V1/Expanded V1, `sci.electronics` was mapped to `Science`. Electronic hardware and digital circuits are **computer systems / hardware engineering**, so they must be re-mapped to **`Technology & Computing`**.
3. **Absence of Genuinely Academic/Educational Material**:
   - The current corpus contains popular science news (`IAB Science`), Usenet astrophysics posts (`sci.space`), and clinical transcripts (`MTSamples`).
   - However, it contains **ZERO university lab manuals, course syllabi, academic courseware, or textbook chapters**.
   - As demonstrated by our real-world document evaluations, academic documents like `LAB MANUAL.pdf` and `Syllabus.pdf` require academic structure signals (course codes, learning objectives, lab exercises, assignment grading criteria) during training so the model recognizes them as `Science & Academics` rather than confusing them with pure `Technology & Computing`.

---

## 4. Current Training Data Distribution (Pre-V2 Expansion)

| Category | Current Document Count | Source Composition | Taxonomy Alignment Rating |
| :--- | :---: | :--- | :---: |
| **Business & Finance** | 6,370 | IAB News (2,972), Reuters (3,162), BBC (236) | 🟢 Excellent |
| **Medical Health** | 6,370 | IAB News (3,684), MTSamples (1,935), 20 Newsgroups (751) | 🟢 Excellent |
| **Sports** | 6,370 | IAB News (5,210), 20 Newsgroups (922), BBC (238) | 🟢 Excellent |
| **Entertainment** | 6,370 | IAB News (6,001), BBC (369) | 🟢 Excellent |
| **Technology & Computing** | 6,370 (+1,282 re-mapped) | 20 Newsgroups (3,213), IAB News (2,910), BBC (247) | 🟡 Needs `sci.crypt` & `sci.electronics` addition |
| **Science & Academics** | 6,370 (-1,282 re-mapped) | IAB News (3,662), 20 Newsgroups (2,708) | 🔴 **Deficient in Academic/Educational Data** |

---

## 5. Audit Conclusion
To fulfill the teacher's final **Taxonomy V2** specification:
1. `sci.crypt` and `sci.electronics` must move from `Science` to `Technology & Computing`.
2. Dedicated academic documents (e.g. arXiv STEM paper abstracts, university courseware, educational lab material) must be integrated into `Science & Academics` to balance the loss of `sci.crypt`/`sci.electronics` and establish true academic signal.
