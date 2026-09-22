# Proposed Production Taxonomy for DocSort AI V2

This document defines the production document taxonomy for DocSort AI V2, analyzing the failure of the news-based taxonomy and establishing functional, organization-oriented categories for desktop document management.

---

## 1. Why the Research News Taxonomy Fails in Production

The frozen research model uses the 6 IAB news categories:
1. `Business and Finance`
2. `Medical Health`
3. `Sports`
4. `Technology & Computing`
5. `Science`
6. `Entertainment`

### Fundamental Mismatch: Subject-Matter Topic vs. Document Function
In journalistic news classification, documents are organized by **subject-matter topics** (e.g., an article *about* a soccer match vs. an article *about* Apple stock).

In desktop document organization, users organize files by **functional utility** (what the document *is* and *how it is used*):
- A user wants their tuition payment receipt filed with **bills and taxes**, not inside an empty "Sports" folder or misrouted to "Technology & Computing" because the student studies AI.
- A student wants course syllabi and lecture slides filed under **academics**, whether the subject is Biology, Computer Vision, or Macroeconomics.
- An engineer wants technical architecture manuals filed under **technical documentation**, not mixed with tech industry news reports.

---

## 2. Production Taxonomy Analysis: Merges, Splits, and Deletions

### 2.1 What Should Be Removed / Re-scoped
- **`Sports` (Deprecate as Core Category)**: Less than 0.5% of real desktop document uploads are sports documents. Desktop users do not save sports news articles as PDFs or Word documents. Tournament brackets or game tickets can be routed to `Personal & Leisure` or `Other`.
- **`Entertainment` (Deprecate as Core Category)**: Box office reports, celebrity gossip, and movie reviews do not represent workplace or personal desktop documents. Concert or movie tickets are transactional items best handled by receipts/other.
- **`Science` (Merge into `Academic & Educational`)**: In research testing, `Science` had the highest confusion boundary with `Technology & Computing` (23 test errors) and `Medical Health` (32 test errors). In real documents, scientific material almost always takes the form of **academic research papers, textbook chapters, or university lab notes**. Maintaining an isolated "Science" folder creates permanent ambiguity with engineering, medicine, and academic materials.

### 2.2 What Must Be Added
- **`Academic & Educational` (High Priority)**: The #1 missing category in the current system. Real users upload course notes, syllabi, slide decks, assignment PDFs, and certificates daily.
- **`Financial & Receipts` (High Priority)**: Must expand beyond Wall Street news to cover invoices, transaction receipts, bank statements, tax forms, and utility bills.
- **`Administrative & Identification` (High Priority)**: Covers official government identity cards, university registration forms, certificates, passports, legal contracts, and residential agreements.
- **`Career & Professional` (Recommended Addition)**: Resumes, CVs, job offer letters, cover letters, and performance evaluations currently have no home and leak heavily into Tech or Business.

---

## 3. The Proposed 6+1 Production Taxonomy for V2

We propose a streamlined **6 functional core categories + 1 unassigned/review category**:

```
DocSort AI Production Taxonomy (V2)
├── 1. Academic & Educational
├── 2. Financial & Receipts
├── 3. Administrative & Identification
├── 4. Technology & Technical Docs
├── 5. Medical & Healthcare
├── 6. Career & Professional
└── 7. Unassigned / Needs Review (Staged in _pending)
```

---

## 4. Category Definitions, Inclusions, and Key Signals

### Category 1: Academic & Educational
- **Functional Definition**: Materials created, distributed, or completed in pursuit of academic coursework, degrees, or scientific study.
- **Included Document Types**:
  - Lecture slide decks (`morphological_Image_Processing.pdf`)
  - Course syllabi and curriculum guides (`Syllabus.pdf`)
  - Homework assignments and lab manuals
  - University and MOOC course certificates (`COURSERA SDV.pdf`)
  - Published research papers, conference preprints (ArXiv)
  - Student study notes and exam review sheets
- **Excluded**:
  - Tuition payment receipts (route to $\to$ *Financial & Receipts*)
  - Student identity cards (route to $\to$ *Administrative & Identification*)
- **Key Lexical Indicators**: `syllabus`, `course`, `lecture`, `assignment`, `semester`, `prerequisites`, `grading`, `credits`, `university`, `professor`, `abstract`, `references`, `doi`, `theorem`, `exercise`, `exam`.

---

### Category 2: Financial & Receipts
- **Functional Definition**: Records of monetary transactions, financial status, billing obligations, or tax accounting.
- **Included Document Types**:
  - Online payment receipts, payment vouchers, transaction slips (`PDFPrint.pdf`)
  - Commercial vendor invoices and utility bills
  - Bank account statements and credit card summaries
  - Income tax filings, W-2s, Form 16, payroll pay slips
  - Corporate financial audits and investment account summaries
- **Excluded**:
  - Job offer compensation letters (route to $\to$ *Career & Professional*)
- **Key Lexical Indicators**: `receipt`, `invoice`, `transaction id`, `order id`, `payment`, `amount paid`, `total`, `balance due`, `subtotal`, `tax`, `inr`, `usd`, `bank`, `account number`, `statement`, `deposit`, `withdrawal`, `billing`.

---

### Category 3: Administrative & Identification
- **Functional Definition**: Legal, institutional, and identity documents that establish authorization, status, or identity.
- **Included Document Types**:
  - Student and employee identity cards (`ID card 25 Mar 2025.pdf`)
  - Government identity cards (Passports, Aadhaar, Driver Licenses, Social Security)
  - College registration and enrollment forms
  - Non-disclosure agreements (NDAs), rental lease agreements, legal contracts
  - Official institution notices, circulars, and certificates of birth/residence
- **Excluded**:
  - Course completion certificates (route to $\to$ *Academic & Educational*)
- **Key Lexical Indicators**: `identity card`, `valid till`, `issued by`, `date of birth`, `card number`, `enrollment`, `registration no`, `government of`, `republic`, `agreement`, `party of the first part`, `witnesseth`, `jurisdiction`, `covenant`.

---

### Category 4: Technology & Technical Docs
- **Functional Definition**: Artifacts dedicated to software engineering, computer hardware, system architecture, programming, or IT operations.
- **Included Document Types**:
  - Source code listings and programming exercises (`Python Basic Programs.pdf`)
  - Software development kits (SDK) and API documentation
  - System architecture guides, server configurations, Docker/Kubernetes files
  - Hardware specifications, processor schematics, data center network topology
  - IT troubleshooting guides and developer walkthroughs
- **Excluded**:
  - Computer science course syllabi (route to $\to$ *Academic & Educational*)
- **Key Lexical Indicators**: `def`, `import`, `function`, `api`, `endpoint`, `docker`, `kubernetes`, `cluster`, `processor`, `database`, `class`, `syntax`, `git`, `repository`, `cpu`, `architecture`, `json`, `yaml`, `framework`.

---

### Category 5: Medical & Healthcare
- **Functional Definition**: Clinical, diagnostic, pharmaceutical, or therapeutic records regarding individual patient health or medical treatments.
- **Included Document Types**:
  - Pathology and blood laboratory reports (CBC, lipid profile, metabolic panel)
  - Doctor prescriptions and outpatient clinical notes
  - Radiology and imaging reports (MRI, CT, ultrasound summaries)
  - Hospital discharge summaries and operative notes
  - Health insurance claim records and vaccination cards
- **Excluded**:
  - Hospital bills / payment receipts (route to $\to$ *Financial & Receipts* if primary purpose is billing)
- **Key Lexical Indicators**: `patient`, `diagnosis`, `rx`, `dosage`, `tablet`, `physician`, `clinic`, `hospital`, `laboratory`, `specimen`, `hemoglobin`, `pathology`, `symptoms`, `treatment`, `discharge`, `clinical`.

---

### Category 6: Career & Professional
- **Functional Definition**: Documents concerning employment, individual career profiles, hiring, and workplace performance.
- **Included Document Types**:
  - Resumes and Curriculum Vitae (CV)
  - Employment offer letters and appointment letters
  - Professional recommendation and cover letters
  - Performance appraisal reviews and promotion records
  - Professional portfolios and work certification summaries
- **Excluded**:
  - Monthly pay slips (route to $\to$ *Financial & Receipts*)
- **Key Lexical Indicators**: `curriculum vitae`, `resume`, `experience`, `education`, `skills`, `employment`, `offer of employment`, `base salary`, `reporting to`, `work history`, `recommendation`, `candidate`, `qualifications`.

---

### Category 7: Unassigned / Needs Review (`_pending`)
- **Functional Definition**: The landing category for documents that fail the dual-gated margin (low absolute support $\max f_c(x) < \tau$ or boundary ambiguity $f_{\text{top1}} - f_{\text{top2}} < \delta$), documents with insufficient text, or documents outside the target domains (e.g., recipes, creative fiction, scanned art).
- Staged in `~/Desktop/SortedDocuments/_pending/` until resolved by the user.

---

## 5. Ambiguity Resolution Guidelines

| Document Scenario | Candidate 1 | Candidate 2 | Governing Rule | Winning Category |
| :--- | :--- | :--- | :--- | :--- |
| **Computer Science Syllabus** | Academic | Technology | Document purpose is curriculum structure, not software documentation. | **Academic & Educational** |
| **College Fee Receipt** | Financial | Academic | Document purpose is financial transaction/audit proof. | **Financial & Receipts** |
| **Doctor Prescription** | Medical | Administrative | Document contains medical treatment and dosage details. | **Medical & Healthcare** |
| **Software Engineer Resume** | Career | Technology | Document is a job application profile, not technical documentation. | **Career & Professional** |
| **Biology Slide Deck** | Academic | Medical | Presentation slides for course instruction. | **Academic & Educational** |
| **Hospital Discharge Bill** | Financial | Medical | Itemized invoice for hospital services. | **Financial & Receipts** |
