# DocSort AI — Academic Error Diagnostic Report (Taxonomy V2 Augmented)

**Date**: September 2026  
**Status**: Diagnostic Complete (No Code or Model Changes Made)  
**Target Artifacts Analyzed**:
- Trained Model Vectorizer & LinearSVC Classifier: `research/taxonomy_v2_augmented/models/`
- Training Dataset: `data/taxonomy_v2_augmented/dataset.csv`
- Target Real-World Files: `LAB MANUAL.pdf`, `Syllabus.pdf`, `Major Project Synopsis.pdf`, `Speaker1_What_to_Speak_Only.docx`

---

## 1. Itemized Real-World Document Diagnostic

### 1.1 `LAB MANUAL.pdf`
- **Ground Truth**: `Science & Academics`
- **Predicted Class**: `Technology & Computing`
- **Decision Margin**: `1.2099` (Top score: +1.0314 vs Second score: -0.1786)
- **Top 3 Decision Scores**:
  1. `Technology & Computing`: **+1.0314**
  2. `Science & Academics`: **-0.1786**
  3. `Sports`: **-1.4405**

#### Top 20 Contributing Features for Predicted Class (`Technology & Computing`):
| Feature | TF-IDF Value | LinearSVC Weight | LinearSVC Contribution | Meaningful vs. Generic |
| :--- | :---: | :---: | :---: | :--- |
| `output` | 0.0893 | +1.5482 | +0.1383 | Generic programming syntax |
| `window` | 0.0668 | +2.0101 | +0.1344 | Computing / GUI term |
| `cnn` | 0.1108 | +1.1386 | +0.1262 | Machine learning term |
| `print` | 0.1000 | +0.8946 | +0.0894 | Generic Python syntax |
| `input` | 0.0978 | +0.8295 | +0.0812 | Generic programming syntax |
| `define` | 0.0745 | +1.0443 | +0.0778 | Generic code keyword |
| `data` | 0.0567 | +1.2919 | +0.0733 | Broad technical term |
| `models` | 0.0835 | +0.8667 | +0.0723 | Technical / ML term |
| `model` | 0.1000 | +0.6772 | +0.0677 | Broad technical term |
| `w1` | 0.0911 | +0.6739 | +0.0614 | Code variable name |
| `random` | 0.0985 | +0.6186 | +0.0609 | Code function name |
| `compile` | 0.1227 | +0.4875 | +0.0598 | Software compilation term |
| `self` | 0.0940 | +0.6285 | +0.0591 | Python OOP keyword |
| `weights` | 0.0840 | +0.6974 | +0.0586 | ML / Neural net term |
| `256` | 0.1048 | +0.4966 | +0.0521 | Technical numeric value |
| `layer` | 0.0812 | +0.6120 | +0.0497 | Neural net term |
| `dense` | 0.1385 | +0.3412 | +0.0472 | Neural net layer term |
| `import` | 0.1477 | +0.3120 | +0.0461 | Python import keyword |
| `return` | 0.0811 | +0.5510 | +0.0447 | Python return keyword |
| `loss` | 0.0712 | +0.6110 | +0.0435 | ML loss term |

#### Top 20 Features Associated with Expected Class (`Science & Academics`):
| Feature | TF-IDF Value | LinearSVC Weight | LinearSVC Contribution | Meaningful vs. Generic |
| :--- | :---: | :---: | :---: | :--- |
| `deep` | 0.0779 | +1.1396 | +0.0888 | Scientific / ML term |
| `evaluate` | 0.0709 | +0.9993 | +0.0709 | Academic evaluation term |
| `accuracy` | 0.0797 | +0.8595 | +0.0685 | Academic evaluation term |
| `import` | 0.1477 | +0.4274 | +0.0632 | Shared code keyword |
| `model` | 0.1000 | +0.5078 | +0.0508 | Shared technical term |
| `learning` | 0.0878 | +0.5728 | +0.0503 | Educational / ML term |
| `neural` | 0.0784 | +0.6060 | +0.0475 | Scientific / AI term |
| `rand` | 0.0884 | +0.4992 | +0.0441 | Code function |
| `shape` | 0.0919 | +0.4780 | +0.0440 | Tensor shape term |
| `objective` | 0.0836 | +0.5183 | +0.0434 | Educational / Lab objective |
| `dense` | 0.1385 | +0.2980 | +0.0413 | Shared neural net term |
| `dataset` | 0.0951 | +0.4266 | +0.0406 | Academic research term |
| `data` | 0.0567 | +0.7117 | +0.0404 | Shared technical term |
| `compare` | 0.0553 | +0.6903 | +0.0381 | Academic comparative term |
| `efficient` | 0.0257 | +1.3067 | +0.0336 | Scientific term |

---

### 1.2 `Syllabus.pdf`
- **Ground Truth**: `Science & Academics`
- **Predicted Class**: `Technology & Computing`
- **Decision Margin**: `2.0217` (Top score: +1.6451 vs Second score: -0.3766)
- **Top 3 Decision Scores**:
  1. `Technology & Computing`: **+1.6451**
  2. `Science & Academics`: **-0.3766**
  3. `Business & Finance`: **-1.8199**

#### Top 20 Contributing Features for Predicted Class (`Technology & Computing`):
| Feature | TF-IDF Value | LinearSVC Weight | LinearSVC Contribution | Meaningful vs. Generic |
| :--- | :---: | :---: | :---: | :--- |
| `software` | 0.0844 | +1.5150 | +0.1279 | Computing software term |
| `program` | 0.1080 | +1.0819 | +0.1168 | Computing / code term |
| `keyboard` | 0.0822 | +1.3340 | +0.1097 | Computer hardware term |
| `threads` | 0.0837 | +1.2059 | +0.1009 | OS concurrency term |
| `compiler` | 0.1010 | +0.9933 | +0.1004 | Systems software term |
| `interface` | 0.0934 | +1.0395 | +0.0971 | Software interface term |
| `code` | 0.0725 | +1.3116 | +0.0951 | Generic code keyword |
| `systems` | 0.1007 | +0.9339 | +0.0940 | Computer systems term |
| `applications` | 0.0656 | +1.3556 | +0.0890 | Software applications |
| `chip` | 0.0382 | +2.1165 | +0.0808 | Hardware chip term |
| `hardware` | 0.0887 | +0.8666 | +0.0769 | Computer hardware term |
| `write` | 0.0956 | +0.7988 | +0.0763 | Generic programming verb |
| `computing` | 0.0427 | +1.6121 | +0.0689 | Computing category keyword |
| `assembly` | 0.0965 | +0.6774 | +0.0654 | Assembly programming |
| `design` | 0.0702 | +0.9141 | +0.0641 | System design term |

#### Top 20 Features Associated with Expected Class (`Science & Academics`):
| Feature | TF-IDF Value | LinearSVC Weight | LinearSVC Contribution | Meaningful vs. Generic |
| :--- | :---: | :---: | :---: | :--- |
| `experiments` | 0.0964 | +1.7047 | +0.1643 | Academic laboratory term |
| `engineering` | 0.0828 | +1.1334 | +0.0939 | Academic discipline |
| `characteristics` | 0.0784 | +0.7634 | +0.0598 | Academic analysis term |
| `institute` | 0.0774 | +0.7450 | +0.0577 | Educational institution |
| `embedded systems` | 0.1674 | +0.3422 | +0.0573 | Academic course subject |
| `national` | 0.0566 | +0.9980 | +0.0565 | Institution descriptor |
| `cortex` | 0.0558 | +0.8801 | +0.0491 | Hardware architecture |
| `co2` | 0.0869 | +0.5595 | +0.0486 | Scientific chemical term |
| `learning` | 0.0843 | +0.5728 | +0.0483 | Educational learning term |
| `embedded` | 0.1830 | +0.2577 | +0.0472 | Embedded systems term |
| `total` | 0.0574 | +0.7689 | +0.0442 | Generic grading term |
| `syllabus` | 0.0954 | +0.4394 | +0.0419 | **Crucial academic indicator** |

---

### 1.3 `Major Project Synopsis.pdf`
- **Ground Truth**: `Technology & Computing`
- **Predicted Class**: `Technology & Computing`
- **Decision Margin**: `3.1000` (Top score: +1.8717 vs Second score: -1.2282)
- **Top 3 Decision Scores**:
  1. `Technology & Computing`: **+1.8717**
  2. `Science & Academics`: **-1.2282**
  3. `Medical Health`: **-1.4151**
- **Top Contributing Features**: `ai` (+0.3261), `computer` (+0.2351), `intelligence` (+0.1389), `internet` (+0.1234), `faster` (+0.1196), `mobile` (+0.1135), `artificial intelligence` (+0.0738), `computer science` (+0.0538).
- **Diagnosis**: Correctly classified. High positive weights for computer science project architecture terms.

---

### 1.4 `Speaker1_What_to_Speak_Only.docx`
- **Ground Truth**: `Medical Health`
- **Predicted Class**: `Medical Health`
- **Decision Margin**: `0.9406` (Top score: +1.7279 vs Second score: +0.7873)
- **Top 3 Decision Scores**:
  1. `Medical Health`: **+1.7279**
  2. `Technology & Computing`: **+0.7873** (due to "AI" in disease diagnosis)
  3. `Entertainment`: **-2.9754**
- **Top Contributing Features**: `doctors` (+0.3060), `medical` (+0.1910), `cancer` (+0.1871), `patient` (+0.1826), `doctor` (+0.1608), `disease` (+0.1549), `diagnosis` (+0.1495), `symptoms` (+0.1400), `healthcare` (+0.1112).
- **Diagnosis**: Correctly classified. Clinical medical terms outweighed AI/computing keywords.

---

## 2. Training Dataset Distribution Inspection (`data/taxonomy_v2_augmented/dataset.csv`)

### 2.1 Category Breakdown
- **`Technology & Computing`**: **8,170 documents**
  - **Sources**: 20 Newsgroups (`comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`, `sci.crypt`, `sci.electronics`): 5,013 documents; IAB Tech News: 2,910 documents; BBC Tech: 247 documents.
- **`Science & Academics`**: **6,370 documents**
  - **Sources**: IAB Science News (space/astronomy): 3,662 documents; 20 Newsgroups (`sci.space`): 908 documents; PubMed QA: 1,000 documents; Syllabi/Courseware templates: 800 documents.

### 2.2 Most Frequent TF-IDF Terms
- **`Technology & Computing`**: `windows`, `drive`, `card`, `software`, `program`, `system`, `file`, `graphics`, `code`, `computer`, `data`, `hardware`, `key`, `encryption`, `chip`, `output`, `input`, `print`, `define`.
- **`Science & Academics`**: `space`, `nasa`, `orbit`, `earth`, `moon`, `launch`, `shuttle`, `satellite`, `scientific`, `patient`, `study`, `research`, `syllabus`, `course`, `lab`, `experiments`.

### 2.3 Do Academic Documents Contain Coding/Technical Terminology?
**YES, heavily.** University engineering syllabi, computer science lab manuals, and computational research papers naturally contain dense programming syntax (`print`, `input`, `define`, `self`, `compile`, `code`, `software`, `program`, `assembly`).

---

## 3. Diagnostic Answers to Specific Questions

### A. Why is `LAB MANUAL.pdf` being classified as Technology?
Because the document contains Python programming and deep learning code (`print`, `input`, `define`, `compile`, `self`, `cnn`, `output`). In the LinearSVC model, these programming keywords carry strong positive weights in `Technology & Computing` derived from 5,013 Usenet computing posts. The raw sum of code keyword weights (+0.138 output, +0.134 window, +0.126 cnn, +0.089 print, +0.081 input) outweighs the academic feature contributions.

### B. Why is `SYLLABUS.pdf` being classified as Technology?
`Syllabus.pdf` is an engineering course syllabus covering microcontrollers and assembly programming. Words like `software`, `program`, `compiler`, `code`, `interface`, `hardware`, `assembly` generate large positive contributions for `Technology & Computing`. Meanwhile, structural academic tokens like `syllabus` possess a relatively modest linear coefficient (+0.4394) because the training set historically contained no syllabi prior to the recent 800 synthetic courseware extracts.

### C. Is this primarily a data-distribution problem, feature problem, or taxonomy-boundary problem?
It is primarily a **data-distribution and feature-weighting problem**:
1. **Data Distribution**: `Technology & Computing` contains 5,013 Usenet forum posts packed with programming syntax, creating large feature weights for unigram code keywords (`print`, `define`, `program`, `code`). `Science & Academics` contains 3,662 news articles and 908 space posts, so academic structural bigrams (`course syllabus`, `lab manual`, `grading policy`, `course objectives`) are relatively dilute.
2. **Feature Representation**: Unigram TF-IDF vectors treat code keywords (`print`, `define`, `code`) independently of the surrounding document context (e.g. `grading policy`, `course objectives`, `prerequisites: calculus`).
3. **Not a Taxonomy Boundary Problem**: The taxonomy definition is clear: *"A Python programming document intended as an academic course/lab document $\rightarrow$ Science & Academics."* The model's feature weights simply favor programming syntax over academic structural metadata.

### D. What is the smallest change likely to improve these two documents?
1. **Academic Structural Bigram / Trigram Feature Weighting**: Ensure explicit academic structural n-grams (`course syllabus`, `lab manual`, `grading policy`, `learning objectives`, `course requirements`, `lab exercise`, `prerequisites`) receive strong feature representation and weighting in `Science & Academics`.
2. **Adding Genuine Academic CS/Engineering Syllabi to `Science & Academics`**: Adding 500–1,000 real university CS/Engineering course syllabi and lab manuals into `Science & Academics` so the model learns that programming code enclosed within academic structural metadata belongs to `Science & Academics`.

---

## 4. Final Concise Recommendations

### What to Change
- **Augment `Science & Academics` with Genuine Computer Science / Engineering Academic Courseware**: Include 500–1,000 real-world computer science course syllabi, university lab manuals, and course outlines into `Science & Academics`.
- **Enrich N-gram Feature Vocabulary**: Include bigrams and trigrams in the TF-IDF vectorizer so structural phrases like `course syllabus`, `lab manual`, `learning objectives`, `grading rubric`, `credit hours` are captured as distinct features.

### What NOT to Change
- **DO NOT change the Six-Class Taxonomy**: The category definitions are sound and approved by the teacher.
- **DO NOT change model family**: `LinearSVC` remains the strongest, fastest, and most explainable model architecture (95.98% macro F1 on base V2 test set).
- **DO NOT tune the Anonymous threshold yet**: Keep Anonymous threshold calibration separate until classification features are aligned.

### Expected Risk
- **Minimal Risk**: Adding genuine academic courseware to `Science & Academics` directly addresses the root cause of the error without altering model architecture or breaking production API contracts.
