# Out-of-Distribution (OOD) & Unknown Document Strategy Analysis

This document provides a mathematical and architectural analysis of the Out-of-Distribution (OOD) / Unknown document failure mode observed during real-world testing of the DocSort AI classifier.

---

## 1. The Core Failure Mode: Unconstrained Argmax over Negative Hyperplanes

### 1.1 Mathematical Formulation of LinearSVC
Linear Support Vector Classification (LinearSVC) solves a one-vs-rest (OvR) formulation for multi-class classification across $K$ classes ($K=6$ in the research model). For an input document vector $x \in \mathbb{R}^D$ (where $D = 100,000$ TF-IDF features):

$$f_c(x) = w_c^T x + b_c \quad \text{for } c \in \{1, \dots, K\}$$

Where:
- $w_c \in \mathbb{R}^D$ is the weight vector defining the separating hyperplane for class $c$.
- $b_c \in \mathbb{R}$ is the intercept scalar.
- $f_c(x)$ represents the signed orthogonal distance from the document vector $x$ to the decision boundary $w_c^T x + b_c = 0$.

### 1.2 Geometrical Interpretation of Signed Distance
- **$f_c(x) > 0$**: The document falls strictly on the **positive side** of the hyperplane, indicating that the lexical features of $x$ actively support membership in class $c$.
- **$f_c(x) < 0$**: The document falls on the **negative side** of the hyperplane, indicating that the model considers $x$ to be an opponent / non-member of class $c$.
- **$f_c(x) \approx 0$**: The document is on or near the margin boundary.

### 1.3 Why Standard Argmax Fails on Out-of-Domain Inputs
In the production application, the predicted class $\hat{c}$ is selected via:

$$\hat{c} = \arg\max_{c \in \{1, \dots, K\}} f_c(x)$$

And the operational decision margin is calculated as:

$$\Delta(x) = f_{\text{top1}}(x) - f_{\text{top2}}(x)$$

**The Fundamental Flaw:** $\Delta(x)$ is purely a **relative** metric. It measures only the distance between the two highest scores, **completely ignoring their absolute location in feature space**.

### 1.4 Forensic Proof from Real-World Testing (`PDFPrint.pdf`)
During live user testing, `PDFPrint.pdf` (an online tuition fee payment receipt) produced the following decision vector:

| Class $c$ | Decision Score $f_c(x)$ | Interpretation |
| :--- | :---: | :--- |
| **Technology & Computing** | $\mathbf{-0.0167}$ | Negative (Opposed) |
| **Sports** | $-0.7107$ | Negative (Opposed) |
| **Business and Finance** | $-0.7224$ | Negative (Opposed) |
| **Entertainment** | $-0.8714$ | Negative (Opposed) |
| **Medical Health** | $-0.9225$ | Negative (Opposed) |
| **Science** | $-0.9728$ | Negative (Opposed) |

**Key Observations:**
1. **Universal Rejection:** $\max_c f_c(x) = -0.0167 < 0$. **Every single class hyperplane in the model rejected this document.**
2. **False Confidence:** Despite unanimous rejection, the difference between the least negative class ($-0.0167$) and the second-least negative class ($-0.7107$) produced:
   $$\Delta(x) = -0.0167 - (-0.7107) = \mathbf{0.6940}$$
3. **Erroneous Auto-Sort:** Because the operational rule used `margin >= 0.50 -> Auto-Sort`, the application auto-sorted a college payment receipt into `~/Desktop/SortedDocuments/Technology & Computing/`!
4. **Why Technology Scored Least Negative:** The receipt contained the student's degree title (`ARTIFICIAL INTELLIGENCE & MACHINE LEARNING`) and department (`COMPUTER SCIENCE & ENGINEERING`). In the absence of receipt/invoice concepts, these two incidental token occurrences prevented Technology from being pushed as deeply negative as the other news classes.

---

## 2. Comparative Analysis of Candidate OOD Mitigation Strategies

We evaluate six potential architectures for handling out-of-domain and unclassifiable documents in DocSort V2.

| Strategy | Mechanism | Complexity | Latency Impact | Pros | Cons |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **1. Absolute Hyperplane Thresholding** | Reject if $\max_c f_c(x) < \tau_{\text{abs}}$ | Minimal ($O(1)$) | Zero ($< 0.1$ ms) | Directly prevents negative-support false positives; trivial to compute | Requires threshold tuning ($\tau_{\text{abs}} \in [0.0, 0.2]$) |
| **2. Dual-Gated Margin (Recommended)** | Accept if $\max_c f_c(x) \ge \tau_{\text{abs}}$ **AND** $\Delta(x) \ge \tau_{\text{rel}}$ | Minimal | Zero | Combines absolute class membership support with relative class separation | Requires tuning two hyperparameters |
| **3. TF-IDF Centroid Cosine Distance** | Compute cosine similarity to class centroids $C_k$; reject if $\max_k \cos(x, C_k) < \theta$ | Low | Minimal ($< 1$ ms) | Bounded metric $[0, 1]$; measures dense vocabulary overlap | Sensitive to document length variations |
| **4. Calibrated Logistic Probabilities** | Platt scaling / Softmax with entropy threshold: $H(p) = -\sum p_i \log p_i$ | Moderate | Low ($< 2$ ms) | Provides probabilistic semantics; entropy flags uniform posteriors | Softmax over-confidence on extreme OOD vectors is well-documented |
| **5. One-Class SVM / Isolation Forest** | Train independent density estimator on valid training space | High | Moderate ($15-50$ ms) | Principled novelty detection | High-dimensional TF-IDF ($100k$) causes curse of dimensionality; high memory |
| **6. Explicit "Other / Unknown" Class** | Train with a 7th class containing diverse OOD documents | Moderate | Zero | Allows model to learn negative boundary representations | Cannot represent the infinite variety of unseen non-target documents |

---

## 3. Detailed Examination of Candidate Strategies

### Strategy 1: Absolute Hyperplane Thresholding
- **Formula:**
  $$\text{Valid Support} \iff \max_{c \in \{1, \dots, K\}} f_c(x) > \tau_{\text{abs}}$$
  Where $\tau_{\text{abs}} = 0.0$ (or a slightly conservative $+0.10$).
- **Impact on Real Tests:**
  - `PDFPrint.pdf`: $\max_c f_c(x) = -0.0167 < 0 \implies$ **REJECTED** (Sent to Manual Review).
  - `ID card 25 Mar 2025.pdf`: $\max_c f_c(x) = -0.4244 < 0 \implies$ **REJECTED** (Sent to Manual Review).
  - `COURSERA SDV.pdf`: $\max_c f_c(x) = -0.4277 < 0 \implies$ **REJECTED** (Sent to Manual Review).
  - `morphological_Image_Processing.pdf`: $\max_c f_c(x) = -0.0937 < 0 \implies$ **REJECTED** (Sent to Manual Review).
- **Evaluation:** This single $O(1)$ check would have successfully caught **100%** of the spurious real-world misclassifications observed in testing!

### Strategy 2: Dual-Gated Margin (The Recommended Core Engine)
A document is classified automatically into folder $c^*$ if and only if **both gates** pass:

$$\text{Auto-Sort} \iff \begin{cases}
f_{c^*}(x) \ge \tau_{\text{support}} & \text{(Gate 1: Absolute Class Support, e.g. } \ge 0.15\text{)} \\
f_{c^*}(x) - f_{\text{runner-up}}(x) \ge \tau_{\text{margin}} & \text{(Gate 2: Relative Decision Margin, e.g. } \ge 0.60\text{)}
\end{cases}$$

If **Gate 1 fails**: Label = `"Low Class Support — Out-of-Distribution / Unknown"`. File staged in `_pending`.
If **Gate 1 passes but Gate 2 fails**: Label = `"Ambiguous — Multi-Class Conflict"`. File staged in `_pending`.
If **both pass**: Label = `"High Confidence Auto-Sort"`. File auto-sorted.

### Strategy 3: Dedicated "Other / Miscellaneous" Training Class
While an absolute threshold rejects vectors that have no positive projection, training an explicit 7th category (`Other / Miscellaneous`) provides positive training signals for common non-sortable items:
- Generic letters, blank templates, license agreements, meeting notes.
- However, machine learning theory dictates that a single class cannot model the open complement of a closed set:
  $$\mathcal{X}_{\text{Other}} = \mathbb{R}^D \setminus \bigcup_{k=1}^K \mathcal{X}_k$$
- Therefore, an explicit `Other` class must be used **in combination with** dual-gated thresholding, not as a standalone solution.

---

## 4. Production V2 Decision Pipeline

```
Raw Upload
    │
    ▼
Text Extraction (with Length & Density Check)
    │
    ├── Text Length < 80 characters ──────────► [STAGING: Extraction Failure / Insufficient Text]
    │
    ▼
TF-IDF Transform
    │
    ▼
LinearSVC Decision Function: f_c(x)
    │
    ├── Gate 1: max_c f_c(x) < 0.15 ──────────► [STAGING: Out-of-Distribution / Unknown Document]
    │
    ├── Gate 2: f_top1 - f_top2 < 0.50 ──────► [STAGING: Ambiguous Class Boundary Review]
    │
    ▼
Auto-Sort into Target Category Folder
```

---

## 5. Summary Recommendation for V2
1. **Implement Dual-Gated Margin immediately** in the inference module. It introduces zero inference latency, requires zero external libraries, and mathematically eliminates negative-hyperplane auto-sorting.
2. **Add a Minimum Text Length Guard** ($L \ge 100$ characters) before vectorization to catch empty scans and stub ID cards.
3. **Introduce an explicit `Administrative & Identification` and `Financial & Receipts` category** in the V2 training data so receipts and cards have a legitimate positive home rather than being forced into OOD rejection.
