# Production Evaluation Strategy for DocSort AI V2

This document establishes the two-tier evaluation framework for DocSort AI V2, defining metrics that prioritize real-world sorting safety over simple in-domain accuracy.

---

## 1. The Production Evaluation Philosophy

In a research paper, the primary metric is typically **Macro F1** on a balanced, held-out test split.

In a production desktop document sorting application, **Macro F1 alone is dangerously inadequate**:
- If an automated sorter achieves 95% accuracy by auto-sorting 100% of files, **5% of the user's files are placed into the wrong local directory without warning**. The user loses track of their files.
- Conversely, asking the user to manually confirm a difficult document via the `_pending` staging queue is **safe, helpful, and expected behavior**.

### The Core Production KPI: False Automatic Sorting Rate (FASR)
$$\text{FASR} = \frac{\text{Number of Auto-Sorted Documents Routed to an Incorrect Folder}}{\text{Total Number of Documents Auto-Sorted}} \times 100\%$$

**Production Target**: $\mathbf{\text{FASR} < 1.5\%}$.
An error where the system quietly puts a college tuition receipt into `Technology & Computing` must be treated as a critical production bug.

---

## 2. Two-Tier Evaluation Benchmark Architecture

```
                               DOCSORT V2 EVALUATION SUITE
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
       BENCHMARK A: IN-DOMAIN                                    BENCHMARK B: REAL-WORLD
       Held-Out Test Split                                       Desktop "Wild" Corpus
       (N = 2,700 documents)                                     (N = 200 documents)
       ├── 6 Balanced Classes (450/class)                        ├── 150 Target Real Documents
       ├── Evaluates Model Capacity                              ├── 50 Deliberate OOD Documents
       └── Standard ML Metrics (F1, Acc)                         └── Safety Metrics (FASR, MRR, OOD)
```

---

## 3. Benchmark Specifications

### Benchmark A: In-Domain Test Split
- **Source**: Stratified 15% split of the curated production dataset (`data/production_v2/splits/test.csv`).
- **Volume**: Exactly 2,700 documents (450 per class).
- **Purpose**: Verifies that the model has generalized across the core vocabulary of all 6 production categories without overfitting.
- **Evaluation Criteria**:
  - Macro F1 $\ge 0.930$
  - Per-class Recall $\ge 0.900$ across all 6 classes
  - Precision $\ge 0.900$ across all 6 classes
  - Confusion Matrix inspection to ensure no persistent class-pair bleeding.

---

### Benchmark B: Real-World "Wild" Desktop Document Benchmark
- **Source**: Dedicated out-of-dataset evaluation folder (`data/production_v2/desktop_wild_test/`).
- **Volume**: Exactly 200 real documents collected from actual desktop workspaces.
- **Composition**:
  - **150 In-Taxonomy Real Documents (25 per class)**:
    - Messy multi-page PDF syllabi, lecture slides, coding exercises, receipt scans, blood test reports, bank PDFs, offer letters.
    - Varying text lengths (from 120-character ID cards to 40,000-character research preprints).
  - **50 Out-of-Distribution (OOD) Negative Controls**:
    - 10 Food recipes and cooking blog PDFs
    - 10 Travel itineraries and boarding passes
    - 10 Fiction novel chapters
    - 10 Scanned graphical flyers / real-estate brochures
    - 10 Corrupt / unextractable PDFs (blank scans)

---

## 4. Key Production Metrics & Target Thresholds

| Metric | Mathematical Definition | Production Target | Failure Action |
| :--- | :--- | :---: | :--- |
| **False Automatic Sorting Rate (FASR)** | $\frac{\text{Auto-Sorted Errors}}{\text{Total Auto-Sorted}} \times 100\%$ | $\mathbf{< 1.5\%}$ | Raise decision margin threshold $\tau_{\text{margin}}$ or absolute support threshold $\tau_{\text{support}}$ |
| **Manual Review Rate (MRR)** | $\frac{N_{\text{staged in } \texttt{\_pending}}}{N_{\text{total}}} \times 100\%$ | $\mathbf{15\% - 25\%}$ | If $> 35\%$, model is overly conservative; if $< 10\%$, FASR will spike |
| **OOD Rejection Recall** | $\frac{N_{\text{OOD staged in } \texttt{\_pending}}}{N_{\text{total OOD}}} \times 100\%$ | $\mathbf{\ge 96\%}$ | Adjust absolute hyperplane support gate ($\max f_c(x) \ge 0.15$) |
| **Macro F1 (In-Domain)** | $\frac{1}{K}\sum_{k=1}^K F1_k$ | $\mathbf{\ge 0.93}$ | Inspect feature weighting, class imbalance, or vocabulary size |
| **Extraction Failure Handling** | Graceful fallback on empty / corrupt files | $\mathbf{100\%}$ | Rejection to UI with user guidance; zero silent crashes |

---

## 5. Dual-Gated Operating Point Optimization

The validation split (`data/production_v2/splits/validation.csv`) must be used to generate a **Safety-vs-Automation Operating Curve**:

```
                       FASR vs. Manual Review Rate Trade-Off
           ▲
    FASR   │  ▲ (High False Sorting - Dangerous)
    (%)    │  │       x (tau_margin = 0.20, FASR = 4.2%, MRR = 8%)
           │  │
           │  │            x (tau_margin = 0.40, FASR = 2.1%, MRR = 14%)
           │  │
    1.5% ──┼──┼─────────────────★ OPERATING TARGET (tau_margin=0.55, tau_support=0.15)
           │  │                   FASR = 0.9%, MRR = 19%
           │  │
           │  └────────────────────────────────────────────────────────►
           0                       15%             25%              MRR (%)
```

### Protocol for Determining Optimal Thresholds:
1. Grid search across $\tau_{\text{support}} \in [0.00, 0.30]$ (step 0.05) and $\tau_{\text{margin}} \in [0.30, 0.80]$ (step 0.05) on the Validation split.
2. Select the operating point $(\tau^*_{\text{support}}, \tau^*_{\text{margin}})$ that minimizes MRR subject to the strict constraint:
   $$\text{FASR} \le 1.0\%$$
3. Freeze $(\tau^*_{\text{support}}, \tau^*_{\text{margin}})$ before touching Benchmark A (Test split) or Benchmark B (Wild test).
