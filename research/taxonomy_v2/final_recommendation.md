# DocSort AI — Taxonomy V2 Final Recommendation

## Recommended Model Architecture
**LinearSVC (C=1.0, hinge loss)** is selected with **94.12% Macro F1**.

## Provisional Anonymous Decision Margin Threshold
- **Recommended Provisional Threshold**: **0.25**
- **Known Acceptance Accuracy**: **99.71%**
- **OOD Rejection Rate**: **71.43%** (5 of 7 real-world OOD files rejected to Anonymous)
- **Note**: Threshold is flagged as **provisional** due to limited real-world OOD sample size.

## Deployment Readiness
- **DO NOT DEPLOY YET**: Production backend/main.py and frontend remain untouched as specified.
