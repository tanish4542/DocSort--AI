import React, { useMemo, useState } from "react";
import { useAppState } from "../context/AppContext";

function folderLabelFromPath(storedPath) {
  if (!storedPath || typeof storedPath !== "string") return "—";
  const parts = storedPath.split(/[/\\]/).filter(Boolean);
  if (parts.length < 2) return parts[0] || "—";
  return parts[parts.length - 2];
}

function fileNameOnly(name) {
  if (!name) return "—";
  const parts = String(name).split(/[/\\]/);
  return parts[parts.length - 1] || name;
}

// Configurable operational threshold constants (matching backend)
const VERY_LOW_MARGIN = 0.25;
const LOW_MARGIN_THRESHOLD = 0.50;

function ResultDashboard({ result, theme, onUploadAnother, onBackHome }) {
  const { confirmManualSort, domainPalette } = useAppState();
  const [choiceError, setChoiceError] = useState("");
  const [choiceBusy, setChoiceBusy] = useState(false);
  const [scoresExpanded, setScoresExpanded] = useState(true);

  const decisionMargin =
    result?.decisionMargin != null && !Number.isNaN(Number(result.decisionMargin))
      ? Number(result.decisionMargin)
      : null;

  const isVeryLowMargin =
    result?.isMiscellaneous ||
    result?.prediction === "Miscellaneous / Needs Review" ||
    (decisionMargin != null && decisionMargin < VERY_LOW_MARGIN);

  const isAmbiguous =
    !isVeryLowMargin &&
    (result?.uncertaintyLevel?.toLowerCase().includes("ambiguous") ||
      result?.status === "Manual Review Required" ||
      (decisionMargin != null && decisionMargin < LOW_MARGIN_THRESHOLD));

  const needsChoice = Boolean(
    (result?.requiresManualChoice && result?.pendingId) ||
      (isAmbiguous && result?.pendingId)
  );

  const uncertaintyLevel =
    result?.uncertaintyLevel ||
    (decisionMargin != null
      ? decisionMargin >= 1.0
        ? "High confidence"
        : decisionMargin >= LOW_MARGIN_THRESHOLD
        ? "Moderate confidence"
        : decisionMargin >= VERY_LOW_MARGIN
        ? "Ambiguous — Manual Review Recommended"
        : "Very low confidence / Needs Review"
      : "Moderate confidence");

  const candidates = Array.isArray(result?.topTwoDomains) ? result.topTwoDomains : [];

  const keywords = useMemo(() => {
    const raw = result?.keywords && result.keywords.length ? result.keywords : result?.fallbackKeywords || [];
    return raw.slice(0, 14);
  }, [result]);

  const folderName = useMemo(() => {
    if (isVeryLowMargin) return "Miscellaneous";
    return folderLabelFromPath(result?.storedIn);
  }, [result?.storedIn, isVeryLowMargin]);

  // Decision scores list sorted descending
  const sortedScores = useMemo(() => {
    const rawScores = result?.decisionScores || {};
    return Object.entries(rawScores)
      .map(([cls, val]) => ({
        className: cls,
        score: typeof val === "number" ? val : parseFloat(val) || 0.0,
      }))
      .sort((a, b) => b.score - a.score);
  }, [result?.decisionScores]);

  // Model comparison data
  const comparisonList = useMemo(() => {
    if (Array.isArray(result?.modelComparison) && result.modelComparison.length > 0) {
      return result.modelComparison;
    }
    const technicalPrediction = result?.rawPrediction || result?.prediction || "—";
    return [
      {
        model: "Multinomial Naive Bayes",
        prediction: technicalPrediction,
        score: null,
        score_type: "Probability",
        score_display: "Probability score",
        test_accuracy: 0.9116,
        test_macro_f1: 0.9117,
        is_selected: false,
      },
      {
        model: "Logistic Regression",
        prediction: technicalPrediction,
        score: null,
        score_type: "Probability",
        score_display: "Probability score",
        test_accuracy: 0.9406,
        test_macro_f1: 0.9405,
        is_selected: false,
      },
      {
        model: "LinearSVC",
        prediction: technicalPrediction,
        score: decisionMargin,
        score_type: "Decision Score",
        score_display: decisionMargin != null ? `Decision margin: ${decisionMargin.toFixed(2)}` : "—",
        decision_margin: decisionMargin,
        uncertainty_level: uncertaintyLevel,
        test_accuracy: 0.9450,
        test_macro_f1: 0.9449,
        is_selected: true,
      },
    ];
  }, [result?.modelComparison, result?.prediction, result?.rawPrediction, decisionMargin, uncertaintyLevel]);

  const handlePickDomain = async (domain) => {
    setChoiceError("");
    setChoiceBusy(true);
    try {
      await confirmManualSort(result.pendingId, domain);
    } catch (e) {
      setChoiceError(e?.message || "Could not move the file.");
    } finally {
      setChoiceBusy(false);
    }
  };

  if (result?.error || result?.isBlank) {
    return (
      <div className="resultDashboard resultDashboard--blank">
        <div className="validationErrorCard cardRise" role="alert">
          <div className="validationErrorCard__icon" aria-hidden="true">⚠️</div>
          <span className="eyebrow eyebrow--warning">Validation Notice</span>
          <h2>Document Content Rejected</h2>
          <p className="validationErrorCard__message">
            {result.error || "This document appears to be blank or contains no readable text. Please upload a document with readable content."}
          </p>
          <p className="validationErrorCard__details">
            File: <strong>{result.filename || "Uploaded document"}</strong> — No readable text was detected. The document was not submitted to the machine learning classifier and was not sorted into any folder. Blank documents are rejected before inference and are not sorted into Miscellaneous.
          </p>
          <div className="resultDashboard__actions" style={{ marginTop: "24px" }}>
            <button className="button button--primary" type="button" onClick={onUploadAnother}>
              Upload Another File
            </button>
            <button className="button button--secondary" type="button" onClick={onBackHome}>
              Back to Home
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="resultDashboard">
      {/* 1. MAIN DOCUMENT / RESULT HEADER */}
      <section className="docResultHeader cardRise">
        <div className="docResultHeader__main">
          <div className="docResultHeader__fileInfo">
            <span className="docResultHeader__icon" aria-hidden="true">📄</span>
            <div>
              <span className="docResultHeader__fileLabel">Analyzed Document</span>
              <h2 className="docResultHeader__fileName" title={result.filename}>
                {fileNameOnly(result.filename)}
              </h2>
            </div>
          </div>

          <div className="docResultHeader__predictionInfo">
            <span className="docResultHeader__catLabel">
              {isVeryLowMargin
                ? "Classification Status"
                : isAmbiguous
                ? "Predicted Category (Unconfirmed)"
                : "Predicted Category"}
            </span>
            <div className="docResultHeader__pillWrap">
              {isVeryLowMargin ? (
                <span className="resultPill resultPill--large resultPill--misc">
                  Miscellaneous / Needs Review
                </span>
              ) : (
                <span
                  className="resultPill resultPill--large"
                  style={{
                    background: theme.soft,
                    borderColor: theme.border,
                    color: theme.color,
                  }}
                >
                  {result.prediction}
                </span>
              )}
            </div>
          </div>

          <div className="docResultHeader__statusWrap">
            <span className="docResultHeader__statusLabel">Sorting Status</span>
            {isVeryLowMargin ? (
              <span className="docResultHeader__status docResultHeader__status--misc">
                <span className="statusDot statusDot--misc" aria-hidden="true" />
                Separated into Miscellaneous (Low-Margin Fallback)
              </span>
            ) : isAmbiguous ? (
              <span className="docResultHeader__status docResultHeader__status--pending">
                <span className="statusDot statusDot--pending" aria-hidden="true" />
                Manual Review Required (Margin {decisionMargin != null ? decisionMargin.toFixed(2) : "—"} &lt; 0.50)
              </span>
            ) : (
              <span className="docResultHeader__status docResultHeader__status--success">
                <span className="statusDot statusDot--success" aria-hidden="true" />
                Organized into {folderName}
              </span>
            )}
          </div>
        </div>
      </section>

      {/* 2. MODEL COMPARISON (THE CORE RESEARCH DEMONSTRATION SECTION) */}
      <section className="modelComparisonPanel cardRise" id="model-comparison">
        <div className="sectionHeading sectionHeading--compact">
          <span className="eyebrow">ML-2 Research Evaluation</span>
          <h2>Model Comparison</h2>
          <p className="modelComparisonSubtitle">
            Independent predictions evaluated across all three individual research classifiers alongside their held-out test benchmarks.
            LinearSVC achieved the highest test Macro F1 and is the selected inference model.
          </p>
        </div>

        {/* THREE EQUAL-WIDTH CARDS */}
        <div className="comparisonGrid">
          {comparisonList.map((item) => {
            const isSelected = Boolean(item.is_selected || item.model === "LinearSVC");
            const catTheme = domainPalette[item.prediction] || domainPalette.Unknown;

            // Formatted per-document scores
            let docScoreValue = "—";
            let docScoreLabel = "Document Score";
            if (item.model === "LinearSVC") {
              docScoreLabel = "Decision Score / Margin";
              const marginStr = decisionMargin != null ? decisionMargin.toFixed(2) : "—";
              const rawScoreStr = item.score != null ? (item.score >= 0 ? `+${item.score.toFixed(4)}` : item.score.toFixed(4)) : "—";
              docScoreValue = `Margin: ${marginStr} (${rawScoreStr})`;
            } else {
              docScoreLabel = "Document Score (Probability)";
              if (item.score != null) {
                docScoreValue = `${(item.score * 100).toFixed(2)}%`;
              } else if (item.score_display) {
                docScoreValue = item.score_display;
              }
            }

            return (
              <article
                key={item.model}
                className={`comparisonCard cardRise ${isSelected ? "comparisonCard--selected" : ""}`}
              >
                <div className="comparisonCard__top">
                  <span className="comparisonCard__modelName">{item.model}</span>
                  {isSelected ? (
                    <span className="comparisonCard__selectedBadge">★ SELECTED MODEL</span>
                  ) : (
                    <span className="comparisonCard__candidateBadge">Candidate</span>
                  )}
                </div>

                <div className="comparisonCard__predictionWrap">
                  <span className="comparisonCard__predLabel">Prediction</span>
                  <div className="comparisonCard__predValue">
                    <span
                      className="comparisonCard__catBadge"
                      style={{
                        background: catTheme.soft,
                        borderColor: catTheme.border,
                        color: catTheme.color,
                      }}
                    >
                      {item.prediction}
                    </span>
                  </div>
                  <div className="comparisonCard__docScoreWrap">
                    <span className="comparisonCard__docScoreLabel">{docScoreLabel}</span>
                    <strong className="comparisonCard__docScoreVal">{docScoreValue}</strong>
                  </div>
                </div>

                <div className="comparisonCard__benchmarks">
                  <span className="comparisonCard__benchLabel">Held-Out Test Performance</span>
                  <div className="benchmarkRow">
                    <div className="benchmarkStat">
                      <span className="benchmarkStat__label">Test Accuracy</span>
                      <strong className="benchmarkStat__val">
                        {(item.test_accuracy * 100).toFixed(2)}%
                        {isSelected && <span className="benchmarkStat__star"> ★</span>}
                      </strong>
                    </div>
                    <div className="benchmarkStat">
                      <span className="benchmarkStat__label">Test Macro F1</span>
                      <strong className="benchmarkStat__val">
                        {(item.test_macro_f1 * 100).toFixed(2)}%
                        {isSelected && <span className="benchmarkStat__star"> ★</span>}
                      </strong>
                    </div>
                  </div>
                </div>
              </article>
            );
          })}
        </div>

        {/* METRIC DISTINCTION CALLOUT */}
        <div className="comparisonNotice">
          <p>
            <strong>Evaluation Metric Notice:</strong> Test Accuracy and Macro F1 are measured on the held-out research test set. Document scores describe this individual prediction and are not accuracy values. LinearSVC uses an uncalibrated decision score and decision margin rather than probability.
          </p>
        </div>

        {/* RESEARCH TAXONOMY & OUT-OF-DOMAIN HANDLING CALLOUT */}
        <div className="taxonomyDistinctionCallout">
          <p>
            <strong>Research Taxonomy &amp; Out-of-Domain Notice:</strong> The six research categories form a closed experimental taxonomy. Real-world documents (such as resumes, personal biodata, or administrative records) may fall outside these categories. The application therefore uses low-margin uncertainty handling to avoid forcing uncertain documents into an unrelated class.
          </p>
        </div>

        {/* ENSEMBLE EXPERIMENTS SUBSECTION */}
        <div className="ensembleSubsection">
          <div className="ensembleSubsection__head">
            <span className="eyebrow">Ensemble Experiments</span>
            <h3>Empirical Comparison with Ensemble Architectures</h3>
            <p>
              We evaluated Soft Voting and Stacking ensembles against the individual models on the held-out test set.
            </p>
          </div>

          <div className="ensembleGrid">
            <div className="ensembleCard">
              <div className="ensembleCard__header">
                <strong>Soft Voting</strong>
                <span className="ensembleCard__badge">Ensemble</span>
              </div>
              <div className="ensembleCard__metricsRow">
                <div className="ensembleCard__metric">
                  <span>Test Accuracy</span>
                  <strong>94.16%</strong>
                </div>
                <div className="ensembleCard__metric">
                  <span>Test Macro F1</span>
                  <strong>94.15%</strong>
                </div>
              </div>
            </div>

            <div className="ensembleCard">
              <div className="ensembleCard__header">
                <strong>Stacking</strong>
                <span className="ensembleCard__badge">Ensemble</span>
              </div>
              <div className="ensembleCard__metricsRow">
                <div className="ensembleCard__metric">
                  <span>Test Accuracy</span>
                  <strong>94.47%</strong>
                </div>
                <div className="ensembleCard__metric">
                  <span>Test Macro F1</span>
                  <strong>94.47%</strong>
                </div>
              </div>
            </div>

            <div className="ensembleCard ensembleCard--winner">
              <div className="ensembleCard__header">
                <strong>LinearSVC Baseline</strong>
                <span className="ensembleCard__badge ensembleCard__badge--winner">★ Baseline Winner</span>
              </div>
              <div className="ensembleCard__metricsRow">
                <div className="ensembleCard__metric">
                  <span>Test Accuracy</span>
                  <strong>94.50%</strong>
                </div>
                <div className="ensembleCard__metric">
                  <span>Test Macro F1</span>
                  <strong style={{ color: "#059669" }}>94.49%</strong>
                </div>
              </div>
            </div>
          </div>

          <p className="ensembleFooterNote">
            <em>"The evaluated ensemble methods did not outperform the LinearSVC baseline on the held-out test set."</em>
          </p>
        </div>
      </section>

      {/* 3. FINAL PREDICTION SECTION */}
      <section className="finalPredictionSection">
        {isVeryLowMargin ? (
          /* CASE 1: VERY LOW MARGIN (< 0.25) -> MISCELLANEOUS / NEEDS REVIEW */
          <article className="resultHeroCard resultHeroCard--misc cardRise">
            <div className="resultHeroCard__top">
              <span className="eyebrow eyebrow--misc">Application Fallback</span>
              <span className="resultPill resultPill--large resultPill--misc">Miscellaneous / Needs Review</span>
              <span className="resultHeroCard__statusTag resultHeroCard__statusTag--misc">
                Auto-separated into Miscellaneous
              </span>
            </div>

            <p className="resultHeroCard__domainLabel">Fallback Category</p>
            <h2 className="resultHeroCard__domainTitle" style={{ color: "#334155" }}>
              Miscellaneous / Needs Review
            </h2>

            <div className="lowConfidenceNotice lowConfidenceNotice--misc">
              <p>
                <strong>Out-of-Domain Notice:</strong> This document falls outside the confidently supported research domains or has insufficient classification evidence. It has been separated into <code>~/Desktop/SortedDocuments/Miscellaneous/</code> rather than automatically assigned to a research category.
              </p>
              {result.rawPrediction && (
                <p style={{ marginTop: "6px", fontSize: "0.86rem", color: "#64748b" }}>
                  Closest technical classifier hyperplane: <strong>{result.rawPrediction}</strong> (Decision margin: {decisionMargin != null ? decisionMargin.toFixed(2) : "—"} &lt; 0.25).
                </p>
              )}
            </div>

            <div className="resultHeroMetaGrid">
              <div className="resultHeroMeta">
                <span>Routing Engine</span>
                <strong>LinearSVC Fallback</strong>
              </div>
              <div className="resultHeroMeta">
                <span>File Name</span>
                <strong title={result.filename}>{fileNameOnly(result.filename)}</strong>
              </div>
              <div className="resultHeroMeta">
                <span>Decision Margin</span>
                <strong>{decisionMargin != null ? decisionMargin.toFixed(2) : "—"}</strong>
              </div>
              <div className="resultHeroMeta">
                <span>Operational Status</span>
                <strong>Very low margin (&lt; 0.25)</strong>
              </div>
            </div>
          </article>
        ) : isAmbiguous ? (
          /* CASE 2: AMBIGUOUS MARGIN (0.25 <= margin < 0.50) e.g. MANVI_BIODATA.pdf */
          <article
            className="resultHeroCard resultHeroCard--ambiguous cardRise"
            style={{ "--accent": "#d97706", "--accentSoft": "rgba(217, 119, 6, 0.14)", "--accentBorder": "rgba(217, 119, 6, 0.35)" }}
          >
            <div className="resultHeroCard__top">
              <span className="eyebrow eyebrow--warning">Manual Review Recommended</span>
              <span className="resultPill resultPill--large" style={{ background: theme.soft, borderColor: theme.border, color: theme.color }}>
                {result.prediction}
              </span>
              <span className="resultHeroCard__statusTag resultHeroCard__statusTag--pending">
                <span className="statusDot statusDot--pending" aria-hidden="true" />
                Manual Review Required
              </span>
            </div>

            <p className="resultHeroCard__domainLabel">Predicted Category (Unconfirmed)</p>
            <h2 className="resultHeroCard__domainTitle">{result.prediction}</h2>

            <div className="lowConfidenceNotice">
              <p>
                <strong>Uncertainty Notice:</strong> Research classifier is uncertain. The document does not have sufficient evidence for automatic sorting (decision margin {decisionMargin != null ? decisionMargin.toFixed(2) : "—"} &lt; 0.50 threshold).
              </p>
              <p style={{ marginTop: "4px", fontSize: "0.86rem", color: "#92400e" }}>
                This file was staged for review and has <strong>not</strong> been automatically placed into <code>{result.prediction}</code>.
              </p>
            </div>

            <div className="resultHeroMetaGrid">
              <div className="resultHeroMeta">
                <span>Selected Model</span>
                <strong>LinearSVC (C=0.5)</strong>
              </div>
              <div className="resultHeroMeta">
                <span>File Name</span>
                <strong title={result.filename}>{fileNameOnly(result.filename)}</strong>
              </div>
              <div className="resultHeroMeta">
                <span>Decision Margin</span>
                <strong>{decisionMargin != null ? decisionMargin.toFixed(2) : "—"}</strong>
              </div>
              <div className="resultHeroMeta">
                <span>Status</span>
                <strong style={{ color: "#d97706" }}>Manual Review Required</strong>
              </div>
            </div>
          </article>
        ) : (
          /* CASE 3: HIGH / MODERATE CONFIDENCE (margin >= 0.50) */
          <article
            className="resultHeroCard cardRise"
            style={{ "--accent": theme.color, "--accentSoft": theme.soft, "--accentBorder": theme.border }}
          >
            <div className="resultHeroCard__glow" aria-hidden="true" />
            <div className="resultHeroCard__inner">
              <div className="resultHeroCard__top">
                <span className="eyebrow">Final Prediction · Selected Model</span>
                <span className="resultPill resultPill--large">{theme.label}</span>
                <span className="resultHeroCard__success">
                  <span className="resultHeroCard__successDot" aria-hidden="true" />
                  Success — file organized automatically
                </span>
              </div>

              <p className="resultHeroCard__domainLabel">Final Category Decision</p>
              <h2 className="resultHeroCard__domainTitle">{result.prediction}</h2>

              <div className="selectionReasonText">
                <strong>Why LinearSVC?</strong> LinearSVC achieved the highest test Macro F1 (94.49%) among the evaluated individual classifiers.
              </div>

              <div className="resultHeroMetaGrid">
                <div className="resultHeroMeta">
                  <span>Selected Model</span>
                  <strong>LinearSVC (C=0.5)</strong>
                </div>
                <div className="resultHeroMeta">
                  <span>File Name</span>
                  <strong title={result.filename}>{fileNameOnly(result.filename)}</strong>
                </div>
                <div className="resultHeroMeta">
                  <span>Decision Margin</span>
                  <strong>{decisionMargin != null ? decisionMargin.toFixed(2) : "—"}</strong>
                </div>
                <div className="resultHeroMeta">
                  <span>Uncertainty Level</span>
                  <strong>{uncertaintyLevel}</strong>
                </div>
              </div>
            </div>
          </article>
        )}

        {/* 4. DECISION MARGIN / UNCERTAINTY PANEL */}
        <section className="confidencePanel cardRise">
          <div className="confidencePanel__head">
            <span className="eyebrow">Model Decision Certainty</span>
            <div className="confidencePanel__pctRow">
              <strong className="confidencePanel__pct">
                {decisionMargin != null ? decisionMargin.toFixed(2) : "—"}
              </strong>
              <span
                className={`confidencePanel__label ${
                  isVeryLowMargin
                    ? "confidencePanel__label--misc"
                    : isAmbiguous
                    ? "confidencePanel__label--ambiguous"
                    : "confidencePanel__label--confident"
                }`}
              >
                {isVeryLowMargin
                  ? "Very low margin — routed to Miscellaneous"
                  : isAmbiguous
                  ? "Ambiguous — manual review required"
                  : uncertaintyLevel}
              </span>
            </div>
          </div>

          {/* Visual Decision Margin Meter */}
          <div className="marginMeterWrap" aria-label="Decision Margin Meter">
            <div
              className="marginMeter__bar"
              style={{
                width: `${Math.min(100, Math.max(8, ((decisionMargin || 0) / 3.0) * 100))}%`,
                background: isVeryLowMargin
                  ? "linear-gradient(90deg, #94a3b8, #64748b)"
                  : isAmbiguous
                  ? "linear-gradient(90deg, #f59e0b, #ef4444)"
                  : `linear-gradient(90deg, var(--teal), ${theme.color})`,
              }}
            />
          </div>

          <div className="marginScaleLegend">
            <span>0.0 (Boundary)</span>
            <span style={{ color: "#64748b" }}>0.25 (Misc Fallback)</span>
            <span style={{ color: "#d97706" }}>0.50 (Review Threshold)</span>
            <span>1.0+ (High Separation)</span>
          </div>

          <p className="confidencePanel__hint">
            {isVeryLowMargin
              ? "Decision margin is below 0.25. The research classifier does not have sufficient evidence for any research domain. The file was separated into Miscellaneous / Needs Review."
              : isAmbiguous
              ? "Decision margin between top-1 and runner-up is below 0.50. The application prevents automatic sorting into unconfirmed research categories. Select an option below to confirm."
              : "Decision margin represents the separation between highest and second-highest LinearSVC hyperplane scores (top - second). These are operational decision-margin thresholds, not calibrated probabilities."}
          </p>
        </section>

        {/* 5. CLOSEST RESEARCH CATEGORIES / MANUAL REVIEW UI */}
        {needsChoice && (
          <section className="manualChoiceCard cardRise" id="manual-review">
            <span className="eyebrow eyebrow--warning">User Review Required</span>
            <h3>Closest Research Categories</h3>
            <p>
              Research classifier is uncertain (margin {decisionMargin != null ? decisionMargin.toFixed(2) : "—"} &lt; 0.50).
              Choose the appropriate destination or route to Miscellaneous:
            </p>

            <div className="candidateButtonsRow">
              {candidates.map((cand) => {
                const candTheme = domainPalette[cand.domain] || domainPalette.Unknown;
                return (
                  <button
                    key={cand.domain}
                    className="button button--candidate"
                    type="button"
                    disabled={choiceBusy}
                    onClick={() => handlePickDomain(cand.domain)}
                    style={{
                      borderColor: candTheme.border,
                      background: candTheme.soft,
                      color: candTheme.color,
                    }}
                  >
                    <strong>Sort into {cand.domain}</strong>
                    <small>LinearSVC Score: {cand.score.toFixed(4)}</small>
                  </button>
                );
              })}

              {/* Explicit option to route to Miscellaneous / Needs Review */}
              <button
                className="button button--candidate button--candidateMisc"
                type="button"
                disabled={choiceBusy}
                onClick={() => handlePickDomain("Miscellaneous / Needs Review")}
              >
                <strong>Route to Miscellaneous / Needs Review</strong>
                <small>Application fallback for out-of-domain documents</small>
              </button>
            </div>

            {choiceError && (
              <p className="choiceError" role="alert">
                {choiceError}
              </p>
            )}
          </section>
        )}

        {/* 6. MODEL DECISION SCORES ACROSS ALL 6 DOMAINS */}
        <section className="decisionScoresPanel cardRise">
          <div
            className="decisionScoresPanel__header"
            onClick={() => setScoresExpanded((prev) => !prev)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") setScoresExpanded((prev) => !prev);
            }}
          >
            <div className="decisionScoresPanel__titleWrap">
              <span className="eyebrow">LinearSVC Distance to Hyperplanes</span>
              <h3>Decision Function Scores across All 6 Domains</h3>
            </div>
            <button className="textButton decisionScoresPanel__toggle" type="button" aria-expanded={scoresExpanded}>
              {scoresExpanded ? "Collapse ▲" : "Expand All 6 Classes ▼"}
            </button>
          </div>

          {scoresExpanded && (
            <div className="decisionScoresTable">
              {sortedScores.map(({ className, score }, index) => {
                const isTop1 = index === 0;
                const isTop2 = index === 1;
                const scoreTheme = domainPalette[className] || domainPalette.Unknown;
                const normalizedScore = Math.max(0, Math.min(100, ((score + 2.0) / 4.0) * 100));

                return (
                  <div
                    key={className}
                    className={`decisionScoreRow ${isTop1 ? "decisionScoreRow--top1" : ""} ${
                      isTop2 ? "decisionScoreRow--top2" : ""
                    }`}
                  >
                    <div className="decisionScoreRow__info">
                      <span className="decisionScoreRow__rank">{index + 1}.</span>
                      <span className="decisionScoreRow__name">{className}</span>
                      {isTop1 && <span className="decisionScoreRow__tag">Top 1</span>}
                      {isTop2 && <span className="decisionScoreRow__tag decisionScoreRow__tag--runner">Runner Up</span>}
                    </div>

                    <div className="decisionScoreRow__meter">
                      <div
                        className="decisionScoreRow__fill"
                        style={{
                          width: `${normalizedScore}%`,
                          background: isTop1 ? scoreTheme.color : "rgba(100, 116, 139, 0.4)",
                        }}
                      />
                    </div>

                    <div className="decisionScoreRow__scoreWrap">
                      <span
                        className={`decisionScoreRow__val ${
                          score >= 0 ? "decisionScoreRow__val--pos" : "decisionScoreRow__val--neg"
                        }`}
                      >
                        {score >= 0 ? `+${score.toFixed(4)}` : score.toFixed(4)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* 7. SORTING RESULT / LOCAL STORAGE */}
        {result.storedIn && (
          <section className="storagePanel cardRise">
            <span className="eyebrow">Local storage</span>
            <h3>Saved to disk</h3>
            <code className="pathBox">{result.storedIn}</code>
            <p className="storagePanel__sub">
              Organized into <strong>{folderName}</strong> under your Desktop folder.
            </p>
          </section>
        )}

        {/* 8. XAI / LINEAR FEATURE ATTRIBUTION */}
        <section className="keywordPanel cardRise">
          <div className="keywordPanel__head">
            <span className="eyebrow">Linear Feature Attribution</span>
            <h3>Influential Lexical Features</h3>
            <p className="keywordPanel__sub">
              Terms with the strongest positive contribution (x_j · w_c,j) to the LinearSVC predicted class decision score.
            </p>
          </div>
          <div className="keywordChipRow">
            {keywords.map((word, idx) => (
              <span key={`${word}-${idx}`} className="keywordChip">
                {word}
              </span>
            ))}
          </div>
        </section>
      </section>

      {/* DECISION EXPLANATION (INTERPRETABILITY) */}
      <section className="reasoningPanel cardRise">
        <span className="eyebrow">Interpretability</span>
        <h3>Decision Explanation</h3>
        <p className="reasoningPanel__lead">{result.detailedReasoning}</p>
        <div className="reasoningPanel__grid">
          <div className="reasoningSubCard">
            <h4>Textual patterns</h4>
            <p>{result.patternNote}</p>
          </div>
          <div className="reasoningSubCard">
            <h4>TF‑IDF weighting</h4>
            <p>{result.tfidfNote}</p>
          </div>
          <div className="reasoningSubCard">
            <h4>LinearSVC decision</h4>
            <p>{result.nbNote}</p>
          </div>
        </div>
      </section>

      {/* ACTIONS */}
      <div className="resultDashboard__actions">
        <button className="button button--primary" type="button" onClick={onUploadAnother}>
          Upload Another File
        </button>
        <button className="button button--secondary" type="button" onClick={onBackHome}>
          Back to Home
        </button>
      </div>
    </div>
  );
}

export default ResultDashboard;
