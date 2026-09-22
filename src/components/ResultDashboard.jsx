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

function ResultDashboard({ result, theme, onUploadAnother, onBackHome }) {
  const { confirmManualSort, domainPalette } = useAppState();
  const [choiceError, setChoiceError] = useState("");
  const [choiceBusy, setChoiceBusy] = useState(false);
  const [scoresExpanded, setScoresExpanded] = useState(true);

  const decisionMargin =
    result?.decisionMargin != null && !Number.isNaN(Number(result.decisionMargin))
      ? Number(result.decisionMargin)
      : null;

  const uncertaintyLevel =
    result?.uncertaintyLevel ||
    (decisionMargin != null
      ? decisionMargin >= 1.0
        ? "High confidence"
        : decisionMargin >= 0.5
        ? "Moderate confidence"
        : "Ambiguous"
      : "Moderate confidence");

  const isAmbiguous =
    uncertaintyLevel.toLowerCase().includes("ambiguous") ||
    (decisionMargin != null && decisionMargin < 0.5);

  const needsChoice = Boolean(result?.requiresManualChoice && result?.pendingId);
  const candidates = Array.isArray(result?.topTwoDomains) ? result.topTwoDomains : [];

  const keywords = useMemo(() => {
    const raw = result?.keywords && result.keywords.length ? result.keywords : result?.fallbackKeywords || [];
    return raw.slice(0, 14);
  }, [result]);

  const folderName = useMemo(() => folderLabelFromPath(result?.storedIn), [result?.storedIn]);

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
    return [
      {
        model: "Multinomial Naive Bayes",
        prediction: result?.prediction || "—",
        score: null,
        score_type: "Probability",
        score_display: "Probability score",
        test_accuracy: 0.9116,
        test_macro_f1: 0.9117,
        is_selected: false,
      },
      {
        model: "Logistic Regression",
        prediction: result?.prediction || "—",
        score: null,
        score_type: "Probability",
        score_display: "Probability score",
        test_accuracy: 0.9406,
        test_macro_f1: 0.9405,
        is_selected: false,
      },
      {
        model: "LinearSVC",
        prediction: result?.prediction || "—",
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
  }, [result?.modelComparison, result?.prediction, decisionMargin, uncertaintyLevel]);

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
            File: <strong>{result.filename || "Uploaded document"}</strong> — No readable text was detected. The document was not submitted to the machine learning classifier and was not sorted into any folder.
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
            <span className="docResultHeader__catLabel">Predicted Category</span>
            <div className="docResultHeader__pillWrap">
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
            </div>
          </div>

          <div className="docResultHeader__statusWrap">
            <span className="docResultHeader__statusLabel">Sorting Status</span>
            {needsChoice ? (
              <span className="docResultHeader__status docResultHeader__status--pending">
                <span className="statusDot statusDot--pending" aria-hidden="true" />
                Pending Review (Ambiguous Margin)
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
        <article
          className="resultHeroCard cardRise"
          style={{ "--accent": theme.color, "--accentSoft": theme.soft, "--accentBorder": theme.border }}
        >
          <div className="resultHeroCard__glow" aria-hidden="true" />
          <div className="resultHeroCard__inner">
            <div className="resultHeroCard__top">
              <span className="eyebrow">Final Prediction · Selected Model</span>
              <span className="resultPill resultPill--large">{theme.label}</span>
              {needsChoice ? (
                <span className="resultHeroCard__success resultHeroCard__success--pending">
                  <span className="resultHeroCard__successDot" aria-hidden="true" />
                  Ambiguous margin — manual review recommended
                </span>
              ) : (
                <span className="resultHeroCard__success">
                  <span className="resultHeroCard__successDot" aria-hidden="true" />
                  Success — file organized automatically
                </span>
              )}
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

        {/* 4. DECISION MARGIN / UNCERTAINTY */}
        <section className="confidencePanel cardRise">
          <div className="confidencePanel__head">
            <span className="eyebrow">Model Decision Certainty</span>
            <div className="confidencePanel__pctRow">
              <strong className="confidencePanel__pct">
                {decisionMargin != null ? decisionMargin.toFixed(2) : "—"}
              </strong>
              <span
                className={`confidencePanel__label ${
                  isAmbiguous ? "confidencePanel__label--ambiguous" : "confidencePanel__label--confident"
                }`}
              >
                {isAmbiguous ? "Ambiguous — manual review recommended" : uncertaintyLevel}
              </span>
            </div>
          </div>

          {/* Visual Decision Margin Meter */}
          <div className="marginMeterWrap" aria-label="Decision Margin Meter">
            <div
              className="marginMeter__bar"
              style={{
                width: `${Math.min(100, Math.max(12, ((decisionMargin || 0) / 3.0) * 100))}%`,
                background: isAmbiguous
                  ? "linear-gradient(90deg, #f59e0b, #ef4444)"
                  : `linear-gradient(90deg, var(--teal), ${theme.color})`,
              }}
            />
          </div>

          <div className="marginScaleLegend">
            <span>0.0 (Boundary)</span>
            <span>0.5 (Moderate)</span>
            <span>1.0+ (High Separation)</span>
          </div>

          <p className="confidencePanel__hint">
            {needsChoice
              ? "Decision margin between top-1 and runner-up is below 0.50. Select one of the two closest categories below to finalize folder sorting."
              : "Decision margin represents the separation between highest and second-highest LinearSVC hyperplane scores (top - second). These are operational decision-margin thresholds, not calibrated probabilities."}
          </p>
        </section>

        {/* 5. MODEL DECISION SCORES ACROSS ALL 6 DOMAINS */}
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

        {/* 6. SORTING RESULT / LOCAL STORAGE */}
        {needsChoice && (
          <section className="manualChoiceCard cardRise">
            <span className="eyebrow">User Review Required</span>
            <h3>Confirm Document Destination</h3>
            <p>
              Decision scores for the top two categories are close (margin &lt; 0.50). Choose the most accurate domain:
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
            </div>

            {choiceError && (
              <p className="choiceError" role="alert">
                {choiceError}
              </p>
            )}
          </section>
        )}

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

        {/* 7. XAI / LINEAR FEATURE ATTRIBUTION */}
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

      {/* 8. ACTIONS */}
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
