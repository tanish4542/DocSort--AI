import React, { useMemo, useState } from "react";
import { useAppState } from "../context/AppContext";

function fileNameOnly(name) {
  if (!name) return "—";
  const parts = String(name).split(/[/\\]/);
  return parts[parts.length - 1] || name;
}

// Configurable operational threshold constants (matching backend)
const VERY_LOW_MARGIN = 0.25;

function ResultDashboard({ result, theme, onUploadAnother, onBackHome }) {
  const {
    domainPalette,
    isFileSystemAccessSupported,
    saveResultToFolder,
    downloadResultFile,
    localDirName,
  } = useAppState();
  const [scoresExpanded, setScoresExpanded] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState("");

  const handleManualSave = async () => {
    try {
      setIsSaving(true);
      setSaveError("");
      await saveResultToFolder(result);
    } catch (err) {
      setSaveError(err?.message || "Failed to save file locally.");
    } finally {
      setIsSaving(false);
    }
  };

  const decisionMargin =
    result?.decisionMargin != null && !Number.isNaN(Number(result.decisionMargin))
      ? Number(result.decisionMargin)
      : null;

  const isAnonymous = Boolean(
    result?.is_anonymous ||
      result?.prediction === "Anonymous" ||
      result?.final_destination === "Anonymous" ||
      (decisionMargin != null && decisionMargin < VERY_LOW_MARGIN)
  );

  const displayCategory = isAnonymous ? "Anonymous" : result?.final_destination || result?.predicted_class || result?.prediction || "Unknown";
  const displayTheme = domainPalette[displayCategory] || domainPalette.Anonymous;
  const originalFileName = fileNameOnly(result?.filename) || "document";

  // Clean, filtered display tokens for feature attribution (removing short, numeric or noisy fragments)
  const cleanKeywords = useMemo(() => {
    const raw = result?.keywords && result.keywords.length ? result.keywords : result?.fallbackKeywords || [];
    return raw
      .filter((word) => {
        if (!word || typeof word !== "string") return false;
        const trimmed = word.trim();
        if (trimmed.length < 3) return false;
        // Exclude purely numeric or alphanumeric noise like "2025", "104", "1a"
        if (/^\d+$/.test(trimmed) || /^\d+[a-z]?$/i.test(trimmed)) return false;
        // Exclude symbols/noise
        if (/^[^a-zA-Z0-9]+$/.test(trimmed) || trimmed.startsWith("_")) return false;
        return true;
      })
      .slice(0, 14);
  }, [result]);

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
            Document Name: <strong>{originalFileName}</strong> — No readable text was detected. Blank documents are rejected before inference and are not sorted into any folder.
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
            <div style={{ minWidth: 0 }}>
              <span className="docResultHeader__fileLabel">DOCUMENT NAME</span>
              <h2 className="docResultHeader__fileName" title={result.filename}>
                {originalFileName}
              </h2>
            </div>
          </div>

          <div className="docResultHeader__predictionInfo">
            <span className="docResultHeader__catLabel">
              {isAnonymous ? "Classification Destination" : "Predicted Category"}
            </span>
            <div className="docResultHeader__pillWrap">
              <span
                className="resultPill resultPill--large"
                style={{
                  background: displayTheme.soft,
                  borderColor: displayTheme.border,
                  color: displayTheme.color,
                }}
              >
                {displayCategory}
              </span>
            </div>
          </div>

          <div className="docResultHeader__statusWrap">
            <span className="docResultHeader__statusLabel">Sorting Status</span>
            <span className="docResultHeader__status docResultHeader__status--success">
              <span className="statusDot statusDot--success" aria-hidden="true" />
              Automatically Sorted
            </span>
          </div>
        </div>
      </section>

      {/* 2. FINAL PREDICTION / ANONYMOUS DISPLAY */}
      <section className="finalPredictionSection">
        {isAnonymous ? (
          <article className="resultHeroCard resultHeroCard--misc cardRise">
            <div className="resultHeroCard__top">
              <span className="eyebrow eyebrow--misc">Low-Confidence Handling</span>
              <span className="resultPill resultPill--large resultPill--misc">Anonymous</span>
              <span className="resultHeroCard__statusTag resultHeroCard__statusTag--misc">
                Automatically Sorted
              </span>
            </div>

            <p className="resultHeroCard__domainLabel">Final Destination</p>
            <h2 className="resultHeroCard__domainTitle" style={{ color: "#334155" }}>
              Anonymous
            </h2>

            <div className="lowConfidenceNotice lowConfidenceNotice--misc">
              <p>
                <strong>Low-confidence document — automatically placed in Anonymous.</strong>
              </p>
              <p style={{ marginTop: "6px", fontSize: "0.86rem", color: "#64748b" }}>
                Decision margin ({decisionMargin != null ? decisionMargin.toFixed(2) : "—"}) fell below the 0.25 threshold. Placed into <code>~/Desktop/SortedDocuments/Anonymous/</code>.
              </p>
            </div>

            <div className="resultHeroMetaGrid">
              <div className="resultHeroMeta">
                <span>DOCUMENT NAME</span>
                <strong title={result.filename} style={{ wordBreak: "break-word" }}>{originalFileName}</strong>
              </div>
              <div className="resultHeroMeta">
                <span>Model</span>
                <strong>TF-IDF + Structural Features + LinearSVC</strong>
              </div>
              <div className="resultHeroMeta">
                <span>Decision Margin</span>
                <strong>{decisionMargin != null ? decisionMargin.toFixed(2) : "—"}</strong>
              </div>
              <div className="resultHeroMeta">
                <span>Sorting Status</span>
                <strong>Automatically Sorted</strong>
              </div>
            </div>
          </article>
        ) : (
          <article
            className="resultHeroCard cardRise"
            style={{ "--accent": displayTheme.color, "--accentSoft": displayTheme.soft, "--accentBorder": displayTheme.border }}
          >
            <div className="resultHeroCard__glow" aria-hidden="true" />
            <div className="resultHeroCard__inner">
              <div className="resultHeroCard__top">
                <span className="eyebrow">Production Inference</span>
                <span className="resultPill resultPill--large">{displayTheme.label}</span>
                <span className="resultHeroCard__success">
                  <span className="resultHeroCard__successDot" aria-hidden="true" />
                  Automatically Sorted
                </span>
              </div>

              <p className="resultHeroCard__domainLabel">Predicted Category</p>
              <h2 className="resultHeroCard__domainTitle">{displayCategory}</h2>

              <div className="resultHeroMetaGrid">
                <div className="resultHeroMeta">
                  <span>DOCUMENT NAME</span>
                  <strong title={result.filename} style={{ wordBreak: "break-word" }}>{originalFileName}</strong>
                </div>
                <div className="resultHeroMeta">
                  <span>Model</span>
                  <strong>TF-IDF + Structural Features + LinearSVC</strong>
                </div>
                <div className="resultHeroMeta">
                  <span>Decision Margin</span>
                  <strong>{decisionMargin != null ? decisionMargin.toFixed(2) : "—"}</strong>
                </div>
                <div className="resultHeroMeta">
                  <span>Sorting Status</span>
                  <strong>Automatically Sorted</strong>
                </div>
              </div>
            </div>
          </article>
        )}

        {/* 3. DECISION MARGIN / UNCERTAINTY PANEL */}
        <section className="confidencePanel cardRise">
          <div className="confidencePanel__head">
            <span className="eyebrow">Decision Margin</span>
            <div className="confidencePanel__pctRow">
              <strong className="confidencePanel__pct">
                {decisionMargin != null ? decisionMargin.toFixed(2) : "—"}
              </strong>
              <span
                className={`confidencePanel__label ${
                  isAnonymous ? "confidencePanel__label--misc" : "confidencePanel__label--confident"
                }`}
              >
                {isAnonymous
                  ? "Anonymous (Margin < 0.25)"
                  : decisionMargin != null && decisionMargin >= 1.0
                  ? "High Separation"
                  : "Moderate Separation"}
              </span>
            </div>
          </div>

          <div className="marginMeterWrap" aria-label="Decision Margin Meter">
            <div
              className="marginMeter__bar"
              style={{
                width: `${Math.min(100, Math.max(8, ((decisionMargin || 0) / 3.0) * 100))}%`,
                background: isAnonymous
                  ? "linear-gradient(90deg, #94a3b8, #64748b)"
                  : `linear-gradient(90deg, var(--teal), ${displayTheme.color})`,
              }}
            />
          </div>

          <div className="marginScaleLegend">
            <span>0.0 (Boundary)</span>
            <span style={{ color: "#64748b", fontWeight: "700" }}>0.25 (Anonymous Threshold)</span>
            <span>1.0+ (High Separation)</span>
          </div>
        </section>

        {/* 4. MODEL DECISION SCORES ACROSS ALL 6 DOMAINS */}
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

        {/* 5. SORTING RESULT / LOCAL STORAGE */}
        <section className="storagePanel cardRise">
          <div className="storagePanel__head">
            <span className="eyebrow">{result.savedLocally ? "Local Folder Organization" : "Sorting Destination"}</span>
            <h3>{result.savedLocally ? "Saved to your local folder" : "Classification Destination"}</h3>
          </div>

          {result.savedLocally ? (
            <div className="storageStatusCard storageStatusCard--success">
              <div className="storageStatusCard__badge">
                <span className="statusDot statusDot--green" />
                <strong>Saved locally</strong>
              </div>
              <p className="storageStatusCard__desc">
                Organized into <strong>{result.localCategory || displayCategory}</strong> as{" "}
                <span className="mono">{result.localFilename || originalFileName}</span>.
              </p>
              <code className="pathBox">
                {result.localPath || `${localDirName || "SortedDocuments"}/${result.localCategory || displayCategory}/${result.localFilename || originalFileName}`}
              </code>
              <div className="storageStatusCard__actions" style={{ marginTop: "14px", display: "flex", gap: "10px", flexWrap: "wrap" }}>
                <button
                  className="button button--secondary"
                  type="button"
                  onClick={() => downloadResultFile(result)}
                  style={{ fontSize: "0.85rem", padding: "8px 16px" }}
                >
                  ⬇️ Download a Copy
                </button>
              </div>
            </div>
          ) : (
            <div className="storageStatusCard storageStatusCard--pending">
              <div className="storageStatusCard__badge" style={{ color: "#92400e" }}>
                <span className="statusDot statusDot--amber" />
                <strong>Destination Category: {result.final_destination || displayCategory}</strong>
              </div>
              <p className="storageStatusCard__desc">
                Target Folder: <strong>{displayCategory}</strong>
                {isAnonymous && <span> (Low Margin &lt; {VERY_LOW_MARGIN.toFixed(2)})</span>}
              </p>
              <p className="storagePanel__sub" style={{ margin: "6px 0 14px 0" }}>
                {isFileSystemAccessSupported
                  ? "Select your local SortedDocuments folder to save this file directly to your computer, or download it."
                  : "Direct folder access is not supported in this browser. You can download the classified file directly."}
              </p>
              {saveError && (
                <p style={{ color: "#dc2626", fontSize: "0.85rem", margin: "0 0 10px" }}>
                  ⚠️ {saveError}
                </p>
              )}
              <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", alignItems: "center" }}>
                {isFileSystemAccessSupported && (
                  <button
                    className="button button--primary"
                    type="button"
                    onClick={handleManualSave}
                    disabled={isSaving}
                    style={{ fontSize: "0.88rem", padding: "10px 18px" }}
                  >
                    {isSaving ? "Saving to Folder..." : "📁 Choose Local Folder & Save"}
                  </button>
                )}
                <button
                  className="button button--secondary"
                  type="button"
                  onClick={() => downloadResultFile(result)}
                  style={{ fontSize: "0.88rem", padding: "10px 18px" }}
                >
                  ⬇️ Download Sorted File
                </button>
              </div>
            </div>
          )}
        </section>

        {/* 6. XAI / LINEAR FEATURE ATTRIBUTION */}
        <section className="keywordPanel cardRise">
          <div className="keywordPanel__head">
            <span className="eyebrow">Linear Feature Attribution</span>
            <h3>Influential Lexical Features</h3>
            <p className="keywordPanel__sub">
              Terms with the strongest positive contribution (x_j · w_c,j) to the LinearSVC predicted class decision score.
            </p>
          </div>
          <div className="keywordChipRow">
            {cleanKeywords.map((word, idx) => (
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
