import React, { useEffect, useMemo, useState } from "react";
import { useAppState } from "../context/AppContext";

const miningOps = [
  {
    key: "extract",
    title: "Text Extraction",
    text: "Raw text is pulled from PDF, DOCX, or TXT using format-specific parsers.",
    icon: "📄",
  },
  {
    key: "noise",
    title: "Noise Removal",
    text: "Non-alphabetic characters are stripped so tokens focus on meaningful words.",
    icon: "🧹",
  },
  {
    key: "token",
    title: "Tokenization",
    text: "The cleaned stream is split into lowercase word tokens for analysis.",
    icon: "✂️",
  },
  {
    key: "stop",
    title: "Stopword Removal",
    text: "High-frequency English stopwords are removed to reduce uninformative noise.",
    icon: "🚫",
  },
  {
    key: "tfidf",
    title: "TF-IDF Vectorization",
    text: "Each term is scored by term frequency and inverse document frequency.",
    icon: "📊",
  },
  {
    key: "weight",
    title: "Feature Weighting",
    text: "Domain-specific vocabulary receives higher weights in the sparse feature vector.",
    icon: "⚖️",
  },
  {
    key: "nb",
    title: "Naive Bayes Classification",
    text: "Class-conditional word probabilities select the most likely domain label.",
    icon: "🧠",
  },
  {
    key: "prob",
    title: "Probability Analysis",
    text: "Posterior scores quantify how strongly each category is supported by the text.",
    icon: "📈",
  },
  {
    key: "sort",
    title: "Folder Sorting",
    text: "The backend moves the file into the predicted domain folder on your machine.",
    icon: "📁",
  },
];

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

function confidenceLabel(pct) {
  if (pct == null || Number.isNaN(pct)) return "Confidence unavailable";
  if (pct >= 80) return "High Confidence";
  if (pct >= 55) return "Medium Confidence";
  return "Low Confidence";
}

function ResultDashboard({ result, theme, onUploadAnother, onBackHome }) {
  const { confirmManualSort, domainPalette } = useAppState();
  const [barWidth, setBarWidth] = useState(0);
  const [choiceError, setChoiceError] = useState("");
  const [choiceBusy, setChoiceBusy] = useState(false);
  const confidence = result.confidence;
  const pct = confidence != null ? Math.min(100, Math.max(0, confidence)) : null;
  const needsChoice = Boolean(result.requiresManualChoice && result.pendingId);
  const candidates = Array.isArray(result.topTwoDomains) ? result.topTwoDomains : [];

  useEffect(() => {
    setBarWidth(0);
    const t = requestAnimationFrame(() => {
      if (pct != null) setBarWidth(pct);
    });
    return () => cancelAnimationFrame(t);
  }, [pct, result.timestamp]);

  const keywords = useMemo(() => {
    const raw = result.keywords && result.keywords.length ? result.keywords : result.fallbackKeywords || [];
    return raw.slice(0, 14);
  }, [result]);

  const folderName = useMemo(() => folderLabelFromPath(result.storedIn), [result.storedIn]);

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

  return (
    <div className="resultDashboard">
      <article
        className="resultHeroCard cardRise"
        style={{ "--accent": theme.color, "--accentSoft": theme.soft, "--accentBorder": theme.border }}
      >
        <div className="resultHeroCard__glow" aria-hidden="true" />
        <div className="resultHeroCard__inner">
          <div className="resultHeroCard__top">
            <span className="eyebrow">Classification complete</span>
            <span className="resultPill resultPill--large">{theme.label}</span>
            {needsChoice ? (
              <span className="resultHeroCard__success resultHeroCard__success--pending">
                <span className="resultHeroCard__successDot" aria-hidden="true" />
                Model confidence is under 100% — pick the correct folder below
              </span>
            ) : (
              <span className="resultHeroCard__success">
                <span className="resultHeroCard__successDot" aria-hidden="true" />
                Success — file organized automatically
              </span>
            )}
          </div>

          <p className="resultHeroCard__domainLabel">Predicted domain</p>
          <h2 className="resultHeroCard__domainTitle">{result.prediction}</h2>

          <div className="resultHeroMetaGrid">
            <div className="resultHeroMeta">
              <span>File name</span>
              <strong title={result.filename}>{fileNameOnly(result.filename)}</strong>
            </div>
            <div className="resultHeroMeta">
              <span>File type</span>
              <strong>{result.fileType}</strong>
            </div>
            <div className="resultHeroMeta">
              <span>File size</span>
              <strong>{result.fileSize}</strong>
            </div>
          </div>
        </div>
      </article>

      {pct != null && (
        <section className="confidencePanel cardRise">
          <div className="confidencePanel__head">
            <span className="eyebrow">Model certainty</span>
            <div className="confidencePanel__pctRow">
              <strong className="confidencePanel__pct">{Math.round(pct)}%</strong>
              <span className="confidencePanel__label">{confidenceLabel(pct)}</span>
            </div>
          </div>
          <div className="confidenceBar" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
            <div
              className="confidenceBar__fill"
              style={{
                width: `${barWidth}%`,
                background: `linear-gradient(90deg, var(--teal), ${theme.color})`,
              }}
            />
          </div>
          <p className="confidencePanel__hint">
            {needsChoice
              ? "The model’s top class did not reach 100% probability. Choose between the two closest domains so the file is sorted correctly."
              : "Derived from Naive Bayes class probabilities — higher values indicate stronger agreement between the document’s TF‑IDF profile and the predicted category."}
          </p>
        </section>
      )}

      {needsChoice && candidates.length > 0 && (
        <section className="domainChoiceSection cardRise" aria-labelledby="domain-choice-heading">
          <div className="domainChoiceSection__head">
            <span className="eyebrow">Your decision</span>
            <h3 id="domain-choice-heading">Which folder should this document use?</h3>
            <p className="domainChoiceSection__sub">
              Select one of the two most likely domains. The file is held in a pending queue until you choose.
            </p>
          </div>
          {choiceError ? <p className="domainChoiceSection__error">{choiceError}</p> : null}
          <div className="domainChoiceGrid">
            {candidates.map((row, idx) => {
              const label = String(row.domain || "").trim();
              const pal = domainPalette[label] || domainPalette.Unknown;
              const confVal =
                row.confidence != null && !Number.isNaN(Number(row.confidence))
                  ? Math.round(Number(row.confidence))
                  : "—";
              return (
                <article
                  key={`${label}-${idx}`}
                  className="domainChoiceCard cardRise"
                  style={{
                    "--accent": pal.color,
                    "--accentSoft": pal.soft,
                    "--accentBorder": pal.border,
                  }}
                >
                  <span className="domainChoiceCard__label">{pal.label}</span>
                  <p className="domainChoiceCard__pct">{confVal}%</p>
                  <p className="domainChoiceCard__hint">Estimated probability for this domain</p>
                  <button
                    type="button"
                    className="button button--primary domainChoiceCard__btn"
                    disabled={choiceBusy}
                    onClick={() => handlePickDomain(label)}
                  >
                    {choiceBusy ? "Sorting…" : `Sort into ${pal.label}`}
                  </button>
                </article>
              );
            })}
          </div>
        </section>
      )}

      <section className="storagePremium cardRise">
        <div className="storagePremium__iconWrap" aria-hidden="true">
          <span className="storagePremium__folderIcon">📂</span>
        </div>
        <div className="storagePremium__body">
          <span className="eyebrow">Local storage</span>
          <h3 className="storagePremium__title">
            {needsChoice ? "File waiting in pending queue" : "Document successfully sorted into"}
          </h3>
          <div className="storagePremium__folderRow">
            <span className="storagePremium__folderName">
              {needsChoice ? "SortedDocuments / _pending" : folderName}
            </span>
            {!needsChoice ? <span className="resultPill storagePremium__domainPill">{theme.label}</span> : null}
          </div>
          <p className="storagePremium__pathLabel">{needsChoice ? "Status" : "Full path"}</p>
          <code className="storagePremium__path">
            {needsChoice
              ? "Choose a domain above to move the file out of the pending folder into the correct category on disk."
              : result.storedIn || "No path returned by the backend."}
          </code>
        </div>
      </section>

      <section className="keywordPanel cardRise">
        <div className="keywordPanel__head">
          <span className="eyebrow">Lexical signals</span>
          <h3>Keywords detected</h3>
          <p className="keywordPanel__sub">
            Terms with the strongest TF‑IDF contribution for this document (live from the vectorizer when available).
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

      <section className="reasoningPanel cardRise">
        <span className="eyebrow">Interpretability</span>
        <h3>Reasoning</h3>
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
            <h4>Naive Bayes decision</h4>
            <p>{result.nbNote}</p>
          </div>
        </div>
      </section>

      <section className="aiSummaryPanel cardRise">
        <span className="eyebrow">Executive summary</span>
        <h3>Document analysis summary</h3>
        <p>{result.aiSummary}</p>
      </section>

      <section className="miningPanel cardRise">
        <div className="miningPanel__head">
          <span className="eyebrow">Pipeline</span>
          <h3>Data mining operations</h3>
          <p>End-to-end preprocessing and modeling steps applied to your upload.</p>
        </div>
        <div className="miningGrid">
          {miningOps.map((op) => (
            <article key={op.key} className="miningCard cardRise">
              <span className="miningCard__icon" aria-hidden="true">
                {op.icon}
              </span>
              <h4>{op.title}</h4>
              <p>{op.text}</p>
            </article>
          ))}
        </div>
      </section>

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
