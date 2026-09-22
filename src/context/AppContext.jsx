import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

const AppContext = createContext(null);

const backendBaseUrl = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8001";
const historyStorageKey = "crossDomainSorter.history";
const predictionStorageKey = "crossDomainSorter.prediction";

const supportedExtensions = ["pdf", "docx", "txt"];
const supportedMimeTypes = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/plain",
];

const domainPalette = {
  "Business and Finance": {
    label: "Business and Finance",
    color: "#10b981",
    soft: "rgba(16, 185, 129, 0.16)",
    border: "rgba(16, 185, 129, 0.32)",
  },
  "Medical Health": {
    label: "Medical Health",
    color: "#0ea5e9",
    soft: "rgba(14, 165, 233, 0.16)",
    border: "rgba(14, 165, 233, 0.32)",
  },
  Sports: {
    label: "Sports",
    color: "#f59e0b",
    soft: "rgba(245, 158, 11, 0.16)",
    border: "rgba(245, 158, 11, 0.32)",
  },
  "Technology & Computing": {
    label: "Technology & Computing",
    color: "#8b5cf6",
    soft: "rgba(139, 92, 246, 0.16)",
    border: "rgba(139, 92, 246, 0.32)",
  },
  Science: {
    label: "Science",
    color: "#f43f5e",
    soft: "rgba(244, 63, 94, 0.16)",
    border: "rgba(244, 63, 94, 0.32)",
  },
  Entertainment: {
    label: "Entertainment",
    color: "#d946ef",
    soft: "rgba(217, 70, 239, 0.16)",
    border: "rgba(217, 70, 239, 0.32)",
  },
  Unknown: {
    label: "Unknown",
    color: "#64748b",
    soft: "rgba(100, 116, 139, 0.12)",
    border: "rgba(100, 116, 139, 0.22)",
  },
  // Backward-compatible aliases
  Finance: {
    label: "Business and Finance",
    color: "#10b981",
    soft: "rgba(16, 185, 129, 0.16)",
    border: "rgba(16, 185, 129, 0.32)",
  },
  Medical: {
    label: "Medical Health",
    color: "#0ea5e9",
    soft: "rgba(14, 165, 233, 0.16)",
    border: "rgba(14, 165, 233, 0.32)",
  },
  Technology: {
    label: "Technology & Computing",
    color: "#8b5cf6",
    soft: "rgba(139, 92, 246, 0.16)",
    border: "rgba(139, 92, 246, 0.32)",
  },
  Education: {
    label: "Science",
    color: "#f43f5e",
    soft: "rgba(244, 63, 94, 0.16)",
    border: "rgba(244, 63, 94, 0.32)",
  },
};

const fallbackKeywordsByDomain = {
  "Business and Finance": ["Revenue", "Investment", "Market", "Equity", "Portfolio", "Earnings", "Capital", "Shares"],
  "Medical Health": ["Diagnosis", "Patient", "Symptoms", "Treatment", "Clinical", "Healthcare", "Therapy", "Hospital"],
  Sports: ["Team", "League", "Match", "Tournament", "Player", "Score", "Season", "Coach"],
  "Technology & Computing": ["Software", "API", "Cloud", "Processor", "Programming", "Data", "Security", "Hardware"],
  Science: ["Research", "Experiment", "Species", "Galaxy", "Physics", "Climate", "Laboratory", "Discovery"],
  Entertainment: ["Film", "Music", "Actor", "Album", "Concert", "Series", "Box Office", "Festival"],
  Unknown: ["Document", "Text", "Classification", "Features", "Model", "Analysis"],
};

const detailedReasoningByDomain = {
  "Business and Finance":
    "The classifier detected salient financial signals including corporate revenue, capital markets, investment securities, and economic indicators. In the 100,000-dimensional TF-IDF vector space, positive linear coefficients mapped these terms to the Business and Finance hyperplane with decisive separation.",
  "Medical Health":
    "The document exhibited strong healthcare terminology including clinical diagnoses, treatments, patient care protocols, and pharmaceutical concepts. LinearSVC weights for Medical Health produced a strong positive decision score over competing categories.",
  Sports:
    "Athletic competition signals—teams, match scores, leagues, players, and tournament narratives—dominated the document's TF-IDF profile, placing the feature vector well into the positive decision region for Sports.",
  "Technology & Computing":
    "Computing, software engineering, cloud architecture, hardware processors, and digital infrastructure tokens carried the largest positive feature attribution under the LinearSVC model weights for Technology & Computing.",
  Science:
    "Empirical methodology, scientific research terminology, astronomy, biodiversity, or physical sciences cues concentrated strongly in the extracted text, yielding the highest decision score for Science.",
  Entertainment:
    "Culture, cinematic productions, music releases, festival coverage, and celebrity media themes dominated the lexical representation, distinguishing this document cleanly under the Entertainment classifier.",
  Unknown:
    "The model's decision scores were near the decision boundary or ambiguous across multiple categories. Manual review is recommended before sorting.",
};

const patternNotesByDomain = {
  "Business and Finance": "Corporate finance, balance sheet metrics, and market commentary formed the primary token distribution.",
  "Medical Health": "Care-delivery lexicon and clinical terminology dominated the token distribution relative to non-medical corpora.",
  Sports: "Match statistics, team rosters, and competition narratives clustered strongly in the extracted text.",
  "Technology & Computing": "Technical infrastructure, programming terminology, and software workflows anchored the vector.",
  Science: "Hypothesis, research observations, and natural sciences vocabulary defined the feature pattern.",
  Entertainment: "Performance arts, media distribution, and entertainment industry terms formed coherent lexical clusters.",
  Unknown: "Lexical cues did not converge decisively on a single domain hyperplane.",
};

const tfidfNotesByDomain = {
  "Business and Finance": "High-information financial n-grams received substantial sublinear TF-IDF weighting.",
  "Medical Health": "Domain-specific medical tokens received strong TF-IDF weights that differentiate clinical text.",
  Sports: "Tournament and sport-specific n-grams dominated the weighted feature vector.",
  "Technology & Computing": "Software and system engineering n-grams carried the largest feature mass in the 100k vocabulary.",
  Science: "Specialized scientific stems and multi-word n-grams produced elevated TF-IDF weights.",
  Entertainment: "Creative industry phrases and media titles received high TF-IDF weighting.",
  Unknown: "TF-IDF weights were evenly dispersed across multiple competing domain vocabularies.",
};

const svcNotesByDomain = {
  "Business and Finance": "LinearSVC (C=0.5) hyperplane distance yielded a dominant positive decision score for Business and Finance.",
  "Medical Health": "LinearSVC weights produced the maximum decision score for the Medical Health hyperplane.",
  Sports: "LinearSVC hyperplane projection strongly separated the document into the Sports decision region.",
  "Technology & Computing": "The Technology & Computing hyperplane evaluation yielded the highest decision score.",
  Science: "LinearSVC decision function evaluated Science as the top-ranking category.",
  Entertainment: "The Entertainment decision hyperplane separated this document with positive margin.",
  Unknown: "LinearSVC decision margins between the top two classes fell below the high-confidence threshold.",
};

const aiSummaryByDomain = {
  "Business and Finance":
    "The uploaded document contains prominent financial and economic subject matter. After TF-IDF extraction across 100,000 features, the finalized LinearSVC model classified it into Business and Finance with clear decision margin.",
  "Medical Health":
    "The document features clinical, healthcare, or biomedical subject matter. Research LinearSVC inference classified it into Medical Health and routed it to local storage.",
  Sports:
    "Sports journalism or athletic event coverage was detected throughout the text. The classifier routed the document to the Sports archive.",
  "Technology & Computing":
    "The text discusses computing, hardware, software, or digital systems. LinearSVC identified Technology & Computing as the top category.",
  Science:
    "Scientific research, natural phenomena, or academic investigation vocabulary was recognized. The document was categorized as Science.",
  Entertainment:
    "Arts, media, or entertainment narratives were recognized in the document text. The file was organized into Entertainment.",
  Unknown:
    "The document decision margin is close to the decision boundary. Please review the suggested categories.",
};

function normalizePrediction(prediction) {
  if (!prediction) {
    return "Unknown";
  }

  const value = String(prediction).trim();
  const lower = value.toLowerCase();

  if (lower.includes("bus") || lower.includes("fin")) return "Business and Finance";
  if (lower.includes("med") || lower.includes("health")) return "Medical Health";
  if (lower.includes("sport")) return "Sports";
  if (lower.includes("tech") || lower.includes("comput")) return "Technology & Computing";
  if (lower.includes("sci")) return "Science";
  if (lower.includes("entert")) return "Entertainment";

  return value;
}

function hydratePredictionFromStorage(stored) {
  if (stored == null) {
    return null;
  }
  if (Array.isArray(stored)) {
    return stored.map((item) => enrichStoredPrediction(item)).filter(Boolean);
  }
  return enrichStoredPrediction(stored);
}

function mapApiResultToFrontend(payload, file) {
  const normalizedPrediction = normalizePrediction(
    typeof payload === "object" ? payload?.prediction || payload?.predicted_class : payload
  );

  const rawKeywords =
    typeof payload === "object" && Array.isArray(payload?.top_keywords)
      ? payload.top_keywords.filter(Boolean).map(String)
      : [];
  const fallbackKeywords =
    fallbackKeywordsByDomain[normalizedPrediction] || fallbackKeywordsByDomain.Unknown;

  const decisionMargin =
    typeof payload === "object" && payload?.decision_margin != null
      ? Number(payload.decision_margin)
      : null;

  const uncertaintyLevel =
    typeof payload === "object" && payload?.uncertainty_level
      ? String(payload.uncertainty_level)
      : decisionMargin != null
      ? decisionMargin >= 1.0
        ? "High confidence"
        : decisionMargin >= 0.5
        ? "Moderate confidence"
        : "Ambiguous"
      : "Moderate confidence";

  const decisionScores =
    typeof payload === "object" && payload?.decision_scores
      ? payload.decision_scores
      : typeof payload === "object" && payload?.ranking
      ? payload.ranking
      : {};

  const requiresManualChoice = Boolean(
    typeof payload === "object" && payload?.requires_manual_choice
  );
  const pendingId =
    typeof payload === "object" && payload?.pending_id ? String(payload.pending_id) : null;
  const topTwoDomains =
    typeof payload === "object" && Array.isArray(payload?.top_two_domains)
      ? payload.top_two_domains
      : [];

  const textLength =
    typeof payload === "object" && payload?.text_length != null
      ? Number(payload.text_length)
      : 0;

  const isBlank = Boolean(typeof payload === "object" && payload?.is_blank);
  const errorMessage = typeof payload === "object" && payload?.error ? String(payload.error) : "";

  const modelComparison =
    typeof payload === "object" && Array.isArray(payload?.model_comparison)
      ? payload.model_comparison
      : [];
  const modelPerformance =
    typeof payload === "object" && payload?.model_performance
      ? payload.model_performance
      : {};
  const ensembleExperiments =
    typeof payload === "object" && payload?.ensemble_experiments
      ? payload.ensemble_experiments
      : {};
  const selectedModel =
    (typeof payload === "object" && payload?.selected_model) || "LinearSVC";
  const selectionReason =
    (typeof payload === "object" && payload?.selection_reason) ||
    "LinearSVC achieved the highest test Macro F1 (94.49%) among the evaluated individual classifiers.";

  const result = {
    filename: (typeof payload === "object" && payload?.filename) || file?.name || "document",
    error: errorMessage,
    isBlank,
    prediction: isBlank ? null : normalizedPrediction,
    predicted_class: isBlank ? null : normalizedPrediction,
    finalPrediction: isBlank ? null : normalizedPrediction,
    selectedModel,
    selectionReason,
    modelComparison: isBlank ? [] : modelComparison,
    modelPerformance,
    ensembleExperiments,
    decisionMargin,
    uncertaintyLevel: isBlank ? "Blank document" : uncertaintyLevel,
    decisionScores: isBlank ? {} : decisionScores,
    textLength,
    storedIn:
      typeof payload === "object" && payload?.stored_in != null ? String(payload.stored_in) : "",
    fileSize: file ? formatFileSize(file.size) : "",
    fileType: file ? getFileTypeLabel(file) : "Unknown",
    timestamp: Date.now(),
    confidence: isBlank ? null : decisionMargin,
    keywords: isBlank ? [] : rawKeywords.length ? rawKeywords : fallbackKeywords,
    fallbackKeywords: isBlank ? [] : fallbackKeywords,
    requiresManualChoice: isBlank ? false : requiresManualChoice,
    pendingId: isBlank ? null : pendingId,
    topTwoDomains: isBlank ? [] : topTwoDomains,
    detailedReasoning: isBlank
      ? ""
      : detailedReasoningByDomain[normalizedPrediction] || detailedReasoningByDomain.Unknown,
    patternNote: isBlank
      ? ""
      : patternNotesByDomain[normalizedPrediction] || patternNotesByDomain.Unknown,
    tfidfNote: isBlank
      ? ""
      : tfidfNotesByDomain[normalizedPrediction] || tfidfNotesByDomain.Unknown,
    nbNote: isBlank
      ? ""
      : svcNotesByDomain[normalizedPrediction] || svcNotesByDomain.Unknown,
    aiSummary: isBlank
      ? ""
      : aiSummaryByDomain[normalizedPrediction] || aiSummaryByDomain.Unknown,
  };

  return enrichStoredPrediction(result);
}

function enrichStoredPrediction(raw) {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  if (raw.isBlank || raw.error) {
    return {
      ...raw,
      prediction: null,
      predicted_class: null,
      finalPrediction: null,
      error: raw.error || "This document appears to be blank or contains no readable text. Please upload a document with readable content.",
      isBlank: true,
      decisionMargin: null,
      confidence: null,
      uncertaintyLevel: "Blank document",
      decisionScores: {},
      keywords: [],
      fallbackKeywords: [],
      topTwoDomains: [],
      modelComparison: [],
      requiresManualChoice: false,
      storedIn: null,
    };
  }

  const pred = normalizePrediction(raw.prediction || raw.predicted_class);
  const fallbackKeywords = fallbackKeywordsByDomain[pred] || fallbackKeywordsByDomain.Unknown;
  const rawKw = Array.isArray(raw.keywords) ? raw.keywords.filter(Boolean).map(String) : [];
  const keywords = rawKw.length ? rawKw : fallbackKeywords;

  const decisionMargin =
    raw.decisionMargin != null && !Number.isNaN(Number(raw.decisionMargin))
      ? Number(raw.decisionMargin)
      : raw.confidence != null && !Number.isNaN(Number(raw.confidence)) && Number(raw.confidence) < 20
      ? Number(raw.confidence)
      : null;

  const uncertaintyLevel =
    raw.uncertaintyLevel ||
    (decisionMargin != null
      ? decisionMargin >= 1.0
        ? "High confidence"
        : decisionMargin >= 0.5
        ? "Moderate confidence"
        : "Ambiguous"
      : "Moderate confidence");

  return {
    ...raw,
    prediction: pred,
    predicted_class: pred,
    finalPrediction: raw.finalPrediction || pred,
    selectedModel: raw.selectedModel || "LinearSVC",
    selectionReason:
      raw.selectionReason ||
      "LinearSVC achieved the highest test Macro F1 (94.49%) among the evaluated individual classifiers.",
    modelComparison: Array.isArray(raw.modelComparison) ? raw.modelComparison : [],
    modelPerformance: raw.modelPerformance || {},
    ensembleExperiments: raw.ensembleExperiments || {},
    decisionMargin,
    uncertaintyLevel,
    decisionScores: raw.decisionScores || raw.ranking || {},
    textLength: raw.textLength || 0,
    confidence: decisionMargin,
    keywords,
    fallbackKeywords,
    requiresManualChoice: Boolean(raw.requiresManualChoice),
    pendingId: raw.pendingId ?? null,
    topTwoDomains: Array.isArray(raw.topTwoDomains) ? raw.topTwoDomains : [],
    detailedReasoning:
      raw.detailedReasoning || detailedReasoningByDomain[pred] || detailedReasoningByDomain.Unknown,
    patternNote: raw.patternNote || patternNotesByDomain[pred] || patternNotesByDomain.Unknown,
    tfidfNote: raw.tfidfNote || tfidfNotesByDomain[pred] || tfidfNotesByDomain.Unknown,
    nbNote: raw.nbNote || svcNotesByDomain[pred] || svcNotesByDomain.Unknown,
    aiSummary: raw.aiSummary || aiSummaryByDomain[pred] || aiSummaryByDomain.Unknown,
  };
}

function readStoredState(storage, key, fallbackValue) {
  if (typeof window === "undefined") {
    return fallbackValue;
  }

  try {
    const storedValue = storage.getItem(key);
    return storedValue ? JSON.parse(storedValue) : fallbackValue;
  } catch {
    return fallbackValue;
  }
}

function writeStoredState(storage, key, value) {
  if (typeof window === "undefined") {
    return;
  }

  try {
    if (value === null || value === undefined || (Array.isArray(value) && value.length === 0)) {
      storage.removeItem(key);
      return;
    }

    storage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore storage errors so the main workflow remains usable.
  }
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const size = bytes / 1024 ** index;
  return `${size.toFixed(size >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function getFileTypeLabel(file) {
  if (!file) {
    return "Unknown";
  }

  const fileName = file.name || "";
  const extension = fileName.includes(".") ? fileName.split(".").pop().toLowerCase() : "";

  if (extension === "pdf") return "PDF";
  if (extension === "docx") return "DOCX";
  if (extension === "txt") return "TXT";

  return file.type || "Unknown";
}

function validateFile(file) {
  if (!file) {
    return "Please upload a PDF, DOCX, or TXT file before continuing.";
  }

  const fileName = file.name || "";
  const extension = fileName.includes(".") ? fileName.split(".").pop().toLowerCase() : "";
  const mimeAllowed = file.type ? supportedMimeTypes.includes(file.type) : false;
  const extensionAllowed = extension ? supportedExtensions.includes(extension) : false;

  if (!mimeAllowed && !extensionAllowed) {
    return "Unsupported file type. Please use PDF, DOCX, or TXT only.";
  }

  return "";
}

function AppProvider({ children }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [prediction, setPrediction] = useState(() =>
    hydratePredictionFromStorage(readStoredState(window.sessionStorage, predictionStorageKey, null))
  );
  const [history, setHistory] = useState(() => readStoredState(window.localStorage, historyStorageKey, []));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    writeStoredState(window.sessionStorage, predictionStorageKey, prediction);
  }, [prediction]);

  useEffect(() => {
    writeStoredState(window.localStorage, historyStorageKey, history);
  }, [history]);

  const resetWorkflow = () => {
    setSelectedFiles([]);
    setPrediction(null);
    setError("");
  };

  const confirmManualSort = async (pendingId, chosenDomain) => {
    if (!pendingId || !chosenDomain) {
      throw new Error("Missing pending sort information.");
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${backendBaseUrl}/confirm-sort`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pending_id: pendingId, chosen_domain: chosenDomain }),
      });

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        const message =
          typeof payload === "object" && payload?.error
            ? String(payload.error)
            : "Could not finalize folder sort. Try again.";
        throw new Error(message);
      }

      const normalizedPrediction = normalizePrediction(payload?.prediction || payload?.predicted_class);
      const displayName = payload?.filename ? String(payload.filename) : "";

      const applyConfirm = (prev) => {
        const rawKeywords =
          Array.isArray(prev?.keywords) && prev.keywords.length
            ? prev.keywords
            : fallbackKeywordsByDomain[normalizedPrediction] || fallbackKeywordsByDomain.Unknown;

        return enrichStoredPrediction({
          ...prev,
          filename: displayName || prev?.filename,
          prediction: normalizedPrediction,
          predicted_class: normalizedPrediction,
          decisionMargin: 10.0,
          uncertaintyLevel: "User confirmed",
          confidence: 10.0,
          storedIn: payload?.stored_in != null ? String(payload.stored_in) : "",
          requiresManualChoice: false,
          pendingId: null,
          topTwoDomains: [],
          keywords: rawKeywords,
          detailedReasoning:
            detailedReasoningByDomain[normalizedPrediction] || detailedReasoningByDomain.Unknown,
          patternNote: patternNotesByDomain[normalizedPrediction] || patternNotesByDomain.Unknown,
          tfidfNote: tfidfNotesByDomain[normalizedPrediction] || tfidfNotesByDomain.Unknown,
          nbNote: svcNotesByDomain[normalizedPrediction] || svcNotesByDomain.Unknown,
          aiSummary: aiSummaryByDomain[normalizedPrediction] || aiSummaryByDomain.Unknown,
          timestamp: Date.now(),
        });
      };

      setPrediction((prev) => {
        if (Array.isArray(prev)) {
          return prev.map((item) =>
            item.pendingId === pendingId ? applyConfirm(item) : item
          );
        }
        return applyConfirm(prev);
      });

      setHistory((currentHistory) =>
        [
          {
            filename: displayName,
            prediction: normalizedPrediction,
            timestamp: Date.now(),
          },
          ...currentHistory,
        ].slice(0, 5)
      );

      return { normalizedPrediction, displayName };
    } catch (errorInstance) {
      const message =
        errorInstance?.message?.trim() || `Could not reach the backend at ${backendBaseUrl}.`;
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  };

  const submitDocument = async (filesInput) => {
    const files = Array.isArray(filesInput) ? filesInput : filesInput ? [filesInput] : [];

    if (files.length === 0) {
      const message = "Please upload a PDF, DOCX, or TXT file before continuing.";
      setError(message);
      throw new Error(message);
    }

    for (const file of files) {
      const validationMessage = validateFile(file);
      if (validationMessage) {
        setError(validationMessage);
        throw new Error(validationMessage);
      }
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      for (const file of files) {
        formData.append("files", file);
      }

      const response = await fetch(`${backendBaseUrl}/predict-bulk`, {
        method: "POST",
        body: formData,
      });

      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json")
        ? await response.json()
        : await response.text();

      if (!response.ok) {
        const message =
          typeof payload === "string"
            ? payload
            : payload?.detail || payload?.error || "Backend error. Please try again.";

        throw new Error(message);
      }

      const rows =
        typeof payload === "object" && Array.isArray(payload?.results) ? payload.results : null;

      if (!rows || rows.length === 0) {
        throw new Error("No classification results returned from the server.");
      }

      // If single file upload and it returned an error (such as blank file):
      if (rows.length === 1 && (rows[0]?.error || rows[0]?.is_blank)) {
        const errorText =
          rows[0].error ||
          "This document appears to be blank or contains no readable text. Please upload a document with readable content.";
        setError(errorText);
        throw new Error(errorText);
      }

      // If all files in a batch returned an error:
      if (rows.every((r) => r?.error || r?.is_blank)) {
        const errorText =
          rows[0]?.error ||
          "This document appears to be blank or contains no readable text. Please upload a document with readable content.";
        setError(errorText);
        throw new Error(errorText);
      }

      const enrichedList = rows.map((row, index) => {
        const file =
          files.find((f) => f.name === row?.filename) ||
          files[index] ||
          files[0];
        return mapApiResultToFrontend(row, file);
      });

      const finalPrediction = enrichedList.length === 1 ? enrichedList[0] : enrichedList;
      setPrediction(finalPrediction);

      const historyEntries = enrichedList
        .filter((item) => !item.requiresManualChoice && !item.isBlank && item.prediction)
        .map((item) => ({
          filename: item.filename,
          prediction: item.prediction,
          timestamp: item.timestamp,
        }));

      if (historyEntries.length > 0) {
        setHistory((currentHistory) => [...historyEntries, ...currentHistory].slice(0, 5));
      }

      return finalPrediction;
    } catch (errorInstance) {
      const message =
        errorInstance?.message?.trim() ||
        `Could not connect to the backend. Ensure FastAPI is running at ${backendBaseUrl}.`;
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  };

  const value = useMemo(
    () => ({
      backendBaseUrl,
      selectedFiles,
      setSelectedFiles,
      prediction,
      setPrediction,
      history,
      loading,
      error,
      setError,
      resetWorkflow,
      submitDocument,
      confirmManualSort,
      validateFile,
      formatFileSize,
      domainPalette,
    }),
    [error, history, loading, prediction, selectedFiles]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

function useAppState() {
  const context = useContext(AppContext);

  if (!context) {
    throw new Error("useAppState must be used within AppProvider");
  }

  return context;
}

export { AppProvider, useAppState };
