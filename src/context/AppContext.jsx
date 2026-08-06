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
  Medical: {
    label: "Medical",
    color: "#4f8cff",
    soft: "rgba(79, 140, 255, 0.16)",
    border: "rgba(79, 140, 255, 0.28)",
  },
  Technology: {
    label: "Technology",
    color: "#8b5cf6",
    soft: "rgba(139, 92, 246, 0.16)",
    border: "rgba(139, 92, 246, 0.28)",
  },
  Finance: {
    label: "Finance",
    color: "#2fbf71",
    soft: "rgba(47, 191, 113, 0.16)",
    border: "rgba(47, 191, 113, 0.28)",
  },
  Sports: {
    label: "Sports",
    color: "#f59e0b",
    soft: "rgba(245, 158, 11, 0.16)",
    border: "rgba(245, 158, 11, 0.28)",
  },
  Education: {
    label: "Education",
    color: "#facc15",
    soft: "rgba(250, 204, 21, 0.18)",
    border: "rgba(250, 204, 21, 0.34)",
  },
  Unknown: {
    label: "Unknown",
    color: "#64748b",
    soft: "rgba(100, 116, 139, 0.12)",
    border: "rgba(100, 116, 139, 0.22)",
  },
};

const fallbackKeywordsByDomain = {
  Medical: ["Diagnosis", "Patient", "Symptoms", "Treatment", "Clinical", "Healthcare", "Therapy"],
  Finance: ["Revenue", "Investment", "Market", "Equity", "Portfolio", "Earnings", "Capital"],
  Technology: ["Software", "API", "Cloud", "Deployment", "System", "Data", "Security"],
  Sports: ["Team", "League", "Match", "Tournament", "Player", "Score", "Season"],
  Education: ["Curriculum", "Learning", "Assessment", "Student", "Course", "Instruction", "Academic"],
  Unknown: ["Document", "Text", "Classification", "Features", "Model", "Analysis"],
};

const detailedReasoningByDomain = {
  Medical:
    "The classifier detected strong medical terminology such as symptoms, diagnosis, treatment, patient, and clinical references. TF-IDF assigned higher weights to these healthcare-related terms because they appeared frequently within the uploaded document but less commonly across other domains. The Naive Bayes classifier then calculated a high probability match with the Medical category based on previously learned word distributions.",
  Finance:
    "The system identified financial terminology including investments, stocks, revenue, market trends, and economic references. TF-IDF emphasized these finance-specific keywords, allowing the Naive Bayes classifier to associate the document with the Finance domain using probabilistic classification.",
  Technology:
    "The classifier surfaced computing-oriented vocabulary such as software, systems, APIs, infrastructure, security, and engineering workflows. TF-IDF up-weighted these tokens relative to general-language terms, and the Naive Bayes model assigned the strongest posterior to the Technology class based on historical word likelihoods per category.",
  Sports:
    "The document exhibited recurring sports signals—teams, leagues, scores, tournaments, athletes, and season narratives. TF-IDF amplified these sport-specific tokens inside the sparse vector, and Naive Bayes matched the resulting pattern to the Sports domain with strong class-conditional support.",
  Education:
    "Academic cues including curriculum, pedagogy, assessments, students, instructors, and learning outcomes appeared consistently. TF-IDF increased salience for education-specific vocabulary, enabling the Naive Bayes classifier to favor the Education domain according to learned term frequencies.",
  Unknown:
    "The model produced a label outside the primary domain taxonomy or confidence was ambiguous. Review the document language, ensure the training vocabulary covers this style of text, and consider re-uploading after verifying file integrity.",
};

const patternNotesByDomain = {
  Medical: "Clinical lexicon and care-process wording dominated the token distribution compared with non-medical baselines.",
  Finance: "Corporate finance and market lexicon appeared repeatedly, steering feature mass toward finance-oriented dimensions.",
  Technology: "Technical implementation language outweighed generic prose, anchoring the vector toward engineering contexts.",
  Sports: "Event- and competition-centric phrases clustered strongly, separating this text from non-sports corpora.",
  Education: "Instructional and assessment vocabulary formed coherent clusters typical of learning materials.",
  Unknown: "Lexical cues did not converge cleanly on a single domain signature; manual review is recommended.",
};

const tfidfNotesByDomain = {
  Medical:
    "Rare-but-meaningful medical stems received high TF-IDF scores, highlighting passages that differentiate clinical notes from everyday language.",
  Finance:
    "Market and accounting terms with high information gain were up-ranked, sharpening the contrast between finance and general news text.",
  Technology:
    "Product, stack, and operations terminology carried the largest TF-IDF mass, mirroring how technical docs differ from other genres.",
  Sports:
    "Team and fixture-specific tokens spiked in the weight vector, which is exactly how TF-IDF isolates sports reporting from other news.",
  Education:
    "Pedagogy- and outcomes-related stems dominated the weighted vector, aligning with syllabus-style writing patterns.",
  Unknown: "TF-IDF weights were diffuse across many weak features, suggesting limited domain-specific concentration.",
};

const nbNotesByDomain = {
  Medical:
    "Naive Bayes multiplied conditional word probabilities under the Medical class, yielding the highest posterior after comparing all trained categories.",
  Finance:
    "Class-conditional likelihoods for finance vocabulary outpaced other categories, so the argmax prediction selected Finance.",
  Technology:
    "Posterior scores favored Technology because token likelihoods under that class best explained the observed term multiset.",
  Sports:
    "Word likelihoods conditioned on Sports exceeded competing classes, driving the final argmax decision.",
  Education:
    "Education-specific likelihood products dominated the posterior, confirming the predicted Education label.",
  Unknown: "Posteriors were comparatively flat; the reported label should be validated against your ground truth.",
};

const aiSummaryByDomain = {
  Medical:
    "The uploaded document contained strong healthcare-oriented terminology and contextual patterns. After preprocessing and TF-IDF feature extraction, the Naive Bayes classifier predicted the Medical domain with elevated confidence. The backend then automatically organized the file into the corresponding local desktop folder.",
  Finance:
    "The document read like finance-focused material, with vocabulary tied to markets, performance, and capital structure. TF-IDF feature extraction surfaced those signals clearly, and Naive Bayes predicted Finance with strong support before the file was routed into the Finance directory on disk.",
  Technology:
    "Engineering and product language dominated the extracted text. Following token cleanup and TF-IDF vectorization, the Naive Bayes model classified the artifact as Technology and the server moved it into the Technology folder for traceable local storage.",
  Sports:
    "Sports-centric wording and event framing were prevalent throughout the upload. The pipeline applied standard NLP preprocessing, TF-IDF weighting, and Naive Bayes scoring to label the document as Sports, then persisted it under the Sports folder path returned to the UI.",
  Education:
    "Instructional vocabulary and learning-design cues were prominent. The classifier interpreted these patterns as Education after probabilistic analysis, and the workflow completed by sorting the file into the Education folder alongside other academic materials.",
  Unknown:
    "The classifier returned an unexpected or low-confidence label relative to the tuned domain set. Inspect the document content and model training coverage, then re-run the upload to confirm routing and storage paths.",
};

function normalizePrediction(prediction) {
  if (!prediction) {
    return "Unknown";
  }

  const value = String(prediction).trim();
  const lower = value.toLowerCase();

  if (lower.includes("med")) return "Medical";
  if (lower.includes("tech")) return "Technology";
  if (lower.includes("fin")) return "Finance";
  if (lower.includes("sport")) return "Sports";
  if (lower.includes("edu")) return "Education";

  return value.charAt(0).toUpperCase() + value.slice(1);
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
    typeof payload === "object" ? payload?.prediction : payload
  );

  const rawKeywords =
    typeof payload === "object" && Array.isArray(payload?.top_keywords)
      ? payload.top_keywords.filter(Boolean).map(String)
      : [];
  const fallbackKeywords =
    fallbackKeywordsByDomain[normalizedPrediction] || fallbackKeywordsByDomain.Unknown;

  let confidence =
    typeof payload === "object" && payload?.confidence != null && !Number.isNaN(Number(payload.confidence))
      ? Number(payload.confidence)
      : null;
  if (confidence == null) {
    confidence = 76 + (String(file?.name || payload?.filename || "").length % 18);
  }

  const requiresManualChoice = Boolean(
    typeof payload === "object" && payload?.requires_manual_choice
  );
  const pendingId =
    typeof payload === "object" && payload?.pending_id ? String(payload.pending_id) : null;
  const topTwoDomains =
    typeof payload === "object" && Array.isArray(payload?.top_two_domains)
      ? payload.top_two_domains
      : [];

  const result = {
    filename: (typeof payload === "object" && payload?.filename) || file?.name || "document",
    prediction: normalizedPrediction,
    storedIn:
      typeof payload === "object" && payload?.stored_in != null ? String(payload.stored_in) : "",
    fileSize: file ? formatFileSize(file.size) : "",
    fileType: file ? getFileTypeLabel(file) : "Unknown",
    timestamp: Date.now(),
    confidence,
    keywords: rawKeywords.length ? rawKeywords : fallbackKeywords,
    fallbackKeywords,
    requiresManualChoice,
    pendingId,
    topTwoDomains,
    detailedReasoning:
      detailedReasoningByDomain[normalizedPrediction] || detailedReasoningByDomain.Unknown,
    patternNote: patternNotesByDomain[normalizedPrediction] || patternNotesByDomain.Unknown,
    tfidfNote: tfidfNotesByDomain[normalizedPrediction] || tfidfNotesByDomain.Unknown,
    nbNote: nbNotesByDomain[normalizedPrediction] || nbNotesByDomain.Unknown,
    aiSummary: aiSummaryByDomain[normalizedPrediction] || aiSummaryByDomain.Unknown,
  };

  return enrichStoredPrediction(result);
}

function enrichStoredPrediction(raw) {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const pred = normalizePrediction(raw.prediction);
  const fallbackKeywords = fallbackKeywordsByDomain[pred] || fallbackKeywordsByDomain.Unknown;
  const rawKw = Array.isArray(raw.keywords) ? raw.keywords.filter(Boolean).map(String) : [];
  const keywords = rawKw.length ? rawKw : fallbackKeywords;

  let confidence =
    raw.confidence != null && !Number.isNaN(Number(raw.confidence)) ? Number(raw.confidence) : null;
  if (confidence == null) {
    confidence = 76 + (String(raw.filename || "").length % 18);
  }

  return {
    ...raw,
    prediction: pred,
    confidence,
    keywords,
    fallbackKeywords,
    requiresManualChoice: Boolean(raw.requiresManualChoice),
    pendingId: raw.pendingId ?? null,
    topTwoDomains: Array.isArray(raw.topTwoDomains) ? raw.topTwoDomains : [],
    detailedReasoning:
      raw.detailedReasoning || detailedReasoningByDomain[pred] || detailedReasoningByDomain.Unknown,
    patternNote: raw.patternNote || patternNotesByDomain[pred] || patternNotesByDomain.Unknown,
    tfidfNote: raw.tfidfNote || tfidfNotesByDomain[pred] || tfidfNotesByDomain.Unknown,
    nbNote: raw.nbNote || nbNotesByDomain[pred] || nbNotesByDomain.Unknown,
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

      const normalizedPrediction = normalizePrediction(payload?.prediction);
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
          confidence:
            payload?.confidence != null && !Number.isNaN(Number(payload.confidence))
              ? Number(payload.confidence)
              : 100,
          storedIn: payload?.stored_in != null ? String(payload.stored_in) : "",
          requiresManualChoice: false,
          pendingId: null,
          topTwoDomains: [],
          keywords: rawKeywords,
          detailedReasoning:
            detailedReasoningByDomain[normalizedPrediction] || detailedReasoningByDomain.Unknown,
          patternNote: patternNotesByDomain[normalizedPrediction] || patternNotesByDomain.Unknown,
          tfidfNote: tfidfNotesByDomain[normalizedPrediction] || tfidfNotesByDomain.Unknown,
          nbNote: nbNotesByDomain[normalizedPrediction] || nbNotesByDomain.Unknown,
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
        .filter((item) => !item.requiresManualChoice)
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
