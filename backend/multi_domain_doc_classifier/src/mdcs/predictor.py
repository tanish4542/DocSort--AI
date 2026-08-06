"""
Inference API: text and file paths → domain, probabilities, keywords.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .config import DOMAINS, MODELS_DIR
from .document_extract import extract_text_auto
from .domain_keywords import DOMAIN_KEYWORDS
from .features import (
    build_tfidf_vectorizer,
    extract_tfidf_keywords_for_doc,
    hstack_tfidf_keywords,
    keyword_match_matrix,
)
from .preprocess import clean_text, preprocess_for_keywords, preprocess_pipeline


def _adjust_proba_with_keyword_evidence(
    prob_by_domain: dict[str, float],
    matched_by_domain: dict[str, list[str]],
) -> dict[str, float]:
    """
    NB+TF-IDF often maps CS/ML text to Medical because tokens like "neural" and
    "learning" are frequent in biomedical training data. When the manual lexicon
    shows many technology phrases and few clinical ones, rebalance probabilities
    before argmax so decisions align with phrase evidence.
    """
    p = {k: float(v) for k, v in prob_by_domain.items()}
    med_n = len(matched_by_domain.get("Medical", []))
    tech_n = len(matched_by_domain.get("Technology", []))

    if med_n <= 2 and tech_n >= 4 and tech_n > med_n:
        p["Medical"] = p.get("Medical", 0.0) * 0.06
        p["Technology"] = p.get("Technology", 0.0) * 5.5 + 1e-8

    s = sum(p.values())
    if s <= 0:
        return prob_by_domain
    return {k: v / s for k, v in p.items()}


@dataclass
class PredictionResult:
    predicted_domain: str
    confidence: float
    probabilities: dict[str, float]
    top_tfidf_terms: list[tuple[str, float]]
    matched_manual_keywords: dict[str, list[str]]
    keyword_domain_scores: dict[str, float]


class DomainClassifier:
    """
    Loads vectorizer + MultinomialNB trained on [TF-IDF | keyword features].
    """

    def __init__(
        self,
        vectorizer,
        model,
        domains: tuple[str, ...] | list[str] | None = None,
        *,
        min_doc_chars: int = 40,
    ):
        self.vectorizer = vectorizer
        self.model = model
        self.domains = list(domains) if domains is not None else list(model.classes_)
        self.min_doc_chars = min_doc_chars

    @classmethod
    def load(cls, models_dir: Path | None = None) -> "DomainClassifier":
        models_dir = models_dir or MODELS_DIR
        with open(models_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(models_dir / "vectorizer.pkl", "rb") as f:
            bundle = pickle.load(f)
        vec = bundle["vectorizer"]
        min_dc = int(bundle.get("min_doc_chars", 40))
        return cls(vec, model, domains=None, min_doc_chars=min_dc)

    def save(self, models_dir: Path | None = None) -> None:
        models_dir = models_dir or MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(models_dir / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        bundle = {
            "vectorizer": self.vectorizer,
            "domains": self.domains,
            "min_doc_chars": self.min_doc_chars,
            "class_order_note": "domains mirror model.classes_ at save time",
        }
        with open(models_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(bundle, f)

    def _transform(self, raw_texts: list[str], cleaned_texts: list[str]) -> sparse.csr_matrix:
        tfidf = self.vectorizer.transform(cleaned_texts)
        kw = keyword_match_matrix(raw_texts, self.domains)
        return hstack_tfidf_keywords(tfidf, kw)

    def predict_proba_dict(self, text: str) -> dict[str, float]:
        proc = preprocess_pipeline(text, min_chars=self.min_doc_chars)
        if proc is None:
            proc = clean_text(text, remove_stopwords=False)
        X = self._transform([text], [proc])
        probs = self.model.predict_proba(X)[0]
        prob_raw = {str(self.domains[i]): float(probs[i]) for i in range(len(self.domains))}
        norm = preprocess_for_keywords(text)
        matched: dict[str, list[str]] = {}
        for d in self.domains:
            hits = [k for k in DOMAIN_KEYWORDS[str(d)] if k.lower() in norm]
            matched[str(d)] = hits[:25]
        prob_adj = _adjust_proba_with_keyword_evidence(prob_raw, matched)
        return prob_adj

    def predict_domain(self, text: str) -> PredictionResult:
        proc = preprocess_pipeline(text, min_chars=self.min_doc_chars)
        if proc is None:
            proc = clean_text(text, remove_stopwords=False)
        X = self._transform([text], [proc])
        probs = self.model.predict_proba(X)[0]
        prob_raw = {str(self.domains[i]): float(probs[i]) for i in range(len(self.domains))}

        tfidf_only = self.vectorizer.transform([proc])
        top_terms = extract_tfidf_keywords_for_doc(self.vectorizer, tfidf_only, top_k=15)

        norm = preprocess_for_keywords(text)
        matched: dict[str, list[str]] = {}
        kw_scores: dict[str, float] = {}
        for d in self.domains:
            hits = [k for k in DOMAIN_KEYWORDS[str(d)] if k.lower() in norm]
            matched[str(d)] = hits[:25]
            kw_scores[str(d)] = float(len(hits))

        prob_adj = _adjust_proba_with_keyword_evidence(prob_raw, matched)
        ordered = [prob_adj[str(d)] for d in self.domains]
        idx = int(np.argmax(ordered))
        pred = str(self.domains[idx])
        conf = float(prob_adj[pred])
        prob_dict = {str(d): float(prob_adj[str(d)]) for d in self.domains}

        s = sum(kw_scores.values()) or 1.0
        kw_scores = {k: round(100.0 * v / s, 2) for k, v in kw_scores.items()}

        return PredictionResult(
            predicted_domain=pred,
            confidence=round(100.0 * conf, 2),
            probabilities={k: round(100.0 * v, 2) for k, v in prob_dict.items()},
            top_tfidf_terms=top_terms,
            matched_manual_keywords=matched,
            keyword_domain_scores=kw_scores,
        )

    def predict_document(self, path: str | Path) -> PredictionResult:
        raw = extract_text_auto(Path(path))
        return self.predict_domain(raw)


def extract_keywords(text: str, top_k: int = 20) -> dict[str, Any]:
    """Standalone helper using a loaded classifier if present; else TF-IDF-only via crude idf."""
    clf = DomainClassifier.load()
    res = clf.predict_domain(text)
    return {
        "tfidf_top_terms": res.top_tfidf_terms[:top_k],
        "manual_matches": {k: v[:top_k] for k, v in res.matched_manual_keywords.items() if v},
    }


def rank_domains(text: str) -> dict[str, Any]:
    clf = DomainClassifier.load()
    res = clf.predict_domain(text)
    model_rank = sorted(res.probabilities.items(), key=lambda x: -x[1])
    kw_rank = sorted(res.keyword_domain_scores.items(), key=lambda x: -x[1])
    return {
        "model_probabilities_percent": model_rank,
        "keyword_presence_percent": kw_rank,
        "predicted": res.predicted_domain,
    }


def predict_domain(text: str) -> dict[str, Any]:
    clf = DomainClassifier.load()
    r = clf.predict_domain(text)
    return {
        "predicted_domain": r.predicted_domain,
        "confidence_percent": r.confidence,
        "probabilities_percent": r.probabilities,
        "top_tfidf_terms": r.top_tfidf_terms,
        "matched_manual_keywords": r.matched_manual_keywords,
        "keyword_domain_scores_percent": r.keyword_domain_scores,
    }


def predict_document(file_path: str | Path) -> dict[str, Any]:
    clf = DomainClassifier.load()
    r = clf.predict_document(file_path)
    return {
        "predicted_domain": r.predicted_domain,
        "confidence_percent": r.confidence,
        "probabilities_percent": r.probabilities,
        "top_tfidf_terms": r.top_tfidf_terms,
        "matched_manual_keywords": r.matched_manual_keywords,
        "keyword_domain_scores_percent": r.keyword_domain_scores,
    }
