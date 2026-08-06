"""
TF-IDF + manual keyword features (sparse hstack).

Why TF-IDF: strong baseline for bag-of-words text; captures discriminative n-grams.
Why keyword side-features: recovers signal when training data omits phrases the user
cares about (e.g. 'venture capital'); complements learned weights for MNB.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .domain_keywords import DOMAIN_KEYWORDS
from .preprocess import preprocess_for_keywords

# Mild boost for domains with small raw corpora so manual priors pull harder vs Medical.
KEYWORD_DOMAIN_WEIGHTS: dict[str, float] = {
    "Finance": 1.0,
    "Medical": 1.0,
    "Sports": 1.25,
    "Technology": 1.6,
}

# Scale manual-keyword block so four dense dimensions can compete with sparse TF-IDF mass.
KEYWORD_BLOCK_SCALE = 12.0


def build_tfidf_vectorizer(
    ngram_max: int = 2,
    max_features: int = 60_000,
    min_df: int = 2,
    max_df: float = 0.9,
) -> TfidfVectorizer:
    """Unigrams + bigrams (+ trigrams if ngram_max>=3)."""
    return TfidfVectorizer(
        ngram_range=(1, ngram_max),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def keyword_match_matrix(texts: Iterable[str], domains: list[str]) -> np.ndarray:
    """
    Rows = documents, cols = domains. Values are normalized hit scores in [0,1].
    Longer phrases matched in full text (preprocess_for_keywords) count more.
    """
    rows = []
    kw_by_domain = [DOMAIN_KEYWORDS[d] for d in domains]
    weights = [KEYWORD_DOMAIN_WEIGHTS.get(d, 1.0) for d in domains]
    for raw in texts:
        norm = preprocess_for_keywords(raw)
        scores = []
        for kws, w in zip(kw_by_domain, weights):
            hits = 0.0
            for phrase in kws:
                p = phrase.strip().lower()
                if not p:
                    continue
                if p in norm:
                    hits += w * (1.0 + 0.15 * (len(p.split()) - 1))
            scores.append(hits)
        v = np.array(scores, dtype=np.float64)
        if v.sum() > 0:
            v = v / (np.sqrt(np.sum(v * v)) + 1e-9)
        rows.append(v)
    return np.vstack(rows)


def hstack_tfidf_keywords(tfidf_X: sparse.csr_matrix, kw_X: np.ndarray) -> sparse.csr_matrix:
    kw_scaled = (kw_X.astype(np.float64)) * KEYWORD_BLOCK_SCALE
    kw_sparse = sparse.csr_matrix(kw_scaled)
    return sparse.hstack([tfidf_X, kw_sparse], format="csr")


def extract_tfidf_keywords_for_doc(
    vectorizer: TfidfVectorizer,
    doc_vector: sparse.csr_matrix,
    top_k: int = 15,
) -> list[tuple[str, float]]:
    """Top weighted n-grams for one TF-IDF row."""
    arr = doc_vector.toarray().ravel()
    idx = np.argsort(arr)[::-1][:top_k]
    feats = np.array(vectorizer.get_feature_names_out())
    out = [(feats[i], float(arr[i])) for i in idx if arr[i] > 0]
    return out
