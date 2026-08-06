"""
Reusable text preprocessing for TF-IDF + Naive Bayes.

Why: normalization reduces spurious features, removes noise (HTML artifacts, extra
whitespace), and aligns train/inference so the vectorizer sees the same token space.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_DIGITS_RE = re.compile(r"\b\d+\b")


def _ensure_nltk() -> None:
    packages = (
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    )
    for finder, pkg in packages:
        try:
            nltk.data.find(finder)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


@lru_cache(maxsize=1)
def _stopwords() -> set[str]:
    _ensure_nltk()
    try:
        return set(stopwords.words("english"))
    except Exception:
        return set()


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    _ensure_nltk()
    return WordNetLemmatizer()


def fix_encoding_and_unicode(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    return text


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    remove_digits: bool = False,
    lemmatize: bool = False,
) -> str:
    text = fix_encoding_and_unicode(text)
    if lowercase:
        text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    if remove_digits:
        text = _DIGITS_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    tokens = text.split()
    if remove_stopwords:
        sw = _stopwords()
        tokens = [t for t in tokens if t not in sw and len(t) > 1]
    if lemmatize and tokens:
        lem = _lemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_pipeline(
    text: str,
    *,
    min_chars: int = 40,
    lemmatize: bool = False,
) -> str | None:
    """
    Full pipeline. Returns None if document is too short after cleaning.
    """
    cleaned = clean_text(text, lemmatize=lemmatize)
    if len(cleaned) < min_chars:
        return None
    return cleaned


def preprocess_for_keywords(text: str) -> str:
    """Light normalization for substring keyword matching (keep phrase integrity)."""
    t = fix_encoding_and_unicode(text).lower()
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t
