"""
Streamlit UI for text/file classification.
Run from project root: streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from mdcs.predictor import DomainClassifier

st.set_page_config(page_title="MDCS — Domain Classifier", layout="wide")
st.title("Multi-Domain Document Classification")


@st.cache_resource
def _clf():
    return DomainClassifier.load()


clf = _clf()

tab1, tab2 = st.tabs(["Paste text", "Upload document"])

with tab1:
    text = st.text_area("Document text", height=240)
    if st.button("Classify text", type="primary") and text.strip():
        r = clf.predict_domain(text)
        st.subheader(f"Prediction: **{r.predicted_domain}** ({r.confidence}%)")
        st.write("All domains (model):", r.probabilities)
        st.write("Keyword presence (curated lexicon):", r.keyword_domain_scores)
        st.write("Top TF-IDF terms:", r.top_tfidf_terms[:12])
        st.expander("Matched manual keywords").write(r.matched_manual_keywords)

with tab2:
    up = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    if up and st.button("Classify file", type="primary"):
        suf = Path(up.name).suffix
        p = ROOT / "data" / "uploads" / f"_tmp_{up.name}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(up.getvalue())
        try:
            r = clf.predict_document(p)
            st.subheader(f"Prediction: **{r.predicted_domain}** ({r.confidence}%)")
            st.write("All domains (model):", r.probabilities)
            st.write("Keyword presence:", r.keyword_domain_scores)
            st.write("Top TF-IDF terms:", r.top_tfidf_terms[:12])
        finally:
            p.unlink(missing_ok=True)
