#!/usr/bin/env bash
# Bundle classifier artifacts for backend integration.
# Artifacts: model.pkl (MultinomialNB), vectorizer.pkl (dict with TfidfVectorizer + min_doc_chars).
# Backend must use the same sklearn major version as training for unpickling.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M)"
OUT="${ROOT}/dist/mdcs_backend_bundle_${STAMP}.tar.gz"
mkdir -p "${ROOT}/dist"
cd "$ROOT"
tar czvf "$OUT" \
  models/model.pkl \
  models/vectorizer.pkl \
  notebooks/01_full_pipeline.ipynb
echo ""
echo "Created: $OUT"
ls -lh "$OUT"
