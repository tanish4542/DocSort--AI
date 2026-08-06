"""
Extract plain text from PDF, DOCX, and TXT for the prediction pipeline.
"""
from __future__ import annotations

from pathlib import Path


def extract_text_txt(path: Path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_text_pdf(path: Path) -> str:
    import pdfplumber

    chunks: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                chunks.append(t)
    return "\n\n".join(chunks)


def extract_text_docx(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


def extract_text_auto(path: Path) -> str:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".txt":
        return extract_text_txt(path)
    if suf == ".pdf":
        return extract_text_pdf(path)
    if suf in (".docx",):
        return extract_text_docx(path)
    raise ValueError(f"Unsupported format: {suf}")
