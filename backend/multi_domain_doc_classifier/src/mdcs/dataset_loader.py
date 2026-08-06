"""
Load heterogeneous sources and emit standardized columns: text, label.

Dataset notes (from inspection):
- Financial.csv: Title, Tag, Content — finance news; label Finance; text = Title + Tag + Content.
- train.csv: PubMed-style lines; `target` is paper section (METHODS, …), not domain — all rows
  are biomedical prose; we aggregate by abstract_id into one Medical document.
- tech.numbers / sports.numbers: Apple Numbers; each populated cell is `domain\\tfile\\ttitle\\tbody`.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_PATHS,
    MAX_ABSTRACTS_MEDICAL,
    MAX_ROWS_FINANCE,
    MIN_DOC_CHARS,
    RANDOM_STATE,
)

# If set (e.g. by --quick training), stop reading medical CSV after this many rows.
MEDICAL_MAX_ROWS: int | None = None


def _parse_numbers_cells(path: Path) -> list[tuple[str, str]]:
    from numbers_parser import Document

    rows: list[tuple[str, str]] = []
    doc = Document(str(path))
    for sheet in doc.sheets:
        for table in sheet.tables:
            for r in range(table.num_rows):
                for c in range(table.num_cols):
                    v = table.cell(r, c).value
                    if v is None:
                        continue
                    sval = str(v).strip()
                    if "\t" not in sval:
                        continue
                    parts = sval.split("\t", 3)
                    if len(parts) < 4:
                        continue
                    dom, _fid, title, body = parts[0].lower(), parts[1], parts[2], parts[3]
                    if dom.startswith("tech"):
                        label = "Technology"
                    elif dom.startswith("sport"):
                        label = "Sports"
                    else:
                        continue
                    text = f"{title} {body}".strip()
                    if text:
                        rows.append((text, label))
    return rows


def load_finance_csv(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_PATHS["finance_csv"]
    df = pd.read_csv(path)
    df = df.rename(columns={"Title": "title", "Tag": "tag", "Content": "content"})
    text = (
        df["title"].fillna("").astype(str)
        + " "
        + df["tag"].fillna("").astype(str)
        + " "
        + df["content"].fillna("").astype(str)
    )
    out = pd.DataFrame({"text": text.str.strip(), "label": "Finance"})
    out = out[out["text"].str.len() >= MIN_DOC_CHARS]
    out = out.drop_duplicates(subset=["text"])
    if len(out) > MAX_ROWS_FINANCE:
        out = out.sample(MAX_ROWS_FINANCE, random_state=RANDOM_STATE)
    return out.reset_index(drop=True)


def load_medical_csv(
    path: Path | None = None,
    max_abstracts: int = MAX_ABSTRACTS_MEDICAL,
    chunk_size: int = 250_000,
) -> pd.DataFrame:
    path = path or DEFAULT_PATHS["medical_csv"]
    agg: dict[int, list[str]] = defaultdict(list)
    total = 0
    for chunk in pd.read_csv(path, usecols=["abstract_id", "abstract_text"], chunksize=chunk_size):
        for aid, txt in zip(chunk["abstract_id"].tolist(), chunk["abstract_text"].tolist()):
            if isinstance(txt, str) and txt.strip():
                agg[int(aid)].append(txt.strip())
        total += len(chunk)
        if MEDICAL_MAX_ROWS is not None and total >= MEDICAL_MAX_ROWS:
            break
    records = []
    for aid, parts in agg.items():
        merged = " ".join(parts)
        if len(merged) >= MIN_DOC_CHARS:
            records.append({"abstract_id": aid, "text": merged})
    abs_df = pd.DataFrame.from_records(records)
    if len(abs_df) > max_abstracts:
        abs_df = abs_df.sample(max_abstracts, random_state=RANDOM_STATE)
    out = pd.DataFrame({"text": abs_df["text"], "label": "Medical"})
    return out.reset_index(drop=True)


def load_technology_numbers(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_PATHS["technology_numbers"]
    pairs = _parse_numbers_cells(path)
    df = pd.DataFrame(pairs, columns=["text", "label"])
    df = df[df["text"].str.len() >= MIN_DOC_CHARS].drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def load_sports_numbers(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_PATHS["sports_numbers"]
    pairs = _parse_numbers_cells(path)
    df = pd.DataFrame(pairs, columns=["text", "label"])
    df = df[df["text"].str.len() >= MIN_DOC_CHARS].drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def analyze_frame(df: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "rows": len(df),
        "missing_text": int(df["text"].isna().sum()),
        "dup_text": int(df["text"].duplicated().sum()),
        "short_docs": int((df["text"].str.len() < MIN_DOC_CHARS).sum()),
        "label_counts": df["label"].value_counts().to_dict(),
    }


def build_master_dataframe(paths: dict[str, Path] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    paths = paths or DEFAULT_PATHS
    parts = [
        load_finance_csv(paths.get("finance_csv")),
        load_medical_csv(paths.get("medical_csv")),
        load_technology_numbers(paths.get("technology_numbers")),
        load_sports_numbers(paths.get("sports_numbers")),
    ]
    master = pd.concat(parts, ignore_index=True)
    reports = [
        analyze_frame(parts[0], "finance"),
        analyze_frame(parts[1], "medical_aggregated"),
        analyze_frame(parts[2], "technology_numbers"),
        analyze_frame(parts[3], "sports_numbers"),
    ]
    return master, reports
