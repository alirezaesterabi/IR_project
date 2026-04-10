"""Build Type 7 RAG context rows from canonical retrieval outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.rag.schemas import ContextRow


DOC_PREVIEW_LIMIT = 800


def load_type7_queries(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    df = pd.DataFrame(payload.get("queries", []))
    if df.empty:
        raise ValueError(f"No Type 7 queries found in {path}")
    return df


def load_run_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"query_id", "doc_id", "rank", "rrf_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Run file missing required columns: {sorted(missing)}")
    return df


def load_document_lookup(path: Path, doc_ids: set[str]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = str(record.get("doc_id", ""))
            if doc_id not in doc_ids:
                continue
            metadata = record.get("metadata", {}) or {}
            lookup[doc_id] = {
                "caption": str(record.get("caption", "")),
                "schema": str(record.get("schema", "")),
                "text_blob": str(record.get("text_blob", ""))[:DOC_PREVIEW_LIMIT],
                "embedding_text": str(record.get("embedding_text", ""))[:DOC_PREVIEW_LIMIT],
                "meta_country": _flatten(metadata.get("country")),
                "meta_programId": _flatten(metadata.get("programId")),
                "meta_datasets": _flatten(metadata.get("datasets")),
            }
    return lookup


def _flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def build_context_rows(
    queries_df: pd.DataFrame,
    run_df: pd.DataFrame,
    docs_lookup: dict[str, dict],
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for query in queries_df.to_dict(orient="records"):
        source_query_id = str(query["source_query_id"])
        matches = (
            run_df[run_df["query_id"] == source_query_id]
            .sort_values("rank")
            .head(top_k)
        )
        for _, match in matches.iterrows():
            doc_id = str(match["doc_id"])
            doc = docs_lookup.get(doc_id, {})
            row = ContextRow(
                query_id=str(query["query_id"]),
                source_query_id=source_query_id,
                source_type=int(query.get("source_type", 0)),
                query_type=int(query.get("query_type", 7)),
                query_text=str(query.get("query_text", "")),
                query_notes=str(query.get("query_notes", "")),
                gold_answer=str(query.get("gold_answer", "")),
                doc_id=doc_id,
                rank=int(match["rank"]),
                rrf_score=float(match["rrf_score"]),
                caption=str(doc.get("caption", "")),
                schema=str(doc.get("schema", "")),
                text_blob=str(doc.get("text_blob", "")),
                embedding_text=str(doc.get("embedding_text", "")),
                meta_country=str(doc.get("meta_country", "")),
                meta_programId=str(doc.get("meta_programId", "")),
                meta_datasets=str(doc.get("meta_datasets", "")),
            )
            rows.append(row.to_dict())

    context_df = pd.DataFrame(rows)
    if context_df.empty:
        raise ValueError(
            "No context rows were built. Check that source_query_id values match "
            "the query_id values in the RRF run."
        )
    return context_df
