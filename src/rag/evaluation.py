"""Helpers for Type 7 answer evaluation with RAGAS."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_context_text(row: pd.Series) -> str:
    parts: list[str] = []
    if row.get("caption"):
        parts.append(f"Entity: {row['caption']}")
    if row.get("schema"):
        parts.append(f"Type: {row['schema']}")
    if row.get("meta_country"):
        parts.append(f"Country: {row['meta_country']}")
    if row.get("meta_programId"):
        parts.append(f"Programme: {row['meta_programId']}")
    text = row.get("embedding_text") or row.get("text_blob") or ""
    if text:
        parts.append(f"Details: {text}")
    return " | ".join(parts) if parts else "No context available"


def build_pseudo_ground_truth(query_group: pd.DataFrame, fallback_question: str) -> str:
    parts: list[str] = []
    for _, row in query_group.iterrows():
        caption = str(row.get("caption", "")).strip()
        program = str(row.get("meta_programId", "")).strip()
        country = str(row.get("meta_country", "")).strip()
        if not caption:
            continue
        segment = caption
        if program:
            segment += f" — {program}"
        if country:
            segment += f" ({country})"
        parts.append(segment)
    return "; ".join(parts) if parts else fallback_question


def build_ragas_rows(answer_df: pd.DataFrame, context_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    grouped_context = {qid: group for qid, group in context_df.groupby("query_id", sort=True)}

    for _, answer_row in answer_df.iterrows():
        query_id = str(answer_row["query_id"])
        context_group = grouped_context.get(query_id)
        if context_group is None or context_group.empty:
            continue
        question = str(answer_row.get("query_text", ""))
        rows.append(
            {
                "query_id": query_id,
                "question": question,
                "answer": str(answer_row.get("answer", "")),
                "contexts": [build_context_text(row) for _, row in context_group.iterrows()],
                "ground_truth": build_pseudo_ground_truth(context_group, question),
            }
        )
    return rows


def evaluate_with_ragas(
    ragas_rows: list[dict],
    *,
    evaluator_model: str,
    base_url: str,
) -> tuple[pd.DataFrame, dict]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "RAGAS dependencies are missing. Install requirements-rag.txt first."
        ) from exc

    llm = _build_langchain_ollama(evaluator_model, base_url)
    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in ragas_rows],
            "answer": [row["answer"] for row in ragas_rows],
            "contexts": [row["contexts"] for row in ragas_rows],
            "ground_truth": [row["ground_truth"] for row in ragas_rows],
        }
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(llm),
        raise_exceptions=False,
    )
    per_query = result.to_pandas()
    per_query["query_id"] = [row["query_id"] for row in ragas_rows]
    per_query["pseudo_ground_truth"] = [row["ground_truth"] for row in ragas_rows]

    summary = {}
    for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if key in per_query.columns:
            summary[key] = float(pd.to_numeric(per_query[key], errors="coerce").mean())
    return per_query, summary


def _build_langchain_ollama(model: str, base_url: str):
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama

    return ChatOllama(model=model, base_url=base_url, temperature=0.0)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
