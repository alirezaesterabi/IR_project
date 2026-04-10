"""Generate grounded Type 7 answers from prepared context rows."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import requests

from src.rag.schemas import AnswerRecord


SYSTEM_PROMPT = """You are a sanctions intelligence analyst.

Write a concise narrative answer grounded only in the retrieved records that are
provided. Do not add outside knowledge. If the records are incomplete, say so.
Support key claims with bracketed record references like [Record 1].
"""


def format_context_block(row: pd.Series, record_number: int) -> str:
    parts = [f"[Record {record_number}]"]
    if row.get("caption"):
        parts.append(f"Entity: {row['caption']}")
    if row.get("schema"):
        parts.append(f"Type: {row['schema']}")
    if row.get("meta_country"):
        parts.append(f"Country: {row['meta_country']}")
    if row.get("meta_programId"):
        parts.append(f"Programme: {row['meta_programId']}")
    if row.get("meta_datasets"):
        parts.append(f"Listed in: {row['meta_datasets']}")
    if row.get("embedding_text"):
        parts.append(f"Details: {row['embedding_text']}")
    elif row.get("text_blob"):
        parts.append(f"Details: {row['text_blob']}")
    parts.append(f"RRF rank: {int(row['rank'])}")
    parts.append(f"RRF score: {float(row['rrf_score']):.6f}")
    return "\n".join(parts)


def build_user_prompt(query_text: str, query_notes: str, context_rows: pd.DataFrame) -> str:
    blocks = [
        format_context_block(row, i)
        for i, (_, row) in enumerate(context_rows.iterrows(), start=1)
    ]
    notes_line = query_notes.strip() if query_notes.strip() else "None"
    return f"""Query:
{query_text}

Analyst notes:
{notes_line}

Retrieved records:
{chr(10).join(chr(10) + block for block in blocks)}

Write a 1-2 paragraph grounded answer that:
1. Summarises the key entities and sanctions context.
2. Highlights the strongest evidence from the retrieved records.
3. Notes important uncertainty or missing information when needed.
4. Uses bracketed record references like [Record 1].
"""


def ollama_generate(
    *,
    user_prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    timeout_seconds: int,
) -> str:
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def generate_answer_records(
    context_df: pd.DataFrame,
    *,
    model: str,
    prompt_version: str,
    base_url: str,
    temperature: float,
    timeout_seconds: int,
) -> list[dict]:
    records: list[dict] = []
    for query_id, group in context_df.groupby("query_id", sort=True):
        first = group.iloc[0]
        prompt = build_user_prompt(
            query_text=str(first.get("query_text", "")),
            query_notes=str(first.get("query_notes", "")),
            context_rows=group,
        )
        try:
            answer = ollama_generate(
                user_prompt=prompt,
                model=model,
                base_url=base_url,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            answer = f"Generation failed: {exc}"
            status = "error"

        record = AnswerRecord(
            query_id=str(query_id),
            source_query_id=str(first.get("source_query_id", "")),
            source_type=int(first.get("source_type", 0)),
            query_type=int(first.get("query_type", 7)),
            query_text=str(first.get("query_text", "")),
            query_notes=str(first.get("query_notes", "")),
            gold_answer=str(first.get("gold_answer", "")),
            top_k=int(len(group)),
            used_doc_ids=[str(value) for value in group["doc_id"].tolist()],
            model_name=model,
            prompt_version=prompt_version,
            answer=answer,
            raw_response=answer,
            generation_status=status,
        )
        records.append(record.to_dict())
    return records


def records_to_dataframe(records: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(records))
