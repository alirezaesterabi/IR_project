"""Helpers for deriving Type 7 queries from Types 3 and 4."""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.schemas import Type7Query


def load_query_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_query_text(query_entry: dict) -> str:
    query_texts = query_entry.get("query_texts", [])
    if isinstance(query_texts, list) and query_texts:
        return str(query_texts[0]).strip()
    if isinstance(query_texts, str):
        return query_texts.strip()
    return ""


def build_type7_text(source_type: int, source_text: str) -> str:
    source_text = source_text.strip().rstrip("?")
    if source_type == 3:
        return (
            "Summarise the sanctions-related background, key entities, and "
            f"supporting evidence for: {source_text}."
        )
    if source_type == 4:
        return (
            "Summarise the entities, relationships, and sanctions-related "
            f"evidence connected to: {source_text}."
        )
    raise ValueError(f"Unsupported source_type: {source_type}")


def build_gold_seed(source_type: int, source_text: str, notes: str) -> str:
    if source_type == 3:
        return (
            f"Ground the answer in retrieved records for the descriptive need: "
            f"{source_text}. Include key entities, sanctions programmes, and "
            "the strongest supporting details."
        )
    return (
        f"Ground the answer in retrieved records for the relational need: "
        f"{source_text}. Highlight linked entities, relationship evidence, and "
        "relevant sanctions context."
    )


def build_type7_queries(type3_data: dict, type4_data: dict) -> dict:
    queries: list[dict] = []
    counter = 1

    for source_type, payload in ((3, type3_data), (4, type4_data)):
        for entry in payload.get("queries", []):
            query_id = f"Q7_{counter:03d}"
            source_text = first_query_text(entry)
            notes = str(entry.get("notes", "")).strip()
            query = Type7Query(
                query_id=query_id,
                source_query_id=str(entry.get("query_id", "")),
                source_type=source_type,
                query_type=7,
                query_text=build_type7_text(source_type, source_text),
                query_notes=notes,
                gold_answer_seed=build_gold_seed(source_type, source_text, notes),
            )
            queries.append(query.to_dict())
            counter += 1

    return {
        "query_type": 7,
        "n_queries": len(queries),
        "queries": queries,
    }
