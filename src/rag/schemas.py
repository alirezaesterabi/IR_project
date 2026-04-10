"""Shared typed schemas for the Type 7 RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Type7Query:
    query_id: str
    source_query_id: str
    source_type: int
    query_type: int
    query_text: str
    query_notes: str = ""
    gold_answer: str = ""
    gold_answer_seed: str = ""
    evaluation_mode: str = "ragas_pseudo_ground_truth"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContextRow:
    query_id: str
    source_query_id: str
    source_type: int
    query_type: int
    query_text: str
    query_notes: str
    gold_answer: str
    doc_id: str
    rank: int
    rrf_score: float
    caption: str
    schema: str
    text_blob: str
    embedding_text: str
    meta_country: str
    meta_programId: str
    meta_datasets: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerRecord:
    query_id: str
    source_query_id: str
    source_type: int
    query_type: int
    query_text: str
    query_notes: str
    gold_answer: str
    top_k: int
    used_doc_ids: list[str]
    model_name: str
    prompt_version: str
    answer: str
    raw_response: str
    generation_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
