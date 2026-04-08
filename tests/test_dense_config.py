"""Tests for dense embedding registry and RUN_TAG helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_build_dense() -> object:
    path = _ROOT / "scripts" / "build_dense_embeddings.py"
    spec = importlib.util.spec_from_file_location("build_dense_embeddings", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


from src.retrieval.dense_config import (
    CHROMA_MAX_BATCH,
    DENSE_MODELS,
    compute_chroma_batch_size,
    compute_run_tag,
    model_file_suffix,
    resolved_paths,
    size_tag_for_limit,
)


def test_dense_model_keys_and_dims() -> None:
    assert set(DENSE_MODELS.keys()) == {"minilm", "bge-m3"}
    assert DENSE_MODELS["minilm"].dim == 384
    assert DENSE_MODELS["bge-m3"].dim == 1024


def test_model_file_suffix() -> None:
    assert model_file_suffix("minilm") == "minilm"
    assert model_file_suffix("bge-m3") == "bge_m3"


def test_size_tag_for_limit() -> None:
    assert size_tag_for_limit(None) == "FULL"
    assert size_tag_for_limit(1000) == "1K"
    assert size_tag_for_limit(50_000) == "50K"
    assert size_tag_for_limit(1_500_000) == "1M"


def test_compute_run_tag_explicit_date() -> None:
    assert compute_run_tag(None, date_tag="20260407") == "FULL_20260407"
    assert compute_run_tag(50_000, date_tag="20260407") == "50K_20260407"


def test_compute_chroma_batch_size_clamped() -> None:
    b = compute_chroma_batch_size(384, available_ram_gb=16.0)
    assert 1000 <= b <= CHROMA_MAX_BATCH


def test_resolved_paths_naming(tmp_path: Path) -> None:
    run_tag = "50K_20260407"
    p = resolved_paths(tmp_path, "bge-m3", run_tag)
    assert p["embedding_cache"] == tmp_path / "doc_embeddings_bge_m3_50K_20260407.npy"
    assert p["doc_ids_cache"] == tmp_path / "doc_ids_bge_m3_50K_20260407.json"
    assert p["chroma_collection"] == "opensanctions_bge_m3_50K_20260407"


def test_build_dense_embeddings_cli_parses() -> None:
    mod = _load_build_dense()
    args = mod.parse_args(["--limit", "3", "--model", "minilm", "--run-tag", "test_cli"])
    assert args.limit == 3
    assert args.model == "minilm"


def test_load_corpus_tiny_jsonl(tmp_path: Path) -> None:
    mod = _load_build_dense()
    load_corpus = mod.load_corpus

    p = tmp_path / "d.jsonl"
    rows = [
        {"doc_id": "a1", "embedding_text": "hello", "caption": "c", "schema": "Person"},
        {"doc_id": "a2", "caption": "x", "schema": "Company", "metadata": {}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ids, texts, docs = load_corpus(p, None)
    assert ids == ["a1", "a2"]
    assert texts[0] == "hello"
    assert len(docs) == 2
