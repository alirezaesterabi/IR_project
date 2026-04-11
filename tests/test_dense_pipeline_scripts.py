"""Tests for the reusable dense pipeline modules and runner scripts."""

from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.retrieval.dense_retriever import DenseRetriever

_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = _ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_dense_retriever_search_converts_distances(monkeypatch, tmp_path: Path) -> None:
    class FakeCollection:
        def query(self, query_embeddings, n_results, include):
            assert n_results == 2
            assert include == ["distances", "metadatas"]
            assert len(query_embeddings) == 1
            return {
                "ids": [["d1", "d2"]],
                "distances": [[0.1, 0.4]],
                "metadatas": [[{"caption": "Doc 1"}, {"caption": "Doc 2"}]],
            }

    class FakeClient:
        def __init__(self, path: str):
            self.path = path

        def get_collection(self, name: str):
            assert name == "opensanctions_minilm_FULL_20260407"
            return FakeCollection()

    class FakeEncoder:
        def __init__(self, hf_name: str, **kwargs):
            assert hf_name == "all-MiniLM-L6-v2"
            assert kwargs["device"] == "cpu"

        def encode(self, texts, normalize_embeddings, convert_to_numpy):
            assert texts == ["arms trafficker"]
            assert normalize_embeddings is True
            assert convert_to_numpy is True
            return np.array([[0.25, 0.75]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(PersistentClient=FakeClient),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeEncoder),
    )

    retriever = DenseRetriever(
        model_key="minilm",
        run_tag="FULL_20260407",
        chroma_dir=tmp_path / "chroma",
        models_dir=tmp_path / "models",
        device="cpu",
    )
    hits = retriever.search("arms trafficker", k=2)

    assert hits == [
        ("d1", pytest.approx(0.9), {"caption": "Doc 1"}),
        ("d2", pytest.approx(0.6), {"caption": "Doc 2"}),
    ]


def test_export_dense_run_writes_shared_schema(monkeypatch, tmp_path: Path) -> None:
    mod = _load_script("export_dense_run.py")
    queries_df = pd.DataFrame(
        [
            {"query_id": "Q1", "query_type": 1, "query_text": "alpha"},
            {"query_id": "Q2", "query_type": 2, "query_text": "beta"},
        ]
    )

    class FakeDenseRetriever:
        def __init__(self, **kwargs):
            assert kwargs["model_key"] == "minilm"
            assert kwargs["run_tag"] == "FULL_20260407"

        def search(self, query_text: str, k: int):
            assert k == 3
            if query_text == "alpha":
                return [("d1", 0.9, {}), ("d2", 0.7, {})]
            return [("d3", 0.8, {})]

    import src.retrieval

    monkeypatch.setattr(src.retrieval, "DenseRetriever", FakeDenseRetriever)

    out = tmp_path / "dense.csv"
    mod.export_dense_run(
        model_key="minilm",
        run_tag="FULL_20260407",
        output_path=out,
        queries_df=queries_df,
        chroma_dir=tmp_path / "chroma",
        models_dir=tmp_path / "models",
        top_k=3,
        device="cpu",
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["query_id"] == "Q1"
    assert rows[0]["doc_id"] == "d1"
    assert rows[0]["rank"] == "1"
    assert rows[0]["dense_score"] == "0.900000000"
    assert rows[-1]["query_id"] == "Q2"
    assert rows[-1]["doc_id"] == "d3"


def test_export_bm25_run_writes_shared_schema(monkeypatch, tmp_path: Path) -> None:
    mod = _load_script("export_bm25_run.py")
    queries_df = pd.DataFrame(
        [{"query_id": "Q1", "query_type": 1, "query_text": "Sanction Vessel"}]
    )

    class FakeBM25Retriever:
        def load(self, path: Path) -> None:
            assert path == tmp_path / "models" / "bm25"

        def search(self, query_tokens, k: int):
            assert query_tokens == ["sanction", "vessel"]
            assert k == 5
            return [("d9", 12.5), ("d3", 11.0)]

    import src.retrieval.classical_ir as classical_ir

    monkeypatch.setattr(classical_ir, "BM25Retriever", FakeBM25Retriever)

    out = tmp_path / "bm25.csv"
    mod.export_bm25_run(
        output_path=out,
        queries_df=queries_df,
        models_dir=tmp_path / "models",
        top_k=5,
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [row["doc_id"] for row in rows] == ["d9", "d3"]
    assert rows[0]["bm25_score"] == "12.500000000"


