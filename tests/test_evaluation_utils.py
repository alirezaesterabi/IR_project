"""Tests for src.evaluation.utils (I/O and tables; ranx tests optional)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.utils import (
    build_bm25_rrf_comparison_table,
    load_qrels_type_json,
    load_queries_type_json,
    load_run_csv,
    merge_all_types_qrels,
    merge_all_types_queries,
    validate_run_coverage,
)


def test_load_qrels_type_json(tmp_path: Path) -> None:
    p = tmp_path / "q.json"
    p.write_text(
        json.dumps(
            {
                "query_type": 1,
                "qrels": {"Q1_001": {"d1": 1, "d2": 0}},
            }
        ),
        encoding="utf-8",
    )
    q = load_qrels_type_json(p)
    assert q["Q1_001"] == {"d1": 1, "d2": 0}


def test_merge_all_types_qrels_duplicate_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"qrels": {"Q1": {"d": 1}}}), encoding="utf-8")
    b.write_text(json.dumps({"qrels": {"Q1": {"d": 2}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        merge_all_types_qrels({1: a, 2: b})


def test_load_queries_type_json(tmp_path: Path) -> None:
    p = tmp_path / "q.json"
    p.write_text(
        json.dumps(
            {
                "query_type": 3,
                "queries": [
                    {
                        "query_id": "Q3_001",
                        "query_type": 3,
                        "query_texts": ["hello"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    df = load_queries_type_json(p)
    assert list(df.columns) == ["query_id", "query_type", "query_text"]
    assert df.iloc[0]["query_id"] == "Q3_001"
    assert df.iloc[0]["query_text"] == "hello"


def test_load_run_csv_with_rrf_score(tmp_path: Path) -> None:
    csv_path = tmp_path / "run.csv"
    csv_path.write_text(
        "query_id,doc_id,rank,rrf_score\n"
        "q1,d1,1,0.5\n"
        "q1,d2,2,0.25\n",
        encoding="utf-8",
    )
    r = load_run_csv(csv_path)
    assert set(r["q1"].keys()) == {"d1", "d2"}
    assert r["q1"]["d1"] == 0.5


def test_load_run_csv_rank_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "run.csv"
    csv_path.write_text(
        "query_id,doc_id,rank\nq1,d1,1\nq1,d2,2\n", encoding="utf-8"
    )
    r = load_run_csv(csv_path)
    assert r["q1"]["d1"] == 1.0
    assert r["q1"]["d2"] == pytest.approx(0.5)


def test_validate_run_coverage() -> None:
    run = {"q1": {"a": 1.0}, "q2": {}}
    df = validate_run_coverage(run, ["q1", "q2", "q3"], min_depth=1)
    assert df.loc[df["query_id"] == "q1", "missing"].iloc[0] is False
    assert df.loc[df["query_id"] == "q2", "missing"].iloc[0] is True
    assert df.loc[df["query_id"] == "q3", "missing"].iloc[0] is True


def test_build_bm25_rrf_comparison_table() -> None:
    df_b = pd.DataFrame(
        {
            "query_type": [1, 2],
            "n_queries": [10, 10],
            "map": [0.2, 0.3],
        }
    )
    df_r = pd.DataFrame(
        {
            "query_type": [1, 2],
            "n_queries": [10, 10],
            "map": [0.25, 0.35],
        }
    )
    out = build_bm25_rrf_comparison_table(df_b, df_r, metric_columns=["map"])
    assert "map_RRF_minus_BM25" in out.columns
    assert out.loc[out["query_type"] == 1, "map_RRF_minus_BM25"].iloc[0] == pytest.approx(
        0.05
    )


def test_evaluate_type_subset_ranx() -> None:
    pytest.importorskip("ranx")
    from src.evaluation.utils import evaluate_type_subset

    qrels = {"q1": {"d1": 1}}
    run = {"q1": {"d1": 10.0, "d2": 1.0}}
    m = evaluate_type_subset(qrels, run, ["q1"], ["map", "precision@1"])
    assert "map" in m
    assert m["precision@1"] == 1.0
