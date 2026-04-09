"""
Lightweight evaluation utilities for Types 1–6 (qrels, runs, ranx metrics).

Assumes per-type JSON qrels under data/qrels/ and runs as CSV with at least
query_id, doc_id, rank (optional score column: score, rrf_score, bm25_score, etc.).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:
    from ranx import Qrels, Run, evaluate
except ImportError as e:  # pragma: no cover
    Qrels = Run = evaluate = None  # type: ignore[misc, assignment]
    _RANX_IMPORT_ERROR = e
else:
    _RANX_IMPORT_ERROR = None

# --- Metric policy (per project / theory.md) ---------------------------------

TYPE_METRICS: dict[int, list[str]] = {
    1: ["precision@1", "mrr", "map"],
    2: ["precision@1", "mrr", "map"],
    3: ["ndcg@10", "map", "recall@10"],
    4: ["ndcg@10", "map", "recall@10"],
    5: ["recall@10", "recall@20", "map"],
    6: ["recall@10", "recall@20", "map"],
}

OVERALL_METRICS: list[str] = ["recall@10", "map"]

# All metric columns that may appear in wide tables (NaN where not computed for a type)
ALL_TABLE_METRICS: list[str] = sorted(
    {m for metrics in TYPE_METRICS.values() for m in metrics} | set(OVERALL_METRICS)
)


def _require_ranx() -> None:
    if _RANX_IMPORT_ERROR is not None:
        raise ImportError(
            "ranx is required for evaluation. Install with: pip install ranx>=0.3.16"
        ) from _RANX_IMPORT_ERROR


def default_data_paths(project_root: Path | str) -> tuple[dict[int, Path], dict[int, Path]]:
    """Default paths to per-type qrels and queries under data/."""
    root = Path(project_root)
    qrels = {
        t: root / "data" / "qrels" / f"qrels_type_{t}.json"
        for t in range(1, 7)
    }
    queries = {
        t: root / "data" / "queries" / f"queries_type_{t}.json"
        for t in range(1, 7)
    }
    return qrels, queries


def load_qrels_type_json(path: str | Path) -> dict[str, dict[str, int]]:
    """
    Load qrels from project JSON: top-level 'qrels' -> {query_id: {doc_id: rel}}.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("qrels")
    if not isinstance(raw, dict):
        raise ValueError(f"Expected 'qrels' object in {path}")
    out: dict[str, dict[str, int]] = {}
    for qid, docs in raw.items():
        if not isinstance(docs, dict):
            continue
        out[str(qid)] = {str(d): int(v) for d, v in docs.items()}
    return out


def load_queries_type_json(path: str | Path) -> pd.DataFrame:
    """
    Load queries JSON; return DataFrame with query_id, query_type, query_text
    (first string in query_texts).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows: list[dict[str, Any]] = []
    for item in data.get("queries", []):
        qid = str(item["query_id"])
        qtype = int(item.get("query_type", data.get("query_type", 0)))
        texts = item.get("query_texts") or []
        qtext = str(texts[0]) if texts else ""
        rows.append(
            {"query_id": qid, "query_type": qtype, "query_text": qtext}
        )
    return pd.DataFrame(rows)


def merge_all_types_qrels(
    qrels_paths_by_type: Mapping[int, str | Path],
) -> dict[str, dict[str, int]]:
    """Merge per-type qrels files; raises if the same query_id appears twice."""
    merged: dict[str, dict[str, int]] = {}
    for t, p in sorted(qrels_paths_by_type.items()):
        part = load_qrels_type_json(p)
        overlap = set(merged) & set(part)
        if overlap:
            raise ValueError(
                f"Duplicate query_id(s) merging type {t} qrels: {sorted(overlap)[:5]}..."
            )
        merged.update(part)
    return merged


def merge_all_types_queries(
    query_paths_by_type: Mapping[int, str | Path],
) -> pd.DataFrame:
    """Concatenate per-type query JSON files into one DataFrame."""
    frames = [
        load_queries_type_json(query_paths_by_type[t])
        for t in sorted(query_paths_by_type.keys())
    ]
    if not frames:
        return pd.DataFrame(columns=["query_id", "query_type", "query_text"])
    return pd.concat(frames, ignore_index=True)


def _detect_score_column(fieldnames: Sequence[str] | None) -> str | None:
    if not fieldnames:
        return None
    lower = {n.lower(): n for n in fieldnames}
    for candidate in (
        "rrf_score",
        "score",
        "bm25_score",
        "dense_score",
        "similarity",
    ):
        if candidate in lower:
            return lower[candidate]
    return None


def load_run_csv(
    path: str | Path,
    score_col: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Read a ranked-list CSV into {query_id: {doc_id: score}} for ranx.Run.
    Higher score = better rank. If no score column, uses 1/rank.
    """
    path = Path(path)
    run: dict[str, dict[str, float]] = defaultdict(dict)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}

        fn = list(reader.fieldnames)
        # Normalise expected columns
        col_map = {c.lower(): c for c in fn}
        q_col = col_map.get("query_id")
        d_col = col_map.get("doc_id")
        r_col = col_map.get("rank")
        if not q_col or not d_col or not r_col:
            raise ValueError(
                f"Run CSV must have query_id, doc_id, rank columns; got {fn!r}"
            )

        use_score = score_col if score_col in fn else _detect_score_column(fn)

        for row in reader:
            qid = str(row[q_col]).strip()
            did = str(row[d_col]).strip()
            if not qid or not did:
                continue
            try:
                rank = int(float(str(row[r_col]).strip()))
            except (TypeError, ValueError):
                continue
            if use_score and row.get(use_score) not in (None, ""):
                try:
                    score = float(str(row[use_score]).strip())
                except (TypeError, ValueError):
                    score = 1.0 / max(rank, 1)
            else:
                score = 1.0 / max(rank, 1)
            run[qid][did] = score

    return {k: dict(v) for k, v in run.items()}


def _qrels_for_queries(
    qrels: Mapping[str, Mapping[str, int]], query_ids: Iterable[str]
) -> dict[str, dict[str, int]]:
    return {qid: dict(qrels[qid]) for qid in query_ids if qid in qrels}


def _run_for_queries(
    run: Mapping[str, Mapping[str, float]], query_ids: Iterable[str]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for qid in query_ids:
        if qid in run:
            out[qid] = dict(run[qid])
        else:
            out[qid] = {}
    return out


def evaluate_type_subset(
    qrels: Mapping[str, Mapping[str, int]],
    run_dict: Mapping[str, Mapping[str, float]],
    query_ids: Sequence[str],
    metrics: Sequence[str],
) -> dict[str, float]:
    """Evaluate ranx metrics on a subset of queries (both must align)."""
    _require_ranx()
    qids = [q for q in query_ids if q in qrels]
    if not qids:
        return {m: float("nan") for m in metrics}
    q_sub = _qrels_for_queries(qrels, qids)
    r_sub = _run_for_queries(run_dict, qids)
    q_obj = Qrels(q_sub)
    r_obj = Run(r_sub)
    return evaluate(q_obj, r_obj, metrics)


def evaluate_all_types(
    qrels: Mapping[str, Mapping[str, int]],
    run_dict: Mapping[str, Mapping[str, float]],
    queries_df: pd.DataFrame,
    type_metrics: Mapping[int, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """
    One row per query type (1–6) with n_queries and all ALL_TABLE_METRICS
    (NaN where not defined for that type).
    """
    tm = type_metrics or TYPE_METRICS
    rows: list[dict[str, Any]] = []

    for qtype in sorted(tm.keys()):
        ids = queries_df.loc[queries_df["query_type"] == qtype, "query_id"].astype(
            str
        )
        qids = [q for q in ids.tolist() if q in qrels]
        metrics = list(tm[qtype])
        scores = evaluate_type_subset(qrels, run_dict, qids, metrics)
        row: dict[str, Any] = {
            "query_type": qtype,
            "n_queries": len(qids),
        }
        for m in ALL_TABLE_METRICS:
            row[m] = scores.get(m, float("nan"))
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_overall_types_1_6(
    qrels: Mapping[str, Mapping[str, int]],
    run_dict: Mapping[str, Mapping[str, float]],
    queries_df: pd.DataFrame,
    overall_metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Macro-style overall metrics over all Types 1–6 queries present in qrels."""
    metrics = list(overall_metrics or OVERALL_METRICS)
    mask = queries_df["query_type"].isin(range(1, 7))
    qids = [
        q
        for q in queries_df.loc[mask, "query_id"].astype(str).tolist()
        if q in qrels
    ]
    return evaluate_type_subset(qrels, run_dict, qids, metrics)


def evaluate_per_query(
    qrels: Mapping[str, Mapping[str, int]],
    run_dict: Mapping[str, Mapping[str, float]],
    query_ids: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Per-query metric rows (for error analysis)."""
    _require_ranx()
    rows: list[dict[str, Any]] = []
    for qid in query_ids:
        if qid not in qrels:
            continue
        q_obj = Qrels({qid: dict(qrels[qid])})
        r_obj = Run({qid: dict(run_dict.get(qid, {}))})
        scores = evaluate(q_obj, r_obj, list(metrics))
        rows.append({"query_id": qid, **scores})
    return pd.DataFrame(rows)


def evaluate_per_query_all_types(
    qrels: Mapping[str, Mapping[str, int]],
    run_dict: Mapping[str, Mapping[str, float]],
    queries_df: pd.DataFrame,
    type_metrics: Mapping[int, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """
    Per-query metrics using each row's query_type to select TYPE_METRICS.
    Adds query_type column; metric columns include NaN where not applicable.
    """
    tm = type_metrics or TYPE_METRICS
    parts: list[pd.DataFrame] = []
    for qtype in sorted(tm.keys()):
        metrics = list(tm[qtype])
        mask = queries_df["query_type"] == qtype
        qids = [
            q
            for q in queries_df.loc[mask, "query_id"].astype(str).tolist()
            if q in qrels
        ]
        if not qids:
            continue
        sub = evaluate_per_query(qrels, run_dict, qids, metrics)
        for m in ALL_TABLE_METRICS:
            if m not in sub.columns:
                sub[m] = float("nan")
        sub.insert(1, "query_type", qtype)
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def validate_run_coverage(
    run_dict: Mapping[str, Mapping[str, float]],
    query_ids: Sequence[str],
    min_depth: int = 10,
) -> pd.DataFrame:
    """
    Report missing queries and shallow runs (fewer than min_depth docs).
    """
    rows: list[dict[str, Any]] = []
    for qid in query_ids:
        docs = run_dict.get(qid, {})
        n = len(docs)
        rows.append(
            {
                "query_id": qid,
                "n_docs": n,
                "missing": n == 0,
                "shallow": 0 < n < min_depth,
            }
        )
    return pd.DataFrame(rows)


def build_bm25_rrf_comparison_table(
    df_bm25: pd.DataFrame,
    df_rrf: pd.DataFrame,
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Merge per-type BM25 and RRF tables on query_type.
    Adds <metric>_RRF_minus_BM25 for each metric column.
    """
    cols = list(metric_columns) if metric_columns is not None else ALL_TABLE_METRICS
    key = "query_type"
    b = df_bm25.copy()
    r = df_rrf.copy()
    b = b[[key, "n_queries"] + [c for c in cols if c in b.columns]]
    r = r[[key, "n_queries"] + [c for c in cols if c in r.columns]]

    merged = b.merge(r, on=key, how="outer", suffixes=("_bm25", "_rrf"))

    # Prefer bm25 n_queries when both exist
    if "n_queries_bm25" in merged.columns and "n_queries_rrf" in merged.columns:
        merged["n_queries"] = merged["n_queries_bm25"].combine_first(
            merged["n_queries_rrf"]
        )
        merged.drop(
            columns=["n_queries_bm25", "n_queries_rrf"],
            inplace=True,
            errors="ignore",
        )
    elif "n_queries_bm25" in merged.columns:
        merged.rename(columns={"n_queries_bm25": "n_queries"}, inplace=True)
    elif "n_queries_rrf" in merged.columns:
        merged.rename(columns={"n_queries_rrf": "n_queries"}, inplace=True)

    out = merged.copy()
    for m in cols:
        cb, cr = f"{m}_bm25", f"{m}_rrf"
        if cb in out.columns and cr in out.columns:
            out[f"{m}_RRF_minus_BM25"] = out[cr] - out[cb]

    # Order columns: query_type, n_queries, pairs per metric
    front = [c for c in ("query_type", "n_queries") if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + sorted(rest)]


def round_metrics_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round float metric columns for display/export."""
    out = df.copy()
    skip = {"query_type", "n_queries", "system"}
    for c in out.columns:
        if c in skip:
            continue
        if out[c].dtype == float or pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return out
