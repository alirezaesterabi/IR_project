"""
Evaluate BM25, dense, and fused run CSVs across query Types 1-6.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def find_root() -> Path:
    marker = "data/raw_data"
    for key in ("IR_PROJECT_ROOT", "VSCODE_WORKSPACE_FOLDER", "CURSOR_WORKSPACE_FOLDER"):
        value = os.environ.get(key)
        if value and (Path(value) / marker).exists():
            return Path(value)
    for path in [Path.cwd()] + list(Path.cwd().parents):
        if (path / marker).exists():
            return path
    fallback = Path(
        "/Users/alireza/Library/CloudStorage/"
        "GoogleDrive-ali.esterabi@gmail.com/My Drive/QMUL_temr_2/IR_project"
    )
    if (fallback / marker).exists():
        return fallback
    raise FileNotFoundError(
        "Cannot find project root. Set IR_PROJECT_ROOT environment variable."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate available retrieval run CSVs across Types 1-6."
    )
    parser.add_argument(
        "--queries-dir",
        type=str,
        default=None,
        help="Directory containing queries_type_*.json (default: data/queries).",
    )
    parser.add_argument(
        "--qrels-dir",
        type=str,
        default=None,
        help="Directory containing qrels_type_*.json (default: data/qrels).",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Directory containing run CSVs (default: results/runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for evaluation CSV outputs (default: results/evaluation).",
    )
    parser.add_argument("--bm25-run", type=str, default=None)
    parser.add_argument("--tfidf-run", type=str, default=None)
    parser.add_argument("--identifier-run", type=str, default=None)
    parser.add_argument("--dense-minilm-run", type=str, default=None)
    parser.add_argument("--dense-bge-m3-run", type=str, default=None)
    parser.add_argument("--rrf-minilm-run", type=str, default=None)
    parser.add_argument("--rrf-bge-m3-run", type=str, default=None)
    parser.add_argument("--rrf-all-run", type=str, default=None)
    parser.add_argument(
        "--min-depth",
        type=int,
        default=10,
        help="Minimum expected retrieved depth for coverage checks (default: 10).",
    )
    return parser.parse_args(argv)


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.evaluation.utils import (
        build_bm25_rrf_comparison_table,
        default_data_paths,
        evaluate_all_types,
        evaluate_overall_types_1_6,
        evaluate_per_query_all_types,
        load_run_csv,
        merge_all_types_qrels,
        merge_all_types_queries,
        round_metrics_df,
        validate_run_coverage,
    )

    queries_dir = Path(args.queries_dir) if args.queries_dir else root / "data" / "queries"
    qrels_dir = Path(args.qrels_dir) if args.qrels_dir else root / "data" / "qrels"
    runs_dir = Path(args.runs_dir) if args.runs_dir else root / "results" / "runs"
    output_dir = Path(args.output_dir) if args.output_dir else root / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    qrel_paths, query_paths = default_data_paths(root)
    if args.queries_dir:
        query_paths = {i: queries_dir / f"queries_type_{i}.json" for i in range(1, 7)}
    if args.qrels_dir:
        qrel_paths = {i: qrels_dir / f"qrels_type_{i}.json" for i in range(1, 7)}

    queries_df = merge_all_types_queries(query_paths)
    qrels = merge_all_types_qrels(qrel_paths)
    all_query_ids = queries_df["query_id"].astype(str).tolist()

    run_paths = {
        "bm25": Path(args.bm25_run) if args.bm25_run else runs_dir / "bm25.csv",
        "tfidf": Path(args.tfidf_run) if args.tfidf_run else runs_dir / "tfidf.csv",
        "identifier": (
            Path(args.identifier_run)
            if args.identifier_run
            else runs_dir / "identifier.csv"
        ),
        "dense_minilm": (
            Path(args.dense_minilm_run)
            if args.dense_minilm_run
            else runs_dir / "dense_minilm.csv"
        ),
        "dense_bge_m3": (
            Path(args.dense_bge_m3_run)
            if args.dense_bge_m3_run
            else runs_dir / "dense_bge_m3.csv"
        ),
        "rrf_minilm": (
            Path(args.rrf_minilm_run)
            if args.rrf_minilm_run
            else runs_dir / "rrf_minilm.csv"
        ),
        "rrf_bge_m3": (
            Path(args.rrf_bge_m3_run)
            if args.rrf_bge_m3_run
            else runs_dir / "rrf_bge_m3.csv"
        ),
        "rrf_all": (
            Path(args.rrf_all_run)
            if args.rrf_all_run
            else runs_dir / "rrf_all.csv"
        ),
    }

    print("=" * 60)
    print("  Evaluate Retrieval Runs")
    print("=" * 60)
    print(f"  Root       : {root}")
    print(f"  Queries    : {queries_dir}")
    print(f"  Qrels      : {qrels_dir}")
    print(f"  Runs dir   : {runs_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Queries    : {len(all_query_ids):,}")
    print()

    evaluated_by_type: dict[str, pd.DataFrame] = {}
    overall_rows: list[dict[str, float | str]] = []
    found_any = False

    for system, path in run_paths.items():
        if not path.exists():
            print(f"[skip] {system}: {path} not found")
            continue

        found_any = True
        run_dict = load_run_csv(path)
        per_type = round_metrics_df(evaluate_all_types(qrels, run_dict, queries_df))
        per_query = round_metrics_df(evaluate_per_query_all_types(qrels, run_dict, queries_df))
        coverage = validate_run_coverage(run_dict, all_query_ids, min_depth=args.min_depth)
        overall = evaluate_overall_types_1_6(qrels, run_dict, queries_df)

        evaluated_by_type[system] = per_type
        overall_rows.append({"system": system, **overall})

        _write_df(per_type, output_dir / f"by_type_{system}.csv")
        _write_df(per_query, output_dir / f"per_query_{system}.csv")
        _write_df(coverage, output_dir / f"coverage_{system}.csv")
        print(f"[ok] {system}: wrote per-type, per-query, and coverage reports")

    if not found_any:
        print("No run CSVs found to evaluate.")
        return 1

    overall_df = round_metrics_df(pd.DataFrame(overall_rows))
    _write_df(overall_df, output_dir / "overall_summary.csv")

    if "bm25" in evaluated_by_type and "rrf_minilm" in evaluated_by_type:
        comp = round_metrics_df(
            build_bm25_rrf_comparison_table(
                evaluated_by_type["bm25"], evaluated_by_type["rrf_minilm"]
            )
        )
        _write_df(comp, output_dir / "comparison_bm25_vs_rrf_minilm.csv")

    if "bm25" in evaluated_by_type and "rrf_bge_m3" in evaluated_by_type:
        comp = round_metrics_df(
            build_bm25_rrf_comparison_table(
                evaluated_by_type["bm25"], evaluated_by_type["rrf_bge_m3"]
            )
        )
        _write_df(comp, output_dir / "comparison_bm25_vs_rrf_bge_m3.csv")

    if "bm25" in evaluated_by_type and "rrf_all" in evaluated_by_type:
        comp = round_metrics_df(
            build_bm25_rrf_comparison_table(
                evaluated_by_type["bm25"], evaluated_by_type["rrf_all"]
            )
        )
        _write_df(comp, output_dir / "comparison_bm25_vs_rrf_all.csv")

    print(f"\nWrote evaluation outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
