"""Prepare Type 7 RAG context from canonical RRF outputs."""

from __future__ import annotations

import argparse
import json
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
    raise FileNotFoundError("Cannot find project root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Type 7 context rows from RRF retrieval results."
    )
    parser.add_argument("--run", type=str, default=None, help="Path to rrf_*.csv.")
    parser.add_argument("--docs", type=str, default=None, help="Path to documents.jsonl.")
    parser.add_argument("--queries", type=str, default=None, help="Path to type7_queries.json.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-tag", type=str, default="latest")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def write_jsonl(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    root = find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.rag.context_builder import (
        build_context_rows,
        load_document_lookup,
        load_run_csv,
        load_type7_queries,
    )

    run_path = Path(args.run) if args.run else root / "results" / "runs" / "rrf_bge_m3.csv"
    docs_path = (
        Path(args.docs)
        if args.docs
        else root / "data" / "json_format_data" / "subset" / "documents.jsonl"
    )
    queries_path = (
        Path(args.queries)
        if args.queries
        else root / "data" / "queries" / "type7_queries.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else root / "results" / "rag"
    output_dir.mkdir(parents=True, exist_ok=True)

    queries_df = load_type7_queries(queries_path)
    run_df = load_run_csv(run_path)
    needed_doc_ids = set(
        run_df[run_df["query_id"].isin(queries_df["source_query_id"])]["doc_id"].astype(str)
    )
    docs_lookup = load_document_lookup(docs_path, needed_doc_ids)
    context_df = build_context_rows(queries_df, run_df, docs_lookup, top_k=args.top_k)

    jsonl_path = output_dir / f"context_{args.run_tag}.jsonl"
    xlsx_path = output_dir / f"context_{args.run_tag}.xlsx"
    write_jsonl(jsonl_path, context_df)
    context_df.to_excel(xlsx_path, index=False)

    print("=" * 60)
    print("  Prepare Type 7 Context")
    print("=" * 60)
    print(f"  Run CSV      : {run_path}")
    print(f"  Docs         : {docs_path}")
    print(f"  Queries      : {queries_path}")
    print(f"  Top-k        : {args.top_k}")
    print(f"  Rows         : {len(context_df):,}")
    print(f"  Queries      : {context_df['query_id'].nunique()}")
    print(f"  JSONL output : {jsonl_path}")
    print(f"  XLSX output  : {xlsx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
