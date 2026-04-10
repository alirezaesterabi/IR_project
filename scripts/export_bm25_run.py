"""
Export BM25 retrieval runs for Types 1-6 queries.

Writes a CSV using the shared run schema expected by fusion and evaluation:
    query_id, doc_id, rank, bm25_score
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Iterable

TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


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


def query_tokens_bm25(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BM25 retrieval runs for Types 1-6 queries."
    )
    parser.add_argument(
        "--queries-dir",
        type=str,
        default=None,
        help="Directory containing queries_type_*.json (default: data/queries).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing the BM25 index (default: models/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/runs/bm25.csv).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of ranked documents to export per query (default: 100).",
    )
    return parser.parse_args(argv)


def iter_query_rows(queries_df) -> Iterable[dict[str, str]]:
    for row in queries_df.itertuples(index=False):
        yield {
            "query_id": str(row.query_id),
            "query_type": str(row.query_type),
            "query_text": str(row.query_text),
        }


def export_bm25_run(*, output_path: Path, queries_df, models_dir: Path, top_k: int) -> None:
    from src.retrieval.classical_ir import BM25Retriever
    from src.preprocessing.text_processing import TextProcessor

    tp = TextProcessor()
    bm25 = BM25Retriever()
    bm25.load(models_dir / "bm25")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_id", "query_type", "doc_id", "rank", "bm25_score"],
        )
        writer.writeheader()
        for row in iter_query_rows(queries_df):
            normalised = tp.normalize(row["query_text"])
            hits = bm25.search(query_tokens_bm25(normalised), k=top_k)
            for rank, (doc_id, score) in enumerate(hits, start=1):
                writer.writerow(
                    {
                        "query_id": row["query_id"],
                        "query_type": row["query_type"],
                        "doc_id": doc_id,
                        "rank": rank,
                        "bm25_score": f"{score:.9f}",
                    }
                )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.evaluation.utils import default_data_paths, merge_all_types_queries

    queries_dir = Path(args.queries_dir) if args.queries_dir else root / "data" / "queries"
    models_dir = Path(args.models_dir) if args.models_dir else root / "models"
    output_path = Path(args.output) if args.output else root / "results" / "runs" / "bm25.csv"

    _, query_paths = default_data_paths(root)
    if args.queries_dir:
        query_paths = {i: queries_dir / f"queries_type_{i}.json" for i in range(1, 7)}
    queries_df = merge_all_types_queries(query_paths)

    print("=" * 60)
    print("  Export BM25 Run")
    print("=" * 60)
    print(f"  Root       : {root}")
    print(f"  Queries    : {queries_dir}")
    print(f"  Models dir : {models_dir}")
    print(f"  Output     : {output_path}")
    print(f"  Queries    : {len(queries_df):,}")
    print(f"  Top-k      : {args.top_k}")
    print()

    export_bm25_run(
        output_path=output_path,
        queries_df=queries_df,
        models_dir=models_dir,
        top_k=args.top_k,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
