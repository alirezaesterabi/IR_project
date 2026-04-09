"""
Export dense retrieval runs for Types 1-6 queries.

Writes one CSV per model using the shared run schema expected by fusion and
evaluation:
    query_id, doc_id, rank, dense_score
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable


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
        description="Export dense retrieval runs for Types 1-6 queries."
    )
    parser.add_argument(
        "--model",
        choices=["minilm", "bge-m3", "all"],
        default="minilm",
        help="Dense model to export (default: minilm).",
    )
    parser.add_argument(
        "--run-tag",
        required=True,
        help="RUN_TAG matching the dense Chroma collection (e.g. FULL_20260407).",
    )
    parser.add_argument(
        "--queries-dir",
        type=str,
        default=None,
        help="Directory containing queries_type_*.json (default: data/queries).",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=None,
        help="Chroma persistence directory (default: chroma_db/).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Models directory used for dense path resolution (default: models/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for dense run CSVs (default: results/runs/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output CSV path. Only valid when exporting a single model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of ranked documents to export per query (default: 100).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Encoding device for query-time model loading (default: auto).",
    )
    return parser.parse_args(argv)


def iter_query_rows(queries_df) -> Iterable[dict[str, str]]:
    for row in queries_df.itertuples(index=False):
        yield {
            "query_id": str(row.query_id),
            "query_type": str(row.query_type),
            "query_text": str(row.query_text),
        }


def export_dense_run(
    *,
    model_key: str,
    run_tag: str,
    output_path: Path,
    queries_df,
    chroma_dir: Path,
    models_dir: Path,
    top_k: int,
    device: str | None = None,
) -> None:
    from src.retrieval import DenseRetriever

    retriever = DenseRetriever(
        model_key=model_key,
        run_tag=run_tag,
        chroma_dir=chroma_dir,
        models_dir=models_dir,
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query_type",
                "doc_id",
                "rank",
                "dense_score",
                "model",
                "run_tag",
            ],
        )
        writer.writeheader()
        for row in iter_query_rows(queries_df):
            hits = retriever.search(row["query_text"], k=top_k)
            for rank, (doc_id, score, _metadata) in enumerate(hits, start=1):
                writer.writerow(
                    {
                        "query_id": row["query_id"],
                        "query_type": row["query_type"],
                        "doc_id": doc_id,
                        "rank": rank,
                        "dense_score": f"{score:.9f}",
                        "model": model_key,
                        "run_tag": run_tag,
                    }
                )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.evaluation.utils import default_data_paths, merge_all_types_queries

    queries_dir = Path(args.queries_dir) if args.queries_dir else root / "data" / "queries"
    chroma_dir = Path(args.chroma_dir) if args.chroma_dir else root / "chroma_db"
    models_dir = Path(args.models_dir) if args.models_dir else root / "models"
    output_dir = Path(args.output_dir) if args.output_dir else root / "results" / "runs"

    _, query_paths = default_data_paths(root)
    if args.queries_dir:
        query_paths = {
            i: queries_dir / f"queries_type_{i}.json"
            for i in range(1, 7)
        }

    queries_df = merge_all_types_queries(query_paths)
    model_keys = ["minilm", "bge-m3"] if args.model == "all" else [args.model]

    if args.output and len(model_keys) != 1:
        raise ValueError("--output can only be used when exporting a single model.")

    print("=" * 60)
    print("  Export Dense Runs")
    print("=" * 60)
    print(f"  Root       : {root}")
    print(f"  Queries    : {queries_dir}")
    print(f"  Chroma dir : {chroma_dir}")
    print(f"  Models dir : {models_dir}")
    print(f"  Run tag    : {args.run_tag}")
    print(f"  Models     : {', '.join(model_keys)}")
    print(f"  Queries    : {len(queries_df):,}")
    print(f"  Top-k      : {args.top_k}")
    print()

    for model_key in model_keys:
        output_path = Path(args.output) if args.output else output_dir / f"dense_{model_key.replace('-', '_')}.csv"
        print(f"[{model_key}] -> {output_path}")
        export_dense_run(
            model_key=model_key,
            run_tag=args.run_tag,
            output_path=output_path,
            queries_df=queries_df,
            chroma_dir=chroma_dir,
            models_dir=models_dir,
            top_k=args.top_k,
            device=args.device,
        )
        print(f"  wrote {output_path}")

    print("\nDense run export complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
