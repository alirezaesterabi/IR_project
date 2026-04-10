"""Evaluate Type 7 answers with RAGAS and export stable artifacts."""

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
        description="Evaluate Type 7 answers with RAGAS."
    )
    parser.add_argument("--answers", type=str, required=True)
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--evaluator-model", type=str, default="gpt-oss:20b")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    root = find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.rag.evaluation import build_ragas_rows, evaluate_with_ragas, write_json

    answers_df = load_jsonl(Path(args.answers))
    context_df = load_jsonl(Path(args.context))
    ragas_rows = build_ragas_rows(answers_df, context_df)
    if not ragas_rows:
        raise ValueError("No RAGAS rows were built from the answer/context inputs.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ragas_input_path = output_dir / "ragas_input.json"
    with ragas_input_path.open("w", encoding="utf-8") as handle:
        json.dump(ragas_rows, handle, indent=2, ensure_ascii=False)

    per_query_df, summary = evaluate_with_ragas(
        ragas_rows,
        evaluator_model=args.evaluator_model,
        base_url=args.ollama_url,
    )

    per_query_path = output_dir / "per_query.csv"
    summary_path = output_dir / "summary.json"
    enriched_path = output_dir / "enriched.xlsx"

    per_query_df.to_csv(per_query_path, index=False)
    write_json(summary_path, summary)
    with pd.ExcelWriter(enriched_path) as writer:
        per_query_df.to_excel(writer, sheet_name="per_query", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)

    print("=" * 60)
    print("  Evaluate Type 7 Answers")
    print("=" * 60)
    print(f"  Answers      : {args.answers}")
    print(f"  Context      : {args.context}")
    print(f"  Evaluator    : {args.evaluator_model}")
    print(f"  Queries      : {len(ragas_rows)}")
    print(f"  RAGAS input  : {ragas_input_path}")
    print(f"  Per-query    : {per_query_path}")
    print(f"  Summary      : {summary_path}")
    print(f"  Enriched XLSX: {enriched_path}")
    print()
    print("Aggregate metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
