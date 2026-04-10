"""Generate Type 7 grounded answers from prepared context rows."""

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
        description="Generate Type 7 answers from prepared RAG context."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to context_*.jsonl.")
    parser.add_argument("--model", type=str, default="gpt-oss:20b")
    parser.add_argument("--prompt-version", type=str, default="v1")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--raw-output", type=str, default=None)
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    root = find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.rag.generator import generate_answer_records, records_to_dataframe

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else root / "results" / "rag" / f"answers_{input_path.stem.replace('context_', '')}.jsonl"
    )
    raw_output_path = (
        Path(args.raw_output)
        if args.raw_output
        else root / "results" / "rag" / f"raw_responses_{input_path.stem.replace('context_', '')}.json"
    )
    xlsx_path = output_path.with_suffix(".xlsx")

    context_df = load_jsonl(input_path)
    records = generate_answer_records(
        context_df,
        model=args.model,
        prompt_version=args.prompt_version,
        base_url=args.ollama_url,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )
    answer_df = records_to_dataframe(records)

    write_jsonl(output_path, records)
    answer_df.to_excel(xlsx_path, index=False)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("  Generate Type 7 Answers")
    print("=" * 60)
    print(f"  Input        : {input_path}")
    print(f"  Model        : {args.model}")
    print(f"  Prompt ver   : {args.prompt_version}")
    print(f"  Queries      : {len(answer_df):,}")
    print(f"  Output JSONL : {output_path}")
    print(f"  Output XLSX  : {xlsx_path}")
    print(f"  Raw output   : {raw_output_path}")
    print()
    status_counts = answer_df["generation_status"].value_counts(dropna=False).to_dict()
    print(f"  Status counts: {status_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
