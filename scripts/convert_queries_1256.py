"""
Convert potential_queries_type_1_2_5_6.xlsx → data/evaluation/queries_1256.json

Source columns:
    query_id, query_type, query_texts (JSON array), filter_criteria, notes

Usage:
    python scripts/convert_queries_1256.py
    python scripts/convert_queries_1256.py --input path/to/file.xlsx
    python scripts/convert_queries_1256.py --output path/to/output.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def find_root() -> Path:
    for key in ("IR_PROJECT_ROOT", "VSCODE_WORKSPACE_FOLDER", "CURSOR_WORKSPACE_FOLDER"):
        v = os.environ.get(key)
        if v:
            p = Path(v)
            if (p / "data" / "raw_data").exists():
                return p
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "data" / "raw_data").exists():
            return base
    raise FileNotFoundError("Cannot find project root")


def convert(input_path: Path, output_path: Path) -> None:
    print(f"Reading  : {input_path}")

    df = pd.read_excel(input_path, sheet_name="Queries", dtype=str)
    df = df.fillna("")

    required = {"query_id", "query_type", "query_texts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    queries = []
    skipped = 0

    for _, row in df.iterrows():
        raw_texts = row.get("query_texts", "").strip()
        if not raw_texts:
            skipped += 1
            continue

        # Parse the JSON array from the cell
        try:
            query_texts = json.loads(raw_texts)
            if not isinstance(query_texts, list) or not query_texts:
                raise ValueError("empty or non-list")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Warning: could not parse query_texts for {row['query_id']}: {e} — skipping")
            skipped += 1
            continue

        queries.append({
            "query_id":        row["query_id"].strip(),
            "query_type":      int(row["query_type"]),
            "query_texts":     query_texts,
            "filter_criteria": row.get("filter_criteria", "").strip(),
            "notes":           row.get("notes", "").strip(),
        })

    output = {"queries": queries}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Written  : {output_path}")
    print(f"Queries  : {len(queries)} exported, {skipped} skipped")

    from collections import Counter
    type_counts = Counter(q["query_type"] for q in queries)
    for qtype in sorted(type_counts):
        print(f"  Type {qtype}: {type_counts[qtype]} queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert potential_queries_type_1_2_5_6.xlsx → queries_1256.json"
    )
    parser.add_argument("--input",  type=str, default=None,
                        help="Path to input Excel (default: data/evaluation/potential_queries_type_1_2_5_6.xlsx)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output JSON (default: data/evaluation/queries_1256.json)")
    args = parser.parse_args()

    root = find_root()
    input_path  = Path(args.input)  if args.input  else root / "data" / "evaluation" / "potential_queries_type_1_2_5_6.xlsx"
    output_path = Path(args.output) if args.output else root / "data" / "evaluation" / "queries_1256.json"

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    convert(input_path, output_path)
