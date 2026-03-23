"""
Convert queries_reviewed.xlsx → data/evaluation/queries.json

Usage:
    python scripts/excel_to_json.py
    python scripts/excel_to_json.py --input data/evaluation/queries_draft.xlsx
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

    required = {"query_id", "query_type", "query_text", "expected_difficulty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    queries = []
    skipped = 0
    for _, row in df.iterrows():
        if not row.get("query_text", "").strip():
            skipped += 1
            continue
        queries.append({
            "query_id":            row["query_id"].strip(),
            "query_type":          int(row["query_type"]) if row["query_type"] else 0,
            "query_text":          row["query_text"].strip(),
            "expected_difficulty": row.get("expected_difficulty", "").strip(),
            "expected_doc_id":     row.get("expected_doc_id", "").strip(),
            "notes":               row.get("notes", "").strip(),
        })

    output = {"queries": queries}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Written  : {output_path}")
    print(f"Queries  : {len(queries)} exported, {skipped} skipped (empty query_text)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert queries Excel → JSON")
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to Excel file (default: data/evaluation/queries_reviewed.xlsx)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to output JSON (default: data/evaluation/queries.json)"
    )
    args = parser.parse_args()

    root = find_root()
    input_path  = Path(args.input)  if args.input  else root / "data" / "evaluation" / "queries_reviewed.xlsx"
    output_path = Path(args.output) if args.output else root / "data" / "evaluation" / "queries.json"

    if not input_path.exists():
        # Fall back to draft if reviewed doesn't exist yet
        draft = root / "data" / "evaluation" / "queries_draft.xlsx"
        if draft.exists():
            print(f"Note: reviewed file not found, using draft: {draft}")
            input_path = draft
        else:
            print(f"Error: {input_path} not found.")
            sys.exit(1)

    convert(input_path, output_path)
