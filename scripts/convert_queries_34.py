"""
Convert queries_type_3_4_before_pooling.xlsx → data/evaluation/queries_34.json

Source columns (new unified format, one row per query):
    query_id, query_type, query_texts (JSON array), filter_criteria, notes, doc_ids (JSON array)

Usage:
    python scripts/convert_queries_34.py
    python scripts/convert_queries_34.py --input data/evaluation/queries_type_3_4_before_pooling.xlsx
    python scripts/convert_queries_34.py --output path/to/output.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def find_root() -> Path:
    for key in ("IR_PROJECT_ROOT", "VSCODE_WORKSPACE_FOLDER", "CURSOR_WORKSPACE_FOLDER"):
        v = os.environ.get(key)
        if v:
            p = Path(v)
            if (p / "data" / "raw_data").exists():
                return p
    for base in [Path.cwd()] + list(Path.cwd().parents):
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
        qid = row["query_id"].strip()
        if not qid:
            skipped += 1
            continue

        raw_texts = row.get("query_texts", "").strip()
        try:
            query_texts = json.loads(raw_texts)
            if not isinstance(query_texts, list) or not query_texts:
                raise ValueError("empty or non-list")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Warning: could not parse query_texts for {qid}: {e} — skipping")
            skipped += 1
            continue

        raw_doc_ids = row.get("doc_ids", "").strip()
        try:
            doc_ids = json.loads(raw_doc_ids) if raw_doc_ids else []
        except json.JSONDecodeError:
            doc_ids = []

        queries.append({
            "query_id":         qid,
            "query_type":       int(row["query_type"]),
            "query_texts":      query_texts,
            "filter_criteria":  row.get("filter_criteria", "").strip(),
            "notes":            row.get("notes", "").strip(),
            "expected_doc_ids": doc_ids,
            "n_relevant":       len(doc_ids),
        })

    output = {"queries": queries}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Written  : {output_path}")
    print(f"Queries  : {len(queries)} exported, {skipped} skipped")

    type_counts = Counter(q["query_type"] for q in queries)
    for qtype in sorted(type_counts):
        print(f"  Type {qtype}: {type_counts[qtype]} queries")

    total = sum(q["n_relevant"] for q in queries)
    print(f"  Total known relevant entities: {total}")
    print()
    print("Sample output:")
    for q in queries[:3]:
        print(f"  [{q['query_id']}] Type {q['query_type']}: {q['query_texts'][0][:70]}")
        print(f"    → {q['n_relevant']} known entities: {q['expected_doc_ids'][:3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert queries_type_3_4_before_pooling.xlsx → queries_34.json"
    )
    parser.add_argument("--input",  type=str, default=None,
                        help="Path to input Excel (default: data/evaluation/queries_type_3_4_before_pooling.xlsx)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output JSON (default: data/evaluation/queries_34.json)")
    args = parser.parse_args()

    root = find_root()
    input_path  = Path(args.input)  if args.input  else root / "data" / "evaluation" / "queries_type_3_4_before_pooling.xlsx"
    output_path = Path(args.output) if args.output else root / "data" / "evaluation" / "queries_34.json"

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    convert(input_path, output_path)
