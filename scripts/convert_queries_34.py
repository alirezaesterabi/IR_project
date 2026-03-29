"""
Convert potential_queries_tyoe_3_4.xlsx → data/evaluation/queries_34.json

Source columns:
    ID, Theme, Type, Query per OpenSanctions, Query Text, Entity ID_URL, Entity ID, Entity Name

The file has one row per relevant entity — multiple rows share the same query ID.
This script groups them and produces one JSON entry per unique query.

Usage:
    python scripts/convert_queries_34.py
    python scripts/convert_queries_34.py --input data/evaluation/queries_type_3_4_before_pooling.xlsx
    python scripts/convert_queries_34.py --output path/to/output.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
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


def parse_query_type(raw: str) -> int:
    """Extract integer type from strings like '3 - Semantic / Descriptive'."""
    try:
        return int(str(raw).strip().split()[0])
    except (ValueError, IndexError):
        return 0


def convert(input_path: Path, output_path: Path) -> None:
    print(f"Reading  : {input_path}")

    df = pd.read_excel(input_path, dtype=str)
    df = df.fillna("")

    required = {"ID", "Type", "Query Text", "Entity ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Group rows by query ID — one row per entity, many rows per query
    query_map: dict = defaultdict(lambda: {
        "query_type":       None,
        "query_text":       None,
        "theme":            None,
        "reference_url":    None,
        "expected_doc_ids": [],
        "entity_names":     [],
    })

    for _, row in df.iterrows():
        qid = row["ID"].strip()
        if not qid:
            continue

        q = query_map[qid]

        # Set query-level fields from first row seen (they repeat across rows)
        if q["query_text"] is None:
            q["query_type"]    = parse_query_type(row["Type"])
            q["query_text"]    = row["Query Text"].strip()
            q["theme"]         = row.get("Theme", "").strip()
            q["reference_url"] = row.get("Query per OpenSanctions", "").strip()

        # Collect entity IDs (ground truth) — deduplicate
        entity_id   = row["Entity ID"].strip()
        entity_name = row.get("Entity Name", "").strip()

        if entity_id and entity_id not in q["expected_doc_ids"]:
            q["expected_doc_ids"].append(entity_id)
            q["entity_names"].append(entity_name)

    # Build output list
    queries = []
    skipped = 0
    for qid, q in query_map.items():
        if not q["query_text"]:
            skipped += 1
            continue
        queries.append({
            "query_id":          qid,
            "query_type":        q["query_type"],
            "query_text":        q["query_text"],
            "theme":             q["theme"],
            "reference_url":     q["reference_url"],
            "expected_doc_ids":  q["expected_doc_ids"],
            "n_relevant":        len(q["expected_doc_ids"]),
        })

    output = {"queries": queries}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Written  : {output_path}")
    print(f"Queries  : {len(queries)} exported, {skipped} skipped (empty query_text)")

    from collections import Counter
    type_counts = Counter(q["query_type"] for q in queries)
    for qtype in sorted(type_counts):
        print(f"  Type {qtype}: {type_counts[qtype]} queries")

    total_entities = sum(q["n_relevant"] for q in queries)
    print(f"  Total known relevant entities across all queries: {total_entities}")
    print()
    print("Sample output:")
    for q in queries[:3]:
        print(f"  [{q['query_id']}] Type {q['query_type']}: {q['query_text'][:70]}")
        print(f"    → {q['n_relevant']} known entities: {q['expected_doc_ids'][:3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert potential_queries_tyoe_3_4.xlsx → queries_34.json"
    )
    parser.add_argument("--input",  type=str, default=None,
                        help="Path to input Excel (default: data/evaluation/potential_queries_tyoe_3_4.xlsx)")
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
