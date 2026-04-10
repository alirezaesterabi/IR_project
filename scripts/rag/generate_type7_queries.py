"""Generate canonical Type 7 queries from the existing Type 3 and Type 4 sets."""

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
        description="Generate canonical Type 7 narrative queries from Types 3 and 4."
    )
    parser.add_argument("--type3", type=str, default=None)
    parser.add_argument("--type4", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--review-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.rag.query_builder import build_type7_queries, load_query_json

    type3_path = Path(args.type3) if args.type3 else root / "data" / "queries" / "queries_type_3.json"
    type4_path = Path(args.type4) if args.type4 else root / "data" / "queries" / "queries_type_4.json"
    output_path = Path(args.output) if args.output else root / "data" / "queries" / "type7_queries.json"
    review_output = (
        Path(args.review_output)
        if args.review_output
        else root / "data" / "queries" / "type7_queries_review.xlsx"
    )

    payload = build_type7_queries(load_query_json(type3_path), load_query_json(type4_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    pd.DataFrame(payload["queries"]).to_excel(review_output, index=False)

    print("=" * 60)
    print("  Generate Type 7 Queries")
    print("=" * 60)
    print(f"  Type 3       : {type3_path}")
    print(f"  Type 4       : {type4_path}")
    print(f"  Output JSON  : {output_path}")
    print(f"  Review XLSX  : {review_output}")
    print(f"  Queries      : {payload['n_queries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
