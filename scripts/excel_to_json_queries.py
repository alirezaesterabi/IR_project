"""
Convert potential_queries Excel files → queries JSON.

Reads all  data/evaluation/potential_queries/queries_type_*.xlsx  (or a single
--input file) and writes one queries JSON per type into data/queries/.

Output format per file (data/queries/queries_type_N.json):
{
  "query_type": 1,
  "n_queries": 50,
  "queries": [
    {
      "query_id": "Q1_001",
      "query_type": 1,
      "query_texts": ["103919088"],
      "filter_criteria": "",
      "notes": "registrationNumber | ..."
    }
  ]
}

Usage
-----
  python scripts/excel_to_json_queries.py
  python scripts/excel_to_json_queries.py --input data/potential_queries/queries_type_3.xlsx
  python scripts/excel_to_json_queries.py --output-dir data/queries
"""

import argparse
import json
import os
import sys
from pathlib import Path


def find_root() -> Path:
    marker = "data/raw_data"
    for key in ("IR_PROJECT_ROOT", "VSCODE_WORKSPACE_FOLDER", "CURSOR_WORKSPACE_FOLDER"):
        v = os.environ.get(key)
        if v and (Path(v) / marker).exists():
            return Path(v)
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / marker).exists():
            return p
    fallback = Path(
        "/Users/alireza/Library/CloudStorage/"
        "GoogleDrive-ali.esterabi@gmail.com/My Drive/QMUL_temr_2/IR_project"
    )
    if (fallback / marker).exists():
        return fallback
    raise FileNotFoundError(
        "Cannot find project root. Set IR_PROJECT_ROOT environment variable."
    )


def parse_json_array(raw: str, field: str, query_id: str) -> list:
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        result = json.loads(raw)
        if not isinstance(result, list):
            raise ValueError("not a list")
        return [str(x) for x in result if str(x).strip()]
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Warning [{query_id}] could not parse {field}: {e} — treating as empty")
        return []


def convert_file(input_path: Path, output_dir: Path) -> None:
    import pandas as pd

    print(f"\nReading : {input_path.name}")

    df = pd.read_excel(input_path, sheet_name="Queries", dtype=str).fillna("")

    required = {"query_id", "query_type", "query_texts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    queries_out = []
    skipped = 0

    for _, row in df.iterrows():
        qid = row["query_id"].strip()
        if not qid:
            skipped += 1
            continue

        query_texts = parse_json_array(row["query_texts"], "query_texts", qid)
        if not query_texts:
            print(f"  Skipping {qid}: empty query_texts")
            skipped += 1
            continue

        queries_out.append({
            "query_id":        qid,
            "query_type":      int(row["query_type"]),
            "query_texts":     query_texts,
            "filter_criteria": row.get("filter_criteria", "").strip(),
            "notes":           row.get("notes", "").strip(),
        })

    stem = input_path.stem                          # queries_type_3
    parts = stem.split("_")
    qtype_num = int(parts[-1]) if parts[-1].isdigit() else 0
    suffix = f"type_{qtype_num}" if qtype_num else stem

    out_path = output_dir / f"queries_{suffix}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"query_type": qtype_num, "n_queries": len(queries_out), "queries": queries_out},
            f, indent=2, ensure_ascii=False,
        )

    print(f"  Exported : {len(queries_out)} queries, {skipped} skipped  → {out_path.name}")
    if queries_out:
        q = queries_out[0]
        print(f"  Sample   : [{q['query_id']}] {q['query_texts'][0][:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert potential_queries Excel files → queries JSON"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Single Excel file (default: all queries_type_*.xlsx)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/queries/)")
    args = parser.parse_args()

    root = find_root()
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "queries"

    if args.input:
        input_files = [Path(args.input)]
    else:
        potential_dir = root / "data" / "potential_queries"
        input_files = sorted(potential_dir.glob("queries_type_*.xlsx"))
        if not input_files:
            print(f"No queries_type_*.xlsx files found in {potential_dir}")
            sys.exit(1)

    print(f"Root       : {root}")
    print(f"Output dir : {output_dir}")
    print(f"Files      : {len(input_files)}")

    for f in input_files:
        convert_file(f, output_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
