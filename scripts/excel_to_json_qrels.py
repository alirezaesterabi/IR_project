"""
Convert potential_queries Excel files → qrels JSON (TREC format).

Reads all  data/evaluation/potential_queries/queries_type_*.xlsx  (or a single
--input file) and writes one qrels JSON per type into data/qrels/.

For query type 6 the expected_doc_ids column in the Excel is only a 20-doc
sample. The full ground truth is derived by scanning documents.jsonl once and
matching filter_criteria (programId / schema / country).

Output format per file (data/qrels/qrels_type_N.json):
{
  "query_type": 1,
  "n_queries": 50,
  "qrels": {
    "Q1_001": {"NK-223CQDBzp8MRkdJMDiqXn3": 1},
    "Q1_002": {"NK-abc456xyz": 1}
  }
}

Usage
-----
  python scripts/excel_to_json_qrels.py
  python scripts/excel_to_json_qrels.py --input data/evaluation/potential_queries/queries_type_6.xlsx
  python scripts/excel_to_json_qrels.py --output-dir data/qrels
  python scripts/excel_to_json_qrels.py --docs data/processed/subset_100k/documents.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
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


def parse_filter_criteria(raw: str) -> dict:
    """Parse 'programId=X | schema=Y | country=Z' into a dict."""
    result = {}
    for part in raw.split("|"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def build_type6_index(docs_path: Path) -> dict:
    """
    Single-pass scan of documents.jsonl.
    Returns {(programId, schema, country): [doc_id, ...]} so all 50 Type 6
    queries can be resolved without re-reading the file.
    """
    index: dict = defaultdict(list)

    if not docs_path.exists():
        print(f"  Warning: {docs_path} not found — Type 6 will fall back to sample doc_ids")
        return index

    print(f"  Building Type-6 index from {docs_path.name} (single pass)…")
    count = 0
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta   = doc.get("metadata", {})
            schema = doc.get("schema", "")
            did    = doc["doc_id"]
            for prog in meta.get("programId", []):
                for ctr in meta.get("country", []):
                    index[(prog, schema, ctr)].append(did)
            count += 1

    print(f"  Indexed {count:,} documents, {len(index):,} (programId, schema, country) combos")
    return index


def convert_file(input_path: Path, output_dir: Path, type6_index: dict) -> None:
    import pandas as pd

    print(f"\nReading : {input_path.name}")

    df = pd.read_excel(input_path, sheet_name="Queries", dtype=str).fillna("")

    required = {"query_id", "query_type", "expected_doc_ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    qrels_out = {}
    skipped = 0

    for _, row in df.iterrows():
        qid = row["query_id"].strip()
        if not qid:
            skipped += 1
            continue

        qtype           = int(row["query_type"])
        filter_criteria = row.get("filter_criteria", "").strip()
        doc_ids_sample  = parse_json_array(row["expected_doc_ids"], "expected_doc_ids", qid)

        if qtype == 6 and filter_criteria and type6_index:
            criteria = parse_filter_criteria(filter_criteria)
            prog     = criteria.get("programId", "")
            schema   = criteria.get("schema", "")
            ctr      = criteria.get("country", "")
            doc_ids  = type6_index.get((prog, schema, ctr), doc_ids_sample)
        else:
            doc_ids = doc_ids_sample

        if doc_ids:
            qrels_out[qid] = {did: 1 for did in doc_ids}
        else:
            skipped += 1

    stem = input_path.stem
    parts = stem.split("_")
    qtype_num = int(parts[-1]) if parts[-1].isdigit() else 0
    suffix = f"type_{qtype_num}" if qtype_num else stem

    out_path = output_dir / f"qrels_{suffix}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"query_type": qtype_num, "n_queries": len(qrels_out), "qrels": qrels_out},
            f, indent=2, ensure_ascii=False,
        )

    total_relevant = sum(len(v) for v in qrels_out.values())
    print(f"  Exported : {len(qrels_out)} queries, {total_relevant} total relevant docs, {skipped} skipped  → {out_path.name}")
    if qrels_out:
        sample_qid = next(iter(qrels_out))
        sample_docs = list(qrels_out[sample_qid].keys())[:3]
        print(f"  Sample   : [{sample_qid}] → {len(qrels_out[sample_qid])} relevant: {sample_docs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert potential_queries Excel files → qrels JSON (TREC format)"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Single Excel file (default: all queries_type_*.xlsx)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/qrels/)")
    parser.add_argument("--docs", type=str, default=None,
                        help="Path to documents.jsonl for Type-6 expansion (default: auto-detect)")
    args = parser.parse_args()

    root = find_root()
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "qrels"

    if args.docs:
        docs_path = Path(args.docs)
    else:
        docs_path = root / "data" / "processed" / "full" / "documents.jsonl"
        if not docs_path.exists():
            docs_path = root / "data" / "processed" / "subset_100k" / "documents.jsonl"

    if args.input:
        input_files = [Path(args.input)]
    else:
        potential_dir = root / "data" / "evaluation" / "potential_queries"
        input_files = sorted(potential_dir.glob("queries_type_*.xlsx"))
        if not input_files:
            print(f"No queries_type_*.xlsx files found in {potential_dir}")
            sys.exit(1)

    print(f"Root       : {root}")
    print(f"Dataset    : {docs_path}")
    print(f"Output dir : {output_dir}")
    print(f"Files      : {len(input_files)}")

    needs_type6 = any("type_6" in f.stem for f in input_files) or not args.input
    type6_index = build_type6_index(docs_path) if needs_type6 else {}

    for f in input_files:
        convert_file(f, output_dir, type6_index)

    print("\nAll done.")


if __name__ == "__main__":
    main()
