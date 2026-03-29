"""
Convert potential_queries Excel files → queries JSON + qrels JSON.

Reads all  data/evaluation/potential_queries/queries_type_*.xlsx  (or a single
--input file) and writes two output files per type:

  data/queries/queries_type_N.json   — query_id / query_texts / filter_criteria / notes
  data/qrels/qrels_type_N.json       — TREC-style relevance judgements {query_id: {doc_id: 1}}

For query type 6 the expected_doc_ids column in the Excel is only a 20-doc
sample.  The full ground truth is derived at runtime by scanning documents.jsonl
and matching the filter_criteria (programId / schema / country).

Usage
-----
  # Convert all 6 types (default)
  python scripts/excel_to_json_queries.py

  # Convert a single file
  python scripts/excel_to_json_queries.py --input data/evaluation/potential_queries/queries_type_3.xlsx

  # Override output directories
  python scripts/excel_to_json_queries.py --queries-dir data/queries --qrels-dir data/qrels
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


# ── Project root detection ────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_json_array(raw: str, field: str, query_id: str) -> list:
    """Parse a JSON array stored as a string in an Excel cell."""
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
    if not raw:
        return result
    for part in raw.split("|"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def build_type6_index(docs_path: Path) -> dict:
    """
    Single-pass scan of documents.jsonl.
    Returns a dict keyed by (programId, schema, country) → [doc_id, ...]
    so that all 50 Type 6 queries can be resolved without re-reading the file.
    """
    from collections import defaultdict
    index: dict = defaultdict(list)

    if not docs_path.exists():
        print(f"  Warning: {docs_path} not found — Type 6 qrels will use sample doc_ids only")
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


def expand_type6_qrels(
    doc_ids_sample: list,
    criteria: dict,
    index: dict,
) -> list:
    """
    Look up ALL doc_ids matching programId + schema + country using the pre-built index.
    Falls back to the sample if the index is empty (file not found).
    """
    if not index:
        return doc_ids_sample

    prog   = criteria.get("programId", "")
    schema = criteria.get("schema", "")
    ctr    = criteria.get("country", "")
    return index.get((prog, schema, ctr), doc_ids_sample)


# ── Core conversion ───────────────────────────────────────────────────────────

def convert_file(
    input_path: Path,
    queries_dir: Path,
    qrels_dir: Path,
    type6_index: dict,
) -> None:
    import pandas as pd

    print(f"\nReading : {input_path.name}")

    df = pd.read_excel(input_path, sheet_name="Queries", dtype=str)
    df = df.fillna("")

    required = {"query_id", "query_type", "query_texts", "expected_doc_ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    queries_out = []
    qrels_out   = {}
    skipped     = 0

    for _, row in df.iterrows():
        qid  = row["query_id"].strip()
        if not qid:
            skipped += 1
            continue

        query_texts = parse_json_array(row["query_texts"], "query_texts", qid)
        if not query_texts:
            print(f"  Skipping {qid}: empty query_texts")
            skipped += 1
            continue

        qtype            = int(row["query_type"])
        filter_criteria  = row.get("filter_criteria", "").strip()
        notes            = row.get("notes", "").strip()
        doc_ids_sample   = parse_json_array(row["expected_doc_ids"], "expected_doc_ids", qid)

        # ── Queries entry (no doc IDs) ─────────────────────────────────────
        queries_out.append({
            "query_id":       qid,
            "query_type":     qtype,
            "query_texts":    query_texts,
            "filter_criteria": filter_criteria,
            "notes":          notes,
        })

        # ── Qrels entry ────────────────────────────────────────────────────
        if qtype == 6 and filter_criteria:
            criteria = parse_filter_criteria(filter_criteria)
            doc_ids  = expand_type6_qrels(doc_ids_sample, criteria, type6_index)
        else:
            doc_ids  = doc_ids_sample

        if doc_ids:
            qrels_out[qid] = {did: 1 for did in doc_ids}

    # Derive type number from filename (e.g. queries_type_3.xlsx → 3)
    stem  = input_path.stem          # queries_type_3
    parts = stem.split("_")
    qtype_num = int(parts[-1]) if parts[-1].isdigit() else 0
    suffix = f"type_{qtype_num}" if qtype_num else stem

    # ── Write queries JSON ─────────────────────────────────────────────────
    queries_path = queries_dir / f"queries_{suffix}.json"
    queries_payload = {
        "query_type": qtype_num,
        "n_queries":  len(queries_out),
        "queries":    queries_out,
    }
    queries_dir.mkdir(parents=True, exist_ok=True)
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries_payload, f, indent=2, ensure_ascii=False)

    # ── Write qrels JSON ───────────────────────────────────────────────────
    qrels_path = qrels_dir / f"qrels_{suffix}.json"
    qrels_payload = {
        "query_type": qtype_num,
        "n_queries":  len(qrels_out),
        "qrels":      qrels_out,
    }
    qrels_dir.mkdir(parents=True, exist_ok=True)
    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump(qrels_payload, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────────────────────
    total_relevant = sum(len(v) for v in qrels_out.values())
    print(f"  Queries : {len(queries_out)} exported, {skipped} skipped  → {queries_path.name}")
    print(f"  Qrels   : {len(qrels_out)} queries, {total_relevant} total relevant docs  → {qrels_path.name}")

    # Sample
    sample_q = queries_out[0]
    sample_qid = sample_q["query_id"]
    sample_docs = list(qrels_out.get(sample_qid, {}).keys())[:3]
    print(f"  Sample  : [{sample_qid}] {sample_q['query_texts'][0][:60]}")
    print(f"            → {len(qrels_out.get(sample_qid, {}))} relevant: {sample_docs}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert potential_queries Excel files → queries JSON + qrels JSON"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Single Excel file to convert (default: all queries_type_*.xlsx in potential_queries/)",
    )
    parser.add_argument(
        "--queries-dir", type=str, default=None,
        help="Output directory for queries JSON (default: data/queries/)",
    )
    parser.add_argument(
        "--qrels-dir", type=str, default=None,
        help="Output directory for qrels JSON (default: data/qrels/)",
    )
    args = parser.parse_args()

    root = find_root()

    queries_dir = Path(args.queries_dir) if args.queries_dir else root / "data" / "queries"
    qrels_dir   = Path(args.qrels_dir)   if args.qrels_dir   else root / "data" / "qrels"
    docs_path   = root / "data" / "processed" / "full" / "documents.jsonl"
    if not docs_path.exists():
        docs_path = root / "data" / "processed" / "subset_100k" / "documents.jsonl"

    if args.input:
        input_files = [Path(args.input)]
    else:
        potential_dir = root / "data" / "evaluation" / "potential_queries"
        input_files   = sorted(potential_dir.glob("queries_type_*.xlsx"))
        if not input_files:
            print(f"No queries_type_*.xlsx files found in {potential_dir}")
            sys.exit(1)

    print(f"Root        : {root}")
    print(f"Dataset     : {docs_path}")
    print(f"Queries out : {queries_dir}")
    print(f"Qrels out   : {qrels_dir}")
    print(f"Files       : {len(input_files)}")

    # Build Type-6 index once (single pass over documents.jsonl)
    needs_type6 = any("type_6" in f.stem for f in input_files) or (
        not args.input  # processing all files
    )
    type6_index = build_type6_index(docs_path) if needs_type6 else {}

    for f in input_files:
        convert_file(f, queries_dir, qrels_dir, type6_index)

    print("\nAll done.")


if __name__ == "__main__":
    main()
