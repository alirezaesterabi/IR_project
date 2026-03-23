"""
End-to-end preprocessing pipeline for OpenSanctions targets.nested.json.

Orchestrates: stream_records → build_document → write JSONL + stats.json

Typical usage
-------------
From the project root:

    python -m src.preprocessing.pipeline

Or from a notebook/script:

    from src.preprocessing.pipeline import run_pipeline
    from pathlib import Path

    run_pipeline(
        input_path=Path("data/raw_data/targets.nested.json"),
        output_dir=Path("data/processed/subset_100k"),
        max_records=100_000,
    )
"""

import json
import time
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .parser import stream_records, _find_project_root
from .text_processing import TextProcessor
from .document_builder import build_document


def run_pipeline(
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
    show_progress: bool = True,
    batch_log_interval: int = 10_000,
) -> dict:
    """
    Stream, preprocess, and write processed documents to output_dir.

    Parameters
    ----------
    input_path : Path, optional
        Source JSONL file. Defaults to data/raw_data/targets.nested.json.
    output_dir : Path, optional
        Output directory. Defaults to data/processed/subset_{max_records}.
        Will be created if it does not exist.
    max_records : int, optional
        Process at most this many records. None = full dataset.
    show_progress : bool
        Show tqdm progress bar (if tqdm installed).
    batch_log_interval : int
        Print a progress line every N records (fallback if no tqdm).

    Returns
    -------
    dict
        Stats summary (also written to output_dir/stats.json).

    Output files
    ------------
    output_dir/documents.jsonl  — one processed document per line
    output_dir/stats.json       — processing statistics
    """
    root = _find_project_root()

    if input_path is None:
        input_path = root / "data" / "raw_data" / "targets.nested.json"
    input_path = Path(input_path)

    if output_dir is None:
        label = f"subset_{max_records // 1000}k" if max_records else "full"
        output_dir = root / "data" / "processed" / label
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_path = output_dir / "documents.jsonl"
    stats_path = output_dir / "stats.json"

    # ------------------------------------------------------------------
    # Initialise shared TextProcessor (loads spaCy once for all docs)
    # ------------------------------------------------------------------
    print("[pipeline] Loading TextProcessor (spaCy + NLTK)...")
    processor = TextProcessor()
    print("[pipeline] TextProcessor ready.\n")

    # ------------------------------------------------------------------
    # Streaming loop
    # ------------------------------------------------------------------
    stats: dict = {
        "source_file":    str(input_path),
        "output_dir":     str(output_dir),
        "max_records":    max_records,
        "total_processed": 0,
        "schema_counts":  {},
        "avg_token_count": 0.0,
        "empty_blob_count": 0,
        "has_identifiers": 0,
        "elapsed_seconds": 0.0,
    }

    schema_counts: Counter = Counter()
    total_tokens = 0
    empty_blobs = 0
    has_identifiers = 0

    start = time.time()
    processed = 0

    print(f"[pipeline] Writing to {docs_path}")
    print(f"[pipeline] Processing {'all' if max_records is None else f'{max_records:,}'} records...\n")

    with open(docs_path, "w", encoding="utf-8") as out_f:
        for record in stream_records(input_path, max_records=max_records,
                                     show_progress=show_progress):
            doc = build_document(record, processor)

            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            # Accumulate stats
            schema_counts[doc["schema"]] += 1
            total_tokens += len(doc["tokens"])
            if not doc["text_blob"]:
                empty_blobs += 1
            if doc["identifiers"]:
                has_identifiers += 1

            processed += 1

            # Fallback progress logging when tqdm is absent
            if not show_progress and processed % batch_log_interval == 0:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  {processed:>10,} records  |  {rate:>7,.0f} rec/s  |  "
                      f"{elapsed:>6.1f}s elapsed")

    elapsed = time.time() - start

    # ------------------------------------------------------------------
    # Compile and write stats
    # ------------------------------------------------------------------
    stats["total_processed"]  = processed
    stats["schema_counts"]    = dict(schema_counts.most_common())
    stats["avg_token_count"]  = round(total_tokens / processed, 2) if processed else 0
    stats["empty_blob_count"] = empty_blobs
    stats["has_identifiers"]  = has_identifiers
    stats["elapsed_seconds"]  = round(elapsed, 2)
    stats["records_per_second"] = round(processed / elapsed, 1) if elapsed > 0 else 0

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n[pipeline] ✓ Done.")
    print(f"  Records processed : {processed:,}")
    print(f"  Elapsed           : {elapsed:.1f}s  ({stats['records_per_second']:,.0f} rec/s)")
    print(f"  Avg tokens/doc    : {stats['avg_token_count']}")
    print(f"  Empty text_blobs  : {empty_blobs}")
    print(f"  With identifiers  : {has_identifiers:,}")
    print(f"  Schema breakdown  :")
    for schema, count in schema_counts.most_common():
        pct = count / processed * 100
        print(f"    {schema:<20} {count:>10,}  ({pct:.1f}%)")
    print(f"\n  Output → {docs_path}")
    print(f"  Stats  → {stats_path}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline on targets.nested.json"
    )
    parser.add_argument(
        "--max-records", type=int, default=100_000,
        help="Number of records to process (default: 100000, 0 = all)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/processed/subset_<N>k)"
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bar (print log lines instead)"
    )
    args = parser.parse_args()

    max_rec = args.max_records if args.max_records > 0 else None
    out_dir = Path(args.output_dir) if args.output_dir else None

    run_pipeline(
        output_dir=out_dir,
        max_records=max_rec,
        show_progress=not args.no_progress,
    )
