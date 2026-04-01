"""
Build and save classical IR indices (BM25, TF-IDF, and Identifier).

Reads documents.jsonl, fits all models, and writes artefacts to models/.

Usage
-----
  # Build all models (default — full dataset)
  python scripts/build_index.py

  # Build only BM25
  python scripts/build_index.py --model bm25

  # Build only TF-IDF
  python scripts/build_index.py --model tfidf

  # Build only Identifier
  python scripts/build_index.py --model identifier

  # Quick test on subset
  python scripts/build_index.py --docs data/json_format_data/subset_100k/documents.jsonl

Saved artefacts
---------------
  models/bm25/index.pkl            — BM25Okapi object
  models/bm25/doc_ids.json         — position → doc_id mapping
  models/tfidf/vectorizer.pkl      — TfidfVectorizer
  models/tfidf/matrix.npz          — sparse TF-IDF matrix
  models/tfidf/doc_ids.json        — row → doc_id mapping
  models/identifier/index.pkl      — normalised value → doc_id inverted index
"""

import argparse
import os
import sys
import time
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and save BM25 / TF-IDF indices"
    )
    parser.add_argument(
        "--model", choices=["bm25", "tfidf", "identifier", "all"], default="all",
        help="Which model to build (default: all)",
    )
    parser.add_argument(
        "--docs", type=str, default=None,
        help="Path to documents.jsonl (default: data/json_format_data/full/documents.jsonl)",
    )
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help="Root directory for saved models (default: models/)",
    )
    args = parser.parse_args()

    root = find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Resolve paths
    if args.docs:
        docs_path = Path(args.docs)
    else:
        docs_path = root / "data" / "json_format_data" / "full" / "documents.jsonl"
        if not docs_path.exists():
            docs_path = root / "data" / "json_format_data" / "subset_100k" / "documents.jsonl"

    models_dir = Path(args.models_dir) if args.models_dir else root / "models"

    if not docs_path.exists():
        print(f"Error: documents.jsonl not found at {docs_path}")
        sys.exit(1)

    print("=" * 60)
    print("  Build Classical IR Indices")
    print("=" * 60)
    print(f"  Root       : {root}")
    print(f"  Dataset    : {docs_path}  ({docs_path.stat().st_size/1e9:.2f} GB)")
    print(f"  Models dir : {models_dir}")
    print(f"  Building   : {args.model}")
    print()

    from src.retrieval.classical_ir import BM25Retriever, TFIDFRetriever, IdentifierRetriever

    total_start = time.time()

    # ── BM25 ──────────────────────────────────────────────────────────
    if args.model in ("bm25", "all"):
        print("─" * 60)
        bm25 = BM25Retriever()
        bm25.build(docs_path, models_dir / "bm25")
        print()

    # ── TF-IDF ────────────────────────────────────────────────────────
    if args.model in ("tfidf", "all"):
        print("─" * 60)
        tfidf = TFIDFRetriever()
        tfidf.build(docs_path, models_dir / "tfidf")
        print()

    # ── Identifier (exact-match on structured IDs like IMO, MMSI) ────
    if args.model in ("identifier", "all"):
        print("─" * 60)
        print("[Identifier] Building inverted index from identifier fields …")
        t_id = time.time()

        import json as _json

        documents = []
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                doc = _json.loads(line)
                documents.append({
                    "doc_id": doc["doc_id"],
                    "identifiers": doc.get("identifiers", {}),
                })

        ident = IdentifierRetriever()
        ident.build_index(documents)
        ident.save(models_dir / "identifier" / "index.pkl")
        print(f"[Identifier] Built in {time.time()-t_id:.1f}s")
        print()

    elapsed = time.time() - total_start
    print("=" * 60)
    print(f"  All done in {elapsed/60:.1f} min")
    print("=" * 60)

    # ── Quick sanity check ────────────────────────────────────────────
    print("\nSanity check — running a test query …")

    test_tokens = ["sanction", "russian", "vessel"]
    test_text   = "sanction russian vessel"

    if args.model in ("bm25", "all"):
        bm25_check = BM25Retriever()
        bm25_check.load(models_dir / "bm25")
        results = bm25_check.search(test_tokens, k=3)
        print(f"\n  BM25 top-3 for {test_tokens}:")
        for doc_id, score in results:
            print(f"    {doc_id}  score={score:.4f}")

    if args.model in ("tfidf", "all"):
        tfidf_check = TFIDFRetriever()
        tfidf_check.load(models_dir / "tfidf")
        results = tfidf_check.search(test_text, k=3)
        print(f"\n  TF-IDF top-3 for '{test_text}':")
        for doc_id, score in results:
            print(f"    {doc_id}  score={score:.4f}")

    if args.model in ("identifier", "all"):
        ident_check = IdentifierRetriever()
        ident_check.load(models_dir / "identifier" / "index.pkl")
        results = ident_check.search("IMO9301407")
        print(f"\n  Identifier lookup for 'IMO9301407':")
        if results:
            for doc_id, score in results:
                print(f"    {doc_id}  score={score:.1f}")
        else:
            print("    (no match — expected if the test IMO is not in dataset)")

    print("\nIndex build complete.")


if __name__ == "__main__":
    main()
