"""
Encode `embedding_text` from documents.jsonl with sentence-transformers; save
`.npy` + `doc_ids.json` under models/; optionally upsert into ChromaDB.

Usage
-----
  python scripts/build_dense_embeddings.py --docs data/json_format_data/full/documents.jsonl \\
      --limit 1000 --model minilm

  python scripts/build_dense_embeddings.py --docs data/json_format_data/subset/documents.jsonl \\
      --model minilm --chroma --run-tag 10K_YYYYMMDD

Environment: HF_TOKEN or HUGGING_FACE_HUB_TOKEN for gated models; optional
--hf-token path to a file containing the token.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# --- project root (same pattern as scripts/build_index.py) -----------------

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


def mem_report(label: str) -> None:
    try:
        import psutil

        p = psutil.Process()
        rss = p.memory_info().rss / 1e9
        avail = psutil.virtual_memory().available / 1e9
        print(f"  [MEM {label}] RSS={rss:.2f} GB | Available={avail:.2f} GB")
    except Exception:
        pass


def read_hf_token(args: argparse.Namespace) -> str | None:
    if getattr(args, "hf_token", None):
        p = Path(args.hf_token)
        if p.is_file():
            return p.read_text(encoding="utf-8").strip() or None
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(key)
        if v:
            return v.strip() or None
    return None


def load_corpus(
    docs_path: Path,
    limit: int | None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Returns doc_ids, embedding texts, and full doc dicts (for Chroma metadata)."""
    import importlib.util

    root = find_root()
    _path = root / "src" / "preprocessing" / "embedding_text.py"
    spec = importlib.util.spec_from_file_location("embedding_text", _path)
    _mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(_mod)
    build_embedding_text = _mod.build_embedding_text

    doc_ids: list[str] = []
    texts: list[str] = []
    docs: list[dict[str, Any]] = []

    with open(docs_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            did = doc.get("doc_id") or doc.get("id") or f"doc_{i}"
            doc_ids.append(str(did))
            t = doc.get("embedding_text")
            if not t:
                t = build_embedding_text(doc)
            texts.append(t)
            docs.append(doc)

    return doc_ids, texts, docs


def chroma_collection_metadata() -> dict[str, Any]:
    return {
        "hnsw:space": "cosine",
        "hnsw:M": 16,
        "hnsw:construction_ef": 100,
        "hnsw:batch_size": 5000,
        "hnsw:sync_threshold": 5000,
    }


def build_chroma_metadatas(
    corpus_slice: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Per-doc metadata for Chroma (string values; matches 07a)."""
    out: list[dict[str, str]] = []
    for doc in corpus_slice:
        meta_raw = doc.get("metadata", {}) or {}
        countries = meta_raw.get("country", [])
        programs = meta_raw.get("programId", [])
        if not isinstance(countries, list):
            countries = []
        if not isinstance(programs, list):
            programs = []
        out.append(
            {
                "caption": str(doc.get("caption", ""))[:200],
                "schema": str(doc.get("schema", "")),
                "country": ", ".join(str(c).upper() for c in countries)[:100],
                "program": ", ".join(str(p) for p in programs)[:200],
            }
        )
    return out


def reset_artefacts(
    models_dir: Path,
    chroma_dir: Path,
    model_keys: list[str],
    run_tag: str,
) -> None:
    from src.retrieval.dense_config import resolved_paths

    print("[reset] Removing cached embeddings and doc_ids for this RUN_TAG …")
    for mk in model_keys:
        sc = resolved_paths(models_dir, mk, run_tag)
        for p in (sc["embedding_cache"], sc["doc_ids_cache"]):
            if p.exists():
                p.unlink()
                print(f"  deleted {p}")

    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    for mk in model_keys:
        name = resolved_paths(models_dir, mk, run_tag)["chroma_collection"]
        try:
            client.delete_collection(name)
            print(f"  deleted Chroma collection {name!r}")
        except Exception as e:
            print(f"  (skip) could not delete {name!r}: {e}")


def encode_one_model(
    model_key: str,
    doc_texts: list[str],
    doc_ids: list[str],
    paths: dict[str, Any],
    device: str,
    hf_token: str | None,
    force: bool,
) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    from src.retrieval.dense_config import DENSE_MODELS

    scfg = paths["spec"]
    emb_path: Path = paths["embedding_cache"]
    ids_path: Path = paths["doc_ids_cache"]
    n_docs = len(doc_texts)

    if (
        emb_path.exists()
        and ids_path.exists()
        and not force
    ):
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if cached_ids == doc_ids:
            print(f"  [{model_key}] Cache hit — loading mmap {emb_path.name}")
            return np.load(str(emb_path), mmap_mode="r")
        print(f"  [{model_key}] Cache doc_ids mismatch — re-encoding.")
        emb_path.unlink(missing_ok=True)
        ids_path.unlink(missing_ok=True)

    batch_size = scfg.encode_batch_size(device)
    chunk_size = scfg.chunk_size

    load_kwargs: dict[str, Any] = {"device": device}
    if scfg.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if hf_token:
        load_kwargs["token"] = hf_token

    print(f"  [{model_key}] Loading {scfg.hf_name!r} on {device.upper()} …")
    model = SentenceTransformer(scfg.hf_name, **load_kwargs)
    if device == "cuda":
        model = model.half()
        print("  Using fp16 inference for speed.")

    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim != scfg.dim:
        print(
            f"  WARNING: registry dim {scfg.dim} vs model {actual_dim} — using model dim."
        )

    cache_path = str(emb_path.resolve())
    tmp_path = cache_path.replace(".npy", "_tmp.npy")
    Path(tmp_path).unlink(missing_ok=True)

    fp = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_docs, actual_dim),
    )
    print(f"  Pre-allocated {n_docs:,} x {actual_dim} float32 (temp → {emb_path.name})")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    pbar = tqdm(total=n_docs, desc=f"Encode [{model_key}]", unit="doc")
    chunk_count = 0
    for chunk_start in range(0, n_docs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_docs)
        chunk = doc_texts[chunk_start:chunk_end]
        emb = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        fp[chunk_start:chunk_end] = emb
        del emb, chunk
        pbar.update(chunk_end - chunk_start)
        chunk_count += 1
        if chunk_count % 4 == 0:
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    pbar.close()
    fp.flush()
    del fp
    gc.collect()

    elapsed = time.time() - start
    print(
        f"  Encoding complete in {elapsed:.1f}s  ({n_docs / max(elapsed, 1e-6):,.0f} docs/sec)"
    )
    if device == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak GPU memory: {peak_mb:.0f} MB")

    shutil.move(tmp_path, cache_path)
    ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")
    print(f"  Saved -> {emb_path.name}")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"  Released {model_key} model from GPU.")

    return np.load(cache_path, mmap_mode="r")


def chroma_upsert(
    model_key: str,
    embeddings: np.ndarray,
    doc_ids: list[str],
    corpus: list[dict[str, Any]],
    paths: dict[str, Any],
    chroma_dir: Path,
    force_chroma: bool,
) -> None:
    import chromadb
    from tqdm import tqdm

    sc = paths
    coll_name = sc["chroma_collection"]
    batch_sz = sc["chroma_batch_size"]
    n_docs = len(doc_ids)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll = client.get_or_create_collection(
        name=coll_name,
        metadata=chroma_collection_metadata(),
    )

    cnt = coll.count()
    if cnt >= n_docs and not force_chroma:
        print(f"  [{model_key}] ChromaDB already has {cnt:,} docs — skipping insert.")
        return

    if force_chroma and cnt >= n_docs:
        print(f"  [{model_key}] --force-chroma: rebuilding collection {coll_name!r} …")
        client.delete_collection(coll_name)
        coll = client.get_or_create_collection(
            name=coll_name,
            metadata=chroma_collection_metadata(),
        )
    elif cnt > 0 and cnt != n_docs:
        print(
            f"  [{model_key}] Count mismatch ({cnt:,} vs {n_docs:,}). Rebuilding collection…"
        )
        client.delete_collection(coll_name)
        coll = client.get_or_create_collection(
            name=coll_name,
            metadata=chroma_collection_metadata(),
        )

    print(f"  [{model_key}] Inserting {n_docs:,} docs (chroma_batch={batch_sz})…")
    mem_report("before ChromaDB insert")
    start = time.time()
    n_batches = (n_docs + batch_sz - 1) // batch_sz
    batch_iter = tqdm(
        range(0, n_docs, batch_sz),
        total=n_batches,
        desc=f"ChromaDB [{model_key}]",
        unit="batch",
    )
    for i in batch_iter:
        batch_end = min(i + batch_sz, n_docs)
        batch_meta = build_chroma_metadatas(corpus[i:batch_end])
        batch_emb = np.ascontiguousarray(embeddings[i:batch_end]).tolist()
        coll.add(
            ids=doc_ids[i:batch_end],
            embeddings=batch_emb,
            metadatas=batch_meta,
        )
        del batch_emb, batch_meta
        gc.collect()
        batch_iter.set_postfix(docs=f"{batch_end:,}")

    elapsed = time.time() - start
    print(f"  [{model_key}] Chroma done in {elapsed:.1f}s — collection {coll_name!r} has {coll.count():,} docs")
    mem_report(f"after {model_key} Chroma insert")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build dense embeddings from embedding_text and optional Chroma upsert.",
    )
    p.add_argument(
        "--docs",
        type=str,
        default=None,
        help="Path to documents.jsonl (default: full/documents.jsonl, else subset/documents.jsonl)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows from JSONL (default: full file)",
    )
    p.add_argument(
        "--model",
        choices=["minilm", "bge-m3", "all"],
        default="minilm",
        help="Which embedding model to run (default: minilm)",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Override RUN_TAG (default: {size}_{YYYYMMDD})",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory for .npy and doc_ids json (default: <root>/models)",
    )
    p.add_argument(
        "--chroma-dir",
        type=str,
        default=None,
        help="ChromaDB persistence directory (default: <root>/chroma_db)",
    )
    p.add_argument(
        "--chroma",
        action="store_true",
        help="Upsert embeddings into ChromaDB after encoding (default: off)",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device for sentence-transformers",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if cached .npy matches doc_ids",
    )
    p.add_argument(
        "--force-chroma",
        action="store_true",
        help="Re-upsert Chroma even if collection already full",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete matching npy/doc_ids (and Chroma collections if --chroma) for this RUN_TAG, then exit",
    )
    p.add_argument(
        "--hf-token",
        type=str,
        default=None,
        metavar="PATH",
        help="File containing Hugging Face token (else HF_TOKEN / HUGGING_FACE_HUB_TOKEN env)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    root = find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    use_chroma = bool(args.chroma)

    if args.docs:
        docs_path = Path(args.docs)
    else:
        docs_path = root / "data" / "json_format_data" / "full" / "documents.jsonl"
        if not docs_path.exists():
            docs_path = root / "data" / "json_format_data" / "subset" / "documents.jsonl"

    models_dir = Path(args.models_dir) if args.models_dir else root / "models"
    chroma_dir = Path(args.chroma_dir) if args.chroma_dir else root / "chroma_db"
    models_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    if not docs_path.exists():
        print(f"Error: documents.jsonl not found at {docs_path}")
        return 1

    from src.retrieval.dense_config import DENSE_MODELS, compute_run_tag, resolved_paths

    run_tag = args.run_tag or compute_run_tag(args.limit)
    if args.model == "all":
        model_keys = list(DENSE_MODELS.keys())
    else:
        model_keys = [args.model]

    if args.reset:
        reset_artefacts(models_dir, chroma_dir, model_keys, run_tag)
        print("[reset] Done.")
        return 0

    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "Error: --device cuda was requested but torch.cuda.is_available() is False.\n"
            "  Install a CUDA-enabled PyTorch build from pytorch.org and confirm the\n"
            "  local machine exposes a working GPU before retrying."
        )
        return 1

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    hf_token = read_hf_token(args)

    print("=" * 60)
    print("  Build dense embeddings")
    print("=" * 60)
    print(f"  Root         : {root}")
    print(f"  Dataset      : {docs_path}")
    print(f"  Models dir   : {models_dir}")
    print(f"  Chroma dir   : {chroma_dir}")
    print(f"  Limit        : {args.limit if args.limit is not None else 'ALL'}")
    print(f"  RUN_TAG      : {run_tag}")
    print(f"  Models       : {', '.join(model_keys)}")
    print(f"  Device       : {device}")
    print(f"  PyTorch      : {getattr(torch, '__version__', '?')}")
    print(f"  Chroma       : {'yes' if use_chroma else 'no'}")
    print()

    if device == "cpu" and args.device == "auto":
        tv = getattr(torch, "__version__", "")
        if "+cpu" in tv:
            print(
                "WARNING: Using CPU for encoding — PyTorch wheel is CPU-only "
                f"({tv!r}). If you expected GPU encoding, install a CUDA-enabled "
                "PyTorch build from pytorch.org and verify that `torch.cuda.is_available()` "
                "returns True.\n"
            )

    t0 = time.time()
    doc_ids, doc_texts, corpus = load_corpus(docs_path, args.limit)
    n_docs = len(doc_ids)
    print(f"Loaded {n_docs:,} documents")
    if args.limit is not None and n_docs < args.limit:
        print(
            f"  Note: requested {args.limit:,} rows but file yielded {n_docs:,}; RUN_TAG still {run_tag!r}."
        )
    mem_report("after corpus load")

    # Rough size hint
    for mk in model_keys:
        spec = DENSE_MODELS[mk]
        bytes_est = n_docs * spec.dim * 4
        print(f"  [{mk}] matrix ~ {bytes_est / 1e9:.2f} GB float32 (approx.)")

    model_embeddings: dict[str, np.ndarray] = {}

    for model_key in model_keys:
        paths = resolved_paths(models_dir, model_key, run_tag)
        print("-" * 60)
        print(
            f"  {model_key}  dim={paths['spec'].dim}  "
            f"encode_batch={paths['spec'].encode_batch_size(device)}  "
            f"chroma_batch={paths['chroma_batch_size']}"
        )
        print(f"  npy: {paths['embedding_cache'].name}")
        print(f"  collection: {paths['chroma_collection']}")
        print("-" * 60)

        emb = encode_one_model(
            model_key,
            doc_texts,
            doc_ids,
            paths,
            device,
            hf_token,
            args.force,
        )
        model_embeddings[model_key] = emb
        mem_report(f"after {model_key} encode")

    if use_chroma:
        for model_key in model_keys:
            paths = resolved_paths(models_dir, model_key, run_tag)
            chroma_upsert(
                model_key,
                model_embeddings[model_key],
                doc_ids,
                corpus,
                paths,
                chroma_dir,
                args.force_chroma,
            )
        del model_embeddings
        gc.collect()

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  Finished in {elapsed / 60:.1f} min")
    print(f"  RUN_TAG      = {run_tag!r}")
    print(f"  MODELS_DIR   = {models_dir}")
    print(f"  CHROMA_DIR   = {chroma_dir}")
    for mk in model_keys:
        sc = resolved_paths(models_dir, mk, run_tag)
        print(f"  [{mk}] {sc['embedding_cache'].name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
