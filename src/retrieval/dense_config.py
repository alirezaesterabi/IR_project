"""
Registry and helpers for dense document embeddings (sentence-transformers + Chroma).

Aligned with notebooks/07a_dense_retrieval_creation_4_0_github.ipynb MODEL_REGISTRY.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ChromaDB internal limit is ~5461; stay under (matches 07a).
CHROMA_MAX_BATCH = 5000


def model_file_suffix(model_key: str) -> str:
    """File / collection token: minilm, bge_m3."""
    return model_key.replace("-", "_")


def size_tag_for_limit(limit: int | None) -> str:
    """Build size tag for RUN_TAG: FULL, 50K, 1M, or raw number."""
    if limit is None:
        return "FULL"
    if limit >= 1_000_000:
        return f"{limit // 1_000_000}M"
    if limit >= 1_000:
        return f"{limit // 1_000}K"
    return str(limit)


def compute_run_tag(
    limit: int | None,
    date_tag: str | None = None,
) -> str:
    """
    RUN_TAG = {size_tag}_{YYYYMMDD} (same convention as 07a).

    If date_tag is None, today (local) is used.
    """
    if date_tag is None:
        date_tag = datetime.now().strftime("%Y%m%d")
    st = size_tag_for_limit(limit)
    return f"{st}_{date_tag}"


def compute_chroma_batch_size(
    dim: int,
    available_ram_gb: float | None = None,
) -> int:
    """
    Target ~50 MB per batch when RAM > 4 GB (same as 07a notebook).
    Falls back without psutil: use CHROMA_MAX_BATCH clamped by dim heuristic.
    """
    if available_ram_gb is None:
        try:
            import psutil  # type: ignore

            available_ram_gb = psutil.virtual_memory().available / 1e9
        except Exception:
            available_ram_gb = 8.0

    target_mb = 50 if available_ram_gb > 4 else 20
    target_bytes = target_mb * 1e6
    batch = int(target_bytes / (dim * 8))
    return max(1_000, min(batch, CHROMA_MAX_BATCH))


@dataclass(frozen=True)
class DenseModelSpec:
    key: str
    hf_name: str
    dim: int
    batch_size_cuda: int
    batch_size_cpu: int
    chunk_size: int
    trust_remote_code: bool
    notes: str

    def encode_batch_size(self, device: str) -> int:
        d = device.lower()
        if d == "cuda":
            return self.batch_size_cuda
        return self.batch_size_cpu


# Keys must match CLI choices: minilm, bge-m3
DENSE_MODELS: dict[str, DenseModelSpec] = {
    "minilm": DenseModelSpec(
        key="minilm",
        hf_name="all-MiniLM-L6-v2",
        dim=384,
        batch_size_cuda=512,
        batch_size_cpu=64,
        chunk_size=100_000,
        trust_remote_code=False,
        notes="Lightweight baseline — fast, good quality",
    ),
    "bge-m3": DenseModelSpec(
        key="bge-m3",
        hf_name="BAAI/bge-m3",
        dim=1024,
        batch_size_cuda=128,
        batch_size_cpu=16,
        chunk_size=50_000,
        trust_remote_code=False,
        notes="Multilingual M3 — dense + sparse + ColBERT",
    ),
}


def resolved_paths(
    models_dir: Path,
    model_key: str,
    run_tag: str,
) -> dict[str, Any]:
    """Paths and Chroma collection name for one model + RUN_TAG."""
    spec = DENSE_MODELS[model_key]
    suffix = model_file_suffix(model_key)
    emb = models_dir / f"doc_embeddings_{suffix}_{run_tag}.npy"
    ids_path = models_dir / f"doc_ids_{suffix}_{run_tag}.json"
    chroma_collection = f"opensanctions_{suffix}_{run_tag}"
    chroma_batch = compute_chroma_batch_size(spec.dim)
    return {
        "spec": spec,
        "suffix": suffix,
        "embedding_cache": emb,
        "doc_ids_cache": ids_path,
        "chroma_collection": chroma_collection,
        "chroma_batch_size": chroma_batch,
    }
