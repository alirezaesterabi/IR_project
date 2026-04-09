"""
Query-time dense retriever backed by sentence-transformers + ChromaDB.

This module turns the notebook-only dense search logic into a reusable class
that can load any model registered in ``dense_config.py`` and query the
matching Chroma collection for a given RUN_TAG.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .dense_config import DENSE_MODELS, DenseModelSpec, resolved_paths


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_device(device: str | None) -> str:
    if device:
        return device.lower()
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _read_hf_token(explicit_token: str | None = None) -> str | None:
    if explicit_token:
        return explicit_token.strip() or None
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value.strip() or None
    return None


class DenseRetriever:
    """
    Query ChromaDB collections produced by ``build_dense_embeddings.py``.

    Parameters
    ----------
    model_key:
        Dense model key from ``DENSE_MODELS`` (e.g. ``minilm`` or ``bge-m3``).
    run_tag:
        RUN_TAG identifying the Chroma collection / cached artefacts.
    chroma_dir:
        Optional path to the Chroma persistence directory.
    models_dir:
        Optional path to the models directory used for naming resolution.
    device:
        ``cuda`` / ``cpu`` / ``mps``. Defaults to automatic detection.
    hf_token:
        Optional Hugging Face token for query model loading.
    """

    def __init__(
        self,
        model_key: str,
        run_tag: str,
        chroma_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
        device: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        if model_key not in DENSE_MODELS:
            raise ValueError(
                f"Unknown dense model {model_key!r}. Available: {sorted(DENSE_MODELS)}"
            )

        root = _default_project_root()
        self.model_key = model_key
        self.run_tag = run_tag
        self.spec: DenseModelSpec = DENSE_MODELS[model_key]
        self.models_dir = Path(models_dir) if models_dir else root / "models"
        self.chroma_dir = Path(chroma_dir) if chroma_dir else root / "chroma_db"
        self.device = _resolve_device(device)
        self.hf_token = _read_hf_token(hf_token)
        self.paths = resolved_paths(self.models_dir, model_key, run_tag)

        self._client: Any | None = None
        self._collection: Any | None = None
        self._encoder: Any | None = None

    @property
    def collection_name(self) -> str:
        return str(self.paths["chroma_collection"])

    def _load_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError as e:  # pragma: no cover - depends on optional env
            raise ImportError(
                "chromadb is required for dense retrieval. "
                "Install dependencies from requirements.txt."
            ) from e

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            self._collection = self._client.get_collection(name=self.collection_name)
        except Exception as e:  # pragma: no cover - backend-specific message
            raise FileNotFoundError(
                f"Chroma collection {self.collection_name!r} not found in "
                f"{self.chroma_dir}. Build dense embeddings for RUN_TAG "
                f"{self.run_tag!r} first."
            ) from e
        return self._collection

    def _load_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on optional env
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install dependencies from requirements.txt."
            ) from e

        load_kwargs: dict[str, Any] = {"device": self.device}
        if self.spec.trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        if self.hf_token:
            load_kwargs["token"] = self.hf_token

        encoder = SentenceTransformer(self.spec.hf_name, **load_kwargs)
        if self.device == "cuda":
            try:
                encoder = encoder.half()
            except Exception:
                pass
        self._encoder = encoder
        return self._encoder

    def search(
        self,
        query_text: str,
        k: int = 20,
        include_metadata: bool = True,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Search the configured Chroma collection.

        Returns
        -------
        list of ``(doc_id, similarity, metadata)`` sorted by rank order.
        ``similarity`` is computed as ``1 - distance`` to match the notebook.
        """
        encoder = self._load_encoder()
        collection = self._load_collection()

        q_emb = encoder.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        include = ["distances"]
        if include_metadata:
            include.append("metadatas")
        results = collection.query(
            query_embeddings=q_emb,
            n_results=k,
            include=include,
        )

        ids = results.get("ids", [[]])
        distances = results.get("distances", [[]])
        metadatas = results.get("metadatas", [[]]) if include_metadata else [[]]

        hits: list[tuple[str, float, dict[str, Any]]] = []
        for idx, doc_id in enumerate(ids[0] if ids else []):
            distance = float(distances[0][idx])
            score = 1.0 - distance
            metadata = (
                dict(metadatas[0][idx]) if include_metadata and metadatas and metadatas[0] else {}
            )
            hits.append((str(doc_id), score, metadata))
        return hits
