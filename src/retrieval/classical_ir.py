"""
Classical IR retrievers: BM25, TF-IDF, and Identifier.

BM25 and TF-IDF follow the same interface:
    .build(docs_path, save_dir)   — fit on documents.jsonl and persist to disk
    .load(save_dir)               — restore from disk
    .search(query, k)             — return top-k [(doc_id, score), ...]

IdentifierRetriever provides exact-match lookup on identifier fields:
    .build_index(documents)       — build inverted index from document dicts
    .save(path) / .load(path)     — persist / restore index
    .search(query)                — return [(doc_id, 1.0), ...] for exact match

Input fields used from documents.jsonl:
    doc_id      — unique document identifier
    tokens      — pre-tokenised list → used by BM25 directly
    text_blob   — space-joined lemmatised string → used by TF-IDF
    identifiers — dict of identifier fields → used by IdentifierRetriever

Saved artefacts
---------------
models/bm25/
    index.pkl      — serialised BM25Okapi object
    doc_ids.json   — ordered list mapping position → doc_id

models/tfidf/
    vectorizer.pkl — fitted TfidfVectorizer
    matrix.npz     — sparse TF-IDF matrix (scipy CSR)
    doc_ids.json   — ordered list mapping row → doc_id

models/identifier/
    index.pkl      — normalised_value → [doc_id, ...] inverted index
"""

import json
import pickle
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import scipy.sparse


# ── BM25 ─────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 retriever backed by rank-bm25 (BM25Okapi).

    Parameters k1=1.2, b=0.75 follow the standard defaults used in the
    learning module (02_ranked_retrieval).
    """

    def __init__(self):
        self._bm25 = None
        self._doc_ids: List[str] = []

    # ------------------------------------------------------------------
    def build(self, docs_path: Path, save_dir: Path) -> None:
        """
        Stream documents.jsonl, fit BM25Okapi, and save to save_dir.

        Memory note: all token lists must fit in RAM (~3–5 GB for 1.24M docs).
        """
        from rank_bm25 import BM25Okapi

        docs_path = Path(docs_path)
        save_dir  = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[BM25] Reading corpus from {docs_path.name} …")
        t0 = time.time()

        doc_ids: List[str] = []
        corpus:  List[List[str]] = []

        with open(docs_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                doc = json.loads(line)
                doc_ids.append(doc["doc_id"])
                corpus.append(doc["tokens"])
                if (i + 1) % 100_000 == 0:
                    print(f"  {i+1:>10,} docs loaded …")

        print(f"  {len(doc_ids):,} docs loaded in {time.time()-t0:.1f}s")

        print("[BM25] Fitting BM25Okapi …")
        t1 = time.time()
        bm25 = BM25Okapi(corpus, k1=1.2, b=0.75)
        print(f"  Fit complete in {time.time()-t1:.1f}s")

        # Save
        index_path   = save_dir / "index.pkl"
        doc_ids_path = save_dir / "doc_ids.json"

        with open(index_path, "wb") as f:
            pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(doc_ids, f)

        self._bm25    = bm25
        self._doc_ids = doc_ids

        size_mb = index_path.stat().st_size / 1e6
        print(f"[BM25] Saved → {index_path}  ({size_mb:.0f} MB)")
        print(f"[BM25] Saved → {doc_ids_path}")

    # ------------------------------------------------------------------
    def load(self, save_dir: Path) -> None:
        """Load BM25 index and doc_ids from disk."""
        save_dir = Path(save_dir)

        with open(save_dir / "index.pkl", "rb") as f:
            self._bm25 = pickle.load(f)

        with open(save_dir / "doc_ids.json", encoding="utf-8") as f:
            self._doc_ids = json.load(f)

        print(f"[BM25] Loaded index with {len(self._doc_ids):,} documents")

    # ------------------------------------------------------------------
    def search(
        self,
        query_tokens: List[str],
        k: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k results for a tokenised query.

        Parameters
        ----------
        query_tokens : list of str
            Pre-tokenised query (same preprocessing as the corpus).
        k : int
            Number of results to return.

        Returns
        -------
        list of (doc_id, score) sorted by score descending.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not loaded. Call .build() or .load() first.")

        scores = self._bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:k]

        return [
            (self._doc_ids[i], float(scores[i]))
            for i in top_k_indices
            if scores[i] > 0
        ]

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None


# ── TF-IDF ───────────────────────────────────────────────────────────────────

class TFIDFRetriever:
    """
    TF-IDF retriever backed by sklearn TfidfVectorizer.

    The TF-IDF matrix is stored as a scipy sparse CSR matrix (.npz).
    Query search uses cosine similarity computed efficiently with sparse
    dot products (no dense materialisation).
    """

    def __init__(self):
        self._vectorizer = None
        self._matrix: Optional[scipy.sparse.csr_matrix] = None
        self._doc_ids: List[str] = []

    # ------------------------------------------------------------------
    def build(self, docs_path: Path, save_dir: Path) -> None:
        """
        Stream documents.jsonl, fit TfidfVectorizer, and save to save_dir.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs_path = Path(docs_path)
        save_dir  = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[TFIDF] Reading corpus from {docs_path.name} …")
        t0 = time.time()

        doc_ids:    List[str] = []
        text_blobs: List[str] = []

        with open(docs_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                doc = json.loads(line)
                doc_ids.append(doc["doc_id"])
                text_blobs.append(doc["text_blob"])
                if (i + 1) % 100_000 == 0:
                    print(f"  {i+1:>10,} docs loaded …")

        print(f"  {len(doc_ids):,} docs loaded in {time.time()-t0:.1f}s")

        print("[TFIDF] Fitting TfidfVectorizer …")
        t1 = time.time()
        vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,       # log(1 + tf) — same as learning module
            min_df=2,                # ignore hapax legomena
            max_df=0.95,             # ignore near-universal terms
        )
        matrix = vectorizer.fit_transform(text_blobs)
        print(f"  Fit complete in {time.time()-t1:.1f}s — "
              f"shape {matrix.shape}, nnz {matrix.nnz:,}")

        # Save
        vec_path     = save_dir / "vectorizer.pkl"
        matrix_path  = save_dir / "matrix.npz"
        doc_ids_path = save_dir / "doc_ids.json"

        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

        scipy.sparse.save_npz(str(matrix_path), matrix.tocsr())

        with open(doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(doc_ids, f)

        self._vectorizer = vectorizer
        self._matrix     = matrix.tocsr()
        self._doc_ids    = doc_ids

        print(f"[TFIDF] Saved → {vec_path}  ({vec_path.stat().st_size/1e6:.0f} MB)")
        print(f"[TFIDF] Saved → {matrix_path}  ({matrix_path.stat().st_size/1e6:.0f} MB)")
        print(f"[TFIDF] Saved → {doc_ids_path}")

    # ------------------------------------------------------------------
    def load(self, save_dir: Path) -> None:
        """Load TF-IDF vectorizer, matrix, and doc_ids from disk."""
        save_dir = Path(save_dir)

        with open(save_dir / "vectorizer.pkl", "rb") as f:
            self._vectorizer = pickle.load(f)

        self._matrix = scipy.sparse.load_npz(str(save_dir / "matrix.npz"))

        with open(save_dir / "doc_ids.json", encoding="utf-8") as f:
            self._doc_ids = json.load(f)

        print(f"[TFIDF] Loaded matrix {self._matrix.shape} "
              f"with {len(self._doc_ids):,} documents")

    # ------------------------------------------------------------------
    def search(
        self,
        query_text: str,
        k: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k results for a query string.

        Uses cosine similarity. The TF-IDF matrix rows are L2-normalised
        by the vectorizer (norm='l2' default), so dot product = cosine.

        Parameters
        ----------
        query_text : str
            Raw query string (same preprocessing as text_blob).
        k : int
            Number of results to return.

        Returns
        -------
        list of (doc_id, score) sorted by score descending.
        """
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("TF-IDF not loaded. Call .build() or .load() first.")

        q_vec = self._vectorizer.transform([query_text])  # (1, vocab)
        scores = (self._matrix @ q_vec.T).toarray().ravel()  # (n_docs,)

        top_k_indices = np.argsort(scores)[::-1][:k]

        return [
            (self._doc_ids[i], float(scores[i]))
            for i in top_k_indices
            if scores[i] > 0
        ]

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._vectorizer is not None


# ── Identifier (exact match) ───────────────────────────────────────────────

class IdentifierRetriever:
    """
    Exact-match retriever for structured identifiers (IMO, MMSI, LEI, etc.).

    Builds an inverted index from the ``identifiers`` dict in each document.
    Both keys and values are normalised (strip + uppercase) so lookups are
    case-insensitive.  Every match scores 1.0.
    """

    # Patterns for looks_like_identifier (compiled once)
    _IMO_RE    = re.compile(r"^IMO\d{7}$", re.IGNORECASE)
    _MMSI_RE   = re.compile(r"^\d{9}$")
    _LEI_RE    = re.compile(r"^[A-Z0-9]{20}$", re.IGNORECASE)
    _EMAIL_RE  = re.compile(r"@")
    _CRYPTO_RE = re.compile(r"^[0-9a-f]{26,}$", re.IGNORECASE)
    _GENERIC_RE = re.compile(r"^[A-Za-z0-9]{6,20}$")

    def __init__(self):
        self._index: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(value: str) -> str:
        """Strip whitespace and uppercase."""
        return value.strip().upper()

    # ------------------------------------------------------------------
    def build_index(self, documents: Iterable[dict]) -> None:
        """
        Build the inverted lookup from an iterable of document dicts.

        Each document must have at least ``doc_id`` (str) and
        ``identifiers`` (dict mapping field name → list of values).
        """
        index: Dict[str, List[str]] = {}

        for doc in documents:
            doc_id = doc["doc_id"]
            identifiers = doc.get("identifiers", {})

            for _field, values in identifiers.items():
                for raw_val in values:
                    key = self._normalise(str(raw_val))
                    if key:
                        index.setdefault(key, []).append(doc_id)

        self._index = index

    # ------------------------------------------------------------------
    def search(self, query: str) -> List[Tuple[str, float]]:
        """
        Exact-match lookup.  Returns ``[(doc_id, 1.0), ...]`` or ``[]``.
        """
        key = self._normalise(query)
        doc_ids = self._index.get(key, [])
        return [(doc_id, 1.0) for doc_id in doc_ids]

    # ------------------------------------------------------------------
    @staticmethod
    def looks_like_identifier(query: str) -> bool:
        """
        Heuristic: return True if *query* resembles a structured identifier.
        """
        q = query.strip()
        if IdentifierRetriever._IMO_RE.match(q):
            return True
        if IdentifierRetriever._MMSI_RE.match(q):
            return True
        if IdentifierRetriever._LEI_RE.match(q):
            return True
        if IdentifierRetriever._EMAIL_RE.search(q):
            return True
        if IdentifierRetriever._CRYPTO_RE.match(q):
            return True
        if IdentifierRetriever._GENERIC_RE.match(q):
            return True
        return False

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Persist the inverted index to *path* (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._index, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Identifier] Saved → {path}  "
              f"({len(self._index):,} keys, {path.stat().st_size/1e6:.1f} MB)")

    # ------------------------------------------------------------------
    def load(self, path: Path) -> None:
        """Restore the inverted index from *path*."""
        path = Path(path)
        with open(path, "rb") as f:
            self._index = pickle.load(f)
        print(f"[Identifier] Loaded index with {len(self._index):,} keys")

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return len(self._index) > 0
