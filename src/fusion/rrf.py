# TODO: INPUT FORMAT CONTRACT — confirm with Alireza (BM25) and Marek (dense)
# that their output CSVs use columns: query_id, doc_id, rank
# If column names differ, update load_ranked_list() only — no other changes needed.
#
# TODO: IMPORTS — requirements for this module are csv, argparse, collections (stdlib only).
# No new dependencies needed. If this changes, update requirements.txt.

"""
Reciprocal Rank Fusion (RRF) for combining ranked retrieval results.

Implements RRF as described in:
    Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods." SIGIR 2009.

Formula
-------
    RRF(d) = sum over all lists L of 1 / (k + rank_L(d))

    where k is a constant (default 60) and rank_L(d) is the 1-indexed rank
    of document d in list L. Documents absent from a list contribute 0.

Input format (assumed — to be confirmed with team)
---------------------------------------------------
Each retrieval lane produces a CSV with columns:
    query_id  — string identifier for the query (e.g. "q01")
    doc_id    — OpenSanctions entity ID string
    rank      — integer, 1-indexed, lower is better

Output format
-------------
Fused CSV with columns:
    query_id   — same as input
    doc_id     — OpenSanctions entity ID string
    rank       — fused rank position (1-indexed, sorted by descending rrf_score)
    rrf_score  — raw RRF score before ranking
"""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple


class ReciprocalRankFusion:
    """Fuse two or more ranked lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60):
        """
        Parameters
        ----------
        k : int
            RRF constant. Default 60 per Cormack et al. (2009).
            Higher k dampens the influence of high ranks; lower k amplifies it.
        """
        self.k = k

    def fuse(
        self, *ranked_lists: List[Tuple[str, int]]
    ) -> List[Tuple[str, float]]:
        """
        Fuse two or more ranked lists for a single query.

        Parameters
        ----------
        *ranked_lists : list of (doc_id, rank) tuples
            Each list represents one retrieval lane's output for a single query.
            rank is 1-indexed (1 = best). Documents appearing in only one list
            still receive a contribution from that list.

        Returns
        -------
        list of (doc_id, rrf_score) sorted by descending rrf_score.
        Ties are broken by doc_id (lexicographic) for deterministic output.
        """
        scores: Dict[str, float] = defaultdict(float)

        for ranked_list in ranked_lists:
            for doc_id, rank in ranked_list:
                scores[doc_id] += 1.0 / (self.k + rank)

        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def load_ranked_list(filepath: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    Read a ranked-list CSV and return {query_id: [(doc_id, rank), ...]}.

    Expected CSV columns: query_id, doc_id, rank
    """
    result: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row["query_id"]
            doc_id = row["doc_id"]
            rank = int(row["rank"])
            result[query_id].append((doc_id, rank))

    return dict(result)


def fuse_and_write(
    bm25_path: str,
    dense_path: str,
    output_path: str,
    k: int = 60,
) -> None:
    """
    Load BM25 and dense ranked lists, fuse with RRF, and write output CSV.

    Parameters
    ----------
    bm25_path : str
        Path to BM25 ranked-list CSV.
    dense_path : str
        Path to dense retrieval ranked-list CSV.
    output_path : str
        Path for the fused output CSV.
    k : int
        RRF constant (default 60).
    """
    bm25_lists = load_ranked_list(bm25_path)
    dense_lists = load_ranked_list(dense_path)

    all_query_ids = sorted(set(bm25_lists.keys()) | set(dense_lists.keys()))

    rrf = ReciprocalRankFusion(k=k)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "doc_id", "rank", "rrf_score"])

        for qid in all_query_ids:
            bm25_ranked = bm25_lists.get(qid, [])
            dense_ranked = dense_lists.get(qid, [])

            fused = rrf.fuse(bm25_ranked, dense_ranked)

            for fused_rank, (doc_id, score) in enumerate(fused, start=1):
                writer.writerow([qid, doc_id, fused_rank, f"{score:.6f}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse BM25 and dense retrieval results using Reciprocal Rank Fusion"
    )
    parser.add_argument(
        "--bm25", required=True, help="Path to BM25 ranked-list CSV"
    )
    parser.add_argument(
        "--dense", required=True, help="Path to dense retrieval ranked-list CSV"
    )
    parser.add_argument(
        "--output", required=True, help="Path for fused output CSV"
    )
    parser.add_argument(
        "--k", type=int, default=60, help="RRF constant (default: 60)"
    )
    args = parser.parse_args()

    fuse_and_write(args.bm25, args.dense, args.output, k=args.k)
    print(f"Fused output written to {args.output}")
