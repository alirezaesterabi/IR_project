"""Unit tests for src.fusion.rrf — Reciprocal Rank Fusion."""

import csv
import os
import tempfile

import pytest

from src.fusion.rrf import ReciprocalRankFusion, load_ranked_list, fuse_and_write


# ── ReciprocalRankFusion.fuse() ──────────────────────────────────────────────


class TestFuseFullOverlap:
    """Two lists containing exactly the same documents."""

    def test_scores_and_order(self):
        rrf = ReciprocalRankFusion(k=60)

        # Both lists rank A=1, B=2, C=3
        list1 = [("A", 1), ("B", 2), ("C", 3)]
        list2 = [("A", 1), ("B", 2), ("C", 3)]

        result = rrf.fuse(list1, list2)
        doc_ids = [doc_id for doc_id, _ in result]
        scores = {doc_id: score for doc_id, score in result}

        # A should be first, C should be last
        assert doc_ids == ["A", "B", "C"]

        # Each doc gets 2 * 1/(k+rank)
        assert scores["A"] == pytest.approx(2.0 / 61)
        assert scores["B"] == pytest.approx(2.0 / 62)
        assert scores["C"] == pytest.approx(2.0 / 63)

    def test_different_orderings(self):
        """Lists agree on documents but disagree on order."""
        rrf = ReciprocalRankFusion(k=60)

        list1 = [("A", 1), ("B", 2)]
        list2 = [("B", 1), ("A", 2)]

        result = rrf.fuse(list1, list2)
        scores = {doc_id: score for doc_id, score in result}

        # Both get 1/61 + 1/62 — same score
        assert scores["A"] == pytest.approx(scores["B"])


class TestFusePartialOverlap:
    """Documents appearing in only one list are still included."""

    def test_partial_overlap(self):
        rrf = ReciprocalRankFusion(k=60)

        list1 = [("A", 1), ("B", 2)]
        list2 = [("B", 1), ("C", 2)]

        result = rrf.fuse(list1, list2)
        doc_ids = [doc_id for doc_id, _ in result]
        scores = {doc_id: score for doc_id, score in result}

        # All three docs should appear
        assert set(doc_ids) == {"A", "B", "C"}

        # B appears in both lists — should have highest score
        assert scores["B"] > scores["A"]
        assert scores["B"] > scores["C"]

        # A (rank 1 in list1 only) and C (rank 2 in list2 only)
        assert scores["A"] == pytest.approx(1.0 / 61)
        assert scores["C"] == pytest.approx(1.0 / 62)

        # B: 1/62 (rank 2 in list1) + 1/61 (rank 1 in list2)
        assert scores["B"] == pytest.approx(1.0 / 62 + 1.0 / 61)

    def test_no_overlap(self):
        """Completely disjoint lists."""
        rrf = ReciprocalRankFusion(k=60)

        list1 = [("A", 1)]
        list2 = [("B", 1)]

        result = rrf.fuse(list1, list2)
        scores = {doc_id: score for doc_id, score in result}

        assert set(scores.keys()) == {"A", "B"}
        assert scores["A"] == pytest.approx(scores["B"])


class TestRankOneBothVsOne:
    """Doc ranked 1st in both lists beats doc ranked 1st in only one."""

    def test_rank1_both_beats_rank1_single(self):
        rrf = ReciprocalRankFusion(k=60)

        # D is rank 1 in both lists; E is rank 1 in list1 only, absent from list2
        list1 = [("D", 1), ("E", 2)]
        list2 = [("D", 1)]

        result = rrf.fuse(list1, list2)
        scores = {doc_id: score for doc_id, score in result}

        # D: 1/61 + 1/61 = 2/61;  E: 1/62
        assert scores["D"] > scores["E"]
        assert scores["D"] == pytest.approx(2.0 / 61)
        assert scores["E"] == pytest.approx(1.0 / 62)


class TestEdgeCases:
    """Edge cases: k=0, empty lists, single list."""

    def test_k_zero_no_crash(self):
        """k=0 means rank 1 gives score 1/(0+1) = 1.0 — should not crash."""
        rrf = ReciprocalRankFusion(k=0)

        list1 = [("A", 1), ("B", 2)]
        list2 = [("A", 1)]

        result = rrf.fuse(list1, list2)
        scores = {doc_id: score for doc_id, score in result}

        assert scores["A"] == pytest.approx(1.0 + 1.0)  # 1/1 + 1/1
        assert scores["B"] == pytest.approx(0.5)  # 1/2

    def test_empty_lists(self):
        """Empty input returns empty output."""
        rrf = ReciprocalRankFusion(k=60)

        result = rrf.fuse([], [])
        assert result == []

    def test_single_empty_list(self):
        """One populated list, one empty."""
        rrf = ReciprocalRankFusion(k=60)

        list1 = [("A", 1)]
        result = rrf.fuse(list1, [])

        assert len(result) == 1
        assert result[0][0] == "A"
        assert result[0][1] == pytest.approx(1.0 / 61)

    def test_three_lists(self):
        """Fuse works with more than two lists."""
        rrf = ReciprocalRankFusion(k=60)

        list1 = [("A", 1)]
        list2 = [("A", 1)]
        list3 = [("A", 1)]

        result = rrf.fuse(list1, list2, list3)
        assert result[0][1] == pytest.approx(3.0 / 61)

    def test_deterministic_tie_breaking(self):
        """Tied scores are broken by doc_id lexicographically."""
        rrf = ReciprocalRankFusion(k=60)

        list1 = [("Z", 1)]
        list2 = [("A", 1)]

        result = rrf.fuse(list1, list2)
        doc_ids = [doc_id for doc_id, _ in result]

        # Same score, A comes before Z
        assert doc_ids == ["A", "Z"]


# ── load_ranked_list() ───────────────────────────────────────────────────────


class TestLoadRankedList:

    def test_load_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "query_id,doc_id,rank\n"
            "q01,NK-001,1\n"
            "q01,NK-002,2\n"
            "q02,NK-003,1\n",
            encoding="utf-8",
        )

        result = load_ranked_list(str(csv_path))

        assert set(result.keys()) == {"q01", "q02"}
        assert result["q01"] == [("NK-001", 1), ("NK-002", 2)]
        assert result["q02"] == [("NK-003", 1)]


# ── fuse_and_write() ────────────────────────────────────────────────────────


class TestFuseAndWrite:

    def test_end_to_end(self, tmp_path):
        bm25_path = tmp_path / "bm25.csv"
        dense_path = tmp_path / "dense.csv"
        output_path = tmp_path / "fused.csv"

        bm25_path.write_text(
            "query_id,doc_id,rank\n"
            "q01,NK-001,1\n"
            "q01,NK-002,2\n"
            "q02,NK-010,1\n",
            encoding="utf-8",
        )
        dense_path.write_text(
            "query_id,doc_id,rank\n"
            "q01,NK-002,1\n"
            "q01,NK-003,2\n"
            "q02,NK-010,1\n",
            encoding="utf-8",
        )

        fuse_and_write(str(bm25_path), str(dense_path), str(output_path), k=60)

        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # q01 should have 3 docs, q02 should have 1
        q01_rows = [r for r in rows if r["query_id"] == "q01"]
        q02_rows = [r for r in rows if r["query_id"] == "q02"]

        assert len(q01_rows) == 3
        assert len(q02_rows) == 1

        # q01: NK-002 appears in both lists — should be rank 1
        assert q01_rows[0]["doc_id"] == "NK-002"
        assert q01_rows[0]["rank"] == "1"

        # q02: NK-010 appears in both — rank 1
        assert q02_rows[0]["doc_id"] == "NK-010"
        assert q02_rows[0]["rank"] == "1"
        assert float(q02_rows[0]["rrf_score"]) == pytest.approx(2.0 / 61, abs=1e-6)

    def test_disjoint_query_ids(self, tmp_path):
        """BM25 and dense have different query sets — union is used."""
        bm25_path = tmp_path / "bm25.csv"
        dense_path = tmp_path / "dense.csv"
        output_path = tmp_path / "fused.csv"

        bm25_path.write_text(
            "query_id,doc_id,rank\nq01,NK-001,1\n", encoding="utf-8"
        )
        dense_path.write_text(
            "query_id,doc_id,rank\nq02,NK-002,1\n", encoding="utf-8"
        )

        fuse_and_write(str(bm25_path), str(dense_path), str(output_path))

        with open(output_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        query_ids = {r["query_id"] for r in rows}
        assert query_ids == {"q01", "q02"}
