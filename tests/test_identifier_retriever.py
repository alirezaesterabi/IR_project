"""Tests for IdentifierRetriever (exact-match identifier lookup)."""

import pickle
from pathlib import Path

import pytest

from src.retrieval.classical_ir import IdentifierRetriever


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs():
    return [
        {
            "doc_id": "NK-001",
            "identifiers": {
                "imoNumber": ["IMO9301407"],
                "mmsi": ["123456789"],
            },
        },
        {
            "doc_id": "NK-002",
            "identifiers": {
                "imoNumber": ["IMO1234567"],
            },
        },
        {
            "doc_id": "NK-003",
            "identifiers": {
                "mmsi": ["123456789"],  # shares MMSI with NK-001
            },
        },
        {
            "doc_id": "NK-004",
            "identifiers": {},
        },
    ]


@pytest.fixture
def retriever(sample_docs):
    r = IdentifierRetriever()
    r.build_index(sample_docs)
    return r


# ── Exact match ─────────────────────────────────────────────────────────────

def test_exact_match(retriever):
    results = retriever.search("IMO9301407")
    assert len(results) == 1
    assert results[0] == ("NK-001", 1.0)


def test_no_match(retriever):
    results = retriever.search("IMO0000000")
    assert results == []


# ── Case-insensitive normalisation ──────────────────────────────────────────

def test_case_insensitive_lookup(retriever):
    results = retriever.search("imo9301407")
    assert len(results) == 1
    assert results[0] == ("NK-001", 1.0)


def test_whitespace_normalisation(retriever):
    results = retriever.search("  IMO9301407  ")
    assert len(results) == 1
    assert results[0] == ("NK-001", 1.0)


# ── Multiple docs sharing an identifier ─────────────────────────────────────

def test_shared_identifier(retriever):
    results = retriever.search("123456789")
    doc_ids = [doc_id for doc_id, _ in results]
    assert "NK-001" in doc_ids
    assert "NK-003" in doc_ids
    assert len(results) == 2
    assert all(score == 1.0 for _, score in results)


# ── looks_like_identifier ───────────────────────────────────────────────────

class TestLooksLikeIdentifier:
    def test_imo(self):
        assert IdentifierRetriever.looks_like_identifier("IMO9301407") is True
        assert IdentifierRetriever.looks_like_identifier("imo9301407") is True

    def test_mmsi(self):
        assert IdentifierRetriever.looks_like_identifier("123456789") is True

    def test_lei(self):
        assert IdentifierRetriever.looks_like_identifier("529900T8BM49AURSDO55") is True

    def test_email(self):
        assert IdentifierRetriever.looks_like_identifier("user@example.com") is True

    def test_crypto_wallet(self):
        assert IdentifierRetriever.looks_like_identifier(
            "1a2b3c4d5e6f7a8b9c0d1e2f3a4b"
        ) is True

    def test_generic_alphanumeric(self):
        assert IdentifierRetriever.looks_like_identifier("ABC123") is True

    def test_plain_name_rejected(self):
        assert IdentifierRetriever.looks_like_identifier("Vladimir Putin") is False

    def test_short_string_rejected(self):
        assert IdentifierRetriever.looks_like_identifier("AB") is False

    def test_long_string_rejected(self):
        assert IdentifierRetriever.looks_like_identifier("A" * 21) is False


# ── Save / load round-trip ──────────────────────────────────────────────────

def test_save_load_roundtrip(retriever, tmp_path):
    save_path = tmp_path / "identifier" / "index.pkl"
    retriever.save(save_path)

    loaded = IdentifierRetriever()
    loaded.load(save_path)

    # Same results after round-trip
    assert retriever.search("IMO9301407") == loaded.search("IMO9301407")
    assert retriever.search("123456789") == loaded.search("123456789")
    assert retriever.search("NOPE") == loaded.search("NOPE")
