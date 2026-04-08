"""Unit tests for RAGPipeline."""

from unittest.mock import patch, MagicMock

import pytest

from src.rag.rag_pipeline import RAGPipeline


def make_doc(
    doc_id="doc1",
    name="Acme Corp",
    text_blob="Acme Corp is a sanctioned entity based in Russia involved in arms trafficking",
    schema="Company",
    country="RU",
    program_id="EU-RU-2022",
    sanctions=None,
    ownership=None,
):
    metadata = {"schema": schema, "country": country, "programId": program_id, "name": name}
    doc = {
        "doc_id": doc_id,
        "text_blob": text_blob,
        "identifiers": {},
        "metadata": metadata,
    }
    if sanctions is not None:
        doc["sanctions"] = sanctions
    if ownership is not None:
        doc["ownership"] = ownership
    return doc


@pytest.fixture
def rag():
    with patch("src.rag.rag_pipeline.pipeline") as mock_pipeline_factory:
        mock_hf = MagicMock()
        mock_hf.return_value = [{"generated_text": "mock summary"}]
        mock_pipeline_factory.return_value = mock_hf
        instance = RAGPipeline(k=3, max_tokens_per_doc=100)
        instance._mock_hf = mock_hf
        yield instance


def test_build_context_contains_entity_name(rag):
    doc = make_doc(name="Acme Corp")
    context = rag.build_context([doc])
    assert "Acme Corp" in context


def test_build_context_truncates_text_blob(rag):
    long_text = " ".join(f"tok{i}" for i in range(500))
    doc = make_doc(text_blob=long_text)
    rag.max_tokens_per_doc = 10
    context = rag.build_context([doc])
    assert "tok9" in context
    assert "tok10" not in context


def test_build_context_omits_sanctions_when_absent(rag):
    doc = make_doc()
    assert "sanctions" not in doc
    context = rag.build_context([doc])
    assert "Sanctions:" not in context


def test_build_context_omits_ownership_when_absent(rag):
    doc = make_doc()
    assert "ownership" not in doc
    context = rag.build_context([doc])
    assert "Ownership:" not in context


def test_build_context_separates_documents_with_dashes(rag):
    doc1 = make_doc(doc_id="d1", name="First Co")
    doc2 = make_doc(doc_id="d2", name="Second Co")
    context = rag.build_context([doc1, doc2])
    assert "---" in context
    assert "First Co" in context
    assert "Second Co" in context


def test_build_context_handles_empty_list(rag):
    context = rag.build_context([])
    assert context == ""


def test_build_prompt_includes_query(rag):
    prompt = rag.build_prompt("who is sanctioned in Russia", "some context")
    assert "who is sanctioned in Russia" in prompt


def test_build_prompt_includes_context(rag):
    prompt = rag.build_prompt("query", "the context block")
    assert "the context block" in prompt


def test_build_prompt_includes_instruction(rag):
    prompt = rag.build_prompt("q", "c")
    assert "Summarise the following sanctioned entities" in prompt
    assert "Only use information from the provided context" in prompt


def test_generate_returns_required_keys(rag):
    doc = make_doc()
    result = rag.generate("a query", [doc])
    assert set(result.keys()) == {"summary", "context", "query"}
    assert result["query"] == "a query"
    assert result["summary"] == "mock summary"


def test_generate_passes_truncation_true(rag):
    doc = make_doc()
    rag.generate("q", [doc])
    _, kwargs = rag._mock_hf.call_args
    assert kwargs.get("truncation") is True
    assert kwargs.get("max_new_tokens") == 200


def test_generate_respects_k(rag):
    rag.k = 2
    docs = [make_doc(doc_id=f"d{i}", name=f"Entity{i}") for i in range(5)]
    with patch.object(rag, "build_context", wraps=rag.build_context) as spy:
        rag.generate("q", docs)
        passed = spy.call_args[0][0]
        assert len(passed) == 2
