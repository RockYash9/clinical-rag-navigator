"""Tests for the hybrid semantic + lexical reranker."""
from clinical_rag.retrieval.reranker import rerank
from clinical_rag.schemas import Chunk


def _make_chunk(chunk_id: str, text: str, source_id: str = "test_source") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_id=source_id,
        source_title="Test Source",
        tier="general_guidelines",
        organization="Test Org",
        url="https://example.com",
        chunk_index=0,
    )


def test_rerank_boosts_exact_keyword_match() -> None:
    # Chunk B has a lower semantic score but contains an exact match for
    # "metformin" — the lexical boost should be able to move it up.
    chunk_a = _make_chunk("a", "General discussion of diabetes management approaches.")
    chunk_b = _make_chunk("b", "Metformin is the first-line pharmacological agent for type 2 diabetes.")

    candidates = [(chunk_a, 0.75), (chunk_b, 0.60)]
    result = rerank("metformin first-line treatment", candidates, top_k=2, lexical_weight=0.5)

    result_ids = [chunk.chunk_id for chunk, _ in result]
    assert "b" in result_ids


def test_rerank_respects_top_k() -> None:
    candidates = [(_make_chunk(str(i), f"chunk {i}"), 1.0 - i * 0.1) for i in range(5)]
    result = rerank("chunk", candidates, top_k=2)
    assert len(result) == 2


def test_rerank_empty_candidates() -> None:
    assert rerank("anything", [], top_k=4) == []


def test_rerank_empty_query_returns_original_order() -> None:
    candidates = [(_make_chunk("a", "text a"), 0.9), (_make_chunk("b", "text b"), 0.5)]
    result = rerank("   ", candidates, top_k=2)
    assert [c.chunk_id for c, _ in result] == ["a", "b"]


def test_rerank_caps_chunks_per_source() -> None:
    # Regression test: a single long/dense document should not be able to
    # fill every retrieval slot when other sources are available.
    candidates = [
        (_make_chunk("b1", "diabetes overview", source_id="big_doc"), 0.90),
        (_make_chunk("b2", "diabetes header noise", source_id="big_doc"), 0.88),
        (_make_chunk("b3", "diabetes more content", source_id="big_doc"), 0.85),
        (_make_chunk("b4", "diabetes even more", source_id="big_doc"), 0.83),
        (_make_chunk("o1", "diabetes from another source", source_id="other_doc"), 0.82),
        (_make_chunk("o2", "diabetes from another source too", source_id="other_doc"), 0.80),
    ]
    result = rerank("diabetes", candidates, top_k=4, max_per_source=2)
    source_ids = [c.source_id for c, _ in result]
    assert source_ids.count("big_doc") <= 2
    assert "other_doc" in source_ids


def test_rerank_backfills_when_diversity_unavailable() -> None:
    # If there genuinely aren't enough distinct sources to fill top_k while
    # respecting the cap, backfill from the capped source rather than
    # returning fewer results than requested.
    candidates = [
        (_make_chunk(str(i), f"chunk {i}", source_id="only_source"), 1.0 - i * 0.1)
        for i in range(5)
    ]
    result = rerank("chunk", candidates, top_k=4, max_per_source=2)
    assert len(result) == 4
