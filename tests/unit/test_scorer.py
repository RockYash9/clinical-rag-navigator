"""Tests for citation mapping and confidence scoring."""
from clinical_rag.citation.scorer import build_citations, score_confidence
from clinical_rag.schemas import Chunk


def _make_chunk(chunk_id: str, source_id: str, text: str = "sample text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_id=source_id,
        source_title=f"Source {source_id}",
        tier="disease_specific",
        organization="Test Org",
        url="https://example.com",
        chunk_index=0,
    )


def test_score_confidence_empty() -> None:
    assert score_confidence([]) == 0.0


def test_score_confidence_single_source() -> None:
    retrieved = [(_make_chunk("1", "src_a"), 0.9), (_make_chunk("2", "src_a"), 0.8)]
    confidence = score_confidence(retrieved)
    assert confidence == round(0.85, 3)  # no diversity bonus — same source


def test_score_confidence_multi_source_bonus() -> None:
    same_source = [(_make_chunk("1", "src_a"), 0.9), (_make_chunk("2", "src_a"), 0.9)]
    multi_source = [(_make_chunk("1", "src_a"), 0.9), (_make_chunk("2", "src_b"), 0.9)]

    assert score_confidence(multi_source) > score_confidence(same_source)


def test_score_confidence_capped_at_one() -> None:
    retrieved = [(_make_chunk(str(i), f"src_{i}"), 1.0) for i in range(5)]
    assert score_confidence(retrieved) <= 1.0


def test_build_citations_preserves_order() -> None:
    retrieved = [
        (_make_chunk("1", "src_a"), 0.9),
        (_make_chunk("2", "src_b"), 0.8),
    ]
    citations = build_citations(retrieved)
    assert [c.chunk_id for c in citations] == ["1", "2"]


def test_build_citations_truncates_long_excerpts() -> None:
    long_text = "word " * 100
    chunk = _make_chunk("1", "src_a", text=long_text)
    citations = build_citations([(chunk, 0.9)])
    assert len(citations[0].excerpt) < len(long_text)
    assert citations[0].excerpt.endswith("…")
