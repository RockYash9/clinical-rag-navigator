"""Tests for degenerate-answer detection in the RAG pipeline.

Covers the failure mode observed in practice: a small local LLM
occasionally returns just a bare citation marker (e.g. "[1]") instead of
a real answer, even when retrieval worked correctly.
"""
from clinical_rag.pipeline import _looks_degenerate


def test_bare_citation_marker_is_degenerate() -> None:
    assert _looks_degenerate("[1]") is True


def test_multiple_bare_citation_markers_is_degenerate() -> None:
    assert _looks_degenerate("[1][2]") is True


def test_bare_citation_marker_with_whitespace_is_degenerate() -> None:
    assert _looks_degenerate("  [1]  ") is True


def test_very_short_answer_is_degenerate() -> None:
    assert _looks_degenerate("Yes.") is True


def test_real_answer_with_citation_is_not_degenerate() -> None:
    answer = "[1] Diabetes is a metabolic disorder characterized by chronic hyperglycemia."
    assert _looks_degenerate(answer) is False


def test_real_short_answer_with_multiple_citations_is_not_degenerate() -> None:
    assert _looks_degenerate("Yes, per [1] and [2].") is False
