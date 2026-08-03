"""Maps retrieved chunks to citations and computes an answer confidence score.

Important scope note: this scores *retrieval* quality (how well the
retrieved passages match the query, and whether multiple independent
sources agree) — it does NOT verify that the LLM's generated text is
faithful to that context. That would need a separate entailment/faithfulness
check comparing the generated answer against the cited passages, which is a
natural next improvement once the basic pipeline is working end-to-end.
"""
from __future__ import annotations

from clinical_rag.schemas import Chunk, Citation

EXCERPT_LENGTH = 200


def score_confidence(retrieved: list[tuple[Chunk, float]]) -> float:
    """Confidence = average retrieval similarity of the chunks actually used,
    with a small bonus when multiple distinct sources agree (cross-source
    corroboration is a meaningful trust signal in a medical context).
    """
    if not retrieved:
        return 0.0

    scores = [score for _, score in retrieved]
    avg_similarity = sum(scores) / len(scores)

    distinct_sources = len({chunk.source_id for chunk, _ in retrieved})
    source_diversity_bonus = min(distinct_sources - 1, 2) * 0.05  # up to +0.10

    confidence = min(avg_similarity + source_diversity_bonus, 1.0)
    return round(max(confidence, 0.0), 3)


def build_citations(retrieved: list[tuple[Chunk, float]]) -> list[Citation]:
    """Builds one Citation per retrieved chunk, in the same order used to
    build the generation prompt — so citation [N] in the answer corresponds
    to the Nth entry in this list.
    """
    citations = []
    for chunk, _score in retrieved:
        excerpt = chunk.text[:EXCERPT_LENGTH]
        if len(chunk.text) > EXCERPT_LENGTH:
            excerpt = excerpt.rsplit(" ", 1)[0] + "…"

        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                source_title=chunk.source_title,
                organization=chunk.organization,
                year=chunk.year,
                url=chunk.url,
                excerpt=excerpt,
            )
        )
    return citations
