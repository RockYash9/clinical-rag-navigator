"""Re-ranks retrieved chunks using a hybrid of vector similarity and lexical overlap.

Pure semantic search can occasionally rank a topically-similar-but-wrong
passage above one with an exact match on a drug name, dosage, or threshold
value — a real risk in a medical context, where "ACE inhibitor" and "ARB"
are semantically close but clinically distinct. This re-ranker nudges the
embedding similarity score using simple keyword overlap as a free, cheap
complement. It's intentionally simple: a cross-encoder re-ranking model is
the natural upgrade path if retrieval quality needs improving later.

It also enforces a per-source diversity cap: a long, densely-written
document can otherwise score well on many of its own chunks for a broad
query and end up filling every slot, starving out other sources entirely
even when they'd add real value (cross-source agreement is also what the
confidence scorer rewards, so source diversity matters twice over).
"""
from __future__ import annotations

import re

from clinical_rag.schemas import Chunk

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def rerank(
    query: str,
    candidates: list[tuple[Chunk, float]],
    top_k: int,
    lexical_weight: float = 0.2,
    max_per_source: int = 2,
) -> list[tuple[Chunk, float]]:
    """`candidates` is (chunk, semantic_score) pairs from vector search, as
    returned by VectorStore.search(). Returns the top_k re-ranked pairs,
    best first, with scores now a weighted blend of semantic + lexical, and
    at most `max_per_source` chunks from any single source_id (backfilled
    past the cap if there aren't enough distinct sources to fill top_k).
    """
    if not candidates:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return candidates[:top_k]

    rescored = []
    for chunk, semantic_score in candidates:
        chunk_terms = _tokenize(chunk.text)
        lexical_overlap = len(query_terms & chunk_terms) / len(query_terms)
        combined_score = (1 - lexical_weight) * semantic_score + lexical_weight * lexical_overlap
        rescored.append((chunk, combined_score))

    rescored.sort(key=lambda pair: pair[1], reverse=True)

    selected: list[tuple[Chunk, float]] = []
    selected_ids: set[str] = set()
    per_source_count: dict[str, int] = {}

    for chunk, score in rescored:
        if per_source_count.get(chunk.source_id, 0) >= max_per_source:
            continue
        selected.append((chunk, score))
        selected_ids.add(chunk.chunk_id)
        per_source_count[chunk.source_id] = per_source_count.get(chunk.source_id, 0) + 1
        if len(selected) >= top_k:
            return selected

    # Cap left us short (too few distinct sources retrieved) — backfill
    # with the next-best chunks regardless of source.
    for chunk, score in rescored:
        if chunk.chunk_id in selected_ids:
            continue
        selected.append((chunk, score))
        if len(selected) >= top_k:
            break

    return selected
