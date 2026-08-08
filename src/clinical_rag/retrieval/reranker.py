"""Re-ranks retrieved chunks, with a pluggable scoring strategy and a shared
per-source diversity cap.

Two scoring strategies are available:

1. Lexical hybrid (default, always available, no extra model): blends the
   embedding similarity score with simple keyword overlap. Pure semantic
   search can occasionally rank a topically-similar-but-wrong passage above
   one with an exact match on a drug name, dosage, or threshold value — a
   real risk in a medical context, where "ACE inhibitor" and "ARB" are
   semantically close but clinically distinct.

2. Cross-encoder (optional, see cross_encoder.py): a small model that scores
   the (query, passage) pair jointly rather than comparing independently
   computed embeddings — meaningfully more accurate at judging relevance,
   especially once a corpus has many similar documents competing for the
   same slots (e.g. several papers all about resistant hypertension).

Both strategies share the same diversity cap: a long, densely-written
document can otherwise score well on many of its own chunks for a broad
query and end up filling every slot, starving out other sources entirely
even when they'd add real value (cross-source agreement is also what the
confidence scorer rewards, so source diversity matters twice over).
"""
from __future__ import annotations

import re
from typing import Protocol

from clinical_rag.schemas import Chunk

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


class CrossEncoderLike(Protocol):
    """Anything with a `.score(query, texts) -> list[float]` method qualifies —
    this is a Protocol rather than a concrete import so unit tests can pass a
    lightweight fake without loading the real model."""

    def score(self, query: str, texts: list[str]) -> list[float]: ...


def _apply_diversity_cap(
    scored: list[tuple[Chunk, float]],
    top_k: int,
    max_per_source: int,
) -> list[tuple[Chunk, float]]:
    """`scored` must already be sorted best-first. Returns up to top_k items
    with at most max_per_source from any single source_id, backfilling past
    the cap if there aren't enough distinct sources to fill top_k.
    """
    selected: list[tuple[Chunk, float]] = []
    selected_ids: set[str] = set()
    per_source_count: dict[str, int] = {}

    for chunk, score in scored:
        if per_source_count.get(chunk.source_id, 0) >= max_per_source:
            continue
        selected.append((chunk, score))
        selected_ids.add(chunk.chunk_id)
        per_source_count[chunk.source_id] = per_source_count.get(chunk.source_id, 0) + 1
        if len(selected) >= top_k:
            return selected

    # Cap left us short (too few distinct sources retrieved) — backfill
    # with the next-best chunks regardless of source.
    for chunk, score in scored:
        if chunk.chunk_id in selected_ids:
            continue
        selected.append((chunk, score))
        if len(selected) >= top_k:
            break

    return selected


def rerank(
    query: str,
    candidates: list[tuple[Chunk, float]],
    top_k: int,
    lexical_weight: float = 0.2,
    max_per_source: int = 2,
    cross_encoder: CrossEncoderLike | None = None,
) -> list[tuple[Chunk, float]]:
    """`candidates` is (chunk, semantic_score) pairs from vector search, as
    returned by VectorStore.search(). Returns the top_k re-ranked pairs,
    best first, with at most `max_per_source` chunks from any single
    source_id.

    If `cross_encoder` is provided, it scores each (query, chunk.text) pair
    directly and that score is used for ranking. Otherwise falls back to
    the lexical + semantic hybrid.
    """
    if not candidates:
        return []

    if cross_encoder is not None:
        texts = [chunk.text for chunk, _ in candidates]
        scores = cross_encoder.score(query, texts)
        rescored = list(zip((chunk for chunk, _ in candidates), scores))
        rescored.sort(key=lambda pair: pair[1], reverse=True)
        return _apply_diversity_cap(rescored, top_k, max_per_source)

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
    return _apply_diversity_cap(rescored, top_k, max_per_source)
