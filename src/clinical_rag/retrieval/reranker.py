"""Re-ranks retrieved chunks using a hybrid of vector similarity and lexical overlap.

Pure semantic search can occasionally rank a topically-similar-but-wrong
passage above one with an exact match on a drug name, dosage, or threshold
value — a real risk in a medical context, where "ACE inhibitor" and "ARB"
are semantically close but clinically distinct. This re-ranker nudges the
embedding similarity score using simple keyword overlap as a free, cheap
complement. It's intentionally simple: a cross-encoder re-ranking model is
the natural upgrade path if retrieval quality needs improving later.
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
) -> list[tuple[Chunk, float]]:
    """`candidates` is (chunk, semantic_score) pairs from vector search, as
    returned by VectorStore.search(). Returns the top_k re-ranked pairs,
    best first, with scores now a weighted blend of semantic + lexical.
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
    return rescored[:top_k]
