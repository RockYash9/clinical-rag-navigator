"""Splits extracted document text into overlapping, source-tagged chunks.

Word-based sliding window chunking. Simple and predictable — good enough for
a first working pipeline. If dosage tables or numbered lists start getting
split awkwardly once real guideline text is chunked, that's the signal to
move to a structure-aware splitter (e.g. splitting on headings first).
"""
from __future__ import annotations

from clinical_rag.schemas import Chunk, SourceDocument


def chunk_text(
    text: str,
    source: SourceDocument,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """Split `text` into overlapping word-count-based chunks tagged with source metadata.

    Args:
        text: Full extracted text of the document.
        source: The SourceDocument this text came from (for provenance tagging).
        chunk_size: Target words per chunk.
        overlap: Words shared between consecutive chunks, so context isn't
            lost at chunk boundaries.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    chunk_index = 0

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break

        chunks.append(
            Chunk(
                chunk_id=f"{source.id}::{chunk_index:04d}",
                text=" ".join(window),
                source_id=source.id,
                source_title=source.title,
                tier=source.tier,
                condition=source.condition,
                organization=source.organization,
                year=source.year,
                url=source.url,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1

        if start + chunk_size >= len(words):
            break

    return chunks
