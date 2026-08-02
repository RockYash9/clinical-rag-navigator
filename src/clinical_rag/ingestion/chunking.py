"""Splits cleaned documents into overlapping chunks for embedding.

TODO:
    - chunk_text(text, chunk_size, overlap) -> list[Chunk]
Chunk should carry: text, source_id, section_title, tier, chunk_index.
Chunking strategy must avoid splitting tables/dosage lists mid-way.
"""
