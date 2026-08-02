"""Shared data models used across ingestion, retrieval, generation, and citation layers.

Keeping these in one place means every layer agrees on what a "chunk" or
"source document" looks like, instead of each module inventing its own dict shape.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Tier = Literal["general_guidelines", "disease_specific"]


class SourceDocument(BaseModel):
    """A single registered source document (one entry from configs/sources.yaml)."""

    id: str
    title: str
    tier: Tier
    condition: Optional[str] = None
    organization: str
    year: Optional[int] = None
    url: str
    license: str
    manual_download: bool = False

    @property
    def local_filename(self) -> str:
        return f"{self.id}.pdf"


class Chunk(BaseModel):
    """A single retrievable unit of text with full source provenance."""

    chunk_id: str
    text: str
    source_id: str
    source_title: str
    tier: Tier
    condition: Optional[str] = None
    organization: str
    year: Optional[int] = None
    url: str
    chunk_index: int = Field(description="Position of this chunk within its source document")


class Citation(BaseModel):
    """A citation attached to a generated answer, pointing back to a chunk."""

    chunk_id: str
    source_title: str
    organization: str
    year: Optional[int] = None
    url: str
    excerpt: str


class QueryResponse(BaseModel):
    """The final response returned by the API for a user query."""

    answer: str
    citations: list[Citation]
    confidence: float
