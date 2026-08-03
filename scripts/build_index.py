"""Entrypoint: embed processed chunks and build the FAISS index -> data/vector_store/.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from clinical_rag.embeddings.embedder import Embedder
from clinical_rag.retrieval.vector_store import VectorStore
from clinical_rag.schemas import Chunk
from clinical_rag.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/vector_store/index.faiss")


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))
    return chunks


def main() -> None:
    load_dotenv()
    setup_logging()

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found — run scripts/ingest.py first")

    chunks = load_chunks(CHUNKS_PATH)
    logger.info("Loaded %d chunks from %s", len(chunks), CHUNKS_PATH)

    if not chunks:
        logger.warning("No chunks to embed — nothing to do")
        return

    embedder = Embedder()
    logger.info(
        "Embedding %d chunks with %s (dimension=%d)",
        len(chunks),
        embedder.model_name,
        embedder.dimension,
    )

    texts = [chunk.text for chunk in chunks]
    vectors = embedder.embed_texts(texts)

    store = VectorStore(dim=vectors.shape[1])
    store.add(chunks, vectors)
    store.save(INDEX_PATH)

    logger.info("Done. Index built with %d vectors -> %s", store.index.ntotal, INDEX_PATH)


if __name__ == "__main__":
    main()
