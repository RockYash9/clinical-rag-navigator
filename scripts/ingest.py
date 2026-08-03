"""Entrypoint: download configured sources, extract text, chunk, and write
the result to data/processed/chunks.jsonl.

Usage:
    python scripts/ingest.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from clinical_rag.ingestion.chunking import chunk_text
from clinical_rag.ingestion.loaders import (
    download_pdf,
    extract_text_from_pdf,
    load_sources_config,
)
from clinical_rag.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/processed/chunks.jsonl")


def main() -> None:
    load_dotenv()
    setup_logging()
    sources = load_sources_config()
    logger.info("Loaded %d source(s) from configs/sources.yaml", len(sources))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_file:
        for source in sources:
            try:
                pdf_path = download_pdf(source)
                text = extract_text_from_pdf(pdf_path)
            except Exception:
                logger.exception("Failed to ingest source %s — skipping", source.id)
                continue

            if not text.strip():
                logger.warning("No text extracted for %s — skipping", source.id)
                continue

            chunks = chunk_text(text, source)
            for chunk in chunks:
                out_file.write(chunk.model_dump_json() + "\n")
            total_chunks += len(chunks)

            logger.info("Ingested %s: %d chunks", source.id, len(chunks))

    logger.info("Done. %d total chunks written to %s", total_chunks, OUTPUT_PATH)


if __name__ == "__main__":
    main()
