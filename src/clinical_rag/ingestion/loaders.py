"""Loaders for raw source documents: config-driven download + text extraction.

Sources are registered in configs/sources.yaml, not hardcoded here — adding a
new document to the corpus means adding a YAML entry, not editing this module.
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests
import yaml
from pypdf import PdfReader

from clinical_rag.schemas import SourceDocument

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
USER_AGENT = "clinical-rag-navigator/0.1 (educational RAG project)"


def load_sources_config(path: str | Path = "configs/sources.yaml") -> list[SourceDocument]:
    """Read and validate the source registry."""
    raw = yaml.safe_load(Path(path).read_text())
    return [SourceDocument(**entry) for entry in raw["sources"]]


class SourceUnavailableError(RuntimeError):
    """Raised when a source can't be auto-downloaded (e.g. bot-blocked host)."""


def download_pdf(source: SourceDocument, dest_dir: str | Path = "data/raw") -> Path:
    """Download a source's PDF into data/raw/<tier>/, skipping if already present.

    Validates that the response is actually a PDF (checks the %PDF magic
    bytes) rather than trusting a 200 status — some hosts (e.g. iris.who.int)
    return a small HTML page to bot-like clients instead of an error code,
    which otherwise fails silently and surfaces later as a cryptic pypdf
    stream error.

    Returns the local path. Raises requests.HTTPError on a bad status, or
    SourceUnavailableError if the response isn't a real PDF — callers should
    catch and log rather than let one bad source kill a whole ingest run.
    """
    tier_dir = Path(dest_dir) / source.tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    dest_path = tier_dir / source.local_filename

    if dest_path.exists():
        logger.info("Already downloaded, skipping: %s", dest_path)
        return dest_path

    if source.manual_download:
        raise SourceUnavailableError(
            f"{source.id} is flagged manual_download in sources.yaml — "
            f"download {source.url} in a browser and save it to {dest_path}"
        )

    logger.info("Downloading %s from %s", source.id, source.url)
    response = requests.get(
        source.url,
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()

    if not response.content.startswith(b"%PDF"):
        raise SourceUnavailableError(
            f"{source.id}: response from {source.url} is not a PDF "
            f"(content-type={response.headers.get('content-type')}). "
            f"The host may be blocking automated requests — try downloading "
            f"manually and saving to {dest_path}, then mark manual_download: "
            f"true in sources.yaml."
        )

    dest_path.write_bytes(response.content)
    logger.info("Saved %s (%d bytes)", dest_path, len(response.content))
    return dest_path


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract raw text from a PDF, page by page, joined with double newlines.

    This is intentionally simple — no layout reconstruction. Tables and
    multi-column layouts will need special handling later if a specific
    guideline's text comes out garbled (flag it in docs/data_sources.md).
    """
    reader = PdfReader(str(path))
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            logger.warning("No extractable text on page %d of %s", i, path)
        pages_text.append(text)
    return "\n\n".join(pages_text)
