"""Loaders for raw source documents: config-driven download + text extraction.

Sources are registered in configs/sources.yaml, not hardcoded here — adding a
new document to the corpus means adding a YAML entry, not editing this module.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from pypdf import PdfReader

from clinical_rag.schemas import SourceDocument

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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

    Also strips repeated running headers/footers (e.g. a journal citation
    line like "DOI: 10.xxxx ... ISSN: xxxx" that appears on nearly every
    page) before joining. Academic PDFs commonly repeat this front-matter
    on every page, and without stripping it, that noise gets chunked
    alongside real content — diluting retrieval and confusing generation
    when a chunk turns out to be mostly repeated metadata rather than
    substantive text.

    Beyond that, this is intentionally simple — no layout reconstruction.
    Tables and multi-column layouts will need special handling later if a
    specific guideline's text comes out garbled (flag it in docs/data_sources.md).
    """
    reader = PdfReader(str(path))
    pages_lines: list[list[str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            logger.warning("No extractable text on page %d of %s", i, path)
        pages_lines.append(text.splitlines())

    num_pages = len(pages_lines)
    if num_pages > 3:
        line_counts: dict[str, int] = {}
        for lines in pages_lines:
            for line in set(line.strip() for line in lines if line.strip()):
                line_counts[line] = line_counts.get(line, 0) + 1

        # A line appearing on most pages is a running header/footer, not
        # article body text — real content rarely repeats verbatim page
        # after page. Require a minimum length so short common words/section
        # numbers aren't mistaken for headers.
        repeat_threshold = max(3, int(num_pages * 0.4))
        boilerplate_lines = {
            line
            for line, count in line_counts.items()
            if count >= repeat_threshold and len(line) > 8
        }
        if boilerplate_lines:
            logger.info(
                "Stripping %d repeated header/footer line(s) from %s",
                len(boilerplate_lines),
                path,
            )
    else:
        boilerplate_lines = set()

    pages_text = [
        "\n".join(line for line in lines if line.strip() not in boilerplate_lines)
        for lines in pages_lines
    ]
    return "\n\n".join(pages_text)


# Phrases that show up on bot-detection / interstitial pages rather than
# real article content — if we see these, something blocked us rather than
# served the actual page, even though the HTTP status was 200.
_BOT_BLOCK_SIGNATURES = (
    "checking your browser",
    "recaptcha",
    "are you a robot",
    "enable javascript",
    "access denied",
)


def download_html(source: SourceDocument, dest_dir: str | Path = "data/raw") -> Path:
    """Download a source's HTML page into data/raw/<tier>/, skipping if already present.

    Same defensive validation philosophy as download_pdf: a 200 status isn't
    enough to trust, since bot-detection interstitials often return 200 with
    a challenge page instead of a real error code.
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
            f"download {source.url} in a browser (save page as HTML) to {dest_path}"
        )

    logger.info("Downloading %s from %s", source.id, source.url)
    response = requests.get(
        source.url,
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type.lower():
        raise SourceUnavailableError(
            f"{source.id}: response from {source.url} is not HTML "
            f"(content-type={content_type})."
        )

    lowered = response.text[:2000].lower()
    if any(sig in lowered for sig in _BOT_BLOCK_SIGNATURES):
        raise SourceUnavailableError(
            f"{source.id}: response from {source.url} looks like a bot-detection "
            f"page, not the real article. Try downloading manually and saving to "
            f"{dest_path}, then mark manual_download: true in sources.yaml."
        )

    dest_path.write_text(response.text, encoding="utf-8")
    logger.info("Saved %s (%d chars)", dest_path, len(response.text))
    return dest_path


def extract_text_from_html(path: str | Path) -> str:
    """Extract article text from a saved HTML page, stripping boilerplate.

    This is a generic heuristic, not site-specific scraping: strip script/
    style/nav/header/footer tags, prefer a <main>/<article>/#maincontent
    container if one exists, and fall back to the full body otherwise. It
    won't be perfect on every site's markup, but it avoids hardcoding
    selectors for one particular domain.
    """
    html = Path(path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find(id="maincontent")
        or soup.find("article")
        or soup.body
        or soup
    )

    lines = [line.strip() for line in main.get_text(separator="\n").splitlines()]
    return "\n".join(line for line in lines if line)


def download_source(source: SourceDocument, dest_dir: str | Path = "data/raw") -> Path:
    """Dispatches to download_pdf or download_html based on source.source_type."""
    if source.source_type == "html":
        return download_html(source, dest_dir)
    return download_pdf(source, dest_dir)


def extract_text(source: SourceDocument, path: str | Path) -> str:
    """Dispatches to the right text extractor based on source.source_type."""
    if source.source_type == "html":
        return extract_text_from_html(path)
    return extract_text_from_pdf(path)
