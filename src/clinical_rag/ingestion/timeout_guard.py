"""Runs text extraction in a subprocess with a hard timeout.

pypdf can occasionally hang or become extremely slow on certain malformed
or unusually complex PDFs (deeply nested XObjects, huge embedded image
streams, etc.) — this happened in practice with a diagram-heavy sensor
review PDF, which required a manual Ctrl+C to recover from.

A thread-based timeout can't actually stop a hung thread in Python (there's
no clean way to kill one), so this uses a subprocess instead, which CAN be
forcibly terminated if it exceeds the timeout. That guarantees one bad PDF
can never block an entire ingestion run again.

IMPORTANT ordering note: this reads from the queue BEFORE calling
process.join(). multiprocessing.Queue is backed by a pipe with a limited
buffer — if the child's result is larger than that buffer, the child
blocks trying to write it until someone reads, while a parent that calls
join() first blocks waiting for the child to exit. Neither can proceed:
a deadlock, lasting until the *outer* join timeout fires regardless of how
fast extraction actually was. queue.get(timeout=...) avoids this by
draining the pipe first, which is what actually lets the child finish.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue as queue_module
from pathlib import Path

from clinical_rag.ingestion.loaders import extract_text
from clinical_rag.schemas import SourceDocument

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 180
JOIN_GRACE_SECONDS = 5


def _extract_worker(source: SourceDocument, path: str, result_queue: "mp.Queue") -> None:
    try:
        text = extract_text(source, path)
        result_queue.put(("ok", text))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def extract_text_with_timeout(
    source: SourceDocument,
    path: str | Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Runs extract_text(source, path) in a subprocess, killing it if it
    exceeds `timeout` seconds.

    Raises TimeoutError if the extraction hung, or RuntimeError wrapping
    whatever exception occurred inside the subprocess — either way, the
    caller can catch and skip this one source rather than the whole run
    getting stuck.
    """
    result_queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_extract_worker, args=(source, str(path), result_queue))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except queue_module.Empty:
        logger.warning("Extraction for %s exceeded %ds — terminating", source.id, timeout)
        process.terminate()
        process.join()
        raise TimeoutError(
            f"Text extraction timed out after {timeout}s for {source.id} — the "
            f"file may be malformed or unusually complex (e.g. many embedded "
            f"images/forms). Consider skipping this source in sources.yaml."
        )

    # Result already received — this should return almost immediately.
    # If the child is somehow still lingering, terminate rather than block.
    process.join(JOIN_GRACE_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join()

    if status == "error":
        raise RuntimeError(f"Extraction failed for {source.id}: {payload}")
    return payload
