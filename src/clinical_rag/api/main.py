"""FastAPI entrypoint exposing the RAG pipeline.

The vector index is loaded once at startup (not per-request) since loading
FAISS + the embedding model has real latency — see the `lifespan` context
manager below. If no index exists yet, the app still starts (so /health
works), but /query returns a clear 503 telling you to build the index first,
rather than crashing on startup.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clinical_rag.pipeline import RAGPipeline
from clinical_rag.schemas import QueryResponse
from clinical_rag.utils.logging_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

_pipeline: RAGPipeline | None = None


class UTF8JSONResponse(JSONResponse):
    """Explicitly declares charset=utf-8 on the Content-Type header.

    Starlette's default JSONResponse omits the charset param for
    application/json (technically spec-compliant, since JSON is UTF-8 by
    definition per RFC 8259) — but some HTTP clients, notably Windows
    PowerShell 5.1's Invoke-RestMethod, don't reliably infer UTF-8 without
    it and mis-decode multi-byte characters (e.g. "≥" becomes "â¥"). Being
    explicit here costs nothing and fixes it for every client.
    """

    media_type = "application/json; charset=utf-8"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    try:
        _pipeline = RAGPipeline()
        logger.info("Pipeline loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("Pipeline not loaded: %s", exc)
    yield


app = FastAPI(
    title="Clinical RAG Navigator",
    version="0.1.0",
    description="RAG system for querying clinical guidelines with cited, confidence-scored answers.",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="A clinical question to answer from the corpus")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded — run scripts/ingest.py then scripts/build_index.py first",
        )
    return _pipeline.query(request.question)


# Mounted last and at the root path, so /health and /query above are matched
# first. StaticFiles(html=True) serves static/index.html at "/" and any
# other file (style.css, app.js) at its own filename — this is the chat UI.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
