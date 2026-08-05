"""Orchestrates the full retrieve -> rerank -> generate -> cite pipeline.

This is the single place that wires together every layer (embeddings,
vector store, reranker, LLM, citation/confidence). The API layer calls this
and stays thin; a future CLI or batch-eval script would call it the same way.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from clinical_rag.citation.scorer import build_citations, score_confidence
from clinical_rag.embeddings.embedder import Embedder
from clinical_rag.generation.llm_client import LLMClient
from clinical_rag.generation.prompt_templates import build_prompt
from clinical_rag.retrieval.reranker import rerank
from clinical_rag.retrieval.vector_store import VectorStore
from clinical_rag.schemas import QueryResponse
from clinical_rag.utils.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = "data/vector_store/index.faiss"

# Small local models occasionally collapse into a near-empty output — e.g.
# just a citation marker like "[1]" with no actual sentence. This is a
# generation-layer failure, not a retrieval problem, so it's handled here
# with a retry rather than by continuing to chase every retrieval-side
# trigger for it.
_MIN_ANSWER_LENGTH = 20
_CITATION_ONLY_RE = re.compile(r"^\s*(\[\d+\]\s*)+\.?\s*$")

FALLBACK_ANSWER = (
    "The system retrieved relevant passages but couldn't generate a complete "
    "answer this time. Try rephrasing the question — the sources below are "
    "still the actual retrieved guideline text and may answer it directly."
)


def _looks_degenerate(answer: str) -> bool:
    stripped = answer.strip()
    return len(stripped) < _MIN_ANSWER_LENGTH or bool(_CITATION_ONLY_RE.match(stripped))


class RAGPipeline:
    def __init__(self, index_path: str | Path = DEFAULT_INDEX_PATH):
        retrieval_config = load_config().get("retrieval", {})
        self.top_k = retrieval_config.get("top_k", 8)
        self.top_k_after_rerank = retrieval_config.get("top_k_after_rerank", 4)
        self.max_per_source = retrieval_config.get("max_per_source", 2)

        logger.info("Loading vector store from %s", index_path)
        self.store = VectorStore.load(index_path)
        logger.info("Vector store loaded: %d chunks indexed", self.store.index.ntotal)

        self.embedder = Embedder()
        self.llm = LLMClient()

    def query(self, question: str) -> QueryResponse:
        query_vector = self.embedder.embed_query(question)
        candidates = self.store.search(query_vector, top_k=self.top_k)

        if not candidates:
            return QueryResponse(
                answer="The provided guidelines do not cover this.",
                citations=[],
                confidence=0.0,
            )

        reranked = rerank(
            question, candidates, top_k=self.top_k_after_rerank, max_per_source=self.max_per_source
        )
        top_chunks = [chunk for chunk, _score in reranked]

        prompt = build_prompt(question, top_chunks)
        answer = self.llm.generate(prompt)

        if _looks_degenerate(answer):
            logger.warning("Generation looked degenerate (%r) — retrying with higher temperature", answer)
            answer = self.llm.generate(prompt, temperature=0.4)

        if _looks_degenerate(answer):
            logger.warning("Retry also degenerate (%r) — returning fallback message", answer)
            answer = FALLBACK_ANSWER

        return QueryResponse(
            answer=answer,
            citations=build_citations(reranked),
            confidence=score_confidence(reranked),
        )
