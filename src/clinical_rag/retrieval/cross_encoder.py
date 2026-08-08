"""Cross-encoder reranker: scores (query, passage) pairs jointly rather than
comparing independently computed embeddings.

This is the natural upgrade path called out in reranker.py's own docstring.
A bi-encoder (what embeds chunks and queries separately for vector search)
has to compress meaning into a fixed vector without seeing the query and
passage together; a cross-encoder sees both at once and can attend across
them, which is a meaningfully stronger relevance signal — at the cost of
being too slow to run over an entire corpus, which is exactly why it's used
here only to re-score the already-retrieved top candidates, not the full
vector search.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — small, free, fast enough to
score a handful of candidates in milliseconds on CPU, and specifically
trained for passage relevance ranking (MS MARCO).

IMPORTANT: this model outputs raw logits, not a bounded [0, 1] score. Both
score_confidence() (citation/scorer.py) and the rest of this codebase treat
retrieval scores as roughly [0, 1] similarities. A sigmoid transform is
applied here — the standard way this specific model's scores are used in
practice — so callers downstream don't need to know or care which scoring
strategy produced the number they're looking at.
"""
from __future__ import annotations

import logging
import math
import os

from clinical_rag.utils.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        # Deferred import: sentence-transformers is already a project
        # dependency, but importing CrossEncoder (and its torch backend)
        # is not free — only pay that cost if cross-encoder reranking is
        # actually enabled.
        from sentence_transformers import CrossEncoder

        config = load_config().get("retrieval", {})
        self.model_name = (
            model_name
            or os.getenv("CROSS_ENCODER_MODEL")
            or config.get("cross_encoder_model", DEFAULT_MODEL)
        )

        logger.info("Loading cross-encoder reranker model: %s", self.model_name)
        self.model = CrossEncoder(self.model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Scores each text's relevance to the query, sigmoid-normalized to
        roughly (0, 1) so downstream confidence scoring behaves the same
        regardless of which reranking strategy produced the score. Higher
        is more relevant.
        """
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        raw_scores = self.model.predict(pairs)
        return [_sigmoid(float(score)) for score in raw_scores]
