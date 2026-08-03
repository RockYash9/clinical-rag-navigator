"""Wraps a sentence-transformers model to embed chunks and queries.

Model choice is resolved in priority order: explicit constructor arg ->
EMBEDDING_MODEL env var (from .env) -> configs/config.yaml default. This lets
the model be swapped (e.g. to a biomedical-specific embedding model later)
without touching code.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from clinical_rag.utils.config import load_config

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str | None = None, batch_size: int | None = None):
        config = load_config().get("embedding", {})

        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.batch_size = batch_size or config.get("batch_size", 32)

        logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Vectors are L2-normalized so that a FAISS
        inner-product index (IndexFlatIP) is equivalent to cosine similarity
        search — normalizing here means retrieval doesn't need to think
        about distance metrics later.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Same normalization as embed_texts."""
        return self.embed_texts([text])[0]
