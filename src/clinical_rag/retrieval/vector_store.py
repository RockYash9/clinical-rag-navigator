"""FAISS-backed vector store: persistence + similarity search.

FAISS only stores vectors and returns integer positions — it knows nothing
about chunk text or source metadata. This class keeps a parallel `chunks`
list in the exact same order as vectors were added, so a search result's
integer index can be mapped straight back to the Chunk it came from. The
index and that chunk list are saved/loaded together as a pair; loading one
without the other would silently desync positions, so both live at the same
base path with different suffixes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

from clinical_rag.schemas import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        # Inner product on L2-normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunks/vectors length mismatch: {len(chunks)} chunks vs {vectors.shape[0]} vectors"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} does not match store dim {self.dim}")

        self.index.add(vectors.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 8) -> list[tuple[Chunk, float]]:
        """Returns (chunk, similarity_score) pairs, best match first."""
        if self.index.ntotal == 0:
            return []

        query_vector = query_vector.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str | Path) -> None:
        """Saves the FAISS index to `path` and chunk metadata to a sibling
        `<path>.chunks.jsonl` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))

        meta_path = path.with_suffix(path.suffix + ".chunks.jsonl")
        with meta_path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json() + "\n")

        logger.info(
            "Saved index (%d vectors) to %s and metadata to %s",
            self.index.ntotal,
            path,
            meta_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No index found at {path} — run scripts/build_index.py first"
            )

        index = faiss.read_index(str(path))

        meta_path = path.with_suffix(path.suffix + ".chunks.jsonl")
        chunks = [
            Chunk.model_validate_json(line)
            for line in meta_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index/metadata mismatch: {index.ntotal} vectors but {len(chunks)} chunks "
                f"loaded from {meta_path}. The two files may be out of sync."
            )

        store = cls(dim=index.d)
        store.index = index
        store.chunks = chunks
        return store
