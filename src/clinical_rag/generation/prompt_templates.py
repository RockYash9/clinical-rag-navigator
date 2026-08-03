"""System/user prompt templates enforcing context-grounded, citation-ready answers.

The core design principle from docs/architecture.md lives here: the model
must answer ONLY from retrieved context and must explicitly say when the
corpus doesn't cover a question, rather than falling back on whatever it
memorized during training.
"""
from __future__ import annotations

from clinical_rag.schemas import Chunk

INSTRUCTIONS = """You are a clinical evidence assistant. Answer the question using ONLY the numbered context passages below, drawn from clinical guidelines and research documents. Do not use any outside knowledge.

Rules:
- Base every claim strictly on the provided context passages.
- If the context does not contain enough information to answer, say exactly: "The provided guidelines do not cover this." Do not guess or fill gaps from general knowledge.
- When you state a specific recommendation, threshold, or fact, cite which passage it came from using its number in brackets, e.g. [1] or [2].
- Be concise and clinically precise."""


def build_prompt(question: str, retrieved_chunks: list[Chunk]) -> str:
    if not retrieved_chunks:
        context_block = "(no relevant passages were retrieved)"
    else:
        context_block = "\n\n".join(
            f"[{i + 1}] (Source: {chunk.source_title}, {chunk.organization}"
            + (f", {chunk.year}" if chunk.year else "")
            + f")\n{chunk.text}"
            for i, chunk in enumerate(retrieved_chunks)
        )

    return (
        f"{INSTRUCTIONS}\n\n"
        f"CONTEXT PASSAGES:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )
