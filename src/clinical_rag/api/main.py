"""FastAPI entrypoint.

TODO:
    - POST /query {question: str} -> {answer, citations, confidence}
    - GET /health
"""
from fastapi import FastAPI

app = FastAPI(title="Clinical RAG Navigator", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
