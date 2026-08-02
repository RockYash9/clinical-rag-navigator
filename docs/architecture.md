# Architecture

## Pipeline overview

```
┌──────────────┐   ┌────────────┐   ┌────────────┐   ┌─────────────┐
│  1. Sources  │ → │ 2. Ingest  │ → │ 3. Embed   │ → │ 4. Vector   │
│  (PubMed,    │   │  & chunk   │   │            │   │    store    │
│  WHO, NICE)  │   │            │   │            │   │  (FAISS)    │
└──────────────┘   └────────────┘   └────────────┘   └─────────────┘
                                                              │
┌──────────────┐   ┌────────────┐   ┌────────────┐          │
│ 8. Interface │ ← │ 7. Citation│ ← │ 6. Generate│ ← ┌───────┴──────┐
│  (API / UI)  │   │ & confidence│   │  (LLM)     │ ← │ 5. Retrieve  │
└──────────────┘   └────────────┘   └────────────┘   └──────────────┘
```

## Layer responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| 1. Sources | `docs/data_sources.md` | Curated list of free, citable clinical sources |
| 2. Ingestion | `src/clinical_rag/ingestion/` | Load raw docs, clean, chunk, tag metadata (source, date, tier) |
| 3. Embeddings | `src/clinical_rag/embeddings/` | Convert chunks/queries to vectors |
| 4. Vector store | `src/clinical_rag/retrieval/vector_store.py` | Persist and search embeddings (FAISS) |
| 5. Retrieval | `src/clinical_rag/retrieval/` | Similarity search + hybrid search + re-ranking |
| 6. Generation | `src/clinical_rag/generation/` | Prompt construction + LLM call, grounded strictly in retrieved context |
| 7. Citation | `src/clinical_rag/citation/` | Map generated claims back to source chunks; compute confidence score |
| 8. Interface | `src/clinical_rag/api/` | FastAPI endpoints serving the pipeline |

## Design principles

1. **Grounded generation only** — the LLM is instructed to answer exclusively
   from retrieved context, and to explicitly say when the corpus doesn't
   cover the question, rather than fall back on parametric knowledge.
2. **Traceable citations** — every chunk carries source metadata from
   ingestion through to the final response, so any claim can be traced back
   to a specific guideline/paper and section.
3. **Confidence scoring** — a composite score based on retrieval similarity,
   source agreement across chunks, and answer-to-source alignment.
4. **Free-tier only** — every component (embedding model, vector store, LLM,
   data sources) is free/open-source or free-tier accessible; no paid APIs
   required to run the system.
5. **Two-tier corpus** — `general_guidelines` and `disease_specific` tiers
   are tagged separately at ingestion so retrieval/citations can indicate
   which kind of source an answer draws from.
