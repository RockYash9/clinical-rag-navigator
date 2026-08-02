# Clinical RAG Navigator

A retrieval-augmented generation (RAG) system for querying clinical guidelines,
research literature, and treatment protocols. Built for healthcare professionals
who need fast, evidence-based answers with traceable source citations and
confidence scoring — not black-box LLM guesses.

## Why this exists

Generic LLMs answer medical questions from memorized training data, which can be
outdated, unsourced, or subtly wrong. This system instead retrieves relevant
passages from a curated, versioned corpus of guidelines and papers, and forces
the generation step to ground every claim in that retrieved text — with
citations back to the exact source.

## Architecture

```
Query → Retrieval (vector search over guideline corpus) → Re-ranking
      → Generation (LLM answers ONLY from retrieved context)
      → Citation mapping + confidence scoring → Response
```

See [docs/architecture.md](docs/architecture.md) for the full pipeline breakdown.

## Project status

🚧 Early development — corpus sourcing and ingestion pipeline in progress.

## Tech stack (all free / open-source)

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` |
| Vector store | FAISS (local, no server) |
| LLM | Ollama (local open-weight models) |
| API | FastAPI |
| Data sources | PubMed/PMC, WHO, NICE, CDC guidelines |

## Getting started

```bash
# clone
git clone https://github.com/<your-username>/clinical-rag-navigator.git
cd clinical-rag-navigator

# environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for local dev/testing

# configure
cp .env.example .env
# edit .env with your local settings

# run tests
pytest
```

## Project structure

```
clinical-rag-navigator/
├── configs/            # YAML configs (pipeline, logging)
├── data/                # raw/processed corpus + vector index (gitignored)
├── docs/                 # architecture & data-source documentation
├── notebooks/            # exploratory analysis (not production code)
├── scripts/               # standalone entrypoints (ingest, build index)
├── src/clinical_rag/       # the actual package
│   ├── ingestion/           # document loading, cleaning, chunking
│   ├── embeddings/          # text -> vector
│   ├── retrieval/            # vector search + re-ranking
│   ├── generation/            # prompting + LLM calls
│   ├── citation/               # source attribution + confidence scoring
│   ├── api/                     # FastAPI app
│   └── utils/                    # shared helpers (logging, config loading)
└── tests/
    ├── unit/
    └── integration/
```

## Disclaimer

This is a research/engineering project, not a certified medical device. It is
not intended to replace clinical judgment and should not be used for direct
patient care decisions without professional oversight.

## License

MIT — see [LICENSE](LICENSE).
