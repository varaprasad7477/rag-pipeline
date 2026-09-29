# Atlas RAG Pipeline

A local-first retrieval-augmented generation workbench that turns private documents into grounded, cited answers. It works without an API key in extractive mode and can use any OpenAI-compatible chat-completions provider for fluent generation.

## What makes it more than a demo

- Hybrid retrieval combines BM25 lexical ranking with deterministic semantic feature hashing and reciprocal-rank fusion.
- Persistent SQLite storage uses WAL mode, content-hash deduplication, indexed chunks, and conversation memory.
- Grounded answers include numbered evidence, retrieval/generation timings, and an explicit no-context response.
- Prompt-injection resistance treats retrieved text as untrusted data and constrains the generator to source evidence.
- Complete product surface: responsive UI, typed FastAPI endpoints, automatic OpenAPI docs, evaluation harness, tests, Docker, and CI.
- No vector-database service is required. TXT, Markdown, CSV, and JSON work out of the box; PDF is optional.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,pdf]"
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. API documentation is at http://127.0.0.1:8000/docs.

## Optional LLM generation

Copy `.env.example` to `.env`, then configure:

```env
RAG_LLM_PROVIDER=openai-compatible
RAG_LLM_BASE_URL=https://api.openai.com/v1
RAG_LLM_MODEL=gpt-4.1-mini
RAG_LLM_API_KEY=your-key
```

The default `extractive` mode is fully local and returns the strongest source sentences with citations.

## API

```bash
curl -F "file=@handbook.md" http://localhost:8000/api/documents
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"question":"What is the leave policy?"}'
```

Key routes: `GET /api/health`, `GET/POST /api/documents`, `DELETE /api/documents/{id}`, and `POST /api/query`.

## Quality checks

```bash
ruff check .
pytest
python -m evals.run
```

## Architecture

`upload → validate/extract → boundary-aware chunks → BM25 + semantic hashing → rank fusion → grounded generator → cited response`

The lightweight built-in embedder makes the repository immediately runnable. For larger deployments, the retrieval interface can be replaced with a hosted embedding model and vector database without changing the API or UI.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

Uploaded knowledge is stored in the named `rag-data` volume. Never commit `.env` or database files.

## Origin

The supplied starter archive contained a small, unrelated customer-support planner patch rather than a RAG application. This repository therefore uses a clean RAG architecture while carrying forward its strongest idea: model output must be constrained, validated, and safely fall back when external generation is unavailable.

