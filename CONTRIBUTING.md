# Contributing to Jalmitra Backend

Thanks for your interest in improving the Jalmitra GraphRAG backend. This guide covers environment setup, external service configuration, and code style.

## Local Setup

1. Fork and clone the repository.
2. Create and activate a virtual environment:
   ```bash
   cd Jalmitra_Backend
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template and fill in the required keys:
   ```bash
   cp .env.example .env
   ```
5. Run the dev server:
   ```bash
   uvicorn server:app --reload --port 8000
   ```

## External Service Configuration

### Neo4j (AuraDB)

1. Create a free instance at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/) (or run Neo4j locally via Docker).
2. Copy the Bolt URI, username, and password into `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`.
3. Seed the graph using `python scripts/insert_graph.py` (loads `data/output/india.json`) and, for Kerala district detail, `python scripts/insert_graph_district.py`. Run these from the repository root.

### Pinecone (optional — semantic search layer)

Pinecone powers an optional semantic-search layer on top of the Neo4j graph. It's
**disabled by default** (`PINECONE_ACTIVATION=false`) because `torch` +
`sentence-transformers` don't fit in the 512MB production instance — see
[README → Deployment modes](README.md#deployment-modes-why-pinecone-is-off-in-production)
for the full explanation. You only need this section if you're running the full
pipeline locally (≥1GB RAM).

1. Create an account and API key at [pinecone.io](https://www.pinecone.io/).
2. Create an index matching your `EMBED_MODEL`'s dimension and **cosine** metric — 384 dimensions for the default `all-MiniLM-L6-v2`, or 768 for `all-mpnet-base-v2` — or run `python scripts/pinecone_setup.py`.
3. Set `PINECONE_ACTIVATION=true`, `PINECONE_API_KEY`, and `PINECONE_INDEX` in `.env`.
4. Populate the index with `python scripts/insert_data.py`.

### Google Gemini

1. Generate an API key at [aistudio.google.com](https://aistudio.google.com/).
2. Set `GENAI_API_KEY` in `.env`.

## Code Style

- Business logic belongs in `services/`; shared graph/embedding utilities belong in `core/`.
- One-off/dev scripts (seeding, debugging, index management) belong in `scripts/` and should be runnable from the repository root.
- Prefer explicit imports (`from core.graphrag import run_cypher`) over wildcard imports.
- Keep endpoint handlers in `server.py` thin — delegate to the relevant `services/*.py` module.
- Use type hints on new functions and Pydantic models for request/response schemas.

## Pull Request Workflow

1. Branch from `main` as `feature/<short-description>` or `fix/<short-description>`.
2. Verify the app still imports and starts cleanly: `python -c "import server"` and `uvicorn server:app --port 8000`.
3. Open a PR describing the change, including any new environment variables or migrations.

## Reporting Bugs

Open a GitHub issue with reproduction steps, the failing endpoint/request, and relevant logs (with secrets redacted). For security issues, see [SECURITY.md](SECURITY.md) instead.
