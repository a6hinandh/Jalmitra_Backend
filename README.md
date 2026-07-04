# Jalmitra — Backend API

> FastAPI backend powering the Jalmitra groundwater intelligence platform. Combines **Neo4j graph queries**, **Pinecone vector search**, and **Google Gemini** into a GraphRAG pipeline that answers natural-language questions about Indian groundwater.

---

## What is Jalmitra?

**Jalmitra** (जलमित्र — "friend of water") is a groundwater intelligence platform for India, built on top of the government's own **CGWB (Central Ground Water Board)** dynamic groundwater resource assessments. Groundwater data in India is real, detailed, and public — but it is published as dense tabular PDFs and spreadsheets that are practically inaccessible to the people who need it most: farmers deciding when to sow, panchayat officials planning recharge structures, researchers looking for trends, and citizens who simply want to know if their district's water table is safe.

Jalmitra exists to close that gap. It turns a static annual assessment into a **living, queryable, multilingual system** that can be asked a question in plain language and answer it with a cited, data-grounded response — and increasingly, go beyond answering into **forecasting, recommending, and alerting**.

### The problem

- CGWB assessment data is authoritative but locked in a format only a specialist can parse.
- A farmer, policymaker, or citizen has no easy way to ask "is my area's groundwater in trouble?" and get a straight, trustworthy answer.
- District-level detail is sparse outside of pilot regions — most publicly consumable tools only go as deep as the state level.
- Existing dashboards show *what happened*, not *what's likely to happen next* or *what should be done about it*.

### The vision

Jalmitra is designed to move along a single axis with every feature it ships:

```
Chatbot (describe)  →  Dashboard (monitor)  →  Predictor (forecast)  →  Advisor (recommend)  →  Actor (intervene/alert)
```

A chatbot answers *"what is the situation?"*. A platform answers *"what should I do about it?"* and *"what happens if I do nothing?"*. Jalmitra started as the former and is deliberately built to grow into the latter — without ever losing the "ask a plain-language question, get a grounded answer" core that makes it usable by someone who has never opened a spreadsheet of groundwater metrics.

### Who it serves

Jalmitra is explicitly **role-aware**, not one-size-fits-all:

- **Farmers** — plain-language answers, and a structured sowing/irrigation advisory that cross-references groundwater stage-of-extraction with crop water requirements.
- **Policymakers** — cross-district comparison, forecasting, and a what-if simulator for testing intervention scenarios (e.g. "what if we cut agricultural draft by 15%?"), plus exportable PDF reports.
- **Researchers** — cited answers (the Cypher query and retrieved passages behind every response), raw CSV/JSON export, and a transparent (not black-box) forecasting methodology.
- **General public** — a friend to ask "is my water safe?" in six Indian languages, with no login and no jargon required.

### How it's grounded (not a generic LLM wrapper)

The single most important design decision in Jalmitra is that answers are never generated from the LLM's own memory. Every response is built from two retrieval sources — a Neo4j **knowledge graph** encoding structured relationships (state → district → year → metric) and a Pinecone **vector index** of semantically searchable assessment text — merged into a context that Gemini is asked to *summarize*, not *invent*. This is what "GraphRAG" means in this codebase: Graph + Retrieval-Augmented Generation, dual-pipeline, not single-pipeline.

### Current data coverage

- CGWB / Ministry of Jal Shakti assessment cycles: **2023 and 2024**
- **28+ states and union territories** at the state level
- **District-level detail for Kerala** (14 districts) — the deepest granularity currently available
- Widening coverage is an active priority, primarily through the crowdsourced **Field Observations** feature (see below), which lets anyone submit a real, timestamped well-depth reading for their district — closing the data gap without waiting on a new government data partnership.

### What Jalmitra can do today

| Capability | What it does |
|---|---|
| **Conversational Q&A** | Role-aware, multilingual, streaming chat answers grounded in the knowledge graph + vector search |
| **Interactive map** | State-level choropleth across 5 metrics and 2 years |
| **Comparative visualization** | State/district/year/metric comparison charts with CSV/PNG export |
| **Forecasting** | Transparent linear-trend projection of stage-of-extraction 1–3 years out, with risk-band threshold-crossing alerts |
| **What-if simulator** | Adjustable draft-change scenario testing built on the forecasting model |
| **Farmer advisory** | Rule-based sowing/irrigation recommendations combining live groundwater data with ICAR/CWC crop water-requirement tables |
| **Crowdsourced field observations** | Community-submitted well-depth readings, clearly separated from official CGWB data |
| **PDF report generation** | Narrative + chart + table reports for a state/district/year range |

### Design principles

- **Grounded over generative** — the LLM synthesizes retrieved facts; it does not invent numbers.
- **Transparent over impressive** — the forecasting model is a deliberately simple linear extrapolation over 2 years of data, explicitly honest about its own confidence, rather than a heavyweight model overfitting on too little history.
- **Public and frictionless** — no authentication anywhere in the product; the barrier to "ask a question" or "submit a reading" is as close to zero as possible.
- **Degrade gracefully** — if the Neo4j graph is temporarily unavailable (e.g. AuraDB free-tier auto-pause), the pipeline falls back to semantic-only results rather than failing outright.

---

## Architecture

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  FastAPI  (server.py)                                  │
│  POST /chat  ·  POST /chat/stream  ·  GET /api/v1/...  │
└────────────────────────────────────────────────────────┘
    │
    ▼
graphrag_chatbot()  (graphrag.py)
┌─────────────────────────────────────────────────────────────────────┐
│  1. detect language                                                 │
│  2. Cypher query  -> Neo4j graph            (always on)             │
│  3. embed query   -> Pinecone   (only if PINECONE_ACTIVATION=true)  │
│  4. merge contexts                                                  │
│  5. Gemini 3.1 Flash Lite -> answer                                 │
└─────────────────────────────────────────────────────────────────────┘
```

Step 3 is gated behind `PINECONE_ACTIVATION` (default `false`) — see
[Deployment modes](#deployment-modes-why-pinecone-is-off-in-production) below.
Production runs steps 1, 2, 4, 5 only (Neo4j + Gemini); the full pipeline
including step 3 is available locally / on hosts with ≥1GB RAM.

**Data sources**
- Neo4j AuraDB — 28-state groundwater knowledge graph (always used)
- Pinecone — sentence embeddings (optional layer, disabled in production — see below)
- IITH INGRES API — district-level Kerala data
- CGWB / Ministry of Jal Shakti datasets (2023–2024)

### Deployment modes: why Pinecone is off in production

The Pinecone semantic-search layer (step 3 above) is **fully implemented but gated
behind the `PINECONE_ACTIVATION` environment flag**, and it is **disabled in
production**. Here's why:

Semantic retrieval needs `torch` + `sentence-transformers` to embed the query.
Importing `torch` alone costs ~250–350MB of RAM, and the embedding model adds
another ~90–420MB on top. Together with FastAPI, pandas, numpy and scikit-learn,
that comfortably exceeds the **512MB memory limit of the free-tier hosting
instance** — the process gets OOM-killed on the first `/chat` request.

So the two supported modes are:

| Mode | `PINECONE_ACTIVATION` | Pipeline | Where |
|---|---|---|---|
| **Production** | `false` (default) | Neo4j graph + Gemini | 512MB free-tier deploy |
| **Full / local** | `true` | Neo4j graph + **Pinecone semantic** + Gemini | local / ≥1GB RAM |

When `PINECONE_ACTIVATION` is `false`, `torch` and `sentence-transformers` are
**never imported** (the imports are deferred into `get_embed_model()`), so the
process stays well within budget. The chat pipeline degrades gracefully to
graph-only retrieval — [`query_pinecone`](core/graphrag.py) simply returns no
semantic hits and Gemini answers from the Neo4j graph results.

To run the full pipeline locally, set `PINECONE_ACTIVATION=true` (and a valid
`PINECONE_API_KEY`) in your `.env` on a machine with ≥1GB RAM.

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A Neo4j instance (local or [AuraDB](https://neo4j.com/cloud/aura/) free tier)
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini 3.1 Flash Lite)
- Optional: a [Pinecone](https://www.pinecone.io/) account with one index, only if you plan to run with `PINECONE_ACTIVATION=true` (see [Deployment modes](#deployment-modes-why-pinecone-is-off-in-production))

### 2. Clone and install

```bash
git clone https://github.com/a6hinandh/Jalmitra_Backend.git
cd Jalmitra_Backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in the four required keys (NEO4J_URI, NEO4J_USER, NEO4J_PASS, GENAI_API_KEY).
# Pinecone vars are only needed if PINECONE_ACTIVATION=true.
```

### 4. Run

```bash
uvicorn server:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `NEO4J_URI` | ✅ | Bolt URI, e.g. `bolt://localhost:7687` or AuraDB URI |
| `NEO4J_USER` | ✅ | Username (default: `neo4j`) |
| `NEO4J_PASS` | ✅ | Password |
| `GENAI_API_KEY` | ✅ | Google Gemini API key |
| `PINECONE_ACTIVATION` | ➖ | `true` to enable the semantic layer (default `false`). See [Deployment modes](#deployment-modes-why-pinecone-is-off-in-production) |
| `PINECONE_API_KEY` | ⚠️ | Required **only** when `PINECONE_ACTIVATION=true`. From [pinecone.io](https://www.pinecone.io/) |
| `PINECONE_INDEX` | ➖ | Index name (must match `EMBED_MODEL` dimension). Used only when activated |
| `EMBED_MODEL` | ➖ | Sentence-transformer model, used only when activated (default: `all-MiniLM-L6-v2`) |
| `ALLOWED_ORIGINS` | ➖ | Comma-separated CORS origins (default: `http://localhost:3001,http://localhost:5173`) |

---

## API Reference

All endpoints return JSON. Error responses use `{"detail": "..."}`.

### Chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Single-turn question → answer |
| `POST` | `/chat/stream` | Same, but Server-Sent Events streaming |

**Request body** (both endpoints):
```json
{
  "query": "What is the groundwater availability in Kerala?",
  "role": "farmer",
  "language": "en"
}
```
`role` ∈ `farmer | policymaker | researcher | general`

**Response** (`/chat`):
```json
{
  "final_answer": "Kerala has ...",
  "sources": ["cypher", "pinecone"],
  "role_used": "farmer",
  "language_detected": "en",
  "processing_time": 1.23
}
```

**SSE stream** (`/chat/stream`): emits `data: {...}` lines:
- `{"token": "partial text"}` — streaming token
- `{"done": true, "final_answer": "...", "sources": [...]}` — completion
- `{"error": "message"}` — error

---

### Data & Visualization

| Method | Path | Description |
|---|---|---|
| `POST` | `/visualize` | Generate chart data for a given comparison |
| `GET` | `/visualization/options` | All available states, districts, metrics, years |
| `GET` | `/api/v1/states` | List all states |
| `GET` | `/api/v1/states/{state}/districts` | Districts for a state |
| `GET` | `/api/v1/metrics` | All metric definitions |
| `GET` | `/api/v1/map/states` | Per-state values for the India map choropleth |

**`GET /api/v1/map/states`** query params:
- `metric` — one of `rainfall | recharge | draft | availability | groundwater`
- `year` — `2023` or `2024`

---

### Utility

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + dependency status |
| `GET` | `/api/v1/suggestions?q=` | Query autocomplete |
| `POST` | `/api/v1/feedback` | Submit a rating/comment |
| `POST` | `/api/v1/data/export` | Download filtered data as CSV |

---

## Docker

```bash
docker build -t jalmitra-backend .

docker run -p 8000:8000 \
  -e NEO4J_URI=... \
  -e NEO4J_USER=... \
  -e NEO4J_PASS=... \
  -e PINECONE_API_KEY=... \
  -e PINECONE_INDEX=... \
  -e GENAI_API_KEY=... \
  jalmitra-backend
```

---

## Rate Limiting

In-memory, per-IP: **30 requests / minute**. Exceeding returns `HTTP 429`.

---

## Project Structure

```
Jalmitra_Backend/
├── server.py               # FastAPI app, all endpoints
├── core/
│   ├── graphrag.py         # GraphRAG pipeline (Neo4j + Gemini; Pinecone optional, see Deployment modes)
│   └── embeddings.py       # Sentence-transformer embedding helpers
├── services/
│   ├── advisory_service.py           # Farmer advisory recommendations
│   ├── field_observations_service.py # Field data submission/retrieval
│   ├── forecast_service.py           # Linear-trend groundwater forecasting
│   └── reports_service.py            # PDF/CSV report generation
├── scripts/                # Dev-only seeding, debug, and indexing scripts
│   ├── fetch_states.py     # Pull CGWB data → data/states, data/output
│   ├── insert_data.py      # Embed + upsert into Pinecone
│   ├── insert_graph.py     # Load india.json into Neo4j
│   ├── insert_graph_district.py
│   ├── pinecone_setup.py
│   ├── query_index.py
│   ├── check_uuids.py / debug_data.py / delete_index.py
│   └── generate_response.py / generate_graph_response.py
├── data/
│   ├── states/             # Per-state CGWB JSON/CSV
│   ├── KERALA/              # Kerala district-level JSON
│   └── output/              # Aggregated india.json / india.csv
├── alerts.db                # SQLite database — created at runtime, gitignored, never committed
├── requirements.txt
├── .env.example
├── Dockerfile
└── .github/workflows/ci.yml
```

> Scripts in `scripts/` are meant to be run from the repository root (e.g. `python scripts/fetch_states.py`), since they resolve data paths relative to `data/`.

> `alerts.db` is regenerated automatically by `alerts_service.py` on first run — it is runtime application state, not source, and is excluded via `.gitignore`. Never commit it; each environment (local/dev/prod) should have its own.
