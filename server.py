"""
Jalmitra Backend — FastAPI server v4.0 (Production)
GraphRAG-powered groundwater intelligence API
"""

import os
import json
import time
import logging
import asyncio
from collections import defaultdict
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

from core.graphrag import graphrag_chatbot, run_cypher
from services import (
    forecast_service,
    advisory_service,
    field_observations_service,
    reports_service,
)

load_dotenv()

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("jalmitra")


# ---------- Rate Limiting (simple in-memory) ----------
_rate_store: Dict[str, List[float]] = defaultdict(list)

def check_rate_limit(ip: str, max_requests: int = 30, window: int = 60) -> bool:
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < window]
    if len(_rate_store[ip]) >= max_requests:
        return False
    _rate_store[ip].append(now)
    return True

async def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 30 requests/minute.")


# ---------- TTL cache (simple in-memory, for map/compare queries) ----------
_ttl_cache: Dict[str, tuple] = {}

def cache_get(key: str):
    entry = _ttl_cache.get(key)
    if not entry:
        return None
    value, expires_at = entry
    if time.time() > expires_at:
        _ttl_cache.pop(key, None)
        return None
    return value

def cache_set(key: str, value, ttl: int = 300):
    _ttl_cache[key] = (value, time.time() + ttl)


# ---------- Scheduled jobs (data-freshness) ----------
scheduler = BackgroundScheduler()

def _scheduled_ingestion_reminder():
    # Placeholder for automated ingestion (roadmap 3.4): CGWB/Jal Shakti datasets are published
    # periodically outside our control, so this job just logs a reminder rather than fabricating
    # a live feed. Once a real upstream feed/URL is available, replace this with an actual fetch
    # + insert_data.py / insert_graph.py invocation.
    logger.info("scheduled ingestion check: no new upstream dataset source configured")


# ---------- App ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Jalmitra API starting up")
    scheduler.add_job(_scheduled_ingestion_reminder, "interval", hours=24, id="ingestion_check", replace_existing=True)
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)
    logger.info("Jalmitra API shutting down")

app = FastAPI(
    title="Jalmitra — Groundwater Intelligence API",
    description=(
        "AI-powered groundwater data access for India. "
        "Dual-pipeline RAG: Pinecone semantic search + Neo4j knowledge graph + Gemini LLM."
    ),
    version="4.0.0",
    contact={"name": "Jalmitra Team"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------- CORS ----------
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3001,http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request logging middleware ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 3)
    logger.info(f"{request.method} {request.url.path} {response.status_code} {duration}s")
    return response


# ---------- Pydantic Models ----------
class ChatHistoryTurn(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural language query")
    role: str = Field(default="general", description="farmer | policymaker | researcher | general")
    debug: bool = Field(default=False)
    history: Optional[List[ChatHistoryTurn]] = Field(default=None, description="Last 5 conversation turns for context")

class ChatResponse(BaseModel):
    query: str
    role: str
    final_answer: str
    processing_time: float
    cypher_used: Optional[str] = None
    semantic_results_count: int
    graph_results_count: int
    interpretation_applied: bool
    error: Optional[str] = None
    debug_info: Optional[dict] = None
    sources: Optional[List[dict]] = None
    chart: Optional[Dict[str, Any]] = None

class DataVisualizationRequest(BaseModel):
    chart_type: str = Field(..., description="bar | line | pie | doughnut | radar")
    comparison_type: str = Field(..., description="state | district | yearly | metric")
    states: Optional[List[str]] = None
    districts: Optional[List[str]] = None
    years: Optional[List[int]] = None
    metrics: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None

class DataVisualizationResponse(BaseModel):
    chart_type: str
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class ExportRequest(BaseModel):
    comparison_type: str
    metrics: List[str]
    states: Optional[List[str]] = None
    districts: Optional[List[str]] = None
    years: Optional[List[int]] = None
    filters: Optional[Dict[str, Any]] = None


# ---------- Constants ----------
VALID_ROLES = ["farmer", "policymaker", "researcher", "general"]
VALID_METRICS = ["rainfall", "recharge", "draft", "availability", "groundwater"]
VALID_YEARS = [2023, 2024]

CHART_TYPE_RESTRICTIONS = {
    "radar": ["metric"],
    "pie": ["state", "district"],
    "doughnut": ["state", "district"],
}

METRIC_META = {
    "rainfall":    {"label": "Rainfall",              "unit": "mm",  "rel": "RAINFALL",    "node": "Rainfall",              "prop": "total"},
    "recharge":    {"label": "Groundwater Recharge",  "unit": "ham", "rel": "RECHARGE",    "node": "Recharge",              "prop": "total"},
    "draft":       {"label": "Groundwater Draft",     "unit": "ham", "rel": "DRAFT",       "node": "Draft",                 "prop": "total"},
    "availability":{"label": "Water Availability",    "unit": "ham", "rel": "AVAILABILITY","node": "Availability",          "prop": "total"},
    "groundwater": {"label": "Groundwater Resources", "unit": "ham", "rel": "GROUND_WATER","node": "GroundWaterAvailability","prop": "total"},
}

STATE_CENTROIDS = {
    "ANDHRA PRADESH":         {"lat": 15.91, "lng": 79.74},
    "ARUNACHAL PRADESH":      {"lat": 28.22, "lng": 94.73},
    "ASSAM":                  {"lat": 26.20, "lng": 92.94},
    "BIHAR":                  {"lat": 25.10, "lng": 85.31},
    "CHHATTISGARH":           {"lat": 21.28, "lng": 81.87},
    "GOA":                    {"lat": 15.30, "lng": 74.12},
    "GUJARAT":                {"lat": 22.26, "lng": 71.19},
    "HARYANA":                {"lat": 29.06, "lng": 76.09},
    "HIMACHAL PRADESH":       {"lat": 31.10, "lng": 77.17},
    "JHARKHAND":              {"lat": 23.61, "lng": 85.28},
    "KARNATAKA":              {"lat": 15.32, "lng": 75.71},
    "KERALA":                 {"lat": 10.85, "lng": 76.27},
    "MADHYA PRADESH":         {"lat": 22.97, "lng": 78.66},
    "MAHARASHTRA":            {"lat": 19.75, "lng": 75.71},
    "MANIPUR":                {"lat": 24.66, "lng": 93.91},
    "MEGHALAYA":              {"lat": 25.47, "lng": 91.37},
    "MIZORAM":                {"lat": 23.16, "lng": 92.94},
    "NAGALAND":               {"lat": 26.16, "lng": 94.56},
    "ODISHA":                 {"lat": 20.95, "lng": 85.10},
    "PUNJAB":                 {"lat": 31.15, "lng": 75.34},
    "RAJASTHAN":              {"lat": 27.02, "lng": 74.22},
    "SIKKIM":                 {"lat": 27.53, "lng": 88.51},
    "TAMILNADU":              {"lat": 11.13, "lng": 78.66},
    "TELANGANA":              {"lat": 18.11, "lng": 79.02},
    "TRIPURA":                {"lat": 23.94, "lng": 91.99},
    "UTTAR PRADESH":          {"lat": 26.85, "lng": 80.95},
    "UTTARAKHAND":            {"lat": 30.07, "lng": 79.02},
    "WEST BENGAL":            {"lat": 22.99, "lng": 87.85},
    "DELHI":                  {"lat": 28.70, "lng": 77.10},
    "JAMMU AND KASHMIR":      {"lat": 33.73, "lng": 76.92},
    "LADAKH":                 {"lat": 34.17, "lng": 77.58},
}


# ---------- Cypher Query Generators ----------
def _metric_parts(metric: str):
    m = METRIC_META.get(metric)
    if not m:
        return None
    return m["rel"], m["node"], m["prop"]

def generate_state_comparison_query(states: List[str], metric: str, year: int = 2024) -> Optional[str]:
    parts = _metric_parts(metric)
    if not parts:
        return None
    rel_type, node_type, prop = parts
    states_str = '", "'.join(s.upper() for s in states)
    return (
        f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State)-[:HAS_YEAR]->(y:Year {{year:{year}}})'
        f'-[:HAS_{rel_type}]->(n:{node_type}) '
        f'WHERE s.name IN ["{states_str}"] AND n.{prop} IS NOT NULL '
        f'RETURN s.name AS entity, n.{prop} AS value ORDER BY s.name'
    )

def generate_district_comparison_query(state: str, districts: List[str], metric: str, year: int = 2024) -> Optional[str]:
    parts = _metric_parts(metric)
    if not parts:
        return None
    rel_type, node_type, prop = parts
    districts_str = '", "'.join(d.upper() for d in districts)
    return (
        f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
        f'-[:HAS_DISTRICT]->(d:District)-[:HAS_YEAR]->(y:Year {{year:{year}}})'
        f'-[:HAS_{rel_type}]->(n:{node_type}) '
        f'WHERE d.name IN ["{districts_str}"] AND n.{prop} IS NOT NULL '
        f'RETURN d.name AS entity, n.{prop} AS value ORDER BY d.name'
    )

def generate_yearly_trend_query(entity: str, entity_type: str, metric: str, years: List[int]) -> Optional[str]:
    parts = _metric_parts(metric)
    if not parts:
        return None
    rel_type, node_type, prop = parts
    years_str = ", ".join(map(str, years))
    if entity_type == "state":
        return (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{entity.upper()}"}})'
            f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel_type}]->(n:{node_type}) '
            f'WHERE y.year IN [{years_str}] AND n.{prop} IS NOT NULL '
            f'RETURN y.year AS entity, n.{prop} AS value ORDER BY y.year'
        )
    return (
        f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"KERALA"}})'
        f'-[:HAS_DISTRICT]->(d:District {{name:"{entity.upper()}"}})'
        f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel_type}]->(n:{node_type}) '
        f'WHERE y.year IN [{years_str}] AND n.{prop} IS NOT NULL '
        f'RETURN y.year AS entity, n.{prop} AS value ORDER BY y.year'
    )

def generate_multi_metric_query(entity: str, entity_type: str, metrics: List[str], year: int = 2024) -> Optional[str]:
    if entity_type == "state":
        base = f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{entity.upper()}"}})-[:HAS_YEAR]->(y:Year {{year:{year}}})'
    else:
        base = (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"KERALA"}})'
            f'-[:HAS_DISTRICT]->(d:District {{name:"{entity.upper()}"}})-[:HAS_YEAR]->(y:Year {{year:{year}}})'
        )
    clauses, returns = [], []
    aliases = {"rainfall": "r", "recharge": "rec", "draft": "dr", "availability": "av", "groundwater": "gw"}
    for m in metrics:
        parts = _metric_parts(m)
        if parts:
            rel_type, node_type, _ = parts
            alias = aliases.get(m, m[:3])
            clauses.append(f'OPTIONAL MATCH (y)-[:HAS_{rel_type}]->({alias}:{node_type})')
            returns.append(f'{alias}.total AS {m}')
    if not returns:
        return None
    return f'{base} {" ".join(clauses)} RETURN {", ".join(returns)}'


# ---------- Chart Formatters ----------
_COLORS_BG = [
    "rgba(59,130,246,0.8)", "rgba(16,185,129,0.8)", "rgba(245,158,11,0.8)",
    "rgba(239,68,68,0.8)",  "rgba(139,92,246,0.8)", "rgba(236,72,153,0.8)",
    "rgba(6,182,212,0.8)",  "rgba(251,113,133,0.8)",
]
_COLORS_BD = [c.replace("0.8", "1") for c in _COLORS_BG]

def _labels(data): return [str(r.get("entity", "?")) for r in data]
def _values(data): return [float(r.get("value", 0) or 0) for r in data]

def fmt_bar(data, title):
    return {"type": "bar", "data": {"labels": _labels(data), "datasets": [{"label": title, "data": _values(data), "backgroundColor": _COLORS_BG, "borderColor": _COLORS_BD, "borderWidth": 2, "borderRadius": 8, "borderSkipped": False}]}, "options": {"responsive": True, "maintainAspectRatio": False, "plugins": {"title": {"display": True, "text": title, "font": {"size": 18, "weight": "bold"}}, "legend": {"display": False}}, "scales": {"y": {"beginAtZero": True}, "x": {"grid": {"display": False}}}}}

def fmt_line(data, title):
    return {"type": "line", "data": {"labels": _labels(data), "datasets": [{"label": title, "data": _values(data), "borderColor": "rgba(59,130,246,1)", "backgroundColor": "rgba(59,130,246,0.1)", "borderWidth": 3, "fill": True, "tension": 0.4, "pointRadius": 6, "pointHoverRadius": 8}]}, "options": {"responsive": True, "maintainAspectRatio": False, "plugins": {"title": {"display": True, "text": title, "font": {"size": 18, "weight": "bold"}}}, "scales": {"y": {"beginAtZero": True}}}}

def fmt_pie(data, title):
    return {"type": "pie", "data": {"labels": _labels(data), "datasets": [{"data": _values(data), "backgroundColor": _COLORS_BG, "borderColor": "#fff", "borderWidth": 3, "hoverBorderWidth": 4}]}, "options": {"responsive": True, "maintainAspectRatio": False, "plugins": {"title": {"display": True, "text": title, "font": {"size": 18, "weight": "bold"}}, "legend": {"position": "right"}}}}

def fmt_radar(data, metrics, title):
    if not data:
        return None
    rec = data[0]
    vals = [float(rec.get(m, 0) or 0) for m in metrics]
    return {"type": "radar", "data": {"labels": [m.title() for m in metrics], "datasets": [{"label": title, "data": vals, "backgroundColor": "rgba(59,130,246,0.2)", "borderColor": "rgba(59,130,246,1)", "borderWidth": 3, "pointRadius": 6}]}, "options": {"responsive": True, "maintainAspectRatio": False, "plugins": {"title": {"display": True, "text": title, "font": {"size": 18, "weight": "bold"}}}, "scales": {"r": {"beginAtZero": True}}}}


# ---------- Helpers ----------
def _valid_year(y): return y if y in VALID_YEARS else VALID_YEARS[-1]


def maybe_build_chat_chart(graph_results: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    """Chart-in-chat (roadmap 3.1): when the Cypher result looks like a trend
    (a 'year'-keyed series with 2+ rows), render it as a small inline line chart
    instead of asking the user to go to Compare for the same numbers."""
    if not graph_results or len(graph_results) < 2:
        return None
    sample = graph_results[0]
    year_key = next((k for k in sample.keys() if k.lower() in ("year", "entity")), None)
    value_key = next((k for k in sample.keys() if k != year_key and isinstance(sample.get(k), (int, float))), None)
    if not year_key or not value_key:
        return None
    rows = [{"entity": r.get(year_key), "value": r.get(value_key)} for r in graph_results if r.get(value_key) is not None]
    if len(rows) < 2:
        return None
    return fmt_line(rows, value_key.replace("_", " ").title())


# ---------- Routes ----------

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Jalmitra Groundwater Intelligence API",
        "version": "4.0.0",
        "status": "healthy",
        "docs": "/docs",
    }

@app.get("/health", tags=["Info"])
async def health():
    from core.graphrag import driver, pine_index
    neo4j_ok, pinecone_ok = False, False
    try:
        with driver.session() as s:
            s.run("RETURN 1")
        neo4j_ok = True
    except Exception:
        pass
    try:
        pine_index.describe_index_stats()
        pinecone_ok = True
    except Exception:
        pass
    return {
        "status": "healthy" if (neo4j_ok and pinecone_ok) else "degraded",
        "services": {"neo4j": "up" if neo4j_ok else "down", "pinecone": "up" if pinecone_ok else "down"},
        "data_availability": {"years": VALID_YEARS, "districts": "Kerala only"},
        "timestamp": time.time(),
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"],
          dependencies=[Depends(rate_limit)])
async def chat_endpoint(request: ChatRequest):
    if request.role.lower() not in VALID_ROLES:
        raise HTTPException(400, f"Invalid role. Use: {', '.join(VALID_ROLES)}")
    logger.info(f"chat query role={request.role} query_len={len(request.query)}")
    try:
        history_dicts = None
        if request.history:
            history_dicts = [{"role": h.role, "content": h.content} for h in request.history[-5:]]
        result = graphrag_chatbot(request.query, role=request.role.lower(), debug_mode=request.debug, history=history_dicts)
        debug_info = None
        if request.debug:
            debug_info = {
                "cypher_query": result.get("cypher_used"),
                "semantic_count": len(result.get("semantic_results", [])),
                "graph_count": len(result.get("graph_results", [])),
            }
        return ChatResponse(
            query=result["query"],
            role=result["role"],
            final_answer=result["final_answer"],
            processing_time=result["processing_time"],
            cypher_used=result.get("cypher_used"),
            semantic_results_count=len(result.get("semantic_results", [])),
            graph_results_count=len(result.get("graph_results", [])),
            interpretation_applied=result.get("interpretation_applied", False),
            error=result.get("error"),
            debug_info=debug_info,
            sources=result.get("sources"),
            chart=maybe_build_chat_chart(result.get("graph_results", []), request.query),
        )
    except Exception as e:
        logger.error(f"chat error: {e}")
        raise HTTPException(500, f"Internal error: {str(e)}")


@app.post("/chat/stream", tags=["Chat"], dependencies=[Depends(rate_limit)])
async def chat_stream(request: ChatRequest):
    """SSE endpoint — streams the answer token by token via Server-Sent Events."""
    if request.role.lower() not in VALID_ROLES:
        raise HTTPException(400, f"Invalid role. Use: {', '.join(VALID_ROLES)}")

    async def generate():
        try:
            history_dicts = None
            if request.history:
                history_dicts = [{"role": h.role, "content": h.content} for h in request.history[-5:]]
            result = await asyncio.to_thread(
                graphrag_chatbot, request.query, request.role.lower(), request.debug, history_dicts
            )
            answer = result.get("final_answer", "")
            words = answer.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'token': chunk})}\n\n"
                await asyncio.sleep(0.02)
            meta = {
                "done": True,
                "processing_time": result.get("processing_time", 0),
                "cypher_used": result.get("cypher_used"),
                "sources": result.get("sources"),
                "chart": maybe_build_chat_chart(result.get("graph_results", []), request.query),
            }
            yield f"data: {json.dumps(meta)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/visualize", response_model=DataVisualizationResponse, tags=["Visualization"])
async def visualize(request: DataVisualizationRequest):
    start = time.time()
    cache_key = f"visualize:{request.model_dump_json()}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    if request.chart_type in CHART_TYPE_RESTRICTIONS:
        allowed = CHART_TYPE_RESTRICTIONS[request.chart_type]
        if request.comparison_type not in allowed:
            raise HTTPException(400, f"'{request.chart_type}' only works with: {allowed}")
    if not request.metrics:
        raise HTTPException(400, "At least one metric required")

    cypher_query = None
    year = _valid_year(request.filters.get("year", 2024) if request.filters else 2024)

    if request.comparison_type == "state" and request.states:
        cypher_query = generate_state_comparison_query(request.states, request.metrics[0], year)
    elif request.comparison_type == "district" and request.districts:
        cypher_query = generate_district_comparison_query("Kerala", request.districts, request.metrics[0], year)
    elif request.comparison_type == "yearly" and request.years:
        entity = (request.filters or {}).get("entity", "Kerala")
        etype  = (request.filters or {}).get("entity_type", "state")
        valid_years = [y for y in request.years if y in VALID_YEARS] or VALID_YEARS
        cypher_query = generate_yearly_trend_query(entity, etype, request.metrics[0], valid_years)
    elif request.comparison_type == "metric" and request.metrics:
        entity = (request.filters or {}).get("entity", "Kerala")
        etype  = (request.filters or {}).get("entity_type", "state")
        cypher_query = generate_multi_metric_query(entity, etype, request.metrics, year)

    if not cypher_query:
        raise HTTPException(400, "Could not build query from given parameters")

    try:
        raw = run_cypher(cypher_query)
    except Exception as e:
        return DataVisualizationResponse(
            chart_type=request.chart_type, data=None,
            metadata={"query_used": cypher_query, "data_points": 0},
            processing_time=round(time.time()-start,2), error=str(e))

    if not raw:
        return DataVisualizationResponse(
            chart_type=request.chart_type, data=None,
            metadata={"query_used": cypher_query, "data_points": 0},
            processing_time=round(time.time()-start,2),
            error="No data found for the given parameters")

    titles = {
        "state":   f"State-wise {request.metrics[0].title()}",
        "district":f"District-wise {request.metrics[0].title()}",
        "yearly":  f"Yearly {request.metrics[0].title()} Trend",
        "metric":  "Multi-metric Analysis",
    }
    title = titles.get(request.comparison_type, "Analysis")

    chart_data = None
    if request.chart_type == "bar":
        chart_data = fmt_bar(raw, title)
    elif request.chart_type == "line":
        chart_data = fmt_line(raw, title)
    elif request.chart_type in ("pie", "doughnut"):
        chart_data = fmt_pie(raw, title)
        if chart_data and request.chart_type == "doughnut":
            chart_data["type"] = "doughnut"
    elif request.chart_type == "radar":
        chart_data = fmt_radar(raw, request.metrics, title)

    response = DataVisualizationResponse(
        chart_type=request.chart_type,
        data=chart_data,
        metadata={"query_used": cypher_query, "data_points": len(raw), "comparison_type": request.comparison_type},
        processing_time=round(time.time()-start, 2),
    )
    cache_set(cache_key, response, ttl=300)
    return response


@app.get("/visualization/options", tags=["Visualization"])
async def visualization_options():
    states_q = 'MATCH (c:Country {name:"India"})-[:HAS_STATE]->(s:State) WHERE toLower(s.name) <> "total" RETURN DISTINCT s.name AS name ORDER BY s.name'
    districts_q = 'MATCH (s:State {name:"KERALA"})-[:HAS_DISTRICT]->(d:District) RETURN DISTINCT d.name AS name ORDER BY d.name'
    try:
        states_data = run_cypher(states_q)
        dist_data   = run_cypher(districts_q)
        states = [r["name"] for r in states_data]
        districts = [r["name"] for r in dist_data]
    except Exception:
        states    = list(STATE_CENTROIDS.keys())[:10]
        districts = ["KOTTAYAM","ERNAKULAM","THRISSUR","PALAKKAD","KOZHIKODE"]
    return {
        "chart_types": [
            {"id": "bar",      "label": "Bar Chart",      "compatible": ["state","district","yearly"]},
            {"id": "line",     "label": "Line Chart",     "compatible": ["yearly"]},
            {"id": "pie",      "label": "Pie Chart",      "compatible": ["state","district"]},
            {"id": "doughnut", "label": "Doughnut Chart", "compatible": ["state","district"]},
            {"id": "radar",    "label": "Radar Chart",    "compatible": ["metric"]},
        ],
        "comparison_types": ["state","district","yearly","metric"],
        "metrics": [{"id": k, **{k2:v for k2,v in v.items() if k2 != "rel" and k2 != "node" and k2 != "prop"}} for k,v in METRIC_META.items()],
        "states": states[:30],
        "districts": {"Kerala": districts},
        "years": VALID_YEARS,
    }


@app.get("/api/v1/states", tags=["Data"])
async def get_states():
    try:
        data = run_cypher('MATCH (c:Country {name:"India"})-[:HAS_STATE]->(s:State) WHERE toLower(s.name) <> "total" RETURN s.name AS name ORDER BY s.name')
        return {"states": [r["name"] for r in data]}
    except Exception:
        return {"states": list(STATE_CENTROIDS.keys())}


@app.get("/api/v1/states/{state}/districts", tags=["Data"])
async def get_districts(state: str):
    try:
        q = f'MATCH (s:State {{name:"{state.upper()}"}})-[:HAS_DISTRICT]->(d:District) RETURN d.name AS name ORDER BY d.name'
        data = run_cypher(q)
        return {"state": state, "districts": [r["name"] for r in data]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/metrics", tags=["Data"])
async def get_metrics():
    return {
        "metrics": [{"id": k, "label": v["label"], "unit": v["unit"]} for k, v in METRIC_META.items()],
        "thresholds": {
            "rainfall": {"very_low": "<500mm", "low": "500-1000mm", "normal": "1000-1500mm", "high": ">1500mm"},
            "stage_extraction": {"safe": "<70%", "semi_critical": "70-90%", "critical": "90-100%", "over_exploited": ">100%"},
        },
    }


@app.get("/api/v1/map/states", tags=["Map"])
async def map_states(metric: str = "availability", year: int = 2024):
    """Returns per-state data with coordinates for map visualization."""
    year = _valid_year(year)
    cache_key = f"map_states:{metric}:{year}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    parts = _metric_parts(metric if metric in METRIC_META else "availability")
    if not parts:
        raise HTTPException(400, "Invalid metric")
    rel_type, node_type, prop = parts
    q = (
        f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State)-[:HAS_YEAR]->(y:Year {{year:{year}}})'
        f'-[:HAS_{rel_type}]->(n:{node_type}) '
        f'WHERE n.{prop} IS NOT NULL AND toLower(s.name) <> "total" RETURN s.name AS state, n.{prop} AS value ORDER BY s.name'
    )
    try:
        rows = run_cypher(q)
    except Exception as e:
        raise HTTPException(500, str(e))

    result = []
    for r in rows:
        name = r.get("state", "")
        coords = STATE_CENTROIDS.get(name, {"lat": 20.5937, "lng": 78.9629})
        result.append({
            "state": name,
            "value": float(r.get("value") or 0),
            "lat": coords["lat"],
            "lng": coords["lng"],
        })
    payload = {"metric": metric, "unit": METRIC_META.get(metric, {}).get("unit", ""), "year": year, "data": result}
    cache_set(cache_key, payload, ttl=300)
    return payload


@app.post("/api/v1/data/export", tags=["Export"])
async def export_data(request: ExportRequest):
    """Export filtered data as CSV string."""
    import csv
    import io
    year = _valid_year((request.filters or {}).get("year", 2024))
    metric = request.metrics[0] if request.metrics else "rainfall"

    if request.comparison_type == "state" and request.states:
        cypher = generate_state_comparison_query(request.states, metric, year)
    elif request.comparison_type == "district" and request.districts:
        cypher = generate_district_comparison_query("Kerala", request.districts, metric, year)
    elif request.comparison_type == "yearly":
        entity = (request.filters or {}).get("entity", "Kerala")
        etype  = (request.filters or {}).get("entity_type", "state")
        valid_years = [y for y in (request.years or VALID_YEARS) if y in VALID_YEARS]
        cypher = generate_yearly_trend_query(entity, etype, metric, valid_years)
    else:
        raise HTTPException(400, "Cannot build export query from given parameters")

    if not cypher:
        raise HTTPException(400, "Could not build query")

    rows = run_cypher(cypher)
    if not rows:
        raise HTTPException(404, "No data found")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    csv_content = output.getvalue()

    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=jalmitra_{metric}_{year}.csv"},
    )


@app.post("/api/v1/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    logger.info(f"feedback rating={request.rating} query_len={len(request.query)}")
    return {"status": "received", "message": "Thank you for your feedback!"}


@app.get("/api/v1/suggestions", tags=["Chat"])
async def get_suggestions(q: str = ""):
    """Query suggestion autocomplete."""
    all_suggestions = [
        "What is the rainfall in Kerala?",
        "Show groundwater draft for Karnataka",
        "Compare recharge rates between Punjab and Haryana",
        "Which states have over-exploited groundwater?",
        "Groundwater availability in Tamil Nadu 2024",
        "Show district-wise data for Kottayam",
        "Yearly trend of rainfall in Maharashtra",
        "Stage of extraction in Rajasthan",
        "Critical groundwater districts in Kerala",
        "Compare water availability across southern states",
        "Groundwater recharge in Gujarat 2023",
        "Over-exploited areas in Uttar Pradesh",
    ]
    if not q:
        return {"suggestions": all_suggestions[:6]}
    filtered = [s for s in all_suggestions if q.lower() in s.lower()]
    return {"suggestions": filtered[:6]}


# ---------- New feature models ----------
class AdvisoryRequest(BaseModel):
    state: str
    crop: str
    district: Optional[str] = None

class SimulateRequest(BaseModel):
    state: str
    district: Optional[str] = None
    draft_change_pct: float = Field(..., ge=-90, le=200, description="Percent change to agricultural draft, e.g. -15 for a 15% reduction")
    horizon: int = Field(default=5, ge=1, le=10)

class ObservationRequest(BaseModel):
    state: str
    district: Optional[str] = None
    well_depth_m: float = Field(..., gt=0, le=1000)
    note: Optional[str] = Field(default=None, max_length=500)


# ---------- 2.1 Forecasting ----------
@app.get("/api/v1/forecast/{state}", tags=["Forecast"])
async def forecast_state(state: str, horizon: int = 3):
    try:
        return forecast_service.build_forecast(state, district=None, horizon=horizon)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/v1/forecast/{state}/{district}", tags=["Forecast"])
async def forecast_district(state: str, district: str, horizon: int = 3):
    try:
        return forecast_service.build_forecast(state, district=district, horizon=horizon)
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- 2.3 Farmer advisory ----------
@app.post("/api/v1/advisory", tags=["Advisory"])
async def advisory(request: AdvisoryRequest):
    result = advisory_service.get_advisory(request.state, request.crop, request.district)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result

@app.get("/api/v1/advisory/crops", tags=["Advisory"])
async def advisory_crops():
    return {"crops": list(advisory_service.CROP_WATER_REQUIREMENTS.keys())}


# ---------- 2.4 What-if simulator ----------
@app.post("/api/v1/simulate", tags=["Simulator"])
async def simulate(request: SimulateRequest):
    try:
        return forecast_service.build_forecast(
            request.state, district=request.district, horizon=request.horizon,
            draft_change_pct=request.draft_change_pct,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- 2.5 Crowdsourced field observations ----------
@app.post("/api/v1/field-observations", tags=["Field Data"], dependencies=[Depends(rate_limit)])
async def submit_field_observation(request: ObservationRequest, req: Request):
    ip = req.client.host if req.client else "unknown"
    try:
        result = field_observations_service.submit_observation(
            request.state, request.district, request.well_depth_m, request.note, ip,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return result

@app.get("/api/v1/field-observations", tags=["Field Data"])
async def get_field_observations(state: Optional[str] = None, district: Optional[str] = None, limit: int = 50):
    return {"observations": field_observations_service.list_observations(state, district, min(limit, 200))}


# ---------- 2.7 PDF report generation ----------
@app.get("/api/v1/reports/{state}", tags=["Reports"])
async def report_state(state: str, years: str = "2023,2024"):
    year_list = [int(y) for y in years.split(",") if y.strip().isdigit()]
    try:
        pdf_bytes = reports_service.generate_report_pdf(state, None, year_list)
    except Exception as e:
        raise HTTPException(500, str(e))
    return StreamingResponse(
        iter([pdf_bytes]), media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=jalmitra_report_{state}.pdf"},
    )

@app.get("/api/v1/reports/{state}/{district}", tags=["Reports"])
async def report_district(state: str, district: str, years: str = "2023,2024"):
    year_list = [int(y) for y in years.split(",") if y.strip().isdigit()]
    try:
        pdf_bytes = reports_service.generate_report_pdf(state, district, year_list)
    except Exception as e:
        raise HTTPException(500, str(e))
    return StreamingResponse(
        iter([pdf_bytes]), media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=jalmitra_report_{state}_{district}.pdf"},
    )


# ---------- 2.7b Full structured data export (Neo4j) ----------
@app.get("/api/v1/data/categories", tags=["Reports"])
async def data_categories():
    """Every Neo4j node category available for selective export."""
    return {"categories": [{"id": k, "label": v["label"]} for k, v in reports_service.DATA_CATEGORIES.items()]}


@app.get("/api/v1/data/export-full", tags=["Reports"])
async def export_full_dataset(
    state: str,
    district: Optional[str] = None,
    years: str = "2023,2024",
    categories: str = "",
    format: str = "json",
):
    year_list = [int(y) for y in years.split(",") if y.strip().isdigit()] or VALID_YEARS
    cat_list = [c.strip() for c in categories.split(",") if c.strip()] or list(reports_service.DATA_CATEGORIES.keys())
    invalid = [c for c in cat_list if c not in reports_service.DATA_CATEGORIES]
    if invalid:
        raise HTTPException(400, f"Unknown categories: {', '.join(invalid)}")

    dataset = reports_service.fetch_full_dataset(state, district, year_list, cat_list)
    label = f"{state}_{district}" if district else state

    if format == "csv":
        csv_content = reports_service.dataset_to_csv(dataset)
        return StreamingResponse(
            iter([csv_content]), media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=jalmitra_full_export_{label}.csv"},
        )
    return dataset


# ---------- 2.7c Pinecone semantic record export ----------
@app.get("/api/v1/data/pinecone-export", tags=["Reports"])
async def export_pinecone_data(state: str, format: str = "json"):
    records = reports_service.export_pinecone_records(state)
    if format == "csv":
        csv_content = reports_service.pinecone_records_to_csv(records)
        return StreamingResponse(
            iter([csv_content]), media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=jalmitra_pinecone_{state}.csv"},
        )
    return {"state": state.upper(), "records": records}


# ---------- 3.4 Data freshness ----------
@app.get("/api/v1/data/freshness", tags=["Data"])
async def data_freshness():
    return {
        "years_available": VALID_YEARS,
        "district_coverage": "Kerala",
        "last_ingested": "2024 CGWB/Jal Shakti assessment",
    }


if __name__ == "__main__":
    import uvicorn
    # `python server.py` is a dev convenience only — production runs via the
    # Dockerfile's `uvicorn server:app --workers 1` (no --reload).
    dev_mode = os.getenv("ENVIRONMENT", "development") != "production"
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=dev_mode, log_level="info")
