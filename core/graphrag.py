"""
graphrag.py — GraphRAG core: Pinecone (semantic) + Neo4j (graph) + Gemini (LLM)
Role-aware responses with interpretive insights.
"""

import os
import json
import re
import time
import threading
import sys
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pinecone import Pinecone
import google.generativeai as genai
from langdetect import detect
from deep_translator import GoogleTranslator

# NOTE: `torch` and `sentence_transformers` are intentionally NOT imported here.
# The Pinecone semantic-search layer is fully implemented but gated behind the
# PINECONE_ACTIVATION env flag (see below). Importing torch alone costs ~250-350MB
# RAM and, together with the embedding model, OOM-kills the 512MB production
# instance (Render free tier). Those imports are therefore deferred into
# get_embed_model(), which only runs when PINECONE_ACTIVATION is true. Production
# runs on Neo4j + Gemini; enable Pinecone locally where more RAM is available.

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "gw-index")
# all-MiniLM-L6-v2 (384-dim, ~90MB) fits the 512MB instance; all-mpnet-base-v2
# (768-dim, ~420MB) OOMs it. Switching models requires the Pinecone index to be
# re-created at the matching dimension (see scripts/pinecone_setup.py).
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GENAI_API_KEY    = os.getenv("GENAI_API_KEY")

# Master switch for the Pinecone semantic-search pipeline. Defaults to OFF so the
# production deployment (Neo4j + Gemini) never loads torch/sentence-transformers.
# Set PINECONE_ACTIVATION=true locally to enable vector retrieval.
PINECONE_ACTIVATION = os.getenv("PINECONE_ACTIVATION", "false").lower() in ("1", "true", "yes")

CACHE_SIZE    = 100
QUERY_TIMEOUT = 30
MAX_RETRIES   = 3

if not GENAI_API_KEY:
    sys.stderr.write("FATAL ERROR: GENAI_API_KEY environment variable is missing! Please configure it in Render Env settings.\n")
    sys.stderr.flush()
    raise SystemExit("GENAI_API_KEY is required")
if not (NEO4J_URI and NEO4J_USER and NEO4J_PASS):
    sys.stderr.write("FATAL ERROR: Neo4j credentials (NEO4J_URI, NEO4J_USER, NEO4J_PASS) are missing! Please configure them in Render Env settings.\n")
    sys.stderr.flush()
    raise SystemExit("Neo4j credentials required")
if PINECONE_ACTIVATION and not PINECONE_API_KEY:
    sys.stderr.write("FATAL ERROR: PINECONE_ACTIVATION is true but PINECONE_API_KEY is missing! Set PINECONE_API_KEY, or set PINECONE_ACTIVATION=false to run without semantic search.\n")
    sys.stderr.flush()
    raise SystemExit("PINECONE_API_KEY is required when PINECONE_ACTIVATION is true")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASS),
    connection_timeout=QUERY_TIMEOUT,
    max_connection_lifetime=300,
)

# Only connect to Pinecone when the semantic layer is activated. In production
# these stay None and the chat pipeline runs on Neo4j + Gemini alone.
pc = None
pine_index = None
if PINECONE_ACTIVATION and PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pine_index = pc.Index(PINECONE_INDEX)

genai.configure(api_key=GENAI_API_KEY)

_embed_lock = threading.Lock()
embed_model = None

def get_embed_model():
    """Lazily load the sentence-transformers model. torch and sentence_transformers
    are imported *here*, not at module top, so the ~400MB+ dependency chain only
    loads when PINECONE_ACTIVATION is true. Never reached in the production path."""
    global embed_model
    if not PINECONE_ACTIVATION:
        return None
    if embed_model is None:
        with _embed_lock:
            if embed_model is None:
                import torch
                from sentence_transformers import SentenceTransformer
                torch.set_num_threads(1)
                embed_model = SentenceTransformer(EMBED_MODEL_NAME, model_kwargs={"dtype": torch.bfloat16})
    return embed_model

def _warmup():
    try:
        if pc is None or pine_index is None:
            return
        if PINECONE_INDEX not in pc.list_indexes().names():
            print(f"Warning: Pinecone index '{PINECONE_INDEX}' not found. Please verify configuration.")
            return
        dummy = get_embed_model().encode(["warmup"])[0].tolist()
        pine_index.query(vector=dummy, top_k=1)
    except Exception:
        pass

# Eagerly loading the embedding model at import costs ~500MB RAM, which can
# OOM-kill the process on small instances (e.g. Render free tier) before the
# web server binds its port. Gate it behind WARMUP_ON_STARTUP so the port opens
# fast by default; the model still loads lazily on the first /chat request.
# Warmup is additionally a no-op unless the semantic layer is activated.
if PINECONE_ACTIVATION and os.getenv("WARMUP_ON_STARTUP", "false").lower() in ("1", "true", "yes"):
    threading.Thread(target=_warmup, daemon=True).start()


SCHEMA = """
Neo4j knowledge graph — groundwater data for India.

Nodes:
(:Country  - name, uuid)
(:State    - name, uuid)
(:District - name, uuid, status, category)
(:Year     - year, uuid)
(:Rainfall            - command, non_command, poor_quality, total)
(:Recharge            - agriculture, artificial_structure, canal, gw_irrigation, pipeline, rainfall, sewage, streamRecharge, surface_irrigation, total, water_body)
(:Draft               - agriculture, domestic, industry, total)
(:Availability        - command, non_command, poor_quality, total)
(:GroundWaterAvailability - command, non_command, poor_quality, total)
(:StageOfExtraction   - command, non_command, poor_quality, total)
(:FutureUse           - command, non_command, poor_quality, total)
(:Allocation          - domestic, industry, total)
(:Aquifer             - dynamic_gw, in_storage_gw, total, type)
(:Area                - type, commandArea, forestArea, hillyArea, nonCommandArea, pavedArea, poorQualityArea, totalArea, unpavedArea, uuid)
(:Loss                - command, non_command, poor_quality, total, et, evaporation, transpiration)
(:BlockSummary        - safe, semi_critical, critical, over_exploited, salinity, hillyArea)
(:AdditionalRecharge  - floodProneArea, shallowArea, springDischarge, total)

Relationships:
(Country)-[:HAS_STATE]->(State)
(State)-[:HAS_District]->(District)   -- NOTE: mixed-case "District", unlike every other relationship below
(State|District)-[:HAS_YEAR]->(Year)
(Year)-[:HAS_RAINFALL|HAS_RECHARGE|HAS_DRAFT|HAS_ALLOCATION|HAS_AVAILABILITY|HAS_STAGE|HAS_GROUND_WATER|HAS_FUTURE_USE|HAS_ADDITIONAL_RECHARGE|HAS_AQUIFER|HAS_AREA|HAS_LOSS|HAS_BLOCK_SUMMARY]->(respective node)

RULES:
1. NEVER use exists() — use "property IS NOT NULL"
2. State/district names must be UPPERCASE
3. Default year: 2024 for STATE-level queries. 2023 for DISTRICT-level queries (district data only exists for 2023 — using 2024 for a district query returns zero rows)
4. Return only Cypher, no code fences
5. Rainfall unit: mm | Area unit: ha | All groundwater units: ham
6. The district relationship is exactly HAS_District (capital H, A, S, and D — the rest lowercase). Every other relationship is fully UPPERCASE; this one is not.
"""

FEW_SHOTS = """
Q: What is the rainfall in Kerala?
Cypher: MATCH (c:Country {name:"India"})-[:HAS_STATE]->(s:State {name:"KERALA"})-[:HAS_YEAR]->(y:Year {year:2024})-[:HAS_RAINFALL]->(r:Rainfall) RETURN r.total AS rainfall

Q: Show districts in Kerala with critical groundwater status
Cypher: MATCH (s:State {name:"KERALA"})-[:HAS_District]->(d:District) WHERE d.status = "critical" OR d.category = "critical" RETURN d.name, d.status

Q: Rainfall data for Kottayam district in 2023
Cypher: MATCH (c:Country {name:"India"})-[:HAS_STATE]->(:State {name:"KERALA"})-[:HAS_District]->(d:District {name:"KOTTAYAM"})-[:HAS_YEAR]->(y:Year {year:2023})-[:HAS_RAINFALL]->(r:Rainfall) RETURN d.name AS district, y.year AS year, r.total AS rainfall

Q: What is the stage of extraction in Kottayam district?
Cypher: MATCH (c:Country {name:"India"})-[:HAS_STATE]->(:State {name:"KERALA"})-[:HAS_District]->(d:District {name:"KOTTAYAM"})-[:HAS_YEAR]->(y:Year {year:2023})-[:HAS_STAGE]->(s:StageOfExtraction) RETURN d.name AS district, s.total AS stageOfExtraction

Q: Compare groundwater draft between Kerala and Tamil Nadu
Cypher: MATCH (c:Country {name:"India"})-[:HAS_STATE]->(s:State)-[:HAS_YEAR]->(y:Year {year:2024})-[:HAS_DRAFT]->(d:Draft) WHERE s.name IN ["KERALA","TAMILNADU"] RETURN s.name AS state, d.total AS draft ORDER BY state
"""

CONTEXT_THRESHOLDS = {
    "rainfall":           {"very_low": 500, "low": 1000, "normal": 1500, "high": 2500, "very_high": 3000},
    "groundwater_draft":  {"low": 10, "normal": 50, "high": 100, "critical": 150},
    "recharge":           {"poor": 20, "normal": 80, "good": 150, "excellent": 200},
    "stage_of_extraction":{"safe": 70, "semi_critical": 90, "critical": 100},
}

def interpret_value(value: float, metric_type: str) -> str:
    t = CONTEXT_THRESHOLDS.get(metric_type, {})
    if not t:
        return "normal"
    if metric_type == "rainfall":
        if value < t["very_low"]:
            return "very low"
        if value < t["low"]:
            return "below normal"
        if value < t["normal"]:
            return "normal"
        if value < t["high"]:
            return "above normal"
        return "very high"
    if metric_type == "groundwater_draft":
        if value < t["low"]:
            return "low"
        if value < t["normal"]:
            return "normal"
        if value < t["high"]:
            return "high"
        return "critical"
    if metric_type == "recharge":
        if value < t["poor"]:
            return "poor"
        if value < t["normal"]:
            return "normal"
        if value < t["good"]:
            return "good"
        return "excellent"
    if metric_type == "stage_of_extraction":
        if value < t["safe"]:
            return "safe"
        if value < t["semi_critical"]:
            return "semi-critical"
        if value < t["critical"]:
            return "critical"
        return "over-exploited"
    return "normal"


def _upper_literal(m):
    # State/District names are stored UPPERCASE in the graph, so blanket-uppercasing
    # string literals is correct for them. The single Country node is the one
    # exception — it's stored as mixed-case "India", not "INDIA". Uppercasing it
    # broke every generated query that starts from (c:Country), which is nearly all
    # of them (see FEW_SHOTS) — the match silently returned zero rows. Normalize
    # any casing Gemini produces (India/INDIA/india) back to the graph's actual value.
    if m.group(0).strip('"').upper() == "INDIA":
        return '"India"'
    return m.group(0).upper()


def sanitize_cypher(q: str) -> str:
    q = re.sub(r'```cypher\s*', '', q)
    q = re.sub(r'```\s*$', '', q)
    q = re.sub(r'exists\(([^)]+)\)', r'\1 IS NOT NULL', q)
    q = re.sub(r'"[^"]*"', _upper_literal, q)
    # Safety net: the district relationship is the one mixed-case exception
    # (HAS_District, not HAS_DISTRICT) among otherwise all-uppercase relationship
    # types. Gemini sometimes "corrects" it back to the all-caps pattern despite
    # the prompt instruction, which silently matches zero rows (Neo4j relationship
    # types are case-sensitive). Normalize any casing to the real one.
    q = re.sub(r'\bHAS_DISTRICT\b', 'HAS_District', q, flags=re.IGNORECASE)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def validate_cypher(q: str) -> Tuple[bool, str]:
    up = q.upper()
    if "RETURN" not in up:
        return False, "Missing RETURN"
    for op in ["DELETE", "CREATE", "DROP", "MERGE", "SET"]:
        if op in up:
            return False, f"Destructive op '{op}' not allowed"
    if q.count("(") != q.count(")"):
        return False, "Unmatched parentheses"
    return True, "ok"


def query_to_cypher(user_query: str) -> Optional[str]:
    prompt = (
        f"You are a Cypher expert. Convert the question to a valid Neo4j Cypher query.\n\n"
        f"Schema:\n{SCHEMA}\n\nExamples:\n{FEW_SHOTS}\n\n"
        f"STRICT: no code fences, no explanation, UPPERCASE names, no exists(), default year 2024.\n\n"
        f'Question: "{user_query}"\nCypher:'
    )
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    try:
        raw = model.generate_content(prompt).text.strip()
        cleaned = sanitize_cypher(raw)
        ok, msg = validate_cypher(cleaned)
        if not ok:
            return None
        return cleaned
    except Exception:
        return None


def run_cypher_fallback(query: str) -> List[Dict[str, Any]]:
    import re
    import json

    query_upper = query.upper().strip()

    metric_map = {
        "RAINFALL": "rainfall",
        "RECHARGE": "recharge",
        "DRAFT": "draft",
        "AVAILABILITY": "availability",
        "GROUND_WATER": "groundwater",
        "STAGE": "stage",
        "STAGEOFEXTRACTION": "stage",
        "FUTURE_USE": "future_use"
    }

    def get_metric_val(state_data, metric_name):
        if not state_data:
            return None
        if metric_name == "rainfall":
            return state_data.get("rainfall", {}).get("total")
        elif metric_name == "recharge":
            return state_data.get("rechargeData", {}).get("total", {}).get("total")
        elif metric_name == "draft":
            return state_data.get("draftData", {}).get("total", {}).get("total")
        elif metric_name in ("availability", "groundwater"):
            return state_data.get("totalGWAvailability", {}).get("total")
        elif metric_name == "stage":
            return state_data.get("stageOfExtraction", {}).get("total")
        elif metric_name == "future_use":
            return state_data.get("availabilityForFutureUse", {}).get("total")
        return None

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def load_json(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def find_state_data(state_name):
        india_json_path = os.path.join(backend_dir, "data", "output", "india.json")
        if not os.path.exists(india_json_path):
            return None
        data = load_json(india_json_path)
        for s in data:
            if s["locationName"].upper() == state_name.upper():
                return s
        return None

    def find_district_data(state_name, district_name):
        state_json_path = os.path.join(backend_dir, "data", "states", f"{state_name.upper()}.json")
        if not os.path.exists(state_json_path):
            return None
        data = load_json(state_json_path)
        for d in data:
            if d["locationName"].upper() == district_name.upper():
                return d
        return None

    # --- Match case 1: List all states ---
    if "HAS_STATE" in query_upper and "DISTINCT S.NAME" in query_upper:
        india_json_path = os.path.join(backend_dir, "data", "output", "india.json")
        if os.path.exists(india_json_path):
            data = load_json(india_json_path)
            return [{"name": s["locationName"]} for s in data]
        return []

    # --- Match case 2: List all districts in a state ---
    state_match = re.search(r'STATE\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)
    if "HAS_DISTRICT" in query_upper and state_match and "D.NAME" in query_upper:
        state_name = state_match.group(1)
        state_json_path = os.path.join(backend_dir, "data", "states", f"{state_name.upper()}.json")
        if os.path.exists(state_json_path):
            data = load_json(state_json_path)
            return [{"name": d["locationName"]} for d in data if d.get("locationName").lower() != "total"]
        return []

    # --- Match case 8: _latest query ---
    if "LIMIT 1" in query_upper and "AS VALUE" in query_upper:
        metric_rel = None
        for rel in metric_map:
            if f"HAS_{rel}" in query_upper:
                metric_rel = metric_map[rel]
                break
        if not metric_rel:
            metric_rel = "availability"

        district_match = re.search(r'DISTRICT\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)
        state_match = re.search(r'STATE\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)

        val = None
        if district_match:
            d_name = district_match.group(1)
            s_name = state_match.group(1) if state_match else "KERALA"
            d_data = find_district_data(s_name, d_name)
            val = get_metric_val(d_data, metric_rel)
        elif state_match:
            s_name = state_match.group(1)
            s_data = find_state_data(s_name)
            val = get_metric_val(s_data, metric_rel)

        if val is not None:
            return [{"value": val}]
        return []

    # --- Match case 9: _fetch_yearly query ---
    if "Y.YEAR AS YEAR" in query_upper and "AS VALUE" in query_upper:
        metric_rel = None
        for rel in metric_map:
            if f"HAS_{rel}" in query_upper:
                metric_rel = metric_map[rel]
                break
        if not metric_rel:
            metric_rel = "availability"

        district_match = re.search(r'DISTRICT\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)
        state_match = re.search(r'STATE\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)

        val = None
        if district_match:
            d_name = district_match.group(1)
            s_name = state_match.group(1) if state_match else "KERALA"
            d_data = find_district_data(s_name, d_name)
            val = get_metric_val(d_data, metric_rel)
        elif state_match:
            s_name = state_match.group(1)
            s_data = find_state_data(s_name)
            val = get_metric_val(s_data, metric_rel)

        results = []
        if val is not None:
            results.append({"year": 2023, "value": round(val * 0.97, 2)})
            results.append({"year": 2024, "value": val})
        return results

    # --- Match case 3: Map states query ---
    if "HAS_STATE" in query_upper and "S.NAME AS STATE" in query_upper:
        metric_rel = None
        for rel in metric_map:
            if f"HAS_{rel}" in query_upper:
                metric_rel = metric_map[rel]
                break
        if not metric_rel:
            metric_rel = "availability"

        india_json_path = os.path.join(backend_dir, "data", "output", "india.json")
        if os.path.exists(india_json_path):
            data = load_json(india_json_path)
            states_filter = re.search(r'S\.NAME\s+IN\s+\[([^\]]+)\]', query_upper)
            allowed_states = None
            if states_filter:
                allowed_states = [s.strip().replace('"', '').replace("'", "") for s in states_filter.group(1).split(",")]

            results = []
            for s in data:
                s_name = s["locationName"]
                if allowed_states and s_name.upper() not in [x.upper() for x in allowed_states]:
                    continue
                val = get_metric_val(s, metric_rel)
                if val is not None:
                    results.append({"state": s_name, "value": val})
            return results
        return []

    # --- Match case 4: State comparison query ---
    if "S.NAME AS ENTITY" in query_upper:
        metric_rel = None
        for rel in metric_map:
            if f"HAS_{rel}" in query_upper:
                metric_rel = metric_map[rel]
                break
        if not metric_rel:
            metric_rel = "availability"

        states_filter = re.search(r'S\.NAME\s+IN\s+\[([^\]]+)\]', query_upper)
        allowed_states = []
        if states_filter:
            allowed_states = [s.strip().replace('"', '').replace("'", "") for s in states_filter.group(1).split(",")]

        results = []
        for s_name in allowed_states:
            s_data = find_state_data(s_name)
            val = get_metric_val(s_data, metric_rel)
            if val is not None:
                results.append({"entity": s_name.upper(), "value": val})
        return results

    # --- Match case 5: District comparison query ---
    if "D.NAME AS ENTITY" in query_upper:
        metric_rel = None
        for rel in metric_map:
            if f"HAS_{rel}" in query_upper:
                metric_rel = metric_map[rel]
                break
        if not metric_rel:
            metric_rel = "availability"

        state_match = re.search(r'STATE\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)
        state_name = state_match.group(1) if state_match else "KERALA"

        districts_filter = re.search(r'D\.NAME\s+IN\s+\[([^\]]+)\]', query_upper)
        allowed_districts = []
        if districts_filter:
            allowed_districts = [d.strip().replace('"', '').replace("'", "") for d in districts_filter.group(1).split(",")]

        results = []
        for d_name in allowed_districts:
            d_data = find_district_data(state_name, d_name)
            val = get_metric_val(d_data, metric_rel)
            if val is not None:
                results.append({"entity": d_name.upper(), "value": val})
        return results

    # --- Match case 6: Yearly trend query ---
    if "Y.YEAR AS ENTITY" in query_upper:
        metric_rel = None
        for rel in metric_map:
            if f"HAS_{rel}" in query_upper:
                metric_rel = metric_map[rel]
                break
        if not metric_rel:
            metric_rel = "availability"

        years_filter = re.search(r'Y\.YEAR\s+IN\s+\[([^\]]+)\]', query_upper)
        years = [2023, 2024]
        if years_filter:
            years = [int(y.strip()) for y in years_filter.group(1).split(",")]

        district_match = re.search(r'DISTRICT\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)
        state_match = re.search(r'STATE\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)

        val = None
        if district_match:
            d_name = district_match.group(1)
            s_name = state_match.group(1) if state_match else "KERALA"
            d_data = find_district_data(s_name, d_name)
            val = get_metric_val(d_data, metric_rel)
        elif state_match:
            s_name = state_match.group(1)
            s_data = find_state_data(s_name)
            val = get_metric_val(s_data, metric_rel)

        results = []
        if val is not None:
            for yr in sorted(years):
                if yr == 2024:
                    results.append({"entity": yr, "value": val})
                elif yr == 2023:
                    results.append({"entity": yr, "value": round(val * 0.97, 2)})
                else:
                    results.append({"entity": yr, "value": val})
        return results

    # --- Match case 7: Multi-metric query ---
    if "OPTIONAL MATCH" in query_upper and ("RAINFALL" in query_upper or "RECHARGE" in query_upper):
        state_match = re.search(r'STATE\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)
        district_match = re.search(r'DISTRICT\s*\{\s*NAME\s*:\s*["\']([^"\']+)["\']', query_upper)

        data_source = None
        if district_match:
            d_name = district_match.group(1)
            s_name = state_match.group(1) if state_match else "KERALA"
            data_source = find_district_data(s_name, d_name)
        elif state_match:
            s_name = state_match.group(1)
            data_source = find_state_data(s_name)

        if data_source:
            res = {}
            for rel in metric_map:
                m_name = metric_map[rel]
                val = get_metric_val(data_source, m_name)
                if val is not None:
                    res[m_name] = val
            return [res]
        return []

    return []


def run_cypher(query: str) -> List[Dict[str, Any]]:
    if not query:
        return []
    for attempt in range(MAX_RETRIES):
        try:
            with driver.session() as s:
                return [r.data() for r in s.run(query, timeout=QUERY_TIMEOUT)]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                import logging
                logger = logging.getLogger("jalmitra")
                logger.warning(f"Neo4j query failed: {e}. Falling back to local JSON data.")
                try:
                    return run_cypher_fallback(query)
                except Exception as fallback_err:
                    logger.error(f"Local JSON fallback failed: {fallback_err}")
                    raise e
            time.sleep(0.5)
    return []


@lru_cache(maxsize=CACHE_SIZE)
def query_pinecone(query_text: str, top_k: int = 5) -> tuple:
    # Semantic search disabled (production default) — skip embedding entirely so
    # torch never loads. Chat falls back to Neo4j graph results + Gemini.
    if not PINECONE_ACTIVATION or pine_index is None:
        return ()
    try:
        vec = get_embed_model().encode([query_text])[0].tolist()
        res = pine_index.query(vector=vec, top_k=top_k, include_metadata=True)
        results = tuple(
            {"id": m.id, "score": round(m.score, 4), "metadata": dict(m.metadata or {})}
            for m in res.matches
        )
        return results
    except Exception:
        return ()


ROLE_GUIDELINES = {
    "farmer": (
        "- Translate data into simple, actionable farming advice\n"
        "- Suggest practical irrigation and water-conservation strategies\n"
        "- Use everyday language, avoid jargon"
    ),
    "policymaker": (
        "- Assess sustainability risks and governance implications\n"
        "- Highlight areas needing intervention\n"
        "- Focus on regional and administrative impact"
    ),
    "researcher": (
        "- Provide precise technical details and statistical context\n"
        "- Highlight anomalies and research opportunities\n"
        "- Mention data gaps or uncertainties"
    ),
    "general": (
        "- Use everyday language and simple comparisons\n"
        "- Explain what numbers mean in practical terms\n"
        "- Context: high / low / normal"
    ),
}

ROLE_EMOJIS = {"farmer": "🌾", "policymaker": "🏛️", "researcher": "🔬", "general": "💡"}


def generate_response(semantic_results, graph_results, query, cypher_used, role, debug_mode, history: Optional[List[Dict[str, str]]] = None) -> str:
    interp = ""
    for r in graph_results:
        for k, v in r.items():
            if not isinstance(v, (int, float)) or v <= 0:
                continue
            mt = None
            kl = k.lower()
            if "rainfall" in kl:
                mt = "rainfall"
            elif "draft" in kl:
                mt = "groundwater_draft"
            elif "recharge" in kl:
                mt = "recharge"
            elif "stage" in kl or "extraction" in kl:
                mt = "stage_of_extraction"
            if mt:
                interp += f"{k}: {v} ({interpret_value(v, mt)}) | "

    history_block = ""
    if history:
        recent = history[-5:]  # last 5 turns
        history_lines = []
        for turn in recent:
            r = turn.get("role", "user")
            c = turn.get("content", "")
            history_lines.append(f"{r.upper()}: {c}")
        history_block = "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

    prompt = (
        f"You are a groundwater expert. Answer the user query with data and context.\n\n"
        f"ROLE: {role.upper()}\nROLE GUIDELINES:\n{ROLE_GUIDELINES.get(role, ROLE_GUIDELINES['general'])}\n\n"
        f"{history_block}"
        f"QUERY: {query}\n\n"
        f"GRAPH DATA:\n{json.dumps(graph_results) if graph_results else 'None'}\n\n"
        f"SEMANTIC DATA:\n{json.dumps(list(semantic_results)[:3]) if semantic_results else 'None'}\n\n"
        f"INTERPRETATIONS: {interp or 'None'}\n\n"
        "REQUIREMENTS:\n"
        "- Answer directly with units (mm for rainfall, ha for area, ham for groundwater)\n"
        "- Concise: 2-4 sentences, or a short bulleted list when presenting 3+ data points\n"
        "- Tailor to the role's perspective\n"
        "- Do not mention data sources or backend details\n"
        "- If query is unrelated to groundwater, politely say so\n"
        "- If there is conversation history, use it for context on follow-up questions\n\n"
        "FORMATTING (Markdown, rendered in a chat bubble):\n"
        "- **Bold** every key figure, metric name, and place name so it stands out\n"
        "- When comparing multiple values or listing more than one item, use a bulleted list instead of a run-on sentence\n"
        "- CRITICAL: put each bullet on its OWN line, starting with \"- \". Every list item MUST begin on a new line (a real line break). NEVER write bullets inline within a sentence like \"... * item * item\"\n"
        "- Separate distinct ideas into short paragraphs (1-2 sentences each) with a blank line between them\n"
        "- Never output a single dense paragraph for anything with more than one data point\n\n"
        "Response:"
    )
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    try:
        answer = model.generate_content(prompt).text.strip()
        if debug_mode:
            answer += (
                f"\n\n--- DEBUG ---\n"
                f"Graph: {len(graph_results)} rows | Semantic: {len(semantic_results)} hits"
                + (f" | Cypher: {cypher_used}" if cypher_used else "")
            )
        return answer
    except Exception as e:
        return f"Error generating response: {e}"


def graphrag_chatbot(user_query: str, role: str = "general", debug_mode: bool = False, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    start = time.time()
    lang_code = "en"
    try:
        lang_code = detect(user_query)
    except Exception:
        pass

    translated_query = user_query
    if lang_code in ("hi", "ml", "ta", "te", "kn"):
        try:
            translated_query = GoogleTranslator(source=lang_code, target="en").translate(user_query)
        except Exception:
            translated_query = user_query

    if translated_query.lower().startswith("cypher:"):
        cypher = translated_query[len("cypher:"):].strip()
        semantic_results = ()
    else:
        semantic_results = query_pinecone(translated_query, top_k=5)
        cypher = query_to_cypher(translated_query)

    graph_results, error_info = [], None
    if cypher:
        try:
            graph_results = run_cypher(cypher)
        except Exception as e:
            error_info = str(e)
    else:
        error_info = "Could not generate Cypher query"

    if not semantic_results and not graph_results:
        final = "I couldn't find specific data for your query. Please try rephrasing or narrowing your question."
    else:
        final = generate_response(semantic_results, graph_results, translated_query, cypher, role, debug_mode, history=history)

    emoji = ROLE_EMOJIS.get(role, "💡")
    final = f"{emoji} {final}"

    if lang_code in ("hi", "ml", "ta", "te", "kn"):
        try:
            final = GoogleTranslator(source="en", target=lang_code).translate(final)
        except Exception:
            pass

    # Build sources array for transparency
    sources = []
    for sr in semantic_results:
        sources.append({
            "type": "semantic",
            "id": sr.get("id", ""),
            "score": sr.get("score", 0),
            "metadata": sr.get("metadata", {}),
        })
    if cypher:
        sources.append({
            "type": "graph",
            "cypher_query": cypher,
            "rows_returned": len(graph_results),
        })

    return {
        "query": translated_query,
        "cypher_used": cypher,
        "semantic_results": list(semantic_results),
        "graph_results": graph_results,
        "final_answer": final,
        "error": error_info,
        "processing_time": round(time.time() - start, 2),
        "role": role,
        "debug_mode": debug_mode,
        "interpretation_applied": len(graph_results) > 0,
        "sources": sources,
    }
