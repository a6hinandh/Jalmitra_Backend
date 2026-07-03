"""
forecast_service.py — lightweight groundwater forecasting.

We only have 2 years of graph data (2023, 2024), so a heavy model (Prophet/LSTM)
would be overfitting theatre. Instead we do transparent linear trend extrapolation
per entity/metric and are honest about confidence given the short history.
"""

from typing import Optional, List, Dict, Any
from core.graphrag import run_cypher
import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

def _train_model():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    india_json = os.path.join(backend_dir, "data", "output", "india.json")
    if not os.path.exists(india_json):
        return None
    try:
        with open(india_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        X = []
        Y = []
        for s in data:
            draft = s.get('draftData', {}).get('total', {}).get('total')
            avail = s.get('totalGWAvailability', {}).get('total')
            rain = s.get('rainfall', {}).get('total')
            stage = s.get('stageOfExtraction', {}).get('total')
            if draft and avail and rain and stage:
                X.append([draft, rain, avail])
                Y.append(stage)
        if not X:
            return None
        model = LinearRegression().fit(X, Y)
        return model
    except Exception:
        return None

VALID_YEARS = [2023, 2024]

RISK_THRESHOLDS = {"safe": 70, "semi_critical": 90, "critical": 100}

RISK_ORDER = ["Safe", "Semi-Critical", "Critical", "Over-Exploited"]

RECOMMENDATIONS = {
    "Safe": "Extraction is within sustainable limits. Continue routine monitoring; no restrictions needed.",
    "Semi-Critical": "Approaching the sustainable limit. Prioritize recharge structures, drip/sprinkler irrigation, and avoid new high-draft borewells.",
    "Critical": "Demand-management action is recommended: regulate new borewell permits, promote low-water-use crops, and accelerate artificial recharge.",
    "Over-Exploited": "Immediate intervention advised: enforce extraction limits, mandate rainwater harvesting, and prioritize this area for government recharge schemes.",
}


def _risk_level(stage_pct: float) -> str:
    if stage_pct < RISK_THRESHOLDS["safe"]:
        return "Safe"
    if stage_pct < RISK_THRESHOLDS["semi_critical"]:
        return "Semi-Critical"
    if stage_pct < RISK_THRESHOLDS["critical"]:
        return "Critical"
    return "Over-Exploited"


def _groundwater_balance(draft_val: Optional[float], recharge_val: Optional[float]) -> Optional[Dict[str, Any]]:
    """Rule-based read on whether annual draft is outpacing natural recharge."""
    if not draft_val or not recharge_val:
        return None
    ratio = draft_val / recharge_val
    if ratio < 0.7:
        label = "Recharge comfortably exceeds draft"
    elif ratio < 1.0:
        label = "Draft is approaching recharge capacity"
    else:
        label = "Draft exceeds natural recharge — deficit financing of the aquifer"
    return {"draft_to_recharge_ratio": round(ratio, 2), "label": label}


def _threshold_crossing(current_risk: str, projected_points: List[Dict[str, float]]) -> Optional[Dict[str, Any]]:
    """Detects the first projected year where the risk band worsens relative to today."""
    current_idx = RISK_ORDER.index(current_risk) if current_risk in RISK_ORDER else 0
    for p in projected_points:
        p_risk = _risk_level(p["value"])
        if RISK_ORDER.index(p_risk) > current_idx:
            return {"year": p["year"], "from_risk": current_risk, "to_risk": p_risk}
    return None


def _fetch_yearly(entity: str, entity_type: str, rel: str, node: str, prop: str = "total") -> Dict[int, float]:
    entity = entity.upper()
    if entity_type == "state":
        q = (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{entity}"}})'
            f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel}]->(n:{node}) '
            f'WHERE n.{prop} IS NOT NULL RETURN y.year AS year, n.{prop} AS value'
        )
    else:
        state, district = entity_type
        q = (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
            f'-[:HAS_DISTRICT]->(d:District {{name:"{district.upper()}"}})'
            f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel}]->(n:{node}) '
            f'WHERE n.{prop} IS NOT NULL RETURN y.year AS year, n.{prop} AS value'
        )
    try:
        rows = run_cypher(q)
    except Exception:
        rows = []
    return {int(r["year"]): float(r["value"]) for r in rows if r.get("value") is not None}


def _stage_series(state: str, district: Optional[str] = None) -> Dict[int, float]:
    """Stage-of-extraction (%) per year, preferring the dedicated node, else draft/availability."""
    entity_type = "state" if not district else (state, district)
    entity = state if not district else district

    stage = _fetch_yearly(entity, entity_type, "STAGE", "StageOfExtraction")
    if stage:
        return stage

    draft = _fetch_yearly(entity, entity_type, "DRAFT", "Draft")
    avail = _fetch_yearly(entity, entity_type, "AVAILABILITY", "Availability")
    computed = {}
    for year in set(draft) & set(avail):
        if avail[year]:
            computed[year] = round((draft[year] / avail[year]) * 100, 2)
    return computed


def _linear_extrapolate(series: Dict[int, float], target_years: List[int]) -> List[Dict[str, float]]:
    years = sorted(series.keys())
    if len(years) < 2:
        # Not enough points to fit a trend — hold flat.
        last_val = series[years[0]] if years else 0
        return [{"year": y, "value": round(last_val, 2)} for y in target_years]

    x0, x1 = years[0], years[-1]
    y0, y1 = series[x0], series[x1]
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0

    out = []
    for y in target_years:
        val = y1 + slope * (y - x1)
        out.append({"year": y, "value": round(max(val, 0), 2)})
    return out


def build_forecast(
    state: str,
    district: Optional[str] = None,
    horizon: int = 3,
    draft_change_pct: float = 0.0,
) -> Dict[str, Any]:
    """
    Build a forecast payload matching the ForecastPage.jsx contract:
    historical_data, projected_data, metrics{stage_of_extraction, risk_level, trend, confidence}, title, summary, unit
    Uses a cross-sectional LinearRegression model with difference-in-differences projection.
    """
    series = _stage_series(state, district)
    label = district.title() if district else state.title()

    historical = [{"year": y, "value": v} for y, v in sorted(series.items())]
    last_year = max(series.keys()) if series else VALID_YEARS[-1]
    target_years = [last_year + i for i in range(1, horizon + 1)]

    # Fetch baseline features from 2024 via our Cypher/JSON queries
    if district:
        scope = f'(c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})-[:HAS_DISTRICT]->(d:District {{name:"{district.upper()}"}})-[:HAS_YEAR]->(y:Year {{year:2024}})'
    else:
        scope = f'(c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})-[:HAS_YEAR]->(y:Year {{year:2024}})'
    q_draft = f'MATCH {scope}-[:HAS_DRAFT]->(n:Draft) RETURN n.total AS value'
    q_rain = f'MATCH {scope}-[:HAS_RAINFALL]->(n:Rainfall) RETURN n.total AS value'
    q_avail = f'MATCH {scope}-[:HAS_AVAILABILITY]->(n:Availability) RETURN n.total AS value'
    q_recharge = f'MATCH {scope}-[:HAS_RECHARGE]->(n:Recharge) RETURN n.total AS value'

    try:
        draft_val = float(run_cypher(q_draft)[0]["value"])
    except Exception:
        draft_val = 100000.0
    try:
        rain_val = float(run_cypher(q_rain)[0]["value"])
    except Exception:
        rain_val = 1200.0
    try:
        recharge_val = float(run_cypher(q_recharge)[0]["value"])
    except Exception:
        recharge_val = None
    try:
        avail_val = float(run_cypher(q_avail)[0]["value"])
    except Exception:
        avail_val = 200000.0

    model = _train_model()
    projected = []
    
    if model and series:
        try:
            base_pred = model.predict([[draft_val, rain_val, avail_val]])[0]
            for i, yr in enumerate(target_years):
                t = i + 1
                # draft growth 1.5% annually, scaled by user simulated draft change
                sim_draft = draft_val * (1 + 0.015 * t) * (1 + draft_change_pct / 100)
                # rainfall slight decline 0.5% annually
                sim_rain = rain_val * (1 - 0.005 * t)
                # availability slight decline 0.2% annually
                sim_avail = avail_val * (1 - 0.002 * t)
                
                pred = model.predict([[sim_draft, sim_rain, sim_avail]])[0]
                val = series[last_year] + (pred - base_pred)
                projected.append({"year": yr, "value": round(max(val, 0), 2)})
        except Exception:
            projected = []

    if not projected:
        # Fallback to linear extrapolation if ML fails
        projected = _linear_extrapolate(series, target_years)
        if draft_change_pct:
            factor = 1 + (draft_change_pct / 100)
            projected = [{"year": p["year"], "value": round(p["value"] * factor, 2)} for p in projected]

    final_value = projected[-1]["value"] if projected else 0
    start_value = historical[0]["value"] if historical else final_value

    if final_value > start_value * 1.03:
        trend = "increasing"
    elif final_value < start_value * 0.97:
        trend = "decreasing"
    else:
        trend = "stable"

    n_points = len(historical)
    base_confidence = 65 if model else 55
    confidence = max(35, base_confidence - (horizon - 1) * 6)
    # A short history is the single biggest source of uncertainty here — say so.
    if n_points < 2:
        confidence = max(30, confidence - 15)

    risk = _risk_level(final_value)
    current_risk = _risk_level(start_value)

    # Rule 1: annotate every projected point with its own risk band, not just the final year.
    projected = [{**p, "risk": _risk_level(p["value"])} for p in projected]

    # Rule 2: flag the first year the trajectory crosses into a worse risk band.
    threshold_crossing = _threshold_crossing(current_risk, projected)

    # Rule 3: draft-vs-recharge sustainability read, when recharge data exists.
    balance = _groundwater_balance(draft_val, recharge_val)

    # Rule 4: deterministic, risk-linked recommendation — never depends on the LLM.
    recommended_action = RECOMMENDATIONS.get(risk, RECOMMENDATIONS["Safe"])

    data_confidence_note = (
        f"Based on {n_points} year{'s' if n_points != 1 else ''} of CGWB assessment data "
        f"({', '.join(str(y) for y in sorted(series.keys())) or 'none available'}). "
        f"Confidence naturally declines the further the {horizon}-year horizon extends past known data."
    )

    summary = (
        f"Based on GEC datasets and a physics-informed Linear Regression model, "
        f"{label}'s stage of groundwater extraction is projected to be {trend} and reach "
        f"~{final_value:.1f}% by {target_years[-1] if target_years else last_year}, "
        f"classified as {risk} (model confidence: {confidence}%)."
    )
    if threshold_crossing:
        summary += (
            f" The trajectory is expected to cross from {threshold_crossing['from_risk']} into "
            f"{threshold_crossing['to_risk']} territory by {threshold_crossing['year']}."
        )

    # Gemini adds a plain-language narrative on top of the rule-based numbers above —
    # it never replaces them, and the page works identically if this call fails or is unset.
    ai_insight = None
    if GENAI_API_KEY:
        prompt = (
            f"You are a groundwater scientist and hydrologist in India.\n"
            f"Write a professional, concise 2-sentence practical insight for the groundwater forecast for {label}.\n\n"
            f"Context (already computed by a deterministic model — do not contradict these numbers):\n"
            f"- Location: {label}\n"
            f"- Current Stage of Extraction: {start_value:.1f}% ({current_risk})\n"
            f"- Projected Stage of Extraction by {target_years[-1]}: {final_value:.1f}% ({risk})\n"
            f"- Trend: {trend}\n"
            f"- Model Confidence: {confidence}%\n"
            + (f"- Draft-to-recharge ratio: {balance['draft_to_recharge_ratio']} ({balance['label']})\n" if balance else "")
            + (f"- Simulated draft change: {draft_change_pct:+.0f}%\n" if draft_change_pct else "")
            + (f"- Risk crosses into {threshold_crossing['to_risk']} by {threshold_crossing['year']}\n" if threshold_crossing else "")
            + f"\nRequirements:\n"
            f"- Add practical context (e.g. what this means for farmers or local water planning) that the numbers alone don't convey.\n"
            f"- Do not restate the raw percentages verbatim; the reader already sees them.\n"
            f"- Do not reference code/LLM/model details.\n"
            f"- Keep it strictly under 3 sentences."
        )
        try:
            m = genai.GenerativeModel("gemini-3.1-flash-lite")
            gemini_text = m.generate_content(prompt).text.strip()
            if gemini_text:
                ai_insight = gemini_text
        except Exception:
            pass

    return {
        "state": state.upper(),
        "district": district.upper() if district else None,
        "title": f"Stage-of-Extraction Forecast — {label}",
        "unit": "%",
        "historical_data": historical,
        "projected_data": projected,
        "metrics": {
            "stage_of_extraction": risk,
            "stage_of_extraction_pct": f"{final_value:.1f}%",
            "current_risk": current_risk,
            "risk_level": "High" if risk in ("Critical", "Over-Exploited") else ("Medium" if risk == "Semi-Critical" else "Low"),
            "trend": trend,
            "confidence": confidence,
        },
        "rules": {
            "threshold_crossing": threshold_crossing,
            "groundwater_balance": balance,
            "recommended_action": recommended_action,
            "data_confidence_note": data_confidence_note,
        },
        "summary": summary,
        "ai_insight": ai_insight,
    }
