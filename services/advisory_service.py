"""
advisory_service.py — personalized farmer sowing/irrigation recommendations.

Combines groundwater stage-of-extraction + rainfall (from the graph) with static
crop water-requirement reference tables (public ICAR/CWC figures) into a structured
recommendation. This is a rule-based advisor, not an ML model — the value is in
combining two data sources a farmer wouldn't otherwise cross-reference themselves.
"""

from typing import Optional, Dict, Any
from core.graphrag import run_cypher
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

# Total crop-water requirement per season, mm (indicative ICAR/CWC ranges), and
# irrigation dependency (share of that requirement typically NOT met by rainfall alone).
CROP_WATER_REQUIREMENTS = {
    "rice":       {"season": "kharif", "requirement_mm": 1200, "irrigation_dependency": 0.6,  "sowing_window": "June – July",       "duration_days": 120, "water_efficient_alt": "Direct-seeded rice (aerobic)"},
    "wheat":      {"season": "rabi",   "requirement_mm": 450,  "irrigation_dependency": 0.75, "sowing_window": "Nov – Dec",         "duration_days": 130, "water_efficient_alt": "Barley"},
    "sugarcane":  {"season": "year-round", "requirement_mm": 1800, "irrigation_dependency": 0.7, "sowing_window": "Feb – Mar / Oct", "duration_days": 330, "water_efficient_alt": "Sweet sorghum"},
    "cotton":     {"season": "kharif", "requirement_mm": 700,  "irrigation_dependency": 0.5,  "sowing_window": "Apr – May",         "duration_days": 180, "water_efficient_alt": "Bt cotton with drip"},
    "maize":      {"season": "kharif", "requirement_mm": 500,  "irrigation_dependency": 0.45, "sowing_window": "Jun – Jul",         "duration_days": 100, "water_efficient_alt": "Pearl millet (bajra)"},
    "groundnut":  {"season": "kharif", "requirement_mm": 500,  "irrigation_dependency": 0.4,  "sowing_window": "Jun – Jul",         "duration_days": 110, "water_efficient_alt": "Moong (green gram)"},
    "pulses":     {"season": "rabi",   "requirement_mm": 350,  "irrigation_dependency": 0.35, "sowing_window": "Oct – Nov",         "duration_days": 90,  "water_efficient_alt": "Chickpea (already low-water)"},
    "vegetables": {"season": "year-round", "requirement_mm": 600, "irrigation_dependency": 0.65, "sowing_window": "Year-round (staggered)", "duration_days": 75, "water_efficient_alt": "Okra/beans with mulching"},
}

# Rule-based water-saving techniques, ranked by relevance to risk level.
WATER_SAVING_TECHNIQUES = {
    "low": [
        "Continue rainfed/conjunctive use; monitor well levels monthly.",
        "Adopt basic mulching to reduce evaporation losses.",
    ],
    "moderate": [
        "Switch to drip or sprinkler irrigation for a 30-50% water saving over flood irrigation.",
        "Use soil-moisture sensors or the 'tensiometer' method to irrigate only when needed.",
        "Apply mulching to cut evaporation losses by up to 25%.",
    ],
    "high": [
        "Prioritize drip irrigation — mandatory for water-intensive crops in this zone.",
        "Consider laser land leveling to reduce water wastage by 25-30%.",
        "Stagger sowing dates to align with expected rainfall and reduce peak irrigation demand.",
        "Join or form a Water User Association for coordinated, metered extraction.",
    ],
    "severe": [
        "Switch to a drought-tolerant alternate crop this season if feasible.",
        "Mandatory drip/micro-irrigation — flood irrigation is not sustainable here.",
        "Apply for government-subsidized recharge structures (check PMKSY schemes).",
        "Delay sowing until post-monsoon recharge is confirmed via local well readings.",
    ],
    "unknown": [
        "Submit a field observation for this location to improve future recommendations.",
    ],
}


def _latest(state: str, district: Optional[str], rel: str, node: str, prop: str = "total") -> Optional[float]:
    if district:
        q = (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
            f'-[:HAS_DISTRICT]->(d:District {{name:"{district.upper()}"}})'
            f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel}]->(n:{node}) '
            f'WHERE n.{prop} IS NOT NULL RETURN n.{prop} AS value ORDER BY y.year DESC LIMIT 1'
        )
    else:
        q = (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
            f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel}]->(n:{node}) '
            f'WHERE n.{prop} IS NOT NULL RETURN n.{prop} AS value ORDER BY y.year DESC LIMIT 1'
        )
    try:
        rows = run_cypher(q)
    except Exception:
        rows = []
    return float(rows[0]["value"]) if rows else None


def get_advisory(state: str, crop: str, district: Optional[str] = None) -> Dict[str, Any]:
    crop_key = crop.lower().strip()
    crop_info = CROP_WATER_REQUIREMENTS.get(crop_key)
    if not crop_info:
        return {
            "error": f"Unknown crop '{crop}'. Supported: {', '.join(CROP_WATER_REQUIREMENTS)}",
        }

    rainfall = _latest(state, district, "RAINFALL", "Rainfall")
    stage = _latest(state, district, "STAGE", "StageOfExtraction")
    if stage is None:
        draft = _latest(state, district, "DRAFT", "Draft")
        avail = _latest(state, district, "AVAILABILITY", "Availability")
        stage = round((draft / avail) * 100, 1) if draft and avail else None

    rainfall = rainfall or 0
    requirement = crop_info["requirement_mm"]
    rainfall_coverage = min(rainfall / requirement, 1.0) if requirement else 0
    irrigation_needed_mm = max(requirement - rainfall, 0)

    # Confidence that groundwater irrigation can safely cover the gap
    if stage is None:
        water_confidence = 50
        risk_flag = "unknown"
    elif stage < 70:
        water_confidence = 85
        risk_flag = "low"
    elif stage < 90:
        water_confidence = 65
        risk_flag = "moderate"
    elif stage < 100:
        water_confidence = 40
        risk_flag = "high"
    else:
        water_confidence = 20
        risk_flag = "severe"

    label = f"{district.title()}, {state.title()}" if district else state.title()

    # Rule: deterministic action — always present, never depends on the LLM.
    if risk_flag in ("high", "severe"):
        action = (
            f"Groundwater in this area is already {'critical' if risk_flag == 'high' else 'over-exploited'}. "
            f"Consider a less water-intensive crop, drip/sprinkler irrigation, or delaying sowing until "
            f"post-monsoon recharge improves availability."
        )
    elif rainfall_coverage >= 0.9:
        action = f"Rainfall alone is expected to cover most of {crop_key}'s water needs — sow as per normal {crop_info['season']} timing with minimal supplemental irrigation."
    else:
        action = (
            f"Plan for supplemental irrigation of roughly {irrigation_needed_mm:.0f}mm beyond rainfall. "
            f"Groundwater conditions here currently support this at {risk_flag} risk."
        )

    # Rule: water-saving technique checklist, ranked by risk level.
    techniques = WATER_SAVING_TECHNIQUES.get(risk_flag, WATER_SAVING_TECHNIQUES["unknown"])

    # Rule: sowing window guidance — pull forward if groundwater is stressed and rain is short.
    sowing_window = crop_info["sowing_window"]
    sowing_note = (
        "Consider delaying sowing until post-monsoon recharge is confirmed."
        if risk_flag in ("high", "severe") and rainfall_coverage < 0.5
        else f"Standard window for {crop_key}: {sowing_window}."
    )

    # Rule: suggest a lower-water alternate crop when groundwater risk is elevated.
    alternate_crop = crop_info.get("water_efficient_alt") if risk_flag in ("high", "severe") else None

    # Rule: irrigation frequency suggestion from crop duration + irrigation dependency.
    if crop_info["irrigation_dependency"] >= 0.6:
        irrigation_frequency = "Every 5-7 days (high dependency crop)"
    elif crop_info["irrigation_dependency"] >= 0.4:
        irrigation_frequency = "Every 10-12 days (moderate dependency crop)"
    else:
        irrigation_frequency = "Every 15+ days / as needed (low dependency crop)"

    # Gemini adds a supplementary, farmer-friendly narrative — it never replaces the
    # rule-based action/techniques above, and the page works identically without it.
    ai_insight = None
    if GENAI_API_KEY:
        prompt = (
            f"You are an agricultural scientist and groundwater expert in India.\n"
            f"Write a short, encouraging, practical note (max 3 sentences) for a farmer in {label} planting {crop_key}.\n"
            f"Do not restate the numbers below verbatim — the farmer already sees them. Add context they wouldn't otherwise know.\n\n"
            f"Context (already computed — do not contradict):\n"
            f"- Crop: {crop_key} ({crop_info['season']} season, sowing window {sowing_window})\n"
            f"- Total Crop Water Requirement: {requirement} mm\n"
            f"- Recent Rainfall: {rainfall:.1f} mm (covers {rainfall_coverage * 100:.1f}% of requirement)\n"
            f"- Additional Irrigation Needed: {irrigation_needed_mm:.1f} mm\n"
            f"- Local Groundwater Stage of Extraction: {stage if stage is not None else 'unknown'}% (classification: {risk_flag})\n"
            + (f"- Suggested lower-water alternate crop: {alternate_crop}\n" if alternate_crop else "")
            + f"\nRequirements:\n"
            f"- Use simple, warm, clear language suitable for a farmer.\n"
            f"- If risk is high/severe, reinforce urgency without being alarmist.\n"
            f"- Do not reference code/LLM/model details.\n"
        )
        try:
            model = genai.GenerativeModel("gemini-3.1-flash-lite")
            gemini_text = model.generate_content(prompt).text.strip()
            if gemini_text:
                ai_insight = gemini_text
        except Exception:
            pass

    return {
        "location": label,
        "crop": crop_key,
        "season": crop_info["season"],
        "sowing_window": sowing_window,
        "sowing_note": sowing_note,
        "duration_days": crop_info["duration_days"],
        "irrigation_frequency": irrigation_frequency,
        "alternate_crop": alternate_crop,
        "water_saving_techniques": techniques,
        "ai_insight": ai_insight,
        "water_requirement_mm": requirement,
        "recent_rainfall_mm": round(rainfall, 1),
        "rainfall_coverage_pct": round(rainfall_coverage * 100, 1),
        "irrigation_needed_mm": round(irrigation_needed_mm, 1),
        "stage_of_extraction_pct": round(stage, 1) if stage is not None else None,
        "water_availability_confidence": water_confidence,
        "risk_flag": risk_flag,
        "recommended_action": action,
    }
