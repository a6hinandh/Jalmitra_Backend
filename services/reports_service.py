"""
reports_service.py — PDF report generation.

Assembles a narrative (via Gemini) + data table into a downloadable PDF for a
state/district/year-range, reusing the same data layer as CSV export.
"""

import io
import csv
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from core.graphrag import run_cypher, embed_model, pine_index

METRICS = ["rainfall", "recharge", "draft", "availability"]
METRIC_REL = {
    "rainfall": ("RAINFALL", "Rainfall"),
    "recharge": ("RECHARGE", "Recharge"),
    "draft": ("DRAFT", "Draft"),
    "availability": ("AVAILABILITY", "Availability"),
}

# Every node category the Neo4j graph stores per state/district/year — the full
# structured dataset behind Jalmitra, not just the 4 metrics shown on the map/compare pages.
DATA_CATEGORIES: Dict[str, Dict[str, str]] = {
    "rainfall":             {"label": "Rainfall",                    "rel": "RAINFALL",            "node": "Rainfall"},
    "recharge":              {"label": "Groundwater Recharge",        "rel": "RECHARGE",             "node": "Recharge"},
    "draft":                 {"label": "Groundwater Draft",           "rel": "DRAFT",                "node": "Draft"},
    "availability":          {"label": "Water Availability",          "rel": "AVAILABILITY",         "node": "Availability"},
    "groundwater":           {"label": "Groundwater Resources",       "rel": "GROUND_WATER",         "node": "GroundWaterAvailability"},
    "stage_of_extraction":   {"label": "Stage of Extraction",         "rel": "STAGE",                "node": "StageOfExtraction"},
    "future_use":            {"label": "Future Use Allocation",       "rel": "FUTURE_USE",           "node": "FutureUse"},
    "allocation":            {"label": "Allocation",                  "rel": "ALLOCATION",           "node": "Allocation"},
    "aquifer":               {"label": "Aquifer",                     "rel": "AQUIFER",              "node": "Aquifer"},
    "area":                  {"label": "Area",                        "rel": "AREA",                 "node": "Area"},
    "loss":                  {"label": "Loss",                        "rel": "LOSS",                 "node": "Loss"},
    "block_summary":         {"label": "Block Summary",               "rel": "BLOCK_SUMMARY",         "node": "BlockSummary"},
    "additional_recharge":   {"label": "Additional Recharge",         "rel": "ADDITIONAL_RECHARGE",   "node": "AdditionalRecharge"},
}


def fetch_category_rows(state: str, district: Optional[str], years: List[int], rel: str, node: str) -> List[Dict[str, Any]]:
    """Fetch every property of a given node category, per year, for a state or district."""
    years_str = ",".join(map(str, years))
    if district:
        q = (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
            f'-[:HAS_DISTRICT]->(d:District {{name:"{district.upper()}"}})'
            f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel}]->(n:{node}) '
            f'WHERE y.year IN [{years_str}] RETURN y.year AS year, properties(n) AS props ORDER BY y.year'
        )
    else:
        q = (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
            f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel}]->(n:{node}) '
            f'WHERE y.year IN [{years_str}] RETURN y.year AS year, properties(n) AS props ORDER BY y.year'
        )
    try:
        rows = run_cypher(q)
    except Exception:
        rows = []
    out = []
    for r in rows:
        props = dict(r.get("props") or {})
        props.pop("uuid", None)
        out.append({"year": r.get("year"), **props})
    return out


def fetch_full_dataset(state: str, district: Optional[str], years: List[int], categories: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    dataset = {}
    for key in categories:
        meta = DATA_CATEGORIES.get(key)
        if not meta:
            continue
        dataset[key] = fetch_category_rows(state, district, years, meta["rel"], meta["node"])
    return dataset


def dataset_to_csv(dataset: Dict[str, List[Dict[str, Any]]]) -> str:
    """Long-format CSV (category, year, field, value) so categories with different
    property sets can all live in a single, spreadsheet-friendly file."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["category", "year", "field", "value"])
    for category, rows in dataset.items():
        for row in rows:
            year = row.get("year")
            for field, value in row.items():
                if field == "year":
                    continue
                writer.writerow([category, year, field, value])
    return output.getvalue()


def export_pinecone_records(state: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Pull the semantic-search records (text + metadata) indexed for a state.
    Uses a metadata filter so we get this state's own record(s), not just whatever
    ranks highest by embedding similarity."""
    try:
        vec = embed_model.encode([f"Groundwater report for {state.title()}"])[0].tolist()
        res = pine_index.query(
            vector=vec, top_k=top_k, include_metadata=True,
            filter={"State": {"$eq": state.title()}},
        )
        matches = res.matches
        if not matches:
            # Fallback: some indexes store the state name differently (e.g. upper case).
            res = pine_index.query(vector=vec, top_k=top_k, include_metadata=True)
            matches = [m for m in res.matches if str(m.metadata.get("State", "")).upper() == state.upper()]
        return [{"id": m.id, "score": round(float(m.score), 4), **dict(m.metadata or {})} for m in matches]
    except Exception:
        return []


def pinecone_records_to_csv(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "id,score\n"
    fieldnames = sorted({k for r in records for k in r.keys()})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def _fetch_metric_rows(state: str, district: Optional[str], years: List[int]):
    rows = {}
    for metric, (rel, node) in METRIC_REL.items():
        if district:
            q = (
                f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
                f'-[:HAS_DISTRICT]->(d:District {{name:"{district.upper()}"}})'
                f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel}]->(n:{node}) '
                f'WHERE y.year IN [{",".join(map(str, years))}] AND n.total IS NOT NULL '
                f'RETURN y.year AS year, n.total AS value ORDER BY y.year'
            )
        else:
            q = (
                f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
                f'-[:HAS_YEAR]->(y:Year)-[:HAS_{rel}]->(n:{node}) '
                f'WHERE y.year IN [{",".join(map(str, years))}] AND n.total IS NOT NULL '
                f'RETURN y.year AS year, n.total AS value ORDER BY y.year'
            )
        try:
            rows[metric] = run_cypher(q)
        except Exception:
            rows[metric] = []
    return rows


def _narrative(state: str, district: Optional[str], years: List[int], rows: dict) -> str:
    label = f"{district.title()}, {state.title()}" if district else state.title()
    data_summary = {m: {r["year"]: r["value"] for r in v} for m, v in rows.items()}
    prompt = (
        f"Write a concise (4-6 sentence) groundwater report narrative for {label} covering years {years}. "
        f"Data: {data_summary}. Mention notable trends and any sustainability concerns. "
        f"Plain professional English, no markdown, no headers."
    )
    try:
        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        return model.generate_content(prompt).text.strip()
    except Exception:
        return (
            f"Groundwater data for {label} across {', '.join(map(str, years))} is summarized in the table below. "
            f"Automated narrative generation was unavailable at report time."
        )


def generate_report_pdf(state: str, district: Optional[str] = None, years: Optional[List[int]] = None) -> bytes:
    years = years or [2023, 2024]
    label = f"{district.title()}, {state.title()}" if district else state.title()
    rows = _fetch_metric_rows(state, district, years)
    narrative = _narrative(state, district, years, rows)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#1e3a8a"))

    elements = [
        Paragraph(f"Jalmitra Groundwater Report — {label}", title_style),
        Paragraph(f"Years covered: {', '.join(map(str, years))}", styles["Normal"]),
        Spacer(1, 0.5 * cm),
        Paragraph("Summary", styles["Heading2"]),
        Paragraph(narrative, styles["BodyText"]),
        Spacer(1, 0.5 * cm),
        Paragraph("Data", styles["Heading2"]),
    ]

    table_data = [["Metric", *[str(y) for y in years], "Unit"]]
    units = {"rainfall": "mm", "recharge": "ham", "draft": "ham", "availability": "ham"}
    for metric in METRICS:
        by_year = {r["year"]: r["value"] for r in rows.get(metric, [])}
        table_data.append([metric.title(), *[f"{by_year.get(y, '—')}" for y in years], units[metric]])

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 1 * cm))
    elements.append(Paragraph(
        "Source: CGWB / Ministry of Jal Shakti groundwater assessment data, via Jalmitra.",
        styles["Italic"],
    ))

    doc.build(elements)
    return buf.getvalue()
