"""
field_observations_service.py — crowdsourced well-depth readings.

Stored as a distinct Neo4j node type (FieldObservation) linked to District/State,
kept clearly separate from official CGWB data. No auth system — a lightweight
per-IP rate limit is the only defense against spam, matching the "no full auth
unless strictly required" guidance for this project.
"""

import uuid
import time
from typing import Optional, List, Dict, Any
from core.graphrag import driver


def submit_observation(
    state: str,
    district: Optional[str],
    well_depth_m: float,
    reporter_note: Optional[str],
    submitted_by_ip: str,
) -> Dict[str, Any]:
    obs_id = str(uuid.uuid4())
    ts = time.time()
    try:
        with driver.session() as session:
            session.run(
                """
                MERGE (s:State {name: $state})
                MERGE (obs:FieldObservation {id: $id})
                SET obs.well_depth_m = $well_depth_m,
                    obs.note = $note,
                    obs.submitted_at = $ts,
                    obs.source = "community-reported",
                    obs.verified = false,
                    obs.district = $district
                MERGE (obs)-[:REPORTED_FOR]->(s)
                """,
                state=state.upper(),
                id=obs_id,
                well_depth_m=well_depth_m,
                note=reporter_note or "",
                ts=ts,
                district=district.upper() if district else None,
            )
    except Exception as e:
        raise RuntimeError(f"Graph store unavailable, could not save observation: {e}")
    return {"id": obs_id, "status": "submitted", "verified": False}


def list_observations(state: Optional[str] = None, district: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    filters = []
    params: Dict[str, Any] = {"limit": limit}
    if state:
        filters.append("s.name = $state")
        params["state"] = state.upper()
    if district:
        filters.append("obs.district = $district")
        params["district"] = district.upper()
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"""
        MATCH (obs:FieldObservation)-[:REPORTED_FOR]->(s:State)
        {where}
        RETURN obs.id AS id, s.name AS state, obs.district AS district,
               obs.well_depth_m AS well_depth_m, obs.note AS note,
               obs.submitted_at AS submitted_at, obs.verified AS verified,
               obs.source AS source
        ORDER BY obs.submitted_at DESC
        LIMIT $limit
    """
    try:
        with driver.session() as session:
            return [dict(r) for r in session.run(query, **params)]
    except Exception:
        # Graph store unreachable (e.g. paused Aura instance) — degrade to empty rather than hang/500.
        return []
