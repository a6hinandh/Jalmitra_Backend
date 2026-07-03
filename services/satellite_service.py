"""
satellite_service.py — remote-sensing overlay (stretch goal, per roadmap 2.8).

TODO(real integration): once BHUVAN_API_KEY or GEE_SERVICE_ACCOUNT is available in .env,
replace `_mock_reading` with a real call to Bhuvan (ISRO) WMS/WFS or Google Earth Engine
for NDVI / soil-moisture rasters clipped to the state boundary. Until then this returns
deterministic mock data (seeded per state) so the endpoint/UI contract is stable and
swapping in the real data source later requires no frontend changes.
"""

import os
import hashlib
from typing import Dict, Any

BHUVAN_API_KEY = os.getenv("BHUVAN_API_KEY")
GEE_SERVICE_ACCOUNT = os.getenv("GEE_SERVICE_ACCOUNT")


def _seeded_value(seed: str, low: float, high: float) -> float:
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    return round(low + (h % 1000) / 1000 * (high - low), 2)


def get_satellite_overlay(state: str, year: int = 2024) -> Dict[str, Any]:
    if not (BHUVAN_API_KEY or GEE_SERVICE_ACCOUNT):
        seed = f"{state.upper()}-{year}"
        return {
            "state": state.upper(),
            "year": year,
            "mock_data": True,
            "note": "No BHUVAN_API_KEY / GEE_SERVICE_ACCOUNT configured — showing deterministic mock values. "
                    "Set one of these env vars and implement the real fetch in satellite_service.py.",
            "ndvi": _seeded_value(seed + "-ndvi", 0.1, 0.8),
            "soil_moisture_pct": _seeded_value(seed + "-soil", 5, 45),
        }

    # Real integration point — not yet implemented.
    raise NotImplementedError(
        "BHUVAN_API_KEY/GEE_SERVICE_ACCOUNT is set but the real fetch is not yet implemented "
        "in satellite_service.get_satellite_overlay()."
    )
