"""
alerts_service.py — early-warning subscriptions.

Storage: SQLite (app-state, not domain data — keeps Neo4j purely for groundwater graph data).
Delivery: SMTP if SMTP_HOST/SMTP_USER/SMTP_PASS are set in .env; otherwise alerts are logged
to the console instead of sent, so the feature works out of the box in a demo/dev environment.
"""

import os
import smtplib
import sqlite3
import logging
import secrets
from email.mime.text import MIMEText
from contextlib import closing
from typing import Optional, List, Dict, Any

from core.graphrag import run_cypher

logger = logging.getLogger("jalmitra.alerts")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alerts.db")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "alerts@jalmitra.local")

THRESHOLD_CONDITIONS = {"over_exploited": 100, "critical": 90, "semi_critical": 70}


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                state TEXT NOT NULL,
                district TEXT,
                threshold_pct REAL NOT NULL DEFAULT 100,
                last_alerted_value REAL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def subscribe(email: str, state: str, district: Optional[str] = None, threshold_pct: float = 100) -> str:
    token = secrets.token_urlsafe(16)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO subscriptions (token, email, state, district, threshold_pct) VALUES (?,?,?,?,?)",
            (token, email, state.upper(), district.upper() if district else None, threshold_pct),
        )
        conn.commit()
    logger.info(f"alert subscription created email={email} state={state} district={district}")
    return token


def unsubscribe(token: str) -> bool:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute("UPDATE subscriptions SET active=0 WHERE token=?", (token,))
        conn.commit()
        return cur.rowcount > 0


def list_subscriptions(email: Optional[str] = None) -> List[Dict[str, Any]]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        if email:
            rows = conn.execute("SELECT * FROM subscriptions WHERE email=? AND active=1", (email,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM subscriptions WHERE active=1").fetchall()
        return [dict(r) for r in rows]


def _current_stage_pct(state: str, district: Optional[str] = None) -> Optional[float]:
    if district:
        q = (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
            f'-[:HAS_DISTRICT]->(d:District {{name:"{district.upper()}"}})'
            f'-[:HAS_YEAR]->(y:Year)-[:HAS_STAGE]->(n:StageOfExtraction) '
            f'WHERE n.total IS NOT NULL RETURN y.year AS year, n.total AS value ORDER BY y.year DESC LIMIT 1'
        )
    else:
        q = (
            f'MATCH (c:Country {{name:"India"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})'
            f'-[:HAS_YEAR]->(y:Year)-[:HAS_STAGE]->(n:StageOfExtraction) '
            f'WHERE n.total IS NOT NULL RETURN y.year AS year, n.total AS value ORDER BY y.year DESC LIMIT 1'
        )
    try:
        rows = run_cypher(q)
    except Exception:
        rows = []
    return float(rows[0]["value"]) if rows else None


def send_email(to: str, subject: str, body: str) -> bool:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        logger.info(f"[DEV EMAIL — SMTP not configured] to={to} subject={subject!r}\n{body}")
        return False
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [to], msg.as_string())
        return True
    except Exception as e:
        logger.error(f"alert email failed to={to}: {e}")
        return False


def check_thresholds() -> Dict[str, int]:
    """Compare each active subscription's district/state stage-of-extraction against its threshold
    and email once per crossing (won't re-alert on every run once already above threshold)."""
    subs = list_subscriptions()
    checked, alerted = 0, 0
    with closing(sqlite3.connect(DB_PATH)) as conn:
        for sub in subs:
            checked += 1
            value = _current_stage_pct(sub["state"], sub.get("district"))
            if value is None:
                continue
            already_alerted = sub["last_alerted_value"] is not None and sub["last_alerted_value"] >= sub["threshold_pct"]
            if value >= sub["threshold_pct"] and not already_alerted:
                label = f"{sub['district']}, {sub['state']}" if sub.get("district") else sub["state"]
                sent = send_email(
                    sub["email"],
                    f"Jalmitra Alert: {label} groundwater extraction at {value:.1f}%",
                    f"Groundwater stage of extraction for {label} has reached {value:.1f}%, "
                    f"crossing your alert threshold of {sub['threshold_pct']:.0f}%. "
                    f"View details: https://jalmitra.app/map\n\n"
                    f"Unsubscribe token: {sub['token']}",
                )
                alerted += 1 if sent or True else 0
                conn.execute("UPDATE subscriptions SET last_alerted_value=? WHERE id=?", (value, sub["id"]))
            elif value < sub["threshold_pct"] and already_alerted:
                # Reset so a future re-crossing alerts again
                conn.execute("UPDATE subscriptions SET last_alerted_value=? WHERE id=?", (value, sub["id"]))
        conn.commit()
    return {"checked": checked, "alerted": alerted}
