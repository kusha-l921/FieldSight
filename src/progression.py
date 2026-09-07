"""
Temporal disease progression ledger and growth velocity tracker for FieldSight-Lite.
Provides SQLite-backed longitudinal logging, daily severity delta (ΔS/day) tracking,
and automated epidemiological risk alerts.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.schemas import (
    ProgressionReport,
    ProgressionState,
    RiskAlertLevel,
    SeverityObservation,
)


def parse_iso_utc(ts_str: str) -> datetime:
    """Parses ISO-8601 UTC timestamp string into a timezone-aware datetime."""
    # Normalize trailing Z if needed
    cleaned = ts_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        # Fallback to current UTC time
        return datetime.now(timezone.utc)


class ProgressionLedger:
    """
    Longitudinal SQLite ledger recording multi-day disease progression observations.
    Computes disease progression velocity and categorizes epidemic trajectories.
    """

    def __init__(
        self,
        db_path: str = "data/temporal_db/field_history.db",
        stable_max_pct_per_day: float = 0.5,
        emerging_max_pct_per_day: float = 1.5,
        moderate_max_pct_per_day: float = 3.5
    ):
        self.db_path = db_path
        self.stable_max_pct_per_day = float(stable_max_pct_per_day)
        self.emerging_max_pct_per_day = float(emerging_max_pct_per_day)
        self.moderate_max_pct_per_day = float(moderate_max_pct_per_day)
        self._init_db()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> ProgressionLedger:
        p_cfg = cfg.get("progression", {})
        s_cfg = cfg.get("storage", {})
        return cls(
            db_path=s_cfg.get("sqlite_db_path", "data/temporal_db/field_history.db"),
            stable_max_pct_per_day=p_cfg.get("stable_max_pct_per_day", 0.5),
            emerging_max_pct_per_day=p_cfg.get("emerging_max_pct_per_day", 1.5),
            moderate_max_pct_per_day=p_cfg.get("moderate_max_pct_per_day", 3.5)
        )

    def _init_db(self) -> None:
        """Initializes SQLite database and tables."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS severity_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_id TEXT NOT NULL,
                    leaf_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    severity_pct REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    raw_data_json TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plant_leaf_ts 
                ON severity_observations (plant_id, leaf_id, timestamp_utc)
            """)
            conn.commit()

    def record_observation(
        self,
        plant_id: str,
        leaf_id: str,
        severity_pct: float,
        confidence_score: float,
        timestamp_utc: Optional[str] = None,
        raw_metadata: Optional[Dict[str, Any]] = None
    ) -> ProgressionReport:
        """
        Inserts a new severity observation into the database and returns a progression analysis.
        """
        now_ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw_json = json.dumps(raw_metadata or {})

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO severity_observations 
                (plant_id, leaf_id, timestamp_utc, severity_pct, confidence_score, raw_data_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (plant_id, leaf_id, now_ts, float(severity_pct), float(confidence_score), raw_json))
            conn.commit()

        return self.analyze_progression(plant_id, leaf_id, current_ts=now_ts)

    def get_history(self, plant_id: str, leaf_id: str) -> List[SeverityObservation]:
        """Fetches chronological observation history for a plant/leaf."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT plant_id, leaf_id, timestamp_utc, severity_pct, confidence_score
                FROM severity_observations
                WHERE plant_id = ? AND leaf_id = ?
                ORDER BY timestamp_utc ASC
            """, (plant_id, leaf_id))
            rows = cursor.fetchall()

        return [
            SeverityObservation(
                plant_id=row[0],
                leaf_id=row[1],
                timestamp_utc=row[2],
                severity_pct=float(row[3]),
                confidence_score=float(row[4])
            )
            for row in rows
        ]

    def list_monitored_entities(self) -> List[Tuple[str, str, int]]:
        """Returns list of unique (plant_id, leaf_id, count) tracked in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT plant_id, leaf_id, COUNT(*)
                FROM severity_observations
                GROUP BY plant_id, leaf_id
                ORDER BY plant_id, leaf_id
            """)
            return [(row[0], row[1], int(row[2])) for row in cursor.fetchall()]

    def analyze_progression(
        self,
        plant_id: str,
        leaf_id: str,
        current_ts: Optional[str] = None
    ) -> ProgressionReport:
        """
        Analyzes historical disease growth curve and computes progression velocity.
        """
        history = self.get_history(plant_id, leaf_id)

        if not history:
            now_ts = current_ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return ProgressionReport(
                plant_id=plant_id,
                leaf_id=leaf_id,
                current_timestamp_utc=now_ts,
                previous_timestamp_utc=None,
                delta_days=0.0,
                delta_severity=0.0,
                progression_rate_daily=None,
                state=ProgressionState.INSUFFICIENT_DATA,
                risk_alert_level=RiskAlertLevel.INDETERMINATE,
                observation_history=()
            )

        current_obs = history[-1]
        now_ts = current_obs.timestamp_utc

        if len(history) == 1:
            return ProgressionReport(
                plant_id=plant_id,
                leaf_id=leaf_id,
                current_timestamp_utc=now_ts,
                previous_timestamp_utc=None,
                delta_days=0.0,
                delta_severity=0.0,
                progression_rate_daily=None,
                state=ProgressionState.INSUFFICIENT_DATA,
                risk_alert_level=RiskAlertLevel.INDETERMINATE,
                observation_history=tuple(history)
            )

        prev_obs = history[-2]
        t_curr = parse_iso_utc(current_obs.timestamp_utc)
        t_prev = parse_iso_utc(prev_obs.timestamp_utc)

        delta_seconds = (t_curr - t_prev).total_seconds()
        delta_days = max(0.0, delta_seconds / 86400.0)
        delta_sev = float(current_obs.severity_pct - prev_obs.severity_pct)

        if delta_days <= 0.001:  # Near identical timestamp
            rate_daily: Optional[float] = None
            state = ProgressionState.STABLE
            alert = RiskAlertLevel.LOW
        else:
            rate_daily = float(delta_sev / delta_days)

            # Categorize progression state
            if rate_daily <= self.stable_max_pct_per_day:
                state = ProgressionState.STABLE
                alert = RiskAlertLevel.LOW
            elif rate_daily <= self.emerging_max_pct_per_day:
                state = ProgressionState.EMERGING
                alert = RiskAlertLevel.MEDIUM
            elif rate_daily <= self.moderate_max_pct_per_day:
                state = ProgressionState.MODERATE
                alert = RiskAlertLevel.HIGH
            else:
                state = ProgressionState.RAPID
                alert = RiskAlertLevel.CRITICAL

        return ProgressionReport(
            plant_id=plant_id,
            leaf_id=leaf_id,
            current_timestamp_utc=now_ts,
            previous_timestamp_utc=prev_obs.timestamp_utc,
            delta_days=delta_days,
            delta_severity=delta_sev,
            progression_rate_daily=rate_daily,
            state=state,
            risk_alert_level=alert,
            observation_history=tuple(history)
        )
