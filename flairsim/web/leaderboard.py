"""
SQLite-backed leaderboard for FlairSim benchmark runs.

Stores completed run results with scenario metadata, player/model info,
success status, and trajectory data.  Provides query methods for
ranking and filtering.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT,
    scenario_id     TEXT NOT NULL,
    player_name     TEXT,
    model_name      TEXT,
    mode            TEXT NOT NULL CHECK (mode IN ('human', 'ai')),
    success         INTEGER NOT NULL DEFAULT 0,
    reason          TEXT,
    steps_taken     INTEGER,
    distance_travelled REAL,
    duration_s      REAL,
    trajectory      TEXT,
    steps_detail    TEXT,
    model_info      TEXT,
    metrics         TEXT,
    score_final     REAL,
    created_at      TEXT NOT NULL
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_runs_scenario ON runs (scenario_id);
"""


class Leaderboard:
    """Persistent leaderboard backed by a SQLite database.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.  Created automatically if
        it does not exist (including parent directories).
    """

    def __init__(self, db_path: str | Path = "data/leaderboard.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._migrate()
        self._conn.commit()
        logger.info("Leaderboard database: %s", self._db_path)

    # ---------------------------------------------------------------- migrate

    def _migrate(self) -> None:
        """Add columns that may be missing in older databases."""
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        migrations = [
            ("steps_detail", "TEXT"),
            ("model_info", "TEXT"),
            ("metrics", "TEXT"),
            ("score_final", "REAL"),
        ]
        for col, dtype in migrations:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {dtype}")
                logger.info("Migrated leaderboard: added column '%s'", col)

    # ---------------------------------------------------------------- write

    def submit_run(self, run_data: Dict[str, Any]) -> str:
        """Record a completed run.

        Parameters
        ----------
        run_data : dict
            Must include ``scenario_id`` and ``mode``.  Optional keys:
            ``session_id``, ``player_name``, ``model_name``, ``success``,
            ``reason``, ``steps_taken``, ``distance_travelled``,
            ``duration_s``, ``trajectory`` (list of dicts),
            ``steps_detail`` (list of dicts), ``model_info`` (dict),
            ``metrics`` (dict), ``score_final`` (float or None).

        Returns
        -------
        str
            The generated run ID (UUID).
        """
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        trajectory = run_data.get("trajectory")
        if trajectory is not None and not isinstance(trajectory, str):
            trajectory = json.dumps(trajectory)

        steps_detail = run_data.get("steps_detail")
        if steps_detail is not None and not isinstance(steps_detail, str):
            steps_detail = json.dumps(steps_detail)

        model_info = run_data.get("model_info")
        if model_info is not None and not isinstance(model_info, str):
            model_info = json.dumps(model_info)

        metrics = run_data.get("metrics")
        if metrics is not None and not isinstance(metrics, str):
            metrics = json.dumps(metrics)

        self._conn.execute(
            """\
            INSERT INTO runs
                (id, session_id, scenario_id, player_name, model_name,
                 mode, success, reason, steps_taken, distance_travelled,
                 duration_s, trajectory, steps_detail, model_info,
                 metrics, score_final, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_data.get("session_id"),
                run_data["scenario_id"],
                run_data.get("player_name"),
                run_data.get("model_name"),
                run_data["mode"],
                int(run_data.get("success", False)),
                run_data.get("reason"),
                run_data.get("steps_taken"),
                run_data.get("distance_travelled"),
                run_data.get("duration_s"),
                trajectory,
                steps_detail,
                model_info,
                metrics,
                run_data.get("score_final"),
                now,
            ),
        )
        self._conn.commit()
        logger.info(
            "Leaderboard: recorded run %s for scenario %s",
            run_id,
            run_data["scenario_id"],
        )
        return run_id

    # ---------------------------------------------------------------- read

    def get_runs(
        self,
        scenario_id: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query runs, ordered by success DESC, steps ASC, distance ASC.

        Parameters
        ----------
        scenario_id : str or None
            Filter by scenario.
        mode : str or None
            Filter by mode (``"human"`` or ``"ai"``).
        limit : int
            Maximum number of results.

        Returns
        -------
        list of dict
        """
        clauses: list[str] = []
        params: list[Any] = []

        if scenario_id is not None:
            clauses.append("scenario_id = ?")
            params.append(scenario_id)
        if mode is not None:
            clauses.append("mode = ?")
            params.append(mode)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        query = f"""\
            SELECT * FROM runs
            {where}
            ORDER BY success DESC, steps_taken ASC, distance_travelled ASC
            LIMIT ?
        """
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single run by ID."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    # ---------------------------------------------------------------- delete

    def delete_run(self, run_id: str) -> bool:
        """Delete a single run by ID.

        Returns
        -------
        bool
            ``True`` if a row was deleted, ``False`` if the ID was not found.
        """
        cursor = self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Leaderboard: deleted run %s", run_id)
        return deleted

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["success"] = bool(d["success"])
        # Deserialise JSON columns.
        for col in ("trajectory", "steps_detail", "model_info", "metrics"):
            val = d.get(col)
            if val and isinstance(val, str):
                try:
                    d[col] = json.loads(val)
                except json.JSONDecodeError:
                    pass
        return d

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
