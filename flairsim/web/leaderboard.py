"""
SQLite-backed leaderboard for FlairSim benchmark runs.

Stores completed run results with scenario metadata, player/model info,
success status, and trajectory data.  Also manages agent profiles.
Provides query methods for ranking, scoring, and filtering.
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
    confidence      REAL,
    fov_coverage    REAL,
    target_seen     INTEGER,
    created_at      TEXT NOT NULL
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_runs_scenario ON runs (scenario_id);
"""

_CREATE_AGENTS_TABLE = """\
CREATE TABLE IF NOT EXISTS agents (
    name        TEXT PRIMARY KEY,
    specs       TEXT,
    created_at  TEXT NOT NULL
);
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
        self._conn.execute(_CREATE_AGENTS_TABLE)
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
            ("confidence", "REAL"),
            ("fov_coverage", "REAL"),
            ("target_seen", "INTEGER"),
        ]
        for col, dtype in migrations:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {dtype}")
                logger.info("Migrated leaderboard: added column '%s'", col)

        tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "agents" not in tables:
            self._conn.execute(_CREATE_AGENTS_TABLE)
            logger.info("Migrated leaderboard: created 'agents' table")

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
            ``metrics`` (dict), ``score_final`` (float or None),
            ``confidence`` (float or None), ``fov_coverage`` (float or None),
            ``target_seen`` (bool or None).

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
                 metrics, score_final, confidence, fov_coverage, target_seen, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                run_data.get("confidence"),
                run_data.get("fov_coverage"),
                int(run_data["target_seen"])
                if run_data.get("target_seen") is not None
                else None,
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

    # ---------------------------------------------------------------- agents

    def create_agent(self, name: str, specs: Optional[Dict[str, Any]]) -> None:
        """Create a new agent profile.

        Raises
        ------
        ValueError
            If an agent with this name already exists.
        """
        now = datetime.now(timezone.utc).isoformat()
        specs_json = json.dumps(specs) if specs is not None else None
        try:
            self._conn.execute(
                "INSERT INTO agents (name, specs, created_at) VALUES (?, ?, ?)",
                (name, specs_json, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Agent '{name}' already exists")
        logger.info("Created agent profile: %s", name)

    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch a single agent profile by name."""
        row = self._conn.execute(
            "SELECT * FROM agents WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("specs") and isinstance(d["specs"], str):
            try:
                d["specs"] = json.loads(d["specs"])
            except json.JSONDecodeError:
                pass
        return d

    def update_agent(self, name: str, specs: Optional[Dict[str, Any]]) -> None:
        """Update specs for an existing agent.

        Raises
        ------
        KeyError
            If the agent does not exist.
        """
        specs_json = json.dumps(specs) if specs is not None else None
        cursor = self._conn.execute(
            "UPDATE agents SET specs = ? WHERE name = ?",
            (specs_json, name),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"Agent '{name}' not found")
        logger.info("Updated agent profile: %s", name)

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
        if d.get("target_seen") is not None:
            d["target_seen"] = bool(d["target_seen"])
        return d

    # ---------------------------------------------------------------- scoring

    def _fetch_reference_mins(self, scenario_id: str) -> Dict[str, Optional[float]]:
        """Query MIN aggregates for successful runs on a given scenario.

        Returns a dict with keys ``dist``, ``steps``, ``time``, each
        holding the minimum value across ALL successful runs (any mode),
        or ``None`` if no eligible rows exist.
        """
        row = self._conn.execute(
            """\
            SELECT
                MIN(distance_travelled),
                MIN(steps_taken),
                MIN(duration_s)
            FROM runs
            WHERE scenario_id = ? AND success = 1
            """,
            (scenario_id,),
        ).fetchone()
        return {
            "dist": row[0],
            "steps": row[1],
            "time": row[2],
        }

    @staticmethod
    def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> float:
        """Return numerator/denominator, or 1.0 when either value is missing/zero.

        The fallback of 1.0 means "neutral" — no bonus, no penalty.
        """
        if numerator is None or denominator is None or denominator == 0:
            return 1.0
        return numerator / denominator

    def compute_score_for_run(self, run: Dict[str, Any], scenario_id: str) -> float:
        """Compute a normalised score for a single completed run.

        For a **successful** run the score S ∈ [0, 100] is:

            S = [0.3·(D_min/D_agent)
               + 0.3·(Step_min/Step_agent)
               + 0.3·(t_min/t_agent)
               + 0.1·c] × 100

        where D_min, Step_min, t_min are the per-scenario minimums drawn
        from the database and c is the run's confidence (0 if absent).
        Missing or zero agent values default the corresponding ratio to 1.0.

        For a **failed** run the score F ∈ [-100, 0] is:

            F = -100 × [0.5·(1 - E) + 0.5·c]

        where E = fov_coverage (0 if absent) and c = confidence (0 if absent).
        If target_seen is True the result is multiplied by 1.5 before clamping.

        Parameters
        ----------
        run : dict
            A run dict as returned by :meth:`get_run` or :meth:`_row_to_dict`.
        scenario_id : str
            The scenario the run belongs to (used to query reference mins).

        Returns
        -------
        float
            Score in [0, 100] for success, or [-100, 0] for failure.
        """
        if run.get("success"):
            return self._score_success(run, scenario_id)
        return self._score_failure(run)

    def _compute_score_with_mins(
        self, run: Dict[str, Any], mins: Dict[str, Any]
    ) -> float:
        """Like :meth:`compute_score_for_run` but uses pre-loaded reference mins.

        Avoids a DB round-trip per run when mins are already known.
        """
        if run.get("success"):
            return self._score_success_with_mins(run, mins)
        return self._score_failure(run)

    def _score_success(self, run: Dict[str, Any], scenario_id: str) -> float:
        """Compute success score S ∈ [0, 100]."""
        mins = self._fetch_reference_mins(scenario_id)
        return self._score_success_with_mins(run, mins)

    def _score_success_with_mins(
        self, run: Dict[str, Any], mins: Dict[str, Any]
    ) -> float:
        """Compute success score S ∈ [0, 100] given pre-loaded reference mins."""
        # Weights: distance 30 %, steps 30 %, time 30 %, confidence 10 %
        WEIGHT_DIST = 0.3
        WEIGHT_STEPS = 0.3
        WEIGHT_TIME = 0.3
        WEIGHT_CONF = 0.1

        ratio_dist = self._safe_ratio(mins["dist"], run.get("distance_travelled"))
        ratio_steps = self._safe_ratio(mins["steps"], run.get("steps_taken"))
        ratio_time = self._safe_ratio(mins["time"], run.get("duration_s"))
        conf = run.get("confidence")
        confidence = conf if conf is not None else 0.0

        raw = (
            WEIGHT_DIST * ratio_dist
            + WEIGHT_STEPS * ratio_steps
            + WEIGHT_TIME * ratio_time
            + WEIGHT_CONF * confidence
        ) * 100.0

        return max(0.0, min(100.0, raw))

    def _score_failure(self, run: Dict[str, Any]) -> float:
        """Compute failure score F ∈ [-100, 0]."""
        # E = exploration (fov_coverage), c = confidence
        WEIGHT_EXPLORE = 0.5
        WEIGHT_CONF = 0.5
        TARGET_SEEN_MULTIPLIER = 1.5

        fov = run.get("fov_coverage")
        exploration = fov if fov is not None else 0.0
        conf = run.get("confidence")
        confidence = conf if conf is not None else 0.0

        raw = -100.0 * (WEIGHT_EXPLORE * (1.0 - exploration) + WEIGHT_CONF * confidence)

        if run.get("target_seen"):
            raw *= TARGET_SEEN_MULTIPLIER

        return max(-100.0, min(0.0, raw))

    def compute_score_breakdown(
        self, run: Dict[str, Any], scenario_id: str
    ) -> Dict[str, Any]:
        """Compute a detailed score breakdown for a single run.

        Returns a dict containing:
        - ``total``: final score (same as :meth:`compute_score_for_run`)
        - ``success``: whether the run was successful
        - ``components``: list of dicts with ``name``, ``value``, ``weight``,
          ``contribution`` for each scoring component
        - ``reference_mins``: the per-scenario minimums used for ratios
          (only for successful runs)

        Parameters
        ----------
        run : dict
            A run dict as returned by :meth:`get_run`.
        scenario_id : str
            The scenario the run belongs to.

        Returns
        -------
        dict
        """
        is_success = bool(run.get("success"))

        if is_success:
            return self._breakdown_success(run, scenario_id)
        return self._breakdown_failure(run)

    def _breakdown_success(
        self, run: Dict[str, Any], scenario_id: str
    ) -> Dict[str, Any]:
        """Build detailed breakdown for a successful run."""
        mins = self._fetch_reference_mins(scenario_id)

        WEIGHT_DIST = 0.3
        WEIGHT_STEPS = 0.3
        WEIGHT_TIME = 0.3
        WEIGHT_CONF = 0.1

        ratio_dist = self._safe_ratio(mins["dist"], run.get("distance_travelled"))
        ratio_steps = self._safe_ratio(mins["steps"], run.get("steps_taken"))
        ratio_time = self._safe_ratio(mins["time"], run.get("duration_s"))
        conf = run.get("confidence")
        confidence = conf if conf is not None else 0.0

        components = [
            {
                "name": "distance",
                "ratio": round(ratio_dist, 4),
                "weight": WEIGHT_DIST,
                "contribution": round(WEIGHT_DIST * ratio_dist * 100.0, 2),
            },
            {
                "name": "steps",
                "ratio": round(ratio_steps, 4),
                "weight": WEIGHT_STEPS,
                "contribution": round(WEIGHT_STEPS * ratio_steps * 100.0, 2),
            },
            {
                "name": "time",
                "ratio": round(ratio_time, 4),
                "weight": WEIGHT_TIME,
                "contribution": round(WEIGHT_TIME * ratio_time * 100.0, 2),
            },
            {
                "name": "confidence",
                "value": round(confidence, 4),
                "weight": WEIGHT_CONF,
                "contribution": round(WEIGHT_CONF * confidence * 100.0, 2),
            },
        ]

        total = self._score_success_with_mins(run, mins)

        return {
            "total": round(total, 2),
            "success": True,
            "components": components,
            "reference_mins": {
                "distance": mins["dist"],
                "steps": mins["steps"],
                "time": mins["time"],
            },
        }

    def _breakdown_failure(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Build detailed breakdown for a failed run."""
        WEIGHT_EXPLORE = 0.5
        WEIGHT_CONF = 0.5
        TARGET_SEEN_MULTIPLIER = 1.5

        fov = run.get("fov_coverage")
        exploration = fov if fov is not None else 0.0
        conf = run.get("confidence")
        confidence = conf if conf is not None else 0.0
        target_seen = bool(run.get("target_seen"))

        explore_contrib = WEIGHT_EXPLORE * (1.0 - exploration)
        conf_contrib = WEIGHT_CONF * confidence
        raw = -100.0 * (explore_contrib + conf_contrib)

        multiplier_applied = False
        if target_seen:
            raw *= TARGET_SEEN_MULTIPLIER
            multiplier_applied = True

        total = max(-100.0, min(0.0, raw))

        components = [
            {
                "name": "exploration",
                "value": round(exploration, 4),
                "weight": WEIGHT_EXPLORE,
                "contribution": round(-100.0 * explore_contrib, 2),
            },
            {
                "name": "confidence",
                "value": round(confidence, 4),
                "weight": WEIGHT_CONF,
                "contribution": round(-100.0 * conf_contrib, 2),
            },
        ]

        result: Dict[str, Any] = {
            "total": round(total, 2),
            "success": False,
            "components": components,
            "target_seen": target_seen,
        }
        if multiplier_applied:
            result["target_seen_multiplier"] = TARGET_SEEN_MULTIPLIER

        return result

    def get_best_runs_per_scenario(
        self, scenario_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Return the single best successful run for each scenario.

        "Best" is defined as: success=True, then fewest steps, then
        shortest distance travelled.  Scenarios with no successful run
        are omitted from the result.

        Parameters
        ----------
        scenario_ids : list of str
            Scenarios to look up.

        Returns
        -------
        dict
            ``{scenario_id: run_dict}`` for each scenario that has at
            least one successful run.
        """
        if not scenario_ids:
            return {}

        placeholders = ",".join("?" * len(scenario_ids))
        rows = self._conn.execute(
            f"""\
            SELECT * FROM runs AS r
            WHERE r.success = 1
              AND r.scenario_id IN ({placeholders})
              AND r.steps_taken = (
                  SELECT MIN(r2.steps_taken)
                  FROM runs r2
                  WHERE r2.scenario_id = r.scenario_id
                    AND r2.success = 1
              )
            ORDER BY r.scenario_id, COALESCE(r.distance_travelled, 9e18) ASC
            """,
            list(scenario_ids),
        ).fetchall()

        # Keep the first row per scenario: fewest steps, then shortest distance.
        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            d = self._row_to_dict(row)
            sid = d["scenario_id"]
            if sid not in result:
                result[sid] = d
        return result

    def get_global_leaderboard(
        self,
        scenario_ids: List[str],
        limit: int = 50,
        mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate per-scenario scores into a global agent ranking.

        For each agent (identified by ``player_name`` or ``model_name``),
        the best run per scenario is selected (success first, then fewest
        steps, then shortest distance).  :meth:`compute_score_for_run` is
        called on each best run and the results are summed.

        Parameters
        ----------
        scenario_ids : list of str
            Only runs belonging to these scenarios are considered.
        limit : int
            Maximum number of agents to return (default 50).
        mode : str or None
            Optional filter (``"ai"`` or ``"human"``).

        Returns
        -------
        list of dict
            Each entry contains:
            ``agent_name``, ``total_score``, ``scenarios_attempted``,
            ``runs`` (list of best-run dicts per scenario with a ``score``
            field added to each).
            Sorted by ``total_score`` descending.
        """
        if not scenario_ids:
            return []

        placeholders = ",".join("?" * len(scenario_ids))
        params: list[Any] = list(scenario_ids)

        mode_clause = ""
        if mode is not None:
            mode_clause = " AND mode = ?"
            params.append(mode)

        rows = self._conn.execute(
            f"""\
            SELECT * FROM runs
            WHERE scenario_id IN ({placeholders}){mode_clause}
            ORDER BY success DESC, steps_taken ASC, distance_travelled ASC
            """,
            params,
        ).fetchall()

        if not rows:
            return []

        # Group all runs by (agent_name, scenario_id), keeping best run only.
        AgentKey = str
        agent_scenario_best: Dict[AgentKey, Dict[str, Dict[str, Any]]] = {}

        for row in rows:
            run = self._row_to_dict(row)
            agent_name: str = (
                run.get("player_name") or run.get("model_name") or "unknown"
            )
            sid: str = run["scenario_id"]

            if agent_name not in agent_scenario_best:
                agent_scenario_best[agent_name] = {}

            # Rows are already ordered best-first; keep first seen per (agent, scenario).
            if sid not in agent_scenario_best[agent_name]:
                agent_scenario_best[agent_name][sid] = run

        # Pre-load reference mins for all scenarios in one pass to avoid N+1 queries.
        ref_mins: Dict[str, Dict[str, Any]] = {
            sid: self._fetch_reference_mins(sid) for sid in scenario_ids
        }

        # Build leaderboard entries.
        entries: List[Dict[str, Any]] = []
        for agent_name, scenario_runs in agent_scenario_best.items():
            total_score = 0.0
            best_runs: List[Dict[str, Any]] = []
            scenario_scores: Dict[str, float] = {}
            for sid, run in scenario_runs.items():
                score = self._compute_score_with_mins(
                    run, ref_mins.get(sid, {"dist": None, "steps": None, "time": None})
                )
                total_score += score
                run_copy = dict(run)
                run_copy["score"] = score
                best_runs.append(run_copy)
                scenario_scores[sid] = round(score, 1)
            entries.append(
                {
                    "agent_name": agent_name,
                    "total_score": total_score,
                    "scenarios_attempted": len(scenario_runs),
                    "scenario_scores": scenario_scores,
                    "runs": best_runs,
                }
            )

        entries.sort(key=lambda e: e["total_score"], reverse=True)
        return entries[:limit]

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
