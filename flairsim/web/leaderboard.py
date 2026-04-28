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
    discovery_coverage    REAL,
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
            ("discovery_coverage", "REAL"),
            ("target_seen", "INTEGER"),
        ]
        for col, dtype in migrations:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {dtype}")
                logger.info("Migrated leaderboard: added column '%s'", col)

        # Copy fov_coverage → discovery_coverage for legacy rows.
        if "fov_coverage" in existing or "fov_coverage" in [
            m[0] for m in migrations if m[0] == "fov_coverage"
        ]:
            self._conn.execute(
                "UPDATE runs SET discovery_coverage = fov_coverage "
                "WHERE discovery_coverage IS NULL AND fov_coverage IS NOT NULL"
            )

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
            ``confidence`` (float or None), ``discovery_coverage`` (float or None),
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
                 metrics, score_final, confidence, discovery_coverage, target_seen, created_at)
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
                run_data.get("discovery_coverage"),
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

    # ---------------------------------------------------------------- Pareto

    @staticmethod
    def compute_pareto_front(
        runs: List[Dict[str, Any]],
        objectives: List[str],
    ) -> List[Dict[str, Any]]:
        """Compute the Pareto front for a list of runs (all objectives minimised).

        Parameters
        ----------
        runs : list of dict
            Run dicts.  Each must contain the keys listed in *objectives*.
        objectives : list of str
            Column names to minimise (e.g. ``["steps_taken", "duration_s",
            "distance_travelled"]``).

        Returns
        -------
        list of dict
            The subset of *runs* that lie on the Pareto front (non-dominated).
        """
        if not runs:
            return []

        def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            """Return True if *a* dominates *b* (all objectives <=, at least one <)."""
            at_least_one_better = False
            for obj in objectives:
                va = a.get(obj) if a.get(obj) is not None else float("inf")
                vb = b.get(obj) if b.get(obj) is not None else float("inf")
                if va > vb:
                    return False
                if va < vb:
                    at_least_one_better = True
            return at_least_one_better

        front: List[Dict[str, Any]] = []
        for candidate in runs:
            is_dominated = False
            new_front: List[Dict[str, Any]] = []
            for existing in front:
                if _dominates(existing, candidate):
                    is_dominated = True
                    new_front.append(existing)
                elif _dominates(candidate, existing):
                    continue
                else:
                    new_front.append(existing)
            if not is_dominated:
                new_front.append(candidate)
            front = new_front
        return front

    @staticmethod
    def select_best_run_pareto(
        runs: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Select the single best run for an agent on a scenario using Pareto.

        Strategy:
        1. Filter successful runs.
        2. Compute Pareto front on (steps_taken, duration_s, distance_travelled).
        3. Pick the run closest to the origin (normalised Euclidean distance).
        4. If no successful runs, pick the run with the fewest steps.

        Returns ``None`` if *runs* is empty.
        """
        if not runs:
            return None

        objectives = ["steps_taken", "duration_s", "distance_travelled"]
        successful = [r for r in runs if r.get("success")]

        if successful:
            front = Leaderboard.compute_pareto_front(successful, objectives)
            if len(front) == 1:
                return front[0]

            # Normalise each objective to [0, 1] across the front, then pick
            # the run with the smallest Euclidean distance to origin.
            mins = {}
            maxs = {}
            for obj in objectives:
                vals = [r.get(obj, 0) if r.get(obj) is not None else 0 for r in front]
                mins[obj] = min(vals) if vals else 0
                maxs[obj] = max(vals) if vals else 1

            best = None
            best_dist = float("inf")
            for r in front:
                dist_sq = 0.0
                for obj in objectives:
                    v = r.get(obj, 0) if r.get(obj) is not None else 0
                    span = maxs[obj] - mins[obj]
                    norm = (v - mins[obj]) / span if span > 0 else 0.0
                    dist_sq += norm**2
                if dist_sq < best_dist:
                    best_dist = dist_sq
                    best = r
            return best

        # No successful runs — pick the one with the fewest steps.
        return min(runs, key=lambda r: r.get("steps_taken") or float("inf"))

    def get_scenario_results(
        self, scenario_id: str, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return flat metrics for a scenario: one best run per agent.

        Parameters
        ----------
        scenario_id : str
            Scenario to query.
        mode : str or None
            Optional filter (``"ai"`` or ``"human"``).

        Returns
        -------
        dict
            ``{"scenario_id": ..., "agents": [...]}``.
            Each agent entry: ``agent_name``, flat metrics, ``run_id``.
        """
        all_runs = self.get_runs(scenario_id=scenario_id, mode=mode, limit=10000)

        # Group runs by agent name.
        by_agent: Dict[str, List[Dict[str, Any]]] = {}
        for run in all_runs:
            agent = run.get("player_name") or run.get("model_name") or "unknown"
            by_agent.setdefault(agent, []).append(run)

        agents: List[Dict[str, Any]] = []
        for agent_name, agent_runs in by_agent.items():
            best = self.select_best_run_pareto(agent_runs)
            if best is None:
                continue
            agents.append(
                {
                    "agent_name": agent_name,
                    "run_id": best.get("id"),
                    "success": best.get("success", False),
                    "steps_taken": best.get("steps_taken"),
                    "duration_s": best.get("duration_s"),
                    "distance_travelled": best.get("distance_travelled"),
                    "target_seen": best.get("target_seen"),
                }
            )

        # Assign Pareto ranks (layered Pareto fronts).
        agents = self._assign_pareto_ranks(agents)
        return {"scenario_id": scenario_id, "agents": agents}

    def get_global_results(
        self,
        scenario_ids: List[str],
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Global results: average metrics for agents that completed ALL scenarios.

        Parameters
        ----------
        scenario_ids : list of str
            All scenario IDs in the benchmark.
        mode : str or None
            Optional filter.

        Returns
        -------
        dict
            ``{"scenario_ids": [...], "agents": [...]}``.
            Each agent entry has averaged metrics across all scenarios.
        """
        if not scenario_ids:
            return {"scenario_ids": [], "agents": []}

        total_scenarios = len(scenario_ids)

        # Collect best run per (agent, scenario).
        agent_scenarios: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for sid in scenario_ids:
            result = self.get_scenario_results(sid, mode=mode)
            for entry in result["agents"]:
                name = entry["agent_name"]
                agent_scenarios.setdefault(name, {})[sid] = entry

        # Keep only agents that participated in ALL scenarios.
        complete_agents: List[Dict[str, Any]] = []
        metric_keys = [
            "steps_taken",
            "duration_s",
            "distance_travelled",
        ]

        for agent_name, scenarios in agent_scenarios.items():
            if len(scenarios) < total_scenarios:
                continue

            successes = sum(1 for e in scenarios.values() if e.get("success"))
            avg_metrics: Dict[str, Any] = {
                "agent_name": agent_name,
                "scenarios_completed": len(scenarios),
                "success_rate": successes / total_scenarios,
            }

            for key in metric_keys:
                vals = [e[key] for e in scenarios.values() if e.get(key) is not None]
                avg_metrics[f"avg_{key}"] = sum(vals) / len(vals) if vals else None

            # Per-scenario breakdown.
            avg_metrics["per_scenario"] = {
                sid: entry for sid, entry in scenarios.items()
            }
            complete_agents.append(avg_metrics)

        # Assign Pareto ranks (layered Pareto fronts on averaged metrics).
        complete_agents = self._assign_pareto_ranks(
            complete_agents,
            key_map={
                "steps_taken": "avg_steps_taken",
                "duration_s": "avg_duration_s",
                "distance_travelled": "avg_distance_travelled",
            },
        )
        return {"scenario_ids": scenario_ids, "agents": complete_agents}

    def _assign_pareto_ranks(
        self,
        agents: List[Dict[str, Any]],
        key_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Assign unique ``pareto_rank`` using sorted rank-vector comparison.

        For each metric the agents are ranked 1..N (lower-is-better for steps,
        time, distance; higher-is-better for coverage — inverted so rank 1 is
        always best).  Each agent's per-metric ranks are **sorted ascending**
        to form a rank vector, then agents are compared **lexicographically**
        on these vectors.  The resulting order gives a unique final rank.

        Successful agents are always ranked above failed agents.

        Parameters
        ----------
        agents : list of dict
            Agent result dicts (must contain success + metric keys).
        key_map : dict or None
            Remap objective keys (e.g. ``{"steps_taken": "avg_steps_taken"}``).
        """
        km = key_map or {}
        # Metrics where lower is better.
        lower_better = [
            km.get("steps_taken", "steps_taken"),
            km.get("duration_s", "duration_s"),
            km.get("distance_travelled", "distance_travelled"),
        ]
        # Metrics where higher is better.
        higher_better: list[str] = []

        ok = [a for a in agents if a.get("success") or a.get("success_rate", 0) > 0]
        fail = [a for a in agents if a not in ok]

        def _rank_group(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not group:
                return []
            n = len(group)
            # For each metric, compute per-agent rank (1-based, ties get same rank).
            per_metric_ranks: Dict[str, List[int]] = {
                a.get("agent_name", str(i)): [] for i, a in enumerate(group)
            }
            all_metrics = lower_better + higher_better
            for metric in all_metrics:
                reverse = metric in higher_better
                vals = []
                for i, a in enumerate(group):
                    v = a.get(metric)
                    if v is None:
                        v = float("inf") if not reverse else float("-inf")
                    vals.append((v, i))
                vals.sort(key=lambda x: x[0], reverse=reverse)
                # Assign ranks with ties.
                ranks = [0] * n
                current_rank = 1
                for pos, (val, idx) in enumerate(vals):
                    if pos > 0 and val != vals[pos - 1][0]:
                        current_rank = pos + 1
                    ranks[idx] = current_rank
                for i, a in enumerate(group):
                    key = a.get("agent_name", str(i))
                    per_metric_ranks[key].append(ranks[i])

            # Sort each agent's rank vector ascending, then compare lexicographically.
            for i, a in enumerate(group):
                key = a.get("agent_name", str(i))
                a["_rank_vector"] = sorted(per_metric_ranks[key])

            group.sort(key=lambda a: a["_rank_vector"])
            return group

        ok = _rank_group(ok)
        fail = _rank_group(fail)

        # Assign final ranks.
        all_agents = ok + fail
        for i, a in enumerate(all_agents):
            a["pareto_rank"] = i + 1
            a.pop("_rank_vector", None)

        return all_agents

    def get_agent_runs(
        self, agent_name: str, scenario_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all runs for a given agent, optionally filtered by scenario.

        Parameters
        ----------
        agent_name : str
            Agent name (matched against player_name or model_name).
        scenario_id : str or None
            Optional scenario filter.

        Returns
        -------
        list of dict
            All runs for this agent, ordered by created_at DESC.
        """
        clauses = ["(player_name = ? OR model_name = ?)"]
        params: list[Any] = [agent_name, agent_name]

        if scenario_id:
            clauses.append("scenario_id = ?")
            params.append(scenario_id)

        where = "WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM runs {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
