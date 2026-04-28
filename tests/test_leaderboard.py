"""
Tests for :mod:`flairsim.web.leaderboard`.

Tests SQLite CRUD operations, filtering, ranking, and edge cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from flairsim.web.leaderboard import Leaderboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lb(tmp_path):
    """Create a Leaderboard with a temp database."""
    db_path = tmp_path / "test_leaderboard.db"
    board = Leaderboard(db_path=db_path)
    yield board
    board.close()


def _make_run(
    scenario_id="find_red_car_D004",
    mode="human",
    success=False,
    steps_taken=50,
    distance_travelled=1200.0,
    player_name="Alice",
    **kwargs,
):
    """Helper to build a run data dict."""
    d = {
        "scenario_id": scenario_id,
        "mode": mode,
        "success": success,
        "steps_taken": steps_taken,
        "distance_travelled": distance_travelled,
        "player_name": player_name,
    }
    d.update(kwargs)
    return d


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------


class TestTableCreation:
    """Verify the database initialises correctly."""

    def test_db_file_created(self, tmp_path):
        db_path = tmp_path / "new_board.db"
        board = Leaderboard(db_path=db_path)
        assert db_path.exists()
        board.close()

    def test_parent_dirs_created(self, tmp_path):
        db_path = tmp_path / "sub" / "dir" / "board.db"
        board = Leaderboard(db_path=db_path)
        assert db_path.exists()
        board.close()


# ---------------------------------------------------------------------------
# Submit + get
# ---------------------------------------------------------------------------


class TestSubmitAndGet:
    """Test submit_run and get_run."""

    def test_submit_returns_uuid(self, lb):
        run_id = lb.submit_run(_make_run())
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID format

    def test_get_run_returns_submitted_data(self, lb):
        run_id = lb.submit_run(
            _make_run(
                player_name="Bob",
                scenario_id="quick_explore",
                mode="ai",
                success=True,
                steps_taken=42,
                distance_travelled=800.5,
            )
        )
        run = lb.get_run(run_id)
        assert run is not None
        assert run["id"] == run_id
        assert run["player_name"] == "Bob"
        assert run["scenario_id"] == "quick_explore"
        assert run["mode"] == "ai"
        assert run["success"] is True
        assert run["steps_taken"] == 42
        assert run["distance_travelled"] == 800.5

    def test_get_run_unknown_returns_none(self, lb):
        assert lb.get_run("nonexistent-uuid") is None

    def test_submit_with_trajectory(self, lb):
        traj = [{"x": 1.0, "y": 2.0, "z": 100.0}, {"x": 3.0, "y": 4.0, "z": 100.0}]
        run_id = lb.submit_run(_make_run(trajectory=traj))
        run = lb.get_run(run_id)
        assert run["trajectory"] == traj

    def test_submit_with_duration(self, lb):
        run_id = lb.submit_run(_make_run(duration_s=45.2))
        run = lb.get_run(run_id)
        assert run["duration_s"] == 45.2

    def test_submit_with_model_name(self, lb):
        run_id = lb.submit_run(_make_run(mode="ai", model_name="GPT-4o"))
        run = lb.get_run(run_id)
        assert run["model_name"] == "GPT-4o"

    def test_submit_with_session_id(self, lb):
        run_id = lb.submit_run(_make_run(session_id="sess-123"))
        run = lb.get_run(run_id)
        assert run["session_id"] == "sess-123"

    def test_success_stored_as_bool(self, lb):
        rid_true = lb.submit_run(_make_run(success=True))
        rid_false = lb.submit_run(_make_run(success=False))
        assert lb.get_run(rid_true)["success"] is True
        assert lb.get_run(rid_false)["success"] is False


# ---------------------------------------------------------------------------
# Querying / filtering
# ---------------------------------------------------------------------------


class TestGetRuns:
    """Test get_runs with filters and ordering."""

    def test_empty_leaderboard(self, lb):
        runs = lb.get_runs()
        assert runs == []

    def test_returns_all_runs(self, lb):
        lb.submit_run(_make_run(player_name="A"))
        lb.submit_run(_make_run(player_name="B"))
        lb.submit_run(_make_run(player_name="C"))
        assert len(lb.get_runs()) == 3

    def test_filter_by_scenario(self, lb):
        lb.submit_run(_make_run(scenario_id="s1"))
        lb.submit_run(_make_run(scenario_id="s2"))
        lb.submit_run(_make_run(scenario_id="s1"))

        runs = lb.get_runs(scenario_id="s1")
        assert len(runs) == 2
        assert all(r["scenario_id"] == "s1" for r in runs)

    def test_filter_by_mode(self, lb):
        lb.submit_run(_make_run(mode="human"))
        lb.submit_run(_make_run(mode="ai"))
        lb.submit_run(_make_run(mode="human"))

        runs = lb.get_runs(mode="ai")
        assert len(runs) == 1
        assert runs[0]["mode"] == "ai"

    def test_filter_by_scenario_and_mode(self, lb):
        lb.submit_run(_make_run(scenario_id="s1", mode="human"))
        lb.submit_run(_make_run(scenario_id="s1", mode="ai"))
        lb.submit_run(_make_run(scenario_id="s2", mode="human"))

        runs = lb.get_runs(scenario_id="s1", mode="human")
        assert len(runs) == 1

    def test_limit(self, lb):
        for i in range(20):
            lb.submit_run(_make_run(player_name=f"P{i}"))

        runs = lb.get_runs(limit=5)
        assert len(runs) == 5

    def test_ordering_success_first(self, lb):
        """Successful runs should rank above failed ones."""
        lb.submit_run(_make_run(success=False, steps_taken=10))
        lb.submit_run(_make_run(success=True, steps_taken=100))
        lb.submit_run(_make_run(success=False, steps_taken=5))

        runs = lb.get_runs()
        assert runs[0]["success"] is True

    def test_ordering_fewer_steps_first(self, lb):
        """Among successful runs, fewer steps ranks higher."""
        lb.submit_run(_make_run(success=True, steps_taken=100, distance_travelled=500))
        lb.submit_run(_make_run(success=True, steps_taken=50, distance_travelled=500))
        lb.submit_run(_make_run(success=True, steps_taken=75, distance_travelled=500))

        runs = lb.get_runs()
        assert runs[0]["steps_taken"] == 50
        assert runs[1]["steps_taken"] == 75
        assert runs[2]["steps_taken"] == 100

    def test_ordering_shorter_distance_first(self, lb):
        """Among runs with same steps, shorter distance ranks higher."""
        lb.submit_run(_make_run(success=True, steps_taken=50, distance_travelled=800))
        lb.submit_run(_make_run(success=True, steps_taken=50, distance_travelled=500))
        lb.submit_run(_make_run(success=True, steps_taken=50, distance_travelled=600))

        runs = lb.get_runs()
        assert runs[0]["distance_travelled"] == 500
        assert runs[1]["distance_travelled"] == 600
        assert runs[2]["distance_travelled"] == 800

    def test_full_ranking_order(self, lb):
        """Integration test for the full ranking: success DESC, steps ASC, dist ASC."""
        lb.submit_run(
            _make_run(
                success=False,
                steps_taken=10,
                distance_travelled=100,
                player_name="Loser",
            )
        )
        lb.submit_run(
            _make_run(
                success=True, steps_taken=30, distance_travelled=200, player_name="OK"
            )
        )
        lb.submit_run(
            _make_run(
                success=True, steps_taken=20, distance_travelled=150, player_name="Best"
            )
        )
        lb.submit_run(
            _make_run(
                success=True, steps_taken=20, distance_travelled=300, player_name="Good"
            )
        )

        runs = lb.get_runs()
        names = [r["player_name"] for r in runs]
        assert names == ["Best", "Good", "OK", "Loser"]


# ---------------------------------------------------------------------------
# New columns (Phase 1.5)
# ---------------------------------------------------------------------------


class TestNewColumns:
    """Test new Phase 1.5 columns: steps_detail, model_info, metrics, score_final."""

    def test_submit_with_steps_detail(self, lb):
        detail = [
            {
                "dx": 1.0,
                "dy": 2.0,
                "dz": 0.0,
                "action_type": "move",
                "reason": "exploring north",
            },
            {
                "dx": -1.0,
                "dy": 0.0,
                "dz": 0.0,
                "action_type": "move",
                "reason": "correcting east",
            },
        ]
        run_id = lb.submit_run(_make_run(steps_detail=detail))
        run = lb.get_run(run_id)
        assert run["steps_detail"] == detail
        assert isinstance(run["steps_detail"], list)
        assert run["steps_detail"][0]["reason"] == "exploring north"

    def test_submit_with_model_info(self, lb):
        info = {
            "model_name": "GPT-4o",
            "provider": "openai",
            "temperature": 0.7,
            "prompt_version": "v2",
        }
        run_id = lb.submit_run(_make_run(mode="ai", model_info=info))
        run = lb.get_run(run_id)
        assert run["model_info"] == info
        assert isinstance(run["model_info"], dict)
        assert run["model_info"]["provider"] == "openai"

    def test_submit_with_metrics(self, lb):
        metrics = {"iou": 0.85, "precision": 0.9, "recall": 0.8}
        run_id = lb.submit_run(_make_run(metrics=metrics))
        run = lb.get_run(run_id)
        assert run["metrics"] == metrics
        assert isinstance(run["metrics"], dict)
        assert run["metrics"]["iou"] == 0.85

    def test_submit_with_score_final(self, lb):
        run_id = lb.submit_run(_make_run(score_final=42.5))
        run = lb.get_run(run_id)
        assert run["score_final"] == 42.5

    def test_score_final_null_by_default(self, lb):
        run_id = lb.submit_run(_make_run())
        run = lb.get_run(run_id)
        assert run["score_final"] is None

    def test_json_columns_null_by_default(self, lb):
        run_id = lb.submit_run(_make_run())
        run = lb.get_run(run_id)
        assert run["steps_detail"] is None
        assert run["model_info"] is None
        assert run["metrics"] is None

    def test_submit_all_new_columns_together(self, lb):
        run_id = lb.submit_run(
            _make_run(
                mode="ai",
                steps_detail=[{"dx": 1, "reason": "go"}],
                model_info={"model_name": "Claude"},
                metrics={"score": 100},
                score_final=99.9,
            )
        )
        run = lb.get_run(run_id)
        assert run["steps_detail"] == [{"dx": 1, "reason": "go"}]
        assert run["model_info"]["model_name"] == "Claude"
        assert run["metrics"]["score"] == 100
        assert run["score_final"] == 99.9


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


class TestMigration:
    """Test that _migrate() adds missing columns to old-schema databases."""

    def test_migrate_adds_columns(self, tmp_path):
        """Create a DB with the old schema (no new columns), then instantiate
        Leaderboard which should auto-migrate."""
        import sqlite3

        db_path = tmp_path / "old_schema.db"
        conn = sqlite3.connect(str(db_path))
        # Old schema without the 4 new columns.
        conn.execute(
            """\
            CREATE TABLE runs (
                id              TEXT PRIMARY KEY,
                session_id      TEXT,
                scenario_id     TEXT NOT NULL,
                player_name     TEXT,
                model_name      TEXT,
                mode            TEXT NOT NULL,
                success         INTEGER NOT NULL DEFAULT 0,
                reason          TEXT,
                steps_taken     INTEGER,
                distance_travelled REAL,
                duration_s      REAL,
                trajectory      TEXT,
                created_at      TEXT NOT NULL
            );
            """
        )
        conn.commit()
        conn.close()

        # Instantiate Leaderboard — should migrate.
        board = Leaderboard(db_path=db_path)

        # Verify new columns exist.
        col_names = {
            row[1] for row in board._conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        assert "steps_detail" in col_names
        assert "model_info" in col_names
        assert "metrics" in col_names
        assert "score_final" in col_names

        # Verify we can insert and read with new columns.
        run_id = board.submit_run(
            {
                "scenario_id": "test",
                "mode": "human",
                "model_info": {"model_name": "test"},
                "score_final": 1.0,
            }
        )
        run = board.get_run(run_id)
        assert run["model_info"]["model_name"] == "test"
        assert run["score_final"] == 1.0
        board.close()


# ---------------------------------------------------------------------------
# New columns Phase 2
# ---------------------------------------------------------------------------


class TestNewColumnsV2:
    """Phase 2 columns: confidence, discovery_coverage, target_seen."""

    def test_submit_with_confidence(self, lb):
        run_id = lb.submit_run(_make_run(confidence=0.85))
        run = lb.get_run(run_id)
        assert run["confidence"] == pytest.approx(0.85)

    def test_confidence_null_by_default(self, lb):
        run_id = lb.submit_run(_make_run())
        run = lb.get_run(run_id)
        assert run["confidence"] is None

    def test_submit_with_discovery_coverage(self, lb):
        run_id = lb.submit_run(_make_run(discovery_coverage=0.42))
        run = lb.get_run(run_id)
        assert run["discovery_coverage"] == pytest.approx(0.42)

    def test_submit_with_target_seen(self, lb):
        run_id = lb.submit_run(_make_run(target_seen=True))
        run = lb.get_run(run_id)
        assert run["target_seen"] is True

    def test_target_seen_false(self, lb):
        run_id = lb.submit_run(_make_run(target_seen=False))
        run = lb.get_run(run_id)
        assert run["target_seen"] is False

    def test_target_seen_null_by_default(self, lb):
        run_id = lb.submit_run(_make_run())
        run = lb.get_run(run_id)
        assert run["target_seen"] is None


class TestAgentsTable:
    """CRUD operations on the agents table."""

    def test_create_agent(self, lb):
        lb.create_agent("gpt4o-v1", {"model": "gpt-4o", "provider": "openai"})
        agent = lb.get_agent("gpt4o-v1")
        assert agent is not None
        assert agent["name"] == "gpt4o-v1"
        assert agent["specs"]["model"] == "gpt-4o"

    def test_create_agent_duplicate_raises(self, lb):
        lb.create_agent("agent-x", {})
        with pytest.raises(ValueError, match="already exists"):
            lb.create_agent("agent-x", {})

    def test_get_agent_not_found(self, lb):
        assert lb.get_agent("no-such-agent") is None

    def test_update_agent_specs(self, lb):
        lb.create_agent("my-bot", {"version": "1"})
        lb.update_agent("my-bot", {"version": "2", "extra": "data"})
        agent = lb.get_agent("my-bot")
        assert agent["specs"]["version"] == "2"
        assert agent["specs"]["extra"] == "data"

    def test_update_agent_not_found_raises(self, lb):
        with pytest.raises(KeyError):
            lb.update_agent("ghost", {"x": 1})

    def test_agent_has_created_at(self, lb):
        lb.create_agent("bot-ts", {})
        agent = lb.get_agent("bot-ts")
        assert agent["created_at"] is not None

    def test_migration_adds_agents_table(self, tmp_path):
        """Old DB (no agents table) is migrated correctly."""
        import sqlite3

        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE runs ("
            "id TEXT PRIMARY KEY, scenario_id TEXT NOT NULL, "
            "mode TEXT NOT NULL, success INTEGER NOT NULL DEFAULT 0, "
            "created_at TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()

        board = Leaderboard(db_path=db_path)
        tables = {
            row[0]
            for row in board._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "agents" in tables
        board.close()

    def test_migration_adds_new_run_columns(self, tmp_path):
        """Old DB without confidence/discovery_coverage/target_seen is migrated."""
        import sqlite3

        db_path = tmp_path / "old2.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE runs ("
            "id TEXT PRIMARY KEY, scenario_id TEXT NOT NULL, "
            "mode TEXT NOT NULL, success INTEGER NOT NULL DEFAULT 0, "
            "created_at TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()

        board = Leaderboard(db_path=db_path)
        col_names = {
            row[1] for row in board._conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        assert "confidence" in col_names
        assert "discovery_coverage" in col_names
        assert "target_seen" in col_names
        board.close()


# ---------------------------------------------------------------------------
# Scoring functions (Task 2)
# ---------------------------------------------------------------------------


def _make_success_run(
    scenario_id: str = "sc1",
    player_name: str = "Alice",
    steps_taken: int = 10,
    distance_travelled: float = 100.0,
    duration_s: float = 10.0,
    **kwargs,
) -> dict:
    """Helper: build a successful run with all scoring-relevant fields."""
    return _make_run(
        scenario_id=scenario_id,
        player_name=player_name,
        success=True,
        steps_taken=steps_taken,
        distance_travelled=distance_travelled,
        duration_s=duration_s,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


class TestParetoFront:
    """Tests for Leaderboard.compute_pareto_front()."""

    def test_empty_returns_empty(self):
        assert Leaderboard.compute_pareto_front([], ["steps_taken"]) == []

    def test_single_run_is_front(self):
        runs = [{"steps_taken": 10, "duration_s": 5}]
        front = Leaderboard.compute_pareto_front(runs, ["steps_taken", "duration_s"])
        assert front == runs

    def test_dominated_run_excluded(self):
        a = {"steps_taken": 10, "duration_s": 5, "id": "a"}
        b = {"steps_taken": 20, "duration_s": 10, "id": "b"}  # dominated by a
        front = Leaderboard.compute_pareto_front([a, b], ["steps_taken", "duration_s"])
        assert len(front) == 1
        assert front[0]["id"] == "a"

    def test_non_dominated_both_kept(self):
        a = {"steps_taken": 10, "duration_s": 20, "id": "a"}
        b = {"steps_taken": 20, "duration_s": 5, "id": "b"}
        front = Leaderboard.compute_pareto_front([a, b], ["steps_taken", "duration_s"])
        assert len(front) == 2

    def test_identical_runs_all_kept(self):
        a = {"steps_taken": 10, "duration_s": 5, "id": "a"}
        b = {"steps_taken": 10, "duration_s": 5, "id": "b"}
        front = Leaderboard.compute_pareto_front([a, b], ["steps_taken", "duration_s"])
        assert len(front) == 2

    def test_none_values_treated_as_inf(self):
        a = {"steps_taken": 10, "duration_s": None, "id": "a"}
        b = {"steps_taken": 20, "duration_s": 5, "id": "b"}
        front = Leaderboard.compute_pareto_front([a, b], ["steps_taken", "duration_s"])
        # Neither dominates the other (a has None=inf on duration)
        assert len(front) == 2


# ---------------------------------------------------------------------------
# Select best run (Pareto)
# ---------------------------------------------------------------------------


class TestSelectBestRunPareto:
    """Tests for Leaderboard.select_best_run_pareto()."""

    def test_empty_returns_none(self):
        assert Leaderboard.select_best_run_pareto([]) is None

    def test_single_success(self):
        r = {
            "success": True,
            "steps_taken": 10,
            "duration_s": 5,
            "distance_travelled": 100,
        }
        assert Leaderboard.select_best_run_pareto([r]) == r

    def test_picks_dominating_success(self):
        a = {
            "success": True,
            "steps_taken": 10,
            "duration_s": 5,
            "distance_travelled": 50,
            "id": "a",
        }
        b = {
            "success": True,
            "steps_taken": 20,
            "duration_s": 10,
            "distance_travelled": 100,
            "id": "b",
        }
        best = Leaderboard.select_best_run_pareto([a, b])
        assert best["id"] == "a"

    def test_no_success_picks_fewest_steps(self):
        a = {"success": False, "steps_taken": 50, "id": "a"}
        b = {"success": False, "steps_taken": 20, "id": "b"}
        best = Leaderboard.select_best_run_pareto([a, b])
        assert best["id"] == "b"

    def test_no_success_none_steps(self):
        a = {"success": False, "steps_taken": None, "id": "a"}
        b = {"success": False, "steps_taken": 10, "id": "b"}
        best = Leaderboard.select_best_run_pareto([a, b])
        assert best["id"] == "b"

    def test_compromise_closest_to_origin(self):
        """Among 2 non-dominated successful runs, picks the one closest to origin."""
        a = {
            "success": True,
            "steps_taken": 10,
            "duration_s": 100,
            "distance_travelled": 100,
            "id": "a",
        }
        b = {
            "success": True,
            "steps_taken": 100,
            "duration_s": 10,
            "distance_travelled": 100,
            "id": "b",
        }
        c = {
            "success": True,
            "steps_taken": 20,
            "duration_s": 20,
            "distance_travelled": 50,
            "id": "c",
        }
        best = Leaderboard.select_best_run_pareto([a, b, c])
        # c should be the compromise (closest to origin after normalisation)
        assert best["id"] == "c"


# ---------------------------------------------------------------------------
# Scenario results
# ---------------------------------------------------------------------------


class TestScenarioResults:
    """Tests for Leaderboard.get_scenario_results()."""

    def test_empty_scenario(self, lb):
        result = lb.get_scenario_results("nonexistent")
        assert result["scenario_id"] == "nonexistent"
        assert result["agents"] == []

    def test_one_agent_one_run(self, lb):
        lb.submit_run(_make_success_run(scenario_id="s1", player_name="Alice"))
        result = lb.get_scenario_results("s1")
        assert len(result["agents"]) == 1
        assert result["agents"][0]["agent_name"] == "Alice"
        assert result["agents"][0]["success"] is True

    def test_two_agents(self, lb):
        lb.submit_run(
            _make_success_run(scenario_id="s1", player_name="Alice", steps_taken=10)
        )
        lb.submit_run(
            _make_success_run(scenario_id="s1", player_name="Bob", steps_taken=20)
        )
        result = lb.get_scenario_results("s1")
        assert len(result["agents"]) == 2
        # Sorted by steps ascending (both successful)
        assert result["agents"][0]["agent_name"] == "Alice"

    def test_best_run_selected_per_agent(self, lb):
        lb.submit_run(
            _make_success_run(scenario_id="s1", player_name="Alice", steps_taken=50)
        )
        lb.submit_run(
            _make_success_run(scenario_id="s1", player_name="Alice", steps_taken=10)
        )
        result = lb.get_scenario_results("s1")
        assert len(result["agents"]) == 1
        assert result["agents"][0]["steps_taken"] == 10

    def test_flat_metrics_present(self, lb):
        lb.submit_run(
            _make_success_run(
                scenario_id="s1",
                player_name="Alice",
                discovery_coverage=0.8,
                target_seen=True,
            )
        )
        result = lb.get_scenario_results("s1")
        agent = result["agents"][0]
        for key in (
            "success",
            "steps_taken",
            "duration_s",
            "distance_travelled",
            "target_seen",
        ):
            assert key in agent


# ---------------------------------------------------------------------------
# Global results
# ---------------------------------------------------------------------------


class TestGlobalResults:
    """Tests for Leaderboard.get_global_results()."""

    def test_empty(self, lb):
        result = lb.get_global_results(["s1", "s2"])
        assert result["agents"] == []

    def test_agent_must_complete_all_scenarios(self, lb):
        lb.submit_run(_make_success_run(scenario_id="s1", player_name="Alice"))
        # Alice only did s1, not s2
        result = lb.get_global_results(["s1", "s2"])
        assert result["agents"] == []

    def test_agent_completes_all(self, lb):
        lb.submit_run(
            _make_success_run(scenario_id="s1", player_name="Alice", steps_taken=10)
        )
        lb.submit_run(
            _make_success_run(scenario_id="s2", player_name="Alice", steps_taken=20)
        )
        result = lb.get_global_results(["s1", "s2"])
        assert len(result["agents"]) == 1
        agent = result["agents"][0]
        assert agent["agent_name"] == "Alice"
        assert agent["success_rate"] == 1.0
        assert agent["avg_steps_taken"] == 15.0

    def test_sorted_by_success_rate(self, lb):
        # Alice: 2/2 success
        lb.submit_run(_make_success_run(scenario_id="s1", player_name="Alice"))
        lb.submit_run(_make_success_run(scenario_id="s2", player_name="Alice"))
        # Bob: 1/2 success
        lb.submit_run(_make_success_run(scenario_id="s1", player_name="Bob"))
        lb.submit_run(
            _make_run(
                scenario_id="s2",
                player_name="Bob",
                success=False,
                discovery_coverage=0.5,
            )
        )
        result = lb.get_global_results(["s1", "s2"])
        assert len(result["agents"]) == 2
        assert result["agents"][0]["agent_name"] == "Alice"

    def test_empty_scenario_list(self, lb):
        result = lb.get_global_results([])
        assert result == {"scenario_ids": [], "agents": []}


# ---------------------------------------------------------------------------
# Agent runs
# ---------------------------------------------------------------------------


class TestAgentRuns:
    """Tests for Leaderboard.get_agent_runs()."""

    def test_empty(self, lb):
        runs = lb.get_agent_runs("nobody")
        assert runs == []

    def test_returns_all_runs(self, lb):
        lb.submit_run(_make_run(scenario_id="s1", player_name="Alice"))
        lb.submit_run(_make_run(scenario_id="s2", player_name="Alice"))
        lb.submit_run(_make_run(scenario_id="s1", player_name="Bob"))
        runs = lb.get_agent_runs("Alice")
        assert len(runs) == 2

    def test_filter_by_scenario(self, lb):
        lb.submit_run(_make_run(scenario_id="s1", player_name="Alice"))
        lb.submit_run(_make_run(scenario_id="s2", player_name="Alice"))
        runs = lb.get_agent_runs("Alice", scenario_id="s1")
        assert len(runs) == 1
        assert runs[0]["scenario_id"] == "s1"

    def test_model_name_match(self, lb):
        lb.submit_run(
            _make_run(scenario_id="s1", player_name=None, model_name="GPT-4o")
        )
        runs = lb.get_agent_runs("GPT-4o")
        assert len(runs) == 1
