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
    """Phase 2 columns: confidence, fov_coverage, target_seen."""

    def test_submit_with_confidence(self, lb):
        run_id = lb.submit_run(_make_run(confidence=0.85))
        run = lb.get_run(run_id)
        assert run["confidence"] == pytest.approx(0.85)

    def test_confidence_null_by_default(self, lb):
        run_id = lb.submit_run(_make_run())
        run = lb.get_run(run_id)
        assert run["confidence"] is None

    def test_submit_with_fov_coverage(self, lb):
        run_id = lb.submit_run(_make_run(fov_coverage=0.42))
        run = lb.get_run(run_id)
        assert run["fov_coverage"] == pytest.approx(0.42)

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
        """Old DB without confidence/fov_coverage/target_seen is migrated."""
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
        assert "fov_coverage" in col_names
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
    confidence: float = 1.0,
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
        confidence=confidence,
        **kwargs,
    )


class TestScoringFunctions:
    """Tests for compute_score_for_run, get_best_runs_per_scenario,
    and get_global_leaderboard."""

    # ------------------------------------------------------------------
    # compute_score_for_run – success path
    # ------------------------------------------------------------------

    def test_compute_score_success_basic(self, lb):
        """Single successful run is also the best on the scenario.

        All ratios = 1.0, confidence = 1.0 → score = 100.
        Formula: (0.3×1 + 0.3×1 + 0.3×1 + 0.1×1) × 100 = 100.0
        """
        run_id = lb.submit_run(
            _make_success_run(
                scenario_id="sc_basic",
                steps_taken=10,
                distance_travelled=100.0,
                duration_s=10.0,
                confidence=1.0,
            )
        )
        run = lb.get_run(run_id)
        score = lb.compute_score_for_run(run, "sc_basic")
        assert score == pytest.approx(100.0)

    def test_compute_score_success_worse_than_best(self, lb):
        """Best run (steps=10, dist=100, t=10) inserted first.

        Worse run (steps=20, dist=200, t=20) should score < 100.
        """
        lb.submit_run(
            _make_success_run(
                scenario_id="sc_worse",
                steps_taken=10,
                distance_travelled=100.0,
                duration_s=10.0,
                confidence=1.0,
            )
        )
        worse_id = lb.submit_run(
            _make_success_run(
                scenario_id="sc_worse",
                steps_taken=20,
                distance_travelled=200.0,
                duration_s=20.0,
                confidence=1.0,
            )
        )
        worse_run = lb.get_run(worse_id)
        score = lb.compute_score_for_run(worse_run, "sc_worse")
        # Each ratio is 0.5, confidence=1 → (0.3×0.5 + 0.3×0.5 + 0.3×0.5 + 0.1×1)×100 = 55
        assert score == pytest.approx(55.0)
        assert score < 100.0

    # ------------------------------------------------------------------
    # compute_score_for_run – failure path
    # ------------------------------------------------------------------

    def test_compute_score_failure_basic(self, lb):
        """Failed run, fov_coverage=0.5, confidence=0.5, target_seen=False.

        F = -100 × [0.5×(1-0.5) + 0.5×0.5] = -100×[0.25+0.25] = -50.0
        """
        run_id = lb.submit_run(
            _make_run(
                scenario_id="sc_fail",
                success=False,
                fov_coverage=0.5,
                confidence=0.5,
                target_seen=False,
            )
        )
        run = lb.get_run(run_id)
        score = lb.compute_score_for_run(run, "sc_fail")
        assert score == pytest.approx(-50.0)

    def test_compute_score_failure_target_seen(self, lb):
        """Same failure but target_seen=True → multiply by 1.5 → -75.0."""
        run_id = lb.submit_run(
            _make_run(
                scenario_id="sc_fail_ts",
                success=False,
                fov_coverage=0.5,
                confidence=0.5,
                target_seen=True,
            )
        )
        run = lb.get_run(run_id)
        score = lb.compute_score_for_run(run, "sc_fail_ts")
        assert score == pytest.approx(-75.0)

    def test_compute_score_failure_clamped(self, lb):
        """Extreme failure values must not go below -100."""
        run_id = lb.submit_run(
            _make_run(
                scenario_id="sc_clamp",
                success=False,
                fov_coverage=0.0,
                confidence=1.0,
                target_seen=True,
            )
        )
        run = lb.get_run(run_id)
        score = lb.compute_score_for_run(run, "sc_clamp")
        # F = -100 × [0.5×1.0 + 0.5×1.0] × 1.5 = -150, clamped to -100
        assert score >= -100.0

    # ------------------------------------------------------------------
    # get_best_runs_per_scenario
    # ------------------------------------------------------------------

    def test_get_best_runs_per_scenario(self, lb):
        """One failed + one successful run on same scenario.

        Best run should be the successful one.
        """
        lb.submit_run(_make_run(scenario_id="sc_best", success=False, steps_taken=5))
        good_id = lb.submit_run(
            _make_success_run(
                scenario_id="sc_best",
                steps_taken=10,
                distance_travelled=100.0,
            )
        )
        best = lb.get_best_runs_per_scenario(["sc_best"])
        assert "sc_best" in best
        assert best["sc_best"]["id"] == good_id
        assert best["sc_best"]["success"] is True

    def test_get_best_runs_per_scenario_empty(self, lb):
        """No runs at all → empty dict returned."""
        best = lb.get_best_runs_per_scenario(["sc_none"])
        assert best == {}

    def test_get_best_runs_per_scenario_selects_fewer_steps(self, lb):
        """Among successful runs, the one with fewer steps wins."""
        lb.submit_run(
            _make_success_run(
                scenario_id="sc_steps",
                steps_taken=20,
                distance_travelled=200.0,
            )
        )
        best_id = lb.submit_run(
            _make_success_run(
                scenario_id="sc_steps",
                steps_taken=10,
                distance_travelled=200.0,
            )
        )
        best = lb.get_best_runs_per_scenario(["sc_steps"])
        assert best["sc_steps"]["id"] == best_id

    # ------------------------------------------------------------------
    # get_global_leaderboard
    # ------------------------------------------------------------------

    def test_get_global_leaderboard_single_agent(self, lb):
        """1 agent, 1 scenario, 1 successful run → appears in leaderboard."""
        lb.submit_run(
            _make_success_run(
                scenario_id="sc_lb",
                player_name="Solo",
                steps_taken=10,
                distance_travelled=100.0,
                duration_s=10.0,
                confidence=1.0,
            )
        )
        board = lb.get_global_leaderboard(["sc_lb"])
        assert len(board) == 1
        entry = board[0]
        assert entry["agent_name"] == "Solo"
        assert entry["scenarios_attempted"] == 1
        assert entry["total_score"] == pytest.approx(100.0)
        assert len(entry["runs"]) == 1

    def test_get_global_leaderboard_multiple_agents(self, lb):
        """2 agents on same scenario → sorted by total_score DESC."""
        # Best agent: all ratios = 1 → 100 pts
        lb.submit_run(
            _make_success_run(
                scenario_id="sc_multi",
                player_name="Best",
                steps_taken=10,
                distance_travelled=100.0,
                duration_s=10.0,
                confidence=1.0,
            )
        )
        # Worse agent: steps=20, dist=200, t=20 → 55 pts
        lb.submit_run(
            _make_success_run(
                scenario_id="sc_multi",
                player_name="Worse",
                steps_taken=20,
                distance_travelled=200.0,
                duration_s=20.0,
                confidence=1.0,
            )
        )
        board = lb.get_global_leaderboard(["sc_multi"])
        assert len(board) == 2
        assert board[0]["agent_name"] == "Best"
        assert board[0]["total_score"] > board[1]["total_score"]

    def test_get_global_leaderboard_empty(self, lb):
        """No runs at all → empty leaderboard."""
        board = lb.get_global_leaderboard(["sc_empty"])
        assert board == []

    def test_get_global_leaderboard_limit(self, lb):
        """limit parameter caps number of agents returned."""
        for i in range(5):
            lb.submit_run(
                _make_success_run(
                    scenario_id="sc_limit",
                    player_name=f"Agent{i}",
                    steps_taken=10 + i,
                    distance_travelled=100.0 + i,
                    duration_s=10.0 + i,
                    confidence=1.0,
                )
            )
        board = lb.get_global_leaderboard(["sc_limit"], limit=3)
        assert len(board) == 3

    def test_get_global_leaderboard_model_name_agent(self, lb):
        """AI runs identified by model_name appear in leaderboard."""
        lb.submit_run(
            _make_success_run(
                scenario_id="sc_ai",
                player_name=None,
                model_name="GPT-4o",
                mode="ai",
                steps_taken=10,
                distance_travelled=100.0,
                duration_s=10.0,
                confidence=1.0,
            )
        )
        board = lb.get_global_leaderboard(["sc_ai"])
        assert len(board) == 1
        assert board[0]["agent_name"] == "GPT-4o"

    def test_get_global_leaderboard_excludes_other_scenarios(self, lb):
        """Runs on scenarios not in the list are excluded."""
        lb.submit_run(
            _make_success_run(
                scenario_id="sc_included",
                player_name="Agent",
                steps_taken=10,
                distance_travelled=100.0,
                duration_s=10.0,
                confidence=1.0,
            )
        )
        lb.submit_run(
            _make_success_run(
                scenario_id="sc_excluded",
                player_name="Agent",
                steps_taken=5,
                distance_travelled=50.0,
                duration_s=5.0,
                confidence=1.0,
            )
        )
        board = lb.get_global_leaderboard(["sc_included"])
        assert len(board) == 1
        entry = board[0]
        # Only 1 scenario counted
        assert entry["scenarios_attempted"] == 1

    # ------------------------------------------------------------------
    # Edge-case: None values in scoring fields
    # ------------------------------------------------------------------

    def test_compute_score_failure_confidence_none(self, lb):
        """Run with confidence=None should treat it as 0.0 in failure formula.

        F = -100 × [0.5×(1-0.5) + 0.5×0.0] = -100 × 0.25 = -25.0
        """
        run_id = lb.submit_run(
            _make_run(
                scenario_id="s1",
                mode="human",
                success=False,
                fov_coverage=0.5,
                confidence=None,
                target_seen=False,
            )
        )
        run = lb.get_run(run_id)
        score = lb.compute_score_for_run(run, "s1")
        assert score == pytest.approx(-25.0)

    def test_get_best_runs_per_scenario_tiebreak_distance(self, lb):
        """Among runs with the same min steps, shortest distance wins."""
        lb.submit_run(
            _make_run(
                scenario_id="s1",
                mode="human",
                success=True,
                steps_taken=10,
                distance_travelled=500.0,
            )
        )
        lb.submit_run(
            _make_run(
                scenario_id="s1",
                mode="human",
                success=True,
                steps_taken=10,
                distance_travelled=200.0,
            )
        )
        result = lb.get_best_runs_per_scenario(["s1"])
        assert result["s1"]["distance_travelled"] == 200.0

    def test_compute_score_success_confidence_none(self, lb):
        """confidence=None should be treated as 0.0 in success formula.

        Single run is its own reference: all ratios = 1.0, confidence = 0.0
        S = (0.3×1 + 0.3×1 + 0.3×1 + 0.1×0) × 100 = 90.0
        """
        run_id = lb.submit_run(
            _make_run(
                scenario_id="sc_conf_none_s",
                mode="human",
                success=True,
                steps_taken=10,
                distance_travelled=100.0,
                duration_s=10.0,
                confidence=None,
            )
        )
        run = lb.get_run(run_id)
        score = lb.compute_score_for_run(run, "sc_conf_none_s")
        assert score == pytest.approx(90.0)
