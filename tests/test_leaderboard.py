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
