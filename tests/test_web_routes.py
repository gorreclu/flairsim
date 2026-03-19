"""
Tests for :mod:`flairsim.web.app` (orchestrator routes).

Tests the FastAPI routes using httpx's AsyncClient with mocked
SessionManager and Leaderboard.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from flairsim.web.leaderboard import Leaderboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenarios_dir(tmp: Path) -> Path:
    """Write minimal scenario YAMLs and return the directory."""
    scenarios_dir = tmp / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    yaml1 = """\
scenario_id: find_target_test
name: Find Target Test
description: A test scenario
objective: find_target
max_steps: 100
environment:
  - urban
  - forest
difficulty: 2
dataset:
  domain: D099-2099
  data_dir: data/D099-2099_AERIAL_RGBI
  roi: AB-S1-01
start:
  x: 800010.0
  y: 6500020.0
  z: 100.0
target:
  x: 800030.0
  y: 6500035.0
  radius: 50.0
"""
    (scenarios_dir / "find_target_test.yaml").write_text(yaml1)
    return scenarios_dir


def _create_app(tmp_path):
    """Build the web app for testing."""
    from flairsim.web.app import create_web_app

    scenarios_dir = _make_scenarios_dir(tmp_path)
    db_path = tmp_path / "test_lb.db"

    app = create_web_app(
        scenarios_dir=str(scenarios_dir),
        data_root=str(tmp_path),
        leaderboard_db=str(db_path),
    )
    return app


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------


class TestStatusRoute:
    """Test the GET /api/status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_ok(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "active_sessions" in data


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class TestScenarioRoutes:
    """Test scenario listing and detail endpoints."""

    @pytest.mark.asyncio
    async def test_list_scenarios(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/scenarios")
            assert resp.status_code == 200
            data = resp.json()
            assert "scenarios" in data
            assert len(data["scenarios"]) >= 1
            assert data["scenarios"][0]["scenario_id"] == "find_target_test"

    @pytest.mark.asyncio
    async def test_get_scenario_found(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/scenarios/find_target_test")
            assert resp.status_code == 200
            data = resp.json()
            assert data["scenario_id"] == "find_target_test"

    @pytest.mark.asyncio
    async def test_get_scenario_not_found(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/scenarios/nonexistent")
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessionRoutes:
    """Test session CRUD endpoints (with mocked subprocess spawning)."""

    @pytest.mark.asyncio
    async def test_create_session_requires_scenario_id(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/sessions", json={})
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_session_invalid_mode(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/sessions",
                json={
                    "scenario_id": "find_target_test",
                    "mode": "invalid",
                },
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_session_nonexistent_scenario(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/sessions",
                json={
                    "scenario_id": "does_not_exist",
                },
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/sessions")
            assert resp.status_code == 200
            data = resp.json()
            assert data["sessions"] == []

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/sessions/nonexistent-uuid")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/api/sessions/nonexistent-uuid")
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


class TestLeaderboardRoutes:
    """Test leaderboard endpoints."""

    @pytest.mark.asyncio
    async def test_get_leaderboard_empty(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/leaderboard")
            assert resp.status_code == 200
            data = resp.json()
            assert data["runs"] == []

    @pytest.mark.asyncio
    async def test_submit_and_get_leaderboard(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Submit a run
            resp = await client.post(
                "/api/leaderboard",
                json={
                    "scenario_id": "find_target_test",
                    "mode": "human",
                    "success": True,
                    "steps_taken": 42,
                    "distance_travelled": 800.0,
                    "player_name": "TestPlayer",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "run_id" in data
            run_id = data["run_id"]

            # Get the run
            resp = await client.get(f"/api/leaderboard/{run_id}")
            assert resp.status_code == 200
            run = resp.json()
            assert run["player_name"] == "TestPlayer"
            assert run["success"] is True

            # List leaderboard
            resp = await client.get("/api/leaderboard")
            assert resp.status_code == 200
            runs = resp.json()["runs"]
            assert len(runs) == 1

    @pytest.mark.asyncio
    async def test_submit_requires_scenario_id(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/leaderboard",
                json={
                    "mode": "human",
                },
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_requires_mode(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/leaderboard",
                json={
                    "scenario_id": "find_target_test",
                },
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_get_leaderboard_run_not_found(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/leaderboard/nonexistent-uuid")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_leaderboard_filter_by_scenario(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/leaderboard",
                json={
                    "scenario_id": "s1",
                    "mode": "human",
                    "player_name": "A",
                },
            )
            await client.post(
                "/api/leaderboard",
                json={
                    "scenario_id": "s2",
                    "mode": "human",
                    "player_name": "B",
                },
            )

            resp = await client.get("/api/leaderboard?scenario_id=s1")
            assert resp.status_code == 200
            runs = resp.json()["runs"]
            assert len(runs) == 1
            assert runs[0]["scenario_id"] == "s1"

    @pytest.mark.asyncio
    async def test_leaderboard_filter_by_mode(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/leaderboard",
                json={
                    "scenario_id": "s1",
                    "mode": "human",
                    "player_name": "A",
                },
            )
            await client.post(
                "/api/leaderboard",
                json={
                    "scenario_id": "s1",
                    "mode": "ai",
                    "player_name": "B",
                },
            )

            resp = await client.get("/api/leaderboard?mode=ai")
            assert resp.status_code == 200
            runs = resp.json()["runs"]
            assert len(runs) == 1
            assert runs[0]["mode"] == "ai"


# ---------------------------------------------------------------------------
# Static files / redirect
# ---------------------------------------------------------------------------


class TestStaticAndRedirect:
    """Test that the root redirects and static files are served."""

    @pytest.mark.asyncio
    async def test_root_redirects_to_static(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as client:
            resp = await client.get("/")
            assert resp.status_code in (301, 302, 307)
            assert "/static/index.html" in resp.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_static_index_served(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/static/index.html")
            assert resp.status_code == 200
            assert "FlairSim" in resp.text

    @pytest.mark.asyncio
    async def test_static_css_served(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/static/style.css")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_static_js_served(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/static/app.js")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# New Phase 1.5 routes
# ---------------------------------------------------------------------------


class TestAISessionValidation:
    """Test API key auth and session creation validation."""

    @pytest.mark.asyncio
    async def test_ai_session_requires_api_key(self, tmp_path):
        """POST /api/sessions with mode=ai but no Bearer token -> 401."""
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/sessions",
                json={
                    "scenario_id": "find_target_test",
                    "mode": "ai",
                },
            )
            # Should be 401 (missing API key) or 503 (API not configured)
            assert resp.status_code in (401, 503)

    @pytest.mark.asyncio
    async def test_ai_session_wrong_api_key(self, tmp_path):
        """POST /api/sessions with mode=ai and wrong Bearer token -> 401."""
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/sessions",
                json={
                    "scenario_id": "find_target_test",
                    "mode": "ai",
                },
                headers={"Authorization": "Bearer wrong-key-123"},
            )
            assert resp.status_code in (401, 503)

    @pytest.mark.asyncio
    async def test_ai_session_with_valid_key_accepted(self, tmp_path):
        """POST /api/sessions with mode=ai and valid Bearer token -> not 401."""
        import os

        # Set the API key for this test
        test_key = "test-api-key-12345"
        os.environ["FLAIRSIM_API_KEY"] = test_key
        try:
            app = _create_app(tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/sessions",
                    json={
                        "scenario_id": "find_target_test",
                        "mode": "ai",
                    },
                    headers={"Authorization": f"Bearer {test_key}"},
                )
                # Should not be 401 (may be 500 if subprocess can't start, but auth passed).
                assert resp.status_code != 401
        finally:
            os.environ.pop("FLAIRSIM_API_KEY", None)

    @pytest.mark.asyncio
    async def test_human_session_no_api_key_ok(self, tmp_path):
        """POST /api/sessions with mode=human and no API key -> not a 401."""
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/sessions",
                json={
                    "scenario_id": "find_target_test",
                    "mode": "human",
                },
            )
            # Should not be 401 (may fail at subprocess spawn, but not auth).
            assert resp.status_code != 401


class TestExplicitSubmitEndpoint:
    """Test the POST /api/leaderboard/submit endpoint (for AI notebooks)."""

    @pytest.mark.asyncio
    async def test_submit_explicit_endpoint(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/leaderboard/submit",
                json={
                    "scenario_id": "find_target_test",
                    "mode": "ai",
                    "success": True,
                    "steps_taken": 10,
                    "distance_travelled": 200.0,
                    "player_name": "AI-Agent",
                    "model_info": {"model_name": "GPT-4o"},
                    "steps_detail": [{"dx": 1.0, "reason": "test"}],
                    "score_final": 88.5,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "run_id" in data

            # Verify the run was stored correctly.
            run_resp = await client.get(f"/api/leaderboard/{data['run_id']}")
            assert run_resp.status_code == 200
            run = run_resp.json()
            assert run["player_name"] == "AI-Agent"
            assert run["model_info"]["model_name"] == "GPT-4o"
            assert run["steps_detail"] == [{"dx": 1.0, "reason": "test"}]
            assert run["score_final"] == 88.5

    @pytest.mark.asyncio
    async def test_submit_explicit_requires_scenario_id(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/leaderboard/submit",
                json={"mode": "ai"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_explicit_requires_mode(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/leaderboard/submit",
                json={"scenario_id": "find_target_test"},
            )
            assert resp.status_code == 400


class TestOverviewEndpoint:
    """Test the GET /api/scenarios/{scenario_id}/overview endpoint."""

    @pytest.mark.asyncio
    async def test_overview_not_found(self, tmp_path):
        """GET /api/scenarios/{id}/overview for non-existent scenario -> 404."""
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/scenarios/nonexistent_scenario/overview")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_overview_no_image_generated(self, tmp_path):
        """GET /api/scenarios/{id}/overview when no image exists -> 404."""
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # The test scenario exists but no overview was generated (no real data).
            resp = await client.get("/api/scenarios/find_target_test/overview")
            assert resp.status_code == 404


class TestScenarioNewFields:
    """Test that scenario endpoints return environment and difficulty."""

    @pytest.mark.asyncio
    async def test_scenario_list_has_environment_and_difficulty(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/scenarios")
            assert resp.status_code == 200
            data = resp.json()
            scenario = data["scenarios"][0]
            assert "environment" in scenario
            assert "difficulty" in scenario
            assert scenario["environment"] == ["urban", "forest"]
            assert scenario["difficulty"] == 2

    @pytest.mark.asyncio
    async def test_scenario_detail_has_environment_and_difficulty(self, tmp_path):
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/scenarios/find_target_test")
            assert resp.status_code == 200
            data = resp.json()
            assert data["environment"] == ["urban", "forest"]
            assert data["difficulty"] == 2
