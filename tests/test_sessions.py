"""
Tests for :mod:`flairsim.web.sessions`.

Tests port allocation, session lifecycle, command building, and cleanup,
all using mocks to avoid spawning real subprocesses.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flairsim.core.scenario import (
    Scenario,
    ScenarioDataset,
    ScenarioStart,
    ScenarioTarget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario_yaml(tmp: Path, scenario_id: str = "test_scenario") -> Path:
    """Write a minimal scenario YAML and return the scenarios dir."""
    scenarios_dir = tmp / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""\
scenario_id: {scenario_id}
name: Test Scenario
description: A test scenario
objective: find_target
max_steps: 100
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
    (scenarios_dir / f"{scenario_id}.yaml").write_text(yaml_content)
    return scenarios_dir


def _make_session_manager(tmp: Path, scenario_id: str = "test_scenario"):
    """Create a SessionManager with a temp scenarios dir."""
    from flairsim.web.sessions import SessionManager

    scenarios_dir = _make_scenario_yaml(tmp, scenario_id)
    return SessionManager(
        scenarios_dir=scenarios_dir,
        data_root=tmp,
        port_range=(19001, 19010),
    )


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------


class TestPortAllocation:
    """Test the port pool allocator."""

    def test_allocate_returns_first_port(self, tmp_path):
        mgr = _make_session_manager(tmp_path)
        port = mgr._allocate_port()
        assert port == 19001

    def test_allocate_sequential(self, tmp_path):
        mgr = _make_session_manager(tmp_path)
        p1 = mgr._allocate_port()
        p2 = mgr._allocate_port()
        p3 = mgr._allocate_port()
        assert p1 == 19001
        assert p2 == 19002
        assert p3 == 19003

    def test_allocate_returns_none_when_exhausted(self, tmp_path):
        mgr = _make_session_manager(tmp_path)
        # Exhaust all 10 ports (19001-19010)
        for _ in range(10):
            assert mgr._allocate_port() is not None
        assert mgr._allocate_port() is None

    def test_port_recycled_after_discard(self, tmp_path):
        mgr = _make_session_manager(tmp_path)
        p1 = mgr._allocate_port()
        mgr._used_ports.discard(p1)
        p2 = mgr._allocate_port()
        assert p2 == p1  # recycled


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


class TestBuildCommand:
    """Test _build_command generates correct CLI args."""

    def test_basic_command(self, tmp_path):
        mgr = _make_session_manager(tmp_path)
        scenario = mgr.scenario_loader.get("test_scenario")
        cmd = mgr._build_command(scenario, 19001)

        assert "--port" in cmd
        assert "19001" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--scenario" in cmd
        assert "test_scenario" in cmd
        assert "--scenarios-dir" in cmd

    def test_command_includes_data_dir(self, tmp_path):
        mgr = _make_session_manager(tmp_path)
        scenario = mgr.scenario_loader.get("test_scenario")
        cmd = mgr._build_command(scenario, 19001)

        # Since source is "auto" and domain is set, should use --domain
        assert "--domain" in cmd
        assert "D099-2099" in cmd

    def test_command_with_roi(self, tmp_path):
        mgr = _make_session_manager(tmp_path)
        scenario = mgr.scenario_loader.get("test_scenario")
        cmd = mgr._build_command(scenario, 19001)

        assert "--roi" in cmd
        assert "AB-S1-01" in cmd


# ---------------------------------------------------------------------------
# Session lifecycle (mocked subprocess)
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """Test session creation and destruction with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_create_session_spawns_process(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            session = await mgr.create_session("test_scenario", mode="human")

            assert session.session_id
            assert session.scenario_id == "test_scenario"
            assert session.mode == "human"
            assert session.port >= 19001
            assert session.status == "ready"
            mock_popen.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_unknown_scenario_raises(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        with pytest.raises(FileNotFoundError):
            await mgr.create_session("nonexistent_scenario")

    @pytest.mark.asyncio
    async def test_destroy_session_kills_process(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            session = await mgr.create_session("test_scenario")
            sid = session.session_id

            assert await mgr.get_session(sid) is not None

            await mgr.destroy_session(sid)
            mock_proc.terminate.assert_called_once()
            assert await mgr.get_session(sid) is None

    @pytest.mark.asyncio
    async def test_destroy_recycles_port(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            session = await mgr.create_session("test_scenario")
            port = session.port
            sid = session.session_id

            assert port in mgr._used_ports
            await mgr.destroy_session(sid)
            assert port not in mgr._used_ports

    @pytest.mark.asyncio
    async def test_list_sessions(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            await mgr.create_session("test_scenario", mode="human")
            await mgr.create_session(
                "test_scenario",
                mode="ai",
                model_info={"model_name": "test-model"},
            )

            sessions = await mgr.list_sessions()
            assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_cleanup_all(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            await mgr.create_session("test_scenario", mode="human")
            await mgr.create_session(
                "test_scenario",
                mode="ai",
                model_info={"model_name": "test-model"},
            )

            await mgr.cleanup_all()
            sessions = await mgr.list_sessions()
            assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_get_session_detects_dead_process(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            session = await mgr.create_session("test_scenario")
            assert session.status == "ready"

            # Simulate process death
            mock_proc.poll.return_value = 1
            updated = await mgr.get_session(session.session_id)
            assert updated.status == "error"

    @pytest.mark.asyncio
    async def test_session_to_dict(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            session = await mgr.create_session(
                "test_scenario", mode="human", player_name="Alice"
            )
            d = session.to_dict()

            assert d["session_id"] == session.session_id
            assert d["scenario_id"] == "test_scenario"
            assert d["mode"] == "human"
            assert d["player_name"] == "Alice"
            assert d["status"] == "ready"
            assert "process" not in d  # process handle excluded

    @pytest.mark.asyncio
    async def test_no_ports_raises_runtime_error(self, tmp_path):
        """Exhaust all ports then try to create a session."""
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            # Create 10 sessions to exhaust ports
            for _ in range(10):
                await mgr.create_session("test_scenario")

            with pytest.raises(RuntimeError, match="No ports available"):
                await mgr.create_session("test_scenario")


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


class TestModelInfo:
    """Test model_info handling in sessions."""

    @pytest.mark.asyncio
    async def test_ai_mode_without_model_info_ok(self, tmp_path):
        """AI sessions no longer require model_info (API key auth is in web layer)."""
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            session = await mgr.create_session("test_scenario", mode="ai")
            assert session.mode == "ai"
            assert session.model_info is None

    @pytest.mark.asyncio
    async def test_ai_session_stores_model_info(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        model_info = {
            "model_name": "GPT-4o",
            "provider": "OpenAI",
            "temperature": 0.7,
        }

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            session = await mgr.create_session(
                "test_scenario", mode="ai", model_info=model_info
            )
            assert session.model_info == model_info
            assert session.mode == "ai"

    @pytest.mark.asyncio
    async def test_model_info_in_to_dict(self, tmp_path):
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        model_info = {"model_name": "Claude"}

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            session = await mgr.create_session(
                "test_scenario", mode="ai", model_info=model_info
            )
            d = session.to_dict()
            assert d["model_info"] == model_info

    @pytest.mark.asyncio
    async def test_human_mode_no_model_info(self, tmp_path):
        """Human sessions do not require model_info."""
        mgr = _make_session_manager(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
        ):
            session = await mgr.create_session("test_scenario", mode="human")
            assert session.model_info is None
            assert session.to_dict()["model_info"] is None
