"""
Session manager for the FlairSim web platform.

Each session maps to a dedicated ``flairsim-server`` subprocess running
on a unique port.  The :class:`SessionManager` handles spawning,
health-checking, and cleanup of these subprocesses.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Port range for simulator subprocesses.
_DEFAULT_PORT_MIN = 9001
_DEFAULT_PORT_MAX = 9099

# Maximum time to wait for a subprocess to become ready.
_HEALTH_CHECK_TIMEOUT = 30.0
_HEALTH_CHECK_INTERVAL = 0.5

# Idle session timeout (seconds).  Sessions with no activity for this
# duration are automatically destroyed.
_IDLE_TIMEOUT = 180  # 3 minutes


@dataclass
class Session:
    """A single simulator session."""

    session_id: str
    scenario_id: str
    mode: str  # "human" | "ai"
    player_name: Optional[str] = None
    model_info: Optional[Dict] = None
    port: int = 0
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    status: str = "starting"  # starting | ready | running | finished | error
    created_at: str = ""
    base_url: str = ""
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.monotonic()

    def to_dict(self) -> Dict:
        """Serialise for API responses (excludes process handle)."""
        return {
            "session_id": self.session_id,
            "scenario_id": self.scenario_id,
            "mode": self.mode,
            "player_name": self.player_name,
            "model_info": self.model_info,
            "port": self.port,
            "status": self.status,
            "created_at": self.created_at,
            "base_url": self.base_url,
        }


class SessionManager:
    """Manage simulator subprocess sessions.

    Parameters
    ----------
    scenarios_dir : Path
        Directory containing scenario YAML files.
    data_root : Path or None
        Root directory for resolving relative data paths in scenarios.
    port_range : tuple of (int, int)
        Range of ports to assign to simulator subprocesses.
    """

    def __init__(
        self,
        scenarios_dir: Path,
        data_root: Optional[Path] = None,
        port_range: tuple[int, int] = (_DEFAULT_PORT_MIN, _DEFAULT_PORT_MAX),
    ) -> None:
        self._scenarios_dir = Path(scenarios_dir).resolve()
        self._data_root = Path(data_root).resolve() if data_root else None
        self._port_min, self._port_max = port_range
        self._sessions: Dict[str, Session] = {}
        self._used_ports: set[int] = set()
        self._idle_check_task: Optional[asyncio.Task] = None

        # Lazy import to avoid pulling in YAML at module level.
        from ..core.scenario import ScenarioLoader

        self._loader = ScenarioLoader(
            scenarios_dir=self._scenarios_dir,
            data_root=self._data_root,
        )
        logger.info(
            "SessionManager: scenarios_dir=%s, data_root=%s, ports=%d-%d",
            self._scenarios_dir,
            self._data_root,
            self._port_min,
            self._port_max,
        )

    # ---------------------------------------------------------------- public

    async def create_session(
        self,
        scenario_id: str,
        mode: str = "human",
        player_name: Optional[str] = None,
        model_info: Optional[Dict] = None,
    ) -> Session:
        """Create a new session and spawn a simulator subprocess.

        Parameters
        ----------
        scenario_id : str
            ID of the scenario to load.
        mode : str
            ``"human"`` or ``"ai"``.
        player_name : str or None
            Display name for the player/model.
        model_info : dict or None
            AI model metadata (model_name, provider, temperature, etc.).
            Required when ``mode="ai"``.

        Returns
        -------
        Session

        Raises
        ------
        FileNotFoundError
            If the scenario does not exist.
        RuntimeError
            If no ports are available or the subprocess fails to start.
        ValueError
            If ``mode="ai"`` but ``model_info`` is not provided.
        """
        # Validate scenario exists.
        scenario = self._loader.get(scenario_id)

        # Allocate port.
        port = self._allocate_port()
        if port is None:
            raise RuntimeError(
                f"No ports available (all {self._port_max - self._port_min + 1} "
                f"ports in use)."
            )

        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            scenario_id=scenario_id,
            mode=mode,
            player_name=player_name,
            model_info=model_info,
            port=port,
            status="starting",
            created_at=datetime.now(timezone.utc).isoformat(),
            base_url=f"http://127.0.0.1:{port}",
        )

        # Build CLI command for the simulator subprocess.
        cmd = self._build_command(scenario, port)
        logger.info("Spawning session %s: %s", session_id, " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            session.process = proc
        except Exception as exc:
            self._used_ports.discard(port)
            raise RuntimeError(f"Failed to spawn simulator subprocess: {exc}") from exc

        self._sessions[session_id] = session

        # Wait for the subprocess to become ready.
        try:
            await self._wait_for_ready(session)
            session.status = "ready"
            logger.info("Session %s ready on port %d", session_id, port)
        except Exception as exc:
            logger.error("Session %s failed health check: %s", session_id, exc)
            session.status = "error"
            await self.destroy_session(session_id)
            raise RuntimeError(f"Simulator subprocess failed to start: {exc}") from exc

        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, or ``None`` if not found."""
        session = self._sessions.get(session_id)
        if session and session.process:
            # Check if process is still alive.
            retcode = session.process.poll()
            if retcode is not None and session.status not in ("finished", "error"):
                session.status = "error"
        return session

    async def list_sessions(self) -> List[Session]:
        """List all active sessions."""
        return list(self._sessions.values())

    async def destroy_session(self, session_id: str) -> None:
        """Kill the subprocess and clean up a session."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        if session.process and session.process.poll() is None:
            logger.info("Killing session %s (port %d)", session_id, session.port)
            session.process.terminate()
            try:
                session.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                session.process.kill()
                session.process.wait(timeout=2)

        session.status = "finished"
        self._used_ports.discard(session.port)
        del self._sessions[session_id]

    async def cleanup_all(self) -> None:
        """Destroy all sessions.  Called on orchestrator shutdown."""
        # Cancel the idle-check background task if running.
        if self._idle_check_task is not None:
            self._idle_check_task.cancel()
            self._idle_check_task = None

        session_ids = list(self._sessions.keys())
        for sid in session_ids:
            try:
                await self.destroy_session(sid)
            except Exception:
                logger.exception("Error cleaning up session %s", sid)

    def touch_session(self, session_id: str) -> None:
        """Update last-activity timestamp for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.touch()

    def start_idle_checker(self) -> None:
        """Start the background task that reaps idle sessions."""
        if self._idle_check_task is None:
            self._idle_check_task = asyncio.ensure_future(self._idle_check_loop())

    async def _idle_check_loop(self) -> None:
        """Periodically destroy sessions that have been idle too long."""
        try:
            while True:
                await asyncio.sleep(30)  # check every 30 seconds
                now = time.monotonic()
                to_destroy = []
                for sid, sess in self._sessions.items():
                    if sess.status in ("finished", "error"):
                        continue
                    if now - sess.last_activity > _IDLE_TIMEOUT:
                        to_destroy.append(sid)
                for sid in to_destroy:
                    logger.info("Destroying idle session %s", sid)
                    try:
                        await self.destroy_session(sid)
                    except Exception:
                        logger.exception("Error destroying idle session %s", sid)
        except asyncio.CancelledError:
            pass

    @property
    def scenario_loader(self):
        """Access the underlying :class:`ScenarioLoader`."""
        return self._loader

    # ---------------------------------------------------------------- internal

    def _allocate_port(self) -> Optional[int]:
        """Find the next available port in the range."""
        for port in range(self._port_min, self._port_max + 1):
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        return None

    def _build_command(self, scenario, port: int) -> list[str]:
        """Build the CLI command to launch a simulator subprocess."""
        from ..core.scenario import Scenario

        cmd = [
            sys.executable,
            "-m",
            "flairsim.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

        ds = scenario.dataset

        # Determine data_dir.  If source is "huggingface" or "auto" and
        # domain is set, use --domain for auto-download.  Otherwise,
        # resolve data_dir from the loader.
        if ds.domain and ds.source in ("huggingface", "auto"):
            cmd.extend(["--domain", ds.domain])
            if ds.modalities and ds.modalities != ["AERIAL_RGBI"]:
                cmd.extend(["--modalities"] + ds.modalities)
        else:
            resolved = self._loader.resolve_data_dir(scenario)
            cmd.extend(["--data-dir", str(resolved)])
            if ds.domain:
                cmd.extend(["--domain", ds.domain])

        if ds.roi:
            cmd.extend(["--roi", ds.roi])

        # Scenario loading — pass scenarios-dir and scenario ID.
        cmd.extend(
            [
                "--scenarios-dir",
                str(self._scenarios_dir),
                "--scenario",
                scenario.scenario_id,
            ]
        )

        # Always pass --data-root so the subprocess's ScenarioLoader
        # resolves data_dir correctly.  Fall back to the loader's data_root
        # (which defaults to CWD) when the user didn't pass --data-root.
        effective_root = self._data_root or self._loader.data_root
        cmd.extend(["--data-root", str(effective_root)])

        return cmd

    async def _wait_for_ready(self, session: Session) -> None:
        """Poll the subprocess until it responds to GET /config."""
        import httpx

        deadline = asyncio.get_event_loop().time() + _HEALTH_CHECK_TIMEOUT
        url = f"{session.base_url}/config"

        while asyncio.get_event_loop().time() < deadline:
            # Check process is still alive.
            if session.process and session.process.poll() is not None:
                stderr = ""
                if session.process.stderr:
                    stderr = session.process.stderr.read().decode(errors="replace")
                raise RuntimeError(
                    f"Subprocess exited with code {session.process.returncode}. "
                    f"stderr: {stderr[:500]}"
                )

            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        return
            except (httpx.ConnectError, httpx.ReadTimeout, OSError):
                pass

            await asyncio.sleep(_HEALTH_CHECK_INTERVAL)

        raise TimeoutError(
            f"Simulator on port {session.port} did not become ready "
            f"within {_HEALTH_CHECK_TIMEOUT}s"
        )
