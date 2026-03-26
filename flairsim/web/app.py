"""
FastAPI orchestrator for the FlairSim web benchmark platform.

Serves the SPA, manages simulator sessions, proxies API calls to
simulator subprocesses, and provides leaderboard endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .leaderboard import Leaderboard
from .sessions import SessionManager

logger = logging.getLogger(__name__)

# Load API key from environment / .env file.
_API_KEY: Optional[str] = None


def _load_api_key() -> Optional[str]:
    """Load FLAIRSIM_API_KEY from env or .env file."""
    key = os.environ.get("FLAIRSIM_API_KEY")
    if key:
        return key
    # Try loading from .env in project root.
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "FLAIRSIM_API_KEY":
                return v.strip()
    return None


# -------------------------------------------------------------------
# App factory
# -------------------------------------------------------------------


def create_web_app(
    scenarios_dir: str | Path,
    data_root: Optional[str | Path] = None,
    leaderboard_db: str | Path = "data/leaderboard.db",
) -> FastAPI:
    """Create the orchestrator FastAPI application.

    Parameters
    ----------
    scenarios_dir : str or Path
        Directory containing scenario YAML files.
    data_root : str or Path or None
        Root for resolving relative ``data_dir`` paths in scenarios.
    leaderboard_db : str or Path
        Path to the SQLite leaderboard database.
    """
    scenarios_dir = Path(scenarios_dir).resolve()
    data_root_path = Path(data_root).resolve() if data_root else None

    # Load API key for AI agent auth.
    global _API_KEY
    _API_KEY = _load_api_key()
    if _API_KEY:
        logger.info("API key loaded (%d chars)", len(_API_KEY))
    else:
        logger.warning("No FLAIRSIM_API_KEY found -- AI API access disabled")

    def _require_api_key(request: Request) -> None:
        """Validate Bearer token in Authorization header."""
        if not _API_KEY:
            raise HTTPException(status_code=503, detail="AI API access not configured")
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != _API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    session_mgr = SessionManager(
        scenarios_dir=scenarios_dir,
        data_root=data_root_path,
    )
    leaderboard = Leaderboard(db_path=leaderboard_db)

    # Overview images directory.
    overview_dir = Path(__file__).parent / "static" / "overviews"
    overview_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Overview generation helpers
    # ---------------------------------------------------------------

    def _generate_overview_for_scenario(scenario_id: str) -> None:
        """Spawn a temporary simulator subprocess, fetch /overview, save JPEG+bounds JSON.

        Runs synchronously (blocking).  Called during lifespan startup.
        """
        import httpx

        loader = session_mgr.scenario_loader
        try:
            scenario = loader.get(scenario_id)
        except FileNotFoundError:
            logger.warning(
                "Cannot generate overview: scenario '%s' not found", scenario_id
            )
            return

        jpeg_path = overview_dir / f"{scenario_id}.jpg"
        bounds_path = overview_dir / f"{scenario_id}.json"
        thumb_path = overview_dir / f"{scenario_id}_thumb.jpg"

        # Skip if all files already generated (overview + thumbnail).
        if jpeg_path.exists() and bounds_path.exists() and thumb_path.exists():
            logger.info(
                "Overview and thumbnail already exist for %s, skipping.", scenario_id
            )
            return

        # Build command for a temporary subprocess.
        ds = scenario.dataset
        cmd = [
            sys.executable,
            "-m",
            "flairsim.server",
            "--host",
            "127.0.0.1",
            "--port",
            "9199",  # Use a high port unlikely to conflict.
        ]

        effective_root = data_root_path or loader.data_root
        if ds.domain and ds.source in ("huggingface", "auto"):
            cmd.extend(["--domain", ds.domain])
        else:
            resolved = loader.resolve_data_dir(scenario)
            cmd.extend(["--data-dir", str(resolved)])
            if ds.domain:
                cmd.extend(["--domain", ds.domain])

        if ds.roi:
            cmd.extend(["--roi", ds.roi])

        cmd.extend(
            [
                "--scenarios-dir",
                str(scenarios_dir),
                "--scenario",
                scenario_id,
                "--data-root",
                str(effective_root),
            ]
        )

        logger.info("Spawning temp subprocess for overview of %s", scenario_id)
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for the subprocess to become ready.
            import time

            base_url = "http://127.0.0.1:9199"
            deadline = time.time() + 60.0
            ready = False
            while time.time() < deadline:
                if proc.poll() is not None:
                    stderr = (
                        proc.stderr.read().decode(errors="replace")
                        if proc.stderr
                        else ""
                    )
                    logger.error(
                        "Overview subprocess for %s exited with code %d: %s",
                        scenario_id,
                        proc.returncode,
                        stderr[:500],
                    )
                    return
                try:
                    resp = httpx.get(f"{base_url}/config", timeout=2.0)
                    if resp.status_code == 200:
                        ready = True
                        break
                except (httpx.ConnectError, httpx.ReadTimeout, OSError):
                    pass
                time.sleep(0.5)

            if not ready:
                logger.error(
                    "Overview subprocess for %s did not become ready", scenario_id
                )
                return

            # Fetch overview image (skip if already exists).
            if not jpeg_path.exists() or not bounds_path.exists():
                resp = httpx.get(f"{base_url}/overview?size=1024", timeout=30.0)
                if resp.status_code != 200:
                    logger.error(
                        "Failed to fetch overview for %s: HTTP %d",
                        scenario_id,
                        resp.status_code,
                    )
                    return

                # Save JPEG.
                jpeg_path.write_bytes(resp.content)

                # Save bounds from response headers.
                bounds_data = {
                    "x_min": float(resp.headers.get("X-Bounds-Xmin", 0)),
                    "y_min": float(resp.headers.get("X-Bounds-Ymin", 0)),
                    "x_max": float(resp.headers.get("X-Bounds-Xmax", 0)),
                    "y_max": float(resp.headers.get("X-Bounds-Ymax", 0)),
                }
                bounds_path.write_text(json.dumps(bounds_data, indent=2))
                logger.info(
                    "Generated overview for %s: %s (%d bytes)",
                    scenario_id,
                    jpeg_path,
                    len(resp.content),
                )
            else:
                logger.info(
                    "Overview already exists for %s, skipping overview fetch.",
                    scenario_id,
                )

            # Also generate start-view thumbnail by resetting the scenario.
            if not thumb_path.exists():
                try:
                    reset_resp = httpx.post(
                        f"{base_url}/reset",
                        json={"scenario_id": scenario_id},
                        timeout=30.0,
                    )
                    if reset_resp.status_code == 200:
                        reset_data = reset_resp.json()
                        img_b64 = reset_data.get("image", "")
                        if img_b64:
                            import base64
                            from PIL import Image as PILImage
                            import io

                            img_bytes = base64.b64decode(img_b64)
                            img = PILImage.open(io.BytesIO(img_bytes))
                            # Resize to thumbnail size (300x300).
                            img = img.resize((300, 300), PILImage.LANCZOS)
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=85)
                            thumb_path.write_bytes(buf.getvalue())
                            logger.info(
                                "Generated start-view thumbnail for %s (%d bytes)",
                                scenario_id,
                                len(buf.getvalue()),
                            )
                        else:
                            logger.warning(
                                "No image in reset response for %s", scenario_id
                            )
                    else:
                        logger.warning(
                            "Failed to reset for thumbnail of %s: HTTP %d",
                            scenario_id,
                            reset_resp.status_code,
                        )
                except Exception:
                    logger.warning(
                        "Could not generate thumbnail for %s",
                        scenario_id,
                        exc_info=True,
                    )

        except Exception:
            logger.exception("Error generating overview for %s", scenario_id)
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("FlairSim web platform starting")

        # Generate overview images for all scenarios (blocking at startup).
        scenario_ids = session_mgr.scenario_loader.list_ids()
        for sid in scenario_ids:
            try:
                await asyncio.to_thread(_generate_overview_for_scenario, sid)
            except Exception:
                logger.exception("Failed to generate overview for %s", sid)

        logger.info("Overview generation complete for %d scenarios", len(scenario_ids))

        # Start background idle-session reaper.
        session_mgr.start_idle_checker()

        yield
        logger.info("FlairSim web platform shutting down -- cleaning up sessions")
        await session_mgr.cleanup_all()
        leaderboard.close()

    app = FastAPI(
        title="FlairSim Benchmark",
        description="Web platform for the FlairSim drone simulator benchmark.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS -- permissive for local dev; in prod behind Cloudflare this is
    # same-origin so CORS headers are ignored by browsers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------
    # Static files (SPA)
    # ---------------------------------------------------------------

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(static_dir), html=True), name="static"
        )

        # Prevent aggressive browser caching of JS/CSS during development.
        @app.middleware("http")
        async def no_cache_static(request: Request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
            return response

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/static/index.html")

    # ---------------------------------------------------------------
    # Health / status
    # ---------------------------------------------------------------

    @app.get("/api/status")
    async def status():
        sessions = await session_mgr.list_sessions()
        return {
            "status": "ok",
            "active_sessions": len(sessions),
            "scenarios_dir": str(scenarios_dir),
        }

    # ---------------------------------------------------------------
    # Scenarios
    # ---------------------------------------------------------------

    @app.get("/api/scenarios")
    async def list_scenarios():
        scenarios = session_mgr.scenario_loader.list_scenarios()
        return {
            "scenarios": [s.to_dict() for s in scenarios],
        }

    @app.get("/api/scenarios/{scenario_id}")
    async def get_scenario(scenario_id: str):
        try:
            scenario = session_mgr.scenario_loader.get(scenario_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Scenario not found")
        return scenario.to_dict()

    # ---------------------------------------------------------------
    # Sessions
    # ---------------------------------------------------------------

    @app.post("/api/sessions")
    async def create_session(request: Request):
        body = await request.json()
        scenario_id = body.get("scenario_id")
        if not scenario_id:
            raise HTTPException(status_code=400, detail="scenario_id is required")

        mode = body.get("mode", "human")
        if mode not in ("human", "ai"):
            raise HTTPException(status_code=400, detail="mode must be 'human' or 'ai'")

        player_name = body.get("player_name")
        model_info = body.get("model_info")

        # AI sessions require a valid API key.
        if mode == "ai":
            _require_api_key(request)

        try:
            session = await session_mgr.create_session(
                scenario_id=scenario_id,
                mode=mode,
                player_name=player_name,
                model_info=model_info,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Scenario not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return session.to_dict()

    @app.get("/api/sessions")
    async def list_sessions():
        sessions = await session_mgr.list_sessions()
        return {"sessions": [s.to_dict() for s in sessions]}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = await session_mgr.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.to_dict()

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        session = await session_mgr.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        await session_mgr.destroy_session(session_id)
        return {"status": "destroyed", "session_id": session_id}

    # ---------------------------------------------------------------
    # Proxy to simulator subprocess
    # ---------------------------------------------------------------

    @app.api_route(
        "/api/sessions/{session_id}/sim/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    )
    async def proxy_to_sim(session_id: str, path: str, request: Request):
        """Forward any request to the simulator subprocess."""
        import httpx

        session = await session_mgr.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.status == "error":
            raise HTTPException(status_code=502, detail="Simulator session has errored")
        if session.status == "starting":
            raise HTTPException(status_code=503, detail="Simulator is still starting")

        target_url = f"{session.base_url}/{path}"

        # Track activity for idle-timeout.
        session_mgr.touch_session(session_id)

        # Read request body for forwarding.
        body = await request.body()

        # Forward query params.
        query_string = str(request.url.query) if request.url.query else ""
        if query_string:
            target_url = f"{target_url}?{query_string}"

        # SSE streaming for /events endpoint.
        if path == "events" and request.method == "GET":
            return await _proxy_sse(target_url, session_id)

        # Standard request forwarding.
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method=request.method,
                    url=target_url,
                    content=body,
                    headers={
                        "content-type": request.headers.get(
                            "content-type", "application/json"
                        ),
                    },
                    timeout=30.0,
                )
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Cannot connect to simulator")
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="Simulator response timed out")

        return JSONResponse(
            content=resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else resp.text,
            status_code=resp.status_code,
        )

    async def _proxy_sse(target_url: str, session_id: str) -> StreamingResponse:
        """Stream SSE events from the simulator to the client."""
        import httpx

        async def event_stream():
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream("GET", target_url, timeout=None) as resp:
                        async for line in resp.aiter_lines():
                            yield line + "\n"
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
                logger.warning("SSE proxy disconnected for session %s", session_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------------------------------------------------------
    # Leaderboard
    # ---------------------------------------------------------------

    @app.get("/api/leaderboard")
    async def get_leaderboard(
        scenario_id: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 50,
    ):
        runs = leaderboard.get_runs(
            scenario_id=scenario_id,
            mode=mode,
            limit=limit,
        )
        return {"runs": runs}

    @app.post("/api/leaderboard")
    async def submit_to_leaderboard(request: Request):
        body = await request.json()

        if "scenario_id" not in body:
            raise HTTPException(status_code=400, detail="scenario_id is required")
        if "mode" not in body:
            raise HTTPException(status_code=400, detail="mode is required")

        run_id = leaderboard.submit_run(body)
        return {"run_id": run_id}

    # ---------------------------------------------------------------
    # Agents
    # ---------------------------------------------------------------

    @app.post("/api/agents")
    async def create_agent(request: Request) -> Dict[str, Any]:
        """Create a new agent profile."""
        body = await request.json()
        name: Optional[str] = body.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        specs: Optional[Dict[str, Any]] = body.get("specs")
        try:
            leaderboard.create_agent(name, specs)
        except ValueError:
            raise HTTPException(status_code=409, detail="Agent already exists")
        return {"name": name, "created": True}

    @app.put("/api/agents/{name}")
    async def update_agent(name: str, request: Request) -> Dict[str, Any]:
        """Update an existing agent's specs."""
        body = await request.json()
        specs: Optional[Dict[str, Any]] = body.get("specs")
        try:
            leaderboard.update_agent(name, specs)
        except KeyError:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"name": name, "updated": True}

    @app.get("/api/agents/{name}")
    async def get_agent(name: str) -> Dict[str, Any]:
        """Get an agent profile by name."""
        agent = leaderboard.get_agent(name)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent

    # ---------------------------------------------------------------
    # D_min helpers — Euclidean start-to-target distance
    # ---------------------------------------------------------------

    def _compute_d_min(scenario_id: str) -> Optional[float]:
        """Compute Euclidean distance between scenario start and target.

        Returns ``None`` if the scenario cannot be loaded or the start
        coordinates are not defined.
        """
        try:
            scenario = session_mgr.scenario_loader.get(scenario_id)
        except (FileNotFoundError, ValueError):
            return None
        if scenario.start.x is None or scenario.start.y is None:
            return None
        return scenario.target.distance_to(scenario.start.x, scenario.start.y)

    def _build_d_min_map(scenario_ids: List[str]) -> Dict[str, float]:
        """Build ``{scenario_id: d_min}`` for all given scenarios."""
        d_mins: Dict[str, float] = {}
        for sid in scenario_ids:
            d = _compute_d_min(sid)
            if d is not None:
                d_mins[sid] = d
        return d_mins

    # ---------------------------------------------------------------
    # Global and per-scenario leaderboards (must be BEFORE /{run_id})
    # ---------------------------------------------------------------

    @app.get("/api/leaderboard/global")
    async def get_global_leaderboard(
        limit: int = 50, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Global cross-scenario agent ranking.

        Optional ``mode`` query parameter (``ai`` or ``human``) filters
        runs before aggregation.
        """
        scenario_ids = session_mgr.scenario_loader.list_ids()
        d_min_map = _build_d_min_map(scenario_ids)
        entries = leaderboard.get_global_leaderboard(
            scenario_ids,
            limit=limit,
            mode=mode,
            d_min_by_scenario=d_min_map,
        )
        # Return scenario IDs so frontend can build per-scenario columns.
        return {"leaderboard": entries, "scenario_ids": scenario_ids}

    @app.get("/api/leaderboard/scenario/{scenario_id}")
    async def get_scenario_leaderboard(
        scenario_id: str, mode: Optional[str] = None, limit: int = 50
    ) -> Dict[str, Any]:
        """Per-scenario scored leaderboard, sorted by score DESC.

        Optional ``mode`` query parameter (``ai`` or ``human``) filters
        runs before scoring so only the requested category is ranked.
        """
        d_min = _compute_d_min(scenario_id)

        # Fetch all runs first; apply limit after scoring so we rank correctly.
        all_runs = leaderboard.get_runs(scenario_id=scenario_id, mode=mode, limit=1000)
        scored_runs = []
        for run in all_runs:
            run_copy = dict(run)
            run_copy["score"] = leaderboard.compute_score_for_run(
                run_copy,
                scenario_id,
                d_min=d_min,
            )
            scored_runs.append(run_copy)
        scored_runs.sort(key=lambda r: r["score"], reverse=True)
        return {"scenario_id": scenario_id, "runs": scored_runs[:limit]}

    @app.get("/api/leaderboard/{run_id}/breakdown")
    async def get_run_breakdown(run_id: str) -> Dict[str, Any]:
        """Return the detailed score breakdown for a single run."""
        run = leaderboard.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        scenario_id = run["scenario_id"]
        d_min = _compute_d_min(scenario_id)
        breakdown = leaderboard.compute_score_breakdown(
            run,
            scenario_id,
            d_min=d_min,
        )
        return breakdown

    @app.get("/api/leaderboard/{run_id}")
    async def get_leaderboard_run(run_id: str) -> Dict[str, Any]:
        run = leaderboard.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.delete("/api/leaderboard/{run_id}")
    async def delete_leaderboard_run(run_id: str, request: Request):
        """Delete a leaderboard run.  Requires admin API key."""
        _require_api_key(request)
        deleted = leaderboard.delete_run(run_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Run not found")
        return Response(status_code=204)

    # Explicit submit endpoint for AI sessions (from notebooks).
    @app.post("/api/leaderboard/submit")
    async def submit_leaderboard_explicit(request: Request):
        """Submit a run to the leaderboard.

        This is the preferred endpoint for AI agents running from
        notebooks.  Accepts the same body as ``POST /api/leaderboard``
        but is more explicit in its URL.
        """
        body = await request.json()

        if "scenario_id" not in body:
            raise HTTPException(status_code=400, detail="scenario_id is required")
        if "mode" not in body:
            raise HTTPException(status_code=400, detail="mode is required")

        run_id = leaderboard.submit_run(body)
        return {"run_id": run_id}

    # ---------------------------------------------------------------
    # Scenario overview images
    # ---------------------------------------------------------------

    @app.get("/api/scenarios/{scenario_id}/overview")
    async def get_scenario_overview(scenario_id: str):
        """Return the pre-generated overview image for a scenario.

        Returns a JPEG image along with bounds metadata in headers.
        """
        jpeg_path = overview_dir / f"{scenario_id}.jpg"
        bounds_path = overview_dir / f"{scenario_id}.json"

        if not jpeg_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Overview not available for scenario '{scenario_id}'",
            )

        jpeg_bytes = jpeg_path.read_bytes()
        headers: Dict[str, str] = {}

        if bounds_path.exists():
            try:
                bounds_data = json.loads(bounds_path.read_text())
                headers["X-Bounds-Xmin"] = str(bounds_data.get("x_min", 0))
                headers["X-Bounds-Ymin"] = str(bounds_data.get("y_min", 0))
                headers["X-Bounds-Xmax"] = str(bounds_data.get("x_max", 0))
                headers["X-Bounds-Ymax"] = str(bounds_data.get("y_max", 0))
            except (json.JSONDecodeError, KeyError):
                pass

        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers=headers,
        )

    @app.get("/api/scenarios/{scenario_id}/thumbnail")
    async def get_scenario_thumbnail(scenario_id: str, size: int = 300):
        """Return a thumbnail for the scenario.

        Prefers the pre-generated start-view image (captured via POST /reset
        during startup).  Falls back to cropping the overview around the
        start position if the start-view file is not available.
        """
        from PIL import Image as PILImage
        import io

        # Check for pre-generated start-view thumbnail first.
        thumb_path = overview_dir / f"{scenario_id}_thumb.jpg"
        if thumb_path.exists():
            thumb_bytes = thumb_path.read_bytes()
            # Optionally resize if caller wants a different size.
            size = min(size, 800)
            img = PILImage.open(io.BytesIO(thumb_bytes))
            if img.size != (size, size):
                img = img.resize((size, size), PILImage.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                thumb_bytes = buf.getvalue()
            return Response(content=thumb_bytes, media_type="image/jpeg")

        # Fallback: crop from overview image.
        jpeg_path = overview_dir / f"{scenario_id}.jpg"
        bounds_path = overview_dir / f"{scenario_id}.json"

        if not jpeg_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Thumbnail not available for scenario '{scenario_id}'",
            )

        img = PILImage.open(str(jpeg_path))
        img_w, img_h = img.size

        x_min = y_min = 0.0
        x_max = float(img_w)
        y_max = float(img_h)
        if bounds_path.exists():
            try:
                bd = json.loads(bounds_path.read_text())
                x_min = float(bd.get("x_min", 0))
                y_min = float(bd.get("y_min", 0))
                x_max = float(bd.get("x_max", img_w))
                y_max = float(bd.get("y_max", img_h))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        try:
            scenario = session_mgr.scenario_loader.get(scenario_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Scenario not found")

        start_x = scenario.start.x
        start_y = scenario.start.y

        world_w = x_max - x_min if x_max != x_min else 1.0
        world_h = y_max - y_min if y_max != y_min else 1.0
        px = (start_x - x_min) / world_w * img_w
        py = (1.0 - (start_y - y_min) / world_h) * img_h

        crop_radius = int(min(img_w, img_h) * 0.25)
        cx, cy = int(px), int(py)

        left = max(0, cx - crop_radius)
        top = max(0, cy - crop_radius)
        right = min(img_w, cx + crop_radius)
        bottom = min(img_h, cy + crop_radius)

        side = min(right - left, bottom - top)
        if right - left > side:
            left = cx - side // 2
            right = left + side
        if bottom - top > side:
            top = cy - side // 2
            bottom = top + side

        left = max(0, left)
        top = max(0, top)
        right = min(img_w, left + side)
        bottom = min(img_h, top + side)

        cropped = img.crop((left, top, right, bottom))
        size = min(size, 800)
        cropped = cropped.resize((size, size), PILImage.LANCZOS)

        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=85)
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
        )

    return app
