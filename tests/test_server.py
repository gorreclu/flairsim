"""
Tests for :mod:`flairsim.server` -- the HTTP REST API.

Uses the FastAPI test client (httpx) against a simulator backed by a
synthetic 3x3 tile grid, identical to the one used in
``test_simulator.py``.  No real data or network is needed.

SSE tests require a real uvicorn server (``httpx.ASGITransport`` does
not support streaming responses).
"""

from __future__ import annotations

import asyncio
import base64
import json as _json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Tuple

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from flairsim.server.app import create_app

# ---------------------------------------------------------------------------
# Synthetic tile grid (same constants as test_simulator.py)
# ---------------------------------------------------------------------------

TILE_PX = 64
TILE_GSD = 0.2
TILE_GROUND = TILE_PX * TILE_GSD  # 12.8 m
N_BANDS = 4
GRID_ROWS = 3
GRID_COLS = 3
ROI = "AB-S1-01"
DOMAIN = "D099-2099"
SENSOR_TYPE = "AERIAL_RGBI"

ORIGIN_X = 800_000.0
ORIGIN_Y = 6_500_000.0 + GRID_ROWS * TILE_GROUND


def _tile_bounds(row: int, col: int) -> Tuple[float, float, float, float]:
    x_min = ORIGIN_X + col * TILE_GROUND
    x_max = x_min + TILE_GROUND
    y_max = ORIGIN_Y - row * TILE_GROUND
    y_min = y_max - TILE_GROUND
    return x_min, y_min, x_max, y_max


def _write_tile(directory: Path, row: int, col: int) -> Path:
    filename = f"{DOMAIN}_{SENSOR_TYPE}_{ROI}_{row}-{col}.tif"
    roi_dir = directory / ROI
    roi_dir.mkdir(parents=True, exist_ok=True)
    filepath = roi_dir / filename

    x_min, y_min, x_max, y_max = _tile_bounds(row, col)
    transform = from_bounds(x_min, y_min, x_max, y_max, TILE_PX, TILE_PX)
    data = np.full((N_BANDS, TILE_PX, TILE_PX), row * 30, dtype=np.uint8)
    data[1] = col * 30
    data[3] = 128

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=TILE_PX,
        width=TILE_PX,
        count=N_BANDS,
        dtype="uint8",
        crs="EPSG:2154",
        transform=transform,
    ) as dst:
        dst.write(data)
    return filepath


@pytest.fixture(scope="module")
def tile_dir():
    """Create a temporary directory with a 3x3 synthetic tile grid."""
    tmpdir = Path(tempfile.mkdtemp(prefix="flairsim_server_test_"))
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            _write_tile(tmpdir, row, col)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def client(tile_dir):
    """FastAPI test client backed by the synthetic tile grid."""
    from fastapi.testclient import TestClient

    from flairsim.drone.camera import CameraConfig
    from flairsim.drone.drone import DroneConfig

    app = create_app(
        data_dir=tile_dir,
        roi=ROI,
        max_steps=50,
        drone_config=DroneConfig(z_min=1.0, z_max=100.0, default_altitude=10.0),
        camera_config=CameraConfig(fov_deg=90.0, image_size=32),
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /config
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for the /config endpoint."""

    def test_config_returns_200(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200

    def test_config_structure(self, client):
        data = client.get("/config").json()
        assert "data_dir" in data
        assert "roi" in data
        assert "map_bounds" in data
        assert "drone" in data
        assert "camera" in data
        assert "max_steps" in data

    def test_config_roi(self, client):
        data = client.get("/config").json()
        assert data["roi"] == ROI

    def test_config_max_steps(self, client):
        data = client.get("/config").json()
        assert data["max_steps"] == 50

    def test_config_map_bounds_valid(self, client):
        bounds = client.get("/config").json()["map_bounds"]
        assert bounds["width"] > 0
        assert bounds["height"] > 0
        assert bounds["x_max"] > bounds["x_min"]
        assert bounds["y_max"] > bounds["y_min"]

    def test_config_drone_params(self, client):
        drone = client.get("/config").json()["drone"]
        assert drone["z_min"] == 1.0
        assert drone["z_max"] == 100.0
        assert drone["default_altitude"] == 10.0

    def test_config_camera_params(self, client):
        camera = client.get("/config").json()["camera"]
        assert camera["fov_deg"] == 90.0
        assert camera["image_size"] == 32


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for the /reset endpoint."""

    def test_reset_default(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 200

    def test_reset_returns_observation(self, client):
        data = client.post("/reset").json()
        assert data["step"] == 0
        assert data["done"] is False
        assert data["result"] is None
        assert "drone_state" in data
        assert "image_base64" in data

    def test_reset_with_position(self, client):
        bounds = client.get("/config").json()["map_bounds"]
        cx = (bounds["x_min"] + bounds["x_max"]) / 2
        cy = (bounds["y_min"] + bounds["y_max"]) / 2

        data = client.post("/reset", json={"x": cx, "y": cy, "z": 20.0}).json()
        state = data["drone_state"]
        assert abs(state["x"] - cx) < 0.1
        assert abs(state["y"] - cy) < 0.1
        assert abs(state["z"] - 20.0) < 0.1

    def test_reset_image_is_valid_png(self, client):
        data = client.post("/reset").json()
        raw = base64.b64decode(data["image_base64"])
        # PNG magic bytes
        assert raw[:4] == b"\x89PNG"

    def test_reset_image_dimensions(self, client):
        data = client.post("/reset").json()
        assert data["image_width"] == 32
        assert data["image_height"] == 32

    def test_reset_ground_values(self, client):
        data = client.post("/reset").json()
        assert data["ground_footprint"] > 0
        assert data["ground_resolution"] > 0

    def test_reset_clears_previous_episode(self, client):
        # Run a few steps
        client.post("/reset")
        client.post("/step", json={"dx": 1.0})
        client.post("/step", json={"dx": 1.0})

        # Reset again
        data = client.post("/reset").json()
        assert data["step"] == 0
        assert data["drone_state"]["step_count"] == 0


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------


class TestStep:
    """Tests for the /step endpoint."""

    def test_step_requires_active_episode(self, client):
        # Force episode end by stepping to max or resetting fresh
        client.post("/reset")
        # Stop episode
        client.post("/step", json={"dx": 0, "dy": 0, "dz": 0, "action_type": "stop"})

        # Now step should fail
        resp = client.post("/step", json={"dx": 1.0})
        assert resp.status_code == 409

    def test_step_move(self, client):
        client.post("/reset")
        data = client.post("/step", json={"dx": 1.0, "dy": 2.0, "dz": 0.0}).json()
        assert data["step"] == 1
        assert data["done"] is False

    def test_step_increments(self, client):
        client.post("/reset")
        for i in range(1, 4):
            data = client.post("/step", json={"dx": 0.5}).json()
            assert data["step"] == i

    def test_step_default_action_type_is_move(self, client):
        client.post("/reset")
        data = client.post("/step", json={"dx": 1.0}).json()
        assert data["done"] is False

    def test_step_found(self, client):
        client.post("/reset")
        data = client.post(
            "/step", json={"dx": 0, "dy": 0, "dz": 0, "action_type": "found"}
        ).json()
        assert data["done"] is True
        assert data["result"] is not None

    def test_step_stop(self, client):
        client.post("/reset")
        data = client.post(
            "/step", json={"dx": 0, "dy": 0, "dz": 0, "action_type": "stop"}
        ).json()
        assert data["done"] is True
        assert data["result"]["success"] is False
        assert "stopped" in data["result"]["reason"].lower()

    def test_step_invalid_action_type(self, client):
        client.post("/reset")
        resp = client.post("/step", json={"dx": 0, "action_type": "fly_backwards"})
        assert resp.status_code == 422

    def test_step_limit_reached(self, client):
        """Episode ends when max_steps is reached (50 for test config)."""
        client.post("/reset")
        data = None
        for _ in range(50):
            data = client.post("/step", json={"dx": 0.1}).json()
            if data["done"]:
                break

        assert data is not None
        assert data["done"] is True
        assert "limit" in data["result"]["reason"].lower()

    def test_step_returns_valid_image(self, client):
        client.post("/reset")
        data = client.post("/step", json={"dx": 1.0}).json()
        raw = base64.b64decode(data["image_base64"])
        assert raw[:4] == b"\x89PNG"

    def test_step_drone_position_updates(self, client):
        reset_data = client.post("/reset").json()
        x0 = reset_data["drone_state"]["x"]

        step_data = client.post("/step", json={"dx": 2.0}).json()
        x1 = step_data["drone_state"]["x"]
        assert abs(x1 - x0 - 2.0) < 0.01


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------


class TestState:
    """Tests for the /state endpoint."""

    def test_state_requires_active_episode(self, client):
        client.post("/reset")
        client.post("/step", json={"action_type": "stop"})
        resp = client.get("/state")
        assert resp.status_code == 409

    def test_state_returns_drone_info(self, client):
        client.post("/reset", json={"z": 15.0})
        data = client.get("/state").json()
        assert abs(data["z"] - 15.0) < 0.1
        assert "x" in data
        assert "y" in data
        assert "heading" in data

    def test_state_does_not_advance(self, client):
        client.post("/reset")
        state1 = client.get("/state").json()
        state2 = client.get("/state").json()
        assert state1 == state2


# ---------------------------------------------------------------------------
# GET /telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    """Tests for the /telemetry endpoint."""

    def test_telemetry_after_reset(self, client):
        client.post("/reset")
        data = client.get("/telemetry").json()
        # Initial record from reset
        assert data["total_steps"] == 1
        assert len(data["records"]) == 1

    def test_telemetry_grows_with_steps(self, client):
        client.post("/reset")
        client.post("/step", json={"dx": 1.0})
        client.post("/step", json={"dx": 2.0})

        data = client.get("/telemetry").json()
        # 1 reset + 2 steps = 3 records
        assert data["total_steps"] == 3
        assert len(data["records"]) == 3

    def test_telemetry_records_positions(self, client):
        client.post("/reset")
        client.post("/step", json={"dx": 5.0, "dy": 3.0})

        records = client.get("/telemetry").json()["records"]
        last = records[-1]
        assert last["dx"] == pytest.approx(5.0, abs=0.1)
        assert last["dy"] == pytest.approx(3.0, abs=0.1)

    def test_telemetry_total_distance(self, client):
        client.post("/reset")
        client.post("/step", json={"dx": 3.0, "dy": 4.0})  # distance = 5

        data = client.get("/telemetry").json()
        assert data["total_distance"] >= 4.9

    def test_telemetry_altitude_range(self, client):
        client.post("/reset", json={"z": 20.0})
        client.post("/step", json={"dz": 5.0})

        data = client.get("/telemetry").json()
        assert data["altitude_range"] is not None
        assert data["altitude_range"][0] <= 20.0
        assert data["altitude_range"][1] >= 25.0


# ---------------------------------------------------------------------------
# Integration: full episode flow
# ---------------------------------------------------------------------------


class TestFullEpisode:
    """End-to-end: reset -> step(N) -> found/stop -> telemetry."""

    def test_full_episode_found(self, client):
        obs = client.post("/reset").json()
        assert obs["done"] is False

        for _ in range(5):
            obs = client.post("/step", json={"dx": 1.0, "dy": -0.5}).json()
            if obs["done"]:
                break

        # Declare found
        if not obs["done"]:
            obs = client.post("/step", json={"action_type": "found"}).json()

        assert obs["done"] is True
        assert obs["result"] is not None

        # Check telemetry
        telem = client.get("/telemetry").json()
        assert telem["total_steps"] > 1
        assert telem["total_distance"] > 0

    def test_full_episode_stop(self, client):
        client.post("/reset")
        client.post("/step", json={"dx": 2.0})
        obs = client.post("/step", json={"action_type": "stop"}).json()
        assert obs["done"] is True
        assert obs["result"]["success"] is False

    def test_multiple_episodes(self, client):
        """Reset should properly clear state between episodes."""
        # Episode 1
        client.post("/reset")
        for _ in range(3):
            client.post("/step", json={"dx": 1.0})
        client.post("/step", json={"action_type": "stop"})

        # Episode 2
        obs = client.post("/reset").json()
        assert obs["step"] == 0
        telem = client.get("/telemetry").json()
        assert telem["total_steps"] == 1  # only reset record

    def test_config_reflects_running_state(self, client):
        # Before reset
        client.post("/reset")
        client.post("/step", json={"action_type": "stop"})
        cfg1 = client.get("/config").json()
        assert cfg1["is_running"] is False

        # After reset
        client.post("/reset")
        cfg2 = client.get("/config").json()
        assert cfg2["is_running"] is True


# ---------------------------------------------------------------------------
# GET /events (Server-Sent Events)
# ---------------------------------------------------------------------------


def _parse_sse_events(raw_text: str) -> list[dict]:
    """Parse raw SSE text into a list of event dicts.

    Each event has ``"type"`` and ``"data"`` keys.  Comments and
    keep-alive lines are ignored.

    Parameters
    ----------
    raw_text : str
        Raw SSE text (``event:`` / ``data:`` / blank-line blocks).

    Returns
    -------
    list[dict]
        Parsed events with ``type`` and ``data`` (parsed JSON).
    """
    events: list[dict] = []
    current_type = ""
    current_data = ""
    for line in raw_text.splitlines():
        if line.startswith("event:"):
            current_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:") :].strip()
        elif line == "" and current_data:
            events.append(
                {
                    "type": current_type,
                    "data": _json.loads(current_data),
                }
            )
            current_type = ""
            current_data = ""
    return events


@pytest.fixture(scope="module")
def sse_server(tile_dir):
    """Start a real uvicorn server for SSE tests.

    SSE requires true streaming which ``httpx.ASGITransport`` does
    not support.  This fixture spins up a real HTTP server on a
    random free port and yields the base URL.
    """
    import socket

    import uvicorn

    from flairsim.drone.camera import CameraConfig
    from flairsim.drone.drone import DroneConfig

    app = create_app(
        data_dir=tile_dir,
        roi=ROI,
        max_steps=50,
        drone_config=DroneConfig(
            z_min=1.0,
            z_max=100.0,
            default_altitude=10.0,
        ),
        camera_config=CameraConfig(fov_deg=90.0, image_size=32),
    )

    # Find a free port.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1.5)  # Wait for server to start.
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5.0)


class TestSSEEvents:
    """Tests for the /events SSE streaming endpoint.

    SSE requires a real HTTP server because ``httpx.ASGITransport``
    buffers responses and cannot handle streaming.  The tests use
    a uvicorn server started in a background thread (see the
    ``sse_server`` fixture) and ``httpx.AsyncClient`` to subscribe
    to the SSE stream concurrently with trigger requests.
    """

    @pytest.mark.asyncio
    async def test_events_returns_200_event_stream(self, sse_server):
        """GET /events returns 200 with text/event-stream."""
        timeout = httpx.Timeout(5.0, read=None)
        async with httpx.AsyncClient(base_url=sse_server, timeout=timeout) as ac:
            async with ac.stream("GET", "/events") as resp:
                assert resp.status_code == 200
                ct = resp.headers.get("content-type", "")
                assert "text/event-stream" in ct

    @pytest.mark.asyncio
    async def test_events_sends_connected_comment(self, sse_server):
        """SSE stream starts with a ``: connected`` comment."""
        timeout = httpx.Timeout(5.0, read=None)
        async with httpx.AsyncClient(base_url=sse_server, timeout=timeout) as ac:
            async with ac.stream("GET", "/events") as resp:
                first = await resp.aiter_text().__anext__()
                assert "connected" in first

    @pytest.mark.asyncio
    async def test_events_pushes_on_reset(self, sse_server):
        """POST /reset pushes an observation event to SSE."""
        timeout = httpx.Timeout(5.0, read=None)
        collected: list[str] = []

        async with httpx.AsyncClient(base_url=sse_server, timeout=timeout) as ac:

            async def _subscribe():
                async with ac.stream("GET", "/events") as resp:
                    async for chunk in resp.aiter_text():
                        collected.append(chunk)
                        if "observation" in chunk:
                            return

            async def _trigger():
                await asyncio.sleep(0.5)
                await ac.post("/reset")

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_subscribe()),
                    asyncio.create_task(_trigger()),
                ],
                timeout=10.0,
            )
            for t in pending:
                t.cancel()

        raw = "".join(collected)
        events = _parse_sse_events(raw)
        assert len(events) >= 1
        assert events[0]["type"] == "observation"
        assert events[0]["data"]["step"] == 0
        assert "image_base64" in events[0]["data"]

    @pytest.mark.asyncio
    async def test_events_pushes_on_step(self, sse_server):
        """POST /step pushes an observation event to SSE."""
        timeout = httpx.Timeout(5.0, read=None)
        collected: list[str] = []
        events_count = 0

        async with httpx.AsyncClient(base_url=sse_server, timeout=timeout) as ac:

            async def _subscribe():
                nonlocal events_count
                async with ac.stream("GET", "/events") as resp:
                    async for chunk in resp.aiter_text():
                        collected.append(chunk)
                        if "observation" in chunk:
                            events_count += 1
                        if events_count >= 2:
                            return

            async def _trigger():
                await asyncio.sleep(0.5)
                await ac.post("/reset")
                await asyncio.sleep(0.2)
                await ac.post("/step", json={"dx": 1.0})

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_subscribe()),
                    asyncio.create_task(_trigger()),
                ],
                timeout=10.0,
            )
            for t in pending:
                t.cancel()

        raw = "".join(collected)
        events = _parse_sse_events(raw)
        assert len(events) >= 2
        assert events[1]["data"]["step"] == 1

    @pytest.mark.asyncio
    async def test_events_observation_structure(self, sse_server):
        """SSE observation events match ObservationResponse schema."""
        timeout = httpx.Timeout(5.0, read=None)
        collected: list[str] = []

        async with httpx.AsyncClient(base_url=sse_server, timeout=timeout) as ac:

            async def _subscribe():
                async with ac.stream("GET", "/events") as resp:
                    async for chunk in resp.aiter_text():
                        collected.append(chunk)
                        if "observation" in chunk:
                            return

            async def _trigger():
                await asyncio.sleep(0.5)
                await ac.post("/reset")

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_subscribe()),
                    asyncio.create_task(_trigger()),
                ],
                timeout=10.0,
            )
            for t in pending:
                t.cancel()

        raw = "".join(collected)
        events = _parse_sse_events(raw)
        assert len(events) >= 1

        data = events[0]["data"]
        assert "step" in data
        assert "done" in data
        assert "drone_state" in data
        assert "ground_footprint" in data
        assert "ground_resolution" in data
        assert "image_base64" in data
        assert "image_width" in data
        assert "image_height" in data

        raw_img = base64.b64decode(data["image_base64"])
        assert raw_img[:4] == b"\x89PNG"

    @pytest.mark.asyncio
    async def test_events_data_matches_reset_response(self, sse_server):
        """SSE event data matches the POST /reset response."""
        timeout = httpx.Timeout(5.0, read=None)
        collected: list[str] = []
        reset_data: dict = {}

        async with httpx.AsyncClient(base_url=sse_server, timeout=timeout) as ac:

            async def _subscribe():
                async with ac.stream("GET", "/events") as resp:
                    async for chunk in resp.aiter_text():
                        collected.append(chunk)
                        if "observation" in chunk:
                            return

            async def _trigger():
                nonlocal reset_data
                await asyncio.sleep(0.5)
                resp = await ac.post("/reset")
                reset_data = resp.json()

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_subscribe()),
                    asyncio.create_task(_trigger()),
                ],
                timeout=10.0,
            )
            for t in pending:
                t.cancel()

        raw = "".join(collected)
        events = _parse_sse_events(raw)
        assert len(events) >= 1

        sse_data = events[0]["data"]
        assert sse_data["step"] == reset_data["step"]
        assert sse_data["done"] == reset_data["done"]
        ds_sse = sse_data["drone_state"]
        ds_rest = reset_data["drone_state"]
        assert ds_sse["x"] == ds_rest["x"]
        assert ds_sse["y"] == ds_rest["y"]
        assert ds_sse["z"] == ds_rest["z"]


# ---------------------------------------------------------------------------
# Phase 1.5: reason field
# ---------------------------------------------------------------------------


class TestStepReason:
    """Tests for the 'reason' field in step requests and telemetry."""

    def test_step_with_reason(self, client):
        """POST /step with a reason field is accepted."""
        client.post("/reset")
        resp = client.post(
            "/step",
            json={"dx": 1.0, "dy": 0.0, "dz": 0.0, "reason": "exploring north"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 1

    def test_step_without_reason_still_works(self, client):
        """POST /step without reason (backward compat)."""
        client.post("/reset")
        resp = client.post("/step", json={"dx": 1.0})
        assert resp.status_code == 200

    def test_reason_in_telemetry(self, client):
        """POST /step with reason, then GET /telemetry shows reason."""
        client.post("/reset")
        client.post(
            "/step",
            json={"dx": 2.0, "dy": 1.0, "reason": "moving toward target"},
        )

        data = client.get("/telemetry").json()
        records = data["records"]
        # Record 0 is the reset (no reason), record 1 is the step.
        assert len(records) >= 2
        step_record = records[-1]
        assert step_record["reason"] == "moving toward target"

    def test_reason_null_when_not_provided(self, client):
        """Telemetry records have null reason when not provided."""
        client.post("/reset")
        client.post("/step", json={"dx": 1.0})

        data = client.get("/telemetry").json()
        records = data["records"]
        # Both reset and step should have null reason.
        for rec in records:
            assert rec["reason"] is None

    def test_multiple_steps_with_reasons(self, client):
        """Multiple steps each with different reasons are tracked."""
        client.post("/reset")
        client.post("/step", json={"dx": 1.0, "reason": "step one"})
        client.post("/step", json={"dx": 1.0, "reason": "step two"})
        client.post("/step", json={"dx": 1.0})  # No reason.

        data = client.get("/telemetry").json()
        records = data["records"]
        assert records[0]["reason"] is None  # Reset.
        assert records[1]["reason"] == "step one"
        assert records[2]["reason"] == "step two"
        assert records[3]["reason"] is None


# ---------------------------------------------------------------------------
# Phase 1.5: overview endpoint
# ---------------------------------------------------------------------------


class TestOverview:
    """Tests for the /overview endpoint."""

    def test_overview_returns_jpeg(self, client):
        """GET /overview returns a JPEG image."""
        resp = client.get("/overview")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        # JPEG magic bytes: FF D8 FF
        assert resp.content[:3] == b"\xff\xd8\xff"

    def test_overview_has_bounds_headers(self, client):
        """GET /overview returns X-Bounds-* headers."""
        resp = client.get("/overview")
        assert resp.status_code == 200
        assert "x-bounds-xmin" in resp.headers
        assert "x-bounds-ymin" in resp.headers
        assert "x-bounds-xmax" in resp.headers
        assert "x-bounds-ymax" in resp.headers
        # Check that bounds are valid numbers.
        x_min = float(resp.headers["x-bounds-xmin"])
        x_max = float(resp.headers["x-bounds-xmax"])
        assert x_max > x_min

    def test_overview_custom_size(self, client):
        """GET /overview?size=256 returns a smaller image."""
        resp = client.get("/overview?size=256")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_overview_cached(self, client):
        """Second call to /overview returns the same bytes (cached)."""
        resp1 = client.get("/overview?size=128")
        resp2 = client.get("/overview?size=128")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.content == resp2.content


# ---------------------------------------------------------------------------
# Phase 1.5: scenarios with new fields
# ---------------------------------------------------------------------------


class TestScenariosNewFields:
    """Tests for environment and difficulty fields in /scenarios (server-side)."""

    def test_scenarios_without_loader(self, client):
        """GET /scenarios without a scenario loader returns empty list."""
        resp = client.get("/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenarios"] == []

    def test_scenarios_with_loader(self, tile_dir, tmp_path):
        """GET /scenarios with a loader returns environment and difficulty."""
        from fastapi.testclient import TestClient

        from flairsim.core.scenario import ScenarioLoader
        from flairsim.drone.camera import CameraConfig
        from flairsim.drone.drone import DroneConfig

        # Write a scenario YAML.
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        (scenarios_dir / "test_sc.yaml").write_text(
            f"""\
scenario_id: test_sc
name: Test Scenario
description: A test
objective: find_target
max_steps: 20
environment:
  - rural
  - forest
difficulty: 3
dataset:
  domain: {DOMAIN}
  data_dir: "{tile_dir}"
  roi: {ROI}
start:
  x: {ORIGIN_X + 10.0}
  y: {ORIGIN_Y - 10.0}
  z: 10.0
target:
  x: {ORIGIN_X + 20.0}
  y: {ORIGIN_Y - 20.0}
  radius: 50.0
"""
        )

        loader = ScenarioLoader(scenarios_dir, data_root=tile_dir)
        app = create_app(
            data_dir=tile_dir,
            roi=ROI,
            max_steps=50,
            drone_config=DroneConfig(z_min=1.0, z_max=100.0, default_altitude=10.0),
            camera_config=CameraConfig(fov_deg=90.0, image_size=32),
            scenario_loader=loader,
        )
        tc = TestClient(app)

        resp = tc.get("/scenarios")
        assert resp.status_code == 200
        scenarios = resp.json()["scenarios"]
        assert len(scenarios) == 1
        sc = scenarios[0]
        assert sc["scenario_id"] == "test_sc"
        assert sc["environment"] == ["rural", "forest"]
        assert sc["difficulty"] == 3


# ---------------------------------------------------------------------------
# GET /snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    """Tests for the /snapshot endpoint (cached last observation)."""

    def test_snapshot_404_before_reset(self, tile_dir):
        """Before any reset, /snapshot should return 404."""
        from fastapi.testclient import TestClient
        from flairsim.drone.camera import CameraConfig
        from flairsim.drone.drone import DroneConfig

        app = create_app(
            data_dir=tile_dir,
            roi=ROI,
            max_steps=50,
            drone_config=DroneConfig(z_min=1.0, z_max=100.0, default_altitude=10.0),
            camera_config=CameraConfig(fov_deg=90.0, image_size=32),
        )
        tc = TestClient(app)
        resp = tc.get("/snapshot")
        assert resp.status_code == 404

    def test_snapshot_200_after_reset(self, client):
        """After reset, /snapshot returns 200 with a full observation."""
        client.post("/reset")
        resp = client.get("/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert "image_base64" in data
        assert "drone_state" in data
        assert "step" in data

    def test_snapshot_updates_after_step(self, client):
        """After a step, /snapshot reflects the new position."""
        client.post("/reset")
        snap_before = client.get("/snapshot").json()

        client.post("/step", json={"dx": 5.0, "dy": 0.0, "dz": 0.0})
        snap_after = client.get("/snapshot").json()

        # The drone_state should differ (moved by dx=5)
        ds_before = snap_before["drone_state"]
        ds_after = snap_after["drone_state"]
        assert (
            ds_after["x"] != ds_before["x"]
            or ds_after["step_count"] != ds_before["step_count"]
        )
