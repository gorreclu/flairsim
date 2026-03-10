"""
FastAPI application exposing the FlairSimulator as a REST API.

Endpoints
---------
POST /reset
    Start a new episode.  Returns initial observation.
POST /step
    Send an action, receive next observation.
GET  /state
    Current drone state without advancing.
GET  /telemetry
    Full flight log for the current episode.
GET  /config
    Simulator configuration and map bounds.
GET  /events
    Server-Sent Events stream of observations (pushed on reset/step).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..core.action import Action, ActionType
from ..core.simulator import FlairSimulator, SimulatorConfig
from ..drone.camera import CameraConfig
from ..drone.drone import DroneConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Body for POST /reset."""

    x: Optional[float] = Field(
        None, description="Start easting (m). None = map centre."
    )
    y: Optional[float] = Field(
        None, description="Start northing (m). None = map centre."
    )
    z: Optional[float] = Field(None, description="Start altitude (m). None = default.")


class StepRequest(BaseModel):
    """Body for POST /step."""

    dx: float = Field(0.0, description="Eastward displacement (m).")
    dy: float = Field(0.0, description="Northward displacement (m).")
    dz: float = Field(0.0, description="Upward displacement (m).")
    action_type: str = Field(
        "move", description="Action type: 'move', 'found', or 'stop'."
    )


class DroneStateResponse(BaseModel):
    """Drone state snapshot."""

    x: float
    y: float
    z: float
    heading: float
    step_count: int
    total_distance: float


class EpisodeResultResponse(BaseModel):
    """Episode outcome (only present when done=True)."""

    success: bool
    reason: str
    steps_taken: int
    distance_travelled: float


class ObservationResponse(BaseModel):
    """Full observation returned by /reset and /step."""

    step: int
    done: bool
    drone_state: DroneStateResponse
    ground_footprint: float
    ground_resolution: float
    image_base64: str = Field(description="PNG-encoded image in base64.")
    image_width: int
    image_height: int
    result: Optional[EpisodeResultResponse] = None
    metadata: Dict[str, Any] = {}


class TelemetryRecordResponse(BaseModel):
    """Single telemetry record."""

    step: int
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    ground_footprint: float
    was_clipped: bool


class TelemetryResponse(BaseModel):
    """Full flight log."""

    total_steps: int
    total_distance: float
    clips_count: int
    altitude_range: Optional[List[float]] = None
    records: List[TelemetryRecordResponse]


class MapBoundsResponse(BaseModel):
    """Spatial extent of the loaded map."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float


class ConfigResponse(BaseModel):
    """Simulator configuration."""

    data_dir: str
    roi: str
    n_tiles: int
    pixel_size_m: float
    tile_ground_size: float
    map_bounds: MapBoundsResponse
    drone: Dict[str, Any]
    camera: Dict[str, Any]
    max_steps: int
    is_running: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image_png(image: np.ndarray) -> str:
    """Encode a (bands, H, W) array as base64 PNG.

    Converts to RGB uint8 (H, W, 3) before encoding.
    """
    if image.ndim == 3 and image.shape[0] >= 3:
        rgb = np.transpose(image[:3], (1, 2, 0))
    elif image.ndim == 3 and image.shape[0] == 1:
        band = image[0]
        rgb = np.stack([band, band, band], axis=-1)
    elif image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image

    rgb = rgb.astype(np.uint8)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _obs_to_response(obs) -> ObservationResponse:
    """Convert an Observation to a serialisable response."""
    state = obs.drone_state
    result = None
    if obs.result is not None:
        result = EpisodeResultResponse(
            success=obs.result.success,
            reason=obs.result.reason,
            steps_taken=obs.result.steps_taken,
            distance_travelled=obs.result.distance_travelled,
        )

    return ObservationResponse(
        step=obs.step,
        done=obs.done,
        drone_state=DroneStateResponse(
            x=state.x,
            y=state.y,
            z=state.z,
            heading=state.heading,
            step_count=state.step_count,
            total_distance=state.total_distance,
        ),
        ground_footprint=obs.ground_footprint,
        ground_resolution=obs.ground_resolution,
        image_base64=_encode_image_png(obs.image),
        image_width=obs.image.shape[-1],
        image_height=obs.image.shape[-2],
        result=result,
        metadata={k: str(v) for k, v in obs.metadata.items()},
    )


def _parse_action_type(raw: str) -> ActionType:
    """Parse action type string to enum, raising HTTPException on invalid."""
    try:
        return ActionType(raw.lower())
    except ValueError:
        valid = [t.value for t in ActionType]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_type '{raw}'. Must be one of: {valid}",
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    data_dir: str | Path,
    *,
    roi: Optional[str] = None,
    max_steps: int = 500,
    drone_config: Optional[DroneConfig] = None,
    camera_config: Optional[CameraConfig] = None,
    preload_tiles: bool = True,
) -> FastAPI:
    """Create a FastAPI application wrapping a FlairSimulator.

    Parameters
    ----------
    data_dir : str or Path
        Path to FLAIR-HUB data directory.
    roi : str or None
        ROI to load.  None auto-selects the largest.
    max_steps : int
        Maximum steps per episode.
    drone_config : DroneConfig or None
        Drone physical limits.
    camera_config : CameraConfig or None
        Camera sensor parameters.
    preload_tiles : bool
        Load all tiles into RAM at start.

    Returns
    -------
    FastAPI
        The configured application instance.
    """
    config = SimulatorConfig(
        drone_config=drone_config,
        camera_config=camera_config,
        max_steps=max_steps,
        roi=roi,
        preload_tiles=preload_tiles,
    )

    sim = FlairSimulator(data_dir=data_dir, config=config)

    # SSE subscriber management.
    # Each connected viewer gets its own asyncio.Queue.  When an
    # observation is produced (by /reset or /step), it is serialised
    # once and pushed to every subscriber queue.
    #
    # Thread-safety note: The ``/reset`` and ``/step`` endpoints are
    # ``async def`` so they run directly on the event loop.  This
    # lets ``_broadcast`` use ``put_nowait`` safely without
    # cross-thread concerns.  The actual simulator calls are
    # offloaded to a thread via ``asyncio.to_thread`` to avoid
    # blocking the loop during image rendering.
    _subscribers: Set[asyncio.Queue[str]] = set()

    def _broadcast(obs_response: ObservationResponse) -> None:
        """Push a serialised observation to all SSE subscribers.

        Must be called from the event-loop thread (i.e. from an
        ``async def`` endpoint).
        """
        if not _subscribers:
            return
        payload = obs_response.model_dump_json()
        dead: list[asyncio.Queue[str]] = []
        for q in list(_subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Slow consumer -- drop oldest event to keep up.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(q)
        for q in dead:
            _subscribers.discard(q)

    app = FastAPI(
        title="FlairSim Drone Simulator",
        description=(
            "REST API for piloting a simulated drone over "
            "FLAIR-HUB aerial imagery.  An external agent sends "
            "actions and receives observations (image + telemetry)."
        ),
        version="0.1.0",
    )

    # ---------------------------------------------------------------- routes

    @app.post("/reset", response_model=ObservationResponse)
    async def reset(body: Optional[ResetRequest] = None):
        """Start a new episode.

        Resets the drone and returns the initial observation.
        If no body is provided, the drone starts at the map centre
        with the default altitude.
        """
        if body is None:
            body = ResetRequest()

        obs = await asyncio.to_thread(sim.reset, x=body.x, y=body.y, z=body.z)
        logger.info(
            "Episode reset at (%.1f, %.1f, %.1f)",
            obs.position[0],
            obs.position[1],
            obs.position[2],
        )
        response = _obs_to_response(obs)
        _broadcast(response)
        return response

    @app.post("/step", response_model=ObservationResponse)
    async def step(body: StepRequest):
        """Advance simulation by one step.

        Applies the given displacement and returns the new observation
        with the updated image and telemetry.
        """
        if not sim.is_running:
            raise HTTPException(
                status_code=409,
                detail="No active episode. Call POST /reset first.",
            )

        action_type = _parse_action_type(body.action_type)
        action = Action(dx=body.dx, dy=body.dy, dz=body.dz, action_type=action_type)
        obs = await asyncio.to_thread(sim.step, action)
        response = _obs_to_response(obs)
        _broadcast(response)
        return response

    @app.get("/state", response_model=DroneStateResponse)
    def get_state():
        """Get current drone state without advancing the simulation."""
        if not sim.is_running:
            raise HTTPException(
                status_code=409,
                detail="No active episode. Call POST /reset first.",
            )

        state = sim.drone.state
        return DroneStateResponse(
            x=state.x,
            y=state.y,
            z=state.z,
            heading=state.heading,
            step_count=state.step_count,
            total_distance=state.total_distance,
        )

    @app.get("/telemetry", response_model=TelemetryResponse)
    def get_telemetry():
        """Get the full flight log for the current episode."""
        log = sim.flight_log
        alt = log.altitude_range
        return TelemetryResponse(
            total_steps=log.total_steps,
            total_distance=round(log.total_distance, 2),
            clips_count=log.clips_count,
            altitude_range=list(alt) if alt else None,
            records=[
                TelemetryRecordResponse(
                    step=r.step,
                    x=r.x,
                    y=r.y,
                    z=r.z,
                    dx=r.dx,
                    dy=r.dy,
                    dz=r.dz,
                    ground_footprint=r.ground_footprint,
                    was_clipped=r.was_clipped,
                )
                for r in log.records
            ],
        )

    @app.get("/config", response_model=ConfigResponse)
    def get_config():
        """Get simulator configuration and map information."""
        bounds = sim.map_bounds
        drone_cfg = sim.drone.config
        cam_cfg = sim.camera.config

        return ConfigResponse(
            data_dir=str(sim._data_dir),
            roi=sim.map_manager.roi_name,
            n_tiles=sim.map_manager.n_tiles_loaded,
            pixel_size_m=sim.map_manager.pixel_size_m,
            tile_ground_size=sim.map_manager.tile_ground_size,
            map_bounds=MapBoundsResponse(
                x_min=bounds.x_min,
                y_min=bounds.y_min,
                x_max=bounds.x_max,
                y_max=bounds.y_max,
                width=bounds.width,
                height=bounds.height,
            ),
            drone={
                "z_min": drone_cfg.z_min,
                "z_max": drone_cfg.z_max,
                "default_altitude": drone_cfg.default_altitude,
                "max_step_distance": drone_cfg.max_step_distance,
            },
            camera={
                "fov_deg": cam_cfg.fov_deg,
                "image_size": cam_cfg.image_size,
            },
            max_steps=sim.max_steps,
            is_running=sim.is_running,
        )

    # ---------------------------------------------------------------- SSE

    @app.get("/events")
    async def events(request: Request):
        """Stream observations via Server-Sent Events.

        Each event is a JSON-encoded :class:`ObservationResponse`
        pushed whenever ``/reset`` or ``/step`` produces a new
        observation.  Multiple viewers can subscribe concurrently.

        The stream stays open until the client disconnects.
        ``EventSourceResponse`` handles client disconnection and
        generator cleanup automatically.
        """
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        _subscribers.add(queue)
        logger.info("SSE client connected (%d subscribers)", len(_subscribers))

        async def _event_generator():
            try:
                # Yield an initial comment to flush HTTP headers
                # immediately, so clients see status 200 and
                # ``Content-Type: text/event-stream`` without waiting
                # for the first real event.
                yield {"comment": "connected"}
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield {"event": "observation", "data": payload}
                    except TimeoutError:
                        # Send keep-alive comment to prevent proxy timeouts.
                        yield {"comment": "keep-alive"}
            except asyncio.CancelledError:
                pass
            finally:
                _subscribers.discard(queue)
                logger.info(
                    "SSE client disconnected (%d subscribers)", len(_subscribers)
                )

        return EventSourceResponse(_event_generator())

    return app
