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
GET  /scenarios
    List available scenarios (when a scenario loader is configured).
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
from ..core.grid import GridOverlay
from ..core.scenario import Scenario, ScenarioLoader
from ..core.simulator import FlairSimulator, SimulatorConfig
from ..drone.camera import CameraConfig
from ..drone.drone import DroneConfig
from ..map.tile_loader import normalize_to_uint8

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
    scenario_id: Optional[str] = Field(
        None, description="Scenario ID to load. None = free flight (use current data)."
    )


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
    images: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-modality PNG images in base64, keyed by modality name "
            "(e.g. 'AERIAL_RGBI', 'DEM_ELEV'). Empty in single-modality mode."
        ),
    )


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
    scenario_id: Optional[str] = None


class ScenarioSummaryResponse(BaseModel):
    """Summary of a single scenario."""

    scenario_id: str
    name: str
    description: str
    max_steps: int
    target_radius: float


class ScenariosListResponse(BaseModel):
    """Response for GET /scenarios."""

    scenarios: List[ScenarioSummaryResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image_png(image: np.ndarray) -> str:
    """Encode a (bands, H, W) array as base64 PNG.

    Converts to RGB uint8 (H, W, 3) before encoding.  Non-uint8 data
    (e.g. float32 DEM, uint16 satellite) is normalised via percentile
    stretch to produce a visually meaningful 8-bit image.
    """
    # Normalise to uint8 if needed (handles uint16, float32, etc.).
    if image.dtype != np.uint8:
        image = normalize_to_uint8(image)

    if image.ndim == 3 and image.shape[0] >= 3:
        rgb = np.transpose(image[:3], (1, 2, 0))
    elif image.ndim == 3 and image.shape[0] == 2:
        # Two bands (e.g. DEM DSM+DTM): show first band as greyscale.
        band = image[0]
        rgb = np.stack([band, band, band], axis=-1)
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


def _apply_grid_overlay(image: np.ndarray, grid: Optional[GridOverlay]) -> np.ndarray:
    """Apply grid overlay to a (bands, H, W) image if grid is set.

    The overlay expects (H, W, 3) uint8, so we convert, draw, and
    convert back to (3, H, W) for the normal encoding pipeline.
    """
    if grid is None:
        return image

    # Normalise to uint8 first so the overlay draws on a sensible image.
    img = image if image.dtype == np.uint8 else normalize_to_uint8(image)

    # Convert (bands, H, W) → (H, W, 3) for the overlay.
    if img.ndim == 3 and img.shape[0] >= 3:
        rgb = np.transpose(img[:3], (1, 2, 0))
    elif img.ndim == 3 and img.shape[0] == 2:
        band = img[0]
        rgb = np.stack([band, band, band], axis=-1)
    elif img.ndim == 3 and img.shape[0] == 1:
        band = img[0]
        rgb = np.stack([band, band, band], axis=-1)
    elif img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1)
    else:
        rgb = img

    rgb = rgb.astype(np.uint8)
    annotated = grid.draw(rgb)
    # Convert back to (3, H, W) so _encode_image_png handles it.
    return np.transpose(annotated, (2, 0, 1))


def _obs_to_response(obs, grid: Optional[GridOverlay] = None) -> ObservationResponse:
    """Convert an Observation to a serialisable response.

    Parameters
    ----------
    obs : Observation
        Simulator observation.
    grid : GridOverlay or None
        If set, the grid is drawn on all images before encoding.
    """
    state = obs.drone_state
    result = None
    if obs.result is not None:
        result = EpisodeResultResponse(
            success=obs.result.success,
            reason=obs.result.reason,
            steps_taken=obs.result.steps_taken,
            distance_travelled=obs.result.distance_travelled,
        )

    # Optionally apply grid overlay to primary image.
    primary_image = _apply_grid_overlay(obs.image, grid)

    # Encode per-modality images (with grid if active).
    images_b64: Dict[str, str] = {}
    if obs.images:
        for mod_name, mod_image in obs.images.items():
            images_b64[mod_name] = _encode_image_png(
                _apply_grid_overlay(mod_image, grid)
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
        image_base64=_encode_image_png(primary_image),
        image_width=obs.image.shape[-1],
        image_height=obs.image.shape[-2],
        result=result,
        metadata={k: str(v) for k, v in obs.metadata.items()},
        images=images_b64,
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
    scenario_loader: Optional[ScenarioLoader] = None,
    scenario_id: Optional[str] = None,
    grid: Optional[int] = None,
    domain: Optional[str] = None,
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
    scenario_loader : ScenarioLoader or None
        Loader for scenario YAML files.  ``None`` disables scenario support.
    scenario_id : str or None
        Initial scenario to load at startup.  ``None`` starts in free-flight mode.
    grid : int or None
        Default grid overlay size (NxN).  ``None`` disables the grid.
        Can be overridden per-request via the ``grid`` query parameter.
    domain : str or None
        FLAIR-HUB domain prefix (e.g. ``"D006-2020"``).  Passed through
        to the simulator for domain-aware modality discovery.

    Returns
    -------
    FastAPI
        The configured application instance.
    """
    _drone_config = drone_config or DroneConfig()
    _camera_config = camera_config or CameraConfig()

    # Load initial scenario if specified.
    initial_scenario: Optional[Scenario] = None
    effective_domain = domain
    if scenario_id and scenario_loader:
        initial_scenario = scenario_loader.get(scenario_id)
        # Override data_dir and roi from the scenario.
        data_dir = scenario_loader.resolve_data_dir(initial_scenario)
        roi = initial_scenario.dataset.roi
        max_steps = initial_scenario.max_steps
        # Use domain from scenario if not explicitly provided via CLI.
        if not effective_domain and initial_scenario.dataset.domain:
            effective_domain = initial_scenario.dataset.domain

    config = SimulatorConfig(
        drone_config=_drone_config,
        camera_config=_camera_config,
        max_steps=max_steps,
        roi=roi,
        preload_tiles=preload_tiles,
    )

    sim = FlairSimulator(
        data_dir=data_dir,
        config=config,
        scenario=initial_scenario,
        domain=effective_domain,
    )

    # Mutable state: the active simulator and scenario can change when
    # /reset is called with a scenario_id.
    class _State:
        current_sim: FlairSimulator = sim
        current_scenario: Optional[Scenario] = initial_scenario
        grid_overlay: Optional[GridOverlay] = (
            GridOverlay(grid) if grid is not None else None
        )

    state = _State()

    # SSE subscriber management.
    _subscribers: Set[asyncio.Queue[str]] = set()

    def _broadcast(obs_response: ObservationResponse) -> None:
        """Push a serialised observation to all SSE subscribers."""
        if not _subscribers:
            return
        payload = obs_response.model_dump_json()
        dead: list[asyncio.Queue[str]] = []
        for q in list(_subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
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
    async def reset(
        body: Optional[ResetRequest] = None,
        grid: Optional[int] = None,
    ):
        """Start a new episode.

        Resets the drone and returns the initial observation.
        If no body is provided, the drone starts at the map centre
        with the default altitude.

        If ``scenario_id`` is provided and a scenario loader is
        configured, the simulator is (re)created with the scenario's
        dataset and the episode starts at the scenario's start position.

        Parameters
        ----------
        grid : int or None
            Grid overlay size (NxN).  Overrides the server default for
            subsequent responses.  ``0`` disables the grid.
        """
        if body is None:
            body = ResetRequest()

        # Handle scenario switching.
        if body.scenario_id is not None:
            if scenario_loader is None:
                raise HTTPException(
                    status_code=400,
                    detail="Scenario support is not configured on this server.",
                )
            try:
                new_scenario = scenario_loader.get(body.scenario_id)
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Scenario '{body.scenario_id}' not found.",
                )

            new_data_dir = scenario_loader.resolve_data_dir(new_scenario)
            new_domain = new_scenario.dataset.domain or domain
            new_config = SimulatorConfig(
                drone_config=_drone_config,
                camera_config=_camera_config,
                max_steps=new_scenario.max_steps,
                roi=new_scenario.dataset.roi,
                preload_tiles=preload_tiles,
            )
            state.current_sim = await asyncio.to_thread(
                FlairSimulator,
                data_dir=new_data_dir,
                config=new_config,
                scenario=new_scenario,
                domain=new_domain,
            )
            state.current_scenario = new_scenario
            logger.info("Loaded scenario: %s", new_scenario.scenario_id)
        elif body.scenario_id is None and state.current_scenario is not None:
            # Explicit reset without scenario_id keeps current scenario.
            pass

        # Update grid overlay if requested.
        if grid is not None:
            if grid == 0:
                state.grid_overlay = None
            else:
                state.grid_overlay = GridOverlay(grid)

        obs = await asyncio.to_thread(
            state.current_sim.reset, x=body.x, y=body.y, z=body.z
        )
        logger.info(
            "Episode reset at (%.1f, %.1f, %.1f)",
            obs.position[0],
            obs.position[1],
            obs.position[2],
        )
        response = _obs_to_response(obs, grid=state.grid_overlay)
        _broadcast(response)
        return response

    @app.post("/step", response_model=ObservationResponse)
    async def step(body: StepRequest, grid: Optional[int] = None):
        """Advance simulation by one step.

        Applies the given displacement and returns the new observation
        with the updated image and telemetry.

        Parameters
        ----------
        grid : int or None
            Grid overlay size (NxN).  Overrides the server default for
            subsequent responses.  ``0`` disables the grid.
        """
        if not state.current_sim.is_running:
            raise HTTPException(
                status_code=409,
                detail="No active episode. Call POST /reset first.",
            )

        # Update grid overlay if requested.
        if grid is not None:
            if grid == 0:
                state.grid_overlay = None
            else:
                state.grid_overlay = GridOverlay(grid)

        action_type = _parse_action_type(body.action_type)
        action = Action(dx=body.dx, dy=body.dy, dz=body.dz, action_type=action_type)
        obs = await asyncio.to_thread(state.current_sim.step, action)
        response = _obs_to_response(obs, grid=state.grid_overlay)
        _broadcast(response)
        return response

    @app.get("/state", response_model=DroneStateResponse)
    def get_state():
        """Get current drone state without advancing the simulation."""
        if not state.current_sim.is_running:
            raise HTTPException(
                status_code=409,
                detail="No active episode. Call POST /reset first.",
            )

        s = state.current_sim.drone.state
        return DroneStateResponse(
            x=s.x,
            y=s.y,
            z=s.z,
            heading=s.heading,
            step_count=s.step_count,
            total_distance=s.total_distance,
        )

    @app.get("/telemetry", response_model=TelemetryResponse)
    def get_telemetry():
        """Get the full flight log for the current episode."""
        log = state.current_sim.flight_log
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
        cur = state.current_sim
        bounds = cur.map_bounds
        drone_cfg = cur.drone.config
        cam_cfg = cur.camera.config

        return ConfigResponse(
            data_dir=str(cur._data_dir),
            roi=cur.map_manager.roi_name,
            n_tiles=cur.map_manager.n_tiles_loaded,
            pixel_size_m=cur.map_manager.pixel_size_m,
            tile_ground_size=cur.map_manager.tile_ground_size,
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
            max_steps=cur.max_steps,
            is_running=cur.is_running,
            scenario_id=(
                state.current_scenario.scenario_id if state.current_scenario else None
            ),
        )

    # ---------------------------------------------------------------- scenarios

    @app.get("/scenarios", response_model=ScenariosListResponse)
    def list_scenarios():
        """List all available scenarios."""
        if scenario_loader is None:
            return ScenariosListResponse(scenarios=[])

        summaries = []
        for sc in scenario_loader.list_scenarios():
            summaries.append(
                ScenarioSummaryResponse(
                    scenario_id=sc.scenario_id,
                    name=sc.name,
                    description=sc.description,
                    max_steps=sc.max_steps,
                    target_radius=sc.target.radius,
                )
            )
        return ScenariosListResponse(scenarios=summaries)

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
