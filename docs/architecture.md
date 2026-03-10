# Architecture

This document describes the internal architecture of FlairSim, its module
decomposition, data flow, and key design decisions.

---

## Overview

FlairSim is structured as a single Python package (`flairsim`) with five
internal subpackages, each with a well-defined responsibility:

```
flairsim/
  map/        Geospatial tile management      (data layer)
  drone/      Drone state, physics, sensors    (agent layer)
  core/       Simulation loop, actions, obs    (engine layer)
  server/     HTTP REST API (FastAPI)          (interface layer)
  viewer/     Real-time visualisation          (presentation layer)
```

Dependencies flow strictly downward:

```
      server
        |
      core        viewer
     /    \         |
  drone   map      core
    \    /
    (numpy)
```

No circular dependencies exist.  The `viewer` depends on `core` (to call
`sim.step()`) and `pygame`, but the simulator itself has no dependency on
the viewer.  The `server` depends on `core` and `fastapi`, but the
simulator runs fine without it (in-process usage is still supported).

---

## Module details

### 1. `flairsim/map/` -- Geospatial tile management

**Purpose**: Load FLAIR-HUB GeoTIFF tiles from disk, assemble them into a
seamless queryable raster surface, and serve arbitrary rectangular
sub-regions to the camera model.

**Key classes**:

| Class | File | Role |
|-------|------|------|
| `Modality` | `modality.py` | Enum of all 10 FLAIR-HUB data modalities with specs |
| `ModalitySpec` | `modality.py` | Frozen dataclass describing a modality's physical properties |
| `TileInfo` | `tile_loader.py` | Lightweight tile metadata (path, row, col, bounds) |
| `TileData` | `tile_loader.py` | In-memory tile raster + metadata |
| `MapBounds` | `map_manager.py` | Axis-aligned bounding box with containment/intersection |
| `MapManager` | `map_manager.py` | Core spatial engine: tile discovery, grid geometry, region extraction |

**Key design decisions**:

- **No spatial index (R-tree)**: FLAIR-HUB tiles are on a regular grid.
  A simple `dict[(row, col) -> Path]` keyed by grid coordinates is
  sufficient and faster than a tree for the grid sizes we encounter
  (< 300 tiles per ROI).

- **Lazy vs eager loading**: `MapManager` supports both modes.  By default
  (`preload=True`), all tiles are loaded at init for maximum runtime
  performance.  Set `preload=False` for large datasets where memory is
  constrained.

- **Coordinate system**: All world coordinates are in EPSG:2154
  (Lambert-93, metres).  The mosaic pixel grid has its origin at the
  north-west corner, with `px` increasing east and `py` increasing south.

- **Band-first layout**: Raster arrays are always `(bands, H, W)`,
  following the rasterio convention.  The `Observation.image_rgb()` method
  converts to channel-last `(H, W, 3)` for display.

**Data flow**:

```
Disk (.tif files)
  --> rasterio.open() + src.read()
  --> TileData(info, data)
  --> MapManager._tile_cache[(row, col)]
  --> MapManager.get_region(x, y, half_extent, output_size)
  --> np.ndarray (bands, H, W)
```

---

### 2. `flairsim/drone/` -- Drone state and sensors

**Purpose**: Model the physical drone (position, movement, bounds
clamping) and the nadir camera that converts altitude to ground
observations.

**Key classes**:

| Class | File | Role |
|-------|------|------|
| `DroneConfig` | `drone.py` | Immutable physical limits (z_min, z_max, max_step_distance) |
| `DroneState` | `drone.py` | Mutable position snapshot (x, y, z, heading, counters) |
| `MoveResult` | `drone.py` | Outcome of a displacement command (requested vs actual) |
| `Drone` | `drone.py` | Main class: reset, move, bounds management |
| `CameraConfig` | `camera.py` | Immutable sensor params (FOV, image_size) |
| `CameraModel` | `camera.py` | Ground footprint geometry, image capture via MapManager |
| `TelemetryRecord` | `telemetry.py` | Frozen per-step state snapshot |
| `FlightLog` | `telemetry.py` | Accumulates records, computes stats, exports CSV/JSON |

**Key design decisions**:

- **Instantaneous movement**: No aerodynamic simulation.  Each `move()`
  call teleports the drone to its new position.  This matches FlySearch's
  simplification and keeps the focus on VLM evaluation.

- **Boundary clamping, not rejection**: When a move would exit the map or
  altitude limits, the position is silently clamped and `was_clipped=True`
  is returned in the `MoveResult`.  This avoids hard failures and lets the
  agent detect boundary situations.

- **State copying**: `Drone.state` returns a *copy* of the internal state
  to prevent accidental external mutation.

- **Simple pinhole camera**: `ground_half_extent = z * tan(fov/2)`.  With
  the default 90-degree FOV, the footprint equals `2 * altitude` metres.
  The output image is always resampled to `image_size x image_size` pixels
  regardless of altitude, matching FlySearch conventions.

---

### 3. `flairsim/core/` -- Simulation engine

**Purpose**: Wire together the map, drone, and camera into a coherent
`reset() -> step() -> Observation` loop.

**Key classes**:

| Class | File | Role |
|-------|------|------|
| `ActionType` | `action.py` | Enum: MOVE, FOUND, STOP |
| `Action` | `action.py` | Frozen displacement command with factory methods |
| `EpisodeResult` | `observation.py` | Episode outcome (success, reason, metrics) |
| `Observation` | `observation.py` | Step output: image, drone state, done flag, metadata |
| `SimulatorConfig` | `simulator.py` | Top-level config (drone, camera, max_steps, ROI) |
| `FlairSimulator` | `simulator.py` | Main engine class |

**The simulation loop**:

```python
sim = FlairSimulator(data_dir="...")
obs = sim.reset()                # drone at map centre, step 0

while not obs.done:
    action = agent.decide(obs)   # agent logic
    obs = sim.step(action)       # physics + capture + termination check

print(obs.result)                # EpisodeResult
print(sim.flight_log)            # FlightLog with full trajectory
```

**Episode termination**:

An episode ends when any of these conditions is met:

1. **FOUND**: Agent calls `Action.found()`.
2. **STOP**: Agent calls `Action.stop()`.  Always `success=False`.
3. **Step limit**: `step_count >= max_steps`.  Always `success=False`.

---

### 4. `flairsim/server/` -- HTTP REST API

**Purpose**: Expose the simulator as a local HTTP service so that external
programs (VLM agents, notebooks, other languages) can pilot the drone via
standard HTTP requests.

**Key classes / functions**:

| Name | File | Role |
|------|------|------|
| `create_app()` | `app.py` | FastAPI app factory. Instantiates simulator and wires routes. |
| `main()` | `cli.py` | CLI entry point (`flairsim-server` command). |

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode. Optional body: `{x, y, z}` |
| `/step` | POST | Send action `{dx, dy, dz, action_type}`, receive observation |
| `/state` | GET | Current drone state without advancing |
| `/telemetry` | GET | Full flight log for current episode |
| `/config` | GET | Simulator config, map bounds, drone/camera params |
| `/events` | GET | Server-Sent Events stream (pushes observations on reset/step) |

**Design decisions**:

- **FastAPI**: Chosen for auto-generated OpenAPI/Swagger docs, Pydantic
  validation, and async support.  The Swagger UI at `/docs` makes manual
  testing trivial.

- **Base64 PNG images**: The observation image is PNG-encoded and base64'd
  in the JSON response.  This avoids multipart responses and keeps the API
  simple.  An agent can decode with `base64.b64decode()`.

- **Async endpoints with thread offloading**: `/reset` and `/step` are
  `async def` and offload the heavy simulator call (`sim.reset()`,
  `sim.step()`) to a thread via `asyncio.to_thread()`.  This keeps the
  event loop free and, critically, ensures that the SSE `_broadcast()`
  call runs on the event loop thread where `asyncio.Queue.put_nowait()`
  is safe.

- **SSE push channel** (`GET /events`): Uses `sse-starlette`'s
  `EventSourceResponse`.  Each connected viewer gets an
  `asyncio.Queue(maxsize=64)`.  `_broadcast()` serialises each
  `ObservationResponse` once and pushes it to all subscriber queues.
  Slow consumers trigger backpressure (oldest event dropped).  A
  keep-alive comment is sent every 30 seconds to prevent proxy timeouts.
  An initial `{"comment": "connected"}` is yielded immediately to flush
  HTTP headers so clients see status 200 without waiting for the first
  real event.

---

### 5. `flairsim/viewer/` -- Desktop visualisation

**Purpose**: Pygame-based interactive viewer for manual flight and
debugging.  Supports three modes of operation.

**Key classes**:

| Class | File | Role |
|-------|------|------|
| `ViewerConfig` | `viewer.py` | Window size, FPS, styling |
| `FlairViewer` | `viewer.py` | Main window + keyboard handling |
| `MinimapConfig` | `minimap.py` | Minimap overlay config |
| `Minimap` | `minimap.py` | Renders scaled-down map + trajectory |
| `HUDConfig` | `hud.py` | HUD overlay config |
| `HUD` | `hud.py` | Renders telemetry text (altitude, coords, step, etc.) |
| `ViewerObservation` | `remote.py` | Adapter bridging local `Observation` and server JSON |
| `ViewerDroneState` | `remote.py` | Lightweight drone state for display |
| `ViewerEpisodeResult` | `remote.py` | Episode outcome for HUD display |

The viewer is fully optional (`uv sync --extra viewer`).  The simulator
runs headless without it.

**Three modes** (selected via `--mode`):

| Mode | Flag | Description |
|------|------|-------------|
| **local** | `--mode local` | Creates a local `FlairSimulator`, flies directly. Default mode. |
| **observe** | `--mode observe` | Connects to a running server via SSE (`GET /events`). Read-only: watches what an external agent does. No keyboard flight. |
| **fly** | `--mode fly` | Connects to a running server, sends `POST /step` on keypresses, receives observation via HTTP response. |

**`ViewerObservation` adapter pattern**: The HUD, minimap, and main
rendering code all consume `ViewerObservation` -- a thin dataclass that
holds exactly `(image_rgb, drone_state, step, done, ...)`.  Two factory
methods construct it from either a local `Observation` object
(`from_observation()`) or a server JSON dict (`from_server_response()`),
keeping the rendering code source-agnostic.

---

## Data flow diagram

```
External Agent (VLM, script, notebook)          Observer Viewer (SSE)
       |                                              ^
       |  HTTP POST /step  {"dx": 10, ...}            | GET /events
       v                                              |
  flairsim.server (FastAPI on localhost) ------> _broadcast()
       |                                         (push to all
       v                                          subscriber queues)
  FlairSimulator.step(action)
       |
       +---> Drone.move(dx, dy, dz)
       |          |
       |          v
       |     DroneState (clamped position)
       |
       +---> CameraModel.capture(map_manager, x, y, z)
       |          |
       |          v
       |     MapManager.get_region()
       |          |
       |          v
       |     np.ndarray (bands, H, W)
       |
       +---> FlightLog.append(TelemetryRecord)
       |
       v
  Observation (image, drone_state, done, result)
       |
       +---> JSON response --> External Agent
       |
       +---> ObservationResponse.model_dump_json()
                  |
                  v
             SSE "observation" event --> Observer Viewers
```

---

## Design rationale

### Why not a 3D engine (UE5, Unity)?

FlySearch uses Unreal Engine 5, which provides photorealistic 3D
environments but requires significant infrastructure (GPU, game engine
install, C++ bridge).  FlairSim deliberately trades 3D fidelity for:

1. **Real imagery**: FLAIR-HUB provides actual aerial photos of France,
   not synthetic renders.
2. **Simplicity**: Pure Python, no game engine dependency.
3. **Reproducibility**: Deterministic, no physics engine randomness.
4. **Dataset diversity**: 78 departments, 19 land-cover classes, multiple
   sensors.

The trade-off is that we only support top-down (nadir) views.  This is
appropriate for many real-world drone survey tasks.

### Why relative displacements instead of waypoints?

Following FlySearch, actions are `(dx, dy, dz)` in metres.  This is:

- **Simple**: No path planning, no coordinate frame confusion.
- **VLM-friendly**: The agent reasons about "move 10m east" rather than
  absolute GPS coordinates.
- **Compatible**: Direct comparison with FlySearch results.

### Why an HTTP API for agent communication?

The simulator and agent are separate concerns.  An HTTP API:

- **Decouples** the simulator from agent implementation (language, framework).
- **Mirrors real systems** where drone components communicate via APIs.
- **Enables parallel development** -- the simulator team and the VLM team
  work independently.
- **Provides free documentation** via FastAPI's auto-generated Swagger UI.

### Why frozen dataclasses?

Configuration objects (`DroneConfig`, `CameraConfig`, `TelemetryRecord`)
are all `@dataclass(frozen=True, slots=True)`.  This provides:

- **Thread safety**: Immutable objects can be shared safely.
- **Hash-ability**: Can be used as dict keys or in sets.
- **Bug prevention**: Accidental mutation is caught at runtime.
- **Memory efficiency**: `slots=True` eliminates `__dict__` overhead.

---

## Testing strategy

The test suite (268 tests, `tests/`) uses four approaches:

1. **Unit tests**: Test individual classes in isolation with mock data.
   No disk I/O, no GeoTIFF files.

2. **Integration tests**: Create temporary synthetic GeoTIFF grids using
   `rasterio` (2x2 or 3x3 tile grids) and validate the full pipeline
   from `MapManager` through `FlairSimulator`.

3. **Server tests**: Use FastAPI's `TestClient` (httpx) to test all HTTP
   endpoints against a synthetic tile grid.  No real network or data.

4. **SSE tests**: Spin up a real `uvicorn.Server` in a background thread,
   connect via `httpx.AsyncClient`, and use `asyncio.wait()` to
   concurrently subscribe and trigger events.  Validates the full SSE
   pipeline (connection, `connected` comment, observation push on
   `/reset` and `/step`, event structure, data consistency).

All tests run in ~5 seconds with no external data dependencies.

```bash
uv run pytest -v      # run all tests
uv run pytest -x      # stop on first failure
uv run pytest -k drone  # run only drone-related tests
```
