# Architecture

This document describes the internal architecture of FlairSim, its module
decomposition, data flow, and key design decisions.

---

## Overview

FlairSim is structured as a single Python package (`flairsim`) with five
internal subpackages, each with a well-defined responsibility:

```
flairsim/
  map/        Geospatial tile management        (data layer)
  drone/      Drone state, physics, sensors      (agent layer)
  core/       Simulation loop, actions, obs,     (engine layer)
              scenarios, grid overlay
  server/     HTTP REST API (FastAPI)            (interface layer)
  viewer/     Real-time visualisation            (presentation layer)
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
sub-regions to the camera model.  Supports all 10 FLAIR-HUB modalities
and provides auto-discovery of available modalities in a data root.

**Key classes**:

| Class | File | Role |
|-------|------|------|
| `Modality` | `modality.py` | Enum of all 10 FLAIR-HUB data modalities with specs |
| `ModalitySpec` | `modality.py` | Frozen dataclass describing a modality's physical properties |
| `TileInfo` | `tile_loader.py` | Lightweight tile metadata (path, row, col, bounds) |
| `TileData` | `tile_loader.py` | In-memory tile raster + metadata |
| `MapBounds` | `map_manager.py` | Axis-aligned bounding box with containment/intersection |
| `MapManager` | `map_manager.py` | Core spatial engine: tile discovery, grid geometry, region extraction |

**Key functions**:

| Function | File | Role |
|----------|------|------|
| `discover_modalities()` | `modality.py` | Scan a data root directory and return all available `Modality` values |
| `pick_primary_modality()` | `modality.py` | Choose the best visual modality from a set (prefers AERIAL > SPOT > S2) |
| `is_single_modality_dir()` | `modality.py` | Detect whether a path points to a single modality directory |
| `normalize_to_uint8()` | `tile_loader.py` | Normalise any-dtype raster to uint8 for display (handles float, uint16, etc.) |

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

- **Multi-modality auto-detection**: `discover_modalities()` scans a data
  root directory (e.g. `FLAIR-HUB_TOY/D006/`) for subdirectories whose
  names match known FLAIR-HUB suffixes (`_AERIAL_RGBI`, `_S2_RGBI`,
  `_DEM_ELEV`, etc.).  Retro-compatibility is maintained: if a single
  modality directory is passed (old `--data-dir` usage), it is detected
  via `is_single_modality_dir()` and wrapped transparently.

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
`reset() -> step() -> Observation` loop.  Also provides predefined
mission scenarios (YAML-based) and a grid overlay system for VLM
prompting.

**Key classes**:

| Class | File | Role |
|-------|------|------|
| `ActionType` | `action.py` | Enum: MOVE, FOUND, STOP |
| `Action` | `action.py` | Frozen displacement command with factory methods |
| `EpisodeResult` | `observation.py` | Episode outcome (success, reason, metrics) |
| `Observation` | `observation.py` | Step output: images dict, drone state, done flag, metadata |
| `SimulatorConfig` | `simulator.py` | Top-level config (drone, camera, max_steps, ROI) |
| `FlairSimulator` | `simulator.py` | Main engine class (multi-modality, scenario-aware) |
| `Scenario` | `scenario.py` | Complete mission definition (dataset, start, target, limits) |
| `ScenarioLoader` | `scenario.py` | Discovers and loads scenarios from a directory of YAML files |
| `ScenarioTarget` | `scenario.py` | Target position with acceptance radius |
| `ScenarioDataset` | `scenario.py` | Dataset reference within a scenario (data_dir, optional ROI) |
| `ScenarioStart` | `scenario.py` | Start position (x, y, z) for a scenario |
| `GridOverlay` | `grid.py` | NxN grid overlay with alphanumeric cell labels for VLM prompting |
| `GridConfig` | `grid.py` | Configuration for grid appearance (line width, font size, alpha) |

**The simulation loop**:

```python
sim = FlairSimulator(data_dir="path/to/D004/")
obs = sim.reset()                # drone at map centre, step 0

while not obs.done:
    action = agent.decide(obs)   # agent logic
    obs = sim.step(action)       # physics + capture + termination check

print(obs.result)                # EpisodeResult
print(sim.flight_log)            # FlightLog with full trajectory
```

**Multi-modality usage**:

```python
# Pass a data root containing multiple modality subdirectories
sim = FlairSimulator(data_dir="path/to/D006/")

obs = sim.reset()
obs.images                       # {"AERIAL_RGBI": np.ndarray, "S2_RGBI": ..., ...}
obs.image                        # Primary modality image (auto-selected)
sim.primary_modality             # e.g. Modality.AERIAL_RGBI
sim.map_managers                 # Dict[str, MapManager] -- one per modality
```

**Scenario-driven usage**:

```python
from flairsim import ScenarioLoader

loader = ScenarioLoader("scenarios/")
scenario = loader.get("find_target_D006")

sim = FlairSimulator(data_dir=scenario.dataset.data_dir, scenario=scenario)
obs = sim.reset()                # drone at scenario start position

# obs.metadata includes "distance_to_target", "target_x", "target_y"
# Episode auto-terminates on FOUND if drone is within target.radius
```

**Episode termination**:

An episode ends when any of these conditions is met:

1. **FOUND**: Agent calls `Action.found()`.
2. **STOP**: Agent calls `Action.stop()`.  Always `success=False`.
3. **Step limit**: `step_count >= max_steps`.  Always `success=False`.

When a scenario is active, `Action.found()` checks if the drone is within
`target.radius` metres of the target position.  If so, `success=True`;
otherwise `success=False`.

---

### 4. `flairsim/server/` -- HTTP REST API

**Purpose**: Expose the simulator as a local HTTP service so that external
programs (VLM agents, notebooks, other languages) can pilot the drone via
standard HTTP requests.  Supports scenarios, multi-modality images, and
grid overlay via query parameters.

**Key classes / functions**:

| Name | File | Role |
|------|------|------|
| `create_app()` | `app.py` | FastAPI app factory. Instantiates simulator and wires routes. |
| `_State` | `app.py` | Mutable server state: current simulator, scenario, grid overlay. |
| `_apply_grid_overlay()` | `app.py` | Apply NxN grid to an image if grid is enabled. |
| `main()` | `cli.py` | CLI entry point (`flairsim-server` command). |

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode. Optional body: `{x, y, z}`. Optional query: `?scenario_id=...&grid=N` |
| `/step` | POST | Send action `{dx, dy, dz, action_type}`, receive observation. Optional query: `?grid=N` |
| `/state` | GET | Current drone state without advancing |
| `/telemetry` | GET | Full flight log for current episode |
| `/config` | GET | Simulator config, map bounds, drone/camera params, active scenario_id |
| `/events` | GET | Server-Sent Events stream (pushes observations on reset/step) |
| `/scenarios` | GET | List all available scenarios (id, name, description) |

**Response changes for multi-modality**:

The `ObservationResponse` includes an `images` field: a dict mapping
modality names to base64-encoded PNG strings.  The legacy `image_base64`
field still exists and contains the primary modality image for
retro-compatibility.

**Grid overlay via query parameter**:

Passing `?grid=N` (e.g. `?grid=4`) on `/reset` or `/step` applies an
NxN grid overlay to all returned images.  `?grid=0` disables the grid.
The grid state persists across steps until changed.

**Design decisions**:

- **FastAPI**: Chosen for auto-generated OpenAPI/Swagger docs, Pydantic
  validation, and async support.  The Swagger UI at `/docs` makes manual
  testing trivial.

- **Base64 PNG images**: The observation image is PNG-encoded and base64'd
  in the JSON response.  This avoids multipart responses and keeps the API
  simple.  An agent can decode with `base64.b64decode()`.

- **Proper normalisation**: All modalities (float32, uint16, 2-band, etc.)
  are normalised to uint8 via `normalize_to_uint8()` before PNG encoding.
  A 2-band image (e.g. DEM with DSM+DTM) is rendered as a 2-channel
  greyscale composite.

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

- **`_State` class**: Holds mutable server state (`current_sim`,
  `current_scenario`, `grid_overlay`) to allow scenario switching and
  grid changes without restarting the server.

---

### 5. `flairsim/viewer/` -- Desktop visualisation

**Purpose**: Pygame-based interactive viewer for manual flight and
debugging.  Supports three modes of operation, grid overlay toggling,
modality cycling, and scenario-aware HUD display.

**Key classes**:

| Class | File | Role |
|-------|------|------|
| `ViewerConfig` | `viewer.py` | Window size, FPS, styling |
| `FlairViewer` | `viewer.py` | Main window + keyboard handling (incl. G for grid, Tab for modality) |
| `MinimapConfig` | `minimap.py` | Minimap overlay config |
| `Minimap` | `minimap.py` | Renders scaled-down map + trajectory + target crosshair |
| `HUDConfig` | `hud.py` | HUD overlay config |
| `HUD` | `hud.py` | Renders telemetry text, scenario info, distance-to-target |
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

**Keyboard controls**:

| Key | Action |
|-----|--------|
| Arrow keys / WASD | Move drone horizontally |
| Q / E | Ascend / descend |
| Space | Mark FOUND |
| Escape | Quit |
| **Tab** | Cycle active modality (multi-modality mode) |
| **G** | Toggle grid overlay on/off |

**`ViewerObservation` adapter pattern**: The HUD, minimap, and main
rendering code all consume `ViewerObservation` -- a thin dataclass that
holds exactly `(image_rgb, drone_state, step, done, ...)`.  Two factory
methods construct it from either a local `Observation` object
(`from_observation()`) or a server JSON dict (`from_server_response()`),
keeping the rendering code source-agnostic.  In multi-modality mode,
`ViewerObservation` also carries `images_rgb` (dict of all modality
images as RGB arrays) and `metadata` (scenario info, distances, etc.).

---

### 6. `flairsim/core/scenario.py` -- Predefined scenarios

**Purpose**: Define reproducible, shareable mission configurations as
YAML files.  A scenario specifies the dataset, drone start position,
target location with acceptance radius, and episode limits.

**YAML format**:

```yaml
scenario_id: find_target_D006
name: Find the industrial building
description: Locate the large warehouse in zone D006.
dataset:
  data_dir: FLAIR-HUB_TOY/D006-2018_AERIAL_RGBI
  roi: null
start:
  x: 800100.0
  y: 6500200.0
  z: 150.0
target:
  x: 800300.0
  y: 6500050.0
  radius: 50.0
max_steps: 200
```

**Key design decisions**:

- **YAML over JSON**: YAML supports comments and is more readable for
  human-authored mission files.  `pyyaml>=6.0` is the only new dependency.

- **Relative data paths**: `dataset.data_dir` is relative to the data
  root passed to the server/simulator.  This makes scenarios portable
  across machines.

- **`ScenarioLoader` caching**: Scenarios are loaded once at startup and
  cached by `scenario_id`.  The `/scenarios` endpoint lists them without
  re-reading disk.

- **Separation from simulator**: `Scenario` is a pure data object.  The
  `FlairSimulator` accepts an optional `scenario` parameter and uses it
  to configure start position, target, and max_steps.  The simulator
  itself remains scenario-agnostic when no scenario is provided.

---

### 7. `flairsim/core/grid.py` -- Grid overlay for VLMs

**Purpose**: Draw an NxN alphanumeric grid over camera images to enable
Set-of-Mark and Scaffold-style spatial prompting for VLMs.

**Background**:

- **Set-of-Mark prompting** (Yang et al., arXiv:2310.11441): Overlays
  visual markers on images so VLMs can reference spatial locations by
  label rather than by coordinates.
- **Scaffold prompting** (Lei et al., arXiv:2402.12058): Uses
  alphanumeric grid cells (A1, B2, ...) specifically to ground VLM
  spatial reasoning.

**Key design decisions**:

- **Pure NumPy/PIL core**: The `GridOverlay.draw()` method operates on
  `(H, W, 3)` uint8 arrays and returns the same.  No pygame dependency.
  A separate `draw_on_surface()` method exists for pygame rendering.

- **Alphanumeric labelling**: Rows are labelled A-Z, columns 1-N.  This
  gives cells like "A1", "B3", "D5".  Max grid size is 26 (limited by
  the alphabet).

- **Coordinate conversion**: `cell_center("B3")` returns `(px_x, px_y)`,
  enabling the agent to convert a VLM's textual cell reference into
  a movement direction.

- **Configurable appearance**: `GridConfig` controls line width, font
  size, label alpha, and line colour.  Default settings are tuned for
  readability on 512x512 images.

- **Server integration**: The grid can be toggled per-request via the
  `?grid=N` query parameter.  The `_State` class persists the active
  `GridOverlay` instance across steps.

---

## Multi-modality data flow

When a data root directory is passed (e.g. `D006/`), the simulator
creates one `MapManager` per detected modality:

```
Data root: D006/
  ├── D006-2018_AERIAL_RGBI/    --> MapManager (4-band uint8)
  ├── D006-2020_S2_RGBI/        --> MapManager (4-band uint16)
  ├── D006-2018_DEM_ELEV/       --> MapManager (2-band float32)
  └── ...

discover_modalities("D006/")
  --> [Modality.AERIAL_RGBI, Modality.S2_RGBI, Modality.DEM_ELEV, ...]

pick_primary_modality({...})
  --> Modality.AERIAL_RGBI  (visual preference: AERIAL > SPOT > S2)
```

On each `step()`, the camera captures from **every** `MapManager`:

```
FlairSimulator.step(action)
  |
  +---> Drone.move(dx, dy, dz)
  |
  +---> for name, mm in self.map_managers.items():
  |         CameraModel.capture(mm, x, y, z)  --> np.ndarray
  |
  +---> Observation(
  |         image=primary_image,        # backward-compatible
  |         images={name: array, ...},  # all modalities
  |     )
  |
  +---> FlightLog.append(TelemetryRecord)
```

**Retro-compatibility**: Passing a single modality path
(`--data-dir D006-2018_AERIAL_RGBI`) still works.  `is_single_modality_dir()`
detects this case and creates a single-entry `map_managers` dict.  The
`obs.image` field always returns the primary modality.

---

## Full data flow diagram

```
External Agent (VLM, script, notebook)          Observer Viewer (SSE)
       |                                              ^
       |  HTTP POST /step  {"dx": 10, ...}            | GET /events
       |  ?grid=4                                     |
       v                                              |
  flairsim.server (FastAPI on localhost) ------> _broadcast()
       |                                         (push to all
       |  scenario_id --> ScenarioLoader             subscriber queues)
       |  grid=N --> GridOverlay
       v
  FlairSimulator.step(action)
       |
       +---> Drone.move(dx, dy, dz)
       |          |
       |          v
       |     DroneState (clamped position)
       |
       +---> for each modality:
       |       CameraModel.capture(map_manager, x, y, z)
       |          |
       |          v
       |       MapManager.get_region()
       |          |
       |          v
       |       np.ndarray (bands, H, W)
       |
       +---> FlightLog.append(TelemetryRecord)
       |
       +---> [if scenario] compute distance_to_target
       |
       v
  Observation (images dict, drone_state, done, result, metadata)
       |
       +---> normalize_to_uint8() for each modality
       |
       +---> [if grid] _apply_grid_overlay()
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

### Why multi-modality?

FLAIR-HUB provides up to 10 modalities per geographic zone (aerial RGB+IR,
Sentinel-2 bands, elevation models, land cover, etc.).  Supporting all of
them simultaneously:

- **Enables richer VLM prompts**: An agent can see both aerial imagery and
  elevation data in a single step.
- **Supports modality-specific research**: Compare VLM performance on
  high-res aerial vs lower-res satellite imagery.
- **Matches real sensor fusion**: Real drones carry multiple sensors;
  FlairSim reflects this.

### Why YAML scenarios?

Predefined scenarios provide:

- **Reproducibility**: Every researcher runs the exact same mission
  (same start, same target, same dataset).
- **Benchmarking**: Compare VLM agents on a standardised set of tasks.
- **Ease of use**: No need to manually configure start positions and
  targets for each experiment.
- **Shareability**: YAML files are human-readable and version-controlled.

### Why grid overlays for VLMs?

Grid overlays (Set-of-Mark / Scaffold prompting) help VLMs:

- **Ground spatial reasoning**: "Move to cell B3" is unambiguous.
- **Reduce hallucination**: Labelled regions provide concrete anchors.
- **Enable structured output**: The agent can refer to grid cells in its
  response, which can be parsed programmatically.

### Why frozen dataclasses?

Configuration objects (`DroneConfig`, `CameraConfig`, `TelemetryRecord`)
are all `@dataclass(frozen=True, slots=True)`.  This provides:

- **Thread safety**: Immutable objects can be shared safely.
- **Hash-ability**: Can be used as dict keys or in sets.
- **Bug prevention**: Accidental mutation is caught at runtime.
- **Memory efficiency**: `slots=True` eliminates `__dict__` overhead.

---

## Testing strategy

The test suite (428 tests across 13 files in `tests/`) uses four
approaches:

1. **Unit tests**: Test individual classes in isolation with mock data.
   No disk I/O, no GeoTIFF files.

2. **Integration tests**: Create temporary synthetic GeoTIFF grids using
   `rasterio` (2x2 or 3x3 tile grids) and validate the full pipeline
   from `MapManager` through `FlairSimulator`.

3. **Server tests**: Use FastAPI's `TestClient` (httpx) to test all HTTP
   endpoints against a synthetic tile grid.  No real network or data.
   Includes tests for scenario endpoints, grid overlay parameters, and
   multi-modality image responses.

4. **SSE tests**: Spin up a real `uvicorn.Server` in a background thread,
   connect via `httpx.AsyncClient`, and use `asyncio.wait()` to
   concurrently subscribe and trigger events.  Validates the full SSE
   pipeline (connection, `connected` comment, observation push on
   `/reset` and `/step`, event structure, data consistency).

**New test files added for the three features**:

| File | Tests | Scope |
|------|-------|-------|
| `test_multimodality.py` | 65 | Multi-modality discovery, MapManager per modality, Observation.images, retro-compatibility |
| `test_grid.py` | 63 | GridOverlay draw, cell labelling, coordinate conversion, GridConfig, edge cases |
| `test_scenario.py` | -- | ScenarioLoader, Scenario parsing, ScenarioTarget, YAML validation |

All tests run in ~5 seconds with no external data dependencies.

```bash
uv run pytest -v      # run all 428 tests
uv run pytest -x      # stop on first failure
uv run pytest -k drone  # run only drone-related tests
```
