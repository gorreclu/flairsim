# FlairSim

**Drone simulator over FLAIR-HUB aerial imagery.**

FlairSim is a lightweight, pure-Python drone simulator that flies a
virtual UAV over real French aerial imagery from the
[FLAIR-HUB](https://arxiv.org/abs/2506.07080) dataset (IGN).  It exposes
a **local HTTP API** so that any external program (VLM agent, notebook,
another language) can pilot the drone and receive observations (image +
telemetry).

> **Academic context** -- *Fil rouge* (capstone project) for the
> Mastere Specialise Big Data / IA at Telecom Paris.

---

## Key features

| Feature | Description |
|---------|-------------|
| **Real aerial imagery** | Flies over 0.2 m/px IGN orthophotos (FLAIR-HUB D004, D005, ...) |
| **Auto-download from HuggingFace** | `--domain D006-2020` auto-downloads ZIP files from IGNF/FLAIR-HUB, extracts to a temp directory, and cleans up on shutdown (Ctrl+C) |
| **Multi-modality** | Auto-discovers and loads all available modalities (RGBI, DEM, SPOT, Sentinel-1/2, labels) from a parent directory. Tab key cycles between them in the viewer |
| **Predefined scenarios** | YAML-based mission scenarios with start position, target location, acceptance radius, and automatic success/failure evaluation |
| **Grid overlay (SoM/Scaffold)** | NxN alphanumeric grid overlay (A1, B2, ...) for VLM spatial prompting, inspired by Set-of-Mark and Scaffold prompting research |
| **HTTP REST API** | Local server on `localhost:8000`. Any program can send actions and receive image + telemetry via HTTP |
| **CLI to launch** | `flairsim-server --data-dir path/to/data` -- one command to start |
| **Interactive viewer** | Pygame-based desktop viewer with HUD, minimap, and three modes (local / observe / fly) |
| **SSE push channel** | `GET /events` streams observations to observer viewers in real time |
| **Telemetry logging** | Full flight log with position, displacement, clipping info |
| **Well tested** | 460 tests (unit + integration + server + SSE + downloader), all passing |

---

## Architecture

```
flairsim/
  data/           # Auto-download from HuggingFace (HubDownloader)
  map/           # GeoTIFF tile loading, spatial queries, modality discovery
  drone/         # Drone state & physics, camera model, telemetry
  core/          # Simulator engine, actions, observations, scenarios, grid overlay
  server/        # HTTP REST API (FastAPI) -- the interface for external agents
  viewer/        # Pygame real-time visualization (FlairViewer, HUD, Minimap)
```

Data flow:

```
External Agent (VLM, script, notebook)
    |
    |  POST /step  {"dx": 10, "dy": 0, "dz": -5}
    v
FlairSim HTTP Server (localhost:8000)
    |
    v
FlairSimulator.step()
    |-- Drone.move()            (apply displacement, clamp to bounds)
    |-- CameraModel.capture()   (extract image from map tiles, all modalities)
    |-- FlightLog.record()      (log telemetry)
    |-- Scenario.evaluate()     (check target proximity, if scenario active)
    |-- GridOverlay.draw()      (annotate image with grid, if enabled)
    |
    v
JSON response:
    {
      "step": 42,
      "done": false,
      "image_base64": "<PNG in base64, with optional grid overlay>",
      "images": {"AERIAL_RGBI": "<base64>", "DEM_ELEV": "<base64>", ...},
      "drone_state": {"x": ..., "y": ..., "z": ..., ...},
      "ground_footprint": 200.0,
      "ground_resolution": 0.4,
      "metadata": {"scenario_name": "...", "distance_to_target": "42.5", ...}
    }
```

---

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/) >= 0.4
- FLAIR-HUB data (at least one `D0xx_AERIAL_RGBI` directory, or a parent directory with multiple modalities)

### Install

```bash
cd simulateur

# Install everything (core + server + viewer + dev tools)
uv sync --all-extras
```

### Dependency groups

| Group | Install command | Packages |
|-------|-----------------|----------|
| Core | `uv sync` | numpy, rasterio, Pillow, pyyaml |
| Server | `uv sync --extra server` | + fastapi, uvicorn, sse-starlette, huggingface_hub |
| Viewer | `uv sync --extra viewer` | + pygame |
| Dev | `uv sync --dev` | + pytest, pytest-asyncio, ruff, httpx |
| Everything | `uv sync --all-extras` | All of the above |

---

## Usage

### 1. Start the simulator server

```bash
# Basic: single modality directory
uv run python -m flairsim.server --data-dir path/to/D004-2021_AERIAL_RGBI

# Multi-modality: parent directory (auto-discovers all modalities)
uv run python -m flairsim.server --data-dir path/to/D006-2020

# Flat FLAIR-HUB layout (data downloaded from HuggingFace):
# Use --domain to select which domain to load from the flat root
uv run python -m flairsim.server \
    --data-dir path/to/FLAIR-HUB \
    --domain D006-2020

# Auto-download from HuggingFace (no --data-dir needed):
# Downloads to a temp directory, cleans up on Ctrl+C / shutdown
uv run python -m flairsim.server --domain D006-2020

# Auto-download with multiple modalities
uv run python -m flairsim.server --domain D006-2020 \
    --modalities AERIAL_RGBI DEM_ELEV

# With scenarios (no --data-dir needed; data resolved from --data-root + scenario YAML)
uv run python -m flairsim.server \
    --data-root path/to/FLAIR-HUB \
    --scenarios-dir scenarios/ \
    --scenario find_target_D006

# With grid overlay (4x4 grid)
uv run python -m flairsim.server --data-dir path/to/data --grid 4

# Or via the installed console script (after `pip install flairsim[server]`)
flairsim-server --data-dir path/to/D004-2021_AERIAL_RGBI
```

> **Note (macOS editable installs)**: With `uv sync` (editable mode),
> macOS may set `UF_HIDDEN` on `.pth` files, causing the `flairsim-server`
> console script to fail with `ModuleNotFoundError`.  Use
> `uv run python -m flairsim.server` instead, which always works.
> Non-editable installs (`pip install .`) are not affected.

The server starts on `http://127.0.0.1:8000`. An auto-generated API doc
is available at `http://127.0.0.1:8000/docs` (Swagger UI).

**Full CLI options:**

```
uv run python -m flairsim.server [OPTIONS]

  --data-dir      Path to data directory (required for free flight unless --domain is used)
  --host          Host to bind (default: 127.0.0.1)
  --port          Port (default: 8000)
  --roi           ROI to load (default: auto-select largest)
  --max-steps     Max steps per episode (default: 500)
  --altitude      Default altitude in metres (default: 100)
  --image-size    Camera output resolution in px (default: 500)
  --fov           Camera FOV in degrees (default: 90)
  --no-preload    Don't preload tiles (saves RAM)
  --scenarios-dir Directory containing scenario YAML files
  --data-root     Root for resolving relative scenario data_dir paths
  --domain        FLAIR-HUB domain prefix (e.g. D006-2020); triggers auto-download if no --data-dir
  --modalities    Modalities to download (default: AERIAL_RGBI). Only used with auto-download
  --scenario      Scenario ID to load at startup (no --data-dir needed)
  --grid N        Enable NxN grid overlay (overridable per-request via ?grid=N)
  -v              Debug logging
```

### 2. Communicate from an external agent

Once the server is running, any program can pilot the drone via HTTP:

```python
import httpx
import base64

client = httpx.Client(base_url="http://127.0.0.1:8000")

# Start an episode (free flight)
obs = client.post("/reset").json()

# Start with a scenario
obs = client.post("/reset", json={"scenario_id": "find_target_D006"}).json()

# Enable a 4x4 grid overlay
obs = client.post("/reset?grid=4").json()

print(f"Drone at ({obs['drone_state']['x']:.0f}, {obs['drone_state']['y']:.0f})")

# Fly the drone
while not obs["done"]:
    obs = client.post("/step", json={"dx": 10.0, "dy": 0.0, "dz": -2.0}).json()

    # Decode the primary image (PNG base64 -> bytes)
    image_bytes = base64.b64decode(obs["image_base64"])

    # Access per-modality images (multi-modality mode)
    for mod_name, mod_b64 in obs["images"].items():
        mod_bytes = base64.b64decode(mod_b64)

    # Use obs["drone_state"], obs["ground_footprint"], obs["ground_resolution"],
    # obs["metadata"]["distance_to_target"] to make decisions

# Check result
if obs["done"]:
    print(obs["result"])
```

This also works from **any language** -- curl, JavaScript, Go, etc.:

```bash
# Start episode
curl -X POST http://127.0.0.1:8000/reset

# Start with scenario and grid overlay
curl -X POST "http://127.0.0.1:8000/reset?grid=4" \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "find_target_D006"}'

# Move drone
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"dx": 10, "dy": 5, "dz": 0}'

# Declare target found
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"dx": 0, "dy": 0, "dz": 0, "action_type": "found"}'

# Get current state
curl http://127.0.0.1:8000/state

# Get flight log
curl http://127.0.0.1:8000/telemetry

# Get config / map info
curl http://127.0.0.1:8000/config

# List available scenarios
curl http://127.0.0.1:8000/scenarios

# Subscribe to SSE observation stream (stays open)
curl -N http://127.0.0.1:8000/events
```

### 3. API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode. Body (optional): `{"x", "y", "z", "scenario_id"}`. Query: `?grid=N` |
| `/step` | POST | Send an action. Body: `{"dx", "dy", "dz", "action_type"}`. Query: `?grid=N` |
| `/state` | GET | Current drone state (no simulation advance) |
| `/telemetry` | GET | Full flight log for current episode |
| `/config` | GET | Simulator config, map bounds, drone/camera params, active scenario |
| `/scenarios` | GET | List available scenarios (requires `--scenarios-dir`) |
| `/events` | GET | Server-Sent Events stream (observations pushed on reset/step) |

**Action types** for `/step`:
- `"move"` (default) -- move the drone by (dx, dy, dz) metres
- `"found"` -- declare target found (ends episode; success depends on proximity to target if scenario active)
- `"stop"` -- voluntarily end episode

**Grid overlay** (`?grid=N` query parameter):
- Available on `/reset` and `/step`
- `?grid=4` enables a 4x4 grid on all subsequent images
- `?grid=0` disables the grid
- The grid persists across requests until changed

### 4. Interactive viewer (pygame)

The viewer supports three modes:

#### Local mode (default)

Fly the drone manually with keyboard controls, using a local simulator
(no server needed):

```bash
# Single modality
uv run python -m flairsim.viewer --data-dir path/to/D004-2021_AERIAL_RGBI

# Multi-modality (auto-discovers all modalities)
uv run python -m flairsim.viewer --data-dir path/to/D006-2020

# With grid overlay
uv run python -m flairsim.viewer --data-dir path/to/data --grid 4
```

#### Observer mode

Connect to a running server and watch what an external agent is doing
in real time (read-only, no keyboard flight):

```bash
# Start server in one terminal
uv run python -m flairsim.server --data-dir path/to/data

# Watch from another terminal
uv run python -m flairsim.viewer --mode observe --server-url http://127.0.0.1:8000
```

The observer connects via SSE (`GET /events`) and receives every
observation pushed by the server whenever `/reset` or `/step` is called.

#### Fly mode

Connect to a running server and pilot the drone via keypresses (sends
`POST /step` on each keypress):

```bash
uv run python -m flairsim.viewer --mode fly --server-url http://127.0.0.1:8000
```

**Keyboard controls:**

| Key | Action |
|-----|--------|
| Z / Up | Move north |
| S / Down | Move south |
| Q / Left | Move west |
| D / Right | Move east |
| A | Descend |
| E | Ascend |
| +/- | Adjust move step size |
| Space | Declare FOUND |
| R | Reset episode |
| H | Toggle HUD |
| M | Toggle minimap |
| Tab | Cycle modality (multi-modality mode) |
| G | Toggle grid overlay (default 4x4, or `--grid N`) |
| Escape | Quit |

**Options:** `--window-size`, `--move-step`, `--altitude`, `--roi`, `--grid`, etc.
Run `uv run python -m flairsim.viewer --help` for the full list.

### 5. Predefined scenarios

Scenarios are YAML files that define complete missions:

```yaml
scenario_id: find_target_D006
name: Find target in Isere
description: >
  Navigate over the Isere department (D006) aerial imagery to locate
  a specific ground target.

dataset:
  data_dir: D006-2020_AERIAL_RGBI
  domain: D006-2020     # optional: inferred from data_dir if omitted
  roi: UU-S2-1

start:
  x: 1018422.0
  y: 6278750.0
  z: 150.0

target:
  x: 1018300.0
  y: 6278200.0
  radius: 50.0

max_steps: 200
```

**Fields**:
- `scenario_id` -- Unique ID (must match the YAML filename stem)
- `dataset.data_dir` -- Path to data (absolute or relative to `--data-root`)
- `dataset.domain` -- FLAIR-HUB domain prefix (e.g. `D006-2020`); optional, inferred from `data_dir` if omitted. Needed for flat layouts to discover sibling modalities
- `dataset.roi` -- ROI to load (`null` = auto-select)
- `start` -- Drone starting position (`null` values use map centre / default altitude)
- `target` -- Target position with acceptance `radius` in metres
- `max_steps` -- Episode length limit

**Evaluation**: When the agent declares FOUND, the simulator checks if the
drone is within `target.radius` metres of `(target.x, target.y)`. If yes,
`result.success = True`; otherwise `result.success = False`.

The viewer HUD shows the active scenario name and distance to target.
The minimap displays a crosshair marker at the target location.

### 6. Multi-modality

FlairSim auto-detects whether `--data-dir` points to a single modality
directory or a parent directory containing multiple modalities:

```
# Single modality (backward-compatible)
--data-dir path/to/D006-2020_AERIAL_RGBI

# Multi-modality from single-modality dir (auto-discovers siblings in parent dir)
--data-dir path/to/FLAIR-HUB/D006-2020_AERIAL_RGBI
  -> primary: AERIAL_RGBI
  -> also discovers: DEM_ELEV, SPOT_RGBI, SENTINEL2_TS, ... (sibling D006-2020_* dirs)

# Flat FLAIR-HUB layout (data from HuggingFace): use --domain to filter
--data-dir path/to/FLAIR-HUB --domain D006-2020
  -> discovers: all D006-2020_* modality dirs

# Auto-download from HuggingFace (no --data-dir needed)
--domain D006-2020 --modalities AERIAL_RGBI DEM_ELEV
  -> downloads and extracts to a temp dir, cleaned up on shutdown
```

In multi-modality mode:
- The primary modality (AERIAL_RGBI preferred) is used for the main `image` field
- All modalities are returned in `obs.images` (dict keyed by modality name)
- The server returns per-modality PNG images in `response.images`
- The viewer lets you cycle through modalities with Tab

### 7. Grid overlay for VLMs

The grid overlay divides the camera image into an NxN grid with
alphanumeric cell labels (A1, A2, B1, B2, ...) following the
**Set-of-Mark** (arXiv:2310.11441) and **Scaffold** (arXiv:2402.12058)
prompting approaches.

This enables VLMs to reference specific image regions using short
labels instead of pixel coordinates:

```python
from flairsim import GridOverlay

overlay = GridOverlay(n=4)                          # 4x4 grid
annotated = overlay.draw(image_rgb)                 # (H,W,3) -> (H,W,3) with grid
bounds = overlay.cell_bounds("B3", w, h)            # -> (x_min, y_min, x_max, y_max)
center = overlay.cell_center("B3", w, h)            # -> (px_x, px_y)
label = overlay.cell_from_pixel(120, 300, w, h)     # -> "B3"
```

**Integration points**:
- **Server API**: `?grid=N` query parameter on `/reset` and `/step`
- **Viewer**: `--grid N` CLI flag + G key toggle
- **Programmatic**: `GridOverlay` class in `flairsim.core.grid`

### 8. Observation response format

```json
{
  "step": 42,
  "done": false,
  "drone_state": {
    "x": 923456.2,
    "y": 6345123.8,
    "z": 100.0,
    "heading": 0.0,
    "step_count": 42,
    "total_distance": 523.4
  },
  "ground_footprint": 200.0,
  "ground_resolution": 0.4,
  "image_base64": "<base64 PNG, with grid overlay if enabled>",
  "image_width": 500,
  "image_height": 500,
  "images": {
    "AERIAL_RGBI": "<base64 PNG>",
    "DEM_ELEV": "<base64 PNG>",
    "SPOT_RGBI": "<base64 PNG>"
  },
  "result": null,
  "metadata": {
    "roi": "UU-S2-1",
    "primary_modality": "AERIAL_RGBI",
    "modalities": "['AERIAL_RGBI', 'DEM_ELEV', 'SPOT_RGBI']",
    "scenario_name": "Find target in Isere",
    "distance_to_target": "42.5"
  }
}
```

When `done=true`, the `result` field contains:
```json
{
  "success": true,
  "reason": "Agent declared FOUND within 42.5 m of target (radius: 50.0 m).",
  "steps_taken": 42,
  "distance_travelled": 523.4
}
```

---

## FLAIR-HUB data

FlairSim uses the [FLAIR-HUB](https://arxiv.org/abs/2506.07080) dataset published by
IGN (Institut national de l'information geographique et forestiere).

### Downloading data from HuggingFace

The full FLAIR-HUB dataset is hosted on HuggingFace:
[IGNF/FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB) (~720 GB total).

**Option 1: Automatic download** (recommended)

FlairSim can automatically download the data you need from HuggingFace.
Just specify `--domain` without `--data-dir`:

```bash
# Download AERIAL_RGBI for domain D006-2020 (default modality)
uv run python -m flairsim.server --domain D006-2020

# Download multiple modalities
uv run python -m flairsim.server --domain D006-2020 \
    --modalities AERIAL_RGBI DEM_ELEV

# With a specific ROI
uv run python -m flairsim.server --domain D006-2020 --roi UU-S2-1
```

Data is downloaded to a temporary directory and **automatically cleaned up**
when the server shuts down (Ctrl+C, SIGTERM, or normal exit). No manual
cleanup is needed.

**Option 2: Official GUI download tool** (for selecting specific domains/modalities)

```bash
# Download the official download script
wget https://huggingface.co/datasets/IGNF/FLAIR-HUB/resolve/main/flair-hub-HF-dl.py

# Run it (requires: pip install huggingface_hub)
python flair-hub-HF-dl.py
# A GUI opens; select the domains/modalities you want and click Download.
# Data is downloaded to ./FLAIR-HUB_download/ by default.
```

**Option 3: Direct download with `huggingface_hub`**

```python
from huggingface_hub import hf_hub_download

# Download a single domain/modality ZIP
hf_hub_download(
    repo_id="IGNF/FLAIR-HUB",
    repo_type="dataset",
    filename="data/D006-2020_AERIAL_RGBI.zip",
    local_dir="./FLAIR-HUB",
)
# Unzip, then point FlairSim at the result:
# flairsim-server --data-dir ./FLAIR-HUB/D006-2020_AERIAL_RGBI
```

**Option 4: Use the toy dataset** (for development/testing)

If you have the `FLAIR-HUB_TOY` subset locally, point directly at it:

```bash
uv run python -m flairsim.server \
    --data-dir /path/to/FLAIR-HUB_TOY/D006-2020_AERIAL_RGBI

# Or with multi-modality (all D006-2020 modalities):
uv run python -m flairsim.server \
    --data-dir /path/to/FLAIR-HUB_TOY \
    --domain D006-2020
```

### Supported modalities

| Modality | Suffix | Resolution | Bands | Type |
|----------|--------|------------|-------|------|
| AERIAL_RGBI | `AERIAL_RGBI` | 0.2 m/px | 4 (RGBI) | uint8 |
| AERIAL_RLT_PAN | `AERIAL-RLT_PAN` | 0.4 m/px | 1 (PAN) | uint8 |
| DEM_ELEV | `DEM_ELEV` | 0.2 m/px | 2 (DSM+DTM) | float32 |
| SPOT_RGBI | `SPOT_RGBI` | 1.6 m/px | 4 (RGBI) | uint16 |
| SENTINEL1_ASC_TS | `SENTINEL1-ASC_TS` | 10.24 m/px | 2 (VV,VH) | float32, TS |
| SENTINEL1_DESC_TS | `SENTINEL1-DESC_TS` | 10.24 m/px | 2 (VV,VH) | float32, TS |
| SENTINEL2_TS | `SENTINEL2_TS` | 10.24 m/px | 10 | uint16, TS |
| LABEL_COSIA | `AERIAL_LABEL-COSIA` | 0.2 m/px | 1 | uint8 |
| LABEL_LPIS | `ALL_LABEL-LPIS` | 0.2 m/px | 3 | uint8 |
| SENTINEL2_MSK_SC | `SENTINEL2_MSK-SC` | 10.24 m/px | 2 | uint16, TS |

All modalities cover exactly 102.4m x 102.4m per patch.

### Data structure

FLAIR-HUB data (whether downloaded from HuggingFace or locally available)
uses a **flat layout**: all domain x modality directories are siblings:

```
FLAIR-HUB/                              # <-- --data-dir or --data-root points here
  D006-2020_AERIAL_RGBI/                # domain=D006-2020, modality=AERIAL_RGBI
    UU-S2-1/
      D006_AERIAL_RGBI_UU-S2-1_000-000.tif
      ...
    FF-S1-14/
      ...
  D006-2020_DEM_ELEV/                   # same domain, different modality
    UU-S2-1/
      ...
  D006-2020_SPOT_RGBI/
    ...
  D012-2019_AERIAL_RGBI/                # different domain, same level
    ...
  D012-2019_DEM_ELEV/
    ...
  GLOBAL_ALL_MTD/                       # metadata (ignored by simulator)
```

Use `--domain D006-2020` (or the scenario's `dataset.domain` field) to
select which domain's modalities to load from a flat root.

Each patch is a 512 x 512 pixel GeoTIFF in EPSG:2154 (Lambert-93) covering
102.4m x 102.4m at 0.2 m/px (for AERIAL_RGBI).  Patches within a ROI form
a contiguous grid that the simulator stitches into a seamless flyable map.

---

## Camera model

The camera always points nadir (straight down). Ground coverage depends on altitude:

| Altitude | Footprint | GSD (m/px) |
|----------|-----------|-----------|
| 10 m | 20 m | 0.04 |
| 50 m | 100 m | 0.20 (native resolution) |
| 100 m | 200 m | 0.40 |
| 200 m | 400 m | 0.80 |
| 500 m | 1000 m | 2.00 |

Formula: `footprint = 2 * altitude * tan(FOV/2)`, GSD = footprint / image_size.

---

## Running the tests

```bash
uv run pytest            # 460 tests, ~6s
uv run pytest -v         # verbose
uv run pytest tests/test_server.py  # server + SSE tests
uv run pytest tests/test_grid.py    # grid overlay tests
uv run pytest tests/test_multimodality.py  # multi-modality tests
uv run pytest tests/test_scenario.py       # scenario tests
uv run pytest tests/test_downloader.py     # HuggingFace downloader tests
```

---

## Project structure

```
simulateur/
  flairsim/                  # Main package
    __init__.py              # Public API exports (22 symbols)
    data/                    # Data acquisition from HuggingFace
      downloader.py          # HubDownloader: download, extract, cleanup
    map/                     # GeoTIFF tile loading and spatial queries
      modality.py            # Modality enum + discover_modalities(), pick_primary_modality()
      tile_loader.py         # TileInfo, TileData, read_tile(), normalize_to_uint8()
      map_manager.py         # MapManager, MapBounds
    drone/                   # Drone state, physics, camera
      drone.py               # Drone, DroneConfig, DroneState
      camera.py              # CameraModel, CameraConfig
      telemetry.py           # FlightLog, TelemetryRecord
    core/                    # Simulator engine
      action.py              # Action, ActionType (MOVE / FOUND / STOP)
      observation.py         # Observation, EpisodeResult
      simulator.py           # FlairSimulator, SimulatorConfig
      scenario.py            # Scenario, ScenarioLoader, ScenarioTarget
      grid.py                # GridOverlay, GridConfig (SoM/Scaffold prompting)
    server/                  # HTTP REST API + SSE
      __main__.py            # `python -m flairsim.server`
      app.py                 # FastAPI app factory, routes, SSE broadcast
      cli.py                 # CLI argument parsing
    viewer/                  # Desktop visualization (pygame, 3 modes)
      __main__.py            # `python -m flairsim.viewer`
      viewer.py              # FlairViewer, keyboard controls, remote modes
      remote.py              # ViewerObservation adapter (local <-> server)
      minimap.py             # Minimap overlay (with target marker)
      hud.py                 # HUD overlay (with scenario info)
  scenarios/                 # Scenario YAML files
    find_target_D006.yaml    # Example scenario
    quick_explore_D012.yaml  # Example scenario
  tests/                     # 460 tests (14 files)
  docs/                      # Technical documentation
  pyproject.toml             # Package config (uv/hatchling)
```

---

## Documentation

Detailed documentation is in the `docs/` directory:

| File | Content |
|------|---------|
| [docs/api.md](docs/api.md) | Complete API reference (all classes, methods, server endpoints, SSE) |
| [docs/architecture.md](docs/architecture.md) | Module architecture, data flow, multi-modality, scenarios, grid overlay |
| [docs/contributing.md](docs/contributing.md) | Development setup, tests, code style, project structure |

When the server is running, interactive API documentation (Swagger UI) is
also available at `http://127.0.0.1:8000/docs`.

---

## References

1. **FLAIR-HUB** -- Music, A. et al. (2025). *FLAIR-HUB: A large-scale multimodal
   dataset for land-cover mapping.* arXiv:2506.07080.

2. **FlySearch** -- Majumdar, A. et al. (2025). *FlySearch: A benchmark for evaluating
   VLM exploration capabilities with a UAV.* NeurIPS 2025. arXiv:2506.02896.

3. **Set-of-Mark** -- Yang, J. et al. (2023). *Set-of-Mark prompting unleashes
   extraordinary visual grounding in GPT-4V.* arXiv:2310.11441.

4. **Scaffold** -- Lei, L. et al. (2024). *Scaffolding coordinates to promote
   vision-language coordination in large multi-modal models.* arXiv:2402.12058.

---

## License

MIT
