<div align="center">

# FlairSim

**Drone simulator over FLAIR-HUB aerial imagery**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-608%20passed-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![uv](https://img.shields.io/badge/pkg-uv-blueviolet.svg)](https://docs.astral.sh/uv/)

*Fly a virtual drone over real French aerial imagery from [FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB) (IGN). Benchmark VLM navigation agents with a web platform, leaderboard, and isolated sessions.*

**Fil rouge** -- MS Big Data / IA, Telecom Paris

</div>

---

## Architecture

```
                               +--------------------------------------+
                               |         Orchestrator (web/)           |
                               |     FastAPI -- port 8080              |
    Browser (SPA) -------------+                                       |
                               |   GET  /api/scenarios     -> list     |
    Notebook (VLM) ------------+   POST /api/sessions      -> create   |
                               |   ANY  /api/sessions/{id}/sim/*       |
                               |   GET  /api/leaderboard   -> rankings |
                               +----------+---------------------------+
                                          | spawns subprocesses
                        +-----------------+-----------------+
                        v                 v                 v
              +--------------+  +--------------+  +--------------+
              | flairsim-srv |  | flairsim-srv |  | flairsim-srv |
              | port 9001    |  | port 9002    |  | port 9003    |
              +--------------+  +--------------+  +--------------+
                        \                |               /
                         +---------------+---------------+
                         |       FLAIR-HUB GeoTIFF       |
                         |       tiles (0.2 m/px)        |
                         +-------------------------------+
```

**Two tiers:**
- **Orchestrator** (`flairsim/web/`) -- manages sessions, serves the SPA, proxies API calls, persists results in SQLite leaderboard
- **Simulator** (`flairsim/server/`) -- one subprocess per session on a unique port (9001-9099), unchanged from standalone usage

**Also available:** Pygame desktop viewer (3 modes: local / observe / fly) and direct simulator API for standalone usage.

---

## Quick start

```bash
# Install (uv required)
uv sync --all-extras

# Option 1: Web platform (browser UI + leaderboard)
uv run python -m flairsim.web --scenarios-dir scenarios/ --data-root data/
# Open http://127.0.0.1:8080

# Option 2: Standalone simulator (direct API)
uv run python -m flairsim.server --data-dir data/D004-2021_AERIAL_RGBI
# API at http://127.0.0.1:8000, Swagger at /docs

# Option 3: Auto-download from HuggingFace (no local data needed)
uv run python -m flairsim.server --domain D006-2020

# Option 4: Pygame viewer
uv run python -m flairsim.viewer --data-dir data/D004-2021_AERIAL_RGBI
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Real aerial imagery** | 0.2 m/px IGN orthophotos from FLAIR-HUB |
| **Web benchmark platform** | Browser SPA with scenario picker, cockpit canvas, minimap, leaderboard |
| **Session isolation** | Each session gets its own simulator subprocess |
| **Multi-modality** | Auto-discovers RGBI, DEM, SPOT, Sentinel-1/2, labels (10 modalities) |
| **Predefined scenarios** | YAML missions with start/target, prompts, evaluation |
| **Auto-download** | `--domain D006-2020` auto-downloads from HuggingFace |
| **SSE real-time push** | `GET /events` streams observations to viewers |
| **Grid overlay** | NxN alphanumeric grid for VLM spatial prompting (Set-of-Mark / Scaffold) |
| **Smooth movement** | Server-side micro-step decomposition + SSE frame queue for fluid animation |
| **SQLite leaderboard** | Ranked by success > fewer steps > shorter distance |
| **API key auth** | Bearer token for AI session creation |
| **Idle timeout** | Sessions auto-destroyed after 3 min inactivity |

---

## Project structure

```
flairsim/
  data/       HuggingFace auto-download (HubDownloader)
  map/        GeoTIFF tile loading, spatial queries, modality discovery
  drone/      Drone state, camera model, telemetry
  core/       Simulator engine, actions, observations, scenarios, grid overlay
  server/     HTTP REST API + SSE (FastAPI) -- simulator process
  web/        Web benchmark platform (orchestrator, leaderboard, SPA)
  viewer/     Pygame desktop viewer (HUD, minimap, 3 modes)
```

---

## Simulator API (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start episode. Body: `{x, y, z, scenario_id}`. Query: `?grid=N` |
| `/step` | POST | Send action. Body: `{dx, dy, dz, action_type, reason}` |
| `/state` | GET | Current drone state |
| `/snapshot` | GET | Last cached observation (for spectators joining mid-session) |
| `/telemetry` | GET | Full flight log |
| `/config` | GET | Simulator config, map bounds |
| `/scenarios` | GET | List available scenarios |
| `/overview` | GET | Full-ROI overview JPEG |
| `/events` | GET | SSE stream of observations |

**Action types:** `move` (default), `found` (declare target), `stop` (end episode)

---

## Orchestrator API (port 8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/scenarios` | GET | List scenarios with full details |
| `/api/scenarios/{id}` | GET | Scenario detail + prompt templates |
| `/api/scenarios/{id}/overview` | GET | Overview JPEG |
| `/api/scenarios/{id}/thumbnail` | GET | Square start-view thumbnail |
| `/api/sessions` | POST | Create session (spawns subprocess) |
| `/api/sessions` | GET | List active sessions |
| `/api/sessions/{id}` | GET | Session status |
| `/api/sessions/{id}` | DELETE | Destroy session |
| `/api/sessions/{id}/sim/{path}` | ANY | Proxy to simulator |
| `/api/leaderboard` | GET | Rankings (`?scenario_id=...&mode=...&limit=50`) |
| `/api/leaderboard/global` | GET | Global agent ranking (`?mode=ai&limit=100`) |
| `/api/leaderboard` | POST | Submit run result |
| `/api/leaderboard/submit` | POST | Alias (preferred for notebooks) |
| `/api/leaderboard/{id}` | GET | Run detail |
| `/api/status` | GET | Health check |

---

## Using from a notebook (AI agent)

```python
import httpx, os

API_KEY = os.environ["FLAIRSIM_API_KEY"]
api = httpx.Client(base_url="http://127.0.0.1:8080/api")

# Create AI session
session = api.post("/sessions", json={
    "scenario_id": "find_red_car_D004",
    "mode": "ai",
    "player_name": "MyAgent",
}, headers={"Authorization": f"Bearer {API_KEY}"}).json()

sid = session["session_id"]

# Reset + step loop
obs = api.post(f"/sessions/{sid}/sim/reset").json()
while not obs["done"]:
    action = {"dx": 10, "dy": 0, "dz": 0, "action_type": "move"}
    obs = api.post(f"/sessions/{sid}/sim/step", json=action).json()

# Submit to leaderboard
api.post("/leaderboard/submit", json={
    "scenario_id": "find_red_car_D004",
    "mode": "ai",
    "model_name": "MyAgent",
    "success": obs["result"]["success"],
    "steps_taken": obs["result"]["steps_taken"],
    "distance_travelled": obs["result"]["distance_travelled"],
})

# Cleanup
api.delete(f"/sessions/{sid}")
```

See the [tutorial notebook](notebooks/tutorial.ipynb) for a complete walkthrough. Launch it with:

```bash
uv run --with jupyter --with matplotlib jupyter lab notebooks/tutorial.ipynb
```

---

## Scenarios

Self-contained YAML missions defining data source, start/target, prompts:

```yaml
scenario_id: find_red_car_D004
name: Find red car in Aube
dataset:
  data_dir: D004-2021_AERIAL_RGBI
  source: local
start: { x: 986699.0, y: 6369901.0, z: 100.0 }
target: { x: 986607.0, y: 6369997.0, radius: 40.0 }
max_steps: 150
prompt:
  system: "You are a drone navigation agent..."
  user_template: "Position: ({x}, {y}, {z}). Step {step}/{max_steps}."
```

---

## Scoring Methodology

Every completed run (human or AI) receives a normalised score reflecting how efficiently the agent navigated to the target.

### Successful Runs — S ∈ [0, 100]

When the agent correctly locates the target and declares it found:

```
S = [ 0.3 · (D_min / D_agent)
    + 0.3 · (Step_min / Step_agent)
    + 0.3 · (t_min / t_agent)
    + 0.1 · c ] × 100
```

| Variable | Description |
|----------|-------------|
| `D_min` | Euclidean (straight-line) distance between the scenario start position and the target position |
| `D_agent` | Distance travelled by the current agent |
| `Step_min` | Minimum steps taken across all successful runs for this scenario |
| `Step_agent` | Steps taken by the current agent |
| `t_min` | Minimum duration (seconds) across all successful runs for this scenario |
| `t_agent` | Duration of the current agent's run |
| `c` | Confidence (0–1) optionally declared by the agent; defaults to 0 |

Reference values: D_min is a fixed geometric property of the scenario (Euclidean distance from start to target). Step_min and t_min are dynamically recomputed as new successful runs are recorded. Each efficiency ratio is capped at 1.0.

### Failed Runs — F ∈ [-100, 0]

When the agent fails (step limit reached or manual stop):

```
F = -100 × [ 0.5 · (1 - E) + 0.5 · c ]
```

| Variable | Description |
|----------|-------------|
| `E` | FOV coverage (exploration ratio, 0–1): fraction of the ROI explored |
| `c` | Confidence (0–1): high confidence on failure is penalised |

If the target was visible at any point during the run (`target_seen = true`), the penalty is multiplied by **1.5×** before clamping to −100.

### Global Ranking

The global leaderboard ranks agents (not individual runs). For each agent, only the best score per scenario is kept. Total scores are summed across all scenarios. AI and human agents are ranked separately.

---

## Keyboard controls

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| Z/W/Up | North | E | Ascend |
| S/Down | South | A | Descend |
| D/Right | East | +/- | Step size |
| Q/Left | West | Space | FOUND |
| Tab | Cycle modality | R | Reset |
| H | Toggle HUD | M | Toggle minimap |
| G | Toggle grid | Esc | Quit |

Works in both web SPA (AZERTY + QWERTY) and Pygame viewer.

---

## Camera model

Nadir pinhole camera. Ground coverage = `2 * altitude * tan(FOV/2)`.

| Altitude | Footprint | GSD |
|----------|-----------|-----|
| 10 m | 20 m | 0.04 m/px |
| 50 m | 100 m | 0.20 m/px (native) |
| 100 m | 200 m | 0.40 m/px |
| 500 m | 1000 m | 2.00 m/px |

---

## Tests

```bash
uv run pytest              # 554 tests, ~9s
uv run pytest -v           # verbose
uv run pytest -x           # stop on first failure
```

| Test file | Tests | Scope |
|-----------|-------|-------|
| `test_action.py` | 13 | Action, ActionType |
| `test_camera.py` | 15 | CameraModel |
| `test_drone.py` | 39 | Drone, DroneConfig |
| `test_map_manager.py` | 44 | MapManager |
| `test_modality.py` | 11 | Modality |
| `test_multimodality.py` | 12 | Multi-modality |
| `test_observation.py` | 14 | Observation |
| `test_scenario.py` | 49 | Scenario, ScenarioLoader |
| `test_server.py` | 48 | HTTP + SSE + Snapshot |
| `test_simulator.py` | 47 | FlairSimulator |
| `test_telemetry.py` | 16 | TelemetryRecord |
| `test_tile_loader.py` | 16 | TileLoader |
| `test_downloader.py` | 22 | HubDownloader |
| `test_grid.py` | 59 | GridOverlay |
| `test_sessions.py` | 52 | SessionManager |
| `test_leaderboard.py` | 30 | Leaderboard |
| `test_web_routes.py` | 67 | Orchestrator routes |
| **Total** | **554** | |

---

## Dependencies

| Group | Command | Packages |
|-------|---------|----------|
| Core | `uv sync` | numpy, rasterio, Pillow, pyyaml |
| Server | `uv sync --extra server` | + fastapi, uvicorn, sse-starlette, huggingface_hub, aiofiles, httpx |
| Viewer | `uv sync --extra viewer` | + pygame |
| Dev | `uv sync --dev` | + pytest, pytest-asyncio, ruff, httpx |
| All | `uv sync --all-extras` | Everything |

---

## Documentation

| Document | Content |
|----------|---------|
| [docs/guide.pdf](docs/guide.pdf) | Complete technical documentation (22 pages): architecture diagrams, all API endpoints, sequence diagrams, data flow |
| [notebooks/tutorial.ipynb](notebooks/tutorial.ipynb) | Interactive tutorial (`uv run --with jupyter jupyter lab notebooks/`) |
| Swagger UI (`/docs`) | Auto-generated API docs when server is running |

---

## References

1. **FLAIR-HUB** -- Music, A. et al. (2025). *A large-scale multimodal dataset for land-cover mapping.* arXiv:2506.07080.
2. **FlySearch** -- Majumdar, A. et al. (2025). *A benchmark for evaluating VLM exploration capabilities with a UAV.* NeurIPS 2025. arXiv:2506.02896.
3. **Set-of-Mark** -- Yang, J. et al. (2023). arXiv:2310.11441.
4. **Scaffold** -- Lei, L. et al. (2024). arXiv:2402.12058.

---

## License

MIT
