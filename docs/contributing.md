# Contributing

Development guide for the FlairSim project.

---

## Development setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) >= 0.4
- Git

uv gere automatiquement la version de Python (3.11, epinglee dans
`.python-version`) et le virtual environment (`.venv/`).

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd simulateur

# Install everything (cree .venv, installe Python 3.11 si besoin,
# resout les deps depuis uv.lock, installe flairsim en editable)
uv sync --all-extras
```

C'est tout. Le lockfile `uv.lock` garantit des builds reproductibles
pour toute l'equipe.

### Commandes courantes

| Action | Commande |
|--------|----------|
| Installer tout | `uv sync --all-extras` |
| Installer sans viewer | `uv sync --extra server --dev` |
| Lancer les tests | `uv run pytest` |
| Lancer le serveur | `uv run python -m flairsim.server --data-dir path/to/data` |
| Ajouter une dependance | `uv add <package>` |
| Ajouter une dep de dev | `uv add --group dev <package>` |
| Mettre a jour les deps | `uv lock --upgrade` |

### Fichiers geres par uv

| Fichier | Role | Git |
|---------|------|-----|
| `pyproject.toml` | Dependances et config projet | oui |
| `uv.lock` | Versions resolues (reproductibilite) | oui |
| `.python-version` | Version Python epinglee (3.11) | oui |
| `.venv/` | Virtual environment local | **non** (dans .gitignore) |

---

## Running tests

```bash
# Full test suite (268 tests)
uv run pytest

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x

# Run specific test module
uv run pytest tests/test_simulator.py

# Run tests matching a keyword
uv run pytest -k "drone"
uv run pytest -k "map_manager"

# Run with coverage (if installed)
uv run pytest --cov=flairsim
```

### Test organisation

| File | Count | Scope |
|------|-------|-------|
| `test_action.py` | 13 | Action, ActionType |
| `test_camera.py` | 15 | CameraConfig, CameraModel |
| `test_drone.py` | 39 | DroneConfig, DroneState, Drone |
| `test_map_manager.py` | 44 | MapManager (integration, synthetic GeoTIFFs) |
| `test_modality.py` | 11 | Modality, ModalitySpec |
| `test_observation.py` | 14 | Observation, EpisodeResult |
| `test_server.py` | 42 | HTTP server endpoints + SSE (FastAPI TestClient / uvicorn) |
| `test_simulator.py` | 47 | FlairSimulator (integration, synthetic GeoTIFFs) |
| `test_telemetry.py` | 16 | TelemetryRecord, FlightLog |
| `test_tile_loader.py` | 16 | TileInfo, TileData, parse/read functions |

**Unit tests** (171 tests): Test classes in isolation with numpy arrays
and mock data.  No disk I/O.

**Integration tests** (91 tests): Create temporary synthetic GeoTIFF grids
on disk using `rasterio`, then run `MapManager` and `FlairSimulator`
against them.  The server tests also fall into this category, using
FastAPI's `TestClient` with synthetic data.

**SSE tests** (6 tests): Start a real `uvicorn.Server` in a background
thread and connect via `httpx.AsyncClient` + `asyncio.wait()`.  These
validate the full SSE pipeline: connection, `connected` comment,
observation push on `/reset` and `/step`, event structure, and data
consistency with the HTTP response.

---

## Code style

### Linting

```bash
# Check for issues
uv run ruff check flairsim/ tests/

# Auto-fix safe issues
uv run ruff check --fix flairsim/ tests/

# Format code
uv run ruff format flairsim/ tests/
```

### Configuration (from `pyproject.toml`)

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "D", "UP"]
ignore = ["D100", "D104", "D203", "D213"]
```

Enabled rule sets:
- **E/W**: pycodestyle errors and warnings
- **F**: pyflakes (unused imports, undefined names)
- **I**: isort (import ordering)
- **N**: pep8-naming
- **D**: pydocstyle (docstring conventions)
- **UP**: pyupgrade (modern Python syntax)

### Conventions

- **Line length**: 88 characters (Black-compatible)
- **Python version**: 3.11 (pinned in `.python-version`)
- **Docstrings**: NumPy-style with `Parameters`, `Returns`, `Raises` sections
- **Type hints**: All public APIs are fully typed
- **Frozen dataclasses**: Use `@dataclass(frozen=True, slots=True)` for
  configuration and value objects
- **Logging**: Use `logging.getLogger(__name__)` in each module. No `print()`.

---

## Project structure

```
simulateur/
  flairsim/          # Main package (6 subpackages)
    __init__.py      # Public API (17 exports)
    map/             # GeoTIFF loading, spatial queries
    drone/           # Drone state, physics, camera, telemetry
    core/            # Simulator engine, actions, observations
    server/          # HTTP REST API (FastAPI) + SSE push
    viewer/          # Pygame visualisation (optional, 3 modes)
      remote.py      # ViewerObservation adapter (local ↔ server)
  tests/             # Test suite (268 tests)
  docs/              # Technical documentation
  pyproject.toml     # Package + deps config
  uv.lock            # Lockfile (reproductibilite)
  .python-version    # Python version pinning
  README.md          # Project overview
```

---

## Adding a new module

1. Create your module file under the appropriate subpackage.
2. Add any new public classes to `flairsim/__init__.py`'s `__all__` list.
3. Write tests in `tests/test_<module>.py`.
4. Run `uv run pytest` and `uv run ruff check` before committing.

---

## Adding a dependency

```bash
# Runtime dependency
uv add numpy>=2.0

# Optional extra
uv add --optional viewer pygame>=2.5

# Dev-only dependency
uv add --group dev coverage
```

After adding, `uv.lock` is automatically updated. Commit both
`pyproject.toml` and `uv.lock`.

---

## Git workflow

- Keep commits focused and well-described.
- Run `uv run pytest` before every commit.
- Run `uv run ruff check` before every commit.
- Always commit `uv.lock` alongside `pyproject.toml` changes.
- Never commit `.venv/`.

---

## Dependencies

### Core (required)

| Package | Min version | Purpose |
|---------|-------------|---------|
| numpy | >= 1.24 | Array operations |
| rasterio | >= 1.3 | GeoTIFF I/O (wraps GDAL) |
| Pillow | >= 9.0 | Image resizing |

### Optional extras

| Package | Extra | Purpose |
|---------|-------|---------|
| pygame | `viewer` | Desktop visualisation |
| fastapi | `server` | HTTP REST API |
| uvicorn | `server` | ASGI server |
| sse-starlette | `server` | Server-Sent Events for push notifications |

### Dev dependencies (`[dependency-groups]`)

| Package | Purpose |
|---------|---------|
| pytest | Test runner |
| pytest-asyncio | Async test support (SSE tests) |
| ruff | Linter / formatter |
| httpx | HTTP client (for FastAPI TestClient + SSE tests) |
