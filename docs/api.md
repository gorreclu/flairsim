# API Reference

Complete API reference for all public classes and functions exported by
the `flairsim` package.

---

## Package-level imports

All commonly-used classes are available directly from the top-level package:

```python
from flairsim import (
    Action, ActionType,
    CameraConfig, CameraModel,
    Drone, DroneConfig, DroneState,
    EpisodeResult, FlairSimulator, FlightLog,
    GridConfig, GridOverlay,
    MapBounds, MapManager,
    Modality, ModalitySpec,
    Observation,
    Scenario, ScenarioLoader, ScenarioTarget,
    SimulatorConfig,
    TelemetryRecord,
    discover_modalities, pick_primary_modality,
)
```

---

## `flairsim.core` -- Simulation engine

### `ActionType` (Enum)

Type of agent action.

| Member | Value | Description |
|--------|-------|-------------|
| `MOVE` | `"move"` | Move the drone (default) |
| `FOUND` | `"found"` | Declare target found |
| `STOP` | `"stop"` | End episode voluntarily |

---

### `Action` (frozen dataclass)

A single agent command.  Displacement is in metres, relative to current
position.

**Attributes**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dx` | `float` | `0.0` | Eastward displacement (metres) |
| `dy` | `float` | `0.0` | Northward displacement (metres) |
| `dz` | `float` | `0.0` | Upward displacement (metres) |
| `action_type` | `ActionType` | `MOVE` | Type of action |

**Factory methods**:

| Method | Description |
|--------|-------------|
| `Action.move(dx, dy, dz)` | Create a MOVE action |
| `Action.found(dx, dy, dz)` | Create a FOUND action (with optional positional adjustment) |
| `Action.stop()` | Create a STOP action (dx=dy=dz=0) |

**Example**:

```python
# Move 10m east, 5m north, descend 3m
action = Action(dx=10.0, dy=5.0, dz=-3.0)

# Using factory methods
action = Action.move(dx=10.0, dy=5.0)
action = Action.found()
action = Action.stop()
```

---

### `EpisodeResult` (frozen dataclass)

Final outcome of an episode (only meaningful when `obs.done == True`).

| Attribute | Type | Description |
|-----------|------|-------------|
| `success` | `bool` | Whether the agent correctly identified the target |
| `reason` | `str` | Human-readable explanation |
| `steps_taken` | `int` | Total steps in the episode |
| `distance_travelled` | `float` | Total horizontal distance (metres) |

---

### `Observation` (dataclass)

Simulator output returned by `reset()` and `step()`.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `image` | `np.ndarray` | Camera image for the *primary* modality, shape `(bands, H, W)` |
| `drone_state` | `DroneState` | Current position and counters |
| `step` | `int` | Current step number (0 on `reset()`) |
| `done` | `bool` | `True` if episode has ended |
| `result` | `EpisodeResult \| None` | Set only when `done=True` |
| `ground_footprint` | `float` | Camera footprint side length (metres) |
| `ground_resolution` | `float` | Ground sampling distance (m/px) |
| `metadata` | `dict` | Extra info (ROI, data_dir, max_steps, scenario info, modalities, ...) |
| `images` | `dict[str, np.ndarray]` | Per-modality images keyed by name (e.g. `"AERIAL_RGBI"`, `"DEM_ELEV"`). Each `(bands, H, W)`. Empty in legacy single-modality mode. |

**Metadata keys** (when applicable):

| Key | Description |
|-----|-------------|
| `roi` | Loaded ROI name |
| `data_dir` | Path to the data directory |
| `max_steps` | Episode step limit |
| `primary_modality` | Name of the primary modality |
| `modalities` | List of all loaded modality names |
| `scenario_id` | Active scenario ID (if scenario mode) |
| `scenario_name` | Active scenario name |
| `distance_to_target` | Distance from drone to target (metres) |
| `target_x`, `target_y` | Target coordinates |
| `target_radius` | Acceptance radius |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `position` | `tuple[float, float, float]` | Shortcut to `(x, y, z)` |
| `altitude` | `float` | Current altitude (metres) |
| `success` | `bool` | Whether episode ended successfully |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `image_rgb()` | `np.ndarray (H, W, 3)` | Extract RGB channels in channel-last layout |

---

### `SimulatorConfig`

Configuration for `FlairSimulator`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drone_config` | `DroneConfig \| None` | `None` | Drone physical limits |
| `camera_config` | `CameraConfig \| None` | `None` | Camera sensor params |
| `max_steps` | `int` | `500` | Max steps per episode |
| `roi` | `str \| None` | `None` | ROI to load (auto-select if `None`) |
| `preload_tiles` | `bool` | `True` | Load all tiles at init |

---

### `FlairSimulator`

Main simulation engine.

**Constructor**:

```python
FlairSimulator(
    data_dir: str | Path,
    config: SimulatorConfig | None = None,
    scenario: Scenario | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Path to FLAIR-HUB data directory. Can be a single-modality dir (e.g. `D004-2021_AERIAL_RGBI`) or a parent dir with multiple modality sub-dirs. |
| `config` | Full configuration. `None` uses defaults. |
| `scenario` | Predefined mission scenario. `None` = free-flight mode. |

**Attributes (public)**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `map_manager` | `MapManager` | The loaded map for the primary modality |
| `map_managers` | `dict[str, MapManager]` | All loaded maps, keyed by modality name |
| `primary_modality` | `str \| None` | Name of the primary modality (e.g. `"AERIAL_RGBI"`) |
| `drone` | `Drone` | The simulated drone |
| `camera` | `CameraModel` | The nadir camera |
| `flight_log` | `FlightLog` | Telemetry log for current episode |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `reset(x, y, z)` | `Observation` | Start a new episode. `None` args default to scenario start / map centre / default altitude |
| `step(action)` | `Observation` | Advance one step. Raises `RuntimeError` if episode ended. |
| `close()` | `None` | Release resources (currently no-op, reserved for future) |
| `random_start_position(rng, margin)` | `(float, float)` | Generate random `(x, y)` within map bounds |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `is_running` | `bool` | Whether an episode is active |
| `step_count` | `int` | Current step number |
| `max_steps` | `int` | Maximum steps per episode |
| `map_bounds` | `MapBounds` | Spatial extent of loaded map |
| `scenario` | `Scenario \| None` | The active scenario, or `None` in free-flight mode |

---

### `Scenario` (frozen dataclass)

A complete mission scenario. Loaded from YAML via `ScenarioLoader`.

**Attributes**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario_id` | `str` | -- | Unique identifier (YAML filename stem) |
| `name` | `str` | -- | Short human-readable name |
| `description` | `str` | `""` | Longer mission description |
| `dataset` | `ScenarioDataset` | -- | Where to load map data from |
| `start` | `ScenarioStart` | `ScenarioStart()` | Drone starting position |
| `target` | `ScenarioTarget` | -- | What the agent must find |
| `max_steps` | `int` | `500` | Episode step limit |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(drone_x, drone_y)` | `bool` | `True` if drone is within target acceptance radius |
| `distance_to_target(drone_x, drone_y)` | `float` | Distance from drone to target centre (metres) |
| `to_dict()` | `dict` | Serialise to JSON-friendly dictionary |

---

### `ScenarioTarget` (frozen dataclass)

Target location the agent must find.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `float` | -- | Target easting (metres) |
| `y` | `float` | -- | Target northing (metres) |
| `radius` | `float` | `50.0` | Acceptance radius (metres) |

**Methods**: `distance_to(px, py) -> float`, `is_within(px, py) -> bool`

---

### `ScenarioLoader`

Load and manage scenario YAML files from a directory.

**Constructor**:

```python
ScenarioLoader(
    scenarios_dir: str | Path,
    data_root: str | Path | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `scenarios_dir` | Directory containing `*.yaml` / `*.yml` scenario files |
| `data_root` | Root directory for resolving relative `data_dir` paths. `None` = cwd |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `get(scenario_id)` | `Scenario` | Load a scenario by its ID (filename stem) |
| `list_ids()` | `list[str]` | Sorted list of available scenario IDs |
| `list_scenarios()` | `list[Scenario]` | Load and return all scenarios |
| `resolve_data_dir(scenario)` | `Path` | Resolve scenario's `data_dir` to an absolute path |

**Properties**: `scenarios_dir -> Path`, `data_root -> Path`

---

### `GridOverlay`

NxN grid overlay with alphanumeric cell labels for VLM prompting.

**Constructor**:

```python
GridOverlay(n: int, config: GridConfig | None = None)
```

| Parameter | Description |
|-----------|-------------|
| `n` | Grid size (rows = columns). Must be in `[1, 26]`. |
| `config` | Visual parameters. `None` uses defaults. |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `n` | `int` | Grid size |
| `config` | `GridConfig` | Visual configuration |
| `cell_labels` | `list[str]` | All labels in row-major order (e.g. `['A1', 'A2', 'B1', 'B2']`) |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `draw(image)` | `np.ndarray (H, W, 3)` | Draw grid on `(H, W, 3)` uint8 image. Returns a new array. |
| `draw_on_surface(surface)` | `None` | Draw grid on a pygame Surface (in-place) |
| `cell_bounds(label, w, h)` | `(x_min, y_min, x_max, y_max)` | Pixel bounding box of a cell |
| `cell_center(label, w, h)` | `(px_x, px_y)` | Pixel centre of a cell |
| `cell_from_pixel(px_x, px_y, w, h)` | `str` | Determine which cell a pixel falls in |

**Example**:

```python
from flairsim import GridOverlay

overlay = GridOverlay(n=4)

# Draw on a camera image
annotated = overlay.draw(image_rgb)  # (H, W, 3) uint8

# Coordinate conversions
bounds = overlay.cell_bounds("B3", 500, 500)  # (250, 125, 375, 250)
center = overlay.cell_center("B3", 500, 500)  # (312, 187)
label = overlay.cell_from_pixel(300, 180, 500, 500)  # "B3"
```

---

### `GridConfig` (frozen dataclass)

Visual parameters for the grid overlay.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_color` | `tuple[int, int, int]` | `(255, 255, 255)` | RGB colour of grid lines |
| `line_alpha` | `float` | `0.6` | Line opacity (0.0-1.0) |
| `line_width` | `int` | `2` | Line width in pixels |
| `label_color` | `tuple[int, int, int]` | `(255, 255, 255)` | Label text colour |
| `label_bg_color` | `tuple[int, int, int] \| None` | `(0, 0, 0)` | Label background colour. `None` = no background |
| `label_bg_alpha` | `float` | `0.5` | Label background opacity |
| `font_scale` | `float` | `1.0` | Relative font size (1.0 = auto-sized) |

---

## `flairsim.drone` -- Drone and sensors

### `DroneConfig` (frozen dataclass)

Immutable physical limits for the drone.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `z_min` | `float` | `10.0` | Minimum altitude (metres). Must be > 0 |
| `z_max` | `float` | `500.0` | Maximum altitude (metres) |
| `max_step_distance` | `float` | `inf` | Max displacement magnitude per step (metres) |
| `default_altitude` | `float` | `100.0` | Spawn altitude when no z is given |

**Validation**: `__post_init__` enforces `z_min > 0`, `z_max > z_min`,
`max_step_distance > 0`, and `z_min <= default_altitude <= z_max`.

---

### `DroneState` (dataclass)

Mutable snapshot of drone kinematic state.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `float` | `0.0` | Easting (EPSG:2154, metres) |
| `y` | `float` | `0.0` | Northing (EPSG:2154, metres) |
| `z` | `float` | `100.0` | Altitude above ground (metres) |
| `heading` | `float` | `0.0` | Compass bearing, degrees CW from north |
| `step_count` | `int` | `0` | Steps since last reset |
| `total_distance` | `float` | `0.0` | Cumulative horizontal distance (metres) |

**Properties**: `position -> (x, y, z)`, `position_2d -> (x, y)`

**Methods**: `copy() -> DroneState`

---

### `MoveResult` (frozen dataclass)

Outcome of a single displacement command.

| Attribute | Type | Description |
|-----------|------|-------------|
| `dx_requested` | `float` | Raw eastward displacement requested |
| `dy_requested` | `float` | Raw northward displacement requested |
| `dz_requested` | `float` | Raw vertical displacement requested |
| `dx_actual` | `float` | Actual eastward displacement after clamping |
| `dy_actual` | `float` | Actual northward displacement after clamping |
| `dz_actual` | `float` | Actual vertical displacement after clamping |
| `was_clipped` | `bool` | `True` if any component was modified |

---

### `Drone`

Simulated quadrotor with 3-DoF position control.

**Constructor**:

```python
Drone(
    config: DroneConfig | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
)
```

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `reset(x, y, z=None, heading=0)` | `DroneState` | Place drone at position, clear counters |
| `move(dx, dy, dz=0)` | `MoveResult` | Apply relative displacement with clamping |
| `set_bounds(x_bounds, y_bounds)` | `None` | Update horizontal bounding box |
| `is_within_bounds()` | `bool` | Check all bounds |

**Properties**: `state -> DroneState` (copy), `config -> DroneConfig`,
`x_bounds`, `y_bounds`

---

### `CameraConfig` (frozen dataclass)

Immutable camera sensor parameters.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `fov_deg` | `float` | `90.0` | Full field-of-view angle (degrees). Must be in (0, 180). |
| `image_size` | `int` | `500` | Output image side length (pixels). Must be >= 1. |

**Properties**: `fov_rad -> float`

---

### `CameraModel`

Nadir-looking camera converting altitude to ground observations.

**Constructor**: `CameraModel(config: CameraConfig | None = None)`

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `ground_half_extent(z)` | `float` | Half footprint at altitude z: `z * tan(fov/2)` |
| `ground_footprint_size(z)` | `float` | Full footprint side length: `2 * half_extent` |
| `ground_resolution(z)` | `float` | Ground sampling distance (m/px) |
| `capture(map_manager, x, y, z)` | `np.ndarray` | Capture nadir image, shape `(bands, image_size, image_size)` |

**Properties**: `config -> CameraConfig`, `image_size -> int`

---

### `TelemetryRecord` (frozen dataclass)

Per-step state snapshot.

| Attribute | Type | Description |
|-----------|------|-------------|
| `step` | `int` | 0-based step index |
| `x`, `y`, `z` | `float` | Drone position after action applied |
| `dx`, `dy`, `dz` | `float` | Actual displacement applied |
| `ground_footprint` | `float` | Camera footprint side length (metres) |
| `was_clipped` | `bool` | Whether displacement was clipped |
| `metadata` | `dict` | Arbitrary extra data |

---

### `FlightLog`

Accumulates telemetry records over an episode.

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `append(record)` | `None` | Add a telemetry record |
| `clear()` | `None` | Remove all records |
| `to_dicts()` | `list[dict]` | Convert to list of plain dicts |
| `to_csv(filepath)` | `None` | Write to CSV file |
| `to_json(filepath, indent=2)` | `None` | Write to JSON file |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `records` | `Sequence[TelemetryRecord]` | All entries (read-only) |
| `total_distance` | `float` | Total horizontal distance (metres) |
| `total_steps` | `int` | Number of steps recorded |
| `altitude_range` | `(float, float) \| None` | `(min_z, max_z)` or `None` |
| `trajectory_2d` | `list[(float, float)]` | Ordered `(x, y)` positions |
| `clips_count` | `int` | Steps where displacement was clipped |

**Other methods**: `bounding_box() -> (x_min, y_min, x_max, y_max) | None`,
`__len__()`, `__getitem__(index)`

---

## `flairsim.map` -- Geospatial tile management

### `ModalitySpec` (frozen dataclass)

Specification of a FLAIR-HUB data modality.

| Attribute | Type | Description |
|-----------|------|-------------|
| `dir_suffix` | `str` | Directory naming suffix (e.g. `"AERIAL_RGBI"`) |
| `pixel_size_m` | `float` | Ground sampling distance (m/px) |
| `patch_pixels` | `int` | Patch side length in pixels |
| `bands` | `int` | Number of spectral bands |
| `dtype` | `str` | NumPy dtype string (e.g. `"uint8"`) |
| `is_time_series` | `bool` | Whether modality is temporal |
| `description` | `str` | Human-readable description |

---

### `Modality` (Enum)

All 10 data modalities available in FLAIR-HUB.

| Member | Suffix | Resolution | Bands | Type |
|--------|--------|------------|-------|------|
| `AERIAL_RGBI` | `AERIAL_RGBI` | 0.2 m/px | 4 (RGBI) | uint8 |
| `AERIAL_RLT_PAN` | `AERIAL-RLT_PAN` | 0.4 m/px | 1 (PAN) | uint8 |
| `DEM_ELEV` | `DEM_ELEV` | 0.2 m/px | 2 (DSM+DTM) | float32 |
| `SPOT_RGBI` | `SPOT_RGBI` | 1.6 m/px | 4 (RGBI) | uint16 |
| `SENTINEL1_ASC_TS` | `SENTINEL1-ASC_TS` | 10.24 m/px | 2 (VV,VH) | float32, TS |
| `SENTINEL1_DESC_TS` | `SENTINEL1-DESC_TS` | 10.24 m/px | 2 (VV,VH) | float32, TS |
| `SENTINEL2_TS` | `SENTINEL2_TS` | 10.24 m/px | 10 | uint16, TS |
| `LABEL_COSIA` | `AERIAL_LABEL-COSIA` | 0.2 m/px | 1 | uint8 |
| `LABEL_LPIS` | `ALL_LABEL-LPIS` | 0.2 m/px | 3 | uint8 |
| `SENTINEL2_MSK_SC` | `SENTINEL2_MSK-SC` | 10.24 m/px | 2 | uint16, TS |

All modalities cover exactly 102.4m x 102.4m per patch.

**Properties**: `spec -> ModalitySpec`, `patch_ground_size_m -> float`

**Class methods**: `from_dir_suffix(suffix) -> Modality | None`

---

### Modality discovery functions

| Function | Returns | Description |
|----------|---------|-------------|
| `discover_modalities(parent_dir)` | `dict[Modality, Path]` | Scan a parent directory for sub-dirs matching known modality suffixes |
| `pick_primary_modality(modalities)` | `Modality` | Choose the primary modality from available ones (prefers AERIAL_RGBI) |
| `is_single_modality_dir(data_dir)` | `bool` | Check if a directory contains tiles directly (single modality) vs. modality sub-dirs |

**`discover_modalities` example**:

```python
from flairsim import discover_modalities, Modality

mods = discover_modalities("FLAIR-HUB_TOY/D006-2018")
# {Modality.AERIAL_RGBI: Path(...), Modality.DEM_ELEV: Path(...), ...}
```

---

### `MapBounds` (frozen dataclass)

Axis-aligned bounding box in world coordinates (EPSG:2154, metres).

| Attribute | Type | Description |
|-----------|------|-------------|
| `x_min` | `float` | Western boundary |
| `y_min` | `float` | Southern boundary |
| `x_max` | `float` | Eastern boundary |
| `y_max` | `float` | Northern boundary |

**Properties**: `width -> float`, `height -> float`, `center -> (float, float)`

**Methods**: `contains(x, y) -> bool`, `intersects(other) -> bool`

---

### `MapManager`

Core spatial engine: tile discovery, grid geometry, region extraction.

**Constructor**:

```python
MapManager(
    data_dir: str | Path,
    roi: str | None = None,
    preload: bool = True,
)
```

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Root directory with `.tif` files for one modality |
| `roi` | ROI to load. `None` auto-selects the largest |
| `preload` | Load all tiles at init (`True`) or lazily (`False`) |

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `bounds` | `MapBounds` | Bounding box of loaded tile grid |
| `grid_rows` | `int` | Number of tile rows |
| `grid_cols` | `int` | Number of tile columns |
| `tile_pixel_size` | `int` | Tile side length in pixels |
| `tile_ground_size` | `float` | Tile side length on ground (metres) |
| `pixel_size_m` | `float` | Ground sampling distance (m/px) |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `get_region(x_center, y_center, half_extent, output_size)` | `np.ndarray` | Extract square region, shape `(bands, H, W)` |
| `get_label_at(x, y)` | `int \| None` | Label value at a point |
| `world_to_grid(x, y)` | `(int, int)` | World coords to grid `(row, col)` |
| `world_to_pixel(x, y)` | `(float, float)` | World coords to mosaic pixel coords |
| `pixel_to_world(px, py)` | `(float, float)` | Mosaic pixel coords to world |
| `list_available_rois()` | `list[str]` | Re-scan and return all ROI names |

**Properties**: `roi_name -> str`, `n_tiles_loaded -> int`, `n_tiles_total -> int`

---

### Tile loader functions

| Function | Returns | Description |
|----------|---------|-------------|
| `parse_tile_coords(filepath)` | `(int, int) \| None` | Extract `(row, col)` from filename |
| `parse_roi_from_path(filepath)` | `str \| None` | Extract ROI id from path |
| `read_tile(filepath)` | `TileData` | Read a GeoTIFF tile from disk |
| `normalize_to_uint8(data, low_pct, high_pct)` | `np.ndarray` | Percentile contrast stretch to uint8 |

---

## `flairsim.viewer` -- Interactive desktop viewer

Optional pygame-based viewer for manual flight and visualisation.
Install with `uv sync --extra viewer`.

### CLI entry point

The viewer supports three modes:

```bash
# Local mode (default) -- run a local simulator and fly manually
uv run python -m flairsim.viewer --data-dir path/to/D004-2021_AERIAL_RGBI

# Multi-modality local mode
uv run python -m flairsim.viewer --data-dir path/to/FLAIR-HUB_TOY/D006-2018

# With grid overlay
uv run python -m flairsim.viewer --data-dir path/to/data --grid 4

# Observe mode -- watch a remote server's activity via SSE
uv run python -m flairsim.viewer --mode observe --server-url http://localhost:8000

# Fly mode -- pilot a remote server via HTTP keypresses
uv run python -m flairsim.viewer --mode fly --server-url http://localhost:8000
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `local` | Viewer mode: `local`, `observe`, or `fly` |
| `--server-url` | `http://localhost:8000` | Server URL (observe/fly modes) |
| `--data-dir` | -- | Path to FLAIR-HUB data directory (local mode only, required) |
| `--roi` | auto | ROI to load (local mode) |
| `--max-steps` | `500` | Max steps per episode (local mode) |
| `--altitude` | `100` | Default altitude (m) (local mode) |
| `--image-size` | `500` | Camera resolution (px) (local mode) |
| `--fov` | `90` | Camera FOV (degrees) (local mode) |
| `--window-size` | `800` | Viewer window size (px) |
| `--move-step` | `20` | Movement per key press (m) |
| `--grid` | -- | Grid overlay size NxN (e.g. `--grid 4`). Press G to toggle. |
| `--no-preload` | -- | Lazy tile loading (local mode) |
| `-v` | -- | Debug logging |

### Viewer modes

| Mode | Flag | Description |
|------|------|-------------|
| **local** | `--mode local` | Creates a local `FlairSimulator`, flies directly. Default mode. |
| **observe** | `--mode observe` | Connects to a running server via SSE (`GET /events`). Read-only: watches what an external agent does. No keyboard flight. |
| **fly** | `--mode fly` | Connects to a running server, sends `POST /step` on keypresses, receives observation via HTTP response. |

### Keyboard controls

| Key | Action | Modes |
|-----|--------|-------|
| Z / Up | Move north (+dy) | local, fly |
| S / Down | Move south (-dy) | local, fly |
| Q / Left | Move west (-dx) | local, fly |
| D / Right | Move east (+dx) | local, fly |
| A | Descend (-dz) | local, fly |
| E | Ascend (+dz) | local, fly |
| +/- | Adjust move step size | local, fly |
| Space | Declare FOUND | local, fly |
| R | Reset episode | local, fly |
| H | Toggle HUD | all |
| M | Toggle minimap | all |
| Tab | Cycle modality (multi-modality mode) | all |
| G | Toggle grid overlay | all |
| Escape | Quit | all |

### `ViewerConfig` (frozen dataclass)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_width` | `int` | `800` | Window width (px) |
| `window_height` | `int` | `800` | Window height (px) |
| `title` | `str` | `"FlairSim Viewer"` | Window title |
| `target_fps` | `int` | `30` | Target frame rate |
| `move_step` | `float` | `20.0` | Default movement step (m) |
| `altitude_step` | `float` | `10.0` | Altitude change per press (m) |
| `show_hud` | `bool` | `True` | Show HUD initially |
| `show_minimap` | `bool` | `True` | Show minimap initially |

### `FlairViewer`

**Constructor**:

```python
FlairViewer(
    config: ViewerConfig | None = None,
    map_bounds: MapBounds | None = None,
    grid: int | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `config` | Window and display settings |
| `map_bounds` | Map extent for minimap |
| `grid` | Grid overlay size NxN. `None` = no grid (toggle with G key). |

| Method | Description |
|--------|-------------|
| `run_manual(simulator)` | Run in local manual flight mode with keyboard controls |
| `run_remote_observe(server_url)` | Connect to server SSE stream and display observations (observe mode) |
| `run_remote_fly(server_url)` | Connect to server and pilot via HTTP keypresses (fly mode) |
| `show(obs) -> bool` | Display one `ViewerObservation`. Returns `False` if window closed |
| `open()` | Open the pygame window |
| `close()` | Close the window |
| `set_map_bounds(bounds)` | Set/update minimap extent |

---

### `ViewerObservation` (dataclass, `flairsim.viewer.remote`)

Source-agnostic observation used by all viewer rendering components.
Decouples the viewer from both the local `Observation` class and the
server JSON format.

| Attribute | Type | Description |
|-----------|------|-------------|
| `image_rgb` | `np.ndarray (H, W, 3)` | RGB image ready for display |
| `drone_state` | `ViewerDroneState` | Position and distance |
| `step` | `int` | Current step number |
| `done` | `bool` | Whether the episode has ended |
| `ground_footprint` | `float` | Camera footprint (m) |
| `ground_resolution` | `float` | Ground sampling distance (m/px) |
| `result` | `ViewerEpisodeResult \| None` | Episode outcome |
| `metadata` | `dict` | Arbitrary metadata (scenario info, modalities, ...) |
| `images_rgb` | `dict[str, np.ndarray]` | Per-modality `(H, W, 3)` uint8 RGB images, keyed by modality name. Empty in single-modality mode. |

**Factory methods**:

| Method | Description |
|--------|-------------|
| `ViewerObservation.from_observation(obs)` | Wrap a local `Observation` |
| `ViewerObservation.from_server_response(data)` | Parse a server JSON dict (from `/reset`, `/step`, or SSE event) |

### `ViewerDroneState` (frozen dataclass, `flairsim.viewer.remote`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `x` | `float` | Easting (m) |
| `y` | `float` | Northing (m) |
| `z` | `float` | Altitude (m) |
| `total_distance` | `float` | Cumulative distance (m) |

### `ViewerEpisodeResult` (frozen dataclass, `flairsim.viewer.remote`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `success` | `bool` | Whether the agent succeeded |
| `reason` | `str` | Human-readable explanation |

---

## `flairsim.server` -- HTTP REST API

The server module exposes the simulator as a local HTTP service so that
external programs (VLM agents, scripts, etc.) can pilot the drone via
simple JSON requests.  Install with `pip install flairsim[server]` or
`uv sync --extra server`.

### `create_app(...)` (factory function)

Create a FastAPI application wrapping a `FlairSimulator`.

```python
from flairsim.server import create_app

app = create_app(
    data_dir="path/to/D004-2021_AERIAL_RGBI",
    roi=None,                # auto-select largest ROI
    max_steps=500,
    drone_config=None,       # defaults
    camera_config=None,      # defaults
    preload_tiles=True,
    scenario_loader=None,    # ScenarioLoader or None
    scenario_id=None,        # initial scenario ID or None
    grid=None,               # grid overlay size or None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str \| Path` | -- | Path to FLAIR-HUB data directory |
| `roi` | `str \| None` | `None` | ROI to load (auto-select if `None`) |
| `max_steps` | `int` | `500` | Maximum steps per episode |
| `drone_config` | `DroneConfig \| None` | `None` | Drone physical limits |
| `camera_config` | `CameraConfig \| None` | `None` | Camera sensor params |
| `preload_tiles` | `bool` | `True` | Load all tiles at startup |
| `scenario_loader` | `ScenarioLoader \| None` | `None` | Loader for scenario YAML files |
| `scenario_id` | `str \| None` | `None` | Initial scenario to load at startup |
| `grid` | `int \| None` | `None` | Default grid overlay size NxN. Overridable per-request via `?grid=N`. |

Returns a `FastAPI` instance ready to be served with uvicorn.

---

### CLI entry point

```bash
# Recommended (works everywhere, including macOS editable installs)
uv run python -m flairsim.server --data-dir path/to/D004-2021_AERIAL_RGBI

# With scenarios and grid
uv run python -m flairsim.server \
    --data-dir path/to/data \
    --scenarios-dir scenarios/ \
    --data-root /data/FLAIR-HUB \
    --scenario find_target_D006 \
    --grid 4

# With options
uv run python -m flairsim.server \
    --data-dir path/to/data \
    --roi D004_2021_0504 \
    --max-steps 1000 \
    --host 127.0.0.1 \
    --port 8000
```

---

### Endpoints

#### `POST /reset`

Start a new episode.  Returns the initial observation.

**Request body** (optional):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `x` | `float \| null` | `null` | Start easting (m). `null` = scenario start or map centre |
| `y` | `float \| null` | `null` | Start northing (m). `null` = scenario start or map centre |
| `z` | `float \| null` | `null` | Start altitude (m). `null` = scenario start or default |
| `scenario_id` | `str \| null` | `null` | Scenario to load. `null` = keep current scenario or free flight |

**Query parameters**:

| Param | Type | Description |
|-------|------|-------------|
| `grid` | `int \| null` | Grid overlay size NxN. `0` disables. Persists across requests. |

**Response**: `ObservationResponse` (see below).

**Side effects**: Broadcasts the observation to all connected SSE
subscribers (see `GET /events`).  If `scenario_id` is set, a new
`FlairSimulator` is created with the scenario's dataset.

---

#### `POST /step`

Advance the simulation by one step.

**Request body**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dx` | `float` | `0.0` | Eastward displacement (m) |
| `dy` | `float` | `0.0` | Northward displacement (m) |
| `dz` | `float` | `0.0` | Upward displacement (m) |
| `action_type` | `str` | `"move"` | `"move"`, `"found"`, or `"stop"` |

**Query parameters**:

| Param | Type | Description |
|-------|------|-------------|
| `grid` | `int \| null` | Grid overlay size NxN. `0` disables. Persists across requests. |

**Response**: `ObservationResponse`.

**Error 409**: If no episode is active (call `/reset` first).

**Side effects**: Broadcasts the observation to all connected SSE
subscribers (see `GET /events`).

---

#### `GET /state`

Get current drone state without advancing the simulation.

**Response**: `DroneStateResponse`.

**Error 409**: If no episode is active.

---

#### `GET /telemetry`

Get the full flight log for the current episode.

**Response**: `TelemetryResponse`.

---

#### `GET /config`

Get simulator configuration and map information.

**Response**: `ConfigResponse`.

---

#### `GET /scenarios`

List all available scenarios.

**Response**: `ScenariosListResponse`.

Returns an empty list if no scenario loader is configured.

---

#### `GET /events`

Server-Sent Events (SSE) streaming endpoint.  Provides a persistent
connection that pushes every observation produced by `/reset` or `/step`
in real time.  Multiple clients can subscribe concurrently.

**Content-Type**: `text/event-stream`

**Protocol**:

1. The server sends a `: connected` comment immediately upon connection
   (used to flush HTTP headers).
2. Each observation is sent as an SSE event with type `observation` and
   a JSON-encoded `ObservationResponse` as the `data` field.
3. If no observation is produced for 30 seconds, the server sends a
   `: keep-alive` comment to prevent proxy/load-balancer timeouts.
4. The stream remains open until the client disconnects.

**Event format**:

```
event: observation
data: {"step": 0, "done": false, "drone_state": {...}, "image_base64": "...", ...}
```

**Backpressure**: Each subscriber has a 64-event queue.  If the queue is
full (slow consumer), the oldest event is dropped to make room for the
new one.

**Example (Python, httpx)**:

```python
import httpx

with httpx.Client(base_url="http://127.0.0.1:8000") as client:
    with client.stream("GET", "/events") as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                import json
                obs = json.loads(line[len("data:"):])
                print(f"Step {obs['step']}, done={obs['done']}")
```

**Example (curl)**:

```bash
curl -N http://127.0.0.1:8000/events
```

---

### Response models

#### `ObservationResponse`

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Current step number |
| `done` | `bool` | Whether the episode has ended |
| `drone_state` | `DroneStateResponse` | Current drone state |
| `ground_footprint` | `float` | Camera footprint side length (m) |
| `ground_resolution` | `float` | Ground sampling distance (m/px) |
| `image_base64` | `str` | PNG-encoded primary image in base64 (with grid overlay if enabled) |
| `image_width` | `int` | Image width in pixels |
| `image_height` | `int` | Image height in pixels |
| `result` | `EpisodeResultResponse \| null` | Set only when `done=true` |
| `metadata` | `dict` | Extra info (ROI, modalities, scenario, distance_to_target, ...) |
| `images` | `dict[str, str]` | Per-modality PNG images in base64, keyed by modality name. Empty in single-modality mode. |

#### `DroneStateResponse`

| Field | Type | Description |
|-------|------|-------------|
| `x` | `float` | Easting (EPSG:2154, m) |
| `y` | `float` | Northing (EPSG:2154, m) |
| `z` | `float` | Altitude (m) |
| `heading` | `float` | Compass bearing (degrees) |
| `step_count` | `int` | Steps since last reset |
| `total_distance` | `float` | Cumulative horizontal distance (m) |

#### `EpisodeResultResponse`

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the target was found |
| `reason` | `str` | Human-readable explanation |
| `steps_taken` | `int` | Total steps in the episode |
| `distance_travelled` | `float` | Total distance (m) |

#### `TelemetryResponse`

| Field | Type | Description |
|-------|------|-------------|
| `total_steps` | `int` | Steps recorded |
| `total_distance` | `float` | Total horizontal distance (m) |
| `clips_count` | `int` | Steps where displacement was clipped |
| `altitude_range` | `[float, float] \| null` | `[min_z, max_z]` or null |
| `records` | `list[TelemetryRecordResponse]` | All telemetry records |

#### `TelemetryRecordResponse`

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | 0-based step index |
| `x`, `y`, `z` | `float` | Drone position after action |
| `dx`, `dy`, `dz` | `float` | Actual displacement applied |
| `ground_footprint` | `float` | Camera footprint (m) |
| `was_clipped` | `bool` | Whether displacement was clipped |

#### `ConfigResponse`

| Field | Type | Description |
|-------|------|-------------|
| `data_dir` | `str` | Path to FLAIR-HUB data directory |
| `roi` | `str` | Loaded ROI name |
| `n_tiles` | `int` | Number of tiles loaded |
| `pixel_size_m` | `float` | Ground sampling distance (m/px) |
| `tile_ground_size` | `float` | Tile side length on ground (m) |
| `map_bounds` | `MapBoundsResponse` | Spatial extent |
| `drone` | `dict` | Drone config (z_min, z_max, ...) |
| `camera` | `dict` | Camera config (fov_deg, image_size) |
| `max_steps` | `int` | Max steps per episode |
| `is_running` | `bool` | Whether an episode is active |
| `scenario_id` | `str \| null` | Active scenario ID, or `null` in free-flight mode |

#### `ScenariosListResponse`

| Field | Type | Description |
|-------|------|-------------|
| `scenarios` | `list[ScenarioSummaryResponse]` | All available scenarios |

#### `ScenarioSummaryResponse`

| Field | Type | Description |
|-------|------|-------------|
| `scenario_id` | `str` | Scenario unique ID |
| `name` | `str` | Short name |
| `description` | `str` | Mission description |
| `max_steps` | `int` | Episode step limit |
| `target_radius` | `float` | Acceptance radius (m) |
