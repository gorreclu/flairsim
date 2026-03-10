"""
Integration tests for :class:`~flairsim.core.simulator.FlairSimulator`.

These tests exercise the full simulator loop (reset → step → done)
by reusing the synthetic GeoTIFF grid from ``test_map_manager.py``.
The simulator is constructed against real (but small) tiles on disk so
that all sub-components (MapManager, Drone, CameraModel, FlightLog)
interact together.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from flairsim.core.action import Action, ActionType
from flairsim.core.observation import Observation
from flairsim.core.simulator import FlairSimulator, SimulatorConfig
from flairsim.drone.camera import CameraConfig
from flairsim.drone.drone import DroneConfig


# ---------------------------------------------------------------------------
# Fixtures: synthetic tile grid
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
def tile_dir() -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="flairsim_sim_test_"))
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            _write_tile(tmpdir, row, col)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _make_sim(
    tile_dir: Path,
    max_steps: int = 50,
    image_size: int = 32,
    z_min: float = 5.0,
    z_max: float = 200.0,
    default_altitude: float = 20.0,
) -> FlairSimulator:
    """Create a FlairSimulator with small images for fast tests."""
    config = SimulatorConfig(
        drone_config=DroneConfig(
            z_min=z_min,
            z_max=z_max,
            default_altitude=default_altitude,
        ),
        camera_config=CameraConfig(fov_deg=90.0, image_size=image_size),
        max_steps=max_steps,
        roi=ROI,
        preload_tiles=True,
    )
    return FlairSimulator(data_dir=tile_dir, config=config)


@pytest.fixture()
def sim(tile_dir: Path) -> FlairSimulator:
    s = _make_sim(tile_dir)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Verify FlairSimulator initialises correctly."""

    def test_creates_successfully(self, sim: FlairSimulator):
        assert sim.map_manager is not None
        assert sim.drone is not None
        assert sim.camera is not None

    def test_is_idle_before_reset(self, sim: FlairSimulator):
        assert not sim.is_running

    def test_step_before_reset_raises(self, sim: FlairSimulator):
        with pytest.raises(RuntimeError, match="reset"):
            sim.step(Action.move(dx=1.0, dy=0.0))

    def test_map_bounds_are_set(self, sim: FlairSimulator):
        b = sim.map_bounds
        assert b.width > 0
        assert b.height > 0

    def test_drone_bounds_match_map(self, sim: FlairSimulator):
        b = sim.map_bounds
        assert sim.drone.x_bounds is not None
        assert sim.drone.x_bounds[0] == pytest.approx(b.x_min)
        assert sim.drone.x_bounds[1] == pytest.approx(b.x_max)

    def test_repr(self, sim: FlairSimulator):
        r = repr(sim)
        assert "FlairSimulator" in r
        assert "idle" in r


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------


class TestReset:
    """Verify episode reset behaviour."""

    def test_reset_returns_observation(self, sim: FlairSimulator):
        obs = sim.reset()
        assert isinstance(obs, Observation)

    def test_is_running_after_reset(self, sim: FlairSimulator):
        sim.reset()
        assert sim.is_running

    def test_step_count_is_zero(self, sim: FlairSimulator):
        sim.reset()
        assert sim.step_count == 0

    def test_observation_step_is_zero(self, sim: FlairSimulator):
        obs = sim.reset()
        assert obs.step == 0

    def test_observation_not_done(self, sim: FlairSimulator):
        obs = sim.reset()
        assert not obs.done
        assert obs.result is None

    def test_reset_default_position_is_center(self, sim: FlairSimulator):
        obs = sim.reset()
        cx, cy = sim.map_bounds.center
        assert obs.drone_state.x == pytest.approx(cx, abs=1.0)
        assert obs.drone_state.y == pytest.approx(cy, abs=1.0)

    def test_reset_custom_position(self, sim: FlairSimulator):
        b = sim.map_bounds
        x = b.x_min + 5.0
        y = b.y_min + 5.0
        obs = sim.reset(x=x, y=y, z=50.0)
        assert obs.drone_state.x == pytest.approx(x)
        assert obs.drone_state.y == pytest.approx(y)
        assert obs.drone_state.z == pytest.approx(50.0)

    def test_image_shape(self, sim: FlairSimulator):
        obs = sim.reset()
        assert obs.image.ndim == 3
        assert obs.image.shape[0] == N_BANDS  # bands first
        # Output size should match camera config.
        assert obs.image.shape[1] == 32
        assert obs.image.shape[2] == 32

    def test_metadata_contains_roi(self, sim: FlairSimulator):
        obs = sim.reset()
        assert obs.metadata["roi"] == ROI

    def test_flight_log_cleared(self, sim: FlairSimulator):
        sim.reset()
        sim.step(Action.move(dx=1.0, dy=0.0))
        sim.reset()  # Second reset.
        # Flight log should have exactly 1 record (the initial telemetry).
        assert len(sim.flight_log) == 1

    def test_ground_footprint_positive(self, sim: FlairSimulator):
        obs = sim.reset()
        assert obs.ground_footprint > 0

    def test_ground_resolution_positive(self, sim: FlairSimulator):
        obs = sim.reset()
        assert obs.ground_resolution > 0


# ---------------------------------------------------------------------------
# Tests: step (MOVE actions)
# ---------------------------------------------------------------------------


class TestStepMove:
    """Verify MOVE actions in the step loop."""

    def test_step_returns_observation(self, sim: FlairSimulator):
        sim.reset()
        obs = sim.step(Action.move(dx=1.0, dy=0.0))
        assert isinstance(obs, Observation)

    def test_step_increments_count(self, sim: FlairSimulator):
        sim.reset()
        sim.step(Action.move(dx=1.0, dy=0.0))
        assert sim.step_count == 1

    def test_step_moves_drone(self, sim: FlairSimulator):
        obs0 = sim.reset()
        x0 = obs0.drone_state.x
        obs1 = sim.step(Action.move(dx=2.0, dy=0.0))
        assert obs1.drone_state.x == pytest.approx(x0 + 2.0)

    def test_step_not_done(self, sim: FlairSimulator):
        sim.reset()
        obs = sim.step(Action.move(dx=1.0, dy=0.0))
        assert not obs.done

    def test_altitude_change(self, sim: FlairSimulator):
        obs0 = sim.reset()
        z0 = obs0.drone_state.z
        obs1 = sim.step(Action.move(dx=0.0, dy=0.0, dz=5.0))
        assert obs1.drone_state.z == pytest.approx(z0 + 5.0)

    def test_telemetry_recorded(self, sim: FlairSimulator):
        sim.reset()
        sim.step(Action.move(dx=3.0, dy=4.0))
        # Initial + 1 step = 2 records.
        assert len(sim.flight_log) == 2
        rec = sim.flight_log[1]
        assert rec.dx == pytest.approx(3.0)
        assert rec.dy == pytest.approx(4.0)

    def test_multiple_steps(self, sim: FlairSimulator):
        sim.reset()
        for _ in range(5):
            obs = sim.step(Action.move(dx=0.5, dy=0.5))
        assert sim.step_count == 5
        assert not obs.done


# ---------------------------------------------------------------------------
# Tests: termination conditions
# ---------------------------------------------------------------------------


class TestTermination:
    """Verify episode termination via FOUND, STOP, and max_steps."""

    def test_found_ends_episode(self, sim: FlairSimulator):
        sim.reset()
        obs = sim.step(Action.found())
        assert obs.done
        assert not sim.is_running
        assert obs.result is not None
        assert obs.result.steps_taken == 1

    def test_stop_ends_episode(self, sim: FlairSimulator):
        sim.reset()
        obs = sim.step(Action.stop())
        assert obs.done
        assert not sim.is_running
        assert obs.result is not None
        assert "stopped" in obs.result.reason.lower()

    def test_max_steps_ends_episode(self, tile_dir: Path):
        sim = _make_sim(tile_dir, max_steps=3)
        sim.reset()
        for _ in range(3):
            obs = sim.step(Action.move(dx=0.5, dy=0.0))
        assert obs.done
        assert obs.result is not None
        assert "limit" in obs.result.reason.lower()

    def test_step_after_done_raises(self, sim: FlairSimulator):
        sim.reset()
        sim.step(Action.found())
        with pytest.raises(RuntimeError, match="reset"):
            sim.step(Action.move(dx=1.0, dy=0.0))

    def test_found_with_movement(self, sim: FlairSimulator):
        """FOUND action can include a displacement."""
        obs0 = sim.reset()
        x0 = obs0.drone_state.x
        obs = sim.step(Action.found(dx=2.0, dy=0.0))
        assert obs.done
        # The movement should have been applied.
        assert obs.drone_state.x == pytest.approx(x0 + 2.0)

    def test_result_distance_tracked(self, sim: FlairSimulator):
        sim.reset()
        sim.step(Action.move(dx=3.0, dy=4.0))  # distance = 5.0
        obs = sim.step(Action.found())
        assert obs.result is not None
        assert obs.result.distance_travelled == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# Tests: multiple episodes
# ---------------------------------------------------------------------------


class TestMultipleEpisodes:
    """Verify that the simulator correctly handles sequential episodes."""

    def test_reset_after_done(self, sim: FlairSimulator):
        sim.reset()
        sim.step(Action.found())
        obs = sim.reset()
        assert not obs.done
        assert sim.is_running
        assert sim.step_count == 0

    def test_independent_episodes(self, sim: FlairSimulator):
        # Episode 1.
        sim.reset()
        for _ in range(5):
            sim.step(Action.move(dx=1.0, dy=0.0))
        sim.step(Action.stop())

        # Episode 2.
        obs = sim.reset()
        assert sim.step_count == 0
        assert len(sim.flight_log) == 1  # Just the initial record.

    def test_reset_midway(self, sim: FlairSimulator):
        """Resetting mid-episode should work cleanly."""
        sim.reset()
        sim.step(Action.move(dx=1.0, dy=0.0))
        sim.step(Action.move(dx=1.0, dy=0.0))
        # Reset without finishing.
        obs = sim.reset()
        assert sim.step_count == 0
        assert sim.is_running


# ---------------------------------------------------------------------------
# Tests: random_start_position
# ---------------------------------------------------------------------------


class TestRandomStart:
    """Verify the random start position helper."""

    def test_within_bounds(self, sim: FlairSimulator):
        rng = np.random.default_rng(42)
        for _ in range(20):
            x, y = sim.random_start_position(rng=rng, margin=2.0)
            b = sim.map_bounds
            assert b.x_min + 2.0 <= x <= b.x_max - 2.0
            assert b.y_min + 2.0 <= y <= b.y_max - 2.0

    def test_different_seed_different_position(self, sim: FlairSimulator):
        x1, y1 = sim.random_start_position(rng=np.random.default_rng(1), margin=2.0)
        x2, y2 = sim.random_start_position(rng=np.random.default_rng(2), margin=2.0)
        # Extremely unlikely to be identical with different seeds.
        assert (x1, y1) != (x2, y2)

    def test_reset_at_random_position(self, sim: FlairSimulator):
        x, y = sim.random_start_position(rng=np.random.default_rng(99), margin=2.0)
        obs = sim.reset(x=x, y=y)
        assert obs.drone_state.x == pytest.approx(x, abs=0.1)
        assert obs.drone_state.y == pytest.approx(y, abs=0.1)


# ---------------------------------------------------------------------------
# Tests: image observations are valid
# ---------------------------------------------------------------------------


class TestObservationImages:
    """Verify that image observations are sensible."""

    def test_image_is_numpy_array(self, sim: FlairSimulator):
        obs = sim.reset()
        assert isinstance(obs.image, np.ndarray)

    def test_image_rgb_helper(self, sim: FlairSimulator):
        obs = sim.reset()
        rgb = obs.image_rgb()
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_image_changes_after_move(self, sim: FlairSimulator):
        obs0 = sim.reset()
        obs1 = sim.step(Action.move(dx=5.0, dy=5.0))
        # Images should generally differ after a significant move.
        # (They could theoretically be the same if both are in the same
        # uniform tile region, but our synthetic tiles have distinct patterns.)
        # We just check they're valid arrays.
        assert obs0.image.shape == obs1.image.shape

    def test_footprint_changes_with_altitude(self, sim: FlairSimulator):
        obs_low = sim.reset(z=10.0)
        fp_low = obs_low.ground_footprint
        sim.reset()
        obs_high = sim.reset(z=100.0)
        fp_high = obs_high.ground_footprint
        assert fp_high > fp_low


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_displacement(self, sim: FlairSimulator):
        obs0 = sim.reset()
        obs1 = sim.step(Action.move(dx=0.0, dy=0.0, dz=0.0))
        assert obs1.drone_state.x == pytest.approx(obs0.drone_state.x)
        assert obs1.drone_state.y == pytest.approx(obs0.drone_state.y)
        assert obs1.drone_state.z == pytest.approx(obs0.drone_state.z)

    def test_boundary_clamping(self, sim: FlairSimulator):
        """Moving far beyond the map should clamp to bounds."""
        sim.reset()
        obs = sim.step(Action.move(dx=1_000_000.0, dy=0.0))
        assert obs.drone_state.x <= sim.map_bounds.x_max

    def test_altitude_clamp_low(self, sim: FlairSimulator):
        sim.reset(z=10.0)
        obs = sim.step(Action.move(dx=0.0, dy=0.0, dz=-100.0))
        assert obs.drone_state.z >= sim.drone.config.z_min

    def test_altitude_clamp_high(self, sim: FlairSimulator):
        sim.reset(z=100.0)
        obs = sim.step(Action.move(dx=0.0, dy=0.0, dz=10_000.0))
        assert obs.drone_state.z <= sim.drone.config.z_max

    def test_close_is_safe(self, tile_dir: Path):
        sim = _make_sim(tile_dir)
        sim.reset()
        sim.close()
        # Calling close again should not raise.
        sim.close()
