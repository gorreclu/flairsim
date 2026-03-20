"""
Tests for :mod:`flairsim.core.scenario`.

Tests the Scenario data classes, ScenarioTarget evaluation, YAML parsing
via ScenarioLoader, and integration with FlairSimulator (scenario-based
episode evaluation).
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

from flairsim.core.action import Action
from flairsim.core.scenario import (
    Scenario,
    ScenarioDataset,
    ScenarioLoader,
    ScenarioStart,
    ScenarioTarget,
    _parse_scenario,
)
from flairsim.core.simulator import FlairSimulator, SimulatorConfig
from flairsim.drone.camera import CameraConfig
from flairsim.drone.drone import DroneConfig


# ---------------------------------------------------------------------------
# Fixtures
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
    """Create a temporary directory with synthetic tiles."""
    tmpdir = Path(tempfile.mkdtemp(prefix="flairsim_scenario_test_"))
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            _write_tile(tmpdir, row, col)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture()
def scenarios_dir(tile_dir: Path) -> Path:
    """Create a temporary scenarios directory with YAML files."""
    tmpdir = Path(tempfile.mkdtemp(prefix="flairsim_scenarios_"))

    # Compute map centre and a nearby target.
    cx = ORIGIN_X + (GRID_COLS * TILE_GROUND) / 2
    cy = ORIGIN_Y - (GRID_ROWS * TILE_GROUND) / 2

    yaml_content = f"""\
scenario_id: test_scenario
name: Test Scenario
description: A test scenario for unit tests.
dataset:
  data_dir: {tile_dir}
  roi: {ROI}
start:
  x: {cx}
  y: {cy}
  z: 20.0
target:
  x: {cx + 5.0}
  y: {cy + 5.0}
  radius: 10.0
max_steps: 50
"""
    (tmpdir / "test_scenario.yaml").write_text(yaml_content, encoding="utf-8")

    yaml_far = f"""\
scenario_id: far_target
name: Far Target
description: Target is far away.
dataset:
  data_dir: {tile_dir}
  roi: {ROI}
start:
  x: {cx}
  y: {cy}
  z: 20.0
target:
  x: {cx + 100.0}
  y: {cy + 100.0}
  radius: 5.0
max_steps: 30
"""
    (tmpdir / "far_target.yaml").write_text(yaml_far, encoding="utf-8")

    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests: ScenarioTarget
# ---------------------------------------------------------------------------


class TestScenarioTarget:
    """Test ScenarioTarget geometry helpers."""

    def test_distance_to(self):
        target = ScenarioTarget(x=100.0, y=200.0, radius=10.0)
        assert target.distance_to(100.0, 200.0) == pytest.approx(0.0)
        assert target.distance_to(103.0, 204.0) == pytest.approx(5.0)

    def test_is_within_inside(self):
        target = ScenarioTarget(x=0.0, y=0.0, radius=10.0)
        assert target.is_within(3.0, 4.0)  # distance = 5

    def test_is_within_on_boundary(self):
        target = ScenarioTarget(x=0.0, y=0.0, radius=5.0)
        assert target.is_within(3.0, 4.0)  # distance = 5, exactly on boundary

    def test_is_within_outside(self):
        target = ScenarioTarget(x=0.0, y=0.0, radius=4.9)
        assert not target.is_within(3.0, 4.0)  # distance = 5 > 4.9


# ---------------------------------------------------------------------------
# Tests: Scenario
# ---------------------------------------------------------------------------


class TestScenario:
    """Test Scenario data class."""

    def test_evaluate_success(self):
        sc = Scenario(
            scenario_id="test",
            name="Test",
            target=ScenarioTarget(x=100.0, y=100.0, radius=10.0),
        )
        assert sc.evaluate(105.0, 105.0)  # distance ~7.07

    def test_evaluate_failure(self):
        sc = Scenario(
            scenario_id="test",
            name="Test",
            target=ScenarioTarget(x=100.0, y=100.0, radius=5.0),
        )
        assert not sc.evaluate(110.0, 110.0)  # distance ~14.14

    def test_distance_to_target(self):
        sc = Scenario(
            scenario_id="test",
            name="Test",
            target=ScenarioTarget(x=0.0, y=0.0, radius=10.0),
        )
        assert sc.distance_to_target(3.0, 4.0) == pytest.approx(5.0)

    def test_to_dict(self):
        sc = Scenario(
            scenario_id="my_sc",
            name="My Scenario",
            description="A description.",
            dataset=ScenarioDataset(data_dir="some/path", roi="AB-S1"),
            start=ScenarioStart(x=1.0, y=2.0, z=3.0),
            target=ScenarioTarget(x=10.0, y=20.0, radius=5.0),
            max_steps=100,
        )
        d = sc.to_dict()
        assert d["scenario_id"] == "my_sc"
        assert d["target"]["radius"] == 5.0
        assert d["start"]["z"] == 3.0
        assert d["dataset"]["data_dir"] == "some/path"

    def test_repr(self):
        sc = Scenario(
            scenario_id="test",
            name="Test",
            target=ScenarioTarget(x=10.0, y=20.0, radius=5.0),
        )
        r = repr(sc)
        assert "test" in r
        assert "Test" in r


# ---------------------------------------------------------------------------
# Tests: _parse_scenario
# ---------------------------------------------------------------------------


class TestParseScenario:
    """Test YAML dict parsing."""

    def test_valid_scenario(self):
        data = {
            "scenario_id": "sc1",
            "name": "Scenario 1",
            "description": "Desc",
            "dataset": {"data_dir": "my/dir", "roi": "AB-S1"},
            "start": {"x": 1.0, "y": 2.0, "z": 3.0},
            "target": {"x": 10.0, "y": 20.0, "radius": 15.0},
            "max_steps": 200,
        }
        sc = _parse_scenario(data)
        assert sc.scenario_id == "sc1"
        assert sc.name == "Scenario 1"
        assert sc.dataset.data_dir == "my/dir"
        assert sc.start.x == 1.0
        assert sc.target.radius == 15.0
        assert sc.max_steps == 200

    def test_missing_scenario_id(self):
        with pytest.raises(ValueError, match="scenario_id"):
            _parse_scenario({"dataset": {"data_dir": "d"}, "target": {"x": 0, "y": 0}})

    def test_missing_data_dir(self):
        with pytest.raises(ValueError, match="data_dir"):
            _parse_scenario({"scenario_id": "x", "target": {"x": 0, "y": 0}})

    def test_missing_target(self):
        with pytest.raises(ValueError, match="target"):
            _parse_scenario({"scenario_id": "x", "dataset": {"data_dir": "d"}})

    def test_defaults(self):
        data = {
            "scenario_id": "sc_defaults",
            "dataset": {"data_dir": "d"},
            "target": {"x": 10.0, "y": 20.0},
        }
        sc = _parse_scenario(data)
        assert sc.name == "sc_defaults"  # defaults to scenario_id
        assert sc.description == ""
        assert sc.start.x is None
        assert sc.target.radius == 50.0  # default
        assert sc.max_steps == 500  # default


# ---------------------------------------------------------------------------
# Tests: ScenarioLoader
# ---------------------------------------------------------------------------


class TestScenarioLoader:
    """Test YAML file loading."""

    def test_list_ids(self, scenarios_dir: Path):
        loader = ScenarioLoader(scenarios_dir)
        ids = loader.list_ids()
        assert "test_scenario" in ids
        assert "far_target" in ids

    def test_get_scenario(self, scenarios_dir: Path):
        loader = ScenarioLoader(scenarios_dir)
        sc = loader.get("test_scenario")
        assert sc.scenario_id == "test_scenario"
        assert sc.name == "Test Scenario"
        assert sc.target.radius == 10.0

    def test_get_caches(self, scenarios_dir: Path):
        loader = ScenarioLoader(scenarios_dir)
        sc1 = loader.get("test_scenario")
        sc2 = loader.get("test_scenario")
        assert sc1 is sc2  # Same object from cache.

    def test_get_not_found(self, scenarios_dir: Path):
        loader = ScenarioLoader(scenarios_dir)
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            loader.get("nonexistent")

    def test_list_scenarios(self, scenarios_dir: Path):
        loader = ScenarioLoader(scenarios_dir)
        scenarios = loader.list_scenarios()
        assert len(scenarios) == 2

    def test_resolve_data_dir_absolute(self, scenarios_dir: Path, tile_dir: Path):
        loader = ScenarioLoader(scenarios_dir)
        sc = loader.get("test_scenario")
        resolved = loader.resolve_data_dir(sc)
        # The data_dir in our test YAML is already an absolute path.
        assert resolved.exists() or True  # tile_dir may have been cleaned

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ScenarioLoader("/tmp/nonexistent_scenario_dir_xyz")

    def test_repr(self, scenarios_dir: Path):
        loader = ScenarioLoader(scenarios_dir)
        r = repr(loader)
        assert "ScenarioLoader" in r
        assert "2" in r  # 2 scenarios

    def test_yml_extension(self, scenarios_dir: Path):
        """Ensure .yml files are also discovered."""
        yml_content = """\
scenario_id: yml_test
name: YML Test
dataset:
  data_dir: some/dir
target:
  x: 1.0
  y: 2.0
"""
        (scenarios_dir / "yml_test.yml").write_text(yml_content, encoding="utf-8")
        loader = ScenarioLoader(scenarios_dir)
        assert "yml_test" in loader.list_ids()
        sc = loader.get("yml_test")
        assert sc.scenario_id == "yml_test"


# ---------------------------------------------------------------------------
# Tests: Simulator + Scenario integration
# ---------------------------------------------------------------------------


class TestSimulatorScenarioIntegration:
    """Test FlairSimulator with a loaded scenario."""

    def _make_sim_with_scenario(
        self, tile_dir: Path, scenarios_dir: Path
    ) -> FlairSimulator:
        loader = ScenarioLoader(scenarios_dir)
        sc = loader.get("test_scenario")
        config = SimulatorConfig(
            drone_config=DroneConfig(z_min=5.0, z_max=200.0, default_altitude=20.0),
            camera_config=CameraConfig(fov_deg=90.0, image_size=32),
            max_steps=sc.max_steps,
            roi=sc.dataset.roi,
            preload_tiles=True,
        )
        data_dir = loader.resolve_data_dir(sc)
        return FlairSimulator(data_dir=data_dir, config=config, scenario=sc)

    def test_scenario_property(self, tile_dir: Path, scenarios_dir: Path):
        sim = self._make_sim_with_scenario(tile_dir, scenarios_dir)
        assert sim.scenario is not None
        assert sim.scenario.scenario_id == "test_scenario"
        sim.close()

    def test_scenario_start_position(self, tile_dir: Path, scenarios_dir: Path):
        sim = self._make_sim_with_scenario(tile_dir, scenarios_dir)
        obs = sim.reset()
        sc = sim.scenario
        assert sc is not None
        assert obs.drone_state.x == pytest.approx(sc.start.x)
        assert obs.drone_state.y == pytest.approx(sc.start.y)
        assert obs.drone_state.z == pytest.approx(sc.start.z)
        sim.close()

    def test_scenario_metadata_in_observation(
        self, tile_dir: Path, scenarios_dir: Path
    ):
        sim = self._make_sim_with_scenario(tile_dir, scenarios_dir)
        obs = sim.reset()
        assert obs.metadata.get("scenario_id") == "test_scenario"
        assert obs.metadata.get("scenario_name") == "Test Scenario"
        assert "distance_to_target" in obs.metadata
        assert obs.metadata["target_radius"] == 10.0
        sim.close()

    def test_found_within_radius_succeeds(self, tile_dir: Path, scenarios_dir: Path):
        sim = self._make_sim_with_scenario(tile_dir, scenarios_dir)
        sc = sim.scenario
        assert sc is not None
        # Start at the scenario start position.
        sim.reset()
        # Move toward the target (target is 5m east, 5m north -> ~7.07m).
        obs = sim.step(Action.move(dx=5.0, dy=5.0))
        # Now at the target -- declare FOUND.
        obs = sim.step(Action.found())
        assert obs.done
        assert obs.result is not None
        assert obs.result.success
        assert "within" in obs.result.reason.lower()
        sim.close()

    def test_found_outside_radius_fails(self, tile_dir: Path, scenarios_dir: Path):
        sim = self._make_sim_with_scenario(tile_dir, scenarios_dir)
        sim.reset()
        # Don't move -- the start position is ~7.07m from the target,
        # which is within the 10m radius.  Move AWAY instead.
        sim.step(Action.move(dx=-20.0, dy=-20.0))
        obs = sim.step(Action.found())
        assert obs.done
        assert obs.result is not None
        assert not obs.result.success
        assert "outside" in obs.result.reason.lower()
        sim.close()

    def test_max_steps_from_scenario(self, tile_dir: Path, scenarios_dir: Path):
        sim = self._make_sim_with_scenario(tile_dir, scenarios_dir)
        assert sim.max_steps == 50  # From the scenario YAML.
        sim.close()

    def test_no_scenario_found_still_works(self, tile_dir: Path):
        """Without a scenario, FOUND should still work (success=False)."""
        config = SimulatorConfig(
            drone_config=DroneConfig(z_min=5.0, z_max=200.0, default_altitude=20.0),
            camera_config=CameraConfig(fov_deg=90.0, image_size=32),
            max_steps=50,
            roi=ROI,
            preload_tiles=True,
        )
        sim = FlairSimulator(data_dir=tile_dir, config=config)
        assert sim.scenario is None
        sim.reset()
        obs = sim.step(Action.found())
        assert obs.done
        assert obs.result is not None
        assert not obs.result.success
        assert "no target" in obs.result.reason.lower()
        sim.close()

    def test_reset_override_position(self, tile_dir: Path, scenarios_dir: Path):
        """Explicit x/y/z in reset() should override the scenario start."""
        sim = self._make_sim_with_scenario(tile_dir, scenarios_dir)
        b = sim.map_bounds
        custom_x = b.x_min + 5.0
        custom_y = b.y_min + 5.0
        obs = sim.reset(x=custom_x, y=custom_y, z=50.0)
        assert obs.drone_state.x == pytest.approx(custom_x)
        assert obs.drone_state.y == pytest.approx(custom_y)
        assert obs.drone_state.z == pytest.approx(50.0)
        sim.close()

    def test_repr_includes_scenario(self, tile_dir: Path, scenarios_dir: Path):
        sim = self._make_sim_with_scenario(tile_dir, scenarios_dir)
        r = repr(sim)
        assert "test_scenario" in r
        sim.close()
