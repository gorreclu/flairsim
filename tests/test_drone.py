"""Unit tests for flairsim.drone.drone."""

import math

import pytest

from flairsim.drone.drone import Drone, DroneConfig, DroneState, MoveResult


# ---------------------------------------------------------------------------
# DroneConfig
# ---------------------------------------------------------------------------


class TestDroneConfig:
    """Tests for DroneConfig validation."""

    def test_defaults(self):
        cfg = DroneConfig()
        assert cfg.z_min == 10.0
        assert cfg.z_max == 500.0
        assert cfg.max_step_distance == float("inf")
        assert cfg.default_altitude == 100.0

    def test_custom_values(self):
        cfg = DroneConfig(z_min=5, z_max=200, max_step_distance=50, default_altitude=80)
        assert cfg.z_min == 5.0
        assert cfg.z_max == 200.0
        assert cfg.max_step_distance == 50.0
        assert cfg.default_altitude == 80.0

    def test_z_min_must_be_positive(self):
        with pytest.raises(ValueError, match="z_min must be positive"):
            DroneConfig(z_min=0)

    def test_z_min_negative(self):
        with pytest.raises(ValueError, match="z_min must be positive"):
            DroneConfig(z_min=-10)

    def test_z_max_must_exceed_z_min(self):
        with pytest.raises(ValueError, match="z_max.*must be greater"):
            DroneConfig(z_min=100, z_max=50)

    def test_z_max_equal_z_min(self):
        with pytest.raises(ValueError, match="z_max.*must be greater"):
            DroneConfig(z_min=100, z_max=100)

    def test_max_step_distance_must_be_positive(self):
        with pytest.raises(ValueError, match="max_step_distance must be positive"):
            DroneConfig(max_step_distance=0)

    def test_default_altitude_out_of_range(self):
        with pytest.raises(ValueError, match="default_altitude"):
            DroneConfig(z_min=10, z_max=100, default_altitude=200)

    def test_frozen(self):
        cfg = DroneConfig()
        with pytest.raises(AttributeError):
            cfg.z_min = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DroneState
# ---------------------------------------------------------------------------


class TestDroneState:
    """Tests for DroneState."""

    def test_defaults(self):
        s = DroneState()
        assert s.x == 0.0
        assert s.y == 0.0
        assert s.z == 100.0
        assert s.heading == 0.0
        assert s.step_count == 0
        assert s.total_distance == 0.0

    def test_position_property(self):
        s = DroneState(x=10.0, y=20.0, z=50.0)
        assert s.position == (10.0, 20.0, 50.0)

    def test_position_2d_property(self):
        s = DroneState(x=10.0, y=20.0, z=50.0)
        assert s.position_2d == (10.0, 20.0)

    def test_copy_independence(self):
        s = DroneState(x=10.0, y=20.0, z=50.0)
        c = s.copy()
        c.x = 999.0
        assert s.x == 10.0  # Original unchanged.

    def test_mutable(self):
        s = DroneState()
        s.x = 42.0
        assert s.x == 42.0


# ---------------------------------------------------------------------------
# MoveResult
# ---------------------------------------------------------------------------


class TestMoveResult:
    """Tests for MoveResult."""

    def test_no_clipping(self):
        r = MoveResult(
            dx_requested=10,
            dy_requested=5,
            dz_requested=0,
            dx_actual=10,
            dy_actual=5,
            dz_actual=0,
            was_clipped=False,
        )
        assert not r.was_clipped
        assert r.dx_requested == r.dx_actual

    def test_with_clipping(self):
        r = MoveResult(
            dx_requested=100,
            dy_requested=0,
            dz_requested=0,
            dx_actual=50,
            dy_actual=0,
            dz_actual=0,
            was_clipped=True,
        )
        assert r.was_clipped
        assert r.dx_requested == 100
        assert r.dx_actual == 50


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------


class TestDrone:
    """Tests for the Drone class."""

    # ---- Reset ----

    def test_reset_basic(self):
        drone = Drone()
        state = drone.reset(x=1000, y=6500, z=80)
        assert state.x == 1000.0
        assert state.y == 6500.0
        assert state.z == 80.0
        assert state.step_count == 0
        assert state.total_distance == 0.0

    def test_reset_default_altitude(self):
        drone = Drone(config=DroneConfig(default_altitude=120))
        state = drone.reset(x=0, y=0)
        assert state.z == 120.0

    def test_reset_clamps_altitude(self):
        drone = Drone(config=DroneConfig(z_min=20, z_max=200))
        state = drone.reset(x=0, y=0, z=5)
        assert state.z == 20.0  # Clamped to z_min.

    def test_reset_heading(self):
        drone = Drone()
        state = drone.reset(x=0, y=0, heading=450)
        assert state.heading == 90.0  # 450 % 360

    def test_reset_clears_counters(self):
        drone = Drone()
        drone.reset(x=0, y=0)
        drone.move(dx=10, dy=0)
        drone.move(dx=10, dy=0)
        assert drone.state.step_count == 2

        state2 = drone.reset(x=0, y=0)
        assert state2.step_count == 0
        assert state2.total_distance == 0.0

    def test_reset_returns_copy(self):
        drone = Drone()
        state = drone.reset(x=100, y=200)
        state.x = 999
        assert drone.state.x == 100.0  # Internal state unchanged.

    # ---- Move ----

    def test_move_simple(self):
        drone = Drone()
        drone.reset(x=0, y=0, z=100)
        result = drone.move(dx=10, dy=5, dz=0)
        assert not result.was_clipped
        assert result.dx_actual == 10.0
        assert result.dy_actual == 5.0
        assert drone.state.x == 10.0
        assert drone.state.y == 5.0

    def test_move_updates_counters(self):
        drone = Drone()
        drone.reset(x=0, y=0)
        drone.move(dx=3, dy=4)  # distance = 5
        assert drone.state.step_count == 1
        assert abs(drone.state.total_distance - 5.0) < 1e-9

    def test_move_altitude(self):
        drone = Drone()
        drone.reset(x=0, y=0, z=100)
        drone.move(dx=0, dy=0, dz=50)
        assert drone.state.z == 150.0

    def test_move_preserves_original_request(self):
        """MoveResult should store the original (dx, dy, dz) request."""
        drone = Drone(config=DroneConfig(max_step_distance=50))
        drone.reset(x=0, y=0, z=100)
        result = drone.move(dx=100, dy=0, dz=0)
        assert result.dx_requested == 100.0
        assert result.dx_actual == pytest.approx(50.0)
        assert result.was_clipped

    # ---- Altitude clamping ----

    def test_altitude_clamp_upper(self):
        drone = Drone(config=DroneConfig(z_max=200))
        drone.reset(x=0, y=0, z=180)
        result = drone.move(dx=0, dy=0, dz=100)
        assert drone.state.z == 200.0
        assert result.was_clipped

    def test_altitude_clamp_lower(self):
        drone = Drone(config=DroneConfig(z_min=20))
        drone.reset(x=0, y=0, z=30)
        result = drone.move(dx=0, dy=0, dz=-50)
        assert drone.state.z == 20.0
        assert result.was_clipped

    # ---- Horizontal clamping ----

    def test_x_bounds_clamp(self):
        drone = Drone(x_bounds=(0, 100))
        drone.reset(x=50, y=0, z=100)
        result = drone.move(dx=200, dy=0)
        assert drone.state.x == 100.0
        assert result.was_clipped

    def test_y_bounds_clamp(self):
        drone = Drone(y_bounds=(0, 100))
        drone.reset(x=0, y=50, z=100)
        result = drone.move(dx=0, dy=-200)
        assert drone.state.y == 0.0
        assert result.was_clipped

    def test_no_bounds_no_clamp(self):
        drone = Drone()
        drone.reset(x=0, y=0, z=100)
        result = drone.move(dx=999999, dy=999999)
        assert not result.was_clipped

    # ---- Max step distance ----

    def test_max_step_distance(self):
        drone = Drone(config=DroneConfig(max_step_distance=50))
        drone.reset(x=0, y=0, z=100)
        result = drone.move(dx=100, dy=0, dz=0)
        # Should be scaled down to 50m total distance.
        assert result.dx_actual == pytest.approx(50.0)
        assert result.dy_actual == pytest.approx(0.0)
        assert result.was_clipped

    def test_max_step_distance_3d(self):
        drone = Drone(config=DroneConfig(max_step_distance=10))
        drone.reset(x=0, y=0, z=100)
        result = drone.move(dx=30, dy=40, dz=0)
        # Original distance = 50, scaled to 10.
        actual_dist = math.sqrt(result.dx_actual**2 + result.dy_actual**2)
        assert actual_dist == pytest.approx(10.0, abs=0.01)

    def test_within_max_step_no_scaling(self):
        drone = Drone(config=DroneConfig(max_step_distance=100))
        drone.reset(x=0, y=0, z=100)
        result = drone.move(dx=3, dy=4)
        assert result.dx_actual == 3.0
        assert result.dy_actual == 4.0
        assert not result.was_clipped

    # ---- Bounds management ----

    def test_set_bounds(self):
        drone = Drone()
        assert drone.x_bounds is None
        drone.set_bounds(x_bounds=(0, 500), y_bounds=(100, 600))
        assert drone.x_bounds == (0, 500)
        assert drone.y_bounds == (100, 600)

    def test_is_within_bounds(self):
        drone = Drone(
            config=DroneConfig(z_min=10, z_max=200),
            x_bounds=(0, 100),
            y_bounds=(0, 100),
        )
        drone.reset(x=50, y=50, z=100)
        assert drone.is_within_bounds()

    def test_is_within_bounds_unbounded(self):
        drone = Drone()
        drone.reset(x=9999, y=9999, z=100)
        assert drone.is_within_bounds()

    # ---- State property returns copy ----

    def test_state_returns_copy(self):
        drone = Drone()
        drone.reset(x=100, y=200, z=50)
        s1 = drone.state
        s1.x = 999
        assert drone.state.x == 100.0

    # ---- Repr ----

    def test_repr(self):
        drone = Drone()
        drone.reset(x=100, y=200, z=80)
        r = repr(drone)
        assert "Drone" in r
        assert "100.0" in r
