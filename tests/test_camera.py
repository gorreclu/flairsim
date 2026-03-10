"""Unit tests for flairsim.drone.camera."""

import math

import pytest

from flairsim.drone.camera import CameraConfig, CameraModel


# ---------------------------------------------------------------------------
# CameraConfig
# ---------------------------------------------------------------------------


class TestCameraConfig:
    """Tests for CameraConfig validation."""

    def test_defaults(self):
        cfg = CameraConfig()
        assert cfg.fov_deg == 90.0
        assert cfg.image_size == 500

    def test_custom(self):
        cfg = CameraConfig(fov_deg=60.0, image_size=256)
        assert cfg.fov_deg == 60.0
        assert cfg.image_size == 256

    def test_fov_rad(self):
        cfg = CameraConfig(fov_deg=90.0)
        assert cfg.fov_rad == pytest.approx(math.pi / 2)

    def test_fov_too_large(self):
        with pytest.raises(ValueError, match="fov_deg"):
            CameraConfig(fov_deg=180.0)

    def test_fov_too_small(self):
        with pytest.raises(ValueError, match="fov_deg"):
            CameraConfig(fov_deg=0.0)

    def test_fov_negative(self):
        with pytest.raises(ValueError, match="fov_deg"):
            CameraConfig(fov_deg=-10.0)

    def test_image_size_too_small(self):
        with pytest.raises(ValueError, match="image_size"):
            CameraConfig(image_size=0)

    def test_frozen(self):
        cfg = CameraConfig()
        with pytest.raises(AttributeError):
            cfg.fov_deg = 45.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CameraModel -- geometry
# ---------------------------------------------------------------------------


class TestCameraModelGeometry:
    """Tests for CameraModel geometric computations."""

    def test_ground_half_extent_90deg(self):
        """At 90° FOV, half_extent = z * tan(45°) = z."""
        cam = CameraModel(CameraConfig(fov_deg=90.0))
        assert cam.ground_half_extent(100.0) == pytest.approx(100.0)

    def test_ground_half_extent_60deg(self):
        """At 60° FOV, half_extent = z * tan(30°)."""
        cam = CameraModel(CameraConfig(fov_deg=60.0))
        expected = 100.0 * math.tan(math.radians(30.0))
        assert cam.ground_half_extent(100.0) == pytest.approx(expected)

    def test_ground_footprint_size(self):
        """Footprint = 2 * half_extent."""
        cam = CameraModel(CameraConfig(fov_deg=90.0))
        assert cam.ground_footprint_size(100.0) == pytest.approx(200.0)

    def test_ground_resolution(self):
        """GSD = footprint / image_size."""
        cam = CameraModel(CameraConfig(fov_deg=90.0, image_size=500))
        # Footprint at 100m = 200m, GSD = 200/500 = 0.4 m/px
        assert cam.ground_resolution(100.0) == pytest.approx(0.4)

    def test_ground_resolution_varies_with_altitude(self):
        cam = CameraModel(CameraConfig(fov_deg=90.0, image_size=500))
        gsd_low = cam.ground_resolution(50.0)
        gsd_high = cam.ground_resolution(200.0)
        assert gsd_high > gsd_low

    def test_altitude_1m(self):
        """Extreme low altitude."""
        cam = CameraModel(CameraConfig(fov_deg=90.0, image_size=500))
        assert cam.ground_footprint_size(1.0) == pytest.approx(2.0)
        assert cam.ground_resolution(1.0) == pytest.approx(0.004)

    def test_different_image_sizes(self):
        cam_small = CameraModel(CameraConfig(fov_deg=90.0, image_size=100))
        cam_large = CameraModel(CameraConfig(fov_deg=90.0, image_size=1000))
        # Same footprint, different GSD.
        assert cam_small.ground_footprint_size(
            100.0
        ) == cam_large.ground_footprint_size(100.0)
        assert cam_small.ground_resolution(100.0) > cam_large.ground_resolution(100.0)

    # ---- Properties ----

    def test_image_size_property(self):
        cam = CameraModel(CameraConfig(image_size=256))
        assert cam.image_size == 256

    def test_config_property(self):
        cfg = CameraConfig(fov_deg=60.0)
        cam = CameraModel(cfg)
        assert cam.config is cfg

    def test_repr(self):
        cam = CameraModel()
        r = repr(cam)
        assert "CameraModel" in r
        assert "90.0" in r
        assert "500" in r
