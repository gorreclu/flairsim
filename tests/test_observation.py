"""Unit tests for flairsim.core.observation."""

import numpy as np
import pytest

from flairsim.core.observation import EpisodeResult, Observation
from flairsim.drone.drone import DroneState


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------


class TestEpisodeResult:
    """Tests for EpisodeResult."""

    def test_success(self):
        r = EpisodeResult(
            success=True,
            reason="Target found!",
            steps_taken=42,
            distance_travelled=1234.5,
        )
        assert r.success is True
        assert r.reason == "Target found!"
        assert r.steps_taken == 42
        assert r.distance_travelled == 1234.5

    def test_failure(self):
        r = EpisodeResult(
            success=False,
            reason="Step limit reached.",
            steps_taken=500,
            distance_travelled=5000.0,
        )
        assert r.success is False

    def test_frozen(self):
        r = EpisodeResult(True, "ok", 1, 1.0)
        with pytest.raises(AttributeError):
            r.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def _make_obs(
    step=0,
    done=False,
    result=None,
    bands=4,
    size=64,
    z=100.0,
):
    """Helper to create an Observation with a dummy image."""
    img = np.zeros((bands, size, size), dtype=np.uint8)
    state = DroneState(x=1000.0, y=6500.0, z=z)
    return Observation(
        image=img,
        drone_state=state,
        step=step,
        done=done,
        result=result,
        ground_footprint=z * 2,
        ground_resolution=z * 2 / size,
    )


class TestObservation:
    """Tests for Observation."""

    def test_basic_properties(self):
        obs = _make_obs(step=5, z=80)
        assert obs.step == 5
        assert obs.altitude == 80.0
        assert obs.position == (1000.0, 6500.0, 80.0)
        assert obs.ground_footprint == 160.0
        assert obs.done is False
        assert obs.result is None
        assert obs.success is False  # Not done.

    def test_done_with_success(self):
        result = EpisodeResult(True, "found", 10, 100.0)
        obs = _make_obs(done=True, result=result)
        assert obs.done is True
        assert obs.success is True

    def test_done_without_success(self):
        result = EpisodeResult(False, "timeout", 500, 5000.0)
        obs = _make_obs(done=True, result=result)
        assert obs.done is True
        assert obs.success is False

    def test_success_not_done(self):
        """success should be False when episode is not done."""
        obs = _make_obs(done=False)
        assert obs.success is False

    def test_metadata_default(self):
        obs = _make_obs()
        assert obs.metadata == {}

    # ---- image_rgb ----

    def test_image_rgb_4band(self):
        """4-band image (RGBI) -> extract first 3 bands as (H, W, 3)."""
        obs = _make_obs(bands=4, size=32)
        rgb = obs.image_rgb()
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_image_rgb_3band(self):
        """3-band image -> should work as (H, W, 3)."""
        obs = _make_obs(bands=3, size=32)
        rgb = obs.image_rgb()
        assert rgb.shape == (32, 32, 3)

    def test_image_rgb_1band(self):
        """1-band image -> replicate to greyscale (H, W, 3)."""
        obs = _make_obs(bands=1, size=16)
        rgb = obs.image_rgb()
        assert rgb.shape == (16, 16, 3)
        # All channels should be identical.
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])

    def test_image_rgb_2d(self):
        """2D image (H, W) -> replicate to (H, W, 3)."""
        img = np.zeros((16, 16), dtype=np.uint8)
        state = DroneState(x=0, y=0, z=100)
        obs = Observation(
            image=img,
            drone_state=state,
            step=0,
            ground_footprint=200,
            ground_resolution=0.4,
        )
        rgb = obs.image_rgb()
        assert rgb.shape == (16, 16, 3)

    def test_image_rgb_returns_copy(self):
        """image_rgb should return a new array, not a view."""
        obs = _make_obs(bands=4, size=8)
        rgb1 = obs.image_rgb()
        rgb1[:] = 255
        rgb2 = obs.image_rgb()
        assert rgb2.sum() == 0  # Original data unchanged.

    def test_image_rgb_content_correct(self):
        """Verify that RGB channels are correctly extracted from RGBI."""
        img = np.zeros((4, 4, 4), dtype=np.uint8)
        img[0, :, :] = 100  # R
        img[1, :, :] = 150  # G
        img[2, :, :] = 200  # B
        img[3, :, :] = 50  # NIR (should be dropped)
        state = DroneState(x=0, y=0, z=100)
        obs = Observation(
            image=img,
            drone_state=state,
            step=0,
            ground_footprint=200,
            ground_resolution=0.4,
        )
        rgb = obs.image_rgb()
        assert rgb[0, 0, 0] == 100  # R
        assert rgb[0, 0, 1] == 150  # G
        assert rgb[0, 0, 2] == 200  # B

    # ---- Repr ----

    def test_repr(self):
        obs = _make_obs(step=3)
        r = repr(obs)
        assert "Observation" in r
        assert "step=3" in r
