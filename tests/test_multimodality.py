"""
Tests for multi-modality support (Feature 1).

Covers:
- Modality discovery (``discover_modalities``, ``pick_primary_modality``,
  ``is_single_modality_dir``) in :mod:`flairsim.map.modality`.
- Multi-modality simulator construction and observation building.
- Backward compatibility when ``--data-dir`` points at a single-modality dir.
- Server-side PNG encoding / normalisation for non-uint8 data.
- ``ViewerObservation`` factory methods with per-modality images.
- ``_bands_to_rgb`` helper.
- ``FlairSimulator._detect_modality_name`` static method.
"""

from __future__ import annotations

import base64
import io
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from flairsim.core.observation import Observation
from flairsim.core.simulator import FlairSimulator, SimulatorConfig
from flairsim.drone.camera import CameraConfig
from flairsim.drone.drone import DroneConfig, DroneState
from flairsim.map.modality import (
    Modality,
    _extract_domain_prefix,
    discover_modalities,
    infer_domain_from_dir,
    is_single_modality_dir,
    pick_primary_modality,
)
from flairsim.map.tile_loader import normalize_to_uint8
from flairsim.server.app import _encode_image_png
from flairsim.viewer.remote import ViewerObservation, _bands_to_rgb


# ---------------------------------------------------------------------------
# Constants (reused across tests)
# ---------------------------------------------------------------------------

TILE_PX = 64
TILE_GSD = 0.2
TILE_GROUND = TILE_PX * TILE_GSD  # 12.8 m
GRID_ROWS = 3
GRID_COLS = 3
ROI = "AB-S1-01"
DOMAIN = "D099-2099"

ORIGIN_X = 800_000.0
ORIGIN_Y = 6_500_000.0 + GRID_ROWS * TILE_GROUND


# ---------------------------------------------------------------------------
# Helpers for creating synthetic tile directories
# ---------------------------------------------------------------------------


def _tile_bounds(row: int, col: int) -> Tuple[float, float, float, float]:
    """Compute geo-bounds for a tile at (row, col)."""
    x_min = ORIGIN_X + col * TILE_GROUND
    x_max = x_min + TILE_GROUND
    y_max = ORIGIN_Y - row * TILE_GROUND
    y_min = y_max - TILE_GROUND
    return x_min, y_min, x_max, y_max


def _write_tile(
    directory: Path,
    row: int,
    col: int,
    n_bands: int = 4,
    dtype: str = "uint8",
    sensor_type: str = "AERIAL_RGBI",
) -> Path:
    """Write a synthetic GeoTIFF tile."""
    filename = f"{DOMAIN}_{sensor_type}_{ROI}_{row}-{col}.tif"
    roi_dir = directory / ROI
    roi_dir.mkdir(parents=True, exist_ok=True)
    filepath = roi_dir / filename

    x_min, y_min, x_max, y_max = _tile_bounds(row, col)
    transform = from_bounds(x_min, y_min, x_max, y_max, TILE_PX, TILE_PX)

    if dtype == "uint8":
        data = np.full((n_bands, TILE_PX, TILE_PX), row * 30, dtype=np.uint8)
        data[0] = col * 30
    elif dtype == "uint16":
        data = np.full((n_bands, TILE_PX, TILE_PX), row * 1000, dtype=np.uint16)
        data[0] = col * 1000
    elif dtype == "float32":
        data = np.full((n_bands, TILE_PX, TILE_PX), row * 10.0, dtype=np.float32)
        data[0] = col * 10.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=TILE_PX,
        width=TILE_PX,
        count=n_bands,
        dtype=dtype,
        crs="EPSG:2154",
        transform=transform,
    ) as dst:
        dst.write(data)
    return filepath


def _make_single_modality_dir(
    tmpdir: Path,
    sensor_type: str = "AERIAL_RGBI",
    n_bands: int = 4,
    dtype: str = "uint8",
) -> Path:
    """Create a single-modality tile directory (old-style layout)."""
    tile_dir = tmpdir / f"{DOMAIN}_{sensor_type}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            _write_tile(
                tile_dir,
                row,
                col,
                n_bands=n_bands,
                dtype=dtype,
                sensor_type=sensor_type,
            )
    return tile_dir


def _make_multi_modality_dir(
    tmpdir: Path,
    modalities: Dict[str, Tuple[int, str]] | None = None,
) -> Path:
    """Create a parent directory with multiple modality sub-dirs.

    Parameters
    ----------
    tmpdir : Path
        Root temporary directory.
    modalities : dict or None
        Mapping from sensor_type to (n_bands, dtype).
        Defaults to AERIAL_RGBI (4, uint8) + DEM_ELEV (2, float32).

    Returns
    -------
    Path
        The parent directory containing modality sub-dirs.
    """
    if modalities is None:
        modalities = {
            "AERIAL_RGBI": (4, "uint8"),
            "DEM_ELEV": (2, "float32"),
        }

    parent = tmpdir / "multi"
    parent.mkdir(parents=True, exist_ok=True)

    for sensor_type, (n_bands, dtype) in modalities.items():
        mod_dir = parent / f"{DOMAIN}_{sensor_type}"
        mod_dir.mkdir(parents=True, exist_ok=True)
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                _write_tile(
                    mod_dir,
                    row,
                    col,
                    n_bands=n_bands,
                    dtype=dtype,
                    sensor_type=sensor_type,
                )

    return parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_tile_dir() -> Path:
    """A single-modality tile directory (AERIAL_RGBI, uint8)."""
    tmpdir = Path(tempfile.mkdtemp(prefix="flairsim_multi_single_"))
    tile_dir = _make_single_modality_dir(tmpdir)
    yield tile_dir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def multi_tile_dir() -> Path:
    """A parent directory with 2 modalities: AERIAL_RGBI + DEM_ELEV."""
    tmpdir = Path(tempfile.mkdtemp(prefix="flairsim_multi_parent_"))
    parent = _make_multi_modality_dir(tmpdir)
    yield parent
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def three_modality_dir() -> Path:
    """Parent directory with 3 modalities: AERIAL_RGBI + DEM_ELEV + SPOT_RGBI."""
    tmpdir = Path(tempfile.mkdtemp(prefix="flairsim_multi_three_"))
    parent = _make_multi_modality_dir(
        tmpdir,
        modalities={
            "AERIAL_RGBI": (4, "uint8"),
            "DEM_ELEV": (2, "float32"),
            "SPOT_RGBI": (4, "uint16"),
        },
    )
    yield parent
    shutil.rmtree(tmpdir, ignore_errors=True)


def _make_sim(
    data_dir: Path,
    max_steps: int = 50,
    image_size: int = 32,
) -> FlairSimulator:
    """Create a FlairSimulator with small images for fast tests."""
    config = SimulatorConfig(
        drone_config=DroneConfig(z_min=5.0, z_max=200.0, default_altitude=20.0),
        camera_config=CameraConfig(fov_deg=90.0, image_size=image_size),
        max_steps=max_steps,
        roi=ROI,
        preload_tiles=True,
    )
    return FlairSimulator(data_dir=data_dir, config=config)


# =========================================================================
# Tests: discover_modalities()
# =========================================================================


class TestDiscoverModalities:
    """Test ``discover_modalities`` from modality.py."""

    def test_discovers_two_modalities(self, multi_tile_dir: Path):
        result = discover_modalities(multi_tile_dir)
        assert len(result) == 2
        assert Modality.AERIAL_RGBI in result
        assert Modality.DEM_ELEV in result

    def test_discovers_three_modalities(self, three_modality_dir: Path):
        result = discover_modalities(three_modality_dir)
        assert len(result) == 3
        assert Modality.AERIAL_RGBI in result
        assert Modality.DEM_ELEV in result
        assert Modality.SPOT_RGBI in result

    def test_paths_are_correct(self, multi_tile_dir: Path):
        result = discover_modalities(multi_tile_dir)
        for mod, path in result.items():
            assert path.is_dir()
            assert mod.value.dir_suffix in path.name

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        empty = tmp_path / "empty_parent"
        empty.mkdir()
        result = discover_modalities(empty)
        assert result == {}

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path):
        result = discover_modalities(tmp_path / "does_not_exist")
        assert result == {}

    def test_single_modality_dir_returns_empty(self, single_tile_dir: Path):
        """A single-modality dir has no modality sub-dirs to discover."""
        result = discover_modalities(single_tile_dir)
        # The single_tile_dir itself is named D099-2099_AERIAL_RGBI and
        # contains ROI sub-dirs, not modality sub-dirs.  The only sub-dir
        # is "AB-S1-01" which doesn't match any modality suffix.
        assert len(result) == 0

    def test_no_false_partial_match(self, tmp_path: Path):
        """Dir named 'foo_RGBI' should not match 'AERIAL_RGBI'."""
        bad_dir = tmp_path / "foo_RGBI"
        bad_dir.mkdir()
        result = discover_modalities(tmp_path)
        # AERIAL_RGBI suffix is "AERIAL_RGBI", not "RGBI".
        assert Modality.AERIAL_RGBI not in result


# =========================================================================
# Tests: domain-aware discovery (flat FLAIR-HUB layout)
# =========================================================================


class TestDomainAwareDiscovery:
    """Test domain-aware ``discover_modalities`` and helpers."""

    @pytest.fixture()
    def flat_layout(self, tmp_path: Path) -> Path:
        """Create a flat FLAIR-HUB layout with two domains."""
        root = tmp_path / "FLAIR-HUB"
        root.mkdir()
        # Domain D006-2020
        for suffix in ("AERIAL_RGBI", "DEM_ELEV", "SPOT_RGBI"):
            d = root / f"D006-2020_{suffix}"
            d.mkdir()
            roi = d / "UU-S2-1"
            roi.mkdir()
            # Write a dummy file so it looks like data.
            (roi / "dummy.tif").touch()
        # Domain D012-2019
        for suffix in ("AERIAL_RGBI", "DEM_ELEV"):
            d = root / f"D012-2019_{suffix}"
            d.mkdir()
            roi = d / "FF-S1-14"
            roi.mkdir()
            (roi / "dummy.tif").touch()
        # A non-matching directory
        (root / "GLOBAL_ALL_MTD").mkdir()
        return root

    def test_discover_with_domain_filter(self, flat_layout: Path):
        result = discover_modalities(flat_layout, domain="D006-2020")
        assert len(result) == 3
        assert Modality.AERIAL_RGBI in result
        assert Modality.DEM_ELEV in result
        assert Modality.SPOT_RGBI in result

    def test_discover_other_domain(self, flat_layout: Path):
        result = discover_modalities(flat_layout, domain="D012-2019")
        assert len(result) == 2
        assert Modality.AERIAL_RGBI in result
        assert Modality.DEM_ELEV in result

    def test_discover_without_domain_finds_all(self, flat_layout: Path):
        """Without a domain filter, both domains' dirs match -- last wins."""
        result = discover_modalities(flat_layout)
        # Both D006 and D012 have AERIAL_RGBI; the alphabetically last
        # (D012-2019_AERIAL_RGBI) will overwrite D006's entry.
        assert Modality.AERIAL_RGBI in result
        # But SPOT_RGBI only exists for D006.
        assert Modality.SPOT_RGBI in result

    def test_discover_nonexistent_domain_empty(self, flat_layout: Path):
        result = discover_modalities(flat_layout, domain="D999-9999")
        assert result == {}

    def test_extract_domain_prefix_normal(self):
        assert _extract_domain_prefix("D006-2020_AERIAL_RGBI") == "D006-2020"

    def test_extract_domain_prefix_historical(self):
        assert _extract_domain_prefix("D006-195X_AERIAL-RLT_PAN") == "D006-195X"

    def test_extract_domain_prefix_no_underscore(self):
        assert _extract_domain_prefix("GLOBAL_ALL_MTD") is None  # "GLOBAL" has no dash

    def test_extract_domain_prefix_no_dash(self):
        assert _extract_domain_prefix("foobar") is None

    def test_infer_domain_from_dir_path(self):
        assert (
            infer_domain_from_dir("/data/FLAIR-HUB/D006-2020_AERIAL_RGBI")
            == "D006-2020"
        )

    def test_infer_domain_from_dir_none_for_generic(self):
        assert infer_domain_from_dir("/data/some_random_dir") is None


# =========================================================================
# Tests: pick_primary_modality()
# =========================================================================


class TestPickPrimaryModality:
    """Test ``pick_primary_modality`` preference logic."""

    def test_prefers_aerial_rgbi(self):
        mods = {
            Modality.DEM_ELEV: Path("/fake/dem"),
            Modality.AERIAL_RGBI: Path("/fake/rgbi"),
            Modality.SPOT_RGBI: Path("/fake/spot"),
        }
        assert pick_primary_modality(mods) == Modality.AERIAL_RGBI

    def test_falls_back_to_spot_rgbi(self):
        mods = {
            Modality.DEM_ELEV: Path("/fake/dem"),
            Modality.SPOT_RGBI: Path("/fake/spot"),
        }
        assert pick_primary_modality(mods) == Modality.SPOT_RGBI

    def test_falls_back_to_first_if_no_preference(self):
        mods = {
            Modality.SENTINEL1_ASC_TS: Path("/fake/s1asc"),
            Modality.SENTINEL1_DESC_TS: Path("/fake/s1desc"),
        }
        result = pick_primary_modality(mods)
        assert result in mods

    def test_single_modality(self):
        mods = {Modality.DEM_ELEV: Path("/fake/dem")}
        assert pick_primary_modality(mods) == Modality.DEM_ELEV

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No modalities"):
            pick_primary_modality({})


# =========================================================================
# Tests: is_single_modality_dir()
# =========================================================================


class TestIsSingleModalityDir:
    """Test ``is_single_modality_dir`` detection."""

    def test_single_modality_dir_is_true(self, single_tile_dir: Path):
        assert is_single_modality_dir(single_tile_dir) is True

    def test_multi_modality_parent_is_false(self, multi_tile_dir: Path):
        # The parent dir contains sub-dirs like D099-2099_AERIAL_RGBI
        # which in turn contain ROI sub-dirs with .tif files.
        # The parent itself does NOT directly contain .tif files.
        # However, is_single_modality_dir checks subdirs for .tif too,
        # so let's verify the actual behaviour.
        #
        # Each sub-dir of multi_tile_dir is D099-2099_AERIAL_RGBI (etc.)
        # and those contain ROI dirs with .tif.  So the sub-dirs DO have
        # sub-sub-dirs with .tif.  But is_single_modality_dir only checks
        # immediate children (one level deep), not grandchildren.
        #
        # multi_tile_dir/D099-2099_AERIAL_RGBI/AB-S1-01/*.tif
        # is_single_modality_dir checks multi_tile_dir/*.tif (no),
        # then multi_tile_dir/D099-2099_AERIAL_RGBI/*.tif (no, .tif is deeper).
        # So it should return False.
        assert is_single_modality_dir(multi_tile_dir) is False

    def test_empty_dir_is_false(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert is_single_modality_dir(empty) is False

    def test_nonexistent_is_false(self, tmp_path: Path):
        assert is_single_modality_dir(tmp_path / "nope") is False

    def test_dir_with_flat_tifs(self, tmp_path: Path):
        """Directory with .tif files directly (no ROI sub-dir)."""
        d = tmp_path / "flat_tifs"
        d.mkdir()
        # Write a minimal tif.
        transform = from_bounds(0.0, 0.0, 12.8, 12.8, 64, 64)
        fp = d / f"{DOMAIN}_AERIAL_RGBI_{ROI}_0-0.tif"
        with rasterio.open(
            fp,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=4,
            dtype="uint8",
            crs="EPSG:2154",
            transform=transform,
        ) as dst:
            dst.write(np.zeros((4, 64, 64), dtype=np.uint8))
        assert is_single_modality_dir(d) is True


# =========================================================================
# Tests: _detect_modality_name (static method)
# =========================================================================


class TestDetectModalityName:
    """Test ``FlairSimulator._detect_modality_name``."""

    def test_aerial_rgbi(self):
        p = Path("/some/path/D004-2021_AERIAL_RGBI")
        assert FlairSimulator._detect_modality_name(p) == "AERIAL_RGBI"

    def test_dem_elev(self):
        p = Path("/some/path/D004-2021_DEM_ELEV")
        assert FlairSimulator._detect_modality_name(p) == "DEM_ELEV"

    def test_spot_rgbi(self):
        p = Path("/some/path/D012-2020_SPOT_RGBI")
        assert FlairSimulator._detect_modality_name(p) == "SPOT_RGBI"

    def test_aerial_rlt_pan(self):
        p = Path("/some/path/D032-1950_AERIAL-RLT_PAN")
        assert FlairSimulator._detect_modality_name(p) == "AERIAL_RLT_PAN"

    def test_unknown_returns_none(self):
        p = Path("/some/path/D004-2021_MYSTERY_DATA")
        assert FlairSimulator._detect_modality_name(p) is None

    def test_bare_suffix(self):
        """A directory named exactly like the suffix (no domain prefix)."""
        p = Path("/some/path/AERIAL_RGBI")
        assert FlairSimulator._detect_modality_name(p) == "AERIAL_RGBI"

    def test_no_partial_match(self):
        """'foo_RGBI' should NOT match AERIAL_RGBI."""
        p = Path("/some/path/foo_RGBI")
        assert FlairSimulator._detect_modality_name(p) is None


# =========================================================================
# Tests: Simulator in single-modality mode (backward compat)
# =========================================================================


class TestSimulatorSingleModality:
    """Verify backward compatibility when data_dir is a single-modality dir."""

    def test_creates_successfully(self, single_tile_dir: Path):
        sim = _make_sim(single_tile_dir)
        assert sim.map_manager is not None
        sim.close()

    def test_has_one_map_manager(self, single_tile_dir: Path):
        sim = _make_sim(single_tile_dir)
        assert len(sim.map_managers) == 1
        sim.close()

    def test_primary_modality_detected(self, single_tile_dir: Path):
        sim = _make_sim(single_tile_dir)
        assert sim.primary_modality == "AERIAL_RGBI"
        sim.close()

    def test_observation_has_images(self, single_tile_dir: Path):
        sim = _make_sim(single_tile_dir)
        obs = sim.reset()
        # In single-modality mode, images is still populated for consistency.
        assert len(obs.images) == 1
        assert "AERIAL_RGBI" in obs.images
        sim.close()

    def test_image_matches_images_primary(self, single_tile_dir: Path):
        sim = _make_sim(single_tile_dir)
        obs = sim.reset()
        # The main image should be the same as the primary modality image.
        primary_img = obs.images[sim.primary_modality]
        np.testing.assert_array_equal(obs.image, primary_img)
        sim.close()

    def test_metadata_has_modality_info(self, single_tile_dir: Path):
        sim = _make_sim(single_tile_dir)
        obs = sim.reset()
        assert obs.metadata["primary_modality"] == "AERIAL_RGBI"
        assert "AERIAL_RGBI" in obs.metadata["modalities"]
        sim.close()


# =========================================================================
# Tests: Simulator in multi-modality mode
# =========================================================================


class TestSimulatorMultiModality:
    """Verify multi-modality simulator construction and observation."""

    def test_creates_successfully(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        assert sim.map_manager is not None
        sim.close()

    def test_multiple_map_managers(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        assert len(sim.map_managers) == 2
        assert "AERIAL_RGBI" in sim.map_managers
        assert "DEM_ELEV" in sim.map_managers
        sim.close()

    def test_primary_is_aerial_rgbi(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        assert sim.primary_modality == "AERIAL_RGBI"
        assert sim.map_manager is sim.map_managers["AERIAL_RGBI"]
        sim.close()

    def test_observation_images_populated(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        obs = sim.reset()
        assert len(obs.images) == 2
        assert "AERIAL_RGBI" in obs.images
        assert "DEM_ELEV" in obs.images
        sim.close()

    def test_each_modality_image_has_correct_bands(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        obs = sim.reset()
        # AERIAL_RGBI: 4 bands
        assert obs.images["AERIAL_RGBI"].shape[0] == 4
        # DEM_ELEV: 2 bands
        assert obs.images["DEM_ELEV"].shape[0] == 2
        sim.close()

    def test_image_spatial_dims_match_camera(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir, image_size=32)
        obs = sim.reset()
        for mod_name, img in obs.images.items():
            assert img.shape[1] == 32, f"{mod_name} height mismatch"
            assert img.shape[2] == 32, f"{mod_name} width mismatch"
        sim.close()

    def test_primary_image_equals_main_image(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        obs = sim.reset()
        np.testing.assert_array_equal(obs.image, obs.images["AERIAL_RGBI"])
        sim.close()

    def test_metadata_lists_all_modalities(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        obs = sim.reset()
        assert set(obs.metadata["modalities"]) == {"AERIAL_RGBI", "DEM_ELEV"}
        sim.close()

    def test_step_updates_all_modality_images(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        sim.reset()
        from flairsim.core.action import Action

        obs = sim.step(Action.move(dx=5.0, dy=5.0))
        # After step, images should still be populated.
        assert len(obs.images) == 2
        for mod_name, img in obs.images.items():
            assert img.ndim == 3
        sim.close()

    def test_three_modalities(self, three_modality_dir: Path):
        sim = _make_sim(three_modality_dir)
        assert len(sim.map_managers) == 3
        obs = sim.reset()
        assert len(obs.images) == 3
        assert "SPOT_RGBI" in obs.images
        # SPOT is uint16 — verify the raw dtype is preserved in the array.
        assert obs.images["SPOT_RGBI"].dtype == np.uint16
        sim.close()

    def test_no_modalities_raises(self, tmp_path: Path):
        """A parent directory with no matching modality sub-dirs should fail."""
        empty = tmp_path / "empty_parent"
        empty.mkdir()
        with pytest.raises(ValueError, match="No FLAIR-HUB modalities"):
            _make_sim(empty)

    def test_repr_shows_modalities(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        r = repr(sim)
        assert "modalities" in r
        sim.close()


# =========================================================================
# Tests: normalize_to_uint8 (tile_loader.py)
# =========================================================================


class TestNormalizeToUint8:
    """Test the ``normalize_to_uint8`` function."""

    def test_uint8_passthrough(self):
        data = np.array([0, 128, 255], dtype=np.uint8)
        result = normalize_to_uint8(data)
        np.testing.assert_array_equal(result, data)
        assert result.dtype == np.uint8

    def test_uint16_stretched(self):
        data = np.linspace(0, 10000, 100, dtype=np.uint16)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255
        # Should use most of the range.
        assert result.max() > 200

    def test_float32_normalized(self):
        data = np.linspace(-50.0, 500.0, 256, dtype=np.float32)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        assert result.max() <= 255

    def test_constant_image(self):
        data = np.full((3, 10, 10), 42.0, dtype=np.float32)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        # Constant -> mid-grey (128).
        assert np.all(result == 128)

    def test_empty_array(self):
        data = np.array([], dtype=np.float32)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        assert result.size == 0

    def test_3d_bands_hw(self):
        data = (
            np.random.default_rng(42).integers(0, 5000, (4, 64, 64)).astype(np.uint16)
        )
        result = normalize_to_uint8(data)
        assert result.shape == (4, 64, 64)
        assert result.dtype == np.uint8


# =========================================================================
# Tests: _encode_image_png (server/app.py)
# =========================================================================


class TestEncodeImagePng:
    """Test the server's ``_encode_image_png`` helper."""

    def test_uint8_rgb(self):
        img = np.random.default_rng(42).integers(0, 256, (4, 32, 32), dtype=np.uint8)
        b64 = _encode_image_png(img)
        assert isinstance(b64, str)
        # Should be valid base64.
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_uint16_encoded_without_error(self):
        img = np.random.default_rng(42).integers(0, 10000, (4, 32, 32), dtype=np.uint16)
        b64 = _encode_image_png(img)
        decoded = base64.b64decode(b64)
        # Verify it's a valid PNG.
        from PIL import Image

        pil_img = Image.open(io.BytesIO(decoded))
        assert pil_img.size == (32, 32)

    def test_float32_encoded_without_error(self):
        img = np.random.default_rng(42).random((2, 32, 32), dtype=np.float32) * 500
        b64 = _encode_image_png(img)
        decoded = base64.b64decode(b64)
        from PIL import Image

        pil_img = Image.open(io.BytesIO(decoded))
        assert pil_img.size == (32, 32)

    def test_single_band_grayscale(self):
        img = np.full((1, 32, 32), 128, dtype=np.uint8)
        b64 = _encode_image_png(img)
        decoded = base64.b64decode(b64)
        from PIL import Image

        pil_img = Image.open(io.BytesIO(decoded)).convert("RGB")
        arr = np.asarray(pil_img)
        # Should be uniform grey.
        assert arr.shape == (32, 32, 3)
        assert np.all(arr[:, :, 0] == 128)


# =========================================================================
# Tests: _bands_to_rgb helper (viewer/remote.py)
# =========================================================================


class TestBandsToRgb:
    """Test the ``_bands_to_rgb`` helper."""

    def test_4_band_uint8(self):
        img = np.random.default_rng(42).integers(0, 256, (4, 16, 16), dtype=np.uint8)
        result = _bands_to_rgb(img, normalize_to_uint8)
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.uint8

    def test_1_band_uint8(self):
        img = np.full((1, 16, 16), 100, dtype=np.uint8)
        result = _bands_to_rgb(img, normalize_to_uint8)
        assert result.shape == (16, 16, 3)
        # Greyscale: all channels equal.
        assert np.all(result[:, :, 0] == 100)
        assert np.all(result[:, :, 1] == 100)
        assert np.all(result[:, :, 2] == 100)

    def test_float32_normalised(self):
        img = np.random.default_rng(42).random((3, 16, 16), dtype=np.float32) * 100
        result = _bands_to_rgb(img, normalize_to_uint8)
        assert result.dtype == np.uint8
        assert result.shape == (16, 16, 3)

    def test_uint16_normalised(self):
        img = np.random.default_rng(42).integers(0, 5000, (4, 16, 16), dtype=np.uint16)
        result = _bands_to_rgb(img, normalize_to_uint8)
        assert result.dtype == np.uint8
        assert result.shape == (16, 16, 3)


# =========================================================================
# Tests: ViewerObservation.from_observation()
# =========================================================================


class TestViewerObservationFromObs:
    """Test ``ViewerObservation.from_observation`` with multi-modality."""

    def _make_observation(
        self,
        modality_images: Dict[str, np.ndarray] | None = None,
    ) -> Observation:
        """Build a minimal Observation for testing."""
        image = np.random.default_rng(42).integers(0, 256, (4, 32, 32), dtype=np.uint8)
        ds = DroneState(
            x=800_010.0,
            y=6_500_010.0,
            z=20.0,
            heading=0.0,
            step_count=0,
            total_distance=0.0,
        )
        return Observation(
            image=image,
            drone_state=ds,
            step=0,
            done=False,
            ground_footprint=40.0,
            ground_resolution=1.25,
            images=modality_images or {},
        )

    def test_empty_images_dict(self):
        obs = self._make_observation()
        vobs = ViewerObservation.from_observation(obs)
        assert vobs.images_rgb == {}

    def test_single_modality_images(self):
        img = np.random.default_rng(42).integers(0, 256, (4, 32, 32), dtype=np.uint8)
        obs = self._make_observation(modality_images={"AERIAL_RGBI": img})
        vobs = ViewerObservation.from_observation(obs)
        assert "AERIAL_RGBI" in vobs.images_rgb
        assert vobs.images_rgb["AERIAL_RGBI"].shape == (32, 32, 3)
        assert vobs.images_rgb["AERIAL_RGBI"].dtype == np.uint8

    def test_multi_modality_images(self):
        rgbi = np.random.default_rng(42).integers(0, 256, (4, 32, 32), dtype=np.uint8)
        dem = np.random.default_rng(43).random((2, 32, 32), dtype=np.float32) * 100
        obs = self._make_observation(
            modality_images={"AERIAL_RGBI": rgbi, "DEM_ELEV": dem}
        )
        vobs = ViewerObservation.from_observation(obs)
        assert len(vobs.images_rgb) == 2
        assert vobs.images_rgb["AERIAL_RGBI"].shape == (32, 32, 3)
        assert vobs.images_rgb["DEM_ELEV"].shape == (32, 32, 3)
        # DEM was float32, should be normalised to uint8.
        assert vobs.images_rgb["DEM_ELEV"].dtype == np.uint8


# =========================================================================
# Tests: ViewerObservation.from_server_response()
# =========================================================================


class TestViewerObservationFromServerResponse:
    """Test ``ViewerObservation.from_server_response`` with per-modality images."""

    @staticmethod
    def _make_b64_png(height: int = 32, width: int = 32) -> str:
        """Create a base64-encoded PNG for testing."""
        from PIL import Image as PILImage

        arr = np.random.default_rng(42).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        )
        pil = PILImage.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def test_no_images_field(self):
        b64 = self._make_b64_png()
        data = {
            "step": 0,
            "done": False,
            "image_base64": b64,
            "drone_state": {"x": 1.0, "y": 2.0, "z": 3.0, "total_distance": 0.0},
            "ground_footprint": 40.0,
            "ground_resolution": 1.25,
        }
        vobs = ViewerObservation.from_server_response(data)
        assert vobs.images_rgb == {}

    def test_with_modality_images(self):
        b64_main = self._make_b64_png()
        b64_rgbi = self._make_b64_png()
        b64_dem = self._make_b64_png()
        data = {
            "step": 5,
            "done": False,
            "image_base64": b64_main,
            "drone_state": {"x": 1.0, "y": 2.0, "z": 3.0, "total_distance": 10.0},
            "ground_footprint": 40.0,
            "ground_resolution": 1.25,
            "images": {
                "AERIAL_RGBI": b64_rgbi,
                "DEM_ELEV": b64_dem,
            },
        }
        vobs = ViewerObservation.from_server_response(data)
        assert len(vobs.images_rgb) == 2
        assert vobs.images_rgb["AERIAL_RGBI"].shape == (32, 32, 3)
        assert vobs.images_rgb["DEM_ELEV"].shape == (32, 32, 3)

    def test_result_field(self):
        b64 = self._make_b64_png()
        data = {
            "step": 10,
            "done": True,
            "image_base64": b64,
            "drone_state": {"x": 1.0, "y": 2.0, "z": 3.0, "total_distance": 50.0},
            "ground_footprint": 40.0,
            "ground_resolution": 1.25,
            "result": {"success": True, "reason": "Target found"},
            "images": {},
        }
        vobs = ViewerObservation.from_server_response(data)
        assert vobs.done is True
        assert vobs.result is not None
        assert vobs.result.success is True


# =========================================================================
# Tests: End-to-end multi-modality with simulator + viewer observation
# =========================================================================


class TestEndToEndMultiModality:
    """Integration: simulator produces multi-modality obs, viewer consumes it."""

    def test_sim_to_viewer_observation(self, multi_tile_dir: Path):
        sim = _make_sim(multi_tile_dir)
        obs = sim.reset()
        vobs = ViewerObservation.from_observation(obs)
        # Both modalities should be present.
        assert "AERIAL_RGBI" in vobs.images_rgb
        assert "DEM_ELEV" in vobs.images_rgb
        # All should be (H, W, 3) uint8.
        for mod_name, img in vobs.images_rgb.items():
            assert img.ndim == 3
            assert img.shape[2] == 3
            assert img.dtype == np.uint8
        sim.close()

    def test_sim_to_server_encode(self, multi_tile_dir: Path):
        """Simulate what the server does: encode all modality images."""
        sim = _make_sim(multi_tile_dir)
        obs = sim.reset()
        # Encode each modality image.
        for mod_name, mod_image in obs.images.items():
            b64 = _encode_image_png(mod_image)
            decoded = base64.b64decode(b64)
            assert len(decoded) > 0
        sim.close()

    def test_three_modalities_end_to_end(self, three_modality_dir: Path):
        sim = _make_sim(three_modality_dir)
        obs = sim.reset()
        vobs = ViewerObservation.from_observation(obs)
        assert len(vobs.images_rgb) == 3
        # SPOT is uint16 in the raw obs, but should be uint8 in viewer.
        assert vobs.images_rgb["SPOT_RGBI"].dtype == np.uint8
        sim.close()
