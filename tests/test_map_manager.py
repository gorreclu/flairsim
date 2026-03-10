"""
Integration tests for :class:`~flairsim.map.map_manager.MapManager`.

These tests create **synthetic GeoTIFF tiles** on disk using ``rasterio``
so that the full MapManager pipeline (discovery, geometry computation,
lazy/eager loading, region extraction, label queries, coordinate
transforms) can be exercised without requiring the multi-gigabyte
FLAIR-HUB dataset.

Tile layout
-----------
We create a 3×3 grid of 64×64 px tiles, each covering 12.8 m × 12.8 m
at 0.2 m/px resolution (matching the real FLAIR-HUB GSD).  The tiles
are placed in ROI sub-directory ``AB-S1-01`` with standard FLAIR-HUB
naming.  Each tile gets a unique per-pixel colour pattern so we can
verify that region extraction stitches tiles correctly.
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

from flairsim.map.map_manager import MapBounds, MapManager


# ---------------------------------------------------------------------------
# Fixtures: synthetic tile grid on disk
# ---------------------------------------------------------------------------

# Tile parameters (mimics FLAIR-HUB at smaller scale).
TILE_PX = 64  # Pixels per side.
TILE_GSD = 0.2  # Metres per pixel.
TILE_GROUND = TILE_PX * TILE_GSD  # 12.8 m per tile side.
N_BANDS = 4  # RGBI.
GRID_ROWS = 3  # 3 rows of tiles.
GRID_COLS = 3  # 3 columns of tiles.
DOMAIN = "D099-2099"
SENSOR_TYPE = "AERIAL_RGBI"
ROI = "AB-S1-01"

# Origin of the grid in Lambert-93 (arbitrary but realistic).
ORIGIN_X = 800_000.0  # Easting of the west edge.
ORIGIN_Y = 6_500_000.0 + GRID_ROWS * TILE_GROUND  # Northing of the north edge.


def _tile_bounds(row: int, col: int) -> Tuple[float, float, float, float]:
    """Return ``(x_min, y_min, x_max, y_max)`` for the given grid cell."""
    x_min = ORIGIN_X + col * TILE_GROUND
    x_max = x_min + TILE_GROUND
    # Rows increase southward → y decreases.
    y_max = ORIGIN_Y - row * TILE_GROUND
    y_min = y_max - TILE_GROUND
    return x_min, y_min, x_max, y_max


def _make_tile_data(row: int, col: int) -> np.ndarray:
    """Create a unique RGBI array for this tile position.

    - Band 0 (R): filled with ``row * 30``, clamped to [0, 255].
    - Band 1 (G): filled with ``col * 30``, clamped to [0, 255].
    - Band 2 (B): gradient across rows (top→bottom).
    - Band 3 (I): constant 128.
    """
    data = np.zeros((N_BANDS, TILE_PX, TILE_PX), dtype=np.uint8)
    data[0] = np.clip(row * 30, 0, 255)
    data[1] = np.clip(col * 30, 0, 255)
    data[2] = np.linspace(0, 255, TILE_PX, dtype=np.uint8)[:, None]
    data[3] = 128
    return data


def _write_tile(directory: Path, row: int, col: int) -> Path:
    """Write a single synthetic GeoTIFF tile and return its path."""
    filename = f"{DOMAIN}_{SENSOR_TYPE}_{ROI}_{row}-{col}.tif"
    roi_dir = directory / ROI
    roi_dir.mkdir(parents=True, exist_ok=True)
    filepath = roi_dir / filename

    x_min, y_min, x_max, y_max = _tile_bounds(row, col)
    transform = from_bounds(x_min, y_min, x_max, y_max, TILE_PX, TILE_PX)
    data = _make_tile_data(row, col)

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
    """Create a temporary directory with a 3×3 grid of synthetic tiles.

    This fixture is module-scoped so tiles are written once and shared
    across all tests in this file.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="flairsim_test_"))
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            _write_tile(tmpdir, row, col)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture()
def mm(tile_dir: Path) -> MapManager:
    """A MapManager loaded from the synthetic tile grid."""
    return MapManager(data_dir=tile_dir, roi=ROI, preload=True)


@pytest.fixture()
def mm_lazy(tile_dir: Path) -> MapManager:
    """A MapManager with lazy loading."""
    return MapManager(data_dir=tile_dir, roi=ROI, preload=False)


# ---------------------------------------------------------------------------
# Tests: MapBounds
# ---------------------------------------------------------------------------


class TestMapBounds:
    """Tests for the MapBounds helper dataclass."""

    def test_width_and_height(self):
        b = MapBounds(0.0, 0.0, 100.0, 50.0)
        assert b.width == 100.0
        assert b.height == 50.0

    def test_center(self):
        b = MapBounds(10.0, 20.0, 30.0, 40.0)
        assert b.center == (20.0, 30.0)

    def test_contains_inside(self):
        b = MapBounds(0.0, 0.0, 10.0, 10.0)
        assert b.contains(5.0, 5.0)

    def test_contains_boundary(self):
        b = MapBounds(0.0, 0.0, 10.0, 10.0)
        assert b.contains(0.0, 0.0)
        assert b.contains(10.0, 10.0)

    def test_contains_outside(self):
        b = MapBounds(0.0, 0.0, 10.0, 10.0)
        assert not b.contains(-1.0, 5.0)
        assert not b.contains(5.0, 11.0)

    def test_intersects_overlap(self):
        a = MapBounds(0.0, 0.0, 10.0, 10.0)
        b = MapBounds(5.0, 5.0, 15.0, 15.0)
        assert a.intersects(b)
        assert b.intersects(a)

    def test_intersects_no_overlap(self):
        a = MapBounds(0.0, 0.0, 10.0, 10.0)
        b = MapBounds(20.0, 20.0, 30.0, 30.0)
        assert not a.intersects(b)

    def test_intersects_touching(self):
        a = MapBounds(0.0, 0.0, 10.0, 10.0)
        b = MapBounds(10.0, 10.0, 20.0, 20.0)
        # Touching at a single corner point — should be True (non-strict).
        assert a.intersects(b)

    def test_frozen(self):
        b = MapBounds(0.0, 0.0, 10.0, 10.0)
        with pytest.raises(AttributeError):
            b.x_min = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: construction and discovery
# ---------------------------------------------------------------------------


class TestConstruction:
    """Verify MapManager construction from synthetic tiles."""

    def test_roi_name(self, mm: MapManager):
        assert mm.roi_name == ROI

    def test_grid_dimensions(self, mm: MapManager):
        assert mm.grid_rows == GRID_ROWS
        assert mm.grid_cols == GRID_COLS

    def test_tile_pixel_size(self, mm: MapManager):
        assert mm.tile_pixel_size == TILE_PX

    def test_tile_ground_size(self, mm: MapManager):
        assert mm.tile_ground_size == pytest.approx(TILE_GROUND, abs=0.01)

    def test_pixel_size_m(self, mm: MapManager):
        assert mm.pixel_size_m == pytest.approx(TILE_GSD, abs=0.001)

    def test_n_tiles(self, mm: MapManager):
        assert mm.n_tiles_total == GRID_ROWS * GRID_COLS
        assert mm.n_tiles_loaded == GRID_ROWS * GRID_COLS  # preloaded

    def test_bounds_extent(self, mm: MapManager):
        b = mm.bounds
        expected_w = GRID_COLS * TILE_GROUND
        expected_h = GRID_ROWS * TILE_GROUND
        assert b.width == pytest.approx(expected_w, abs=0.1)
        assert b.height == pytest.approx(expected_h, abs=0.1)

    def test_bounds_origin(self, mm: MapManager):
        b = mm.bounds
        assert b.x_min == pytest.approx(ORIGIN_X, abs=0.1)
        # y_min is south edge = ORIGIN_Y - GRID_ROWS * TILE_GROUND.
        assert b.y_min == pytest.approx(ORIGIN_Y - GRID_ROWS * TILE_GROUND, abs=0.1)

    def test_nonexistent_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            MapManager(data_dir=tmp_path / "nonexistent")

    def test_empty_directory_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No valid FLAIR-HUB tiles"):
            MapManager(data_dir=empty)

    def test_invalid_roi_raises(self, tile_dir: Path):
        with pytest.raises(ValueError, match="not found"):
            MapManager(data_dir=tile_dir, roi="ZZ-NONEXISTENT")

    def test_auto_select_roi(self, tile_dir: Path):
        # With roi=None, the largest ROI should be auto-selected.
        mm = MapManager(data_dir=tile_dir, roi=None, preload=False)
        assert mm.roi_name == ROI

    def test_repr(self, mm: MapManager):
        r = repr(mm)
        assert "MapManager" in r
        assert ROI in r


# ---------------------------------------------------------------------------
# Tests: lazy loading
# ---------------------------------------------------------------------------


class TestLazyLoading:
    """Verify that lazy loading defers tile reads."""

    def test_no_tiles_loaded_initially(self, mm_lazy: MapManager):
        # After construction with preload=False, only the tiles needed
        # for geometry computation should be loaded (corner tiles).
        # The geometry builder loads corner tiles, so loaded count is > 0
        # but < total.
        assert mm_lazy.n_tiles_loaded <= mm_lazy.n_tiles_total

    def test_region_triggers_loading(self, mm_lazy: MapManager):
        loaded_before = mm_lazy.n_tiles_loaded
        cx, cy = mm_lazy.bounds.center
        mm_lazy.get_region(cx, cy, half_extent=2.0)
        # At least one more tile should now be loaded.
        assert mm_lazy.n_tiles_loaded >= loaded_before


# ---------------------------------------------------------------------------
# Tests: coordinate transforms
# ---------------------------------------------------------------------------


class TestCoordinateTransforms:
    """Verify world ↔ pixel ↔ grid conversions."""

    def test_world_to_grid_origin(self, mm: MapManager):
        # A point just inside the NW corner tile (row=0, col=0).
        x = ORIGIN_X + 1.0
        y = ORIGIN_Y - 1.0
        row, col = mm.world_to_grid(x, y)
        assert row == 0
        assert col == 0

    def test_world_to_grid_other_tile(self, mm: MapManager):
        # Centre of tile (1, 2).
        x = ORIGIN_X + 2.5 * TILE_GROUND
        y = ORIGIN_Y - 1.5 * TILE_GROUND
        row, col = mm.world_to_grid(x, y)
        assert row == 1
        assert col == 2

    def test_world_to_pixel_round_trip(self, mm: MapManager):
        x_orig, y_orig = mm.bounds.center
        px, py = mm.world_to_pixel(x_orig, y_orig)
        x_back, y_back = mm.pixel_to_world(px, py)
        assert x_back == pytest.approx(x_orig, abs=0.001)
        assert y_back == pytest.approx(y_orig, abs=0.001)

    def test_pixel_origin_is_nw_corner(self, mm: MapManager):
        px, py = mm.world_to_pixel(mm.bounds.x_min, mm.bounds.y_max)
        assert px == pytest.approx(0.0, abs=0.01)
        assert py == pytest.approx(0.0, abs=0.01)

    def test_pixel_se_corner(self, mm: MapManager):
        px, py = mm.world_to_pixel(mm.bounds.x_max, mm.bounds.y_min)
        expected_total_px = GRID_COLS * TILE_PX
        expected_total_py = GRID_ROWS * TILE_PX
        assert px == pytest.approx(expected_total_px, abs=1.0)
        assert py == pytest.approx(expected_total_py, abs=1.0)


# ---------------------------------------------------------------------------
# Tests: region extraction
# ---------------------------------------------------------------------------


class TestGetRegion:
    """Test the get_region() method (image extraction)."""

    def test_single_tile_centre(self, mm: MapManager):
        """Extract a region fully inside the centre tile."""
        # Centre of tile (1, 1).
        cx = ORIGIN_X + 1.5 * TILE_GROUND
        cy = ORIGIN_Y - 1.5 * TILE_GROUND
        half = TILE_GROUND * 0.25  # Small window inside one tile.
        region = mm.get_region(cx, cy, half_extent=half)
        assert region.ndim == 3
        assert region.shape[0] == N_BANDS

    def test_region_with_output_size(self, mm: MapManager):
        """Extraction with resampling to a fixed output size."""
        cx, cy = mm.bounds.center
        region = mm.get_region(cx, cy, half_extent=10.0, output_size=64)
        assert region.shape == (N_BANDS, 64, 64)

    def test_region_spanning_multiple_tiles(self, mm: MapManager):
        """A large extraction that should span multiple tiles."""
        cx, cy = mm.bounds.center
        half = TILE_GROUND * 1.2  # Larger than one tile.
        region = mm.get_region(cx, cy, half_extent=half)
        assert region.ndim == 3
        assert region.shape[0] == N_BANDS
        # Should be wider than a single tile in pixels.
        assert region.shape[2] > TILE_PX

    def test_region_partially_outside_bounds(self, mm: MapManager):
        """Extraction near the map edge, partially out of bounds."""
        # Near the NW corner.
        x = mm.bounds.x_min + 1.0
        y = mm.bounds.y_max - 1.0
        region = mm.get_region(x, y, half_extent=5.0)
        assert region.ndim == 3
        # Out-of-bounds area should be zero-filled.
        # At least some pixels should be zero (the part outside the map).
        assert region.shape[0] == N_BANDS

    def test_region_completely_outside(self, mm: MapManager):
        """Extraction completely outside the map should return zeros."""
        x = mm.bounds.x_min - 1000.0
        y = mm.bounds.y_min - 1000.0
        region = mm.get_region(x, y, half_extent=5.0, output_size=32)
        assert region.shape == (N_BANDS, 32, 32)
        assert np.all(region == 0)

    def test_region_dtype_matches_tiles(self, mm: MapManager):
        """Output dtype should match the source tiles."""
        cx, cy = mm.bounds.center
        region = mm.get_region(cx, cy, half_extent=3.0)
        assert region.dtype == np.uint8

    def test_region_content_from_known_tile(self, mm: MapManager):
        """Verify pixel values match the synthetic pattern.

        Tile (1, 1) has R=30, G=30, I=128.
        """
        # Deep inside tile (1, 1).
        cx = ORIGIN_X + 1.5 * TILE_GROUND
        cy = ORIGIN_Y - 1.5 * TILE_GROUND
        # Very small region to stay within one tile.
        half = TILE_GSD * 2  # ~0.4 m → ~2 pixels radius.
        region = mm.get_region(cx, cy, half_extent=half)
        # Band 0 (R) should be ~30 (row=1 → 1*30=30).
        # Band 1 (G) should be ~30 (col=1 → 1*30=30).
        # Band 3 (I) should be 128.
        mid_h = region.shape[1] // 2
        mid_w = region.shape[2] // 2
        assert region[0, mid_h, mid_w] == 30  # R
        assert region[1, mid_h, mid_w] == 30  # G
        assert region[3, mid_h, mid_w] == 128  # I


# ---------------------------------------------------------------------------
# Tests: label queries
# ---------------------------------------------------------------------------


class TestGetLabelAt:
    """Test get_label_at() with single-band label-like tiles.

    Since our synthetic tiles have 4 bands, get_label_at reads band 0.
    For tile (r, c), band 0 is filled with ``r * 30``.
    """

    def test_label_at_tile_00(self, mm: MapManager):
        x = ORIGIN_X + 0.5 * TILE_GROUND
        y = ORIGIN_Y - 0.5 * TILE_GROUND
        label = mm.get_label_at(x, y)
        assert label == 0  # row=0 → 0*30=0

    def test_label_at_tile_10(self, mm: MapManager):
        x = ORIGIN_X + 0.5 * TILE_GROUND
        y = ORIGIN_Y - 1.5 * TILE_GROUND
        label = mm.get_label_at(x, y)
        assert label == 30  # row=1 → 1*30=30

    def test_label_at_tile_21(self, mm: MapManager):
        x = ORIGIN_X + 1.5 * TILE_GROUND
        y = ORIGIN_Y - 2.5 * TILE_GROUND
        label = mm.get_label_at(x, y)
        assert label == 60  # row=2 → 2*30=60

    def test_label_outside_returns_none(self, mm: MapManager):
        x = mm.bounds.x_min - 100.0
        y = mm.bounds.y_min - 100.0
        label = mm.get_label_at(x, y)
        assert label is None


# ---------------------------------------------------------------------------
# Tests: list_available_rois
# ---------------------------------------------------------------------------


class TestListRois:
    """Test introspection methods."""

    def test_list_includes_our_roi(self, mm: MapManager):
        rois = mm.list_available_rois()
        assert ROI in rois

    def test_list_returns_sorted(self, mm: MapManager):
        rois = mm.list_available_rois()
        assert rois == sorted(rois)


# ---------------------------------------------------------------------------
# Tests: sparse grid (missing tiles)
# ---------------------------------------------------------------------------


class TestSparseGrid:
    """Verify MapManager handles grids with missing tiles."""

    @pytest.fixture()
    def sparse_dir(self, tmp_path: Path) -> Path:
        """Create a grid where only 4 out of 9 tiles exist."""
        for row, col in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            _write_tile(tmp_path, row, col)
        return tmp_path

    def test_sparse_construction(self, sparse_dir: Path):
        mm = MapManager(data_dir=sparse_dir, roi=ROI, preload=True)
        assert mm.n_tiles_total == 4
        assert mm.grid_rows == 3  # 0..2
        assert mm.grid_cols == 3  # 0..2

    def test_sparse_region_missing_tile(self, sparse_dir: Path):
        """Region centred on a missing tile should return zeros there."""
        mm = MapManager(data_dir=sparse_dir, roi=ROI, preload=True)
        # Centre of tile (1, 1) — which doesn't exist.
        cx = ORIGIN_X + 1.5 * TILE_GROUND
        cy = ORIGIN_Y - 1.5 * TILE_GROUND
        half = TILE_GROUND * 0.4
        region = mm.get_region(cx, cy, half_extent=half, output_size=32)
        assert region.shape == (N_BANDS, 32, 32)
        # Since tile (1,1) is missing, centre should be zeros.
        assert np.all(region[:, 14:18, 14:18] == 0)

    def test_sparse_label_missing_tile(self, sparse_dir: Path):
        mm = MapManager(data_dir=sparse_dir, roi=ROI, preload=True)
        cx = ORIGIN_X + 1.5 * TILE_GROUND
        cy = ORIGIN_Y - 1.5 * TILE_GROUND
        assert mm.get_label_at(cx, cy) is None
