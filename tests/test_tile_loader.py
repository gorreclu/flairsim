"""Unit tests for flairsim.map.tile_loader."""

from pathlib import Path

import numpy as np
import pytest

from flairsim.map.tile_loader import (
    normalize_to_uint8,
    parse_roi_from_path,
    parse_tile_coords,
)


# ---------------------------------------------------------------------------
# parse_tile_coords
# ---------------------------------------------------------------------------


class TestParseTileCoords:
    """Tests for filename coordinate parsing."""

    def test_standard_filename(self):
        p = Path("D004-2021_AERIAL_RGBI_AA-S1-32_3-7.tif")
        assert parse_tile_coords(p) == (3, 7)

    def test_large_indices(self):
        p = Path("D005-2018_AERIAL_RGBI_BB-S2-10_125-42.tif")
        assert parse_tile_coords(p) == (125, 42)

    def test_zero_indices(self):
        p = Path("D004-2021_AERIAL_RGBI_AA-S1-32_0-0.tif")
        assert parse_tile_coords(p) == (0, 0)

    def test_with_parent_directory(self):
        p = Path("/data/FLAIR/AA-S1-32/D004-2021_AERIAL_RGBI_AA-S1-32_5-12.tif")
        assert parse_tile_coords(p) == (5, 12)

    def test_non_flair_filename(self):
        p = Path("random_image.tif")
        assert parse_tile_coords(p) is None

    def test_partial_match(self):
        p = Path("file_with_3-but_no_tif.txt")
        # stem doesn't end with _ROW-COL
        assert parse_tile_coords(Path("something.tif")) is None

    def test_dem_modality(self):
        p = Path("D004-2021_DEM_ELEV_AA-S1-32_10-20.tif")
        assert parse_tile_coords(p) == (10, 20)

    def test_label_modality(self):
        p = Path("D004-2021_AERIAL_LABEL-COSIA_AA-S1-32_1-1.tif")
        assert parse_tile_coords(p) == (1, 1)


# ---------------------------------------------------------------------------
# parse_roi_from_path
# ---------------------------------------------------------------------------


class TestParseRoiFromPath:
    """Tests for ROI extraction from file paths."""

    def test_from_parent_directory(self):
        p = Path("/data/D004/AA-S1-32/D004-2021_AERIAL_RGBI_AA-S1-32_3-7.tif")
        assert parse_roi_from_path(p) == "AA-S1-32"

    def test_from_different_roi(self):
        p = Path("/data/BB-S2-10/D005_AERIAL_RGBI_BB-S2-10_1-1.tif")
        assert parse_roi_from_path(p) == "BB-S2-10"

    def test_non_roi_directory(self):
        """When parent is not a standard ROI name, try filename parsing."""
        p = Path("/data/images/D004-2021_AERIAL_RGBI_CC-S3-5_2-4.tif")
        # Parent "images" doesn't match ROI pattern.
        # Filename parsing should extract CC-S3-5.
        result = parse_roi_from_path(p)
        assert result == "CC-S3-5"

    def test_no_roi_found(self):
        p = Path("/data/images/random_file_3-7.tif")
        result = parse_roi_from_path(p)
        assert result is None


# ---------------------------------------------------------------------------
# normalize_to_uint8
# ---------------------------------------------------------------------------


class TestNormalizeToUint8:
    """Tests for radiometric normalisation."""

    def test_uint8_passthrough(self):
        """uint8 data should be returned as-is (copy)."""
        data = np.array([0, 128, 255], dtype=np.uint8)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, data)
        # Should be a copy, not the same object.
        assert result is not data

    def test_uint16_stretch(self):
        """uint16 data should be stretched to [0, 255]."""
        data = np.array([100, 500, 1000], dtype=np.uint16)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_float32_stretch(self):
        """float32 data should be stretched to [0, 255]."""
        data = np.array([-10.0, 0.0, 50.0, 100.0], dtype=np.float32)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_image(self):
        """A constant image should map to mid-grey (128)."""
        data = np.full((10, 10), 42.0, dtype=np.float64)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, 128)

    def test_empty_array(self):
        """Empty array should return empty uint8 array."""
        data = np.array([], dtype=np.float32)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        assert result.size == 0

    def test_3d_array(self):
        """Should work with (bands, H, W) arrays."""
        data = np.random.randint(0, 65535, size=(4, 64, 64), dtype=np.uint16)
        result = normalize_to_uint8(data)
        assert result.dtype == np.uint8
        assert result.shape == (4, 64, 64)

    def test_custom_percentiles(self):
        """Custom percentile values should be respected."""
        data = np.arange(0, 1000, dtype=np.float32)
        result_default = normalize_to_uint8(data, low_pct=2.0, high_pct=98.0)
        result_narrow = normalize_to_uint8(data, low_pct=10.0, high_pct=90.0)
        assert result_default.dtype == np.uint8
        assert result_narrow.dtype == np.uint8
        # Both should span [0, 255] but with different distributions.
