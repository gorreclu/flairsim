"""
Tests for grid overlay (Feature 2).

Covers:
- ``GridOverlay`` construction, validation, and properties.
- Cell label generation (``cell_labels``).
- Cell geometry: ``cell_bounds``, ``cell_center``, ``cell_from_pixel``.
- Label parsing and validation (``_parse_label``).
- ``draw()`` — pure NumPy/PIL drawing on (H, W, 3) images.
- ``GridConfig`` customisation.
- ``_apply_grid_overlay`` server helper.
- Server endpoint ``?grid=N`` query parameter.
- Edge cases: 1x1 grid, 26x26 grid, non-square images, boundary pixels.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from PIL import Image

from flairsim.core.grid import MAX_GRID_SIZE, GridConfig, GridOverlay, _blend


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestGridOverlayConstruction:
    """Tests for GridOverlay.__init__ and properties."""

    def test_default_construction(self):
        overlay = GridOverlay(n=4)
        assert overlay.n == 4
        assert isinstance(overlay.config, GridConfig)

    def test_custom_config(self):
        cfg = GridConfig(line_color=(255, 0, 0), line_alpha=1.0, line_width=5)
        overlay = GridOverlay(n=3, config=cfg)
        assert overlay.config.line_color == (255, 0, 0)
        assert overlay.config.line_alpha == 1.0
        assert overlay.config.line_width == 5

    def test_min_grid_size(self):
        overlay = GridOverlay(n=1)
        assert overlay.n == 1

    def test_max_grid_size(self):
        overlay = GridOverlay(n=26)
        assert overlay.n == 26

    def test_invalid_grid_size_zero(self):
        with pytest.raises(ValueError, match="between 1 and"):
            GridOverlay(n=0)

    def test_invalid_grid_size_negative(self):
        with pytest.raises(ValueError, match="between 1 and"):
            GridOverlay(n=-1)

    def test_invalid_grid_size_too_large(self):
        with pytest.raises(ValueError, match="between 1 and"):
            GridOverlay(n=27)

    def test_repr(self):
        overlay = GridOverlay(n=4)
        assert "GridOverlay(n=4)" in repr(overlay)


# ---------------------------------------------------------------------------
# Cell labels
# ---------------------------------------------------------------------------


class TestCellLabels:
    """Tests for cell_labels property."""

    def test_2x2_labels(self):
        overlay = GridOverlay(n=2)
        assert overlay.cell_labels == ["A1", "A2", "B1", "B2"]

    def test_3x3_labels(self):
        overlay = GridOverlay(n=3)
        expected = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
        assert overlay.cell_labels == expected

    def test_1x1_labels(self):
        overlay = GridOverlay(n=1)
        assert overlay.cell_labels == ["A1"]

    def test_label_count(self):
        for n in [1, 2, 4, 8, 16, 26]:
            overlay = GridOverlay(n=n)
            assert len(overlay.cell_labels) == n * n

    def test_first_and_last_labels_large_grid(self):
        overlay = GridOverlay(n=10)
        labels = overlay.cell_labels
        assert labels[0] == "A1"
        assert labels[-1] == "J10"

    def test_26x26_last_label(self):
        overlay = GridOverlay(n=26)
        labels = overlay.cell_labels
        assert labels[-1] == "Z26"


# ---------------------------------------------------------------------------
# Cell geometry
# ---------------------------------------------------------------------------


class TestCellBounds:
    """Tests for cell_bounds()."""

    def test_top_left_cell(self):
        overlay = GridOverlay(n=4)
        x_min, y_min, x_max, y_max = overlay.cell_bounds("A1", 400, 400)
        assert x_min == 0
        assert y_min == 0
        assert x_max == 100
        assert y_max == 100

    def test_bottom_right_cell(self):
        overlay = GridOverlay(n=4)
        x_min, y_min, x_max, y_max = overlay.cell_bounds("D4", 400, 400)
        assert x_min == 300
        assert y_min == 300
        assert x_max == 400
        assert y_max == 400

    def test_middle_cell(self):
        overlay = GridOverlay(n=4)
        x_min, y_min, x_max, y_max = overlay.cell_bounds("B3", 400, 400)
        assert x_min == 200
        assert y_min == 100
        assert x_max == 300
        assert y_max == 200

    def test_non_square_image(self):
        overlay = GridOverlay(n=2)
        x_min, y_min, x_max, y_max = overlay.cell_bounds("A2", 600, 400)
        assert x_min == 300
        assert y_min == 0
        assert x_max == 600
        assert y_max == 200

    def test_1x1_grid_covers_whole_image(self):
        overlay = GridOverlay(n=1)
        x_min, y_min, x_max, y_max = overlay.cell_bounds("A1", 500, 300)
        assert x_min == 0
        assert y_min == 0
        assert x_max == 500
        assert y_max == 300

    def test_invalid_label_raises(self):
        overlay = GridOverlay(n=4)
        with pytest.raises(ValueError):
            overlay.cell_bounds("E1", 400, 400)  # Row E doesn't exist in 4x4
        with pytest.raises(ValueError):
            overlay.cell_bounds("A5", 400, 400)  # Col 5 doesn't exist in 4x4


class TestCellCenter:
    """Tests for cell_center()."""

    def test_center_top_left(self):
        overlay = GridOverlay(n=4)
        cx, cy = overlay.cell_center("A1", 400, 400)
        assert cx == 50
        assert cy == 50

    def test_center_bottom_right(self):
        overlay = GridOverlay(n=4)
        cx, cy = overlay.cell_center("D4", 400, 400)
        assert cx == 350
        assert cy == 350

    def test_center_1x1(self):
        overlay = GridOverlay(n=1)
        cx, cy = overlay.cell_center("A1", 100, 200)
        assert cx == 50
        assert cy == 100


class TestCellFromPixel:
    """Tests for cell_from_pixel()."""

    def test_top_left_corner(self):
        overlay = GridOverlay(n=4)
        assert overlay.cell_from_pixel(0, 0, 400, 400) == "A1"

    def test_bottom_right_corner(self):
        overlay = GridOverlay(n=4)
        # Last valid pixel is (399, 399)
        assert overlay.cell_from_pixel(399, 399, 400, 400) == "D4"

    def test_middle_pixel(self):
        overlay = GridOverlay(n=4)
        assert overlay.cell_from_pixel(250, 150, 400, 400) == "B3"

    def test_boundary_pixel_horizontal(self):
        overlay = GridOverlay(n=4)
        # Pixel at exactly x=100 should be in column 2.
        assert overlay.cell_from_pixel(100, 50, 400, 400) == "A2"

    def test_boundary_pixel_vertical(self):
        overlay = GridOverlay(n=4)
        # Pixel at exactly y=100 should be in row B.
        assert overlay.cell_from_pixel(50, 100, 400, 400) == "B1"

    def test_pixel_outside_raises(self):
        overlay = GridOverlay(n=4)
        with pytest.raises(ValueError, match="outside image"):
            overlay.cell_from_pixel(400, 0, 400, 400)
        with pytest.raises(ValueError, match="outside image"):
            overlay.cell_from_pixel(0, 400, 400, 400)
        with pytest.raises(ValueError, match="outside image"):
            overlay.cell_from_pixel(-1, 0, 400, 400)

    def test_1x1_grid_all_pixels_same_cell(self):
        overlay = GridOverlay(n=1)
        assert overlay.cell_from_pixel(0, 0, 100, 100) == "A1"
        assert overlay.cell_from_pixel(99, 99, 100, 100) == "A1"
        assert overlay.cell_from_pixel(50, 50, 100, 100) == "A1"

    def test_roundtrip_center_to_pixel(self):
        """cell_from_pixel(cell_center(label)) should return the same label."""
        overlay = GridOverlay(n=4)
        w, h = 400, 400
        for label in overlay.cell_labels:
            cx, cy = overlay.cell_center(label, w, h)
            assert overlay.cell_from_pixel(cx, cy, w, h) == label


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------


class TestLabelParsing:
    """Tests for _parse_label (indirectly through public API)."""

    def test_lowercase_label(self):
        """Labels should be case-insensitive."""
        overlay = GridOverlay(n=4)
        # cell_bounds internally calls _parse_label.
        b1 = overlay.cell_bounds("a1", 400, 400)
        b2 = overlay.cell_bounds("A1", 400, 400)
        assert b1 == b2

    def test_multi_digit_column(self):
        overlay = GridOverlay(n=12)
        # "A12" should work — row A, column 12.
        x_min, y_min, x_max, y_max = overlay.cell_bounds("A12", 1200, 1200)
        assert x_min > 0  # Not the first column.
        assert x_max == 1200  # Last column.

    def test_empty_label_raises(self):
        overlay = GridOverlay(n=4)
        with pytest.raises(ValueError):
            overlay.cell_bounds("", 400, 400)

    def test_single_char_label_raises(self):
        overlay = GridOverlay(n=4)
        with pytest.raises(ValueError):
            overlay.cell_bounds("A", 400, 400)

    def test_invalid_row_raises(self):
        overlay = GridOverlay(n=4)
        with pytest.raises(ValueError):
            overlay.cell_bounds("1A", 400, 400)  # digit as row

    def test_invalid_column_raises(self):
        overlay = GridOverlay(n=4)
        with pytest.raises(ValueError):
            overlay.cell_bounds("AB", 400, 400)  # letter as column


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


class TestDraw:
    """Tests for draw() — pure NumPy/PIL overlay."""

    def test_output_shape_and_dtype(self):
        overlay = GridOverlay(n=4)
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        result = overlay.draw(img)
        assert result.shape == (400, 400, 3)
        assert result.dtype == np.uint8

    def test_original_not_modified(self):
        overlay = GridOverlay(n=4)
        img = np.full((400, 400, 3), 128, dtype=np.uint8)
        original = img.copy()
        _ = overlay.draw(img)
        np.testing.assert_array_equal(img, original)

    def test_grid_lines_visible(self):
        """White grid lines on a black image should produce non-zero pixels."""
        cfg = GridConfig(line_color=(255, 255, 255), line_alpha=1.0, line_width=2)
        overlay = GridOverlay(n=4, config=cfg)
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        result = overlay.draw(img)
        # Vertical line at x=100 should have non-zero pixels.
        assert result[:, 100, :].max() > 0

    def test_labels_visible(self):
        """After drawing, the result should differ from just lines."""
        overlay = GridOverlay(n=2)
        img = np.full((200, 200, 3), 100, dtype=np.uint8)
        result = overlay.draw(img)
        # The result should not be identical to the input.
        assert not np.array_equal(img, result)

    def test_1x1_grid_no_lines(self):
        """A 1x1 grid has no internal lines (but may have labels)."""
        cfg = GridConfig(
            line_color=(255, 0, 0),
            line_alpha=1.0,
            label_bg_color=None,  # No label background
        )
        overlay = GridOverlay(n=1, config=cfg)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = overlay.draw(img)
        # No red lines should appear (only the label text in white).
        # Check that no pixel is pure red (255, 0, 0).
        red_mask = (
            (result[:, :, 0] == 255) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)
        )
        assert not red_mask.any()

    def test_invalid_input_shape_raises(self):
        overlay = GridOverlay(n=4)
        with pytest.raises(ValueError, match="Expected .* image"):
            overlay.draw(np.zeros((400, 400), dtype=np.uint8))  # 2D
        with pytest.raises(ValueError, match="Expected .* image"):
            overlay.draw(np.zeros((400, 400, 4), dtype=np.uint8))  # 4 channels

    def test_small_image(self):
        """Grid should work on very small images without crashing."""
        overlay = GridOverlay(n=2)
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        result = overlay.draw(img)
        assert result.shape == (20, 20, 3)

    def test_non_square_image(self):
        overlay = GridOverlay(n=3)
        img = np.zeros((300, 600, 3), dtype=np.uint8)
        result = overlay.draw(img)
        assert result.shape == (300, 600, 3)


# ---------------------------------------------------------------------------
# GridConfig defaults
# ---------------------------------------------------------------------------


class TestGridConfig:
    """Tests for GridConfig dataclass."""

    def test_defaults(self):
        cfg = GridConfig()
        assert cfg.line_color == (255, 255, 255)
        assert cfg.line_alpha == 0.6
        assert cfg.line_width == 2
        assert cfg.label_color == (255, 255, 255)
        assert cfg.label_bg_color == (0, 0, 0)
        assert cfg.label_bg_alpha == 0.5
        assert cfg.font_scale == 1.0

    def test_frozen(self):
        cfg = GridConfig()
        with pytest.raises(AttributeError):
            cfg.line_width = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _blend utility
# ---------------------------------------------------------------------------


class TestBlend:
    """Tests for the _blend alpha-blending utility."""

    def test_alpha_zero_returns_base(self):
        base = np.full((10, 10, 3), 100, dtype=np.uint8)
        over = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = _blend(base, over, 0.0)
        np.testing.assert_array_equal(result, base)

    def test_alpha_one_returns_overlay(self):
        base = np.full((10, 10, 3), 100, dtype=np.uint8)
        over = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = _blend(base, over, 1.0)
        np.testing.assert_array_equal(result, over)

    def test_alpha_half(self):
        base = np.full((10, 10, 3), 100, dtype=np.uint8)
        over = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = _blend(base, over, 0.5)
        assert result.dtype == np.uint8
        # (100 * 0.5 + 200 * 0.5) = 150
        np.testing.assert_array_equal(result, 150)


# ---------------------------------------------------------------------------
# Server integration: _apply_grid_overlay
# ---------------------------------------------------------------------------


class TestApplyGridOverlay:
    """Tests for the _apply_grid_overlay helper in app.py."""

    def test_no_grid_returns_original(self):
        from flairsim.server.app import _apply_grid_overlay

        img = np.random.randint(0, 256, (3, 100, 100), dtype=np.uint8)
        result = _apply_grid_overlay(img, None)
        np.testing.assert_array_equal(result, img)

    def test_with_grid_changes_image(self):
        from flairsim.server.app import _apply_grid_overlay

        overlay = GridOverlay(n=4)
        img = np.full((3, 100, 100), 128, dtype=np.uint8)
        result = _apply_grid_overlay(img, overlay)
        assert result.shape[0] == 3  # (3, H, W) format preserved
        assert result.shape[1:] == (100, 100)
        # The result should differ from the input.
        assert not np.array_equal(result, img)

    def test_with_grid_on_float32_image(self):
        from flairsim.server.app import _apply_grid_overlay

        overlay = GridOverlay(n=2)
        img = np.random.rand(3, 100, 100).astype(np.float32) * 1000
        result = _apply_grid_overlay(img, overlay)
        assert result.dtype == np.uint8
        assert result.shape == (3, 100, 100)

    def test_with_grid_on_1band_image(self):
        from flairsim.server.app import _apply_grid_overlay

        overlay = GridOverlay(n=2)
        img = np.full((1, 100, 100), 128, dtype=np.uint8)
        result = _apply_grid_overlay(img, overlay)
        assert result.shape == (3, 100, 100)

    def test_with_grid_on_2band_image(self):
        from flairsim.server.app import _apply_grid_overlay

        overlay = GridOverlay(n=2)
        img = np.full((2, 100, 100), 128, dtype=np.uint8)
        result = _apply_grid_overlay(img, overlay)
        assert result.shape == (3, 100, 100)

    def test_with_grid_on_2d_image(self):
        from flairsim.server.app import _apply_grid_overlay

        overlay = GridOverlay(n=2)
        img = np.full((100, 100), 128, dtype=np.uint8)
        result = _apply_grid_overlay(img, overlay)
        assert result.shape == (3, 100, 100)


# ---------------------------------------------------------------------------
# Server endpoint integration (FastAPI TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture
def _tile_dir(tmp_path: Path) -> Path:
    """Create a minimal 2x2 tile directory for server tests.

    Follows the FLAIR-HUB naming convention expected by MapManager:
    ``<data_dir>/<ROI>/<DOMAIN>_<SENSOR>_<ROI>_<row>-<col>.tif``
    """
    import rasterio
    from rasterio.transform import from_bounds

    TILE_PX = 64
    TILE_GSD = 0.2
    N_BANDS = 4
    ORIGIN_X = 800_000.0
    ORIGIN_Y = 6_400_000.0
    DOMAIN = "D099-2099"
    SENSOR_TYPE = "AERIAL_RGBI"
    ROI = "AB-S1-01"

    # data_dir is the top-level directory (named like the dataset).
    data_dir = tmp_path / f"{DOMAIN}_{SENSOR_TYPE}"
    roi_dir = data_dir / ROI
    roi_dir.mkdir(parents=True)

    tile_ground = TILE_PX * TILE_GSD
    for row in range(2):
        for col in range(2):
            x0 = ORIGIN_X + col * tile_ground
            y0 = ORIGIN_Y + row * tile_ground
            x1 = x0 + tile_ground
            y1 = y0 + tile_ground

            tfm = from_bounds(x0, y0, x1, y1, TILE_PX, TILE_PX)
            # Use the correct naming: {row}-{col} (not zero-padded).
            fname = f"{DOMAIN}_{SENSOR_TYPE}_{ROI}_{row}-{col}.tif"
            fp = roi_dir / fname
            with rasterio.open(
                fp,
                "w",
                driver="GTiff",
                width=TILE_PX,
                height=TILE_PX,
                count=N_BANDS,
                dtype="uint8",
                crs="EPSG:2154",
                transform=tfm,
            ) as dst:
                for b in range(1, N_BANDS + 1):
                    dst.write(
                        np.random.randint(0, 256, (TILE_PX, TILE_PX), dtype=np.uint8),
                        b,
                    )

    return data_dir


@pytest.fixture
def server_app(_tile_dir: "Path"):
    """Create a FastAPI app with a minimal tile set."""
    from flairsim.server.app import create_app

    return create_app(data_dir=str(_tile_dir), preload_tiles=True, grid=None)


class TestServerGridEndpoints:
    """Test ?grid=N query parameter on /reset and /step."""

    def test_reset_with_grid(self, server_app):
        from fastapi.testclient import TestClient

        client = TestClient(server_app)
        # Reset with grid=4.
        resp = client.post("/reset?grid=4")
        assert resp.status_code == 200
        data = resp.json()
        assert "image_base64" in data

        # Decode the image and check it's valid PNG.
        img_bytes = base64.b64decode(data["image_base64"])
        img = Image.open(io.BytesIO(img_bytes))
        assert img.size[0] > 0 and img.size[1] > 0

    def test_step_with_grid(self, server_app):
        from fastapi.testclient import TestClient

        client = TestClient(server_app)
        client.post("/reset")
        resp = client.post(
            "/step?grid=3",
            json={"dx": 1.0, "dy": 0.0, "dz": 0.0, "action_type": "move"},
        )
        assert resp.status_code == 200

    def test_grid_zero_disables(self, server_app):
        from fastapi.testclient import TestClient

        client = TestClient(server_app)
        # Enable grid.
        client.post("/reset?grid=4")
        # Disable with grid=0.
        resp = client.post(
            "/step?grid=0",
            json={"dx": 0.0, "dy": 0.0, "dz": 0.0, "action_type": "move"},
        )
        assert resp.status_code == 200

    def test_reset_without_grid_no_overlay(self, server_app):
        """When no grid param is given, images are returned as-is."""
        from fastapi.testclient import TestClient

        client = TestClient(server_app)
        resp = client.post("/reset")
        assert resp.status_code == 200


class TestServerGridDefaultParam:
    """Test that --grid CLI default is passed through create_app."""

    def test_default_grid_applied_on_reset(self, _tile_dir):
        from fastapi.testclient import TestClient
        from flairsim.server.app import create_app

        app = create_app(data_dir=str(_tile_dir), preload_tiles=True, grid=4)
        client = TestClient(app)
        resp = client.post("/reset")
        assert resp.status_code == 200
        data = resp.json()
        # Image should be valid.
        img_bytes = base64.b64decode(data["image_base64"])
        img = Image.open(io.BytesIO(img_bytes))
        assert img.size[0] > 0


# ---------------------------------------------------------------------------
# Import from package root
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Verify GridOverlay and GridConfig are importable from flairsim."""

    def test_grid_overlay_import(self):
        from flairsim import GridOverlay as GO

        assert GO is GridOverlay

    def test_grid_config_import(self):
        from flairsim import GridConfig as GC

        assert GC is GridConfig
