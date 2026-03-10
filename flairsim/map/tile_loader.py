"""
Tile loader for FLAIR-HUB GeoTIFF patches.

This module handles the low-level reading of individual GeoTIFF files
produced by the FLAIR-HUB dataset pipeline.  It parses the FLAIR-HUB
file-naming convention to extract spatial coordinates and provides
radiometric normalisation utilities.

File naming convention (from the FLAIR-HUB data paper, Section III):

    {DOMAIN}_{SENSOR}_{DATATYPE}_{ROI}_{ROW}-{COL}.tif

    Example: D004-2021_AERIAL_RGBI_AA-S1-32_3-7.tif
             ^^^^^^^^^ ^^^^^^^^^^^^^ ^^^^^^^^ ^^^
             domain    sensor+type   ROI id   row-col in ROI grid

Each patch covers exactly 102.4 m x 102.4 m on the ground.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

# Regex for extracting the row-column position from a FLAIR-HUB filename.
# Matches patterns like ``_3-7.tif`` at the end of the filename stem.
_TILE_COORD_RE = re.compile(r"_(\d+)-(\d+)$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TileInfo:
    """Lightweight metadata about a single GeoTIFF tile on disk.

    Attributes
    ----------
    path : Path
        Absolute path to the ``.tif`` file.
    row : int
        Row index of this tile in the ROI grid (0-based by convention,
        though FLAIR-HUB uses 1-based in filenames -- we store as-is).
    col : int
        Column index in the ROI grid.
    x_min : float
        Western boundary in the map CRS (metres, EPSG:2154).
    y_min : float
        Southern boundary in the map CRS.
    x_max : float
        Eastern boundary.
    y_max : float
        Northern boundary.
    """

    path: Path
    row: int
    col: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class TileData:
    """In-memory raster data for a single tile.

    Attributes
    ----------
    info : TileInfo
        Spatial metadata for this tile.
    data : np.ndarray
        Pixel array with shape ``(bands, height, width)``.  The band
        dimension is always first (rasterio convention).
    """

    info: TileInfo
    data: np.ndarray = field(repr=False)

    @property
    def height(self) -> int:
        """Tile height in pixels."""
        return self.data.shape[1]

    @property
    def width(self) -> int:
        """Tile width in pixels."""
        return self.data.shape[2]

    @property
    def bands(self) -> int:
        """Number of bands."""
        return self.data.shape[0]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_tile_coords(filepath: Path) -> Optional[Tuple[int, int]]:
    """Extract the ``(row, col)`` grid position from a FLAIR-HUB filename.

    Parameters
    ----------
    filepath : Path
        Path (or just filename) of a ``.tif`` file following the
        FLAIR-HUB naming convention.

    Returns
    -------
    tuple[int, int] or None
        ``(row, col)`` if the filename matches, otherwise ``None``.

    Examples
    --------
    >>> parse_tile_coords(Path("D004-2021_AERIAL_RGBI_AA-S1-32_3-7.tif"))
    (3, 7)
    >>> parse_tile_coords(Path("not_a_flair_file.tif")) is None
    True
    """
    match = _TILE_COORD_RE.search(filepath.stem)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_roi_from_path(filepath: Path) -> Optional[str]:
    """Extract the ROI identifier from the parent directory name.

    In the FLAIR-HUB layout each ROI has its own sub-directory
    (e.g. ``AA-S1-32/``).  When files are stored flat (no ROI
    sub-directory), we fall back to parsing the filename itself.

    Parameters
    ----------
    filepath : Path
        Path to a ``.tif`` file.

    Returns
    -------
    str or None
        ROI identifier string, or ``None`` if it cannot be determined.
    """
    # Try the parent directory first (standard FLAIR-HUB layout).
    parent_name = filepath.parent.name
    if re.match(r"^[A-Z]{2}-S\d+-\d+$", parent_name):
        return parent_name

    # Fallback: parse from the filename.
    # Pattern: {DOMAIN}_{SENSOR}_{DATATYPE}_{ROI}_{ROW}-{COL}.tif
    parts = filepath.stem.rsplit("_", maxsplit=2)
    if len(parts) >= 2:
        candidate = parts[-2]
        if re.match(r"^[A-Z]{2}-S\d+-\d+$", candidate):
            return candidate

    return None


# ---------------------------------------------------------------------------
# Tile loading
# ---------------------------------------------------------------------------


def read_tile(filepath: Path) -> TileData:
    """Read a single GeoTIFF tile from disk.

    The tile's geographic bounds are read from the file's embedded
    affine transform (EPSG:2154 Lambert-93 coordinates).

    Parameters
    ----------
    filepath : Path
        Path to the ``.tif`` file.

    Returns
    -------
    TileData
        In-memory tile with raster data and spatial metadata.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the filename does not follow the FLAIR-HUB naming convention
        (i.e. we cannot extract a row/col position).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Tile file not found: {filepath}")

    coords = parse_tile_coords(filepath)
    if coords is None:
        raise ValueError(
            f"Cannot extract (row, col) from filename: {filepath.name}. "
            "Expected FLAIR-HUB naming convention ending with _ROW-COL.tif"
        )
    row, col = coords

    with rasterio.open(filepath) as src:
        data = src.read()  # shape: (bands, height, width)
        bounds = src.bounds  # BoundingBox(left, bottom, right, top)

    info = TileInfo(
        path=filepath,
        row=row,
        col=col,
        x_min=bounds.left,
        y_min=bounds.bottom,
        x_max=bounds.right,
        y_max=bounds.top,
    )

    logger.debug(
        "Loaded tile %s  row=%d col=%d  shape=%s  bounds=(%.1f, %.1f, %.1f, %.1f)",
        filepath.name,
        row,
        col,
        data.shape,
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
    )

    return TileData(info=info, data=data)


# ---------------------------------------------------------------------------
# Radiometric normalisation
# ---------------------------------------------------------------------------


def normalize_to_uint8(
    data: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """Percentile-based contrast stretch to ``uint8`` range.

    This is the standard approach used for visualising aerial and
    satellite imagery where the raw dynamic range may be much wider
    than 8 bits (e.g. ``uint16`` reflectance or ``float32`` elevation).

    Parameters
    ----------
    data : np.ndarray
        Input array of any numeric dtype.  Can be 2-D ``(H, W)`` or
        3-D ``(bands, H, W)`` / ``(H, W, bands)``.
    low_pct : float
        Lower percentile for clipping (default 2 %).
    high_pct : float
        Upper percentile for clipping (default 98 %).

    Returns
    -------
    np.ndarray
        Array of dtype ``uint8`` with values in ``[0, 255]``.
    """
    if data.size == 0:
        return data.astype(np.uint8)

    # Already uint8 -- return a copy to avoid side-effects.
    if data.dtype == np.uint8:
        return data.copy()

    data_f = data.astype(np.float64)
    lo = np.percentile(data_f, low_pct)
    hi = np.percentile(data_f, high_pct)

    if hi - lo < 1e-8:
        # Constant image -- map everything to mid-grey.
        return np.full_like(data, 128, dtype=np.uint8)

    clipped = np.clip(data_f, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)
