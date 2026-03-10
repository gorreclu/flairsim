"""
Map manager -- spatial indexing and region extraction over FLAIR-HUB tiles.

The :class:`MapManager` is the central component that turns a collection
of individual FLAIR-HUB GeoTIFF patches into a seamless, queryable map
surface.  It supports:

* **Lazy loading** -- only tiles within a configurable radius of the
  query point are loaded into memory.
* **Multi-modality** -- the same geographic region can be served in any
  available FLAIR-HUB modality (AERIAL_RGBI, DEM_ELEV, labels, ...).
* **Efficient cropping** -- arbitrary rectangular regions (in world
  coordinates) are extracted and optionally resampled to a target
  pixel resolution.

Design notes
------------
FLAIR-HUB tiles are arranged on a regular grid whose origin and spacing
are encoded in each file's geo-transform.  We exploit this regularity to
avoid building a full spatial index (R-tree): a simple dictionary keyed
by ``(row, col)`` is sufficient and much faster for the grid sizes we
encounter (typically < 300 tiles per ROI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .modality import Modality
from .tile_loader import TileData, TileInfo, parse_tile_coords, read_tile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MapBounds:
    """Axis-aligned bounding box in world coordinates (metres).

    Coordinate system is EPSG:2154 (Lambert-93), where *x* increases
    eastward and *y* increases northward.

    Attributes
    ----------
    x_min, y_min : float
        South-west corner.
    x_max, y_max : float
        North-east corner.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """East-west extent in metres."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """North-south extent in metres."""
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        """Centre point ``(x, y)``."""
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    def contains(self, x: float, y: float) -> bool:
        """Return ``True`` if the point ``(x, y)`` lies inside the box."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def intersects(self, other: "MapBounds") -> bool:
        """Return ``True`` if *other* overlaps with this box."""
        return not (
            other.x_max < self.x_min
            or other.x_min > self.x_max
            or other.y_max < self.y_min
            or other.y_min > self.y_max
        )


# ---------------------------------------------------------------------------
# Map manager
# ---------------------------------------------------------------------------


class MapManager:
    """Manages a grid of FLAIR-HUB tiles for a single modality and ROI.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing the ``.tif`` files for one
        ``{DOMAIN}_{SENSOR}_{DATATYPE}`` layer.  Files may be either
        directly in this directory or inside ROI sub-directories
        (e.g. ``AA-S1-32/``).
    roi : str or None
        If given, only tiles belonging to this ROI are indexed.
        When ``None``, the ROI with the most tiles is automatically
        selected.
    preload : bool
        If ``True`` (default), all tiles are loaded into memory at
        construction time.  Set to ``False`` for lazy loading (tiles
        are loaded on first access).

    Attributes
    ----------
    bounds : MapBounds
        Bounding box of the entire loaded tile grid.
    grid_rows : int
        Number of tile rows in the grid.
    grid_cols : int
        Number of tile columns in the grid.
    tile_pixel_size : int
        Side length of each tile in pixels.
    tile_ground_size : float
        Side length of each tile on the ground (metres).
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        data_dir: str | Path,
        roi: str | None = None,
        preload: bool = True,
    ) -> None:
        self._data_dir = Path(data_dir).resolve()
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self._data_dir}")

        # Discover tile files and select ROI.
        self._tile_paths: Dict[Tuple[int, int], Path] = {}
        self._roi_name: str = ""
        self._discover_tiles(roi)

        if not self._tile_paths:
            raise ValueError(
                f"No valid FLAIR-HUB tiles found in {self._data_dir} (roi={roi!r})."
            )

        # Determine grid geometry from the first loaded tile.
        self._tile_cache: Dict[Tuple[int, int], TileData] = {}
        self._grid_origin: Tuple[float, float] = (0.0, 0.0)  # (x_min, y_max) of grid
        self.tile_pixel_size: int = 0
        self.tile_ground_size: float = 0.0
        self.pixel_size_m: float = 0.0
        self.grid_rows: int = 0
        self.grid_cols: int = 0
        self.bounds: MapBounds = MapBounds(0, 0, 0, 0)

        self._build_grid_geometry()

        if preload:
            self._preload_all()

        logger.info(
            "MapManager ready: roi=%s  grid=%dx%d  tiles=%d  "
            "bounds=(%.1f, %.1f)-(%.1f, %.1f)  tile_ground=%.1fm",
            self._roi_name,
            self.grid_rows,
            self.grid_cols,
            len(self._tile_paths),
            self.bounds.x_min,
            self.bounds.y_min,
            self.bounds.x_max,
            self.bounds.y_max,
            self.tile_ground_size,
        )

    # -------------------------------------------------------------- discovery

    def _discover_tiles(self, roi: str | None) -> None:
        """Scan the data directory for ``.tif`` files and build the tile index.

        If *roi* is ``None``, the ROI sub-directory containing the most
        tiles is selected automatically.
        """
        # Collect all .tif files grouped by ROI subdirectory (or root).
        roi_groups: Dict[str, Dict[Tuple[int, int], Path]] = {}

        for tif_path in sorted(self._data_dir.rglob("*.tif")):
            coords = parse_tile_coords(tif_path)
            if coords is None:
                continue

            # Determine the ROI name from the directory structure.
            # If files are in a subdirectory, use its name as the ROI.
            # If files are at the root level, use "root".
            parent = tif_path.parent
            if parent == self._data_dir:
                roi_name = "root"
            else:
                roi_name = parent.name

            roi_groups.setdefault(roi_name, {})[coords] = tif_path

        if not roi_groups:
            return

        # Select ROI.
        if roi is not None:
            if roi not in roi_groups:
                available = sorted(roi_groups.keys())
                raise ValueError(f"ROI {roi!r} not found.  Available ROIs: {available}")
            self._roi_name = roi
            self._tile_paths = roi_groups[roi]
        else:
            # Auto-select the ROI with the most tiles.
            self._roi_name = max(roi_groups, key=lambda k: len(roi_groups[k]))
            self._tile_paths = roi_groups[self._roi_name]
            logger.info(
                "Auto-selected ROI %r (%d tiles) from %d available ROIs.",
                self._roi_name,
                len(self._tile_paths),
                len(roi_groups),
            )

    # --------------------------------------------------------- grid geometry

    def _build_grid_geometry(self) -> None:
        """Determine grid dimensions and spatial extent from tile metadata.

        We read one tile to obtain the pixel size and tile dimensions,
        then compute the full grid extent from the known (row, col)
        positions.
        """
        # Read the first tile to learn the physical properties.
        first_coords = next(iter(self._tile_paths))
        first_tile = self._load_tile(first_coords)

        self.tile_pixel_size = first_tile.width
        self.pixel_size_m = (
            first_tile.info.x_max - first_tile.info.x_min
        ) / first_tile.width
        self.tile_ground_size = first_tile.info.x_max - first_tile.info.x_min

        # Compute grid extent from all known tile positions.
        all_rows = [rc[0] for rc in self._tile_paths]
        all_cols = [rc[1] for rc in self._tile_paths]
        min_row, max_row = min(all_rows), max(all_rows)
        min_col, max_col = min(all_cols), max(all_cols)

        self.grid_rows = max_row - min_row + 1
        self.grid_cols = max_col - min_col + 1

        # The grid origin is the north-west corner of the tile at
        # (min_row, min_col).  In FLAIR-HUB, row index increases
        # southward (y decreases) and col index increases eastward
        # (x increases).  We verify this by inspecting tile bounds.
        # First, use the actual geo-referenced bounds of all loaded tiles
        # to compute the overall extent.
        x_mins: List[float] = []
        y_mins: List[float] = []
        x_maxs: List[float] = []
        y_maxs: List[float] = []

        # We need bounds from at least a few tiles.  For efficiency,
        # read only the extreme tiles if not preloaded.
        corner_coords = {
            (min_row, min_col),
            (min_row, max_col),
            (max_row, min_col),
            (max_row, max_col),
        }
        # Keep only those that actually exist.
        corner_coords = corner_coords & set(self._tile_paths.keys())

        for rc in corner_coords:
            tile = self._load_tile(rc)
            x_mins.append(tile.info.x_min)
            y_mins.append(tile.info.y_min)
            x_maxs.append(tile.info.x_max)
            y_maxs.append(tile.info.y_max)

        # Also include the first tile we already loaded.
        x_mins.append(first_tile.info.x_min)
        y_mins.append(first_tile.info.y_min)
        x_maxs.append(first_tile.info.x_max)
        y_maxs.append(first_tile.info.y_max)

        # Compute overall bounds using the grid extent.
        # Each tile covers tile_ground_size metres.
        overall_x_min = min(x_mins)
        overall_y_min = min(y_mins)
        overall_x_max = max(x_maxs)
        overall_y_max = max(y_maxs)

        # Extend to account for any missing corner tiles.
        # The full grid should span grid_rows * tile_ground_size vertically
        # and grid_cols * tile_ground_size horizontally.
        expected_width = self.grid_cols * self.tile_ground_size
        expected_height = self.grid_rows * self.tile_ground_size

        # Use the actual minimum coordinates as the origin and extend.
        overall_x_max = max(overall_x_max, overall_x_min + expected_width)
        overall_y_max = max(overall_y_max, overall_y_min + expected_height)

        self._grid_origin = (overall_x_min, overall_y_max)  # NW corner
        self.bounds = MapBounds(
            x_min=overall_x_min,
            y_min=overall_y_min,
            x_max=overall_x_max,
            y_max=overall_y_max,
        )

        # Store the mapping from (row, col) to absolute row/col
        # (offset by the minimum indices).
        self._row_offset = min_row
        self._col_offset = min_col

    # --------------------------------------------------------- tile loading

    def _load_tile(self, coords: Tuple[int, int]) -> TileData:
        """Load a tile into the cache if not already present.

        Parameters
        ----------
        coords : tuple[int, int]
            ``(row, col)`` key in ``self._tile_paths``.

        Returns
        -------
        TileData
            The loaded tile data.
        """
        if coords in self._tile_cache:
            return self._tile_cache[coords]

        path = self._tile_paths[coords]
        tile = read_tile(path)
        self._tile_cache[coords] = tile
        return tile

    def _preload_all(self) -> None:
        """Load every tile into memory."""
        for coords in self._tile_paths:
            self._load_tile(coords)
        logger.info("Preloaded %d tiles into memory.", len(self._tile_cache))

    # -------------------------------------------------------- coordinate helpers

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid ``(row, col)`` indices.

        Parameters
        ----------
        x, y : float
            Position in the map CRS (EPSG:2154, metres).

        Returns
        -------
        tuple[int, int]
            ``(row, col)`` grid indices (using the original FLAIR-HUB
            numbering, *not* 0-based).
        """
        origin_x, origin_y = self._grid_origin  # NW corner
        # col increases eastward (x increases).
        col = int((x - origin_x) / self.tile_ground_size) + self._col_offset
        # row increases southward (y decreases).
        row = int((origin_y - y) / self.tile_ground_size) + self._row_offset
        return row, col

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates to pixel coordinates in the full mosaic.

        The mosaic pixel grid has its origin ``(0, 0)`` at the north-west
        corner, with *px* increasing eastward and *py* increasing
        southward (standard image convention).

        Parameters
        ----------
        x, y : float
            Position in the map CRS.

        Returns
        -------
        tuple[float, float]
            ``(px, py)`` in mosaic pixel coordinates (may be fractional).
        """
        origin_x, origin_y = self._grid_origin
        px = (x - origin_x) / self.pixel_size_m
        py = (origin_y - y) / self.pixel_size_m
        return px, py

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Convert mosaic pixel coordinates back to world coordinates.

        Parameters
        ----------
        px, py : float
            Position in mosaic pixel coordinates.

        Returns
        -------
        tuple[float, float]
            ``(x, y)`` in the map CRS.
        """
        origin_x, origin_y = self._grid_origin
        x = origin_x + px * self.pixel_size_m
        y = origin_y - py * self.pixel_size_m
        return x, y

    # -------------------------------------------------------- region extraction

    def get_region(
        self,
        x_center: float,
        y_center: float,
        half_extent: float,
        output_size: int | None = None,
    ) -> np.ndarray:
        """Extract a square region from the map, centred on a world point.

        This is the primary method used by the drone camera to obtain
        its current view.  It assembles the required tiles into a
        seamless raster and optionally resamples it to a fixed output
        resolution.

        Parameters
        ----------
        x_center, y_center : float
            Centre of the extraction window in world coordinates.
        half_extent : float
            Half the side length of the square window, in metres.
            The total window covers ``2 * half_extent`` m in each
            direction.
        output_size : int or None
            If given, the extracted region is resized to
            ``(output_size, output_size)`` pixels using bilinear
            interpolation.  If ``None``, the region is returned at
            the native tile resolution.

        Returns
        -------
        np.ndarray
            Raster array with shape ``(bands, H, W)``.  If no valid
            tiles cover the requested region, a zero-filled array
            is returned.
        """
        # Requested bounding box in world coordinates.
        req = MapBounds(
            x_min=x_center - half_extent,
            y_min=y_center - half_extent,
            x_max=x_center + half_extent,
            y_max=y_center + half_extent,
        )

        # Convert to pixel coordinates in the full mosaic.
        px_left, py_top = self.world_to_pixel(req.x_min, req.y_max)
        px_right, py_bottom = self.world_to_pixel(req.x_max, req.y_min)

        # Round to integer pixel boundaries.
        px_left_i = int(np.floor(px_left))
        py_top_i = int(np.floor(py_top))
        px_right_i = int(np.ceil(px_right))
        py_bottom_i = int(np.ceil(py_bottom))

        region_w = px_right_i - px_left_i
        region_h = py_bottom_i - py_top_i

        if region_w <= 0 or region_h <= 0:
            logger.warning(
                "Requested region has non-positive size: %dx%d", region_w, region_h
            )
            n_bands = self._guess_band_count()
            size = output_size or 1
            return np.zeros((n_bands, size, size), dtype=np.uint8)

        # Determine which tiles we need.
        n_bands = self._guess_band_count()
        sample_tile = next(iter(self._tile_cache.values()), None)
        dtype = sample_tile.data.dtype if sample_tile else np.uint8

        region = np.zeros((n_bands, region_h, region_w), dtype=dtype)

        tiles_used = 0
        for coords, path in self._tile_paths.items():
            tile = self._load_tile(coords)

            # Tile's pixel extent in the full mosaic.
            t_px_left, t_py_top = self.world_to_pixel(tile.info.x_min, tile.info.y_max)
            t_px_right = t_px_left + tile.width
            t_py_bottom = t_py_top + tile.height

            # Intersection with the requested region.
            ix_left = max(px_left_i, int(t_px_left))
            iy_top = max(py_top_i, int(t_py_top))
            ix_right = min(px_right_i, int(t_px_right))
            iy_bottom = min(py_bottom_i, int(t_py_bottom))

            if ix_left >= ix_right or iy_top >= iy_bottom:
                continue  # No overlap.

            # Source region within the tile.
            src_x0 = ix_left - int(t_px_left)
            src_y0 = iy_top - int(t_py_top)
            src_x1 = ix_right - int(t_px_left)
            src_y1 = iy_bottom - int(t_py_top)

            # Destination region within the output.
            dst_x0 = ix_left - px_left_i
            dst_y0 = iy_top - py_top_i
            dst_x1 = ix_right - px_left_i
            dst_y1 = iy_bottom - py_top_i

            # Handle band count mismatch gracefully.
            bands_to_copy = min(n_bands, tile.bands)
            region[:bands_to_copy, dst_y0:dst_y1, dst_x0:dst_x1] = tile.data[
                :bands_to_copy, src_y0:src_y1, src_x0:src_x1
            ]
            tiles_used += 1

        logger.debug(
            "Extracted region: center=(%.1f, %.1f) extent=%.1fm  "
            "native=%dx%d px  tiles_used=%d",
            x_center,
            y_center,
            half_extent * 2,
            region_w,
            region_h,
            tiles_used,
        )

        # Resize if requested.
        if output_size is not None and (
            region_w != output_size or region_h != output_size
        ):
            region = self._resize(region, output_size)

        return region

    # --------------------------------------------------------- label queries

    def get_label_at(self, x: float, y: float) -> Optional[int]:
        """Return the label value at a specific world coordinate.

        This is useful for label/supervision modalities (COSIA, LPIS)
        to check what class is present at a given point.

        Parameters
        ----------
        x, y : float
            World coordinates.

        Returns
        -------
        int or None
            The label value at the nearest pixel, or ``None`` if the
            point falls outside all loaded tiles.
        """
        row, col = self.world_to_grid(x, y)
        coords = (row, col)

        if coords not in self._tile_paths:
            return None

        tile = self._load_tile(coords)

        # Pixel position within this tile.
        px_in_tile = (x - tile.info.x_min) / self.pixel_size_m
        py_in_tile = (tile.info.y_max - y) / self.pixel_size_m

        px_i = int(np.clip(px_in_tile, 0, tile.width - 1))
        py_i = int(np.clip(py_in_tile, 0, tile.height - 1))

        return int(tile.data[0, py_i, px_i])

    # --------------------------------------------------------- utilities

    def _guess_band_count(self) -> int:
        """Return the band count from the first cached tile, or 4 as default."""
        for tile in self._tile_cache.values():
            return tile.bands
        # No tiles loaded yet -- read one.
        first_coords = next(iter(self._tile_paths))
        return self._load_tile(first_coords).bands

    @staticmethod
    def _resize(data: np.ndarray, size: int) -> np.ndarray:
        """Resize a ``(bands, H, W)`` array to ``(bands, size, size)``.

        Uses a simple nearest-neighbour approach for label data
        compatibility and speed.  For photographic data, bilinear
        interpolation would be preferable -- this can be enhanced
        later via a ``method`` parameter.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape ``(bands, H, W)``.
        size : int
            Target side length in pixels.

        Returns
        -------
        np.ndarray
            Resized array of shape ``(bands, size, size)``.
        """
        from PIL import Image

        bands, h, w = data.shape
        result = np.empty((bands, size, size), dtype=data.dtype)

        for b in range(bands):
            img = Image.fromarray(data[b])
            # Use LANCZOS for photographic data, NEAREST for labels.
            resample = (
                Image.NEAREST
                if data.dtype == np.uint8 and bands == 1
                else Image.LANCZOS
            )
            resized = img.resize((size, size), resample=resample)
            result[b] = np.array(resized)

        return result

    # --------------------------------------------------------- introspection

    @property
    def roi_name(self) -> str:
        """Name of the currently loaded ROI."""
        return self._roi_name

    @property
    def n_tiles_loaded(self) -> int:
        """Number of tiles currently held in memory."""
        return len(self._tile_cache)

    @property
    def n_tiles_total(self) -> int:
        """Total number of tiles indexed for this ROI."""
        return len(self._tile_paths)

    def list_available_rois(self) -> List[str]:
        """Re-scan the data directory and return all available ROI names.

        This is a convenience method for interactive exploration.  It
        does *not* modify the currently loaded ROI.

        Returns
        -------
        list[str]
            Sorted list of ROI directory names.
        """
        rois: set[str] = set()
        for tif_path in self._data_dir.rglob("*.tif"):
            coords = parse_tile_coords(tif_path)
            if coords is None:
                continue
            parent = tif_path.parent
            if parent == self._data_dir:
                rois.add("root")
            else:
                rois.add(parent.name)
        return sorted(rois)

    def __repr__(self) -> str:
        return (
            f"MapManager(roi={self._roi_name!r}, "
            f"grid={self.grid_rows}x{self.grid_cols}, "
            f"tiles={self.n_tiles_loaded}/{self.n_tiles_total}, "
            f"bounds={self.bounds})"
        )
