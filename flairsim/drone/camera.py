"""
Camera model for the nadir-looking drone sensor.

The :class:`CameraModel` converts the drone's altitude into a ground
footprint (in metres) and produces image observations by querying the
:class:`~flairsim.map.map_manager.MapManager`.

Design decisions
----------------
* **Simple pinhole geometry** -- we model a square sensor with a fixed
  field-of-view (FOV) angle.  At altitude *z*, the ground footprint
  half-extent is ``z * tan(fov / 2)``.
* **Always nadir** -- the camera always points straight down, so there
  is no perspective distortion or off-nadir geometry.
* **Configurable output resolution** -- regardless of altitude (and
  therefore ground sampling distance), the output image is always
  resampled to ``(image_size, image_size)`` pixels.  This mirrors the
  FlySearch benchmark where the agent always receives a fixed-size image.

Coordinate conventions
----------------------
The returned image is oriented with **north at the top**, consistent
with standard cartographic convention and the FLAIR-HUB tile layout.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Immutable camera sensor parameters.

    Attributes
    ----------
    fov_deg : float
        Full field-of-view angle in degrees.  The default 90° gives
        a ground footprint equal to ``2 * altitude`` in each direction.
    image_size : int
        Side length of the square output image in pixels.  Following
        the FlySearch benchmark, the default is 500 px.
    """

    fov_deg: float = 90.0
    image_size: int = 500

    def __post_init__(self) -> None:
        if not (0.0 < self.fov_deg < 180.0):
            raise ValueError(f"fov_deg must be in (0, 180), got {self.fov_deg}")
        if self.image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {self.image_size}")

    @property
    def fov_rad(self) -> float:
        """Field-of-view in radians."""
        return math.radians(self.fov_deg)


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------


class CameraModel:
    """Nadir-looking camera that converts altitude to ground observations.

    The camera captures a square region of the map centred on the drone's
    horizontal position.  The size of that region (the *ground footprint*)
    depends on the drone's altitude and the configured FOV.

    Parameters
    ----------
    config : CameraConfig or None
        Sensor parameters.  ``None`` uses the defaults.

    Examples
    --------
    >>> cam = CameraModel()
    >>> cam.ground_half_extent(z=100.0)
    100.0
    >>> cam.ground_footprint_size(z=100.0)
    200.0
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self._config = config or CameraConfig()

    # ---------------------------------------------------------- geometry

    def ground_half_extent(self, z: float) -> float:
        """Half the side length of the ground footprint at altitude *z*.

        Parameters
        ----------
        z : float
            Altitude above ground in metres.

        Returns
        -------
        float
            Half-extent in metres: ``z * tan(fov / 2)``.
        """
        return z * math.tan(self._config.fov_rad / 2.0)

    def ground_footprint_size(self, z: float) -> float:
        """Full side length of the ground footprint at altitude *z*.

        Parameters
        ----------
        z : float
            Altitude above ground in metres.

        Returns
        -------
        float
            Total footprint side length in metres.
        """
        return 2.0 * self.ground_half_extent(z)

    def ground_resolution(self, z: float) -> float:
        """Ground sampling distance (metres per output pixel).

        Parameters
        ----------
        z : float
            Altitude above ground in metres.

        Returns
        -------
        float
            Metres per pixel in the output image.
        """
        return self.ground_footprint_size(z) / self._config.image_size

    # ---------------------------------------------------------- capture

    def capture(
        self,
        map_manager: "MapManager",  # noqa: F821 -- forward ref
        x: float,
        y: float,
        z: float,
    ) -> np.ndarray:
        """Capture a nadir image from the given drone position.

        Parameters
        ----------
        map_manager : MapManager
            The map source providing raster data.
        x, y : float
            Drone horizontal position in world coordinates.
        z : float
            Drone altitude above ground.

        Returns
        -------
        np.ndarray
            Image array with shape ``(bands, H, W)`` where
            ``H = W = config.image_size``.  The dtype matches
            the underlying map data (typically ``uint8`` for
            AERIAL_RGBI).
        """
        half_ext = self.ground_half_extent(z)

        logger.debug(
            "Camera capture: pos=(%.1f, %.1f, %.1f)  "
            "half_ext=%.1f m  footprint=%.1f m  GSD=%.3f m/px",
            x,
            y,
            z,
            half_ext,
            half_ext * 2.0,
            self.ground_resolution(z),
        )

        return map_manager.get_region(
            x_center=x,
            y_center=y,
            half_extent=half_ext,
            output_size=self._config.image_size,
        )

    # ---------------------------------------------------------- properties

    @property
    def config(self) -> CameraConfig:
        """Camera configuration (immutable)."""
        return self._config

    @property
    def image_size(self) -> int:
        """Output image side length in pixels."""
        return self._config.image_size

    # ---------------------------------------------------------- repr

    def __repr__(self) -> str:
        return (
            f"CameraModel(fov={self._config.fov_deg:.1f}°, "
            f"image_size={self._config.image_size}px)"
        )
