"""
Drone state, configuration, and movement logic.

The :class:`Drone` is the agent's physical embodiment in the simulator.
It tracks position, enforces altitude and speed limits, and applies
displacement commands relative to its current location.

Design decisions
----------------
* **Relative commands** -- following the FlySearch benchmark convention,
  the drone accepts ``(dx, dy, dz)`` displacements in metres.  ``dx``
  is eastward, ``dy`` is northward, ``dz`` is upward.
* **Instantaneous movement** -- we skip aerodynamic physics (drag,
  inertia, settling time) because the focus is on VLM evaluation, not
  flight dynamics.  Each step teleports the drone to its new position.
* **Boundary clamping** -- the drone is silently clamped to the map
  bounds and altitude limits.  No exceptions are raised; instead, the
  actual displacement is returned so the caller can detect clipping.
* **No heading** -- the camera is always nadir (straight down) and
  north-aligned, so heading is irrelevant for now.  We keep a heading
  field for future extensibility.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DroneConfig:
    """Immutable physical limits for the drone.

    All values are in SI units (metres, metres per second).

    Attributes
    ----------
    z_min : float
        Minimum allowed altitude above ground (metres).  Must be > 0
        to prevent ground collisions.
    z_max : float
        Maximum allowed altitude (metres).
    max_step_distance : float
        Maximum displacement magnitude per step (metres).  Steps
        requesting a larger displacement are scaled down to this length.
        Set to ``inf`` to disable.
    default_altitude : float
        Altitude at which the drone spawns when no explicit z is given.
    """

    z_min: float = 10.0
    z_max: float = 500.0
    max_step_distance: float = float("inf")
    default_altitude: float = 100.0

    def __post_init__(self) -> None:
        if self.z_min <= 0:
            raise ValueError(f"z_min must be positive, got {self.z_min}")
        if self.z_max <= self.z_min:
            raise ValueError(
                f"z_max ({self.z_max}) must be greater than z_min ({self.z_min})"
            )
        if self.max_step_distance <= 0:
            raise ValueError(
                f"max_step_distance must be positive, got {self.max_step_distance}"
            )
        if not (self.z_min <= self.default_altitude <= self.z_max):
            raise ValueError(
                f"default_altitude ({self.default_altitude}) must be in "
                f"[{self.z_min}, {self.z_max}]"
            )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DroneState:
    """Mutable snapshot of the drone's kinematic state.

    Coordinate system
    -----------------
    * ``x`` -- easting in EPSG:2154 metres (increases eastward).
    * ``y`` -- northing in EPSG:2154 metres (increases northward).
    * ``z`` -- altitude above ground in metres (increases upward).
    * ``heading`` -- compass bearing in degrees clockwise from north
      (0 = north, 90 = east).  Currently unused but reserved.

    Attributes
    ----------
    x, y, z : float
        3-D position.
    heading : float
        Heading in degrees [0, 360).
    step_count : int
        Number of movement steps applied since the last reset.
    total_distance : float
        Cumulative horizontal distance travelled (metres).
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 100.0
    heading: float = 0.0
    step_count: int = 0
    total_distance: float = 0.0

    @property
    def position(self) -> Tuple[float, float, float]:
        """Current position as ``(x, y, z)``."""
        return (self.x, self.y, self.z)

    @property
    def position_2d(self) -> Tuple[float, float]:
        """Horizontal position as ``(x, y)``."""
        return (self.x, self.y)

    def copy(self) -> "DroneState":
        """Return an independent copy of this state."""
        return DroneState(
            x=self.x,
            y=self.y,
            z=self.z,
            heading=self.heading,
            step_count=self.step_count,
            total_distance=self.total_distance,
        )


# ---------------------------------------------------------------------------
# Displacement result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MoveResult:
    """Outcome of a single displacement command.

    This allows the caller to detect whether the requested displacement
    was clipped by boundaries or step-size limits.

    Attributes
    ----------
    dx_requested, dy_requested, dz_requested : float
        The raw displacement requested by the agent.
    dx_actual, dy_actual, dz_actual : float
        The displacement actually applied after clamping.
    was_clipped : bool
        ``True`` if any component was modified by boundary enforcement.
    """

    dx_requested: float
    dy_requested: float
    dz_requested: float
    dx_actual: float
    dy_actual: float
    dz_actual: float
    was_clipped: bool


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------


class Drone:
    """Simulated quadrotor drone with 3-DoF position control.

    The drone starts at a given position and accepts relative displacement
    commands.  Position is clamped to the configured altitude range and
    (optionally) to a horizontal bounding box representing the map extent.

    Parameters
    ----------
    config : DroneConfig or None
        Physical limits.  ``None`` uses the defaults.
    x_bounds : tuple[float, float] or None
        ``(x_min, x_max)`` horizontal limits in world coordinates.
        ``None`` means unbounded horizontally.
    y_bounds : tuple[float, float] or None
        ``(y_min, y_max)`` vertical limits in world coordinates.
        ``None`` means unbounded.

    Examples
    --------
    >>> drone = Drone()
    >>> drone.reset(x=1000.0, y=6500.0, z=80.0)
    >>> result = drone.move(dx=10.0, dy=-5.0, dz=20.0)
    >>> drone.state.x
    1010.0
    """

    def __init__(
        self,
        config: Optional[DroneConfig] = None,
        x_bounds: Optional[Tuple[float, float]] = None,
        y_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self._config = config or DroneConfig()
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds
        self._state = DroneState(z=self._config.default_altitude)

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        heading: float = 0.0,
    ) -> DroneState:
        """Place the drone at a specific position and clear counters.

        Parameters
        ----------
        x, y : float
            Horizontal position in world coordinates.
        z : float or None
            Altitude.  ``None`` uses ``config.default_altitude``.
        heading : float
            Initial heading in degrees (default 0 = north).

        Returns
        -------
        DroneState
            The freshly initialised state (a copy).
        """
        z_val = z if z is not None else self._config.default_altitude
        z_val = self._clamp_z(z_val)

        self._state = DroneState(
            x=x,
            y=y,
            z=z_val,
            heading=heading % 360.0,
            step_count=0,
            total_distance=0.0,
        )

        logger.info(
            "Drone reset at (%.1f, %.1f, %.1f)  heading=%.1f°",
            x,
            y,
            z_val,
            heading,
        )
        return self._state.copy()

    # ------------------------------------------------------------------ move

    def move(self, dx: float, dy: float, dz: float = 0.0) -> MoveResult:
        """Apply a relative displacement to the drone.

        Steps larger than ``config.max_step_distance`` are proportionally
        scaled down.  The final position is clamped to altitude limits
        and horizontal bounds.

        Parameters
        ----------
        dx : float
            Eastward displacement (metres).
        dy : float
            Northward displacement (metres).
        dz : float
            Upward displacement (metres, default 0).

        Returns
        -------
        MoveResult
            Struct describing the requested vs. actual displacement.
        """
        # Preserve the original request before any clamping.
        dx_orig, dy_orig, dz_orig = dx, dy, dz
        clipped = False

        # --- Enforce maximum step distance ---
        step_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if step_dist > self._config.max_step_distance:
            scale = self._config.max_step_distance / step_dist
            dx *= scale
            dy *= scale
            dz *= scale
            clipped = True
            logger.debug(
                "Step scaled from %.1f m to %.1f m (max=%.1f)",
                step_dist,
                self._config.max_step_distance,
                self._config.max_step_distance,
            )

        # --- Compute candidate position ---
        new_x = self._state.x + dx
        new_y = self._state.y + dy
        new_z = self._state.z + dz

        # --- Clamp altitude ---
        clamped_z = self._clamp_z(new_z)
        if clamped_z != new_z:
            clipped = True

        # --- Clamp horizontal position ---
        clamped_x = new_x
        clamped_y = new_y

        if self._x_bounds is not None:
            clamped_x = max(self._x_bounds[0], min(self._x_bounds[1], new_x))
            if clamped_x != new_x:
                clipped = True

        if self._y_bounds is not None:
            clamped_y = max(self._y_bounds[0], min(self._y_bounds[1], new_y))
            if clamped_y != new_y:
                clipped = True

        # --- Apply ---
        actual_dx = clamped_x - self._state.x
        actual_dy = clamped_y - self._state.y
        actual_dz = clamped_z - self._state.z
        horiz_dist = math.sqrt(actual_dx**2 + actual_dy**2)

        self._state.x = clamped_x
        self._state.y = clamped_y
        self._state.z = clamped_z
        self._state.step_count += 1
        self._state.total_distance += horiz_dist

        logger.debug(
            "Drone moved to (%.1f, %.1f, %.1f)  step=%d  clipped=%s",
            self._state.x,
            self._state.y,
            self._state.z,
            self._state.step_count,
            clipped,
        )

        return MoveResult(
            dx_requested=dx_orig,
            dy_requested=dy_orig,
            dz_requested=dz_orig,
            dx_actual=actual_dx,
            dy_actual=actual_dy,
            dz_actual=actual_dz,
            was_clipped=clipped,
        )

    # ------------------------------------------------------------------ queries

    @property
    def state(self) -> DroneState:
        """Current drone state (read-only copy).

        Returns a copy to prevent accidental external mutation.
        """
        return self._state.copy()

    @property
    def config(self) -> DroneConfig:
        """Drone configuration (immutable)."""
        return self._config

    @property
    def x_bounds(self) -> Optional[Tuple[float, float]]:
        """Horizontal x-axis limits, or ``None`` if unbounded."""
        return self._x_bounds

    @property
    def y_bounds(self) -> Optional[Tuple[float, float]]:
        """Horizontal y-axis limits, or ``None`` if unbounded."""
        return self._y_bounds

    def set_bounds(
        self,
        x_bounds: Optional[Tuple[float, float]] = None,
        y_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Update the horizontal bounding box.

        This is called by the simulator after the map is loaded to
        constrain the drone to the map extent.

        Parameters
        ----------
        x_bounds : tuple[float, float] or None
            ``(x_min, x_max)`` limits.
        y_bounds : tuple[float, float] or None
            ``(y_min, y_max)`` limits.
        """
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds
        logger.debug("Drone bounds updated: x=%s y=%s", x_bounds, y_bounds)

    def is_within_bounds(self) -> bool:
        """Check whether the drone is currently within all bounds."""
        s = self._state
        if not (self._config.z_min <= s.z <= self._config.z_max):
            return False
        if self._x_bounds is not None:
            if not (self._x_bounds[0] <= s.x <= self._x_bounds[1]):
                return False
        if self._y_bounds is not None:
            if not (self._y_bounds[0] <= s.y <= self._y_bounds[1]):
                return False
        return True

    # ------------------------------------------------------------------ internal

    def _clamp_z(self, z: float) -> float:
        """Clamp altitude to the configured range."""
        return max(self._config.z_min, min(self._config.z_max, z))

    # ------------------------------------------------------------------ repr

    def __repr__(self) -> str:
        s = self._state
        return (
            f"Drone(pos=({s.x:.1f}, {s.y:.1f}, {s.z:.1f}), "
            f"steps={s.step_count}, dist={s.total_distance:.1f}m)"
        )
