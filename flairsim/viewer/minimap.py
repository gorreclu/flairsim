"""
Minimap inset showing the drone's position on the full ROI grid.

The :class:`Minimap` draws a small overview of the tile grid in a corner
of the viewer window, with:

* The tile grid as a coloured rectangle.
* The drone's current position as a marker.
* The camera footprint as a rectangle.
* The flight trajectory as a polyline.

The minimap provides spatial awareness -- the agent (or human operator)
can see where on the overall map they are and where they have been.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pygame

from ..map.map_manager import MapBounds


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MinimapConfig:
    """Visual parameters for the minimap inset.

    Attributes
    ----------
    size : int
        Side length of the minimap square in pixels.
    margin : int
        Margin from the viewer edge in pixels.
    position : str
        Corner placement: ``"top-left"``, ``"top-right"``,
        ``"bottom-left"``, or ``"bottom-right"``.
    bg_color : tuple[int, int, int, int]
        RGBA background colour.
    grid_color : tuple[int, int, int]
        Colour for the tile grid outline.
    drone_color : tuple[int, int, int]
        Colour for the drone position marker.
    footprint_color : tuple[int, int, int, int]
        RGBA colour for the camera footprint rectangle.
    trail_color : tuple[int, int, int]
        Colour for the flight trajectory line.
    target_color : tuple[int, int, int]
        Colour for the scenario target marker.
    drone_radius : int
        Radius of the drone marker dot in pixels.
    target_radius : int
        Radius of the target marker dot in pixels.
    trail_width : int
        Width of the trajectory line in pixels.
    """

    size: int = 180
    margin: int = 8
    position: str = "bottom-right"
    bg_color: Tuple[int, int, int, int] = (20, 20, 20, 200)
    grid_color: Tuple[int, int, int] = (80, 80, 80)
    drone_color: Tuple[int, int, int] = (255, 50, 50)
    footprint_color: Tuple[int, int, int, int] = (255, 255, 0, 80)
    trail_color: Tuple[int, int, int] = (100, 200, 255)
    target_color: Tuple[int, int, int] = (50, 255, 50)
    drone_radius: int = 4
    target_radius: int = 5
    trail_width: int = 1


# ---------------------------------------------------------------------------
# Minimap
# ---------------------------------------------------------------------------


class Minimap:
    """Inset map showing the drone's position on the full tile grid.

    Parameters
    ----------
    map_bounds : MapBounds
        Spatial extent of the loaded map in world coordinates.
    config : MinimapConfig or None
        Visual configuration.  ``None`` uses defaults.
    """

    def __init__(
        self,
        map_bounds: MapBounds,
        config: Optional[MinimapConfig] = None,
    ) -> None:
        self._bounds = map_bounds
        self._config = config or MinimapConfig()
        self._trail: List[Tuple[float, float]] = []
        self._target: Optional[Tuple[float, float]] = None

    # ---------------------------------------------------------------- trail

    def reset_trail(self) -> None:
        """Clear the trajectory trail (call on episode reset)."""
        self._trail.clear()

    def add_trail_point(self, x: float, y: float) -> None:
        """Add a point to the trajectory trail.

        Parameters
        ----------
        x, y : float
            World coordinates.
        """
        self._trail.append((x, y))

    # ---------------------------------------------------------------- target

    def set_target(self, x: float, y: float) -> None:
        """Set the scenario target position for display.

        Parameters
        ----------
        x, y : float
            Target world coordinates.
        """
        self._target = (x, y)

    def clear_target(self) -> None:
        """Remove the target marker."""
        self._target = None

    # ---------------------------------------------------------------- render

    def render(
        self,
        surface: pygame.Surface,
        drone_x: float,
        drone_y: float,
        footprint_size: float = 0.0,
    ) -> None:
        """Draw the minimap onto the given surface.

        Parameters
        ----------
        surface : pygame.Surface
            Target display surface.
        drone_x, drone_y : float
            Current drone position in world coordinates.
        footprint_size : float
            Side length of the camera footprint in metres.
        """
        cfg = self._config
        size = cfg.size

        # Create the minimap surface with alpha.
        mm_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        mm_surface.fill(cfg.bg_color)

        # Draw the grid outline.
        pygame.draw.rect(mm_surface, cfg.grid_color, (0, 0, size, size), 1)

        # Draw camera footprint.
        if footprint_size > 0:
            fp_px, fp_py = self._world_to_minimap(drone_x, drone_y)
            # Convert footprint size to minimap pixels.
            fp_half = self._scale_to_minimap(footprint_size / 2.0)
            fp_rect = pygame.Rect(
                int(fp_px - fp_half),
                int(fp_py - fp_half),
                int(fp_half * 2),
                int(fp_half * 2),
            )
            fp_surf = pygame.Surface((fp_rect.width, fp_rect.height), pygame.SRCALPHA)
            fp_surf.fill(cfg.footprint_color)
            mm_surface.blit(fp_surf, fp_rect.topleft)

        # Draw trajectory trail.
        if len(self._trail) >= 2:
            trail_points = [self._world_to_minimap(tx, ty) for tx, ty in self._trail]
            # Convert to integer tuples.
            trail_int = [(int(p[0]), int(p[1])) for p in trail_points]
            pygame.draw.lines(
                mm_surface, cfg.trail_color, False, trail_int, cfg.trail_width
            )

        # Draw drone marker.
        dx_px, dy_px = self._world_to_minimap(drone_x, drone_y)
        pygame.draw.circle(
            mm_surface,
            cfg.drone_color,
            (int(dx_px), int(dy_px)),
            cfg.drone_radius,
        )

        # Draw target marker (crosshair).
        if self._target is not None:
            tx_px, ty_px = self._world_to_minimap(*self._target)
            txi, tyi = int(tx_px), int(ty_px)
            r = cfg.target_radius
            # Draw a diamond/crosshair marker.
            pygame.draw.circle(mm_surface, cfg.target_color, (txi, tyi), r, 1)
            pygame.draw.line(
                mm_surface,
                cfg.target_color,
                (txi - r - 2, tyi),
                (txi + r + 2, tyi),
                1,
            )
            pygame.draw.line(
                mm_surface,
                cfg.target_color,
                (txi, tyi - r - 2),
                (txi, tyi + r + 2),
                1,
            )

        # Position the minimap on the main surface.
        sw, sh = surface.get_size()
        if cfg.position == "top-left":
            pos = (cfg.margin, cfg.margin)
        elif cfg.position == "top-right":
            pos = (sw - size - cfg.margin, cfg.margin)
        elif cfg.position == "bottom-left":
            pos = (cfg.margin, sh - size - cfg.margin)
        elif cfg.position == "bottom-right":
            pos = (sw - size - cfg.margin, sh - size - cfg.margin)
        else:
            pos = (sw - size - cfg.margin, sh - size - cfg.margin)

        surface.blit(mm_surface, pos)

    # ---------------------------------------------------------------- internal

    def _world_to_minimap(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates to minimap pixel coordinates.

        The minimap maps the full map bounds to a ``(size, size)`` square.
        North is at the top (y increases upward in world, but downward
        in pixel space).
        """
        b = self._bounds
        size = self._config.size

        # Normalise to [0, 1] within the map bounds.
        if b.width > 0:
            nx = (x - b.x_min) / b.width
        else:
            nx = 0.5

        if b.height > 0:
            # Flip y: world y increases northward, pixel y increases downward.
            ny = 1.0 - (y - b.y_min) / b.height
        else:
            ny = 0.5

        # Apply a small margin inside the minimap.
        margin_frac = 0.05
        usable = 1.0 - 2 * margin_frac
        px = (margin_frac + nx * usable) * size
        py = (margin_frac + ny * usable) * size

        return (px, py)

    def _scale_to_minimap(self, metres: float) -> float:
        """Convert a distance in metres to minimap pixels."""
        b = self._bounds
        map_extent = max(b.width, b.height)
        if map_extent <= 0:
            return 0.0
        usable = (1.0 - 0.1) * self._config.size  # Match margin_frac=0.05
        return metres / map_extent * usable

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        return f"Minimap(size={self._config.size}px, pos={self._config.position!r})"
