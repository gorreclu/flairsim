"""
Heads-up display overlay for the drone viewer.

The :class:`HUD` renders flight telemetry information (altitude, position,
step count, ground footprint, GSD, etc.) as a semi-transparent overlay on
the main viewer window.

Design
------
All rendering is done directly onto a pygame surface.  The HUD is drawn
as a translucent panel in a corner of the screen with monospaced text
lines.  Colours and layout are configurable.

The HUD accepts a :class:`~flairsim.viewer.remote.ViewerObservation`,
which is a lightweight data class that works with both local simulator
observations and remote server responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import pygame

from .remote import ViewerObservation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HUDConfig:
    """Visual parameters for the HUD overlay.

    Attributes
    ----------
    font_size : int
        Text size in pixels.
    font_name : str or None
        Pygame font name (``None`` uses the default monospace font).
    text_color : tuple[int, int, int]
        RGB colour for text.
    bg_color : tuple[int, int, int, int]
        RGBA colour for the background panel.
    padding : int
        Padding inside the panel in pixels.
    line_spacing : int
        Vertical spacing between text lines in pixels.
    position : str
        Corner placement: ``"top-left"``, ``"top-right"``,
        ``"bottom-left"``, or ``"bottom-right"``.
    """

    font_size: int = 16
    font_name: Optional[str] = None
    text_color: Tuple[int, int, int] = (255, 255, 255)
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 160)
    padding: int = 10
    line_spacing: int = 4
    position: str = "top-left"


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------


class HUD:
    """Flight telemetry overlay.

    The HUD displays key information from the current
    :class:`~flairsim.viewer.remote.ViewerObservation` as text lines
    on a semi-transparent panel.

    Parameters
    ----------
    config : HUDConfig or None
        Visual configuration.  ``None`` uses defaults.
    """

    def __init__(self, config: Optional[HUDConfig] = None) -> None:
        self._config = config or HUDConfig()
        self._font: Optional[pygame.font.Font] = None
        self._extra_lines: List[str] = []

    # ---------------------------------------------------------------- init

    def init(self) -> None:
        """Initialise the pygame font (must be called after ``pygame.init()``)."""
        if self._config.font_name:
            self._font = pygame.font.SysFont(
                self._config.font_name, self._config.font_size
            )
        else:
            # Default monospace font.
            self._font = pygame.font.SysFont("monospace", self._config.font_size)

    # ---------------------------------------------------------------- render

    def render(
        self,
        surface: pygame.Surface,
        obs: ViewerObservation,
        fps: float = 0.0,
    ) -> None:
        """Draw the HUD overlay onto the given surface.

        Parameters
        ----------
        surface : pygame.Surface
            The target surface (typically the main display).
        obs : ViewerObservation
            Current observation to extract telemetry from.
        fps : float
            Current frame rate for display (0 = not shown).
        """
        if self._font is None:
            self.init()
        assert self._font is not None

        lines = self._build_lines(obs, fps)
        self._draw_panel(surface, lines)

    def set_extra_lines(self, lines: Sequence[str]) -> None:
        """Set additional lines to display below the standard telemetry.

        Parameters
        ----------
        lines : sequence of str
            Extra text lines (e.g. scenario info, agent status).
        """
        self._extra_lines = list(lines)

    # ---------------------------------------------------------------- internal

    def _build_lines(self, obs: ViewerObservation, fps: float) -> List[str]:
        """Assemble the text lines to display."""
        s = obs.drone_state
        lines = [
            f"Step: {obs.step}",
            f"Pos:  ({s.x:.0f}, {s.y:.0f})",
            f"Alt:  {s.z:.0f} m",
            f"Foot: {obs.ground_footprint:.0f} m",
            f"GSD:  {obs.ground_resolution:.3f} m/px",
            f"Dist: {s.total_distance:.0f} m",
        ]

        if fps > 0:
            lines.append(f"FPS:  {fps:.0f}")

        if obs.done and obs.result is not None:
            lines.append("")
            lines.append(f"DONE: {obs.result.reason}")
            if obs.result.success:
                lines.append(">> SUCCESS <<")

        if self._extra_lines:
            lines.append("")
            lines.extend(self._extra_lines)

        return lines

    def _draw_panel(self, surface: pygame.Surface, lines: List[str]) -> None:
        """Render the background panel and text lines."""
        assert self._font is not None
        cfg = self._config

        # Measure text extents.
        rendered = [self._font.render(line, True, cfg.text_color) for line in lines]
        max_width = max((r.get_width() for r in rendered), default=0)
        line_height = self._font.get_height()
        total_height = len(lines) * line_height + (len(lines) - 1) * cfg.line_spacing

        panel_w = max_width + 2 * cfg.padding
        panel_h = total_height + 2 * cfg.padding

        # Create translucent panel.
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(cfg.bg_color)

        # Position the panel.
        sw, sh = surface.get_size()
        if cfg.position == "top-left":
            panel_x, panel_y = 8, 8
        elif cfg.position == "top-right":
            panel_x, panel_y = sw - panel_w - 8, 8
        elif cfg.position == "bottom-left":
            panel_x, panel_y = 8, sh - panel_h - 8
        elif cfg.position == "bottom-right":
            panel_x, panel_y = sw - panel_w - 8, sh - panel_h - 8
        else:
            panel_x, panel_y = 8, 8

        surface.blit(panel, (panel_x, panel_y))

        # Draw text lines.
        y_cursor = panel_y + cfg.padding
        for rendered_line in rendered:
            surface.blit(rendered_line, (panel_x + cfg.padding, y_cursor))
            y_cursor += line_height + cfg.line_spacing

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        return f"HUD(font_size={self._config.font_size}, pos={self._config.position!r})"
