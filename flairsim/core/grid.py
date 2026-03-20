"""
Grid overlay for Set-of-Mark / Scaffold prompting.

Draws an NxN grid with alphanumeric cell labels on top of a camera
image.  This is designed to help Vision-Language Models (VLMs) refer
to specific regions of an observation using short, unambiguous
identifiers (e.g. ``"A1"``, ``"B3"``).

Background
----------
- **Set-of-Mark prompting** (Yang et al., arXiv:2310.11441) overlays
  visual markers on images so VLMs can reference spatial locations
  by label rather than by coordinates.
- **Scaffold prompting** (Lei et al., arXiv:2402.12058) uses
  alphanumeric grid cells specifically (A1, B2, ...) to ground VLM
  spatial reasoning.

Usage
-----
::

    from flairsim.core.grid import GridOverlay

    overlay = GridOverlay(n=4)          # 4x4 grid
    annotated = overlay.draw(image)     # (H, W, 3) uint8 -> (H, W, 3) uint8

The overlay can also be used to convert cell labels to/from pixel
coordinates::

    overlay.cell_center("B3")           # -> (px_x, px_y)
    overlay.cell_from_pixel(120, 300)   # -> "B3"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Maximum supported grid size (26 rows = A..Z).
MAX_GRID_SIZE = 26

# Row labels (letters) and column labels (digits).
_ROW_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Visual parameters for the grid overlay.

    Attributes
    ----------
    line_color : tuple[int, int, int]
        RGB colour of the grid lines (default: white).
    line_alpha : float
        Opacity of the grid lines, 0.0 (transparent) to 1.0 (opaque).
    line_width : int
        Width of the grid lines in pixels.
    label_color : tuple[int, int, int]
        RGB colour of the cell labels.
    label_bg_color : tuple[int, int, int] or None
        Background colour for label boxes.  ``None`` = no background.
    label_bg_alpha : float
        Opacity of the label background box.
    font_scale : float
        Relative font size (1.0 = auto-sized to cell).
    """

    line_color: Tuple[int, int, int] = (255, 255, 255)
    line_alpha: float = 0.6
    line_width: int = 2
    label_color: Tuple[int, int, int] = (255, 255, 255)
    label_bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0)
    label_bg_alpha: float = 0.5
    font_scale: float = 1.0


# ---------------------------------------------------------------------------
# GridOverlay
# ---------------------------------------------------------------------------


class GridOverlay:
    """NxN grid overlay with alphanumeric cell labels.

    The grid divides an image into ``n`` rows (labelled A, B, C, ...)
    and ``n`` columns (labelled 1, 2, 3, ...).  Each cell is identified
    by its row letter and column number, e.g. ``"A1"`` (top-left),
    ``"B3"`` (second row, third column).

    Parameters
    ----------
    n : int
        Grid size (number of rows = number of columns).  Must be in
        ``[1, 26]``.
    config : GridConfig or None
        Visual parameters.  ``None`` uses defaults.

    Raises
    ------
    ValueError
        If *n* is outside ``[1, 26]``.
    """

    def __init__(self, n: int, config: Optional[GridConfig] = None) -> None:
        if not 1 <= n <= MAX_GRID_SIZE:
            raise ValueError(
                f"Grid size must be between 1 and {MAX_GRID_SIZE}, got {n}."
            )
        self._n = n
        self._config = config or GridConfig()

    # ---------------------------------------------------------------- properties

    @property
    def n(self) -> int:
        """Grid size (number of rows and columns)."""
        return self._n

    @property
    def config(self) -> GridConfig:
        """Visual configuration."""
        return self._config

    @property
    def cell_labels(self) -> list[str]:
        """All cell labels in row-major order (e.g. ``['A1', 'A2', 'B1', 'B2']``)."""
        labels = []
        for r in range(self._n):
            for c in range(self._n):
                labels.append(f"{_ROW_LABELS[r]}{c + 1}")
        return labels

    # ---------------------------------------------------------------- geometry

    def cell_bounds(
        self, label: str, img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        """Pixel bounding box of a cell.

        Parameters
        ----------
        label : str
            Cell label (e.g. ``"A1"``).
        img_width, img_height : int
            Image dimensions.

        Returns
        -------
        tuple[int, int, int, int]
            ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.

        Raises
        ------
        ValueError
            If the label is invalid.
        """
        row, col = self._parse_label(label)
        cell_w = img_width / self._n
        cell_h = img_height / self._n
        x_min = int(col * cell_w)
        y_min = int(row * cell_h)
        x_max = int((col + 1) * cell_w)
        y_max = int((row + 1) * cell_h)
        return x_min, y_min, x_max, y_max

    def cell_center(
        self, label: str, img_width: int, img_height: int
    ) -> Tuple[int, int]:
        """Pixel centre of a cell.

        Parameters
        ----------
        label : str
            Cell label (e.g. ``"B3"``).
        img_width, img_height : int
            Image dimensions.

        Returns
        -------
        tuple[int, int]
            ``(px_x, px_y)`` pixel coordinates of the cell centre.
        """
        x_min, y_min, x_max, y_max = self.cell_bounds(label, img_width, img_height)
        return (x_min + x_max) // 2, (y_min + y_max) // 2

    def cell_from_pixel(
        self, px_x: int, px_y: int, img_width: int, img_height: int
    ) -> str:
        """Determine which cell a pixel coordinate falls in.

        Parameters
        ----------
        px_x, px_y : int
            Pixel coordinates.
        img_width, img_height : int
            Image dimensions.

        Returns
        -------
        str
            Cell label (e.g. ``"A1"``).

        Raises
        ------
        ValueError
            If the pixel is outside the image.
        """
        if not (0 <= px_x < img_width and 0 <= px_y < img_height):
            raise ValueError(
                f"Pixel ({px_x}, {px_y}) is outside image ({img_width}x{img_height})."
            )
        cell_w = img_width / self._n
        cell_h = img_height / self._n
        col = min(int(px_x / cell_w), self._n - 1)
        row = min(int(px_y / cell_h), self._n - 1)
        return f"{_ROW_LABELS[row]}{col + 1}"

    # ---------------------------------------------------------------- drawing

    def draw(self, image: np.ndarray) -> np.ndarray:
        """Draw the grid overlay on an image.

        Parameters
        ----------
        image : np.ndarray
            Input image with shape ``(H, W, 3)`` and dtype ``uint8``.

        Returns
        -------
        np.ndarray
            A new ``(H, W, 3)`` uint8 array with the grid drawn on top.
            The original image is not modified.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got shape {image.shape}.")

        result = image.copy()
        h, w = result.shape[:2]
        cfg = self._config

        cell_w = w / self._n
        cell_h = h / self._n

        # Draw grid lines with alpha blending.
        line_overlay = result.copy()

        lw = cfg.line_width

        # Vertical lines.
        for i in range(1, self._n):
            x = int(round(i * cell_w))
            x0 = max(0, x - lw // 2)
            x1 = min(w, x0 + lw)
            line_overlay[:, x0:x1] = cfg.line_color

        # Horizontal lines.
        for i in range(1, self._n):
            y = int(round(i * cell_h))
            y0 = max(0, y - lw // 2)
            y1 = min(h, y0 + lw)
            line_overlay[y0:y1, :] = cfg.line_color

        # Alpha-blend lines.
        alpha = cfg.line_alpha
        result = _blend(result, line_overlay, alpha)

        # Draw cell labels.
        result = self._draw_labels(result, w, h, cell_w, cell_h)

        return result

    def draw_on_surface(self, surface: "pygame.Surface") -> None:
        """Draw the grid overlay directly on a pygame Surface.

        This avoids a numpy round-trip when the viewer already has a
        pygame surface.

        Parameters
        ----------
        surface : pygame.Surface
            The surface to draw on (modified in-place).
        """
        import pygame

        w, h = surface.get_size()
        cfg = self._config
        cell_w = w / self._n
        cell_h = h / self._n
        lw = cfg.line_width

        # Create a transparent overlay surface.
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)

        line_color_a = (*cfg.line_color, int(cfg.line_alpha * 255))

        # Vertical lines.
        for i in range(1, self._n):
            x = int(round(i * cell_w))
            pygame.draw.line(overlay, line_color_a, (x, 0), (x, h), lw)

        # Horizontal lines.
        for i in range(1, self._n):
            y = int(round(i * cell_h))
            pygame.draw.line(overlay, line_color_a, (0, y), (w, y), lw)

        # Draw labels.
        font_size = self._auto_font_size(cell_w, cell_h)
        try:
            font = pygame.font.SysFont("monospace", font_size, bold=True)
        except Exception:
            font = pygame.font.Font(None, font_size)

        for r in range(self._n):
            for c in range(self._n):
                label = f"{_ROW_LABELS[r]}{c + 1}"
                text_surf = font.render(label, True, cfg.label_color)
                tw, th = text_surf.get_size()

                # Position: top-left of cell with small margin.
                margin = max(2, int(cell_w * 0.05))
                lx = int(c * cell_w) + margin
                ly = int(r * cell_h) + margin

                # Label background.
                if cfg.label_bg_color is not None:
                    pad = 2
                    bg_rect = pygame.Rect(
                        lx - pad, ly - pad, tw + 2 * pad, th + 2 * pad
                    )
                    bg_color_a = (*cfg.label_bg_color, int(cfg.label_bg_alpha * 255))
                    pygame.draw.rect(overlay, bg_color_a, bg_rect)

                overlay.blit(text_surf, (lx, ly))

        surface.blit(overlay, (0, 0))

    # ---------------------------------------------------------------- internal

    def _draw_labels(
        self,
        image: np.ndarray,
        w: int,
        h: int,
        cell_w: float,
        cell_h: float,
    ) -> np.ndarray:
        """Draw alphanumeric labels on each cell using PIL."""
        from PIL import Image, ImageDraw, ImageFont

        cfg = self._config
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img, "RGBA")

        font_size = self._auto_font_size(cell_w, cell_h)
        try:
            font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("Courier New Bold.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        for r in range(self._n):
            for c in range(self._n):
                label = f"{_ROW_LABELS[r]}{c + 1}"

                # Measure text.
                bbox = font.getbbox(label)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                # Position: top-left of cell with small margin.
                margin = max(2, int(cell_w * 0.05))
                lx = int(c * cell_w) + margin
                ly = int(r * cell_h) + margin

                # Label background.
                if cfg.label_bg_color is not None:
                    pad = 2
                    bg_alpha = int(cfg.label_bg_alpha * 255)
                    bg = (*cfg.label_bg_color, bg_alpha)
                    draw.rectangle(
                        [lx - pad, ly - pad, lx + tw + pad, ly + th + pad],
                        fill=bg,
                    )

                draw.text((lx, ly), label, fill=cfg.label_color, font=font)

        return np.asarray(pil_img)

    def _auto_font_size(self, cell_w: float, cell_h: float) -> int:
        """Compute font size that fits well within a cell."""
        # Target: label occupies ~30% of cell width.
        # A 2-char label at font_size N is roughly N * 1.2 pixels wide.
        target_w = cell_w * 0.30 * self._config.font_scale
        # For n>=10, labels are 3 chars (e.g. "A10"), need more space.
        n_chars = 2 if self._n < 10 else 3
        size = int(target_w / (n_chars * 0.6))
        return max(8, min(size, int(cell_h * 0.4)))

    def _parse_label(self, label: str) -> Tuple[int, int]:
        """Parse a cell label into (row, col) indices (0-based).

        Parameters
        ----------
        label : str
            e.g. ``"A1"``, ``"C12"``.

        Returns
        -------
        tuple[int, int]
            ``(row, col)`` 0-based indices.

        Raises
        ------
        ValueError
            If the label format is invalid or out of range.
        """
        if len(label) < 2:
            raise ValueError(f"Invalid cell label: {label!r}")

        row_char = label[0].upper()
        col_str = label[1:]

        if row_char not in _ROW_LABELS:
            raise ValueError(f"Invalid row letter in label: {label!r}")

        row = _ROW_LABELS.index(row_char)
        if row >= self._n:
            raise ValueError(
                f"Row '{row_char}' is out of range for a {self._n}x{self._n} grid."
            )

        try:
            col = int(col_str) - 1  # 1-based to 0-based.
        except ValueError:
            raise ValueError(f"Invalid column number in label: {label!r}")

        if not 0 <= col < self._n:
            raise ValueError(
                f"Column {col + 1} is out of range for a {self._n}x{self._n} grid."
            )

        return row, col

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        return f"GridOverlay(n={self._n})"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend *overlay* onto *base* (both uint8, same shape)."""
    return (
        base.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha
    ).astype(np.uint8)
