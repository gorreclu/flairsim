"""
Observation returned by the simulator at each step.

An :class:`Observation` bundles everything the agent needs to make its
next decision: the camera image, the drone's state, episode metadata,
and termination signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..drone.drone import DroneState


# ---------------------------------------------------------------------------
# Episode outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Final outcome of an episode (only meaningful when ``done=True``).

    Attributes
    ----------
    success : bool
        ``True`` if the agent correctly identified the target.
    reason : str
        Human-readable explanation of why the episode ended
        (e.g. "target found", "step limit reached", "agent stopped").
    steps_taken : int
        Total number of steps in the episode.
    distance_travelled : float
        Total horizontal distance in metres.
    """

    success: bool
    reason: str
    steps_taken: int
    distance_travelled: float


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Observation:
    """Simulator output returned by ``reset()`` and ``step()``.

    This is the agent's interface to the world.  It contains everything
    needed to decide the next action.

    Attributes
    ----------
    image : np.ndarray
        Camera image with shape ``(bands, H, W)`` for the *primary*
        modality.  For AERIAL_RGBI, bands are ``[R, G, B, NIR]`` in
        ``uint8``.  The image is always ``(image_size, image_size)``
        regardless of altitude.
    drone_state : DroneState
        Current position, heading, and counters.
    step : int
        Current step number (0 on ``reset()``).
    done : bool
        ``True`` if the episode has ended.
    result : EpisodeResult or None
        Set only when ``done=True``.
    ground_footprint : float
        Side length of the current camera ground footprint (metres).
    ground_resolution : float
        Ground sampling distance (metres per pixel).
    metadata : dict
        Arbitrary extra info (scenario name, target description, etc.).
    images : dict[str, np.ndarray]
        Per-modality images keyed by modality name (e.g.
        ``"AERIAL_RGBI"``, ``"DEM_ELEV"``).  Each value has shape
        ``(bands, H, W)``.  Empty when running in single-modality
        mode (in that case only ``image`` is populated).
    """

    image: np.ndarray = field(repr=False)
    drone_state: DroneState
    step: int
    done: bool = False
    result: Optional[EpisodeResult] = None
    ground_footprint: float = 0.0
    ground_resolution: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)
    images: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- helpers

    @property
    def position(self) -> Tuple[float, float, float]:
        """Shortcut to drone ``(x, y, z)``."""
        return self.drone_state.position

    @property
    def altitude(self) -> float:
        """Current altitude in metres."""
        return self.drone_state.z

    @property
    def success(self) -> bool:
        """Whether the episode ended successfully.

        Returns ``False`` if the episode is still running or ended
        without success.
        """
        if self.result is not None:
            return self.result.success
        return False

    def image_rgb(self) -> np.ndarray:
        """Return the image as ``(H, W, 3)`` uint8 RGB for display.

        Extracts the first three bands (R, G, B) and transposes to
        channel-last layout suitable for matplotlib / pygame.

        Returns
        -------
        np.ndarray
            ``uint8`` array with shape ``(H, W, 3)``.
        """
        if self.image.ndim == 3 and self.image.shape[0] >= 3:
            # (bands, H, W) -> take first 3 bands -> (H, W, 3)
            return np.transpose(self.image[:3], (1, 2, 0)).copy()
        elif self.image.ndim == 3 and self.image.shape[0] == 1:
            # Single band -> replicate to greyscale RGB.
            band = self.image[0]
            return np.stack([band, band, band], axis=-1)
        elif self.image.ndim == 2:
            # Already (H, W) -> replicate.
            return np.stack([self.image, self.image, self.image], axis=-1)
        else:
            # Fallback: return as-is.
            return self.image

    def __repr__(self) -> str:
        pos = self.drone_state.position
        return (
            f"Observation(step={self.step}, "
            f"pos=({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}), "
            f"done={self.done}, "
            f"image={self.image.shape})"
        )
