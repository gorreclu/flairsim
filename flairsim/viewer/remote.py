"""
Remote observation adapter for the viewer.

The :class:`ViewerObservation` is a lightweight data class that holds
exactly the fields the viewer needs to render a frame (image, drone
position, telemetry, episode status).  It decouples the viewer from
both the simulator's :class:`~flairsim.core.observation.Observation`
and the server's JSON response format.

Two factory methods are provided:

* :meth:`ViewerObservation.from_observation` -- wraps a local
  ``Observation`` object (used when the viewer is directly attached
  to a simulator).
* :meth:`ViewerObservation.from_server_response` -- reconstructs
  the view from a server JSON response dict (used in remote
  observe/fly modes).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from ..core.observation import Observation


# ---------------------------------------------------------------------------
# Lightweight sub-structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViewerDroneState:
    """Minimal drone state for display purposes.

    Attributes
    ----------
    x, y, z : float
        Position in world coordinates (metres).
    total_distance : float
        Cumulative distance travelled (metres).
    """

    x: float
    y: float
    z: float
    total_distance: float


@dataclass(frozen=True, slots=True)
class ViewerEpisodeResult:
    """Episode outcome for HUD display.

    Attributes
    ----------
    success : bool
        Whether the agent succeeded.
    reason : str
        Human-readable explanation.
    """

    success: bool
    reason: str


# ---------------------------------------------------------------------------
# ViewerObservation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ViewerObservation:
    """Everything the viewer needs to render one frame.

    This is intentionally a thin, framework-agnostic data class with no
    dependency on the simulator core, the server, or pygame.  It serves
    as the single interface between the data source (local or remote)
    and the rendering components (HUD, minimap, main image).

    Attributes
    ----------
    image_rgb : np.ndarray
        ``(H, W, 3)`` uint8 RGB image ready for display.
    drone_state : ViewerDroneState
        Current position and distance.
    step : int
        Current step number.
    done : bool
        Whether the episode has ended.
    ground_footprint : float
        Camera ground footprint in metres.
    ground_resolution : float
        Ground sampling distance (m/px).
    result : ViewerEpisodeResult or None
        Episode outcome (set only when ``done=True``).
    """

    image_rgb: np.ndarray = field(repr=False)
    drone_state: ViewerDroneState
    step: int
    done: bool = False
    ground_footprint: float = 0.0
    ground_resolution: float = 0.0
    result: Optional[ViewerEpisodeResult] = None

    # ---------------------------------------------------------------- factories

    @classmethod
    def from_observation(cls, obs: Observation) -> ViewerObservation:
        """Create from a local :class:`Observation`.

        Parameters
        ----------
        obs : Observation
            Observation from the simulator.

        Returns
        -------
        ViewerObservation
        """
        result = None
        if obs.result is not None:
            result = ViewerEpisodeResult(
                success=obs.result.success,
                reason=obs.result.reason,
            )

        return cls(
            image_rgb=obs.image_rgb(),
            drone_state=ViewerDroneState(
                x=obs.drone_state.x,
                y=obs.drone_state.y,
                z=obs.drone_state.z,
                total_distance=obs.drone_state.total_distance,
            ),
            step=obs.step,
            done=obs.done,
            ground_footprint=obs.ground_footprint,
            ground_resolution=obs.ground_resolution,
            result=result,
        )

    @classmethod
    def from_server_response(cls, data: Dict[str, Any]) -> ViewerObservation:
        """Create from a server JSON response dict.

        Decodes the base64-encoded PNG image and extracts telemetry
        fields from the ``ObservationResponse`` structure.

        Parameters
        ----------
        data : dict
            Parsed JSON from ``POST /reset``, ``POST /step``, or an
            SSE ``observation`` event.

        Returns
        -------
        ViewerObservation
        """
        # Decode base64 PNG to numpy RGB array.
        png_bytes = base64.b64decode(data["image_base64"])
        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        image_rgb = np.asarray(pil_img, dtype=np.uint8)

        ds = data["drone_state"]
        drone_state = ViewerDroneState(
            x=ds["x"],
            y=ds["y"],
            z=ds["z"],
            total_distance=ds["total_distance"],
        )

        result = None
        if data.get("result") is not None:
            result = ViewerEpisodeResult(
                success=data["result"]["success"],
                reason=data["result"]["reason"],
            )

        return cls(
            image_rgb=image_rgb,
            drone_state=drone_state,
            step=data["step"],
            done=data["done"],
            ground_footprint=data["ground_footprint"],
            ground_resolution=data["ground_resolution"],
            result=result,
        )
