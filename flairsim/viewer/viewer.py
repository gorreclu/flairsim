"""
Real-time pygame viewer for FlairSim.

The :class:`FlairViewer` opens a desktop window showing the drone's
camera view, with a HUD overlay and minimap inset.  It supports three
connection modes:

* **Local manual flight** -- the viewer directly controls a local
  ``FlairSimulator`` instance with keyboard controls.
* **Remote observe** -- the viewer connects to a running FlairSim
  server via SSE and passively displays what an agent does.
* **Remote fly** -- the viewer connects to a running FlairSim server
  via SSE *and* sends ``POST /step`` / ``POST /reset`` on keypresses.

Keyboard controls (manual / remote fly modes)
----------------------------------------------
+-----------+----------------------------+
| Key       | Action                     |
+===========+============================+
| Z / Up    | Move north (+dy)           |
| S / Down  | Move south (-dy)           |
| Q / Left  | Move west  (-dx)           |
| D / Right | Move east  (+dx)           |
| A         | Descend    (-dz)           |
| E         | Ascend     (+dz)           |
| +/-       | Adjust move step size      |
| Space     | Declare FOUND              |
| Escape    | Quit / stop episode        |
| R         | Reset episode              |
| H         | Toggle HUD                 |
| M         | Toggle minimap             |
+-----------+----------------------------+
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame

from ..core.action import Action, ActionType
from ..map.map_manager import MapBounds
from .hud import HUD, HUDConfig
from .minimap import Minimap, MinimapConfig
from .remote import ViewerObservation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViewerConfig:
    """Configuration for the viewer window.

    Attributes
    ----------
    window_width : int
        Window width in pixels.
    window_height : int
        Window height in pixels.
    title : str
        Window title.
    target_fps : int
        Target frame rate.
    move_step : float
        Default movement step size in metres per key press.
    altitude_step : float
        Altitude change per A/E press in metres.
    show_hud : bool
        Whether to show the HUD initially.
    show_minimap : bool
        Whether to show the minimap initially.
    hud_config : HUDConfig or None
        HUD visual parameters.
    minimap_config : MinimapConfig or None
        Minimap visual parameters.
    """

    window_width: int = 800
    window_height: int = 800
    title: str = "FlairSim Viewer"
    target_fps: int = 30
    move_step: float = 20.0
    altitude_step: float = 10.0
    show_hud: bool = True
    show_minimap: bool = True
    hud_config: Optional[HUDConfig] = None
    minimap_config: Optional[MinimapConfig] = None


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class FlairViewer:
    """Real-time pygame viewer for drone flight visualisation.

    The viewer can operate in three modes:

    1. **Local manual mode** -- call :meth:`run_manual` with a local
       simulator to fly with keyboard controls.
    2. **Remote observe mode** -- call :meth:`run_remote_observe` with
       a server URL to passively watch an agent fly.
    3. **Remote fly mode** -- call :meth:`run_remote_fly` with a
       server URL to fly the drone via the HTTP API.

    You can also call :meth:`show` with individual
    :class:`ViewerObservation` objects for custom agent loops.

    Parameters
    ----------
    config : ViewerConfig or None
        Window and display parameters.
    map_bounds : MapBounds or None
        Map extent for the minimap.  If ``None``, the minimap is
        disabled until :meth:`set_map_bounds` is called.
    """

    def __init__(
        self,
        config: Optional[ViewerConfig] = None,
        map_bounds: Optional[MapBounds] = None,
    ) -> None:
        self._config = config or ViewerConfig()
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._running = False
        self._show_hud = self._config.show_hud
        self._show_minimap = self._config.show_minimap
        self._move_step = self._config.move_step

        # Sub-components.
        self._hud = HUD(config=self._config.hud_config)
        self._minimap: Optional[Minimap] = None
        if map_bounds is not None:
            self._minimap = Minimap(
                map_bounds=map_bounds,
                config=self._config.minimap_config,
            )

    # ---------------------------------------------------------------- lifecycle

    def open(self) -> None:
        """Initialise pygame and open the viewer window."""
        if self._screen is not None:
            return  # Already open.

        pygame.init()
        self._screen = pygame.display.set_mode(
            (self._config.window_width, self._config.window_height)
        )
        pygame.display.set_caption(self._config.title)
        self._clock = pygame.time.Clock()
        self._hud.init()
        self._running = True

        logger.info(
            "Viewer opened: %dx%d @ %d FPS",
            self._config.window_width,
            self._config.window_height,
            self._config.target_fps,
        )

    def close(self) -> None:
        """Close the viewer window and quit pygame."""
        self._running = False
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._clock = None
            logger.info("Viewer closed.")

    def set_map_bounds(self, map_bounds: MapBounds) -> None:
        """Set or update the map bounds for the minimap.

        Parameters
        ----------
        map_bounds : MapBounds
            Spatial extent of the map.
        """
        self._minimap = Minimap(
            map_bounds=map_bounds,
            config=self._config.minimap_config,
        )

    # ---------------------------------------------------------------- observer mode

    def show(self, obs: ViewerObservation) -> bool:
        """Display a single observation (observer mode).

        Call this from an external agent loop to visualise each step.
        Handles pygame events (including quit) and returns whether the
        viewer is still open.

        Parameters
        ----------
        obs : ViewerObservation
            The observation to display.

        Returns
        -------
        bool
            ``True`` if the viewer is still open, ``False`` if the user
            closed the window.
        """
        if self._screen is None:
            self.open()
        assert self._screen is not None and self._clock is not None

        # Process events (allow closing the window).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return False
                if event.key == pygame.K_h:
                    self._show_hud = not self._show_hud
                if event.key == pygame.K_m:
                    self._show_minimap = not self._show_minimap

        self._render_frame(obs)
        self._clock.tick(self._config.target_fps)
        return True

    # ---------------------------------------------------------------- local manual mode

    def run_manual(self, simulator: "FlairSimulator") -> None:  # noqa: F821
        """Run the viewer in local manual flight mode.

        Opens the window, resets the simulator, and enters a keyboard
        control loop.  The loop continues until the user presses Escape
        or closes the window.

        Parameters
        ----------
        simulator : FlairSimulator
            The simulator to control.
        """
        self.set_map_bounds(simulator.map_bounds)
        self.open()
        assert self._screen is not None and self._clock is not None

        raw_obs = simulator.reset()
        obs = ViewerObservation.from_observation(raw_obs)

        # Reset minimap trail.
        if self._minimap is not None:
            self._minimap.reset_trail()
            self._minimap.add_trail_point(obs.drone_state.x, obs.drone_state.y)

        self._running = True

        while self._running:
            action = self._process_manual_events()

            if action is None:
                # No movement this frame, just re-render.
                self._render_frame(obs)
                self._clock.tick(self._config.target_fps)
                continue

            if action.action_type == ActionType.STOP:
                # Escape pressed.
                break

            if action == "RESET":
                # R pressed -- reset the episode.
                raw_obs = simulator.reset()
                obs = ViewerObservation.from_observation(raw_obs)
                if self._minimap is not None:
                    self._minimap.reset_trail()
                    self._minimap.add_trail_point(obs.drone_state.x, obs.drone_state.y)
                self._render_frame(obs)
                self._clock.tick(self._config.target_fps)
                continue

            # Apply action.
            if not obs.done:
                raw_obs = simulator.step(action)
                obs = ViewerObservation.from_observation(raw_obs)
                if self._minimap is not None:
                    self._minimap.add_trail_point(obs.drone_state.x, obs.drone_state.y)

            self._render_frame(obs)
            self._clock.tick(self._config.target_fps)

        self.close()

    # ---------------------------------------------------------------- remote observe

    def run_remote_observe(self, server_url: str) -> None:
        """Run the viewer in remote observe mode.

        Connects to the server's SSE ``/events`` endpoint and passively
        displays each observation as it arrives.  No keyboard interaction
        with the drone -- only HUD/minimap toggles and quit are handled.

        Parameters
        ----------
        server_url : str
            Base URL of the FlairSim server (e.g. ``http://localhost:8000``).
        """
        import httpx

        base = server_url.rstrip("/")

        # Fetch config to set up minimap.
        cfg = httpx.get(f"{base}/config", timeout=10.0).json()
        mb = cfg["map_bounds"]
        self.set_map_bounds(
            MapBounds(
                x_min=mb["x_min"],
                y_min=mb["y_min"],
                x_max=mb["x_max"],
                y_max=mb["y_max"],
            )
        )

        self.open()
        assert self._screen is not None and self._clock is not None

        if self._minimap is not None:
            self._minimap.reset_trail()

        self._running = True

        # SSE listener runs in a background thread so the pygame event
        # loop stays responsive.
        latest_obs: list[Optional[ViewerObservation]] = [None]
        sse_error: list[Optional[str]] = [None]

        def _sse_listener():
            try:
                with httpx.Client(timeout=None) as http:
                    with http.stream("GET", f"{base}/events") as resp:
                        resp.raise_for_status()
                        buffer = ""
                        event_type = ""
                        for line in resp.iter_lines():
                            if not self._running:
                                break
                            if line.startswith("event:"):
                                event_type = line[len("event:") :].strip()
                            elif line.startswith("data:"):
                                buffer = line[len("data:") :].strip()
                            elif line == "" and buffer:
                                if event_type == "observation":
                                    data = json.loads(buffer)
                                    latest_obs[0] = (
                                        ViewerObservation.from_server_response(data)
                                    )
                                buffer = ""
                                event_type = ""
                            elif line.startswith(":"):
                                # Comment / keep-alive, ignore.
                                pass
            except Exception as exc:
                if self._running:
                    sse_error[0] = str(exc)
                    logger.error("SSE listener error: %s", exc)

        sse_thread = threading.Thread(target=_sse_listener, daemon=True)
        sse_thread.start()

        logger.info("Observing server at %s (SSE)", base)

        current_obs: Optional[ViewerObservation] = None

        while self._running:
            # Process quit / toggle events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._running = False
                        break
                    if event.key == pygame.K_h:
                        self._show_hud = not self._show_hud
                    if event.key == pygame.K_m:
                        self._show_minimap = not self._show_minimap

            if not self._running:
                break

            # Check for new observation from SSE.
            new_obs = latest_obs[0]
            if new_obs is not None and new_obs is not current_obs:
                current_obs = new_obs
                latest_obs[0] = None  # Consume.
                if self._minimap is not None:
                    if current_obs.step == 0:
                        self._minimap.reset_trail()
                    self._minimap.add_trail_point(
                        current_obs.drone_state.x, current_obs.drone_state.y
                    )

            if current_obs is not None:
                self._render_frame(current_obs)
            else:
                # No observation yet -- show a waiting screen.
                self._screen.fill((30, 30, 30))
                self._draw_centered_text("Waiting for observations...")
                pygame.display.flip()

            # Check SSE errors.
            if sse_error[0] is not None:
                logger.error("SSE connection lost: %s", sse_error[0])
                self._screen.fill((30, 30, 30))
                self._draw_centered_text(f"SSE error: {sse_error[0]}")
                pygame.display.flip()
                time.sleep(1.0)
                sse_error[0] = None

            self._clock.tick(self._config.target_fps)

        self.close()
        sse_thread.join(timeout=2.0)

    # ---------------------------------------------------------------- remote fly

    def run_remote_fly(self, server_url: str) -> None:
        """Run the viewer in remote fly mode.

        Connects to the server's SSE ``/events`` endpoint *and* sends
        actions via ``POST /step`` and ``POST /reset`` in response to
        keyboard input.

        Parameters
        ----------
        server_url : str
            Base URL of the FlairSim server (e.g. ``http://localhost:8000``).
        """
        import httpx

        base = server_url.rstrip("/")

        # Fetch config to set up minimap.
        cfg = httpx.get(f"{base}/config", timeout=10.0).json()
        mb = cfg["map_bounds"]
        self.set_map_bounds(
            MapBounds(
                x_min=mb["x_min"],
                y_min=mb["y_min"],
                x_max=mb["x_max"],
                y_max=mb["y_max"],
            )
        )

        self.open()
        assert self._screen is not None and self._clock is not None

        if self._minimap is not None:
            self._minimap.reset_trail()

        self._running = True

        http = httpx.Client(base_url=base, timeout=10.0)

        # Reset to start an episode.
        try:
            resp = http.post("/reset")
            resp.raise_for_status()
            obs = ViewerObservation.from_server_response(resp.json())
        except Exception as exc:
            logger.error("Failed to reset: %s", exc)
            self._screen.fill((30, 30, 30))
            self._draw_centered_text(f"Server error: {exc}")
            pygame.display.flip()
            time.sleep(2.0)
            self.close()
            http.close()
            return

        if self._minimap is not None:
            self._minimap.add_trail_point(obs.drone_state.x, obs.drone_state.y)

        while self._running:
            action = self._process_manual_events()

            if action is None:
                # No movement this frame, just re-render.
                self._render_frame(obs)
                self._clock.tick(self._config.target_fps)
                continue

            if isinstance(action, str) and action == "RESET":
                try:
                    resp = http.post("/reset")
                    resp.raise_for_status()
                    obs = ViewerObservation.from_server_response(resp.json())
                    if self._minimap is not None:
                        self._minimap.reset_trail()
                        self._minimap.add_trail_point(
                            obs.drone_state.x, obs.drone_state.y
                        )
                except Exception as exc:
                    logger.error("Reset failed: %s", exc)
                self._render_frame(obs)
                self._clock.tick(self._config.target_fps)
                continue

            if isinstance(action, Action) and action.action_type == ActionType.STOP:
                # Escape -- send stop to server and exit.
                try:
                    http.post(
                        "/step",
                        json={"dx": 0, "dy": 0, "dz": 0, "action_type": "stop"},
                    )
                except Exception:
                    pass
                break

            # Send action to server.
            if isinstance(action, Action) and not obs.done:
                body = {
                    "dx": action.dx,
                    "dy": action.dy,
                    "dz": action.dz,
                    "action_type": action.action_type.value,
                }
                try:
                    resp = http.post("/step", json=body)
                    resp.raise_for_status()
                    obs = ViewerObservation.from_server_response(resp.json())
                    if self._minimap is not None:
                        self._minimap.add_trail_point(
                            obs.drone_state.x, obs.drone_state.y
                        )
                except Exception as exc:
                    logger.error("Step failed: %s", exc)

            self._render_frame(obs)
            self._clock.tick(self._config.target_fps)

        http.close()
        self.close()

    # ---------------------------------------------------------------- rendering

    def _render_frame(self, obs: ViewerObservation) -> None:
        """Render a complete frame: image + HUD + minimap."""
        assert self._screen is not None

        # Convert observation image to a pygame surface.
        rgb = obs.image_rgb  # (H, W, 3) uint8
        img_surface = self._array_to_surface(rgb)

        # Scale to fill the window.
        scaled = pygame.transform.scale(
            img_surface,
            (self._config.window_width, self._config.window_height),
        )
        self._screen.blit(scaled, (0, 0))

        # Draw HUD.
        if self._show_hud:
            fps = self._clock.get_fps() if self._clock else 0.0
            self._hud.render(self._screen, obs, fps=fps)

        # Draw minimap.
        if self._show_minimap and self._minimap is not None:
            self._minimap.render(
                self._screen,
                drone_x=obs.drone_state.x,
                drone_y=obs.drone_state.y,
                footprint_size=obs.ground_footprint,
            )

        pygame.display.flip()

    def _draw_centered_text(self, text: str) -> None:
        """Draw centred text on the screen (used for waiting/error states)."""
        assert self._screen is not None
        font = pygame.font.SysFont("monospace", 20)
        rendered = font.render(text, True, (200, 200, 200))
        rect = rendered.get_rect(
            center=(self._config.window_width // 2, self._config.window_height // 2)
        )
        self._screen.blit(rendered, rect)

    @staticmethod
    def _array_to_surface(rgb: np.ndarray) -> pygame.Surface:
        """Convert an ``(H, W, 3)`` uint8 array to a pygame Surface.

        Parameters
        ----------
        rgb : np.ndarray
            RGB image array with shape ``(H, W, 3)`` and dtype ``uint8``.

        Returns
        -------
        pygame.Surface
            The corresponding pygame surface.
        """
        # Ensure contiguous C-order array.
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)

        h, w, _ = rgb.shape
        return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

    # ---------------------------------------------------------------- input

    def _process_manual_events(self) -> Optional[Action | str]:
        """Process pygame events and return an Action or control signal.

        Returns
        -------
        Action, str, or None
            - An :class:`Action` if the user pressed a movement key.
            - The string ``"RESET"`` if R was pressed.
            - ``None`` if no action-relevant event occurred this frame.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return Action.stop()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                    return Action.stop()

                if event.key == pygame.K_h:
                    self._show_hud = not self._show_hud
                    return None

                if event.key == pygame.K_m:
                    self._show_minimap = not self._show_minimap
                    return None

                if event.key == pygame.K_r:
                    return "RESET"

                if event.key == pygame.K_SPACE:
                    return Action.found()

                # Adjust step size.
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._move_step = min(self._move_step * 1.5, 500.0)
                    logger.debug("Move step: %.1f m", self._move_step)
                    return None
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._move_step = max(self._move_step / 1.5, 1.0)
                    logger.debug("Move step: %.1f m", self._move_step)
                    return None

        # Check held keys for continuous movement.
        keys = pygame.key.get_pressed()
        dx, dy, dz = 0.0, 0.0, 0.0

        if keys[pygame.K_z] or keys[pygame.K_UP]:
            dy += self._move_step
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy -= self._move_step
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += self._move_step
        if keys[pygame.K_q] or keys[pygame.K_LEFT]:
            dx -= self._move_step
        if keys[pygame.K_e]:
            dz += self._config.altitude_step
        if keys[pygame.K_a]:
            dz -= self._config.altitude_step

        if dx != 0 or dy != 0 or dz != 0:
            return Action.move(dx=dx, dy=dy, dz=dz)

        return None

    # ---------------------------------------------------------------- properties

    @property
    def is_open(self) -> bool:
        """Whether the viewer window is currently open."""
        return self._screen is not None and self._running

    @property
    def move_step(self) -> float:
        """Current movement step size in metres."""
        return self._move_step

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        return (
            f"FlairViewer(status={status}, "
            f"{self._config.window_width}x{self._config.window_height})"
        )
