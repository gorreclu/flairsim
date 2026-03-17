"""
FlairSimulator -- the main simulation engine.

This module provides the central :class:`FlairSimulator` class that
implements the ``reset() -> Observation`` / ``step(Action) -> Observation``
interface.  It wires together:

* :class:`~flairsim.map.map_manager.MapManager` for terrain data,
* :class:`~flairsim.drone.drone.Drone` for physical state,
* :class:`~flairsim.drone.camera.CameraModel` for image observations,
* :class:`~flairsim.drone.telemetry.FlightLog` for trajectory recording.

Usage
-----
::

    from flairsim.core.simulator import FlairSimulator

    sim = FlairSimulator(data_dir="path/to/D004-2021_AERIAL_RGBI")
    obs = sim.reset()

    while not obs.done:
        action = Action(dx=10.0, dy=0.0, dz=0.0)
        obs = sim.step(action)

    print(sim.flight_log)
    sim.close()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..drone.camera import CameraConfig, CameraModel
from ..drone.drone import Drone, DroneConfig
from ..drone.telemetry import FlightLog, TelemetryRecord
from ..map.map_manager import MapManager
from ..map.modality import (
    Modality,
    discover_modalities,
    infer_domain_from_dir,
    is_single_modality_dir,
    pick_primary_modality,
)
from .action import Action, ActionType
from .observation import EpisodeResult, Observation
from .scenario import Scenario

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulator configuration
# ---------------------------------------------------------------------------


class SimulatorConfig:
    """Configuration for the FlairSimulator.

    Bundles the sub-component configurations together with episode-level
    parameters.

    Parameters
    ----------
    drone_config : DroneConfig or None
        Physical limits for the drone.
    camera_config : CameraConfig or None
        Sensor parameters for the camera.
    max_steps : int
        Maximum number of steps per episode before auto-termination.
    roi : str or None
        Which ROI to load.  ``None`` auto-selects the largest.
    preload_tiles : bool
        Whether to load all tiles into memory at start.
    """

    def __init__(
        self,
        drone_config: Optional[DroneConfig] = None,
        camera_config: Optional[CameraConfig] = None,
        max_steps: int = 500,
        roi: Optional[str] = None,
        preload_tiles: bool = True,
    ) -> None:
        self.drone_config = drone_config or DroneConfig()
        self.camera_config = camera_config or CameraConfig()
        self.max_steps = max_steps
        self.roi = roi
        self.preload_tiles = preload_tiles


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class FlairSimulator:
    """Main simulation engine over FLAIR-HUB aerial imagery.

    The simulator manages a single episode at a time.  Call :meth:`reset`
    to start a new episode, then :meth:`step` repeatedly with agent
    actions.  The episode ends when:

    1. The agent calls ``Action.found()`` (success is evaluated).
    2. The agent calls ``Action.stop()`` (voluntary termination).
    3. The step limit is reached (auto-termination, no success).

    Parameters
    ----------
    data_dir : str or Path
        Path to a FLAIR-HUB data directory.  This can be either:

        * A **single-modality** directory (e.g. ``D004-2021_AERIAL_RGBI``)
          containing GeoTIFF tiles directly -- backward-compatible mode.
          Sibling modalities will be auto-discovered in the parent dir.
        * A **flat FLAIR-HUB root** directory containing modality sub-dirs
          for multiple domains (e.g. ``FLAIR-HUB/``) -- use *domain* to
          select which domain to load.
        * A **parent directory** containing modality sub-dirs for a single
          domain -- the simulator will auto-discover all modalities.

    config : SimulatorConfig or None
        Full configuration.  ``None`` uses defaults.
    scenario : Scenario or None
        If set, the simulator runs in scenario mode.
    domain : str or None
        FLAIR-HUB domain prefix (e.g. ``"D006-2020"``).  When the data
        directory has a flat layout with multiple domains as siblings,
        this is **required** to select which domain's modalities to load.
        When *data_dir* points to a single modality directory, the domain
        is inferred from the directory name if not provided.

    Attributes
    ----------
    map_manager : MapManager
        The loaded map for the primary modality (used for bounds,
        coordinate queries, and the ``image`` field in observations).
    map_managers : dict[str, MapManager]
        All loaded map managers, keyed by modality name (e.g.
        ``"AERIAL_RGBI"``).  In single-modality mode this contains
        exactly one entry.
    primary_modality : str or None
        Name of the primary modality (e.g. ``"AERIAL_RGBI"``), or
        ``None`` in legacy single-modality mode.
    drone : Drone
        The simulated drone.
    camera : CameraModel
        The nadir camera.
    flight_log : FlightLog
        Telemetry log for the current episode.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: Optional[SimulatorConfig] = None,
        scenario: Optional[Scenario] = None,
        domain: Optional[str] = None,
    ) -> None:
        self._config = config or SimulatorConfig()
        self._data_dir = Path(data_dir).resolve()
        self._scenario = scenario
        self._domain = domain

        # If a scenario is active, override max_steps from the scenario.
        if scenario is not None:
            self._config.max_steps = scenario.max_steps

        # --- Initialise sub-components ---
        logger.info("Initialising FlairSimulator with data: %s", self._data_dir)

        # --- Multi-modality discovery ---
        self.map_managers: Dict[str, MapManager] = {}
        self.primary_modality: Optional[str] = None

        if is_single_modality_dir(self._data_dir):
            # Single-modality mode: the user pointed directly at a modality
            # directory (e.g. D006-2020_AERIAL_RGBI/).
            # Also try to discover sibling modalities in the parent dir.
            mm = MapManager(
                data_dir=self._data_dir,
                roi=self._config.roi,
                preload=self._config.preload_tiles,
            )
            mod_name = self._detect_modality_name(self._data_dir)
            if mod_name:
                self.map_managers[mod_name] = mm
                self.primary_modality = mod_name
            else:
                self.map_managers["default"] = mm
                self.primary_modality = "default"
            self.map_manager = mm

            # Try auto-discovering sibling modalities in the parent directory
            # using the inferred domain prefix.
            inferred_domain = domain or infer_domain_from_dir(self._data_dir)
            if inferred_domain and self._data_dir.parent.is_dir():
                siblings = discover_modalities(
                    self._data_dir.parent, domain=inferred_domain
                )
                for mod, mod_path in siblings.items():
                    if mod.name not in self.map_managers:
                        try:
                            sibling_mm = MapManager(
                                data_dir=mod_path,
                                roi=self._config.roi,
                                preload=self._config.preload_tiles,
                            )
                            self.map_managers[mod.name] = sibling_mm
                        except Exception as exc:
                            logger.warning(
                                "Skipping sibling modality %s: %s",
                                mod.name,
                                exc,
                            )
                if len(self.map_managers) > 1:
                    logger.info(
                        "Auto-discovered %d sibling modalities for domain %s",
                        len(self.map_managers) - 1,
                        inferred_domain,
                    )
        else:
            # Parent directory mode: auto-discover modalities.
            # Use the domain filter if provided.
            effective_domain = domain
            if not effective_domain:
                # Try to infer domain from the data_dir name itself.
                effective_domain = infer_domain_from_dir(self._data_dir)

            discovered = discover_modalities(self._data_dir, domain=effective_domain)
            if not discovered:
                raise ValueError(
                    f"No FLAIR-HUB modalities found in {self._data_dir}"
                    + (f" for domain '{effective_domain}'" if effective_domain else "")
                    + ". Pass a single-modality directory, or a parent directory "
                    "containing modality sub-directories (with --domain if needed)."
                )

            primary_mod = pick_primary_modality(discovered)
            self.primary_modality = primary_mod.name

            for mod, mod_path in discovered.items():
                mm = MapManager(
                    data_dir=mod_path,
                    roi=self._config.roi,
                    preload=self._config.preload_tiles,
                )
                self.map_managers[mod.name] = mm

            self.map_manager = self.map_managers[self.primary_modality]
            logger.info(
                "Multi-modality mode: %d modalities, primary=%s",
                len(self.map_managers),
                self.primary_modality,
            )

        self.drone = Drone(config=self._config.drone_config)
        self.camera = CameraModel(config=self._config.camera_config)
        self.flight_log = FlightLog()

        # Set drone bounds from the primary map extent.
        bounds = self.map_manager.bounds
        self.drone.set_bounds(
            x_bounds=(bounds.x_min, bounds.x_max),
            y_bounds=(bounds.y_min, bounds.y_max),
        )

        # Episode state.
        self._step_count: int = 0
        self._done: bool = True  # True until reset() is called.
        self._result: Optional[EpisodeResult] = None

        logger.info(
            "FlairSimulator ready: map=%s  drone=%s  camera=%s  scenario=%s  modalities=%s",
            self.map_manager,
            self.drone,
            self.camera,
            self._scenario.scenario_id if self._scenario else "none",
            list(self.map_managers.keys()),
        )

    # ---------------------------------------------------------------- reset

    def reset(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> Observation:
        """Start a new episode.

        If a scenario is active and no explicit position is given, the
        scenario's start position is used.  Otherwise, if no start
        position is given, the drone is placed at the centre of the
        map at the default altitude.

        Parameters
        ----------
        x, y : float or None
            Starting horizontal position.  ``None`` uses the scenario
            start (if active) or the map centre.
        z : float or None
            Starting altitude.  ``None`` uses the scenario start (if
            active) or ``drone_config.default_altitude``.

        Returns
        -------
        Observation
            The initial observation (step 0, done=False).
        """
        # Apply scenario start position as defaults (explicit args take priority).
        if self._scenario is not None:
            s = self._scenario.start
            if x is None and s.x is not None:
                x = s.x
            if y is None and s.y is not None:
                y = s.y
            if z is None and s.z is not None:
                z = s.z

        # Default to map centre.
        cx, cy = self.map_manager.bounds.center
        start_x = x if x is not None else cx
        start_y = y if y is not None else cy

        # Reset drone.
        self.drone.reset(x=start_x, y=start_y, z=z)

        # Reset episode state.
        self._step_count = 0
        self._done = False
        self._result = None
        self.flight_log.clear()

        # Capture initial observation.
        obs = self._make_observation()

        # Record initial telemetry.
        self._record_telemetry(dx=0.0, dy=0.0, dz=0.0, was_clipped=False)

        logger.info(
            "Episode started at (%.1f, %.1f, %.1f)",
            start_x,
            start_y,
            obs.altitude,
        )

        return obs

    # ---------------------------------------------------------------- step

    def step(self, action: Action) -> Observation:
        """Advance the simulation by one step.

        Parameters
        ----------
        action : Action
            The agent's command for this step.

        Returns
        -------
        Observation
            The resulting observation after applying the action.

        Raises
        ------
        RuntimeError
            If the episode has already ended (call ``reset()`` first).
        """
        if self._done:
            raise RuntimeError(
                "Episode has ended.  Call reset() to start a new episode."
            )

        # --- Apply movement ---
        move_result = self.drone.move(dx=action.dx, dy=action.dy, dz=action.dz)
        self._step_count += 1

        # --- Record telemetry ---
        self._record_telemetry(
            dx=move_result.dx_actual,
            dy=move_result.dy_actual,
            dz=move_result.dz_actual,
            was_clipped=move_result.was_clipped,
        )

        # --- Check termination conditions ---
        if action.action_type == ActionType.FOUND:
            self._done = True
            # Evaluate success against scenario target if available.
            state = self.drone.state
            if self._scenario is not None:
                success = self._scenario.evaluate(state.x, state.y)
                dist = self._scenario.distance_to_target(state.x, state.y)
                if success:
                    reason = (
                        f"Agent declared FOUND within target radius "
                        f"(distance={dist:.1f}m, "
                        f"radius={self._scenario.target.radius:.0f}m)."
                    )
                else:
                    reason = (
                        f"Agent declared FOUND but is outside target radius "
                        f"(distance={dist:.1f}m, "
                        f"radius={self._scenario.target.radius:.0f}m)."
                    )
            else:
                success = False
                reason = "Agent declared FOUND (no target configured for evaluation)."
            self._result = EpisodeResult(
                success=success,
                reason=reason,
                steps_taken=self._step_count,
                distance_travelled=self.drone.state.total_distance,
            )
            logger.info("Agent declared FOUND at step %d.", self._step_count)

        elif action.action_type == ActionType.STOP:
            self._done = True
            self._result = EpisodeResult(
                success=False,
                reason="Agent voluntarily stopped.",
                steps_taken=self._step_count,
                distance_travelled=self.drone.state.total_distance,
            )
            logger.info("Agent stopped at step %d.", self._step_count)

        elif self._step_count >= self._config.max_steps:
            self._done = True
            self._result = EpisodeResult(
                success=False,
                reason=f"Step limit reached ({self._config.max_steps}).",
                steps_taken=self._step_count,
                distance_travelled=self.drone.state.total_distance,
            )
            logger.info(
                "Step limit reached (%d) at step %d.",
                self._config.max_steps,
                self._step_count,
            )

        # --- Build observation ---
        return self._make_observation()

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        """Release resources.

        Currently a no-op, but reserved for future cleanup (e.g.
        closing file handles, GPU contexts, viewer windows).
        """
        logger.info("FlairSimulator closed.")

    # ---------------------------------------------------------------- helpers

    def _make_observation(self) -> Observation:
        """Capture the current camera view and build an Observation."""
        state = self.drone.state

        # Capture the primary modality image (always present).
        image = self.camera.capture(
            map_manager=self.map_manager,
            x=state.x,
            y=state.y,
            z=state.z,
        )

        # Capture all modalities for the images dict.
        images: Dict[str, np.ndarray] = {}
        if len(self.map_managers) > 1:
            for mod_name, mm in self.map_managers.items():
                mod_image = self.camera.capture(
                    map_manager=mm,
                    x=state.x,
                    y=state.y,
                    z=state.z,
                )
                images[mod_name] = mod_image
        elif self.primary_modality:
            # Single modality: still populate images for consistency.
            images[self.primary_modality] = image

        metadata: dict = {
            "roi": self.map_manager.roi_name,
            "data_dir": str(self._data_dir),
            "max_steps": self._config.max_steps,
            "primary_modality": self.primary_modality,
            "modalities": list(self.map_managers.keys()),
        }

        # Inject scenario metadata if present.
        if self._scenario is not None:
            sc = self._scenario
            metadata["scenario_id"] = sc.scenario_id
            metadata["scenario_name"] = sc.name
            metadata["scenario_description"] = sc.description
            metadata["target_x"] = sc.target.x
            metadata["target_y"] = sc.target.y
            metadata["target_radius"] = sc.target.radius
            metadata["distance_to_target"] = sc.distance_to_target(state.x, state.y)

        return Observation(
            image=image,
            drone_state=state,
            step=self._step_count,
            done=self._done,
            result=self._result,
            ground_footprint=self.camera.ground_footprint_size(state.z),
            ground_resolution=self.camera.ground_resolution(state.z),
            metadata=metadata,
            images=images,
        )

    def _record_telemetry(
        self,
        dx: float,
        dy: float,
        dz: float,
        was_clipped: bool,
    ) -> None:
        """Append a telemetry record for the current state."""
        state = self.drone.state
        record = TelemetryRecord(
            step=self._step_count,
            x=state.x,
            y=state.y,
            z=state.z,
            dx=dx,
            dy=dy,
            dz=dz,
            ground_footprint=self.camera.ground_footprint_size(state.z),
            was_clipped=was_clipped,
        )
        self.flight_log.append(record)

    # ---------------------------------------------------------------- properties

    @property
    def is_running(self) -> bool:
        """Whether an episode is currently active."""
        return not self._done

    @property
    def scenario(self) -> Optional[Scenario]:
        """The active scenario, or ``None`` if free-flight mode."""
        return self._scenario

    @property
    def step_count(self) -> int:
        """Current step number within the episode."""
        return self._step_count

    @property
    def max_steps(self) -> int:
        """Maximum steps allowed per episode."""
        return self._config.max_steps

    @property
    def map_bounds(self) -> "MapBounds":  # noqa: F821
        """Spatial extent of the loaded map."""
        return self.map_manager.bounds

    # ---------------------------------------------------------------- spawn helpers

    @staticmethod
    def _detect_modality_name(data_dir: Path) -> Optional[str]:
        """Try to detect the modality name from a data directory path.

        Checks if the directory name ends with a known FLAIR-HUB
        modality suffix.

        Parameters
        ----------
        data_dir : Path
            The data directory path.

        Returns
        -------
        str or None
            The modality name (e.g. ``"AERIAL_RGBI"``), or ``None``
            if no known suffix matches.
        """
        name = data_dir.name
        for member in Modality:
            suffix = member.value.dir_suffix
            prefix_end = len(name) - len(suffix)
            if name.endswith(suffix) and (
                prefix_end == 0 or name[prefix_end - 1] == "_"
            ):
                return member.name
        return None

    def random_start_position(
        self,
        rng: Optional[np.random.Generator] = None,
        margin: float = 50.0,
    ) -> Tuple[float, float]:
        """Generate a random start position within the map bounds.

        Parameters
        ----------
        rng : np.random.Generator or None
            Random number generator.  ``None`` creates a new one.
        margin : float
            Inset from map edges in metres (default 50 m) to avoid
            spawning at the very edge where the camera view would be
            mostly out of bounds.

        Returns
        -------
        tuple[float, float]
            ``(x, y)`` position.
        """
        if rng is None:
            rng = np.random.default_rng()

        b = self.map_manager.bounds
        x = rng.uniform(b.x_min + margin, b.x_max - margin)
        y = rng.uniform(b.y_min + margin, b.y_max - margin)
        return (x, y)

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        status = "running" if self.is_running else "idle"
        sc = f", scenario={self._scenario.scenario_id!r}" if self._scenario else ""
        mods = list(self.map_managers.keys())
        mod_info = f", modalities={mods}" if len(mods) > 1 else ""
        return (
            f"FlairSimulator(status={status}, step={self._step_count}, "
            f"map={self.map_manager.roi_name}, "
            f"drone={self.drone}{sc}{mod_info})"
        )
