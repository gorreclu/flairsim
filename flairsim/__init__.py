"""
FlairSim -- Drone simulator over FLAIR-HUB aerial imagery.

A lightweight, modular drone simulator designed for evaluating Vision-Language
Models (VLMs) on exploration and search tasks over real French aerial imagery
from the FLAIR-HUB dataset (IGN).

Typical usage::

    from flairsim import FlairSimulator, Action

    sim = FlairSimulator(data_dir="path/to/D004-2021_AERIAL_RGBI")
    obs = sim.reset()

    while not obs.done:
        action = Action(dx=10.0, dy=0.0, dz=-5.0)
        obs = sim.step(action)

    print(f"Success: {obs.success}")
    sim.close()
"""

__version__ = "0.1.0"

# Public API -- import the most commonly used classes at package level.
from .core.action import Action, ActionType  # noqa: F401
from .core.grid import GridConfig, GridOverlay  # noqa: F401
from .core.observation import EpisodeResult, Observation  # noqa: F401
from .core.scenario import Scenario, ScenarioLoader, ScenarioTarget  # noqa: F401
from .core.simulator import FlairSimulator, SimulatorConfig  # noqa: F401
from .drone.camera import CameraConfig, CameraModel  # noqa: F401
from .drone.drone import Drone, DroneConfig, DroneState  # noqa: F401
from .drone.telemetry import FlightLog, TelemetryRecord  # noqa: F401
from .map.map_manager import MapBounds, MapManager  # noqa: F401
from .map.modality import Modality, ModalitySpec  # noqa: F401
from .map.modality import discover_modalities, pick_primary_modality  # noqa: F401

__all__ = [
    "Action",
    "ActionType",
    "CameraConfig",
    "CameraModel",
    "Drone",
    "DroneConfig",
    "DroneState",
    "EpisodeResult",
    "FlairSimulator",
    "FlightLog",
    "GridConfig",
    "GridOverlay",
    "MapBounds",
    "MapManager",
    "Modality",
    "ModalitySpec",
    "Observation",
    "Scenario",
    "ScenarioLoader",
    "ScenarioTarget",
    "SimulatorConfig",
    "TelemetryRecord",
    "discover_modalities",
    "pick_primary_modality",
]
