"""
Predefined mission scenarios for the FlairSimulator.

A :class:`Scenario` defines a complete mission: which dataset/ROI to fly over,
where the drone starts, what the target is (position + acceptance radius), and
episode limits.  Scenarios are loaded from YAML files via :class:`ScenarioLoader`.

YAML format
------------
.. code-block:: yaml

    scenario_id: find_building_D006
    name: Find the industrial building
    description: Locate the large warehouse in zone D006.
    dataset:
      data_dir: D006-2020_AERIAL_RGBI   # path relative to data root
      domain: D006-2020                  # optional, inferred from data_dir
      roi: null                          # auto-select largest
      modalities:                        # optional, default ["AERIAL_RGBI"]
        - AERIAL_RGBI
      source: auto                       # "local", "huggingface", or "auto"
    start:
      x: 800100.0
      y: 6500200.0
      z: 150.0
    target:
      x: 800300.0
      y: 6500050.0
      radius: 50.0
    max_steps: 200
    prompt:                              # optional VLM prompt template
      system: |
        You are a drone navigation agent...
      user_template: |
        Position: ({x}, {y}, {z}). Steps remaining: {steps_remaining}.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioDataset:
    """Dataset location for a scenario.

    Attributes
    ----------
    data_dir : str
        Path to the data directory, relative to the data root.
    roi : str or None
        Specific ROI to load.  ``None`` auto-selects the largest.
    domain : str or None
        FLAIR-HUB domain prefix (e.g. ``"D006-2020"``).  When the
        data root has a flat layout with multiple domains, this tells
        the simulator which domain's modalities to load.  If ``None``,
        the domain is inferred from *data_dir*.
    modalities : list of str
        Modality names to load (e.g. ``["AERIAL_RGBI", "DEM_ELEV"]``).
        Defaults to ``["AERIAL_RGBI"]``.
    source : str
        Where to find the data: ``"local"`` (must exist on disk),
        ``"huggingface"`` (download from HF), or ``"auto"`` (try
        local first, fall back to HuggingFace download).
    """

    data_dir: str
    roi: Optional[str] = None
    domain: Optional[str] = None
    modalities: List[str] = field(default_factory=lambda: ["AERIAL_RGBI"])
    source: str = "auto"


@dataclass(frozen=True, slots=True)
class ScenarioStart:
    """Starting position for the drone.

    Attributes
    ----------
    x : float or None
        Easting in metres.  ``None`` uses the map centre.
    y : float or None
        Northing in metres.  ``None`` uses the map centre.
    z : float or None
        Altitude in metres.  ``None`` uses the default altitude.
    """

    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ScenarioTarget:
    """Target location the agent must find.

    Attributes
    ----------
    x : float
        Target easting in metres.
    y : float
        Target northing in metres.
    radius : float
        Acceptance radius in metres.  The agent succeeds if it declares
        FOUND while within this distance of ``(x, y)``.
    """

    x: float
    y: float
    radius: float = 50.0

    def distance_to(self, px: float, py: float) -> float:
        """Euclidean distance from a point to the target centre."""
        return math.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)

    def is_within(self, px: float, py: float) -> bool:
        """Return ``True`` if ``(px, py)`` is within the acceptance radius."""
        return self.distance_to(px, py) <= self.radius


@dataclass(frozen=True, slots=True)
class ScenarioPrompt:
    """VLM prompt template for a scenario.

    Attributes
    ----------
    system : str
        System prompt template for the VLM.  Empty string means no
        system prompt is specified (the agent is free to choose).
    user_template : str
        User prompt template with ``{placeholders}`` that are filled
        at each step.  Available placeholders: ``{x}``, ``{y}``,
        ``{z}``, ``{steps_remaining}``, ``{distance}``,
        ``{step}``, ``{max_steps}``.
    """

    system: str = ""
    user_template: str = ""


@dataclass(frozen=True, slots=True)
class Scenario:
    """A complete mission scenario.

    Attributes
    ----------
    scenario_id : str
        Unique identifier (also the YAML filename stem).
    name : str
        Short human-readable name.
    description : str
        Longer description of the mission.
    dataset : ScenarioDataset
        Where to load map data from.
    start : ScenarioStart
        Drone starting position.
    target : ScenarioTarget
        What the agent must find.
    max_steps : int
        Maximum steps before the episode is terminated.
    prompt : ScenarioPrompt
        Optional VLM prompt template.
    environment : list of str
        Tags describing the terrain type (e.g. ``["urban"]``,
        ``["rural", "forest"]``).  Defaults to an empty list.
    difficulty : int
        Difficulty level from 1 (easy) to 3 (hard).  Defaults to 1.
    """

    scenario_id: str
    name: str
    description: str = ""
    dataset: ScenarioDataset = field(
        default_factory=lambda: ScenarioDataset(data_dir="")
    )
    start: ScenarioStart = field(default_factory=ScenarioStart)
    target: ScenarioTarget = field(default_factory=lambda: ScenarioTarget(x=0.0, y=0.0))
    max_steps: int = 500
    prompt: ScenarioPrompt = field(default_factory=ScenarioPrompt)
    environment: List[str] = field(default_factory=list)
    difficulty: int = 1

    def evaluate(self, drone_x: float, drone_y: float) -> bool:
        """Evaluate whether the agent succeeded.

        Returns ``True`` if the drone is within the target acceptance radius.

        Parameters
        ----------
        drone_x, drone_y : float
            Current drone position when FOUND was declared.
        """
        return self.target.is_within(drone_x, drone_y)

    def distance_to_target(self, drone_x: float, drone_y: float) -> float:
        """Distance from the drone to the target centre."""
        return self.target.distance_to(drone_x, drone_y)

    def to_dict(self) -> Dict[str, object]:
        """Serialise to a JSON-friendly dictionary."""
        result: Dict[str, object] = {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "dataset": {
                "data_dir": self.dataset.data_dir,
                "roi": self.dataset.roi,
                "domain": self.dataset.domain,
                "modalities": list(self.dataset.modalities),
                "source": self.dataset.source,
            },
            "start": {
                "x": self.start.x,
                "y": self.start.y,
                "z": self.start.z,
            },
            "target": {
                "x": self.target.x,
                "y": self.target.y,
                "radius": self.target.radius,
            },
            "max_steps": self.max_steps,
            "environment": list(self.environment),
            "difficulty": self.difficulty,
        }
        # Include prompt only if non-empty.
        if self.prompt.system or self.prompt.user_template:
            result["prompt"] = {
                "system": self.prompt.system,
                "user_template": self.prompt.user_template,
            }
        return result

    def __repr__(self) -> str:
        return (
            f"Scenario(id={self.scenario_id!r}, name={self.name!r}, "
            f"target=({self.target.x:.0f}, {self.target.y:.0f}), "
            f"radius={self.target.radius:.0f}m, "
            f"difficulty={self.difficulty}, env={self.environment})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_scenario(data: Dict, source: str = "<unknown>") -> Scenario:
    """Parse a raw YAML dictionary into a :class:`Scenario`.

    Parameters
    ----------
    data : dict
        Parsed YAML content.
    source : str
        File path or description for error messages.

    Returns
    -------
    Scenario

    Raises
    ------
    ValueError
        If required fields are missing.
    """
    # --- scenario_id ---
    scenario_id = data.get("scenario_id")
    if not scenario_id:
        raise ValueError(f"Missing 'scenario_id' in scenario file: {source}")

    name = data.get("name", scenario_id)
    description = data.get("description", "")

    # --- dataset ---
    ds_raw = data.get("dataset", {})
    if not ds_raw.get("data_dir"):
        raise ValueError(
            f"Missing 'dataset.data_dir' in scenario '{scenario_id}': {source}"
        )
    raw_modalities = ds_raw.get("modalities")
    if raw_modalities is not None and not isinstance(raw_modalities, list):
        raw_modalities = [raw_modalities]
    dataset = ScenarioDataset(
        data_dir=ds_raw["data_dir"],
        roi=ds_raw.get("roi"),
        domain=ds_raw.get("domain"),
        modalities=raw_modalities or ["AERIAL_RGBI"],
        source=ds_raw.get("source", "auto"),
    )

    # --- start ---
    st_raw = data.get("start", {})
    start = ScenarioStart(
        x=st_raw.get("x"),
        y=st_raw.get("y"),
        z=st_raw.get("z"),
    )

    # --- target ---
    tg_raw = data.get("target", {})
    if "x" not in tg_raw or "y" not in tg_raw:
        raise ValueError(
            f"Missing 'target.x' and/or 'target.y' in scenario "
            f"'{scenario_id}': {source}"
        )
    target = ScenarioTarget(
        x=float(tg_raw["x"]),
        y=float(tg_raw["y"]),
        radius=float(tg_raw.get("radius", 50.0)),
    )

    max_steps = int(data.get("max_steps", 500))

    # --- prompt ---
    pr_raw = data.get("prompt", {})
    prompt = ScenarioPrompt(
        system=pr_raw.get("system", ""),
        user_template=pr_raw.get("user_template", ""),
    )

    # --- environment & difficulty ---
    raw_env = data.get("environment", [])
    if isinstance(raw_env, str):
        raw_env = [raw_env]
    environment: List[str] = list(raw_env) if raw_env else []

    difficulty = int(data.get("difficulty", 1))
    if difficulty < 1 or difficulty > 3:
        logger.warning(
            "Scenario '%s': difficulty=%d out of range [1, 3], clamping.",
            scenario_id,
            difficulty,
        )
        difficulty = max(1, min(3, difficulty))

    return Scenario(
        scenario_id=scenario_id,
        name=name,
        description=description,
        dataset=dataset,
        start=start,
        target=target,
        max_steps=max_steps,
        prompt=prompt,
        environment=environment,
        difficulty=difficulty,
    )


class ScenarioLoader:
    """Load and manage scenario YAML files from a directory.

    Parameters
    ----------
    scenarios_dir : str or Path
        Directory containing ``*.yaml`` / ``*.yml`` scenario files.
    data_root : str or Path or None
        Root directory prepended to relative ``data_dir`` paths in
        scenarios.  ``None`` defaults to the current working directory.

    Examples
    --------
    ::

        loader = ScenarioLoader("scenarios/", data_root="/data/FLAIR-HUB")
        scenario = loader.get("find_building_D006")
        all_ids = loader.list_ids()
    """

    def __init__(
        self,
        scenarios_dir: str | Path,
        data_root: Optional[str | Path] = None,
    ) -> None:
        self._scenarios_dir = Path(scenarios_dir).resolve()
        self._data_root = Path(data_root).resolve() if data_root else Path.cwd()
        self._cache: Dict[str, Scenario] = {}

        if not self._scenarios_dir.is_dir():
            raise FileNotFoundError(
                f"Scenarios directory not found: {self._scenarios_dir}"
            )

        logger.info(
            "ScenarioLoader: dir=%s, data_root=%s",
            self._scenarios_dir,
            self._data_root,
        )

    # ---------------------------------------------------------------- public

    def list_ids(self) -> List[str]:
        """Return a sorted list of available scenario IDs.

        The scenario ID is the YAML filename stem (without extension).
        """
        ids = []
        for path in self._scenarios_dir.iterdir():
            if path.suffix in (".yaml", ".yml") and path.is_file():
                ids.append(path.stem)
        return sorted(ids)

    def list_scenarios(self) -> List[Scenario]:
        """Load and return all scenarios."""
        return [self.get(sid) for sid in self.list_ids()]

    def get(self, scenario_id: str) -> Scenario:
        """Load a scenario by its ID.

        Parameters
        ----------
        scenario_id : str
            The scenario ID (YAML filename stem).

        Returns
        -------
        Scenario

        Raises
        ------
        FileNotFoundError
            If no YAML file matches the ID.
        ValueError
            If the YAML is malformed.
        """
        if scenario_id in self._cache:
            return self._cache[scenario_id]

        path = self._find_file(scenario_id)
        scenario = self._load_file(path)

        # Sanity: the file's scenario_id should match the filename stem.
        if scenario.scenario_id != scenario_id:
            logger.warning(
                "Scenario file '%s' has scenario_id='%s' (expected '%s'). "
                "Using file-based ID.",
                path.name,
                scenario.scenario_id,
                scenario_id,
            )

        self._cache[scenario_id] = scenario
        return scenario

    def resolve_data_dir(self, scenario: Scenario) -> Path:
        """Resolve the scenario's ``data_dir`` to an absolute path.

        If the scenario's ``data_dir`` is relative, it is resolved
        against the loader's ``data_root``.

        Parameters
        ----------
        scenario : Scenario
            The scenario to resolve.

        Returns
        -------
        Path
            Absolute path to the data directory.
        """
        raw = Path(scenario.dataset.data_dir)
        if raw.is_absolute():
            return raw
        return (self._data_root / raw).resolve()

    @property
    def scenarios_dir(self) -> Path:
        """The directory where scenario YAML files are stored."""
        return self._scenarios_dir

    @property
    def data_root(self) -> Path:
        """The root directory for resolving relative data paths."""
        return self._data_root

    # ---------------------------------------------------------------- internal

    def _find_file(self, scenario_id: str) -> Path:
        """Find the YAML file for a scenario ID."""
        for ext in (".yaml", ".yml"):
            candidate = self._scenarios_dir / f"{scenario_id}{ext}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No scenario file found for ID '{scenario_id}' in {self._scenarios_dir}"
        )

    def _load_file(self, path: Path) -> Scenario:
        """Load and parse a single YAML file."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for scenario support. "
                "Install it with: pip install pyyaml"
            ) from exc

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping in {path}, got {type(data)}")

        return _parse_scenario(data, source=str(path))

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        n = len(self.list_ids())
        return (
            f"ScenarioLoader(dir={self._scenarios_dir}, "
            f"data_root={self._data_root}, "
            f"scenarios={n})"
        )
