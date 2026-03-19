"""
Flight telemetry recording.

The :class:`FlightLog` accumulates :class:`TelemetryRecord` entries at
each simulation step, providing a complete trajectory history that can
be serialised, analysed, or replayed.

This is essential for:

* **Evaluation** -- computing metrics like path efficiency, area
  coverage, and time-to-target.
* **Debugging** -- replaying an agent's trajectory to understand its
  decision process.
* **Visualisation** -- drawing the flight path on the minimap.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-step record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """Snapshot of drone state and action at a single simulation step.

    Attributes
    ----------
    step : int
        0-based step index within the episode.
    x, y, z : float
        Drone position after the action was applied.
    dx, dy, dz : float
        Displacement that was actually applied (may differ from the
        agent's request if clamping occurred).
    ground_footprint : float
        Side length of the camera's ground footprint (metres).
    was_clipped : bool
        Whether any component of the displacement was clipped.
    metadata : dict
        Arbitrary extra data (e.g. agent confidence, label at position).
    reason : str or None
        AI agent's reasoning for this step.  ``None`` for human players.
    """

    step: int
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    ground_footprint: float
    was_clipped: bool
    metadata: Dict[str, object] = field(default_factory=dict)
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Flight log
# ---------------------------------------------------------------------------


class FlightLog:
    """Accumulates telemetry records over an episode.

    Provides summary statistics and export utilities.

    Examples
    --------
    >>> log = FlightLog()
    >>> log.append(TelemetryRecord(step=0, x=100, y=200, z=80,
    ...     dx=0, dy=0, dz=0, ground_footprint=160, was_clipped=False))
    >>> len(log)
    1
    >>> log.total_distance
    0.0
    """

    def __init__(self) -> None:
        self._records: List[TelemetryRecord] = []

    # ---------------------------------------------------------------- mutate

    def append(self, record: TelemetryRecord) -> None:
        """Add a telemetry record."""
        self._records.append(record)

    def clear(self) -> None:
        """Remove all records (for episode reset)."""
        self._records.clear()

    # ---------------------------------------------------------------- access

    @property
    def records(self) -> Sequence[TelemetryRecord]:
        """All recorded telemetry entries (read-only view)."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TelemetryRecord:
        return self._records[index]

    # ---------------------------------------------------------------- stats

    @property
    def total_distance(self) -> float:
        """Total horizontal distance travelled (metres).

        Computed from the sum of actual horizontal displacements.
        """
        import math

        return sum(math.sqrt(r.dx**2 + r.dy**2) for r in self._records)

    @property
    def total_steps(self) -> int:
        """Number of steps recorded."""
        return len(self._records)

    @property
    def altitude_range(self) -> Optional[tuple[float, float]]:
        """``(min_z, max_z)`` observed during the episode, or ``None``."""
        if not self._records:
            return None
        zs = [r.z for r in self._records]
        return (min(zs), max(zs))

    @property
    def trajectory_2d(self) -> List[tuple[float, float]]:
        """Ordered list of ``(x, y)`` positions for path plotting."""
        return [(r.x, r.y) for r in self._records]

    @property
    def clips_count(self) -> int:
        """Number of steps where the displacement was clipped."""
        return sum(1 for r in self._records if r.was_clipped)

    def bounding_box(self) -> Optional[tuple[float, float, float, float]]:
        """``(x_min, y_min, x_max, y_max)`` of all visited positions.

        Returns ``None`` if the log is empty.
        """
        if not self._records:
            return None
        xs = [r.x for r in self._records]
        ys = [r.y for r in self._records]
        return (min(xs), min(ys), max(xs), max(ys))

    # ---------------------------------------------------------------- export

    def to_dicts(self) -> List[Dict[str, object]]:
        """Convert all records to a list of plain dictionaries.

        Useful for JSON serialisation or DataFrame construction.
        """
        return [asdict(r) for r in self._records]

    def to_csv(self, filepath: str | Path) -> None:
        """Write the telemetry log to a CSV file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.  Parent directories must exist.
        """
        filepath = Path(filepath)
        if not self._records:
            logger.warning("No records to write to %s", filepath)
            return

        fieldnames = [
            "step",
            "x",
            "y",
            "z",
            "dx",
            "dy",
            "dz",
            "ground_footprint",
            "was_clipped",
            "reason",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in self._records:
                row = asdict(record)
                # Remove metadata from CSV (it's a nested dict).
                row.pop("metadata", None)
                writer.writerow(row)

        logger.info("Wrote %d telemetry records to %s", len(self._records), filepath)

    def to_json(self, filepath: str | Path, indent: int = 2) -> None:
        """Write the telemetry log to a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        indent : int
            JSON indentation level (default 2).
        """
        filepath = Path(filepath)
        data = {
            "total_steps": self.total_steps,
            "total_distance_m": round(self.total_distance, 2),
            "clips_count": self.clips_count,
            "altitude_range": self.altitude_range,
            "records": self.to_dicts(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)

        logger.info("Wrote telemetry JSON to %s", filepath)

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        return (
            f"FlightLog(steps={self.total_steps}, "
            f"distance={self.total_distance:.1f}m, "
            f"clips={self.clips_count})"
        )
