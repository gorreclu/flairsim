"""Unit tests for flairsim.drone.telemetry."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from flairsim.drone.telemetry import FlightLog, TelemetryRecord


# ---------------------------------------------------------------------------
# TelemetryRecord
# ---------------------------------------------------------------------------


class TestTelemetryRecord:
    """Tests for TelemetryRecord."""

    def test_creation(self):
        r = TelemetryRecord(
            step=0,
            x=100,
            y=200,
            z=80,
            dx=0,
            dy=0,
            dz=0,
            ground_footprint=160,
            was_clipped=False,
        )
        assert r.step == 0
        assert r.x == 100
        assert r.z == 80
        assert r.ground_footprint == 160
        assert not r.was_clipped
        assert r.metadata == {}

    def test_with_metadata(self):
        r = TelemetryRecord(
            step=1,
            x=0,
            y=0,
            z=100,
            dx=10,
            dy=5,
            dz=0,
            ground_footprint=200,
            was_clipped=False,
            metadata={"label": "building", "confidence": 0.9},
        )
        assert r.metadata["label"] == "building"
        assert r.metadata["confidence"] == 0.9

    def test_frozen(self):
        r = TelemetryRecord(
            step=0,
            x=0,
            y=0,
            z=100,
            dx=0,
            dy=0,
            dz=0,
            ground_footprint=200,
            was_clipped=False,
        )
        with pytest.raises(AttributeError):
            r.step = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FlightLog
# ---------------------------------------------------------------------------


def _make_record(step, x, y, z, dx=0, dy=0, dz=0, clipped=False):
    """Helper to create a TelemetryRecord."""
    return TelemetryRecord(
        step=step,
        x=x,
        y=y,
        z=z,
        dx=dx,
        dy=dy,
        dz=dz,
        ground_footprint=z * 2,
        was_clipped=clipped,
    )


class TestFlightLog:
    """Tests for FlightLog."""

    def test_empty_log(self):
        log = FlightLog()
        assert len(log) == 0
        assert log.total_steps == 0
        assert log.total_distance == 0.0
        assert log.altitude_range is None
        assert log.bounding_box() is None
        assert log.trajectory_2d == []
        assert log.clips_count == 0

    def test_append_and_len(self):
        log = FlightLog()
        log.append(_make_record(0, 0, 0, 100))
        assert len(log) == 1
        log.append(_make_record(1, 10, 0, 100, dx=10))
        assert len(log) == 2

    def test_getitem(self):
        log = FlightLog()
        r = _make_record(0, 42, 99, 80)
        log.append(r)
        assert log[0] is r

    def test_clear(self):
        log = FlightLog()
        log.append(_make_record(0, 0, 0, 100))
        log.append(_make_record(1, 10, 0, 100))
        log.clear()
        assert len(log) == 0

    def test_total_distance(self):
        log = FlightLog()
        log.append(_make_record(0, 0, 0, 100, dx=0, dy=0))
        log.append(_make_record(1, 3, 4, 100, dx=3, dy=4))  # dist=5
        log.append(_make_record(2, 3, 4, 100, dx=0, dy=0))  # dist=0
        assert log.total_distance == pytest.approx(5.0)

    def test_total_distance_multiple(self):
        log = FlightLog()
        log.append(_make_record(0, 0, 0, 100))
        log.append(_make_record(1, 10, 0, 100, dx=10, dy=0))  # 10
        log.append(_make_record(2, 10, 20, 100, dx=0, dy=20))  # 20
        assert log.total_distance == pytest.approx(30.0)

    def test_altitude_range(self):
        log = FlightLog()
        log.append(_make_record(0, 0, 0, 50))
        log.append(_make_record(1, 0, 0, 200))
        log.append(_make_record(2, 0, 0, 100))
        assert log.altitude_range == (50, 200)

    def test_trajectory_2d(self):
        log = FlightLog()
        log.append(_make_record(0, 10, 20, 100))
        log.append(_make_record(1, 30, 40, 100))
        assert log.trajectory_2d == [(10, 20), (30, 40)]

    def test_clips_count(self):
        log = FlightLog()
        log.append(_make_record(0, 0, 0, 100, clipped=False))
        log.append(_make_record(1, 10, 0, 100, clipped=True))
        log.append(_make_record(2, 20, 0, 100, clipped=True))
        log.append(_make_record(3, 20, 0, 100, clipped=False))
        assert log.clips_count == 2

    def test_bounding_box(self):
        log = FlightLog()
        log.append(_make_record(0, 10, 20, 100))
        log.append(_make_record(1, 50, 5, 100))
        log.append(_make_record(2, 30, 40, 100))
        bb = log.bounding_box()
        assert bb == (10, 5, 50, 40)

    def test_to_dicts(self):
        log = FlightLog()
        log.append(_make_record(0, 10, 20, 80))
        dicts = log.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["step"] == 0
        assert dicts[0]["x"] == 10
        assert dicts[0]["z"] == 80

    # ---- Export ----

    def test_to_csv(self):
        log = FlightLog()
        log.append(_make_record(0, 10, 20, 80))
        log.append(_make_record(1, 30, 40, 100, dx=20, dy=20))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        log.to_csv(csv_path)

        # Read back and verify.
        content = Path(csv_path).read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 records
        assert "step" in lines[0]
        assert "10" in lines[1]  # x value

        Path(csv_path).unlink()

    def test_to_csv_empty(self):
        """Empty log should not crash."""
        log = FlightLog()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name
        log.to_csv(csv_path)
        Path(csv_path).unlink()

    def test_to_json(self):
        log = FlightLog()
        log.append(_make_record(0, 10, 20, 80))
        log.append(_make_record(1, 30, 40, 100, dx=20, dy=20, clipped=True))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        log.to_json(json_path)

        data = json.loads(Path(json_path).read_text())
        assert data["total_steps"] == 2
        assert data["clips_count"] == 1
        assert len(data["records"]) == 2
        assert data["records"][0]["x"] == 10

        Path(json_path).unlink()

    def test_repr(self):
        log = FlightLog()
        r = repr(log)
        assert "FlightLog" in r
        assert "steps=0" in r
