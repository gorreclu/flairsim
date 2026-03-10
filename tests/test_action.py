"""Unit tests for flairsim.core.action."""

import pytest

from flairsim.core.action import Action, ActionType


# ---------------------------------------------------------------------------
# ActionType
# ---------------------------------------------------------------------------


class TestActionType:
    """Tests for ActionType enum."""

    def test_members(self):
        assert ActionType.MOVE.value == "move"
        assert ActionType.FOUND.value == "found"
        assert ActionType.STOP.value == "stop"

    def test_count(self):
        assert len(ActionType) == 3


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class TestAction:
    """Tests for Action dataclass."""

    def test_defaults(self):
        a = Action()
        assert a.dx == 0.0
        assert a.dy == 0.0
        assert a.dz == 0.0
        assert a.action_type == ActionType.MOVE

    def test_custom(self):
        a = Action(dx=10, dy=-5, dz=3, action_type=ActionType.FOUND)
        assert a.dx == 10.0
        assert a.dy == -5.0
        assert a.dz == 3.0
        assert a.action_type == ActionType.FOUND

    def test_frozen(self):
        a = Action()
        with pytest.raises(AttributeError):
            a.dx = 5.0  # type: ignore[misc]

    # ---- Factory methods ----

    def test_move_factory(self):
        a = Action.move(dx=10, dy=20, dz=-5)
        assert a.action_type == ActionType.MOVE
        assert a.dx == 10.0
        assert a.dy == 20.0
        assert a.dz == -5.0

    def test_move_factory_defaults(self):
        a = Action.move()
        assert a.dx == 0.0
        assert a.dy == 0.0
        assert a.dz == 0.0
        assert a.action_type == ActionType.MOVE

    def test_found_factory(self):
        a = Action.found()
        assert a.action_type == ActionType.FOUND
        assert a.dx == 0.0
        assert a.dy == 0.0
        assert a.dz == 0.0

    def test_found_factory_with_movement(self):
        a = Action.found(dx=1, dy=2, dz=3)
        assert a.action_type == ActionType.FOUND
        assert a.dx == 1.0

    def test_stop_factory(self):
        a = Action.stop()
        assert a.action_type == ActionType.STOP
        assert a.dx == 0.0
        assert a.dy == 0.0
        assert a.dz == 0.0

    # ---- Repr ----

    def test_repr_move(self):
        a = Action.move(dx=10, dy=5)
        r = repr(a)
        assert "MOVE" in r
        assert "10" in r

    def test_repr_found(self):
        a = Action.found()
        r = repr(a)
        assert "FOUND" in r

    def test_repr_stop(self):
        a = Action.stop()
        assert "STOP" in repr(a)
