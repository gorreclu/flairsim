"""
Agent action definitions.

An :class:`Action` is the interface through which an agent (human, random,
or VLM) communicates its intent to the simulator.  Following the FlySearch
benchmark, actions are relative displacements in metres.

Action types
------------
* **MOVE** -- the default; the drone moves by ``(dx, dy, dz)`` metres.
* **FOUND** -- the agent declares that it has found the target.  The
  simulator checks visibility / proximity and returns success or failure.
  Movement is still applied (the agent may adjust position before calling
  FOUND).
* **STOP** -- the agent voluntarily ends the episode without declaring
  a target found.  Useful for "give up" or "exploration complete" signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique


@unique
class ActionType(Enum):
    """Type of agent action."""

    MOVE = "move"
    FOUND = "found"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class Action:
    """A single agent command.

    Parameters
    ----------
    dx : float
        Eastward displacement in metres (default 0).
    dy : float
        Northward displacement in metres (default 0).
    dz : float
        Upward displacement in metres (default 0).
    action_type : ActionType
        The type of action (default ``MOVE``).

    Examples
    --------
    Move 10 m east, 5 m north, descend 3 m:

    >>> Action(dx=10.0, dy=5.0, dz=-3.0)
    Action(dx=10.0, dy=5.0, dz=-3.0, type=MOVE)

    Declare target found (with optional positional adjustment):

    >>> Action(dx=0.0, dy=0.0, dz=0.0, action_type=ActionType.FOUND)
    Action(dx=0.0, dy=0.0, dz=0.0, type=FOUND)
    """

    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    action_type: ActionType = ActionType.MOVE

    # ---------------------------------------------------------------- factory

    @classmethod
    def move(cls, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Action":
        """Create a MOVE action."""
        return cls(dx=dx, dy=dy, dz=dz, action_type=ActionType.MOVE)

    @classmethod
    def found(cls, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Action":
        """Create a FOUND action (declare target located)."""
        return cls(dx=dx, dy=dy, dz=dz, action_type=ActionType.FOUND)

    @classmethod
    def stop(cls) -> "Action":
        """Create a STOP action (end episode without finding target)."""
        return cls(dx=0.0, dy=0.0, dz=0.0, action_type=ActionType.STOP)

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        return (
            f"Action(dx={self.dx}, dy={self.dy}, dz={self.dz}, "
            f"type={self.action_type.name})"
        )
