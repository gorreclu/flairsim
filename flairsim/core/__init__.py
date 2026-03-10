"""
Core sub-package -- simulator engine, actions, and observations.

This package provides the main :class:`FlairSimulator` class and the
:class:`Action` / :class:`Observation` data structures that define
the ``sim.step(action) -> observation`` API.

Modules
-------
action
    :class:`Action` dataclass for agent commands.
observation
    :class:`Observation` dataclass returned by the simulator.
simulator
    :class:`FlairSimulator` -- the central simulation loop.
"""
