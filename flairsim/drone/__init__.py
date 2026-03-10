"""
Drone sub-package -- state, physics, camera model, and telemetry.

This package models a simplified quadrotor UAV flying over the FLAIR-HUB
map surface.  The drone has three degrees of freedom (east, north, altitude)
and always looks straight down (nadir / top-down view).

Modules
-------
drone
    Core :class:`Drone` class and its state/configuration dataclasses.
camera
    :class:`CameraModel` linking altitude to ground footprint and image
    capture.
telemetry
    Per-step :class:`TelemetryRecord` and cumulative :class:`FlightLog`.
"""
