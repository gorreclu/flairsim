"""
Viewer sub-package -- real-time pygame visualisation.

This package provides a desktop GUI for watching the drone fly over
the FLAIR-HUB map in real time.  Three modes are supported:

* **Local manual** -- ``run_manual(simulator)`` for direct control
  of a local :class:`~flairsim.core.simulator.FlairSimulator`.
* **Remote observe** -- ``run_remote_observe(url)`` to watch a
  running server via SSE.
* **Remote fly** -- ``run_remote_fly(url)`` to pilot a running
  server via HTTP.

Modules
-------
viewer
    Main :class:`FlairViewer` class managing the pygame window.
hud
    :class:`HUD` overlay with flight telemetry readouts.
minimap
    :class:`Minimap` inset showing the drone's position on the full ROI.
remote
    :class:`ViewerObservation` adapter bridging local and remote data.
"""
