"""
FlairSim HTTP server -- REST API for drone simulation.

Exposes the :class:`~flairsim.core.simulator.FlairSimulator` as a local
HTTP service so that external programs (VLM agents, notebooks, other
languages) can pilot the drone via standard HTTP requests.

Launch via CLI::

    uv run python -m flairsim.server --data-dir path/to/D004-2021_AERIAL_RGBI

Or programmatically::

    from flairsim.server import create_app

    app = create_app(data_dir="path/to/data")
"""

from .app import create_app

__all__ = ["create_app"]
