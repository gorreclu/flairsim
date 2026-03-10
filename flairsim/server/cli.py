"""
CLI entry point for the FlairSim HTTP server.

Usage::

    flairsim-server --data-dir path/to/D004-2021_AERIAL_RGBI
    flairsim-server --data-dir path/to/data --port 8080 --roi AA-S1-32
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and launch the server."""
    parser = argparse.ArgumentParser(
        prog="flairsim-server",
        description="Launch the FlairSim drone simulator as a local HTTP server.",
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to FLAIR-HUB data directory (e.g. D004-2021_AERIAL_RGBI).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="ROI to load. Omit to auto-select the largest.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500).",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=100.0,
        help="Default starting altitude in metres (default: 100).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=500,
        help="Camera output resolution in pixels (default: 500).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Camera field of view in degrees (default: 90).",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Do not preload tiles into memory (saves RAM, slower runtime).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    # --- Logging ---
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Lazy imports (so --help is fast) ---
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed. "
            "Install the server extra: uv sync --extra server",
            file=sys.stderr,
        )
        sys.exit(1)

    from ..drone.camera import CameraConfig
    from ..drone.drone import DroneConfig
    from .app import create_app

    drone_config = DroneConfig(default_altitude=args.altitude)
    camera_config = CameraConfig(fov_deg=args.fov, image_size=args.image_size)

    app = create_app(
        data_dir=args.data_dir,
        roi=args.roi,
        max_steps=args.max_steps,
        drone_config=drone_config,
        camera_config=camera_config,
        preload_tiles=not args.no_preload,
    )

    logging.getLogger(__name__).info(
        "Starting FlairSim server on %s:%d", args.host, args.port
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
