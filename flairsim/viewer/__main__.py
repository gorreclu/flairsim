"""
CLI entry point for the FlairSim viewer.

Three modes are available:

* **local** (default) -- start a local simulator and fly manually::

      uv run python -m flairsim.viewer --data-dir path/to/D004

* **observe** -- connect to a running server and watch via SSE::

      uv run python -m flairsim.viewer --mode observe --server-url http://localhost:8000

* **fly** -- connect to a running server and pilot via HTTP::

      uv run python -m flairsim.viewer --mode fly --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and launch the viewer."""
    parser = argparse.ArgumentParser(
        prog="flairsim-viewer",
        description=(
            "Launch the FlairSim viewer.  Supports local manual flight, "
            "remote observation (SSE), and remote piloting (HTTP)."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["local", "observe", "fly"],
        default="local",
        help=(
            "Viewer mode: 'local' runs a local simulator, "
            "'observe' watches a remote server via SSE, "
            "'fly' pilots a remote server via HTTP (default: local)."
        ),
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help=("Server URL for observe/fly modes (default: http://localhost:8000)."),
    )

    # --- Local mode options ---
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Path to FLAIR-HUB data directory (required for local mode, "
            "ignored in remote modes)."
        ),
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="ROI to load (local mode). Omit to auto-select the largest.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (local mode, default: 500).",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=100.0,
        help="Default starting altitude in metres (local mode, default: 100).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=500,
        help="Camera output resolution in pixels (local mode, default: 500).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Camera field of view in degrees (local mode, default: 90).",
    )

    # --- Common options ---
    parser.add_argument(
        "--window-size",
        type=int,
        default=800,
        help="Viewer window size in pixels (default: 800).",
    )
    parser.add_argument(
        "--move-step",
        type=float,
        default=20.0,
        help="Movement step size per key press in metres (default: 20).",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Do not preload tiles into memory (local mode only).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Enable NxN grid overlay on the viewer (e.g. --grid 4 for a 4x4 grid). "
            "Press G to toggle the grid on/off during flight."
        ),
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
    log = logging.getLogger(__name__)

    # --- Check pygame ---
    try:
        import pygame  # noqa: F401
    except ImportError:
        print(
            "Error: pygame is not installed. "
            "Install the viewer extra: uv sync --extra viewer",
            file=sys.stderr,
        )
        sys.exit(1)

    from .viewer import FlairViewer, ViewerConfig

    viewer_config = ViewerConfig(
        window_width=args.window_size,
        window_height=args.window_size,
        move_step=args.move_step,
    )

    # ----- Local mode -----
    if args.mode == "local":
        if args.data_dir is None:
            parser.error("--data-dir is required for local mode.")

        from ..core.simulator import FlairSimulator, SimulatorConfig
        from ..drone.camera import CameraConfig
        from ..drone.drone import DroneConfig

        drone_config = DroneConfig(default_altitude=args.altitude)
        camera_config = CameraConfig(fov_deg=args.fov, image_size=args.image_size)

        sim_config = SimulatorConfig(
            drone_config=drone_config,
            camera_config=camera_config,
            max_steps=args.max_steps,
            roi=args.roi,
            preload_tiles=not args.no_preload,
        )

        log.info("Loading simulator from %s ...", args.data_dir)
        sim = FlairSimulator(data_dir=args.data_dir, config=sim_config)

        log.info("Starting local viewer (%dx%d)", args.window_size, args.window_size)
        viewer = FlairViewer(config=viewer_config, grid=args.grid)
        viewer.run_manual(sim)

    # ----- Observe mode -----
    elif args.mode == "observe":
        # Check httpx is available.
        try:
            import httpx  # noqa: F401
        except ImportError:
            print(
                "Error: httpx is not installed. "
                "Install dev dependencies: uv sync --dev",
                file=sys.stderr,
            )
            sys.exit(1)

        log.info("Observing server at %s", args.server_url)
        viewer = FlairViewer(config=viewer_config, grid=args.grid)
        viewer.run_remote_observe(args.server_url)

    # ----- Fly mode -----
    elif args.mode == "fly":
        try:
            import httpx  # noqa: F401
        except ImportError:
            print(
                "Error: httpx is not installed. "
                "Install dev dependencies: uv sync --dev",
                file=sys.stderr,
            )
            sys.exit(1)

        log.info("Flying via server at %s", args.server_url)
        viewer = FlairViewer(config=viewer_config, grid=args.grid)
        viewer.run_remote_fly(args.server_url)


if __name__ == "__main__":
    main()
