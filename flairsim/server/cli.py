"""
CLI entry point for the FlairSim HTTP server.

Usage::

    # Free flight (single or multi-modality)
    flairsim-server --data-dir path/to/D004-2021_AERIAL_RGBI
    flairsim-server --data-dir path/to/D006-2020 --port 8080 --roi UU-S2-1

    # Flat FLAIR-HUB layout with --domain
    flairsim-server --data-dir path/to/FLAIR-HUB --domain D006-2020

    # Auto-download from HuggingFace (no --data-dir needed)
    flairsim-server --domain D006-2020
    flairsim-server --domain D006-2020 --modalities AERIAL_RGBI DEM_ELEV

    # Scenario mode (--data-dir not needed, resolved from --data-root + scenario YAML)
    flairsim-server --data-root path/to/FLAIR-HUB --scenarios-dir scenarios/ --scenario find_target_D006
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Optional


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and launch the server."""
    parser = argparse.ArgumentParser(
        prog="flairsim-server",
        description="Launch the FlairSim drone simulator as a local HTTP server.",
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Path to FLAIR-HUB data directory (single modality like "
            "D004-2021_AERIAL_RGBI, or parent directory for multi-modality). "
            "Required for free flight unless --domain is used for auto-download."
        ),
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

    # --- Scenario options ---
    parser.add_argument(
        "--scenarios-dir",
        default=None,
        help="Directory containing scenario YAML files.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Root directory for resolving relative data_dir paths in "
            "scenarios. Defaults to current working directory."
        ),
    )
    parser.add_argument(
        "--domain",
        default=None,
        help=(
            "FLAIR-HUB domain prefix (e.g. D006-2020). When --data-dir "
            "is omitted, triggers automatic download from HuggingFace. "
            "When --data-dir is provided, filters modality discovery to "
            "this domain only."
        ),
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help=(
            "Modalities to download from HuggingFace (only used with "
            "auto-download, i.e. --domain without --data-dir). "
            "Default: AERIAL_RGBI. "
            "Example: --modalities AERIAL_RGBI DEM_ELEV"
        ),
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario ID to load at startup (requires --scenarios-dir).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Enable NxN grid overlay on images (e.g. --grid 4 for a 4x4 grid). "
            "Can be overridden per-request via the ?grid=N query parameter."
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

    from ..core.scenario import ScenarioLoader
    from ..drone.camera import CameraConfig
    from ..drone.drone import DroneConfig
    from .app import create_app

    drone_config = DroneConfig(default_altitude=args.altitude)
    camera_config = CameraConfig(fov_deg=args.fov, image_size=args.image_size)

    # --- Build scenario loader ---
    scenario_loader = None
    if args.scenarios_dir:
        scenario_loader = ScenarioLoader(
            scenarios_dir=args.scenarios_dir,
            data_root=args.data_root,
        )
        log.info(
            "Scenario loader: %s (%d scenarios)",
            args.scenarios_dir,
            len(scenario_loader.list_ids()),
        )

    if args.scenario and not scenario_loader:
        print(
            "Error: --scenario requires --scenarios-dir to be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Auto-download from HuggingFace if --domain without --data-dir ---
    downloader: Optional["HubDownloader"] = None  # noqa: F821
    if not args.data_dir and args.domain and not args.scenario:
        from ..data.downloader import HubDownloader

        modalities = args.modalities or ["AERIAL_RGBI"]
        downloader = HubDownloader(
            domain=args.domain,
            modalities=modalities,
        )

        log.info(
            "Auto-downloading domain %s modalities %s from HuggingFace ...",
            args.domain,
            modalities,
        )
        try:
            downloader.download()
        except Exception:
            log.exception("Download failed. Cleaning up ...")
            downloader.cleanup()
            sys.exit(1)

        effective_data_dir = str(downloader.data_dir)
        log.info("Data downloaded to: %s", effective_data_dir)

    elif not args.data_dir and not args.scenario:
        # No --data-dir, no --domain, no --scenario: error.
        parser.error(
            "--data-dir is required for free flight. "
            "Use --domain to auto-download from HuggingFace, "
            "or use --scenario to load from a scenario file."
        )
    else:
        # --data-dir provided, or --scenario mode.
        # In scenario mode without --data-dir, use a placeholder; create_app
        # will override it from the scenario's dataset.data_dir.
        effective_data_dir = args.data_dir or ""

    # Warn if --modalities is used with --data-dir (it's ignored).
    if args.modalities and args.data_dir:
        log.warning(
            "--modalities is only used with auto-download (--domain without "
            "--data-dir). Ignoring --modalities since --data-dir is set."
        )

    app = create_app(
        data_dir=effective_data_dir,
        roi=args.roi,
        max_steps=args.max_steps,
        drone_config=drone_config,
        camera_config=camera_config,
        preload_tiles=not args.no_preload,
        scenario_loader=scenario_loader,
        scenario_id=args.scenario,
        grid=args.grid,
        domain=args.domain,
        downloader=downloader,
    )

    # --- Register signal handlers for cleanup ---
    def _shutdown_handler(signum, frame):
        """Handle SIGINT/SIGTERM: clean up downloaded data, then exit."""
        sig_name = signal.Signals(signum).name
        log.info("Received %s, shutting down ...", sig_name)
        if downloader is not None:
            downloader.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    log.info("Starting FlairSim server on %s:%d", args.host, args.port)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        # Belt-and-suspenders: also clean up when uvicorn exits normally
        # (e.g. via SIGINT caught by uvicorn internally before our handler).
        if downloader is not None:
            downloader.cleanup()


if __name__ == "__main__":
    main()
