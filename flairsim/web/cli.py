"""
CLI entry point for the FlairSim web benchmark platform (orchestrator).

Usage::

    # Minimal -- auto-download from HuggingFace
    flairsim-web --scenarios-dir scenarios/

    # With local FLAIR-HUB data
    flairsim-web --scenarios-dir scenarios/ --data-root /path/to/FLAIR-HUB

    # Custom port and leaderboard path
    flairsim-web --scenarios-dir scenarios/ --port 8080 \
                 --leaderboard-db data/leaderboard.db
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and launch the orchestrator."""
    parser = argparse.ArgumentParser(
        prog="flairsim-web",
        description=(
            "Launch the FlairSim web benchmark platform. "
            "Serves a browser UI and manages simulator sessions."
        ),
    )

    parser.add_argument(
        "--scenarios-dir",
        required=True,
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
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080).",
    )
    parser.add_argument(
        "--leaderboard-db",
        default="data/leaderboard.db",
        help="Path to the SQLite leaderboard database (default: data/leaderboard.db).",
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
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "Error: uvicorn is not installed. "
            "Install the server extra: uv sync --extra server",
            file=sys.stderr,
        )
        sys.exit(1)

    from .app import create_web_app

    app = create_web_app(
        scenarios_dir=args.scenarios_dir,
        data_root=args.data_root,
        leaderboard_db=args.leaderboard_db,
    )

    log.info(
        "Starting FlairSim web platform on %s:%d",
        args.host,
        args.port,
    )
    log.info("Scenarios: %s", args.scenarios_dir)
    log.info("Leaderboard DB: %s", args.leaderboard_db)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
