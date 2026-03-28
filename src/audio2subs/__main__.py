"""CLI entry point for Audio2Subs."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from audio2subs.config import ServiceConfig
from audio2subs.service import SubtitleService


def setup_logging(config: ServiceConfig, log_dir: str) -> None:
    """Configure logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if config.log_to_file:
        log_path = os.path.join(log_dir, 'subtitle_service.log')
        handlers.append(logging.FileHandler(log_path, mode='w', encoding='utf-8'))

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        handlers=handlers
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Subtitle Service for MPV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--socket",
        required=True,
        help="MPV IPC socket path"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only mode"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Build config
    config = ServiceConfig.from_env()
    config.socket_path = args.socket
    config.log_level = args.log_level

    if args.cpu:
        config.transcription.device = "cpu"

    # Setup logging
    if sys.platform == "win32":
        log_dir = Path(os.environ.get("APPDATA", os.path.expanduser("~"))) / "MPV-Audio2Subs"
    else:
        log_dir = Path(os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))) / "mpv-audio2subs"

    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config, str(log_dir))

    # Run service
    try:
        service = SubtitleService(config)
        service.run()
        return 0
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
