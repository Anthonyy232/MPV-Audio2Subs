"""End-to-end test: run TranscriptionEngine directly on a video file."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from audio2subs.config import ServiceConfig
from audio2subs.engine import TranscriptionEngine
from audio2subs.transcription import CohereTranscriber

VIDEO_PATH = r"C:\Users\Antho\Videos\2026-02-23 14-00-40.mp4"


def _get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("test_e2e.log", mode="w", encoding="utf-8"),
        ],
    )

    config = ServiceConfig()
    # Use default chunk duration (30 s)

    transcriber = CohereTranscriber(config.transcription, config.subtitle)

    # Load model up front so timing is clear
    logging.info("Loading ASR model...")
    transcriber.load()
    logging.info("ASR model loaded.")

    duration = _get_duration(VIDEO_PATH)
    logging.info(f"Video duration: {duration:.1f}s")

    subtitle_path = os.path.splitext(VIDEO_PATH)[0] + ".ai.ass"

    def on_progress(done: int, total: int) -> None:
        pct = int(done / total * 100) if total else 0
        logging.info(f"Progress: {done}/{total} chunks ({pct}%)")

    engine = TranscriptionEngine(
        video_path=VIDEO_PATH,
        duration=duration,
        transcriber=transcriber,
        config=config,
        video_width=1920,
        video_height=1080,
        on_progress=on_progress,
    )

    logging.info("Starting engine...")
    engine.start()

    # Kick off the whole file immediately by simulating time updates
    # across the full duration so all chunks get queued at idle priority.
    # The engine's idle queue handles the rest automatically.

    try:
        while not engine.is_finished:
            time.sleep(5)
        logging.info("Engine finished.")
    except KeyboardInterrupt:
        logging.info("Interrupted — stopping engine.")
        engine.stop()
        return

    engine.stop()

    # Show a sample of the output
    if os.path.exists(subtitle_path):
        with open(subtitle_path, encoding="utf-8") as f:
            lines = f.readlines()
        dialogue = [l.strip() for l in lines if l.startswith("Dialogue:")]
        logging.info(f"Total subtitle lines: {len(dialogue)}")
        logging.info("=== First 20 lines ===")
        for l in dialogue[:20]:
            logging.info(l)
        logging.info("=== Last 10 lines ===")
        for l in dialogue[-10:]:
            logging.info(l)
    else:
        logging.warning(f"Subtitle file not found: {subtitle_path}")


if __name__ == "__main__":
    main()
