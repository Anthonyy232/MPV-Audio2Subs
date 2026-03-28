"""Transcription engine - coordinates audio extraction, transcription, and subtitle generation."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

from audio2subs.audio import AudioExtractor
from audio2subs.config import ServiceConfig
from audio2subs.subtitle import SubtitleWriter, generate_subtitle_path
from audio2subs.transcription.base import BaseTranscriber

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """One-shot transcription engine for a single video.

    Extracts the full audio, transcribes it in one pass via the Cohere +
    stable-ts pipeline, and writes the ASS subtitle file.
    """

    def __init__(
        self,
        video_path: str,
        duration: float,
        transcriber: BaseTranscriber,
        config: ServiceConfig,
        video_width: int | None = None,
        video_height: int | None = None,
        audio_track_id: int | None = None,
        on_subtitle_update: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ):
        self.video_path = video_path
        self.video_filename = os.path.basename(video_path)
        self.duration = duration
        self.transcriber = transcriber
        self.config = config
        self.on_progress = on_progress

        self.subtitle_path = generate_subtitle_path(video_path, config.subtitle.subtitle_suffix)

        self._audio = AudioExtractor(video_path, duration, audio_track_id)
        self._subtitle_writer = SubtitleWriter(
            self.subtitle_path,
            config.subtitle,
            video_width,
            video_height,
            on_update=on_subtitle_update,
        )

        self._stop_event = threading.Event()
        self._finished_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._log_prefix = f"[Engine:{self.video_filename[:20]}]"

    def __repr__(self) -> str:
        return f"TranscriptionEngine(video={self.video_filename}, finished={self.is_finished})"

    @property
    def is_finished(self) -> bool:
        return self._finished_event.is_set()

    def start(self) -> None:
        """Start audio extraction immediately, then kick off the worker."""
        logger.info(f"{self._log_prefix} Starting engine (duration={self.duration:.1f}s)")
        self._subtitle_writer.initialize()

        # Start FFmpeg now so audio extraction runs in parallel with model loading.
        self._audio.extract_streaming()

        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name=f"Engine-{self.video_filename[:10]}",
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the engine and clean up resources."""
        logger.info(f"{self._log_prefix} Stopping engine")
        self._stop_event.set()
        self._audio.close()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info(f"{self._log_prefix} Engine stopped")

    def _worker(self) -> None:
        """Worker thread: wait for model + audio → transcribe → write subtitles."""
        logger.info(f"{self._log_prefix} Worker started")
        try:
            # Wait for transcription model (interruptible).
            if not self.transcriber.is_loaded:
                logger.info(f"{self._log_prefix} Waiting for transcription model...")
                while not self.transcriber.is_loaded and not self._stop_event.is_set():
                    self.transcriber.wait_for_load(timeout=1.0)

            if self._stop_event.is_set() or not self.transcriber.is_loaded:
                return

            # Wait for audio extraction (started in start(), interruptible).
            if not self._audio.is_ready:
                logger.info(f"{self._log_prefix} Waiting for audio extraction...")
                while not self._audio.is_ready and not self._stop_event.is_set():
                    self._audio.wait_for_completion(timeout=1.0)

            if self._stop_event.is_set():
                return

            audio_data = self._audio.read_all()
            if not audio_data:
                logger.warning(f"{self._log_prefix} No audio data extracted")
                return

            # Transcribe entire file in one pass
            logger.info(
                f"{self._log_prefix} Transcribing {self.duration:.1f}s "
                f"({len(audio_data) // 1024}KB)..."
            )
            start_perf = time.perf_counter()
            result = self.transcriber.transcribe(audio_data)
            elapsed = time.perf_counter() - start_perf

            if self._stop_event.is_set():
                return

            if result.is_empty:
                logger.info(f"{self._log_prefix} No speech detected ({elapsed:.1f}s)")
            else:
                self._subtitle_writer.write_ass(result.ass_content)
                logger.info(f"{self._log_prefix} Done in {elapsed:.1f}s")

            if self.on_progress:
                self.on_progress(1, 1)

            logger.info(f"{self._log_prefix} Transcription complete!")

        except Exception as e:
            logger.error(f"{self._log_prefix} Worker error: {e}", exc_info=True)
        finally:
            self._finished_event.set()
            logger.info(f"{self._log_prefix} Worker stopped")
