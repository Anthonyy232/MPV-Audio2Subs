"""Audio extraction utilities using FFmpeg."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import BinaryIO

from audio2subs.exceptions import AudioExtractionError
from audio2subs.transcription.base import SAMPLE_RATE, CHANNELS

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extracts audio from video files using FFmpeg in a background thread."""

    def __init__(self, video_path: str, duration: float, audio_track_id: int | None = None):
        self.video_path = video_path
        self.duration = duration
        self.audio_track_id = audio_track_id

        self._temp_file_path: str | None = None
        self._file_handle: BinaryIO | None = None
        self._ffmpeg_process: subprocess.Popen | None = None
        self._closed = False
        self._extraction_complete = threading.Event()
        self._extraction_error: Exception | None = None

    @property
    def is_ready(self) -> bool:
        """Whether audio extraction is complete."""
        return self._extraction_complete.is_set()

    def extract_streaming(self) -> None:
        """Start audio extraction in a background thread."""
        self._create_temp_file()
        self._open_for_reading()
        thread = threading.Thread(
            target=self._extraction_worker,
            daemon=True,
            name="AudioExtractor",
        )
        thread.start()

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for extraction to complete. Returns True if done, False on timeout."""
        return self._extraction_complete.wait(timeout=timeout)

    def read_all(self) -> bytes | None:
        """Read the entire extracted audio as raw PCM bytes.

        Must be called after extraction is complete.
        """
        fh = self._file_handle  # snapshot to avoid race with close()
        if fh is None:
            return None
        try:
            fh.seek(0)
            return fh.read()
        except (IOError, ValueError) as e:
            logger.error(f"Failed to read audio: {e}")
            return None

    def close(self) -> None:
        """Kill FFmpeg if running and release all resources."""
        self._closed = True

        if self._ffmpeg_process is not None and self._ffmpeg_process.poll() is None:
            self._ffmpeg_process.kill()
            self._ffmpeg_process = None

        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None

        if self._temp_file_path and os.path.exists(self._temp_file_path):
            try:
                os.remove(self._temp_file_path)
                logger.debug(f"Cleaned up temp audio file: {self._temp_file_path}")
            except OSError as e:
                logger.warning(f"Failed to remove temp file: {e}")

        self._temp_file_path = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extraction_worker(self) -> None:
        try:
            self._run_ffmpeg()
        except Exception as e:
            self._extraction_error = e
            if not self._closed:
                logger.error(f"Audio extraction failed for {os.path.basename(self.video_path)}: {e}")
        finally:
            self._extraction_complete.set()

    def _create_temp_file(self) -> None:
        fd, self._temp_file_path = tempfile.mkstemp(suffix=".raw")
        os.close(fd)
        logger.debug(f"Created temp audio file: {self._temp_file_path}")

    def _open_for_reading(self) -> None:
        if not self._temp_file_path:
            raise AudioExtractionError("No audio file to open")
        self._file_handle = open(self._temp_file_path, "rb")

    def _run_ffmpeg(self) -> None:
        if not self._temp_file_path:
            raise AudioExtractionError("Temp file not created")

        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", self.video_path]

        if self.audio_track_id is not None:
            cmd.extend(["-map", f"0:a:{self.audio_track_id}"])

        cmd.extend([
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-f", "s16le",
            self._temp_file_path,
        ])

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        timeout = max(self.duration * 1.5, 60) if self.duration > 0 else 600

        filename = os.path.basename(self.video_path)
        logger.info(f"Extracting audio from '{filename}'...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._ffmpeg_process = process

        try:
            start_time = time.time()
            while process.poll() is None:
                if time.time() - start_time > timeout:
                    process.kill()
                    raise AudioExtractionError(f"FFmpeg timed out after {timeout}s for {filename}")
                time.sleep(0.1)

            if process.returncode != 0:
                stderr = process.stderr.read().decode(errors="ignore") if process.stderr else ""
                raise AudioExtractionError(
                    f"FFmpeg exited with code {process.returncode}",
                    stderr=stderr,
                )
        finally:
            if process.poll() is None:
                process.kill()
            if process.stderr:
                process.stderr.close()

        logger.info("Audio extraction complete")

    def __enter__(self) -> "AudioExtractor":
        return self

    def __exit__(self, *args) -> None:
        self.close()
