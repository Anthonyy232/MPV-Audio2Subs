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
from audio2subs.transcription.base import SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extracts and manages audio from video files using FFmpeg.
    
    Supports both blocking (full extraction) and streaming (non-blocking) modes.
    """
    
    def __init__(self, video_path: str, duration: float, audio_track_id: int | None = None):
        """Initialize the audio extractor.
        
        Args:
            video_path: Path to the source video file
            duration: Video duration in seconds
            audio_track_id: MPV audio track ID to extract (None = first track)
        """
        self.video_path = video_path
        self.duration = duration
        self.audio_track_id = audio_track_id
        
        self._temp_file_path: str | None = None
        self._file_handle: BinaryIO | None = None
        self._extraction_complete = threading.Event()
        self._extraction_thread: threading.Thread | None = None
        self._extraction_error: Exception | None = None
        self._bytes_written = 0
        self._lock = threading.Lock()
    
    @property
    def is_ready(self) -> bool:
        """Whether audio extraction is complete and file is ready."""
        return self._extraction_complete.is_set()
    
    @property
    def bytes_available(self) -> int:
        """Number of bytes currently available for reading."""
        with self._lock:
            return self._bytes_written
    
    def extract_blocking(self) -> None:
        """Extract full audio synchronously (blocking).
        
        Raises:
            AudioExtractionError: If FFmpeg fails
        """
        self._create_temp_file()
        self._run_ffmpeg()
        self._open_for_reading()
    
    def extract_streaming(self) -> None:
        """Start audio extraction in background (non-blocking).
        
        Call `bytes_available` to check progress and `is_ready` for completion.
        """
        self._create_temp_file()
        self._extraction_thread = threading.Thread(
            target=self._streaming_extraction_worker,
            daemon=True,
            name="AudioExtractor"
        )
        self._extraction_thread.start()
    
    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for streaming extraction to complete.
        
        Args:
            timeout: Maximum seconds to wait (None = wait forever)
            
        Returns:
            True if completed, False if timeout
        """
        return self._extraction_complete.wait(timeout=timeout)
    
    def _streaming_extraction_worker(self) -> None:
        """Background worker for streaming extraction."""
        try:
            self._run_ffmpeg()
            self._open_for_reading()
        except Exception as e:
            with self._lock:
                self._extraction_error = e
            logger.error(f"Streaming extraction failed: {e}")
        finally:
            self._extraction_complete.set()
    
    def _create_temp_file(self) -> None:
        """Create a temporary file for audio data."""
        fd, self._temp_file_path = tempfile.mkstemp(suffix=".raw")
        os.close(fd)
        logger.debug(f"Created temp audio file: {self._temp_file_path}")
    
    def _run_ffmpeg(self) -> None:
        """Run FFmpeg to extract audio."""
        if not self._temp_file_path:
            raise AudioExtractionError("Temp file not created")
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y', '-i', self.video_path]
        
        # Select specific audio track if specified
        if self.audio_track_id is not None:
            cmd.extend(['-map', f'0:a:{self.audio_track_id}'])
        
        cmd.extend([
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', str(CHANNELS),
            '-f', 's16le',
            self._temp_file_path
        ])
        
        # Windows: hide console window
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        
        # Calculate reasonable timeout
        timeout = max(self.duration * 1.5, 60) if self.duration > 0 else 600
        
        logger.info(f"Extracting audio to {os.path.basename(self._temp_file_path)}...")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,  # Prevent potential deadlock
                stderr=subprocess.PIPE,
                creationflags=creationflags
            )
            
            try:
                # Monitor progress
                start_time = time.time()
                while process.poll() is None:
                    if time.time() - start_time > timeout:
                        process.kill()
                        raise AudioExtractionError(f"FFmpeg timed out after {timeout}s")
                    
                    # Update bytes written so far
                    if self._temp_file_path and os.path.exists(self._temp_file_path):
                        try:
                            size = os.path.getsize(self._temp_file_path)
                            with self._lock:
                                self._bytes_written = size
                        except OSError:
                            pass
                    
                    time.sleep(0.5)
                
                if process.returncode != 0:
                    stderr = process.stderr.read().decode(errors='ignore') if process.stderr else ""
                    raise AudioExtractionError(
                        f"FFmpeg exited with code {process.returncode}",
                        stderr=stderr
                    )
            finally:
                # Ensure process is terminated on any internal error
                if process.poll() is None:
                    process.kill()
                if process.stderr:
                    process.stderr.close()
            
            # Final update
            if self._temp_file_path and os.path.exists(self._temp_file_path):
                with self._lock:
                    self._bytes_written = os.path.getsize(self._temp_file_path)
            
            logger.info("Audio extraction complete")
            
        except subprocess.TimeoutExpired as e:
            raise AudioExtractionError(
                f"FFmpeg timed out after {timeout}s",
                stderr=str(e)
            ) from e
    
    def _open_for_reading(self) -> None:
        """Open the extracted audio file for reading."""
        if not self._temp_file_path:
            raise AudioExtractionError("No audio file to open")
        
        self._file_handle = open(self._temp_file_path, 'rb')
    
    def read_chunk(self, start_time: float, end_time: float) -> bytes | None:
        """Read a chunk of audio data for the given time range.
        
        Blocks if data is not yet available in streaming mode.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Raw PCM audio bytes, or None if not available or extraction failed
        """
        if not self._file_handle:
            with self._lock:
                if self._extraction_error:
                    raise self._extraction_error
            return None
        
        bytes_per_sample_frame = SAMPLE_WIDTH * CHANNELS
        start_offset = int(start_time * SAMPLE_RATE) * bytes_per_sample_frame
        num_bytes = int((end_time - start_time) * SAMPLE_RATE) * bytes_per_sample_frame
        
        if num_bytes <= 0:
            return None
        
        # In streaming mode, wait for data to become available
        while True:
            with self._lock:
                available = self._bytes_written
                complete = self._extraction_complete.is_set()
                err = self._extraction_error
            
            if err:
                raise err
            
            if start_offset + num_bytes <= available:
                break # Data is ready
                
            if complete:
                # Extraction finished but we don't have enough bytes?
                # This can happen at the very end of a file due to rounding
                if start_offset < available:
                    num_bytes = available - start_offset
                    break
                return None
                
            # Wait for more data
            time.sleep(0.1)
        
        try:
            self._file_handle.seek(start_offset)
            return self._file_handle.read(num_bytes)
        except (IOError, ValueError) as e:
            logger.error(f"Failed to read audio chunk: {e}")
            return None
    
    def close(self) -> None:
        """Release all resources."""
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
    
    def __enter__(self) -> "AudioExtractor":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
