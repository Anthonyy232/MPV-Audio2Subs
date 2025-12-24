"""Transcription engine - coordinates audio extraction, transcription, and subtitle generation."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from queue import PriorityQueue, Empty
from typing import Callable

from audio2subs.audio import AudioExtractor
from audio2subs.config import ServiceConfig
from audio2subs.subtitle import SubtitleWriter, SubtitleSegment, generate_subtitle_path
from audio2subs.transcription.base import BaseTranscriber, TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """Manages the transcription workflow for a single video.
    
    Coordinates:
    - Audio extraction
    - Chunk scheduling with priority queue
    - Transcription via the ASR backend
    - Subtitle file generation
    """
    
    IDLE_PRIORITY_OFFSET = 1_000_000.0
    
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
        """Initialize the transcription engine.
        
        Args:
            video_path: Path to the video file
            duration: Video duration in seconds
            transcriber: ASR backend to use
            config: Service configuration
            video_width: Video width for subtitle styling
            video_height: Video height for subtitle styling
            audio_track_id: MPV audio track ID to extract
            on_subtitle_update: Callback when subtitle file is updated
            on_progress: Callback with (completed_chunks, total_chunks)
        """
        self.video_path = video_path
        self.video_filename = os.path.basename(video_path)
        self.duration = duration
        self.transcriber = transcriber
        self.config = config
        self.on_subtitle_update = on_subtitle_update
        self.on_progress = on_progress
        
        # Calculate chunk info
        self.chunk_duration = config.chunk_duration_seconds
        self.total_chunks = max(1, math.ceil(duration / self.chunk_duration)) if duration > 0 else 1
        
        # Generate subtitle path
        self.subtitle_path = generate_subtitle_path(video_path, config.subtitle.subtitle_suffix)
        
        # Initialize components
        self._audio = AudioExtractor(video_path, duration, audio_track_id)
        self._subtitle_writer = SubtitleWriter(
            self.subtitle_path,
            config.subtitle,
            video_width,
            video_height,
            on_update=on_subtitle_update
        )
        
        # Threading state
        self._stop_event = threading.Event()
        self._finished_event = threading.Event()
        self._rewrite_needed = threading.Event()
        
        # Chunk tracking
        self._task_queue: PriorityQueue = PriorityQueue()
        self._queue_lock = threading.Lock()
        self._processed_chunks: set[float] = set()
        self._queued_chunks: set[float] = set()
        self._last_known_time = 0.0
        self._idle_pointer = 0
        
        # Worker threads
        self._worker_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        
        self._log_prefix = f"[{self.video_filename[:15]}...]"
    
    @property
    def is_finished(self) -> bool:
        """Whether all chunks have been processed."""
        return self._finished_event.is_set()
    
    @property
    def progress(self) -> tuple[int, int]:
        """Current progress as (processed_chunks, total_chunks)."""
        return (len(self._processed_chunks) + len(self._queued_chunks)), self.total_chunks
    
    @property
    def completed_count(self) -> int:
        """Number of successfully processed chunks."""
        return len(self._processed_chunks)
    
    def start(self) -> None:
        """Start the transcription engine."""
        logger.info(f"{self._log_prefix} Starting engine (duration={self.duration:.1f}s, chunks={self.total_chunks})")
        
        # Extract audio (streaming mode)
        try:
            self._audio.extract_streaming()
        except Exception as e:
            logger.error(f"{self._log_prefix} Audio extraction failed: {e}")
            raise
        
        # Initialize subtitle file
        self._subtitle_writer.initialize()
        
        # Start worker threads
        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name=f"Engine-{self.video_filename[:10]}"
        )
        self._writer_thread = threading.Thread(
            target=self._writer_worker,
            daemon=True,
            name=f"Writer-{self.video_filename[:10]}"
        )
        
        self._worker_thread.start()
        self._writer_thread.start()
    
    def stop(self) -> None:
        """Stop the engine and clean up resources."""
        logger.info(f"{self._log_prefix} Stopping engine")
        self._stop_event.set()
        
        # Clear the queue to unblock worker
        with self._queue_lock:
            self._task_queue = PriorityQueue()
            self._queued_chunks.clear()
        
        # Signal writer to do final flush
        self._rewrite_needed.set()
        
        # Wait for threads to finish
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)
        
        # Cleanup audio resources
        self._audio.close()
        
        logger.info(f"{self._log_prefix} Engine stopped")
    
    def process_time_update(self, current_time: float) -> None:
        """Handle playback position update - schedule nearby chunks.
        
        Args:
            current_time: Current playback position in seconds
        """
        if current_time is None or self._finished_event.is_set():
            return
        
        # Detect large seek and clear active queue
        if abs(current_time - self._last_known_time) > self.chunk_duration:
            logger.info(f"{self._log_prefix} Large seek detected, clearing queue")
            with self._queue_lock:
                self._clear_queue()
        
        self._last_known_time = current_time
        
        # Queue chunks around current position
        current_chunk = int(current_time / self.chunk_duration)
        for offset in range(-1, 4):
            chunk_idx = current_chunk + offset
            if 0 <= chunk_idx < self.total_chunks:
                self._queue_chunk(chunk_idx, current_time)
    
    def _worker(self) -> None:
        """Worker thread - processes transcription tasks."""
        logger.info(f"{self._log_prefix} Worker started")
        
        while not self._stop_event.is_set():
            # Check if done
            if len(self._processed_chunks) >= self.total_chunks and self._task_queue.empty():
                break
            
            try:
                priority, task = self._task_queue.get(timeout=2.0)
                if self._stop_event.is_set():
                    break
                self._process_chunk(task)
            except Empty:
                # Queue idle chunks when nothing else to do
                if len(self._processed_chunks) >= self.total_chunks:
                    break
                self._queue_idle_chunk()
        
        if not self._stop_event.is_set():
            logger.info(f"{self._log_prefix} All {len(self._processed_chunks)} chunks processed")
            self._finished_event.set()
        
        logger.info(f"{self._log_prefix} Worker stopped")
    
    def _writer_worker(self) -> None:
        """Writer thread - handles throttled subtitle file writes."""
        logger.info(f"{self._log_prefix} Writer started")
        
        first_write = True
        while not self._stop_event.is_set():
            if self._rewrite_needed.wait(timeout=1.0):
                self._rewrite_needed.clear()
                
                # Throttle subsequent writes, but allow first write to happen immediately
                if not first_write:
                    time.sleep(self.config.rewrite_throttle_seconds)
                
                self._write_subtitles()
                first_write = False
        
        # Final flush
        if self._subtitle_writer.segment_count > 0:
            logger.info(f"{self._log_prefix} Final subtitle write")
            self._write_subtitles()
        
        logger.info(f"{self._log_prefix} Writer stopped")
    
    def _process_chunk(self, task: tuple[float, float, int]) -> None:
        """Process a single transcription chunk."""
        start_time, end_time, chunk_idx = task
        
        try:
            # Read audio chunk
            audio_data = self._audio.read_chunk(start_time, end_time)
            if not audio_data:
                logger.warning(f"{self._log_prefix} No audio data for chunk {chunk_idx}")
                return
            
            # Transcribe
            result = self.transcriber.transcribe(audio_data)
            
            if self._stop_event.is_set():
                return
            
            if result.is_empty:
                logger.debug(f"{self._log_prefix} Chunk {chunk_idx} had no speech")
                return
            
            # Add words to subtitle writer
            count = self._subtitle_writer.add_words(result.words, time_offset=start_time)
            
            if count > 0:
                logger.info(f"{self._log_prefix} Added {count} segments from chunk {chunk_idx}")
                self._rewrite_needed.set()
            
            # Report progress (even on failure to ensure UI reaches 100%)
            if self.on_progress:
                processed = len(self._processed_chunks) + 1
                self.on_progress(processed, self.total_chunks)
                
        except Exception as e:
            logger.error(f"{self._log_prefix} Chunk {chunk_idx} failed: {e}", exc_info=True)
        finally:
            with self._queue_lock:
                self._queued_chunks.discard(start_time)
                self._processed_chunks.add(start_time)
            try:
                self._task_queue.task_done()
            except ValueError:
                pass
    
    def _write_subtitles(self) -> None:
        """Write the subtitle file."""
        try:
            self._subtitle_writer.write()
        except Exception as e:
            logger.error(f"{self._log_prefix} Subtitle write failed: {e}")
    
    def _queue_chunk(self, chunk_idx: int, current_time: float) -> None:
        """Queue a chunk for processing with priority based on distance."""
        start_time = chunk_idx * self.chunk_duration
        
        with self._queue_lock:
            if start_time in self._processed_chunks or start_time in self._queued_chunks:
                return
            
            end_time = min(start_time + self.chunk_duration, self.duration)
            priority = abs(start_time - current_time)
            
            logger.debug(f"{self._log_prefix} Queuing chunk {chunk_idx} (P={priority:.1f})")
            self._queued_chunks.add(start_time)
            self._task_queue.put((priority, (start_time, end_time, chunk_idx)))
    
    def _queue_idle_chunk(self) -> None:
        """Queue the next unprocessed chunk at low priority."""
        with self._queue_lock:
            while self._idle_pointer < self.total_chunks:
                start_time = self._idle_pointer * self.chunk_duration
                
                if start_time not in self._processed_chunks and start_time not in self._queued_chunks:
                    end_time = min(start_time + self.chunk_duration, self.duration)
                    priority = self.IDLE_PRIORITY_OFFSET + self._idle_pointer
                    
                    logger.debug(f"{self._log_prefix} Queuing idle chunk {self._idle_pointer}")
                    self._queued_chunks.add(start_time)
                    self._task_queue.put((priority, (start_time, end_time, self._idle_pointer)))
                    self._idle_pointer += 1
                    return
                
                self._idle_pointer += 1
    
    def _clear_queue(self) -> None:
        """Clear active chunks from queue, keep idle chunks."""
        old_queue = self._task_queue
        self._task_queue = PriorityQueue()
        self._queued_chunks.clear()
        
        # Re-add idle priority tasks
        while not old_queue.empty():
            try:
                priority, task = old_queue.get_nowait()
                if priority >= self.IDLE_PRIORITY_OFFSET:
                    self._task_queue.put((priority, task))
                    self._queued_chunks.add(task[0])
            except Empty:
                break
