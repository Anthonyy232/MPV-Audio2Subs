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
from audio2subs.refinement import QwenRefiner

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
    BATCH_SIZE = 16  # max chunks per GPU inference call
    
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
        
        # Refinement logic
        self._refiner: QwenRefiner | None = None
        if config.refinement.enabled:
            self._refiner = QwenRefiner(config.refinement)
        
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
        self._first_task_done = False  # first task processed alone for fast time-to-first-subtitle
        
        # Worker threads
        self._worker_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        
        self._log_prefix = f"[Engine:{self.video_filename[:20]}]"
    
    def __repr__(self) -> str:
        return f"TranscriptionEngine(video={self.video_filename}, progress={self.completed_count}/{self.total_chunks})"
    
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
        
        # Clear the queue contents to unblock worker gracefully
        with self._queue_lock:
            with self._task_queue.mutex:
                self._task_queue.queue.clear()
            self._queued_chunks.clear()
            
            # Insert poison pill to immediately wake up waiting worker block
            self._task_queue.put((-1, (0.0, 0.0, -1)))
        
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
                priority, first_task = self._task_queue.get(timeout=0.5)
                if self._stop_event.is_set():
                    break

                # Check for poison pill
                if first_task[2] == -1:
                    break

                # Always process the very first task alone so the first subtitle
                # appears as fast as possible. Batching would delay TTF because
                # we'd need audio for all N chunks before inference starts.
                if not self._first_task_done:
                    self._first_task_done = True
                    self._process_chunk(first_task)
                else:
                    # Greedily collect additional tasks for batched inference.
                    # Track priorities so extras can be correctly re-enqueued if needed.
                    batch = [first_task]
                    extra_priorities: list[float] = []
                    while len(batch) < self.BATCH_SIZE:
                        try:
                            extra_p, extra_task = self._task_queue.get_nowait()
                            if extra_task[2] == -1:
                                # Return poison pill and stop collecting
                                self._task_queue.put((-1, (0.0, 0.0, -1)))
                                break
                            batch.append(extra_task)
                            extra_priorities.append(extra_p)
                        except Empty:
                            break

                    if len(batch) == 1:
                        self._process_chunk(batch[0])
                    else:
                        self._process_batch(batch)
            except Empty:
                # Queue idle chunks when nothing else to do
                if len(self._processed_chunks) >= self.total_chunks:
                    break
                self._queue_all_idle_chunks()
            except Exception as e:
                logger.error(f"{self._log_prefix} Worker loop error: {e}", exc_info=True)
                time.sleep(1.0)
        
        # Post-transcription refinement
        if not self._stop_event.is_set() and self._refiner:
            try:
                logger.info(f"{self._log_prefix} Starting subtitle refinement...")

                # UNLOAD ASR model first to free VRAM
                logger.info(f"{self._log_prefix} Unloading ASR model to free VRAM for refinement...")
                self.transcriber.close()

                # Load Refiner
                self._refiner.load()

                # Perform refinement
                count = self._subtitle_writer.refine(self._refiner)

                if count > 0:
                    logger.info(f"{self._log_prefix} Refinement complete: {count} segments polished")
                    self._rewrite_needed.set()

            except Exception as e:
                logger.error(f"{self._log_prefix} Refinement failed: {e}", exc_info=True)
            finally:
                # Always unload refiner to free VRAM
                self._refiner.close()

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
        start_perf = time.perf_counter()
        
        try:
            # Wait for transcriber to be fully loaded with timeout
            if not self.transcriber.is_loaded:
                logger.debug(f"{self._log_prefix} Waiting for transcription model (chunk {chunk_idx})...")
                wait_start = time.time()
                while not self.transcriber.is_loaded and not self._stop_event.is_set():
                    if time.time() - wait_start > 300: # 5 minute sanity timeout
                        raise RuntimeError("Transcription model failed to load within 5 minutes")
                        
                    if hasattr(self.transcriber, 'wait_for_load'):
                        if self.transcriber.wait_for_load(1.0):
                            break
                    else:
                        time.sleep(1.0)
                        
                if self._stop_event.is_set():
                    return
            
            # Read audio chunk
            audio_data = self._audio.read_chunk(start_time, end_time)
            if not audio_data:
                logger.warning(f"{self._log_prefix} No audio data for chunk {chunk_idx} ({start_time:.1f}s-{end_time:.1f}s)")
                return

            # Transcribe
            logger.info(f"{self._log_prefix} Transcribing chunk {chunk_idx} ({start_time:.1f}s-{end_time:.1f}s, {len(audio_data)//2000:.0f}KB)...")
            result = self.transcriber.transcribe(audio_data)
            
            if self._stop_event.is_set():
                return
            
            if result.is_empty:
                logger.debug(f"{self._log_prefix} Chunk {chunk_idx} processed (no speech detected)")
                return
            
            # Add words to subtitle writer
            count = self._subtitle_writer.add_words(result.words, time_offset=start_time)
            
            elapsed = time.perf_counter() - start_perf
            if count > 0:
                logger.info(f"{self._log_prefix} Chunk {chunk_idx} done: added {count} segments in {elapsed:.2f}s")
                self._rewrite_needed.set()
            else:
                logger.debug(f"{self._log_prefix} Chunk {chunk_idx} done: no segments added (duration: {elapsed:.2f}s)")
            
            # Report progress
            if self.on_progress:
                processed = len(self._processed_chunks) + 1
                self.on_progress(processed, self.total_chunks)
                
        except Exception as e:
            logger.error(f"{self._log_prefix} Failed to process chunk {chunk_idx}: {e}", exc_info=True)
            # Re-queue if it was a transient error? For now just log and move on to prioritize playback
        finally:
            with self._queue_lock:
                self._queued_chunks.discard(start_time)
                self._processed_chunks.add(start_time)
            try:
                self._task_queue.task_done()
            except ValueError:
                pass
    
    def _process_batch(self, tasks: list[tuple[float, float, int]]) -> None:
        """Process multiple chunks in a single batched model call.

        Reads audio for all tasks, runs one batch inference, then distributes
        results. Each task is marked processed in the finally block.
        """
        start_perf = time.perf_counter()
        chunk_ids = [t[2] for t in tasks]
        logger.info(f"{self._log_prefix} Batch processing {len(tasks)} chunks: {chunk_ids}")

        # Wait for model load (same as _process_chunk)
        if not self.transcriber.is_loaded:
            wait_start = time.time()
            while not self.transcriber.is_loaded and not self._stop_event.is_set():
                if time.time() - wait_start > 300:
                    raise RuntimeError("Transcription model failed to load within 5 minutes")
                if hasattr(self.transcriber, 'wait_for_load'):
                    if self.transcriber.wait_for_load(1.0):
                        break
                else:
                    time.sleep(1.0)
            if self._stop_event.is_set():
                return

        # Read audio for all tasks; keep track of which have valid data
        audio_buffers: list[bytes | None] = []
        for start_time, end_time, chunk_idx in tasks:
            try:
                data = self._audio.read_chunk(start_time, end_time)
                audio_buffers.append(data if data else None)
                if not data:
                    logger.warning(f"{self._log_prefix} No audio for chunk {chunk_idx}")
            except Exception as e:
                logger.error(f"{self._log_prefix} Audio read failed for chunk {chunk_idx}: {e}")
                audio_buffers.append(None)

        # Build list of (task, buffer) pairs that have valid audio
        valid_pairs = [
            (tasks[i], audio_buffers[i])
            for i in range(len(tasks))
            if audio_buffers[i] is not None
        ]

        if valid_pairs and not self._stop_event.is_set():
            try:
                valid_tasks = [p[0] for p in valid_pairs]
                valid_bufs = [p[1] for p in valid_pairs]
                results = self.transcriber.transcribe_batch(valid_bufs)

                any_new = False
                for (start_time, end_time, chunk_idx), result in zip(valid_tasks, results):
                    if result.is_empty:
                        logger.debug(f"{self._log_prefix} Chunk {chunk_idx}: no speech")
                        continue
                    count = self._subtitle_writer.add_words(result.words, time_offset=start_time)
                    if count > 0:
                        any_new = True
                        logger.debug(f"{self._log_prefix} Chunk {chunk_idx}: added {count} segments")

                elapsed = time.perf_counter() - start_perf
                logger.info(
                    f"{self._log_prefix} Batch {chunk_ids} done in {elapsed:.2f}s "
                    f"({elapsed/len(tasks):.2f}s/chunk)"
                )
                if any_new:
                    self._rewrite_needed.set()

            except Exception as e:
                logger.error(f"{self._log_prefix} Batch transcription failed: {e}", exc_info=True)

        # Mark all tasks processed and release queue slots
        for start_time, _end_time, _chunk_idx in tasks:
            with self._queue_lock:
                self._queued_chunks.discard(start_time)
                self._processed_chunks.add(start_time)
            try:
                self._task_queue.task_done()
            except ValueError:
                pass

        if self.on_progress:
            self.on_progress(len(self._processed_chunks), self.total_chunks)

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

    def _queue_all_idle_chunks(self) -> None:
        """Queue ALL remaining unprocessed chunks at low priority in one pass.

        Called when the worker queue runs dry so the worker never stalls waiting
        for the 2-second get() timeout to fire chunk by chunk.
        """
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
    
    def _clear_queue(self) -> None:
        """Clear active chunks from queue, keep idle chunks."""
        # Find idle tasks before clearing
        idle_tasks = [item for item in self._task_queue.queue if item[0] >= self.IDLE_PRIORITY_OFFSET]
        
        # Clear the queue contents
        with self._task_queue.mutex:
            self._task_queue.queue.clear()
        self._queued_chunks.clear()
        
        # Re-add idle priority tasks efficiently
        for priority, task in idle_tasks:
            self._task_queue.put((priority, task))
            self._queued_chunks.add(task[0])
