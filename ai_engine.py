import bisect
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from queue import PriorityQueue, Empty
from typing import Optional, TextIO, Callable

from transcription import TranscriptionInterface, SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH

class AITranscriptionEngine:
    # Coordinates chunked transcription and atomic subtitle updates

    # --- Configuration Constants ---
    IDLE_PRIORITY_OFFSET = 1000000.0
    SUBTITLE_REWRITE_THROTTLE_S = 3.00 # Batch rapid update requests to reduce IO
    # --- Subtitle Rules ---
    SUB_MIN_DURATION_S, SUB_MAX_DURATION_S, SUB_MIN_GAP_S = 0.8, 7.0, 0.15
    SUB_MAX_CPL, SUB_PADDING_S = 42, 0.150
    # --- ASS Style ---
    ASS_FONT_NAME, ASS_FONT_SIZE = "Arial", 65
    ASS_PRIMARY_COLOR, ASS_OUTLINE_COLOR = "&H00FFFFFF", "&H00000000"

    def __init__(self, video_metadata: dict, config: dict, transcription_model: TranscriptionInterface, on_update_callback: Callable[[str], None]):
        # persist video and configuration for the engine lifecycle
        self.video_path = video_metadata.get('path')
        self.video_filename = video_metadata.get('filename')
        self.duration = video_metadata.get('duration')
        self.video_width = video_metadata.get('width')
        self.video_height = video_metadata.get('height')
        self.config = config
        self.transcription_model = transcription_model
        self.on_update_callback = on_update_callback

        self.CHUNK_DURATION = self.config.get('CHUNK_DURATION_SECONDS', 30)
        self.total_chunks_count = math.ceil(self.duration / self.CHUNK_DURATION) if self.duration and self.duration > 0 else 1
        self.total_chunks_index = self.total_chunks_count - 1

        self.video_context = f"[Video: {self.video_filename[:15]}...]"
        self.subtitle_path = self._generate_subtitle_filepath()

        # Threading primitives and shared state
        self.is_finished = threading.Event()
        self.stop_worker = threading.Event()
        self.rewrite_needed = threading.Event() # Signaler to trigger batched subtitle writes
        self.subtitle_segments = []
        self.subtitle_lock = threading.Lock()
        self.task_queue = PriorityQueue()
        self.queue_lock = threading.Lock()
        self.processed_chunks, self.queued_chunks = set(), set()
        self.last_known_time, self.sequential_idle_pointer = 0.0, 0
        self.audio_file_path: Optional[str] = None
        self.audio_file_handle: Optional[TextIO] = None

        self._pre_extract_audio()
        self._initialize_subtitle_file()

        # Worker threads that perform transcription and write aggregated subtitles
        self.worker_thread = threading.Thread(target=self._worker, daemon=True, name=f"AIEngine-Worker-{self.video_filename[:10]}")
        self.writer_thread = threading.Thread(target=self._subtitle_writer_worker, daemon=True, name=f"AIEngine-Writer-{self.video_filename[:10]}")
        self.worker_thread.start()
        self.writer_thread.start()
        logging.info(f"{self.video_context} Engine initialized (Duration: {self.duration:.2f}s, Chunks: {self.total_chunks_count}).")

    def _worker(self):
        # Process tasks until all chunks are handled or a stop is requested
        logging.info(f"{self.video_context} Worker thread started.")
        while not self.stop_worker.is_set():
            try:
                # fast-path: exit when no more work remains
                if len(self.processed_chunks) >= self.total_chunks_count and self.task_queue.empty():
                    break

                _, (start_time, end_time, chunk_index) = self.task_queue.get(timeout=2.0)
                if self.stop_worker.is_set(): break
                self._process_one_chunk(start_time, end_time, chunk_index)
            except Empty:
                # when idle, schedule lower-priority background chunks to fill throughput
                if len(self.processed_chunks) >= self.total_chunks_count:
                    break
                else:
                    self._queue_idle_chunk()

        if not self.stop_worker.is_set():
            logging.info(f"{self.video_context} All {len(self.processed_chunks)} chunks processed. Worker thread is shutting down.")
            self.is_finished.set()
        else:
            logging.info(f"{self.video_context} Worker received stop signal.")

    def _subtitle_writer_worker(self):
        """Dedicated thread to handle throttled subtitle file writes."""
        logging.info(f"{self.video_context} Subtitle writer thread started.")
        while not self.stop_worker.is_set():
            # Wait until a rewrite is needed or until timeout to check for stop signal
            if self.rewrite_needed.wait(timeout=1.0):
                # Once signaled, clear the event immediately
                self.rewrite_needed.clear()
                # Then, sleep for the throttle duration to batch subsequent rapid requests
                time.sleep(self.SUBTITLE_REWRITE_THROTTLE_S)

                self._rewrite_subtitle_file()

        logging.info(f"{self.video_context} Writer received stop signal, performing final write.")
        # Perform one final, un-throttled write to flush all pending changes
        if len(self.subtitle_segments) > 0:
            self._rewrite_subtitle_file()

    def shutdown(self):
        """Stop threads and flush subtitle writes safely."""
        logging.info(f"{self.video_context} Shutdown requested for engine.")
        self.stop_worker.set()
        # Clear queued tasks to unblock the worker quickly.
        with self.queue_lock:
            self.task_queue = PriorityQueue()
            self.queued_chunks.clear()

        # Unblock the writer thread if it's waiting on the event
        self.rewrite_needed.set()

        # Allow a brief, non-blocking join so threads can exit cleanly.
        self.worker_thread.join(timeout=1.5)
        self.writer_thread.join(timeout=1.5)

        self._cleanup()

    def _rewrite_subtitle_file(self):
        # Atomically update subtitle file to avoid partial reads by the player
        header = self._get_ass_header()

        # Prevent race condition on self.subtitle_segments while creating the lines
        with self.subtitle_lock:
            dialogue_lines = [
                f"Dialogue: 0,{self._format_ass_timestamp(start)},{self._format_ass_timestamp(end)},Default,,0,0,0,,{text}"
                for start, end, text in self.subtitle_segments
            ]
        content = header + "\n" + "\n".join(dialogue_lines) + "\n"

        temp_subtitle_path = self.subtitle_path + ".tmp"

        try:
            with open(temp_subtitle_path, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(temp_subtitle_path, self.subtitle_path)
            # Notify the main service about subtitle updates for immediate reload
            self.on_update_callback(self.subtitle_path)
        except Exception as e:
            logging.error(f"{self.video_context} An unexpected error occurred during subtitle rewrite: {e}", exc_info=True)
            if os.path.exists(temp_subtitle_path):
                os.remove(temp_subtitle_path)

    def _process_one_chunk(self, start_time: float, end_time: float, chunk_index: int):
        # Convert audio chunk to timestamped subtitles and insert them safely
        try:
            audio_buffer = self._extract_audio_chunk(start_time, end_time)
            if not audio_buffer:
                return

            transcription_data = self.transcription_model.transcribe(audio_buffer)
            # If a shutdown was requested during transcription, abort post-processing.
            if self.stop_worker.is_set():
                logging.info(f"{self.video_context} Stop signal received mid-chunk. Aborting post-processing.")
                return

            if not transcription_data or not transcription_data.get('words'):
                return

            refined_segments = self._apply_subtitle_rules(transcription_data)

            new_segments_added = 0
            # hold lock only while mutating shared subtitle list to minimize blocking
            with self.subtitle_lock:
                for item in refined_segments:
                    abs_start = start_time + item['start']
                    abs_end = start_time + item['end']
                    text = item['text'].strip()
                    if not text: continue

                    new_segment = (abs_start, abs_end, text)
                    bisect.insort(self.subtitle_segments, new_segment)
                    new_segments_added += 1

            if new_segments_added > 0:
                logging.info(f"{self.video_context} Added {new_segments_added} segments from chunk {chunk_index}. Total: {len(self.subtitle_segments)}.")
                # Signal the writer thread instead of writing directly to avoid IO on worker threads
                self.rewrite_needed.set()
        except Exception as e:
            logging.error(f"{self.video_context} Critical error during chunk processing ({start_time:.2f}s): {e}", exc_info=True)
        finally:
            with self.queue_lock:
                self.queued_chunks.discard(start_time)
                self.processed_chunks.add(start_time)
            self.task_queue.task_done()

    def _cleanup(self):
        # Remove temporary audio artifacts to free disk and memory
        if self.audio_file_handle:
            self.audio_file_handle.close()
            self.audio_file_handle = None
        if self.audio_file_path and os.path.exists(self.audio_file_path):
            try:
                os.remove(self.audio_file_path)
                logging.info(f"{self.video_context} Cleaned up temporary audio file: {self.audio_file_path}")
            except OSError as e:
                logging.warning(f"{self.video_context} Failed to remove temporary audio file: {e}")
        self.audio_file_path = None

    def _pre_extract_audio(self):
        # Pre-extract full audio to enable efficient random-access reads
        with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as fp:
            self.audio_file_path = fp.name

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        command = [
            'ffmpeg', '-y', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-f', 's16le', self.audio_file_path
        ]
        logging.info(f"{self.video_context} Starting full audio extraction to {os.path.basename(self.audio_file_path)}...")
        try:
            timeout = self.duration * 1.5 if self.duration and self.duration > 0 else 600
            subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True, creationflags=creationflags)
            self.audio_file_handle = open(self.audio_file_path, 'rb')
            logging.info(f"{self.video_context} Full audio extraction complete and file is open for reading.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr = e.stderr if hasattr(e, 'stderr') else 'Timeout'
            logging.critical(f"{self.video_context} FFMPEG pre-extraction failed: {stderr}")
            self._cleanup()
            raise RuntimeError("Audio extraction failed.") from e
        except Exception as e:
            logging.critical(f"{self.video_context} FFMPEG pre-extraction failed with an unexpected error: {e}")
            self._cleanup()
            raise RuntimeError("Audio extraction failed.") from e

    def _get_ass_header(self) -> str:
        # Build ASS header aligned to player resolution and chosen style
        header_lines = ["[Script Info]", "Title: AI Generated Subtitles", "ScriptType: v4.00+"]
        if self.video_width and self.video_height:
            header_lines.extend([f"PlayResX: {self.video_width}", f"PlayResY: {self.video_height}"])

        style_line = (f"Style: Default,{self.ASS_FONT_NAME},{self.ASS_FONT_SIZE},{self.ASS_PRIMARY_COLOR},&H000000FF,{self.ASS_OUTLINE_COLOR},"
                      "&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1")

        header_lines.extend(["", "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            style_line, "", "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"])
        return "\n".join(header_lines)

    def _initialize_subtitle_file(self):
        # Create initial subtitle file so the player can load a placeholder immediately
        header = self._get_ass_header()
        try:
            with self.subtitle_lock:
                with open(self.subtitle_path, 'w', encoding='utf-8') as f: f.write(header)
            logging.info(f"{self.video_context} Initialized new subtitle file: '{os.path.basename(self.subtitle_path)}'")
        except IOError as e:
            logging.error(f"{self.video_context} Failed to initialize subtitle file: {e}", exc_info=True)

    def _format_ass_timestamp(self, total_seconds: float) -> str:
        # Format seconds into ASS timestamp with centiseconds precision
        centiseconds = int((total_seconds % 1) * 100)
        total_seconds = int(total_seconds)
        seconds = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    def _generate_subtitle_filepath(self) -> str:
        # Save generated subtitle beside the video so MPV auto-loads it
        base_name = os.path.splitext(self.video_path)[0]
        return f"{base_name}.ai.ass"

    def process_update(self, current_time: float):
        # Prioritize transcription around the current playback position for responsiveness
        if current_time is None: return
        if abs(current_time - self.last_known_time) > self.CHUNK_DURATION:
            logging.info(f"{self.video_context} Large seek detected. Clearing active processing queue.")
            with self.queue_lock:
                self._clear_task_queue()

        self.last_known_time = current_time
        current_chunk_index = int(current_time / self.CHUNK_DURATION)
        for offset in range(-1, 4):
            chunk_index = current_chunk_index + offset
            if not (0 <= chunk_index <= self.total_chunks_index): continue

            chunk_start_time = chunk_index * self.CHUNK_DURATION
            with self.queue_lock:
                if chunk_start_time in self.processed_chunks or chunk_start_time in self.queued_chunks:
                    continue

                priority = abs(chunk_start_time - current_time)
                chunk_end_time = min(chunk_start_time + self.CHUNK_DURATION, self.duration)
                logging.debug(f"{self.video_context} Queueing ACTIVE chunk {chunk_index} ({chunk_start_time:.2f}s) with P={priority:.2f}")
                self.queued_chunks.add(chunk_start_time)
                self.task_queue.put((priority, (chunk_start_time, chunk_end_time, chunk_index)))

    def _clear_task_queue(self):
        """Empties the task queue. Must be called within queue_lock."""
        old_queue = self.task_queue
        self.task_queue = PriorityQueue()
        self.queued_chunks.clear()

        # Re-queue idle tasks that were cleared so background progress resumes
        while not old_queue.empty():
            try:
                priority, task = old_queue.get_nowait()
                if priority >= self.IDLE_PRIORITY_OFFSET:
                    self.task_queue.put((priority, task))
                    self.queued_chunks.add(task[0]) # task[0] is start_time
            except Empty:
                break

    def _queue_idle_chunk(self):
        # Backgroundly schedule remaining chunks to avoid impacting playback
        with self.queue_lock:
            while self.sequential_idle_pointer <= self.total_chunks_index:
                start_time = self.sequential_idle_pointer * self.CHUNK_DURATION
                if start_time not in self.processed_chunks and start_time not in self.queued_chunks:
                    chunk_index = self.sequential_idle_pointer
                    end_time = min(start_time + self.CHUNK_DURATION, self.duration)
                    priority = self.IDLE_PRIORITY_OFFSET + chunk_index
                    logging.debug(f"{self.video_context} Queueing IDLE chunk {chunk_index} ({start_time:.2f}s) with P={priority:.2f}")
                    self.queued_chunks.add(start_time)
                    self.task_queue.put((priority, (start_time, end_time, chunk_index)))
                    self.sequential_idle_pointer += 1
                    return
                self.sequential_idle_pointer += 1

    def _extract_audio_chunk(self, start_time: float, end_time: float) -> Optional[bytes]:
        # Read byte-aligned audio slices to preserve frame integrity
        if not self.audio_file_handle:
            logging.error(f"{self.video_context} Cannot extract chunk: Audio file handle is not available.")
            return None

        bytes_per_sample_frame = SAMPLE_WIDTH * CHANNELS
        start_offset = int(start_time * SAMPLE_RATE) * bytes_per_sample_frame
        num_bytes_to_read = int((end_time - start_time) * SAMPLE_RATE) * bytes_per_sample_frame

        if num_bytes_to_read <= 0:
            return None

        try:
            self.audio_file_handle.seek(start_offset)
            audio_buffer = self.audio_file_handle.read(num_bytes_to_read)
            return audio_buffer
        except (IOError, ValueError) as e:
            logging.error(f"{self.video_context} Failed to read audio chunk from temp file: {e}", exc_info=True)
            return None

    def _apply_subtitle_rules(self, transcription_data: dict) -> list[dict]:
        # Adjust timing and line breaks to improve subtitle readability
        words = transcription_data.get('words', [])
        if not words: return []

        initial_segments = []
        current_line_words = []

        for i, word_info in enumerate(words):
            current_line_words.append(word_info)
            current_text = " ".join(w['word'] for w in current_line_words)

            is_last_word = (i == len(words) - 1)
            punct_break = any(p in word_info['word'] for p in ".?!")
            cpl_exceeded = len(current_text) > self.SUB_MAX_CPL

            if is_last_word or punct_break or cpl_exceeded:
                if cpl_exceeded and len(current_line_words) > 1:
                    segment_words = current_line_words[:-1]
                    current_line_words = current_line_words[-1:]
                else:
                    segment_words = current_line_words
                    current_line_words = []

                if not segment_words:
                    continue

                text = " ".join(w['word'] for w in segment_words)
                start_time = segment_words[0]['start']
                end_time = segment_words[-1]['end']
                initial_segments.append({'text': text, 'start': start_time, 'end': end_time})

        if not initial_segments: return []

        processed_segments = []
        for i, seg in enumerate(initial_segments):
            seg['start'] = max(0.0, seg['start'] - self.SUB_PADDING_S)
            seg['end'] += self.SUB_PADDING_S
            if seg['end'] - seg['start'] < self.SUB_MIN_DURATION_S: seg['end'] = seg['start'] + self.SUB_MIN_DURATION_S
            if seg['end'] - seg['start'] > self.SUB_MAX_DURATION_S: seg['end'] = seg['start'] + self.SUB_MAX_DURATION_S
            if processed_segments:
                prev_end = processed_segments[-1]['end']
                if seg['start'] < prev_end + self.SUB_MIN_GAP_S: seg['start'] = prev_end + self.SUB_MIN_GAP_S
            if seg['end'] <= seg['start']: continue

            if len(seg['text']) > self.SUB_MAX_CPL:
                split_point = seg['text'].rfind(' ', 0, self.SUB_MAX_CPL)
                if split_point != -1: seg['text'] = f"{seg['text'][:split_point]}\\N{seg['text'][split_point+1:]}"

            processed_segments.append(seg)

        return processed_segments