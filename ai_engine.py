import bisect
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from queue import PriorityQueue, Empty

import requests

# --- Audio Extraction Constants ---
# Defines the raw audio format for WhisperX compatibility.
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1         # Mono
SAMPLE_WIDTH = 2     # 16-bit PCM (s16le)
# Calculated size of a 30-second audio chunk in bytes.
AUDIO_CHUNK_SIZE_BYTES = 30 * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS

class TranscriptionInterface(ABC):
    """Abstract base class for transcription models."""
    @abstractmethod
    def transcribe(self, audio_buffer: bytes) -> list[dict]:
        pass
    def close(self) -> None:
        logging.info(f"[Model] Closing transcription model: {self.__class__.__name__}")
        pass

class WhisperXWebClient(TranscriptionInterface):
    """Client for communicating with the external WhisperX Flask server (Docker container)."""
    def __init__(self, server_url="http://localhost:5000"):
        self.transcribe_url = f"{server_url}/transcribe"
        self.health_url = f"{server_url}/health"
        logging.info(f"[Model] WhisperX web client initialized. Target: {server_url}")
        # Blocks initialization until the AI model is loaded and ready.
        self._wait_for_server()

    def _wait_for_server(self):
        """Polls the health endpoint until the transcription server responds (200 OK)."""
        logging.info("[Model] Checking transcription server readiness...")
        
        max_wait_time = 300
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(self.health_url, timeout=5)
                if response.status_code == 200:
                    logging.info("[Model] Transcription server is ready (200 OK).")
                    return
            except requests.exceptions.RequestException:
                logging.debug("[Model] Server not yet reachable. Retrying in 2 seconds.")
                time.sleep(2)
        
        logging.critical("FATAL: Transcription server did not become ready within 300 seconds.")
        raise RuntimeError("Could not connect to the transcription server.")

    def transcribe(self, audio_buffer: bytes) -> list[dict]:
        """Sends raw audio bytes to the server and requests word-level timestamps."""
        if not audio_buffer:
            return []

        data_payload = {
            'language': 'en',
            'word_timestamps': 'true'
        }

        files = {'audio': ('audio.s16le', audio_buffer, 'application/octet-stream')}
        
        try:
            logging.info(f"[API] Sending {len(audio_buffer)} bytes for transcription.")
            response = requests.post(self.transcribe_url, files=files, data=data_payload, timeout=60)
            response.raise_for_status()
            
            segments = response.json()

            if isinstance(segments, list):
                if segments:
                    logging.info(f"[API] Received {len(segments)} segments from server.")
                else:
                    logging.info("[API] Received empty segment list from server.")
                return segments
            else:
                logging.error(f"[API] API contract violation: Server returned unexpected data type: {type(segments)}")
                return []
        
        except requests.exceptions.RequestException as e:
            logging.error(f"[API] Request failed: {e}")
            return []
        except requests.exceptions.JSONDecodeError as e:
            logging.error(f"[API] JSON Decode failed: {e}. Response text snippet: {response.text[:100]}...")
            return []
        except Exception as e:
            logging.error(f"[API] Unexpected error during API call: {e}", exc_info=True)
            return []


class AITranscriptionEngine:
    """Manages transcription tasks and subtitle generation for a single video file."""

    def __init__(self, video_metadata: dict, config: dict, on_chunk_completed_callback: callable, transcription_model: TranscriptionInterface):
        self.video_path = video_metadata.get('path')
        self.video_filename = video_metadata.get('filename')
        self.duration = video_metadata.get('duration')
        self.config = config
        self.on_chunk_completed = on_chunk_completed_callback
        self.transcription_model = transcription_model

        self.CHUNK_DURATION = self.config.get('CHUNK_DURATION_SECONDS', 30)
        
        # Calculates the correct maximum chunk index.
        # e.g., a 60s video has chunks 0 and 1, so total_chunks (max index) is 1.
        if self.duration > 0:
            num_chunks = math.ceil(self.duration / self.CHUNK_DURATION)
            self.total_chunks = int(num_chunks - 1)
        else:
            self.total_chunks = -1 # No chunks to process
        
        self.video_context = f"[Video: {self.video_filename[:15]}...]"
        self.subtitle_path = self._generate_subtitle_filepath()
        
        # Critical state management
        self.subtitle_segments = []
        self.subtitle_lock = threading.Lock()
        self.max_segment_end_time = 0.0
        
        self.task_queue = PriorityQueue()
        self.queue_lock = threading.Lock()
        self.processed_chunks = set()
        self.queued_chunks = set()
        self.last_known_time = 0.0

        # Pointer for sequential background transcription.
        self.sequential_idle_pointer = 0
        
        # Audio extraction optimization
        self.audio_file_path = None
        # Executes FFMPEG once to extract the full audio stream to a temporary file.
        self._pre_extract_audio()

        # Writes the ASS header to the file.
        self._initialize_subtitle_file()

        self.worker_thread = threading.Thread(target=self._worker, daemon=True, name=f"AIEngine-{self.video_filename[:10]}")
        self.worker_thread.start()

        logging.info(f"{self.video_context} Engine initialized (Duration: {self.duration:.2f}s, Chunks: {self.total_chunks + 1}).")

    def _cleanup(self):
        """Removes the temporary raw audio file."""
        self._cleanup_audio_file()

    def _cleanup_audio_file(self):
        """Deletes the temporary audio file if it exists."""
        if self.audio_file_path and os.path.exists(self.audio_file_path):
            try:
                os.remove(self.audio_file_path)
                logging.info(f"{self.video_context} Cleaned up temporary audio file: {self.audio_file_path}")
            except OSError as e:
                logging.warning(f"{self.video_context} Failed to remove temporary audio file: {e}")
        self.audio_file_path = None

    def _pre_extract_audio(self):
        """Extracts the entire video audio track to a temporary raw PCM file.
        This eliminates repeated FFMPEG process launches during chunk processing."""
        self.audio_file_path = os.path.join(tempfile.gettempdir(), f"ai_audio_{os.getpid()}_{threading.get_ident()}.raw")
        
        command = [
            'ffmpeg', '-y', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-f', 's16le', self.audio_file_path
        ]
        
        logging.info(f"{self.video_context} Starting full audio extraction to {os.path.basename(self.audio_file_path)}...")
        try:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process.wait(timeout=self.duration * 2)
            
            if process.returncode != 0:
                logging.error(f"{self.video_context} FFMPEG failed during full audio extraction (Return code: {process.returncode}).")
                self._cleanup_audio_file()
                raise RuntimeError("Audio extraction failed.")
            
            logging.info(f"{self.video_context} Full audio extraction complete.")
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
            logging.critical(f"{self.video_context} FFMPEG execution failed during pre-extraction: {e}")
            self._cleanup_audio_file()
            raise RuntimeError("FFMPEG pre-extraction failed.")

    def get_max_segment_end_time(self) -> float:
        """Returns the end time of the latest transcribed segment."""
        with self.subtitle_lock:
            return self.max_segment_end_time

    def _format_ass_timestamp(self, total_seconds: float) -> str:
        """Formats seconds into the ASS timestamp format (H:MM:SS.CC)."""
        centiseconds = int((total_seconds % 1) * 100)
        total_seconds = int(total_seconds)
        seconds = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    def _generate_subtitle_filepath(self) -> str:
        """Generates the path for the output ASS subtitle file."""
        base_name = os.path.splitext(self.video_path)[0]
        return f"{base_name}.ai.ass"

    def _initialize_subtitle_file(self):
        """Writes the required ASS header to the subtitle file."""
        header = (
            "[Script Info]\nTitle: AI Generated Subtitles\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        try:
            with self.subtitle_lock:
                with open(self.subtitle_path, 'w', encoding='utf-8') as f:
                    f.write(header)
            logging.info(f"{self.video_context} Initialized new subtitle file: '{os.path.basename(self.subtitle_path)}'")
        except IOError as e:
            logging.error(f"{self.video_context} Failed to initialize subtitle file: {e}", exc_info=True)

    def _clear_queue_on_seek(self):
        """Clears the task queue and resets the idle pointer after a large seek."""
        with self.queue_lock:
            while not self.task_queue.empty():
                try: self.task_queue.get_nowait()
                except Empty: break
            self.queued_chunks.clear()
            self.sequential_idle_pointer = 0
            logging.info(f"{self.video_context} Seek detected. Task queue cleared.")

    def process_update(self, current_time: float):
        """Updates the task queue based on the current playback position."""
        if current_time is None: return

        if abs(current_time - self.last_known_time) > self.CHUNK_DURATION:
            logging.info(f"{self.video_context} Large seek detected from {self.last_known_time:.2f}s to {current_time:.2f}s. Resetting queue.")
            self._clear_queue_on_seek()

        self.last_known_time = current_time

        active_horizon = [-1, 0, 1, 2, 3]
        current_chunk_index = int(current_time / self.CHUNK_DURATION)

        for offset in active_horizon:
            chunk_index = current_chunk_index + offset
            if not (0 <= chunk_index <= self.total_chunks):
                continue

            chunk_start_time = chunk_index * self.CHUNK_DURATION

            with self.queue_lock:
                if chunk_start_time in self.processed_chunks or chunk_start_time in self.queued_chunks:
                    continue

                priority = abs(chunk_start_time - current_time)
                chunk_end_time = min(chunk_start_time + self.CHUNK_DURATION, self.duration)

                logging.info(f"{self.video_context} Queueing ACTIVE chunk {chunk_index} ({chunk_start_time:.2f}s) with priority {priority:.2f}")
                self.queued_chunks.add(chunk_start_time)
                self.task_queue.put((priority, (chunk_start_time, chunk_end_time, chunk_index)))

    def _worker(self):
        """The main worker loop that processes transcription tasks."""
        logging.info(f"{self.video_context} Worker thread started.")
        while True:
            try:
                priority, (start_time, end_time, chunk_index) = self.task_queue.get(timeout=2.0)
                logging.info(f"{self.video_context} Processing task: Chunk {chunk_index} at {start_time:.2f}s (P={priority:.2f})")
                self._process_one_chunk(start_time, end_time, chunk_index)
            except Empty:
                logging.debug(f"{self.video_context} Queue empty. Attempting to queue idle chunk.")
                self._queue_idle_chunk()

    def _queue_idle_chunk(self):
        """Queues the next sequential chunk that has not yet been processed."""
        with self.queue_lock:
            if self.sequential_idle_pointer > self.total_chunks:
                return

            chunk_index = self.sequential_idle_pointer
            start_time = chunk_index * self.CHUNK_DURATION
            
            while start_time in self.processed_chunks and chunk_index <= self.total_chunks:
                chunk_index += 1
                start_time = chunk_index * self.CHUNK_DURATION
            
            if chunk_index > self.total_chunks:
                return

            self.sequential_idle_pointer = chunk_index + 1
            
            end_time = min(start_time + self.CHUNK_DURATION, self.duration)
            priority = 1000000.0 + chunk_index

            logging.info(f"{self.video_context} Queueing IDLE chunk {chunk_index} ({start_time:.2f}s) with priority {priority:.2f}")
            self.queued_chunks.add(start_time)
            self.task_queue.put((priority, (start_time, end_time, chunk_index)))

    def _extract_audio_chunk(self, start_time: float, end_time: float) -> bytes | None:
        """
        Reads audio bytes from the pre-extracted file. If the chunk is shorter
        than the standard chunk duration (i.e., it's the last chunk), it is
        padded with silence to ensure the model receives a consistent input size.
        """
        if not self.audio_file_path or not os.path.exists(self.audio_file_path):
            logging.error(f"{self.video_context} Cannot extract chunk: Audio file not available.")
            return None

        # Calculate byte offset and size based on time and audio format constants.
        start_offset_bytes = int(start_time * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        duration_seconds = end_time - start_time
        size_bytes = int(duration_seconds * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        
        try:
            with open(self.audio_file_path, 'rb') as f:
                f.seek(start_offset_bytes)
                audio_bytes = f.read(size_bytes)
            
            # Define the target size for a full chunk.
            target_size_bytes = int(self.CHUNK_DURATION * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)

            # If the extracted audio is shorter than a full chunk, pad it with silence.
            if len(audio_bytes) < target_size_bytes:
                bytes_to_pad = target_size_bytes - len(audio_bytes)
                # Create a silence buffer (bytes of zeros).
                silence = b'\x00' * bytes_to_pad
                audio_bytes += silence
                logging.info(f"{self.video_context} Padded final chunk with {bytes_to_pad} bytes of silence.")

            logging.debug(f"{self.video_context} Extracted {len(audio_bytes)} bytes from temp file (post-padding).")
            return audio_bytes
        except IOError as e:
            logging.error(f"{self.video_context} Failed to read audio chunk from temp file: {e}")
            return None

    def _refine_and_resegment(self, original_results: list[dict]) -> list[dict]:
        """
        Transforms raw ASR results into professional-grade subtitle segments.

        This multi-pass pipeline applies a series of rules to ensure readability,
        correct timing, and adherence to subtitling standards. The order of
        operations is critical, as later passes refine or correct the output
        of earlier ones.

        Pass 1: Re-segment based on CPL and pauses.
        Pass 2: Correct trailing silences using word-level timestamps.
        Pass 3: Enforce minimum/maximum duration and reading speed (CPS).
        Pass 4: Prevent overlapping segments.
        Pass 5: Final validation and cleanup.
        """
        # --- Subtitling Standards & Configuration ---
        CPS_RATE = 15          # Characters Per Second reading speed.
        MAX_CPL = 42           # Max Characters Per Line.
        MIN_DURATION_S = 1.0   # Minimum time a subtitle should be on screen.
        MAX_DURATION_S = 7.0   # Maximum time a subtitle should be on screen.
        PAUSE_THRESHOLD_S = 0.5 # A pause in speech long enough to justify a new subtitle.
        MIN_GAP_S = 0.15       # Minimum silent gap between consecutive subtitles.
        START_CUSHION_S = 0.15 # Time to add before a word starts for better sync.
        END_CUSHION_S = 0.10   # Time to add after a word ends for better sync.

        # --- PASS 1: Re-segment long sentences into readable lines ---
        # The model may return one long segment for a full sentence. We must break
        # it down into lines that fit on screen and are timed to speech.
        intermediate_segments = []
        for segment in original_results:
            if 'words' not in segment or not segment['words']:
                continue

            words = segment['words']
            current_line_words = []

            for i, word_data in enumerate(words):
                if not all(k in word_data for k in ['word', 'start', 'end']):
                    continue

                potential_line_words = current_line_words + [word_data]
                potential_text = " ".join([w['word'].strip() for w in potential_line_words])
                
                # Check for split conditions
                is_cpl_violated = len(potential_text) > MAX_CPL
                is_pause_split = False
                if current_line_words:
                    prev_word_end = current_line_words[-1]['end']
                    current_word_start = word_data['start']
                    if (current_word_start - prev_word_end) > PAUSE_THRESHOLD_S:
                        is_pause_split = True

                if current_line_words and (is_cpl_violated or is_pause_split):
                    # Finalize the previous line and start a new one
                    text = " ".join([w['word'].strip() for w in current_line_words])
                    start_time = current_line_words[0]['start'] - START_CUSHION_S
                    end_time = current_line_words[-1]['end'] + END_CUSHION_S
                    intermediate_segments.append({'start': start_time, 'end': end_time, 'text': text, 'words': current_line_words})
                    current_line_words = [word_data]
                else:
                    current_line_words = potential_line_words
            
            # Add the last remaining line
            if current_line_words:
                text = " ".join([w['word'].strip() for w in current_line_words])
                start_time = current_line_words[0]['start'] - START_CUSHION_S
                end_time = current_line_words[-1]['end'] + END_CUSHION_S
                intermediate_segments.append({'start': start_time, 'end': end_time, 'text': text, 'words': current_line_words})

        if not intermediate_segments:
            return []
        
        final_segments = intermediate_segments

        # --- PASS 2: Intelligent End-Time Correction ---
        # Corrects segments where the model included a long trailing silence. This is
        # common with short, sharp phrases (e.g., "Okay."). We use the end time of
        # the last word as a more accurate anchor. This must run before duration enforcement.
        for seg in final_segments:
            text_len = len(seg['text'])
            current_duration = seg['end'] - seg['start']
            
            # Heuristic: A short phrase with a disproportionately long duration.
            if text_len < 15 and current_duration > (text_len * 0.4):
                if seg['words']:
                    # Calculate a more accurate end time based on the last spoken word.
                    target_end = seg['words'][-1]['end'] + END_CUSHION_S
                    
                    # Only apply the correction if it's a significant reduction (>100ms).
                    if target_end < seg['end'] - 0.1:
                        seg['end'] = target_end

        # --- PASS 3: Enforce Duration and Reading Speed Rules ---
        # Ensures every subtitle is readable and doesn't linger too long.
        for seg in final_segments:
            num_chars = len(seg['text'])
            current_duration = seg['end'] - seg['start']
            
            # Calculate minimum required duration based on CPS and absolute minimum.
            required_cps_duration = num_chars / CPS_RATE
            target_duration = max(current_duration, required_cps_duration, MIN_DURATION_S)
            
            # Cap the duration to the maximum allowed.
            final_duration = min(target_duration, MAX_DURATION_S)

            # Extend the end time if the new duration is longer.
            if final_duration > current_duration:
                seg['end'] = seg['start'] + final_duration

        # --- PASS 4: Prevent Overlaps ---
        # Ensures a minimum silent gap between consecutive subtitles for clean presentation.
        for i in range(len(final_segments) - 1):
            current_seg = final_segments[i]
            next_seg = final_segments[i+1]
            
            # If the current segment's end time is too close to the next one's start...
            if current_seg['end'] > (next_seg['start'] - MIN_GAP_S):
                #...snap it back to create the required gap.
                current_seg['end'] = next_seg['start'] - MIN_GAP_S

        # --- PASS 5: Final Sanity Check and Validation ---
        # This is a critical safety net to catch any invalid timings created by
        # the previous passes (e.g., negative start times or durations).
        for seg in final_segments:
            # Clamp start time to 0.0 to avoid negative values.
            seg['start'] = max(0.0, seg['start'])
            
            # If a segment now has a zero or negative duration (e.g., from overlap
            # prevention), give it the minimum allowed duration.
            if seg['end'] <= seg['start']:
                seg['end'] = seg['start'] + MIN_DURATION_S
                
        return final_segments

    def _rewrite_subtitle_file(self):
        """Rewrites the entire subtitle file, including the header and all current segments.
        
        This method is called to ensure chronological order and prevent duplication after
        new segments are added. It assumes the caller is already holding self.subtitle_lock.
        """
        header = (
            "[Script Info]\nTitle: AI Generated Subtitles\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        dialogue_lines = []
        for start, end, text in self.subtitle_segments:
            start_str = self._format_ass_timestamp(start)
            end_str = self._format_ass_timestamp(end)
            # The text is already cleaned, so we only need to remove potential newlines for ASS format.
            formatted_text = text.replace('\n', ' ')
            dialogue_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{formatted_text}")
        content = header + "\n" + "\n".join(dialogue_lines) + "\n"
        
        try:
            with open(self.subtitle_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.debug(f"{self.video_context} Subtitle file rewritten. Total segments: {len(self.subtitle_segments)}.")
        except IOError as e:
            logging.error(f"{self.video_context} Failed to rewrite subtitle file: {e}", exc_info=True)

    def _clean_and_format_text(self, text: str) -> str:
        """
        Cleans and standardizes raw transcribed text for better subtitle readability.
        
        This process includes:
        1. Normalizing whitespace.
        2. Capitalizing standalone "i" and common contractions (e.g., "i'm" -> "I'm").
        3. Capitalizing the first letter of the entire subtitle line.
        4. Standardizing ellipses.
        5. Correcting spacing before common punctuation marks.
        """
        if not text or not text.strip():
            return ""

        # Normalize whitespace: collapses multiple spaces, strips leading/trailing space.
        cleaned_text = " ".join(text.split())

        # Capitalize standalone "i" and common contractions using case-insensitive regex.
        # The `\b` ensures we match whole words only.
        cleaned_text = re.sub(r"\bi'm\b", "I'm", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\bi've\b", "I've", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\bi'd\b", "I'd", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\bi'll\b", "I'll", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\bi\b', 'I', cleaned_text, flags=re.IGNORECASE)

        # Capitalize the first letter of the entire subtitle line.
        if cleaned_text:
            cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]

        # Standardize ellipses for better typography.
        cleaned_text = cleaned_text.replace('...', '…')

        # Correct spacing before common punctuation marks.
        for punc in [',', '.', '?', '!', '…']:
            cleaned_text = cleaned_text.replace(f' {punc}', punc)

        return cleaned_text

    def _process_one_chunk(self, start_time: float, end_time: float, chunk_index: int):
        """Handles the full transcription pipeline for a single audio chunk."""
        MIN_DURATION_S = 1.0
        try:
            audio_buffer = self._extract_audio_chunk(start_time, end_time)
            if not audio_buffer:
                logging.warning(f"{self.video_context} Skipping chunk {start_time:.2f}s due to missing audio buffer.")
                return
            original_results = self.transcription_model.transcribe(audio_buffer)
            if not original_results:
                logging.info(f"{self.video_context} No transcription results found for chunk {start_time:.2f}s.")
                return
            refined_results = self._refine_and_resegment(original_results)
            
            with self.subtitle_lock:
                latest_end_time_in_chunk = 0.0
                for item in refined_results:
                    abs_start = start_time + item['start']
                    abs_end = start_time + item['end']
                    
                    cleaned_text = self._clean_and_format_text(item['text'])
                    if not cleaned_text:
                        continue # Skip segments that become empty after cleaning.
                    
                    if abs_end <= abs_start:
                        abs_end = abs_start + MIN_DURATION_S

                    new_segment = (abs_start, abs_end, cleaned_text)
                    latest_end_time_in_chunk = max(latest_end_time_in_chunk, abs_end)
                    
                    # Insert sorted by start time
                    bisect.insort(self.subtitle_segments, new_segment, key=lambda x: x[0])
                
                if refined_results:
                    self.max_segment_end_time = max(self.max_segment_end_time, latest_end_time_in_chunk)
                    logging.info(f"{self.video_context} Added {len(refined_results)} segments from chunk {start_time:.2f}s. Total segments: {len(self.subtitle_segments)}. Max end time: {self.max_segment_end_time:.2f}s.")
                    
                    self._rewrite_subtitle_file()

            self.on_chunk_completed(self.subtitle_path, self.max_segment_end_time)
        except Exception as e:
            logging.error(f"{self.video_context} Critical error during chunk transcription ({start_time:.2f}s): {e}", exc_info=True)
        finally:
            with self.queue_lock:
                self.queued_chunks.discard(start_time)
                self.processed_chunks.add(start_time)
            self.task_queue.task_done()