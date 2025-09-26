import logging
import os
import threading
import subprocess
from queue import PriorityQueue, Empty
import bisect

class TranscriptionInterface:
    """Placeholder for the actual transcription model interface."""
    def transcribe(self, audio_buffer: bytes) -> list[dict]:
        raise NotImplementedError

class AITranscriptionEngine:
    """AI transcription engine for a single video file.

    Maintains an in-memory, sorted list of all transcribed subtitle segments.
    For each new chunk of transcription data, it merges the new segments into the
    master list and regenerates the entire ASS subtitle file.
    """

    def __init__(self, video_metadata: dict, config: dict, on_chunk_completed_callback: callable, transcription_model: TranscriptionInterface):
        self.video_path = video_metadata.get('path')
        self.video_filename = video_metadata.get('filename')
        self.duration = video_metadata.get('duration')
        self.config = config
        self.on_chunk_completed = on_chunk_completed_callback
        self.transcription_model = transcription_model

        self.CHUNK_DURATION = self.config.get('CHUNK_DURATION_SECONDS', 30)
        self.total_chunks = int(self.duration / self.CHUNK_DURATION)
        
        # Contextual data used for logging
        self.video_context = f"[Video: {self.video_filename[:15]}...]" 

        self.subtitle_path = self._generate_subtitle_filepath()
        
        # Critical state management
        self.subtitle_segments = []
        self.subtitle_lock = threading.Lock()
        
        self.task_queue = PriorityQueue()
        self.queue_lock = threading.Lock()
        self.processed_chunks = set()
        self.queued_chunks = set()
        self.last_known_time = 0.0

        # Pointers for idle background queueing
        self.idle_pointer_start = 0
        self.idle_pointer_end = self.total_chunks
        self.idle_from_start = True

        self._initialize_subtitle_file()

        self.worker_thread = threading.Thread(target=self._worker, daemon=True, name=f"AIEngine-{self.video_filename[:10]}")
        self.worker_thread.start()

        logging.info(f"{self.video_context} Engine initialized (Duration: {self.duration:.2f}s, Chunks: {self.total_chunks+1}).")

    def _format_ass_timestamp(self, total_seconds: float) -> str:
        centiseconds = int((total_seconds % 1) * 100)
        total_seconds = int(total_seconds)
        seconds = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    def _generate_subtitle_filepath(self) -> str:
        base_name = os.path.splitext(self.video_path)[0]
        return f"{base_name}.ai.ass"

    def _initialize_subtitle_file(self):
        # The ASS header defines the subtitle format and default style.
        # This structure is required for mpv to correctly interpret the file.
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
        with self.queue_lock:
            while not self.task_queue.empty():
                try: self.task_queue.get_nowait()
                except Empty: break
            self.queued_chunks.clear()
            logging.info(f"{self.video_context} Seek detected. Task queue cleared.")

    def process_update(self, current_time: float):
        if current_time is None: return

        # Detect large seeks (jumps greater than one chunk duration)
        if abs(current_time - self.last_known_time) > self.CHUNK_DURATION:
            logging.info(f"{self.video_context} Large seek detected from {self.last_known_time:.2f}s to {current_time:.2f}s. Resetting queue.")
            self._clear_queue_on_seek()

        self.last_known_time = current_time

        # Define the 'active horizon' of chunks immediately surrounding the current playback position.
        # These chunks are prioritized for immediate transcription.
        active_horizon = [-1, 0, 1, 2, 3]
        current_chunk_index = int(current_time / self.CHUNK_DURATION)

        for offset in active_horizon:
            chunk_index = current_chunk_index + offset
            if not (0 <= chunk_index <= self.total_chunks):
                continue

            chunk_start_time = chunk_index * self.CHUNK_DURATION

            with self.queue_lock:
                if chunk_start_time in self.processed_chunks or chunk_start_time in self.queued_chunks:
                    logging.debug(f"{self.video_context} Chunk {chunk_index} already processed or queued.")
                    continue

                # Priority is based on proximity to current time (lower is better).
                priority = abs(chunk_start_time - current_time)
                chunk_end_time = min(chunk_start_time + self.CHUNK_DURATION, self.duration)

                logging.info(f"{self.video_context} Queueing ACTIVE chunk {chunk_index} ({chunk_start_time:.2f}s) with priority {priority:.2f}")
                self.queued_chunks.add(chunk_start_time)
                self.task_queue.put((priority, (chunk_start_time, chunk_end_time)))

    def _worker(self):
        logging.info(f"{self.video_context} Worker thread started.")
        while True:
            try:
                priority, (start_time, end_time) = self.task_queue.get(timeout=2.0)
                logging.info(f"{self.video_context} Processing task: Chunk at {start_time:.2f}s (P={priority:.2f})")
                self._process_one_chunk(start_time, end_time)
            except Empty:
                # If the active queue is empty, attempt to fill the queue with idle chunks.
                logging.debug(f"{self.video_context} Queue empty. Attempting to queue idle chunk.")
                self._queue_idle_chunk()

    def _queue_idle_chunk(self):
        with self.queue_lock:
            if len(self.processed_chunks) + len(self.queued_chunks) > self.total_chunks:
                return

            chunk_to_queue = -1

            # Alternate between scanning from the start and scanning from the end
            # to ensure full coverage of the video file during idle time.
            if self.idle_from_start:
                while self.idle_pointer_start <= self.total_chunks:
                    start_time = self.idle_pointer_start * self.CHUNK_DURATION
                    if start_time not in self.processed_chunks and start_time not in self.queued_chunks:
                        chunk_to_queue = self.idle_pointer_start
                        self.idle_pointer_start += 1
                        break
                    self.idle_pointer_start += 1
            else:
                while self.idle_pointer_end >= 0:
                    start_time = self.idle_pointer_end * self.CHUNK_DURATION
                    if start_time not in self.processed_chunks and start_time not in self.queued_chunks:
                        chunk_to_queue = self.idle_pointer_end
                        self.idle_pointer_end -= 1
                        break
                    self.idle_pointer_end -= 1

            self.idle_from_start = not self.idle_from_start

            if chunk_to_queue != -1:
                start_time = chunk_to_queue * self.CHUNK_DURATION
                end_time = min(start_time + self.CHUNK_DURATION, self.duration)
                # Assign a very high priority to idle tasks to ensure they are only run
                # when the active queue is completely empty.
                priority = 1000000.0 + chunk_to_queue 

                logging.info(f"{self.video_context} Queueing IDLE chunk {chunk_to_queue} ({start_time:.2f}s) with priority {priority:.2f}")
                self.queued_chunks.add(start_time)
                self.task_queue.put((priority, (start_time, end_time)))

    def _extract_audio_chunk(self, start_time: float, end_time: float) -> bytes | None:
        chunk_duration = end_time - start_time
        # FFMPEG command extracts raw PCM audio (s16le, 16kHz, mono) suitable for WhisperX.
        # Using '-' as the output file writes the raw data to stdout.
        command = [
            'ffmpeg', '-y', '-ss', str(start_time), '-i', self.video_path,
            '-t', str(chunk_duration), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-f', 's16le', '-'
        ]
        try:
            logging.debug(f"{self.video_context} Extracting audio chunk {start_time:.2f}s to {end_time:.2f}s using FFMPEG.")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            audio_bytes, _ = process.communicate()
            if process.returncode != 0: 
                logging.warning(f"{self.video_context} FFMPEG failed to extract audio chunk {start_time:.2f}s (Return code: {process.returncode}).")
                return None
            logging.debug(f"{self.video_context} Extracted {len(audio_bytes)} bytes of audio.")
            return audio_bytes
        except (subprocess.CalledProcessError, FileNotFoundError) as e: 
            logging.error(f"{self.video_context} FFMPEG execution failed for chunk {start_time:.2f}s: {e}")
            return None

    def _refine_and_resegment(self, original_results: list[dict]) -> list[dict]:
        """
        Refines and re-segments transcription results based on professional subtitling standards.
        This logic ensures readability by enforcing character limits, minimum/maximum durations,
        and splitting lines based on significant pauses.
        """
        CPS_RATE = 15
        MAX_CPL = 42
        MIN_DURATION_S = 1.0
        MAX_DURATION_S = 7.0
        PAUSE_THRESHOLD_S = 0.5
        MIN_GAP_S = 0.15

        START_CUSHION_S = 0.15
        END_CUSHION_S = 0.10

        final_segments = []

        for segment in original_results:
            if not 'words' in segment or not segment['words']:
                logging.warning(f"Refinement failed: Segment lacks word-level timestamps. Text: \"{segment.get('text', '')[:50]}...\"")
                continue

            words = segment['words']
            current_line_words = []

            for i, word_data in enumerate(words):
                if not all(k in word_data for k in ['word', 'start', 'end']): continue

                potential_line_words = current_line_words + [word_data]
                potential_text = " ".join([w['word'].strip() for w in potential_line_words])

                violated_cpl = len(potential_text) > MAX_CPL

                is_pause_split = False
                if current_line_words:
                    prev_word_end = current_line_words[-1]['end']
                    current_word_start = word_data['start']
                    # Split if the gap between words exceeds the pause threshold.
                    if (current_word_start - prev_word_end) > PAUSE_THRESHOLD_S:
                        is_pause_split = True

                if current_line_words and (violated_cpl or is_pause_split):
                    text = " ".join([w['word'].strip() for w in current_line_words])

                    raw_start_time = current_line_words[0]['start']
                    raw_end_time = current_line_words[-1]['end']

                    # Apply cushions to make the subtitle appear slightly before and after the speech.
                    start_time = raw_start_time - START_CUSHION_S
                    end_time = raw_end_time + END_CUSHION_S

                    final_segments.append({'start': start_time, 'end': end_time, 'text': text, 'words': current_line_words})

                    current_line_words = [word_data]
                else:
                    current_line_words = potential_line_words

            if current_line_words:
                text = " ".join([w['word'].strip() for w in current_line_words])

                raw_start_time = current_line_words[0]['start']
                raw_end_time = current_line_words[-1]['end']

                start_time = raw_start_time - START_CUSHION_S
                end_time = raw_end_time + END_CUSHION_S

                final_segments.append({'start': start_time, 'end': end_time, 'text': text, 'words': current_line_words})

        if not final_segments: return []

        # Post-processing: Adjust duration based on reading speed (CPS) and minimum/maximum limits.
        for seg in final_segments:
            text_len = len(seg['text'])
            current_duration = seg['end'] - seg['start']

            # Aggressively shorten segments that are too long relative to their content length.
            if text_len < 15 and current_duration > (text_len * 0.4): 
                if seg['words']:
                    target_end = seg['words'][-1]['end'] + END_CUSHION_S
                    if target_end < seg['end'] - 0.1:
                        seg['end'] = target_end
                        logging.debug(f"Refinement: Aggressively shortened end time for: \"{seg['text'][:30]}...\"")

        for seg in final_segments:
            num_chars = len(seg['text'])
            required_cps_duration = num_chars / CPS_RATE
            current_duration = seg['end'] - seg['start']

            target_duration = max(current_duration, required_cps_duration, MIN_DURATION_S)
            final_duration = min(target_duration, MAX_DURATION_S)

            if final_duration > current_duration:
                seg['end'] = seg['start'] + final_duration

        # Post-processing: Ensure segments do not overlap or have insufficient gap.
        for i in range(len(final_segments) - 1):
            current_seg = final_segments[i]
            next_seg = final_segments[i+1]

            # If segments overlap or the gap is too small, force the current segment to end early.
            if current_seg['end'] > (next_seg['start'] - MIN_GAP_S):
                current_seg['end'] = next_seg['start'] - MIN_GAP_S

            current_seg['start'] = max(0.0, current_seg['start'])

        return final_segments

    def _process_one_chunk(self, start_time: float, end_time: float):
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
                for item in refined_results:
                    abs_start = start_time + item['start']
                    abs_end = start_time + item['end']
                    text = item['text'].strip()
                    if not text: continue

                    if abs_end <= abs_start:
                        abs_end = abs_start + MIN_DURATION_S

                    new_segment = (abs_start, abs_end, text)
                    
                    # Use bisect.insort to maintain the list of segments sorted by start time,
                    # which is crucial for correct ASS file generation order.
                    bisect.insort(self.subtitle_segments, new_segment, key=lambda x: x[0])

                if refined_results:
                    logging.info(f"{self.video_context} Added {len(refined_results)} segments from chunk {start_time:.2f}s. Total segments: {len(self.subtitle_segments)}.")
                    self._regenerate_subtitle_file()

            self.on_chunk_completed(self.subtitle_path)
        except Exception as e:
            logging.error(f"{self.video_context} Critical error during chunk transcription ({start_time:.2f}s): {e}", exc_info=True)
        finally:
            with self.queue_lock:
                self.queued_chunks.discard(start_time)
                self.processed_chunks.add(start_time)
            self.task_queue.task_done()

    def _regenerate_subtitle_file(self):
        logging.info(f"{self.video_context} Regenerating subtitle file with {len(self.subtitle_segments)} total segments.")
        header = (
            "[Script Info]\nTitle: AI Generated Subtitles\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        lines_to_write = []
        for start, end, text in self.subtitle_segments:
            start_str = self._format_ass_timestamp(start)
            end_str = self._format_ass_timestamp(end)
            formatted_text = text.replace('\n', ' ').strip()
            lines_to_write.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{formatted_text}")
        try:
            with open(self.subtitle_path, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write("\n".join(lines_to_write) + "\n")
            logging.debug(f"{self.video_context} Subtitle file write complete.")
        except IOError as e:
            logging.error(f"{self.video_context} Failed to regenerate subtitle file: {e}", exc_info=True)