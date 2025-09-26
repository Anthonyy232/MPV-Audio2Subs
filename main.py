import sys
import time
import argparse
import logging
import os
import tempfile
import atexit
from contextlib import suppress
try:
    import portalocker
    _HAS_PORTALOCKER = True
except Exception:
    portalocker = None
    _HAS_PORTALOCKER = False
from python_mpv_jsonipc import MPV, MPVError
from ai_engine import AITranscriptionEngine 
from transcription import WhisperXWebClient 

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, 'subtitle_service.log')

# Standardized logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

CLIENT_NAME = "ai_subtitle_service"
CONFIG = {"CHUNK_DURATION_SECONDS": 30}
LOCK_FILE_PATH = os.path.join(tempfile.gettempdir(), 'mpv_ai_subtitle_service.lock')
_lock_file_handle = None


def acquire_lock():
    """Acquires an exclusive, non-blocking lock to ensure only one service instance runs."""
    global _lock_file_handle

    if _HAS_PORTALOCKER and portalocker is not None:
        try:
            fh = open(LOCK_FILE_PATH, 'a+', encoding='utf-8')
            fh.seek(0); fh.truncate(0); fh.write(str(os.getpid())); fh.flush()
            portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
            _lock_file_handle = fh
            logging.info(f"[Lock] Acquired portalocker lock and wrote PID {os.getpid()} to {LOCK_FILE_PATH}.")
            return True
        except (IOError, OSError, portalocker.exceptions.LockException) as e:
            logging.warning(f"[Lock] Could not acquire portalocker lock: {e}")
            with suppress(Exception): fh.close()
            return False

    # Fallback: simple PID file heuristic.
    if os.path.exists(LOCK_FILE_PATH):
        try:
            with open(LOCK_FILE_PATH, 'r', encoding='utf-8') as f:
                pid_text = f.read().strip()
                logging.warning(f"[Lock] Lock file {LOCK_FILE_PATH} exists with contents: '{pid_text}'.")
        except IOError:
            logging.warning(f"[Lock] Lock file {LOCK_FILE_PATH} exists but could not be read.")
        return False

    try:
        with open(LOCK_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(str(os.getpid()))
        logging.info(f"[Lock] Wrote PID {os.getpid()} to {LOCK_FILE_PATH} (fallback lock).")
        return True
    except IOError:
        logging.critical("[Lock] Failed to write fallback lock file. Check permissions.", exc_info=True)
        return False


def release_lock():
    """Releases the process lock and removes the lock file."""
    global _lock_file_handle
    if _lock_file_handle is not None:
        try:
            if _HAS_PORTALOCKER and portalocker is not None:
                portalocker.unlock(_lock_file_handle)
            _lock_file_handle.close()
            logging.info("[Lock] Released portalocker lock and closed handle.")
        except Exception as e:
            logging.debug(f"[Lock] Error releasing portalocker handle: {e}")
        finally: _lock_file_handle = None

    if os.path.exists(LOCK_FILE_PATH):
        try:
            with open(LOCK_FILE_PATH, 'r', encoding='utf-8') as f:
                try: pid = int(f.read().strip())
                except ValueError: pid = None
            if pid is None or pid == os.getpid():
                os.remove(LOCK_FILE_PATH)
                logging.info("[Lock] Lock file removed.")
            else:
                logging.info(f"[Lock] Not removing lock file owned by PID {pid}.")
        except (IOError, OSError) as e:
            logging.debug(f"[Lock] Error checking/removing lock file: {e}")


class SubtitleService:
    """Manages the connection to MPV and coordinates transcription engines."""
    def __init__(self, socket_path: str):
        self.running = True
        self.player = None
        # Caches transcription engines per video path.
        self.engines = {}
        self.current_engine = None
        self.subtitle_file_loaded = False
        self.observer_ids = []
        self.transcription_model = self._load_ai_model()

        # Subtitle Reload Optimization Parameters
        self.RELOAD_DEBOUNCE_SECONDS = 0.5
        self.RELOAD_PROXIMITY_SECONDS = 60.0 
        self.RELOAD_THROTTLE_SECONDS = 5.0 
        
        self._reload_pending = False
        self._last_reload_time = 0.0
        # Tracks the end time of the latest transcribed segment across all chunks.
        self._last_transcribed_end_time = 0.0 

        self.last_known_time_pos = 0.0
        self._pending_fast_reload_after_seek = False

        try:
            self.player = MPV(start_mpv=False, ipc_socket=socket_path)
            logging.info(f"[MPV] Connected to mpv via socket: {socket_path}")
        except Exception:
            logging.critical("[MPV] Fatal: Could not connect to mpv socket. Exiting.", exc_info=True)
            sys.exit(1)

        self._setup_observers_and_events()
        logging.info(f"[Service] Python service registered as '{CLIENT_NAME}' and is ready.")

    def _load_ai_model(self):
        """Initializes the external transcription client."""
        logging.info("[Model] Initializing WhisperX Web Client...")
        try:
            model = WhisperXWebClient()
            logging.info(f"[Model] AI model client '{model.__class__.__name__}' initialized successfully.")
            return model
        except Exception as e:
            logging.critical(f"[Model] Fatal: Failed to connect to AI server: {e}", exc_info=True)
            sys.exit(1)

    def _setup_observers_and_events(self):
        """Binds MPV events and property observers."""
        self.player.bind_event('shutdown', self.handle_shutdown)
        self.player.bind_event('client-message', self.handle_message)
        self.observer_ids.append(self.player.bind_property_observer('path', self._on_path_change))
        self.observer_ids.append(self.player.bind_property_observer('time-pos', self._on_time_pos_change))
        self.observer_ids.append(self.player.bind_property_observer('pause', self._on_pause_change))
        self.player.command("script-binding", CLIENT_NAME)

    def run(self):
        """Main loop to keep the service alive and monitor MPV connection."""
        logging.info("[Service] Service is now running. Waiting for mpv events...")
        while self.running:
            try:
                _ = self.player.time_pos
                time.sleep(1)
            except (MPVError, BrokenPipeError):
                logging.warning("[MPV] Connection to mpv lost. Shutting down.")
                self.running = False
        self._cleanup()
        logging.info("[Service] Python service has shut down.")

    def _cleanup(self):
        """Cleans up MPV connection and ensures all engines delete temporary files."""
        logging.info("[Service] Cleaning up resources...")
        # Ensures all active engines clean up their temporary audio files.
        for engine in self.engines.values():
            engine._cleanup()
            
        if self.player:
            try:
                for obs_id in self.observer_ids: self.player.unbind_property_observer(obs_id)
                self.player.terminate()
                logging.debug("[MPV] MPV connection terminated.")
            except (MPVError, BrokenPipeError): 
                logging.debug("[MPV] MPV connection already broken during cleanup.")
            finally: self.player = None

    def _on_path_change(self, _, path: str | None):
        """Handles video file loading/unloading, initializing or reusing the engine."""
        if self.current_engine:
            self.current_engine._cleanup()
            
        self.subtitle_file_loaded = False
        self._last_transcribed_end_time = 0.0
        self._reload_pending = False
        self._last_reload_time = 0.0
        self.last_known_time_pos = 0.0
        self._pending_fast_reload_after_seek = False

        if path is None:
            logging.info("[Video] Player stopped or file closed.")
            self.current_engine = None
            return

        logging.info(f"[Video] New video loaded: {os.path.basename(path)}")
        if path not in self.engines:
            try:
                video_metadata = {'path': path, 'duration': self.player.duration, 'filename': self.player.filename}
                self.engines[path] = AITranscriptionEngine(video_metadata, CONFIG, self.on_chunk_completed, self.transcription_model)
                logging.info(f"[Engine] Created new transcription engine for '{os.path.basename(path)}'.")
            except RuntimeError as e:
                logging.error(f"[Engine] Failed to initialize engine due to audio extraction error: {e}")
                self.current_engine = None
                return
        else:
            logging.info(f"[Engine] Reusing existing transcription engine for '{os.path.basename(path)}'.")
            
        self.current_engine = self.engines[path]
        max_end_time = self.current_engine.get_max_segment_end_time()
        self.on_chunk_completed(self.current_engine.subtitle_path, max_end_time)

    def _on_time_pos_change(self, _, time_pos: float | None):
        """Handles playback position updates and manages subtitle reloading."""
        if self.current_engine is None or time_pos is None:
            return

        # Detect large seeks (jumps greater than one chunk duration)
        if abs(time_pos - self.last_known_time_pos) > self.current_engine.CHUNK_DURATION:
            logging.info(f"[Service] Large seek detected from {self.last_known_time_pos:.2f}s to {time_pos:.2f}s. Prioritizing next reload.")
            self._pending_fast_reload_after_seek = True
        
        self.last_known_time_pos = time_pos

        if not self.player.pause:
            self.current_engine.process_update(time_pos)

            # --- Subtitle Reload Optimization ---
            if self._reload_pending:
                time_since_last_reload = time.time() - self._last_reload_time
                
                # Heuristic 1: Debounce check (minimum time between reloads)
                if time_since_last_reload < self.RELOAD_DEBOUNCE_SECONDS:
                    logging.debug(f"[Sub] Reload skipped due to debounce ({time_since_last_reload:.2f}s).")
                    return

                # Calculate gap between current playback and the latest transcribed segment end time.
                proximity_gap = self._last_transcribed_end_time - time_pos
                should_reload = False

                if self._pending_fast_reload_after_seek:
                    # Heuristic 2a: Force reload immediately after a seek.
                    logging.info("[Sub] Performing fast reload due to recent seek.")
                    should_reload = True
                    self._pending_fast_reload_after_seek = False
                elif proximity_gap < self.RELOAD_PROXIMITY_SECONDS:
                    # Heuristic 2b: High Priority - Reload if close to the transcription front.
                    should_reload = True
                elif time_since_last_reload >= self.RELOAD_THROTTLE_SECONDS:
                    # Heuristic 2c: Low Priority - Reload if far behind, but the throttle window is met.
                    logging.info(f"[Sub] Far behind transcription front (Gap: {proximity_gap:.2f}s). Performing slow reload.")
                    should_reload = True
                else:
                    logging.debug(f"[Sub] Reload skipped: Far behind transcription front (Gap: {proximity_gap:.2f}s) and throttle window not met.")

                if should_reload:
                    self._perform_subtitle_reload()


    def _on_pause_change(self, _, is_paused: bool):
        """Handles playback pause/resume events."""
        if self.current_engine:
            logging.info(f"[Playback] Playback {'paused' if is_paused else 'resumed'}.")
            if not is_paused: self.current_engine.process_update(self.player.time_pos)

    def _perform_subtitle_reload(self):
        """Executes the MPV sub_reload command and updates state."""
        try:
            self.player.sub_reload()
            self._last_reload_time = time.time()
            self._reload_pending = False
            logging.debug("[Sub] Executed debounced sub_reload.")
        except (MPVError, BrokenPipeError) as e:
            logging.warning(f"[MPV] Failed to execute sub_reload: {e}")

    def on_chunk_completed(self, subtitle_path: str, latest_segment_end_time: float):
        """Callback triggered by the engine when new segments are appended to the file."""
        try:
            if not os.path.exists(subtitle_path): 
                logging.debug(f"[Sub] Subtitle file not found at {subtitle_path}.")
                return
            
            self._last_transcribed_end_time = max(self._last_transcribed_end_time, latest_segment_end_time)

            if not self.subtitle_file_loaded:
                # Add the subtitle file to MPV's track list on first completion.
                track_list = self.player.track_list
                sub_already_loaded = any(t.get("type") == "sub" and t.get("path") == subtitle_path for t in track_list)
                if not sub_already_loaded:
                    self.player.sub_add(subtitle_path)
                    logging.info(f"[Sub] Added new AI subtitle file: {os.path.basename(subtitle_path)}")
                self.subtitle_file_loaded = True
                self._last_reload_time = time.time()
                self._reload_pending = False
            else:
                # Set flag to trigger debounced reload in the time-pos observer.
                self._reload_pending = True
                logging.debug(f"[Sub] Reload pending. Latest transcribed time: {self._last_transcribed_end_time:.2f}s.")

        except (MPVError, BrokenPipeError) as e: 
            logging.warning(f"[MPV] Failed to interact with subtitle track: {e}")

    def handle_message(self, event: dict):
        """Handles custom IPC messages from MPV scripts."""
        args = event.get("args", [])
        if args and args[0] == "ai-service-event" and args[1] == '"stop"':
            logging.info("[Service] Received 'stop' command via IPC.")
            self.running = False

    def handle_shutdown(self, event=None):
        """Handles MPV shutdown event."""
        logging.info("[Service] MPV shutdown event received.")
        self.running = False

if __name__ == "__main__":
    atexit.register(release_lock)

    if not acquire_lock():
        sys.exit(1)
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--socket", required=True)
        args = parser.parse_args()
        service = SubtitleService(socket_path=args.socket)
        service.run()
    except Exception as e:
        logging.critical(f"[Service] An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)