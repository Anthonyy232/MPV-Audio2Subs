import sys
import time
import argparse
import logging
import os
import tempfile
import atexit
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

def acquire_lock():
    """
    Acquires a file lock to ensure only one instance of the Python client
    is running at a time, preventing conflicts with MPV IPC and resource usage.
    """
    if os.path.exists(LOCK_FILE_PATH):
        logging.warning(f"[Lock] Lock file {LOCK_FILE_PATH} already exists. Checking owner PID.")
        try:
            with open(LOCK_FILE_PATH, 'r') as f:
                pid = f.read().strip()
                logging.warning(f"[Lock] Lock file owned by PID: {pid}. Cannot start.")
        except IOError:
            pass
        return False
    try:
        with open(LOCK_FILE_PATH, 'w') as f:
            f.write(str(os.getpid()))
        logging.info(f"[Lock] Acquired lock file with PID {os.getpid()}.")
        return True
    except IOError:
        logging.critical("[Lock] Failed to acquire lock file. Check permissions.", exc_info=True)
        return False

def release_lock():
    if os.path.exists(LOCK_FILE_PATH):
        try:
            with open(LOCK_FILE_PATH, 'r') as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                os.remove(LOCK_FILE_PATH)
                logging.info("[Lock] Lock file released.")
            else:
                logging.warning(f"[Lock] Not releasing lock file (owned by PID {pid}).")
        except (IOError, ValueError):
            try:
                os.remove(LOCK_FILE_PATH)
                logging.warning("[Lock] Lock file released (owner PID verification failed).")
            except OSError as e:
                logging.error(f"[Lock] Error removing lock file: {e}")

class SubtitleService:
    def __init__(self, socket_path: str):
        self.running = True
        self.player = None
        # Caches transcription engines per video path to avoid re-transcribing
        # the entire file if the user closes and reopens the same video.
        self.engines = {}
        self.current_engine = None
        self.subtitle_file_loaded = False
        self.observer_ids = []
        self.transcription_model = self._load_ai_model()

        try:
            self.player = MPV(start_mpv=False, ipc_socket=socket_path)
            logging.info(f"[MPV] Connected to mpv via socket: {socket_path}")
        except Exception:
            logging.critical("[MPV] Fatal: Could not connect to mpv socket. Exiting.", exc_info=True)
            sys.exit(1)

        self._setup_observers_and_events()
        logging.info(f"[Service] Python service registered as '{CLIENT_NAME}' and is ready.")

    def _load_ai_model(self):
        logging.info("[Model] Initializing WhisperX Web Client...")
        try:
            model = WhisperXWebClient()
            logging.info(f"[Model] AI model client '{model.__class__.__name__}' initialized successfully.")
            return model
        except Exception as e:
            logging.critical(f"[Model] Fatal: Failed to connect to AI server: {e}", exc_info=True)
            sys.exit(1)

    def _setup_observers_and_events(self):
        self.player.bind_event('shutdown', self.handle_shutdown)
        self.player.bind_event('client-message', self.handle_message)
        self.observer_ids.append(self.player.bind_property_observer('path', self._on_path_change))
        self.observer_ids.append(self.player.bind_property_observer('time-pos', self._on_time_pos_change))
        self.observer_ids.append(self.player.bind_property_observer('pause', self._on_pause_change))
        self.player.command("script-binding", CLIENT_NAME)

    def run(self):
        logging.info("[Service] Service is now running. Waiting for mpv events...")
        while self.running:
            try:
                # Heartbeat check to detect if the MPV connection is still alive.
                _ = self.player.time_pos
                time.sleep(1)
            except (MPVError, BrokenPipeError):
                logging.warning("[MPV] Connection to mpv lost. Shutting down.")
                self.running = False
        self._cleanup()
        logging.info("[Service] Python service has shut down.")

    def _cleanup(self):
        logging.info("[Service] Cleaning up resources...")
        if self.player:
            try:
                for obs_id in self.observer_ids: self.player.unbind_property_observer(obs_id)
                self.player.terminate()
                logging.debug("[MPV] MPV connection terminated.")
            except (MPVError, BrokenPipeError): 
                logging.debug("[MPV] MPV connection already broken during cleanup.")
            finally: self.player = None

    def _on_path_change(self, _, path: str | None):
        self.subtitle_file_loaded = False
        if path is None:
            logging.info("[Video] Player stopped or file closed.")
            self.current_engine = None
            return

        logging.info(f"[Video] New video loaded: {os.path.basename(path)}")
        if path not in self.engines:
            video_metadata = {'path': path, 'duration': self.player.duration, 'filename': self.player.filename}
            self.engines[path] = AITranscriptionEngine(video_metadata, CONFIG, self.on_chunk_completed, self.transcription_model)
            logging.info(f"[Engine] Created new transcription engine for '{os.path.basename(path)}'.")
        else:
            logging.info(f"[Engine] Reusing existing transcription engine for '{os.path.basename(path)}'.")
            
        self.current_engine = self.engines[path]
        self.on_chunk_completed(self.current_engine.subtitle_path)

    def _on_time_pos_change(self, _, time_pos: float | None):
        # High-frequency event, only process if playing.
        if self.current_engine and not self.player.pause and time_pos is not None:
            logging.debug(f"[Time] Current time position: {time_pos:.2f}s. Processing update.")
            self.current_engine.process_update(time_pos)

    def _on_pause_change(self, _, is_paused: bool):
        if self.current_engine:
            logging.info(f"[Playback] Playback {'paused' if is_paused else 'resumed'}.")
            if not is_paused: self.current_engine.process_update(self.player.time_pos)

    def on_chunk_completed(self, subtitle_path: str):
        try:
            if not os.path.exists(subtitle_path): 
                logging.debug(f"[Sub] Subtitle file not found at {subtitle_path}.")
                return
            
            if not self.subtitle_file_loaded:
                track_list = self.player.track_list
                sub_already_loaded = any(t.get("type") == "sub" and t.get("path") == subtitle_path for t in track_list)
                if not sub_already_loaded:
                    # Manually add the subtitle file to MPV's track list.
                    self.player.sub_add(subtitle_path)
                    logging.info(f"[Sub] Added new AI subtitle file: {os.path.basename(subtitle_path)}")
                self.subtitle_file_loaded = True
            else:
                # MPV requires a reload command to display changes made to a subtitle file
                # that is currently loaded, ensuring real-time updates.
                self.player.sub_reload()
                logging.debug("[Sub] Reloaded existing AI subtitle file.")
        except (MPVError, BrokenPipeError) as e: 
            logging.warning(f"[MPV] Failed to interact with subtitle track: {e}")

    def handle_message(self, event: dict):
        args = event.get("args", [])
        if args and args[0] == "ai-service-event" and args[1] == '"stop"':
            logging.info("[Service] Received 'stop' command via IPC.")
            self.running = False

    def handle_shutdown(self, event=None):
        logging.info("[Service] MPV shutdown event received.")
        self.running = False

if __name__ == "__main__":
    # Register the cleanup function to be called automatically upon script exit.
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