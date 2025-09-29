import sys
import time
import argparse
import logging
import os
import tempfile
import atexit
import threading
import subprocess
from contextlib import suppress

try:
    import portalocker
    _HAS_PORTALOCKER = True
except ImportError:
    portalocker = None
    _HAS_PORTALOCKER = False

from python_mpv_jsonipc import MPV, MPVError
from ai_engine import AITranscriptionEngine
from transcription import ParakeetLocalClient

# Logging to file and stdout
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, 'subtitle_service.log')
os.chdir(script_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

CLIENT_NAME = "ai_subtitle_service"
CONFIG = {"CHUNK_DURATION_SECONDS": 300}
LOCK_FILE_PATH = os.path.join(tempfile.gettempdir(), 'mpv_ai_subtitle_service.lock')
_lock_file_handle = None


def _is_pid_running(pid: int) -> bool:
    """True if PID exists."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # Use tasklist command on Windows
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}"],
                stderr=subprocess.DEVNULL
            ).decode(errors='ignore')
            return str(pid) in output
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    else:
        # Use kill -0 on Unix
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True


def acquire_lock():
    global _lock_file_handle
    if os.path.exists(LOCK_FILE_PATH):
        try:
            with open(LOCK_FILE_PATH, 'r', encoding='utf-8') as f:
                stale_pid_str = f.read().strip()
                if stale_pid_str.isdigit():
                    stale_pid = int(stale_pid_str)
                    if not _is_pid_running(stale_pid):
                        logging.warning(f"[Lock] Found stale lock file from dead PID {stale_pid}. Removing it.")
                        os.remove(LOCK_FILE_PATH)
                    else:
                        logging.warning(f"[Lock] Lock file {LOCK_FILE_PATH} exists and PID {stale_pid} is running.")
                        return False
                else:
                    # Corrupted lock file
                    logging.warning("[Lock] Found corrupted lock file. Removing it.")
                    os.remove(LOCK_FILE_PATH)
        except (IOError, ValueError) as e:
            logging.error(f"[Lock] Error checking stale lock file, removing it: {e}")
            try:
                os.remove(LOCK_FILE_PATH)
            except OSError:
                pass # Ignore if removal fails

    # Try portalocker, else write PID file
    if _HAS_PORTALOCKER and portalocker is not None:
        try:
            fh = open(LOCK_FILE_PATH, 'w', encoding='utf-8')
            portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
            fh.write(str(os.getpid()))
            fh.flush()
            _lock_file_handle = fh
            logging.info(f"[Lock] Portalocker lock acquired; PID written.")
            return True
        except (IOError, OSError, portalocker.exceptions.LockException) as e:
            logging.warning(f"[Lock] Portalocker lock failed: {e}")
            with suppress(Exception): fh.close()
            return False
    try:
        _lock_file_handle = open(LOCK_FILE_PATH, 'w', encoding='utf-8')
        _lock_file_handle.write(str(os.getpid()))
        _lock_file_handle.flush()
        logging.info(f"[Lock] PID file written (fallback).")
        return True
    except IOError:
        logging.critical("[Lock] Failed to write PID file.", exc_info=True)
        return False


def release_lock():
    global _lock_file_handle
    if _lock_file_handle:
        try:
            if _HAS_PORTALOCKER and portalocker is not None:
                portalocker.unlock(_lock_file_handle)
            _lock_file_handle.close()
            _lock_file_handle = None
        except Exception as e:
            logging.warning(f"[Lock] Error releasing lock: {e}")

    if os.path.exists(LOCK_FILE_PATH):
        try:
            os.remove(LOCK_FILE_PATH)
            logging.info("[Lock] Lock file removed.")
        except OSError as e:
            logging.warning(f"[Lock] Could not remove lock file: {e}.")


class SubtitleService:
    def __init__(self, socket_path: str):
        self.running = True
        self.shutdown_event = threading.Event()
        self.player = None
        self.engines = {}
        self.current_engine = None
        self.observer_ids = []
        self.transcription_model = self._load_ai_model()
        self.ai_subtitle_track_id = None  # AI subtitle track id

        try:
            self.player = MPV(start_mpv=False, ipc_socket=socket_path)
            logging.info(f"[MPV] Connected to mpv via socket: {socket_path}")
        except Exception:
            logging.critical("[MPV] Fatal: Could not connect to mpv socket. Exiting.", exc_info=True)
            sys.exit(1)

        self._setup_observers_and_events()
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True, name="MPVHeartbeat")
        self.heartbeat_thread.start()

        logging.info(f"[Service] Python service registered as '{CLIENT_NAME}' and is ready.")

    def _load_ai_model(self):
        logging.info("[Model] Init Parakeet client")
        try:
            model = ParakeetLocalClient()
            logging.info(f"[Model] {model.__class__.__name__} initialized")
            return model
        except Exception as e:
            logging.critical(f"[Model] Fatal: Failed to load AI model: {e}", exc_info=True)
            sys.exit(1)

    def _heartbeat_worker(self):
        logging.info("[Heartbeat] started")
        while self.running:
            time.sleep(2.0)
            try:
                _ = self.player.pause
            except (MPVError, BrokenPipeError, AttributeError):
                logging.warning("[Heartbeat] MPV connection lost. Shutting down.")
                self.shutdown()
                break
        logging.info("[Heartbeat] stopped")
        
    def _setup_observers_and_events(self):
        # Bind mpv events
        self.player.bind_event('shutdown', self.handle_shutdown)
        self.player.bind_event('client-message', self.handle_message)
        self.observer_ids.append(self.player.bind_property_observer('path', self._on_path_change))
        self.observer_ids.append(self.player.bind_property_observer('time-pos', self._on_time_pos_change))
        self.observer_ids.append(self.player.bind_property_observer('pause', self._on_pause_change))
        self.player.command("script-binding", CLIENT_NAME)

    def run(self):
        logging.info("[Service] running; awaiting events")
        try:
            self.shutdown_event.wait()
        except (KeyboardInterrupt, SystemExit):
            logging.info("[Service] Received exit signal.")
        finally:
            self.shutdown()
        self._cleanup()
        logging.info("[Service] Python service has shut down.")
        
    def _cleanup(self):
        logging.info("[Service] cleaning up")
        for engine in list(self.engines.values()):
            engine.shutdown()
        if self.player:
            try:
                for obs_id in self.observer_ids:
                    self.player.unbind_property_observer(obs_id)
                self.player.terminate()
            except (MPVError, BrokenPipeError):
                pass
            finally:
                self.player = None

    def _on_path_change(self, _, path: str | None):
        if self.current_engine:
            logging.info(f"[Engine] Cleaning old engine for '{os.path.basename(self.current_engine.video_path)}'.")
            if self.current_engine.video_path in self.engines:
                self.engines[self.current_engine.video_path].shutdown()
                del self.engines[self.current_engine.video_path]

        self.current_engine = None
        self.ai_subtitle_track_id = None  # reset for new video

        if path is None:
            logging.info("[Video] stopped or closed")
            return

        logging.info(f"[Video] loaded: {os.path.basename(path)}")
        if path not in self.engines:
            try:
                video_params = self.player.video_params
                video_metadata = {
                    'path': path,
                    'duration': self.player.duration,
                    'filename': self.player.filename,
                    'width': video_params.get('w'),
                    'height': video_params.get('h')
                }
                self.engines[path] = AITranscriptionEngine(
                    video_metadata, CONFIG, self.transcription_model, self._on_subtitle_update
                )
                logging.info(f"[Engine] New transcription engine for '{os.path.basename(path)}'.")
            except (RuntimeError, MPVError, BrokenPipeError) as e:
                logging.error(f"[Engine] Init failed: {e}")
                return
        else:
            logging.info(f"[Engine] Reusing engine for '{os.path.basename(path)}'.")

        self.current_engine = self.engines[path]
        self._on_subtitle_update(self.current_engine.subtitle_path)

    def _on_time_pos_change(self, _, time_pos: float | None):
        if self.current_engine is None or time_pos is None or self.current_engine.is_finished.is_set():
            return
        if not self.player.pause:
            self.current_engine.process_update(time_pos)

    def _on_pause_change(self, _, is_paused: bool):
        if self.current_engine and not self.current_engine.is_finished.is_set():
            logging.info(f"[Playback] {'paused' if is_paused else 'resumed'}")
            if not is_paused:
                self.current_engine.process_update(self.player.time_pos)

    def _get_ai_subtitle_track(self, subtitle_path: str) -> dict | None:
        try:
            abs_subtitle_path = os.path.abspath(subtitle_path)
            for track in self.player.track_list:
                if track.get('type') == 'sub':
                    track_filename = track.get('external-filename')
                    if track_filename and os.path.abspath(track_filename) == abs_subtitle_path:
                        return track
        except (MPVError, BrokenPipeError, AttributeError) as e:
            logging.warning(f"[MPV] Could not query track list: {e}")
        return None

    def _on_subtitle_update(self, subtitle_path: str):
        """
        Add or reload the AI subtitle track, respecting the user's current selection.
        If subtitles are disabled, it adds the track without selecting it.
        If subtitles are enabled, it adds and selects the track.
        If the track already exists, it just reloads the content.
        """
        if not self.player or not os.path.exists(subtitle_path):
            return

        try:
            # If we already know the track ID, just reload it
            if self.ai_subtitle_track_id is not None:
                logging.debug(f"[Sub] Reloading AI track id={self.ai_subtitle_track_id}.")
                self.player.sub_reload(self.ai_subtitle_track_id)
                return

            # If ID is unknown, find the track or add it for the first time ---
            ai_track = self._get_ai_subtitle_track(subtitle_path)

            if ai_track:
                # The track is already loaded but we didn't have its ID. Store it now.
                self.ai_subtitle_track_id = ai_track.get('id')
                logging.info(f"[Sub] Found existing AI track, stored id={self.ai_subtitle_track_id}")
                self.player.sub_reload(self.ai_subtitle_track_id)
            else:
                # The track is not loaded. We need to add it.
                is_sub_active = self.player.sid != 'no' and self.player.sub_visibility

                # If subtitles are active, we'll select our new track
                add_flag = 'select' if is_sub_active else 'auto'
                logging.info(f"[Sub] Adding new AI track with flag '{add_flag}' (user subs active: {is_sub_active}).")
                self.player.sub_add(subtitle_path, add_flag)

                # Give mpv a moment to process before we query for the new track's ID
                time.sleep(0.2)

                # Now, find the newly added track to get its ID for future reloads
                newly_added_track = self._get_ai_subtitle_track(subtitle_path)
                if newly_added_track:
                    self.ai_subtitle_track_id = newly_added_track.get('id')
                    logging.info(f"[Sub] Successfully added AI track, stored id={self.ai_subtitle_track_id}")
                else:
                    logging.warning("[Sub] Could not find the AI subtitle track immediately after adding it.")

        except (MPVError, BrokenPipeError) as e:
            logging.warning(f"[MPV] Subtitle interaction failed: {e}")
            self.ai_subtitle_track_id = None
        except Exception as e:
            logging.error(f"[Sub] Unexpected error during subtitle update: {e}", exc_info=True)

    def shutdown(self):
        if self.running:
            self.running = False
            self.shutdown_event.set()

    def handle_message(self, event: dict):
        args = event.get("args", [])
        if args and args[0] == "ai-service-event" and args[1] == '"stop"':
            logging.info("[Service] Received 'stop' command via IPC.")
            self.shutdown()

    def handle_shutdown(self, event=None):
        logging.info("[Service] MPV shutdown event received.")
        self.shutdown()


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