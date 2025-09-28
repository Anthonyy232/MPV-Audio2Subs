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

# Configure logging to file and stdout for diagnostics
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, 'subtitle_service.log')

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


def _is_pid_running(pid: int) -> bool:
    """Return True if a process with the given PID exists."""
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
        # Use kill -0 on Unix-like systems
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

    # Attempt a platform-aware exclusive lock, fall back to pid file write
    if _HAS_PORTALOCKER and portalocker is not None:
        try:
            fh = open(LOCK_FILE_PATH, 'w', encoding='utf-8')
            portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
            fh.write(str(os.getpid()))
            fh.flush()
            _lock_file_handle = fh
            logging.info(f"[Lock] Acquired portalocker lock and wrote PID {os.getpid()} to {LOCK_FILE_PATH}.")
            return True
        except (IOError, OSError, portalocker.exceptions.LockException) as e:
            logging.warning(f"[Lock] Could not acquire portalocker lock: {e}")
            with suppress(Exception): fh.close()
            return False
    try:
        _lock_file_handle = open(LOCK_FILE_PATH, 'w', encoding='utf-8')
        _lock_file_handle.write(str(os.getpid()))
        _lock_file_handle.flush()
        logging.info(f"[Lock] Wrote PID {os.getpid()} to {LOCK_FILE_PATH} (fallback lock).")
        return True
    except IOError:
        logging.critical("[Lock] Failed to write fallback lock file. Check permissions.", exc_info=True)
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
            logging.warning(f"[Lock] Error releasing lock handle: {e}")

    if os.path.exists(LOCK_FILE_PATH):
        try:
            os.remove(LOCK_FILE_PATH)
            logging.info("[Lock] Released lock and removed lock file.")
        except OSError as e:
            logging.warning(f"[Lock] Could not remove lock file on exit: {e}.")


class SubtitleService:
    def __init__(self, socket_path: str):
        self.running = True
        self.shutdown_event = threading.Event()
        self.player = None
        self.engines = {}
        self.current_engine = None
        self.observer_ids = []
        self.transcription_model = self._load_ai_model()
        self.subtitle_file_loaded = False

        try:
            self.player = MPV(start_mpv=False, ipc_socket=socket_path)
            logging.info(f"[MPV] Connected to mpv via socket: {socket_path}")
        except Exception:
            logging.critical("[MPV] Fatal: Could not connect to mpv socket. Exiting.", exc_info=True)
            sys.exit(1)

        self._setup_observers_and_events()
        # Start a background heartbeat to detect lost MPV connection
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True, name="MPVHeartbeat")
        self.heartbeat_thread.start()

        logging.info(f"[Service] Python service registered as '{CLIENT_NAME}' and is ready.")

    def _load_ai_model(self):
        logging.info("[Model] Initializing Parakeet Local Client...")
        try:
            model = ParakeetLocalClient()
            logging.info(f"[Model] AI model client '{model.__class__.__name__}' initialized successfully.")
            return model
        except Exception as e:
            logging.critical(f"[Model] Fatal: Failed to load AI model: {e}", exc_info=True)
            sys.exit(1)

    def _heartbeat_worker(self):
        """Background monitor that checks MPV connection and triggers shutdown if lost."""
        logging.info("[Heartbeat] Heartbeat monitor started.")
        while self.running:
            time.sleep(2.0)
            try:
                # Lightweight property access to validate the IPC connection
                _ = self.player.pause
            except (MPVError, BrokenPipeError, AttributeError):
                logging.warning("[Heartbeat] Connection to MPV lost. Triggering shutdown.")
                self.shutdown()
                break
        logging.info("[Heartbeat] Heartbeat monitor stopped.")
        
    def _setup_observers_and_events(self):
        # Bind MPV events and property observers
        self.player.bind_event('shutdown', self.handle_shutdown)
        self.player.bind_event('client-message', self.handle_message)
        self.observer_ids.append(self.player.bind_property_observer('path', self._on_path_change))
        self.observer_ids.append(self.player.bind_property_observer('time-pos', self._on_time_pos_change))
        self.observer_ids.append(self.player.bind_property_observer('pause', self._on_pause_change))
        self.player.command("script-binding", CLIENT_NAME)

    def run(self):
        logging.info("[Service] Service is now running. Waiting for mpv events...")
        try:
            # This will block until shutdown_event is set
            self.shutdown_event.wait()
        except (KeyboardInterrupt, SystemExit):
            logging.info("[Service] Received exit signal.")
        finally:
            self.shutdown()

        self._cleanup()
        logging.info("[Service] Python service has shut down.")
        
    def _cleanup(self):
        logging.info("[Service] Cleaning up resources...")
        for engine in list(self.engines.values()):
            # Shutdown each engine (idempotent)
            engine.shutdown()
        if self.player:
            try:
                for obs_id in self.observer_ids: self.player.unbind_property_observer(obs_id)
                self.player.terminate()
            except (MPVError, BrokenPipeError): pass
            finally: self.player = None

    def _on_path_change(self, _, path: str | None):
        if self.current_engine:
            logging.info(f"[Engine] Cleaning up old engine for '{os.path.basename(self.current_engine.video_path)}'.")
            if self.current_engine.video_path in self.engines:
                self.engines[self.current_engine.video_path].shutdown()
                del self.engines[self.current_engine.video_path]

        self.subtitle_file_loaded = False
        self.current_engine = None

        if path is None:
            logging.info("[Video] Player stopped or file closed.")
            return

        logging.info(f"[Video] New video loaded: {os.path.basename(path)}")
        if path not in self.engines:
            try:
                video_params = self.player.video_params
                video_metadata = {
                    'path': path, 'duration': self.player.duration, 'filename': self.player.filename,
                    'width': video_params.get('w'), 'height': video_params.get('h')
                }
                self.engines[path] = AITranscriptionEngine(
                    video_metadata, CONFIG, self.transcription_model, self._on_subtitle_update
                )
                logging.info(f"[Engine] Created new transcription engine for '{os.path.basename(path)}'.")
            except (RuntimeError, MPVError, BrokenPipeError) as e:
                logging.error(f"[Engine] Failed to initialize engine: {e}")
                return
        else:
            logging.info(f"[Engine] Reusing existing transcription engine for '{os.path.basename(path)}'.")

        self.current_engine = self.engines[path]
        self._on_subtitle_update(self.current_engine.subtitle_path)

    def _on_time_pos_change(self, _, time_pos: float | None):
        if self.current_engine is None or time_pos is None or self.current_engine.is_finished.is_set():
            return
        if not self.player.pause:
            self.current_engine.process_update(time_pos)

    def _on_pause_change(self, _, is_paused: bool):
        if self.current_engine and not self.current_engine.is_finished.is_set():
            logging.info(f"[Playback] Playback {'paused' if is_paused else 'resumed'}.")
            if not is_paused:
                self.current_engine.process_update(self.player.time_pos)

    def _on_subtitle_update(self, subtitle_path: str):
        if not os.path.exists(subtitle_path):
            return
        try:
            if not self.subtitle_file_loaded:
                self.player.sub_add(subtitle_path)
                logging.info(f"[Sub] Added new AI subtitle file: {os.path.basename(subtitle_path)}")
                self.subtitle_file_loaded = True
            else:
                logging.info("[Sub] Subtitle file has been updated. Reloading.")
                self.player.sub_reload()
        except (MPVError, BrokenPipeError) as e:
            logging.warning(f"[MPV] Failed to interact with subtitle track during update: {e}")
        except Exception as e:
            logging.error(f"[Sub] An unexpected error occurred during subtitle update: {e}", exc_info=True)

    def shutdown(self):
        """Idempotent method to initiate a graceful shutdown."""
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