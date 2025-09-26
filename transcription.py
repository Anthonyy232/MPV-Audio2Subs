import logging
import requests
import time
from abc import ABC, abstractmethod
import json

class TranscriptionInterface(ABC):
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