import logging
from abc import ABC, abstractmethod
import numpy as np
import torch

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    nemo_asr = None
    logging.critical("NeMo toolkit not found. Please install it via 'pip install nemo_toolkit[asr]'")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2


class TranscriptionInterface(ABC):
    # Interface so different transcription backends can be swapped seamlessly
    @abstractmethod
    def transcribe(self, audio_buffer: bytes) -> dict:
        pass
    def close(self) -> None:
        logging.info(f"[Model] Closing transcription model: {self.__class__.__name__}")
        pass


class ParakeetLocalClient(TranscriptionInterface):
    # NeMo-backed client intended for local, resource-efficient inference
    def __init__(self):
        if nemo_asr is None:
            raise RuntimeError("NeMo toolkit is not available. Please check installation.")
        logging.info("[Model] Initializing local Parakeet model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"[Model] Using device: {self.device}")
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
            self.model.to(self.device)
            logging.info("[Model] Parakeet model loaded successfully onto device.")
        except Exception as e:
            logging.critical(f"[Model] FATAL: Could not load Parakeet model. Error: {e}", exc_info=True)
            raise RuntimeError("Failed to load Parakeet model.")

    def transcribe(self, audio_buffer: bytes) -> dict:
        # Convert raw PCM to model-ready floats and request timestamps for alignment
        if not audio_buffer:
            return {}
        try:
            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            transcription_result = self.model.transcribe(
                [audio_np],
                batch_size=1,
                timestamps=True,
                verbose=False
            )
            
            if not transcription_result or not transcription_result[0] or not transcription_result[0].timestamp:
                logging.info("[Model] Parakeet transcription yielded no results or timestamps.")
                return {}

            word_stamps_raw = transcription_result[0].timestamp.get('word', [])
            
            word_stamps = [
                {'word': w['word'], 'start': w['start'], 'end': w['end']}
                for w in word_stamps_raw
            ]

            if not word_stamps:
                return {'segments': [], 'words': []}

            full_text = transcription_result[0].text
            segments = [{'text': full_text, 'start': word_stamps[0]['start'], 'end': word_stamps[-1]['end']}]

            logging.info(f"[Model] Parakeet processed {len(word_stamps)} words from chunk.")
            return {'segments': segments, 'words': word_stamps}
        except Exception as e:
            logging.error(f"[Model] Parakeet transcription failed for a chunk: {e}", exc_info=True)
            return {}

    def close(self) -> None:
        # Free large model resources promptly to reduce memory pressure
        logging.info("[Model] Releasing Parakeet model from memory.")
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        super().close()