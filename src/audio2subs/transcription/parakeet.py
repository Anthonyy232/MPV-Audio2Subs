"""NVIDIA Parakeet ASR transcription backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from audio2subs.config import TranscriptionConfig
from audio2subs.exceptions import TranscriptionError
from audio2subs.transcription.base import (
    BaseTranscriber,
    TranscriptionResult,
    WordTimestamp,
)

if TYPE_CHECKING:
    import torch
    import nemo.collections.asr as nemo_asr

logger = logging.getLogger(__name__)


class ParakeetTranscriber(BaseTranscriber):
    """NeMo Parakeet ASR transcriber for local, high-quality transcription."""
    
    def __init__(self, config: TranscriptionConfig | None = None):
        super().__init__()
        self.config = config or TranscriptionConfig()
        self._model: "nemo_asr.models.ASRModel | None" = None
        self._device: str = ""
    
    def load(self) -> None:
        """Load the Parakeet model onto the configured device.
        
        This is separated from __init__ to support lazy/background loading.
        """
        if self._is_loaded:
            return
            
        logger.info(f"Loading Parakeet model: {self.config.model_name}")
        
        try:
            import torch
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise TranscriptionError(
                "NeMo toolkit not installed. Run: pip install nemo_toolkit[asr]"
            ) from e
        
        self._device = self.config.get_device()
        logger.info(f"Using device: {self._device}")
        
        try:
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.config.model_name
            )
            self._model.to(self._device)
            self._is_loaded = True
            logger.info("Parakeet model loaded successfully")
        except Exception as e:
            raise TranscriptionError(f"Failed to load Parakeet model: {e}") from e
    
    def transcribe(self, audio_buffer: bytes) -> TranscriptionResult:
        """Transcribe raw PCM audio using Parakeet.
        
        Args:
            audio_buffer: Raw PCM audio bytes (16kHz, mono, 16-bit)
            
        Returns:
            TranscriptionResult with word-level timestamps
        """
        if not self._is_loaded or self._model is None:
            self.load()
        
        if not audio_buffer:
            logger.debug("Empty audio buffer provided, returning empty result")
            return TranscriptionResult()
        
        try:
            # Convert raw PCM to normalized float32
            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            result = self._model.transcribe(
                [audio_np],
                batch_size=1,
                timestamps=True,
                verbose=False
            )
            
            if not result or not result[0] or not result[0].timestamp:
                logger.debug("Parakeet returned no results for chunk")
                return TranscriptionResult()
            
            word_stamps_raw = result[0].timestamp.get('word', [])
            words = [
                WordTimestamp(
                    word=w['word'],
                    start=w['start'],
                    end=w['end']
                )
                for w in word_stamps_raw
            ]
            
            if not words:
                logger.debug("No words extracted from transcription result")
                return TranscriptionResult()
            
            full_text = result[0].text
            logger.debug(f"Transcribed {len(words)} words")
            
            return TranscriptionResult(words=words, full_text=full_text)
            
        except Exception as e:
            logger.error(f"Parakeet transcription failed: {e}", exc_info=True)
            return TranscriptionResult()
    
    def close(self) -> None:
        """Release GPU memory held by the model."""
        super().close()
        
        if self._model is not None:
            logger.info("Releasing Parakeet model from memory")
            del self._model
            self._model = None
            
            if self._device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
    
    def __enter__(self) -> "ParakeetTranscriber":
        """Context manager entry - load model if not loaded."""
        if not self._is_loaded:
            self.load()
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit - close resources."""
        self.close()
