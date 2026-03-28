"""Base classes for transcription backends."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Audio format constants (expected by all transcribers)
SAMPLE_RATE = 16000
CHANNELS = 1


@dataclass
class TranscriptionResult:
    """Result from a transcription operation."""

    full_text: str = ""
    ass_content: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.full_text


class BaseTranscriber(ABC):
    """Abstract base class for transcription backends."""

    def __init__(self):
        self._is_loaded = False
        self._loaded_event = threading.Event()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def wait_for_load(self, timeout: float | None = None) -> bool:
        """Wait for the transcriber to be loaded. Returns True if loaded, False if timeout."""
        return self._loaded_event.wait(timeout)

    @abstractmethod
    def transcribe(self, audio_buffer: bytes) -> TranscriptionResult:
        """Transcribe raw PCM audio.

        Args:
            audio_buffer: Raw PCM audio bytes (16kHz, mono, 16-bit)

        Returns:
            TranscriptionResult with full_text and ass_content
        """
        pass

    def close(self) -> None:
        """Release resources held by the transcriber."""
        logger.info(f"Closing transcriber: {self.__class__.__name__}")
        self._is_loaded = False
