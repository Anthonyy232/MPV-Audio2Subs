"""Base classes and protocols for transcription backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# Audio format constants (expected by all transcribers)
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit audio


@dataclass
class WordTimestamp:
    """A single word with its timing information."""
    
    word: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class TranscriptionResult:
    """Result from a transcription operation."""
    
    words: list[WordTimestamp] = field(default_factory=list)
    full_text: str = ""
    
    @property
    def is_empty(self) -> bool:
        return len(self.words) == 0


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for transcription backends."""
    
    def transcribe(self, audio_buffer: bytes) -> TranscriptionResult:
        """Transcribe raw PCM audio to timestamped words."""
        ...
    
    def close(self) -> None:
        """Release resources."""
        ...


class BaseTranscriber(ABC):
    """Abstract base class for transcription backends with common utilities."""
    
    def __init__(self):
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        return self._is_loaded
    
    @abstractmethod
    def transcribe(self, audio_buffer: bytes) -> TranscriptionResult:
        """Transcribe raw PCM audio to timestamped words.
        
        Args:
            audio_buffer: Raw PCM audio bytes (16kHz, mono, 16-bit)
            
        Returns:
            TranscriptionResult with word-level timestamps
        """
        pass
    
    def close(self) -> None:
        """Release resources held by the transcriber."""
        logger.info(f"Closing transcriber: {self.__class__.__name__}")
        self._is_loaded = False
