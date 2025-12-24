"""Transcription backend interfaces and implementations."""

from audio2subs.transcription.base import (
    TranscriptionResult,
    WordTimestamp,
    BaseTranscriber,
)
from audio2subs.transcription.parakeet import ParakeetTranscriber

__all__ = [
    "TranscriptionResult",
    "WordTimestamp",
    "BaseTranscriber",
    "ParakeetTranscriber",
]
