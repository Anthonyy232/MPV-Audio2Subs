"""Transcription backend interfaces and implementations."""

from audio2subs.transcription.base import (
    TranscriptionResult,
    WordTimestamp,
    BaseTranscriber,
)
from audio2subs.transcription.cohere import CohereTranscriber

__all__ = [
    "TranscriptionResult",
    "WordTimestamp",
    "BaseTranscriber",
    "CohereTranscriber",
]
