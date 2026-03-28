"""Transcription backend interfaces and implementations."""

from audio2subs.transcription.base import (
    TranscriptionResult,
    BaseTranscriber,
)
from audio2subs.transcription.cohere import CohereTranscriber

__all__ = [
    "TranscriptionResult",
    "BaseTranscriber",
    "CohereTranscriber",
]
