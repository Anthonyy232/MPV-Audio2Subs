"""Transcription backend interfaces and implementations."""

from audio2subs.transcription.base import (
    TranscriptionResult,
    WordTimestamp,
    BaseTranscriber,
)
from audio2subs.transcription.qwen import QwenTranscriber

__all__ = [
    "TranscriptionResult",
    "WordTimestamp",
    "BaseTranscriber",
    "QwenTranscriber",
]
