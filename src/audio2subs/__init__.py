"""MPV-Audio2Subs: Real-time AI subtitle generation for MPV."""

__version__ = "2.0.0"
__author__ = "Anthonyy232"

from audio2subs.config import ServiceConfig, SubtitleConfig, TranscriptionConfig
from audio2subs.exceptions import Audio2SubsError

__all__ = [
    "ServiceConfig",
    "SubtitleConfig", 
    "TranscriptionConfig",
    "Audio2SubsError",
    "__version__",
]
