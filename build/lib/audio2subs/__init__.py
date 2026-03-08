"""MPV-Audio2Subs: Real-time AI subtitle generation for MPV."""

__version__ = "2.0.0"
__author__ = "Anthonyy232"

from audio2subs.config import ServiceConfig, SubtitleConfig, TranscriptionConfig
from audio2subs.exceptions import Audio2SubsError

# Windows compatibility: Monkeypatch signal module for NeMo
import signal
import sys

if sys.platform == "win32":
    if not hasattr(signal, "SIGKILL"):
        # On Windows, SIGKILL is not available, but SIGTERM is often used as a fallback
        # for forced termination logic in cross-platform libraries.
        signal.SIGKILL = signal.SIGTERM
    if not hasattr(signal, "SIGALRM"):
        # SIGALRM is also POSIX-only, provide a dummy value if missing
        signal.SIGALRM = 14


__all__ = [
    "ServiceConfig",
    "SubtitleConfig", 
    "TranscriptionConfig",
    "Audio2SubsError",
    "__version__",
]
