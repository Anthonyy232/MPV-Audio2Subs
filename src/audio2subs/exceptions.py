"""Custom exceptions for Audio2Subs."""

from __future__ import annotations


class Audio2SubsError(Exception):
    """Base exception for all Audio2Subs errors."""
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.args[0]!r})" if self.args else f"{self.__class__.__name__}()"


class AudioExtractionError(Audio2SubsError):
    """Raised when FFmpeg audio extraction fails."""

    def __init__(self, message: str, stderr: str | None = None):
        super().__init__(message)
        self.stderr = stderr

    def __repr__(self) -> str:
        if self.stderr:
            return f"AudioExtractionError({self.args[0]!r}, stderr={self.stderr[:100]!r}...)"
        return f"AudioExtractionError({self.args[0]!r})"


class TranscriptionError(Audio2SubsError):
    """Raised when ASR model inference fails."""
    pass


class MPVConnectionError(Audio2SubsError):
    """Raised when MPV IPC communication fails."""
    
    def __init__(self, message: str, socket_path: str | None = None):
        super().__init__(message)
        self.socket_path = socket_path
    
    def __repr__(self) -> str:
        return f"MPVConnectionError({self.args[0]!r}, socket_path={self.socket_path!r})"


class MPVCommandError(Audio2SubsError):
    """Raised when an MPV command fails."""
    pass
