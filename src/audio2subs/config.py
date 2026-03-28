"""Configuration dataclasses for Audio2Subs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


def _resolve_device(device: Literal["cuda", "cpu", "auto"]) -> str:
    """Resolve 'auto' to the actual available device."""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


@dataclass
class TranscriptionConfig:
    """Configuration for the ASR transcription backend."""

    model_name: str = "CohereLabs/cohere-transcribe-03-2026"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    language: str = "en"  # ISO-639-1 language code (e.g. 'en', 'fr', 'de')

    def get_device(self) -> str:
        """Resolve 'auto' to the actual device."""
        return _resolve_device(self.device)


@dataclass
class SubtitleConfig:
    """Configuration for subtitle generation and styling."""

    # ASS styling
    font_name: str = "Arial"
    font_size: int = 65
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"

    # File naming
    subtitle_suffix: str = ".ai.ass"


@dataclass
class ServiceConfig:
    """Main service configuration."""

    # MPV socket path (works on both Linux/macOS and Windows)
    socket_path: str = "/tmp/mpv-socket"

    # Service behavior
    auto_select_subtitle: bool = True  # Auto-select AI subs on first generation

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

    # Nested configs
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Create config from environment variables."""
        config = cls()

        if socket := os.environ.get("MPV_SOCKET"):
            config.socket_path = socket
        if os.environ.get("AUDIO2SUBS_CPU_ONLY", "").lower() in ("1", "true"):
            config.transcription.device = "cpu"

        return config

    def __repr__(self) -> str:
        return (
            f"ServiceConfig(socket={self.socket_path!r}, "
            f"device={self.transcription.device!r})"
        )
