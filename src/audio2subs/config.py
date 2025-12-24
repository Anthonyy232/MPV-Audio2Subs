"""Configuration dataclasses for Audio2Subs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TranscriptionConfig:
    """Configuration for the ASR transcription backend."""
    
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    
    def get_device(self) -> str:
        """Resolve 'auto' to the actual device."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


@dataclass
class SubtitleConfig:
    """Configuration for subtitle generation and styling."""
    
    # Timing rules
    min_duration_s: float = 0.8
    max_duration_s: float = 7.0
    min_gap_s: float = 0.15
    padding_s: float = 0.150
    
    # Line formatting
    max_chars_per_line: int = 42
    
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
    
    # Chunking
    chunk_duration_seconds: int = 60
    
    # MPV socket path (works on both Linux/macOS and Windows)
    socket_path: str = "/tmp/mpv-socket"
    
    # Service behavior
    persistent_mode: bool = False  # Keep model in memory when toggled off
    auto_select_subtitle: bool = True  # Auto-select AI subs on first generation
    
    # Subtitle rewrite throttling
    rewrite_throttle_seconds: float = 3.0
    
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
        if chunk_dur := os.environ.get("AUDIO2SUBS_CHUNK_DURATION"):
            try:
                value = int(chunk_dur)
                if value > 0:
                    config.chunk_duration_seconds = value
            except ValueError:
                pass  # Ignore invalid values
        if os.environ.get("AUDIO2SUBS_PERSISTENT_MODE", "").lower() in ("1", "true"):
            config.persistent_mode = True
        if os.environ.get("AUDIO2SUBS_CPU_ONLY", "").lower() in ("1", "true"):
            config.transcription.device = "cpu"
            
        return config
    
    def __repr__(self) -> str:
        return (
            f"ServiceConfig(socket={self.socket_path!r}, "
            f"chunk_duration={self.chunk_duration_seconds}s, "
            f"device={self.transcription.device!r})"
        )
