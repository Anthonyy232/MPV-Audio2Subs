"""Main service - orchestrates MPV integration and transcription engines."""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from audio2subs.config import ServiceConfig
from audio2subs.engine import TranscriptionEngine
from audio2subs.mpv_client import MPVClient
from audio2subs.transcription import ParakeetTranscriber

if TYPE_CHECKING:
    from audio2subs.transcription.base import BaseTranscriber

logger = logging.getLogger(__name__)


class SubtitleService:
    """Main service coordinating MPV events and transcription engines.
    
    Listens for MPV events (file loads, seeks, pause) and manages
    per-video transcription engines.
    """
    
    def __init__(self, config: ServiceConfig):
        """Initialize the subtitle service.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Components
        self._mpv: MPVClient | None = None
        self._transcriber: BaseTranscriber | None = None
        self._current_engine: TranscriptionEngine | None = None
        self._engines: dict[str, TranscriptionEngine] = {}
        
        # Model loading state
        self._model_loading = False
        self._model_loaded = threading.Event()
    
    def start(self) -> None:
        """Start the service and connect to MPV."""
        logger.info("Starting subtitle service")
        self._running = True
        
        # Connect to MPV
        self._mpv = MPVClient(self.config.socket_path)
        
        # Send "starting" status to Lua
        self._mpv.show_osd("AI Subtitle Service: Loading model...", 15000)
        self._mpv.send_message("ai-subs/starting")
        
        # Load transcription model (can be slow)
        try:
            self._load_model()
        except Exception:
            self._running = False
            return
        
        # Set up MPV event handlers
        self._setup_observers()
        
        # Send "ready" status
        self._mpv.show_osd("AI Subtitle Service: Ready", 3000)
        self._mpv.send_message("ai-subs/ready")
        
        # Check if a video is already playing
        if self._mpv.path:
            self._on_path_change(None, self._mpv.path)
        
        logger.info("Subtitle service started")
    
    def run(self) -> None:
        """Run the service until shutdown."""
        if not self._running:
            self.start()
        
        logger.info("Service running, awaiting events...")
        
        try:
            self._shutdown_event.wait()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Received exit signal")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the service and clean up resources."""
        if not self._running:
            return
        
        logger.info("Stopping subtitle service")
        self._running = False
        self._shutdown_event.set()
        
        # Stop all engines
        for engine in list(self._engines.values()):
            engine.stop()
        self._engines.clear()
        self._current_engine = None
        
        # Close transcriber
        if self._transcriber:
            self._transcriber.close()
            self._transcriber = None
        
        # Close MPV connection
        if self._mpv:
            self._mpv.send_message("ai-subs/stopped")
            self._mpv.close()
            self._mpv = None
        
        logger.info("Subtitle service stopped")
    
    def _load_model(self) -> None:
        """Load the transcription model."""
        logger.info("Loading transcription model...")
        self._model_loading = True
        
        try:
            self._transcriber = ParakeetTranscriber(self.config.transcription)
            self._transcriber.load()
            self._model_loaded.set()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.critical(f"Failed to load model: {e}", exc_info=True)
            if self._mpv:
                self._mpv.show_osd(f"AI Service Error: {e}", 10000)
                self._mpv.send_message("ai-subs/error", str(e))
            raise
        finally:
            self._model_loading = False
    
    def _setup_observers(self) -> None:
        """Set up MPV event observers."""
        if not self._mpv:
            return
        
        self._mpv.bind_event('shutdown', self._on_shutdown)
        self._mpv.bind_event('client-message', self._on_message)
        self._mpv.observe_property('path', self._on_path_change)
        self._mpv.observe_property('time-pos', self._on_time_pos)
        self._mpv.observe_property('pause', self._on_pause)
        self._mpv.observe_property('sid', self._on_sid_change)
    
    def _on_path_change(self, _: str | None, path: str | None) -> None:
        """Handle video file change."""
        # Clean up old engine
        if self._current_engine:
            video_path = self._current_engine.video_path
            logger.info(f"Cleaning up engine for: {os.path.basename(video_path)}")
            self._current_engine.stop()
            self._engines.pop(video_path, None)
        
        self._current_engine = None
        
        if self._mpv:
            self._mpv.reset_track_state()
        
        if not path:
            logger.info("Video closed")
            return
        
        logger.info(f"New video: {os.path.basename(path)}")
        
        # Create new engine
        if path not in self._engines:
            try:
                engine = self._create_engine(path)
                self._engines[path] = engine
                engine.start()
                
                if self._mpv:
                    self._mpv.send_message("ai-subs/started", os.path.basename(path))
                    
            except Exception as e:
                logger.error(f"Failed to create engine: {e}", exc_info=True)
                if self._mpv:
                    self._mpv.show_osd(f"AI Service Error: {e}", 5000)
                return
        
        self._current_engine = self._engines[path]
        
        # Try to add/update subtitle file
        self._update_subtitle_track()
    
    def _create_engine(self, video_path: str) -> TranscriptionEngine:
        """Create a transcription engine for a video."""
        if not self._mpv or not self._transcriber:
            raise RuntimeError("Service not properly initialized")
        
        duration = self._mpv.duration or 0
        video_params = self._mpv.video_params
        audio_id = self._mpv.aid
        
        # Convert MPV audio ID to FFmpeg stream index (MPV is 1-based)
        audio_track = (audio_id - 1) if audio_id and audio_id > 0 else None
        
        return TranscriptionEngine(
            video_path=video_path,
            duration=duration,
            transcriber=self._transcriber,
            config=self.config,
            video_width=video_params.get('w'),
            video_height=video_params.get('h'),
            audio_track_id=audio_track,
            on_subtitle_update=self._on_subtitle_update,
            on_progress=self._on_progress,
        )
    
    def _on_time_pos(self, _: str | None, time_pos: float | None) -> None:
        """Handle playback position change."""
        if not self._current_engine or time_pos is None:
            return
        if self._current_engine.is_finished:
            return
        if self._mpv and not self._mpv.pause:
            self._current_engine.process_time_update(time_pos)
    
    def _on_pause(self, _: str | None, is_paused: bool) -> None:
        """Handle pause/resume."""
        if not self._current_engine or self._current_engine.is_finished:
            return
        
        if not is_paused and self._mpv:
            time_pos = self._mpv.time_pos
            if time_pos is not None:
                self._current_engine.process_time_update(time_pos)
    
    def _on_sid_change(self, _: str | None, sid: int | str | None) -> None:
        """Handle subtitle track change - detect user deselection."""
        if not self._mpv or not self._current_engine:
            return
        
        # If user deselected to 'no' or switched to a different track
        # while we have an AI track, mark it as user-deselected
        ai_track_id = self._mpv.ai_track_id
        if ai_track_id is not None:
            if sid in (None, 'no', False) or (isinstance(sid, int) and sid != ai_track_id):
                self._mpv.mark_user_deselected()
    
    def _on_subtitle_update(self, subtitle_path: str) -> None:
        """Handle subtitle file update - reload in MPV."""
        if not self._mpv or not os.path.exists(subtitle_path):
            return
        
        self._mpv.add_subtitle(subtitle_path, auto_select=self.config.auto_select_subtitle)
    
    def _update_subtitle_track(self) -> None:
        """Update subtitle track in MPV."""
        if not self._current_engine or not self._mpv:
            return
        
        subtitle_path = self._current_engine.subtitle_path
        if os.path.exists(subtitle_path):
            self._mpv.add_subtitle(subtitle_path, auto_select=self.config.auto_select_subtitle)
    
    def _on_progress(self, completed: int, total: int) -> None:
        """Handle transcription progress update."""
        if not self._mpv:
            return
        
        percent = int(completed / total * 100) if total > 0 else 0
        
        # Send progress to Lua
        self._mpv.send_message("ai-subs/progress", str(percent), str(completed), str(total))
        
        # Update OSD periodically
        if completed == total:
            self._mpv.show_osd("AI Subtitles: Complete", 2000)
            self._mpv.send_message("ai-subs/complete")
    
    def _on_message(self, event: dict) -> None:
        """Handle IPC message from Lua."""
        args = event.get("args", [])
        if not args:
            return
        
        command = args[0]
        
        if command == "ai-subs/stop" or (command == "ai-service-event" and len(args) > 1 and '"stop"' in args[1]):
            logger.info("Received stop command via IPC")
            self._shutdown_event.set()
    
    def _on_shutdown(self, event: dict | None = None) -> None:
        """Handle MPV shutdown event."""
        logger.info("MPV shutdown event received")
        self._shutdown_event.set()
