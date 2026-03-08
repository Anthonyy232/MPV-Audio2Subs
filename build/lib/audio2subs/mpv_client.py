"""MPV IPC client wrapper with bidirectional communication."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable

from audio2subs.exceptions import MPVConnectionError, MPVCommandError

logger = logging.getLogger(__name__)

# Try to import the MPV library
try:
    from python_mpv_jsonipc import MPV, MPVError
    _HAS_MPV = True
except ImportError:
    MPV = None
    MPVError = Exception
    _HAS_MPV = False


class MPVClient:
    """Wrapper for MPV IPC communication with enhanced features.
    
    Features:
    - Bidirectional messaging (send status updates to Lua)
    - Smart subtitle track management
    - Clean observer lifecycle
    """
    
    CLIENT_NAME = "ai_subtitle_service"
    
    def __init__(self, socket_path: str):
        """Connect to MPV via IPC socket.
        
        Args:
            socket_path: Path to MPV's IPC socket
            
        Raises:
            MPVConnectionError: If connection fails
        """
        if not _HAS_MPV:
            raise MPVConnectionError(
                "python-mpv-jsonipc not installed",
                socket_path=socket_path
            )
        
        self.socket_path = socket_path
        self._mpv: MPV | None = None
        self._observer_ids: list[int] = []
        self._event_handlers: dict[str, Callable] = {}
        self._ai_track_id: int | None = None
        self._user_deselected: bool = False  # Smart selection tracking
        self._shutdown_event = threading.Event()
        
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to MPV."""
        try:
            self._mpv = MPV(start_mpv=False, ipc_socket=self.socket_path)
            logger.info(f"Connected to MPV: {self.socket_path}")
        except Exception as e:
            raise MPVConnectionError(
                f"Failed to connect to MPV: {e}",
                socket_path=self.socket_path
            ) from e
    
    @property
    def is_connected(self) -> bool:
        """Whether the MPV connection is alive."""
        if not self._mpv:
            return False
        try:
            _ = self._mpv.pause
            return True
        except (MPVError, BrokenPipeError, AttributeError):
            return False
    
    # --- Property Access ---
    
    def get_property(self, name: str) -> Any:
        """Get an MPV property value."""
        if not self._mpv:
            return None
        try:
            return getattr(self._mpv, name.replace("-", "_"))
        except (MPVError, BrokenPipeError, AttributeError) as e:
            logger.warning(f"Failed to get property '{name}': {e}")
            return None
    
    @property
    def path(self) -> str | None:
        return self.get_property("path")
    
    @property
    def duration(self) -> float | None:
        return self.get_property("duration")
    
    @property
    def time_pos(self) -> float | None:
        return self.get_property("time_pos")
    
    @property
    def pause(self) -> bool:
        return self.get_property("pause") or False
    
    @property
    def filename(self) -> str | None:
        return self.get_property("filename")
    
    @property
    def video_params(self) -> dict:
        return self.get_property("video_params") or {}
    
    @property
    def aid(self) -> int | None:
        """Current audio track ID."""
        return self.get_property("aid")
    
    @property
    def sid(self) -> int | str | None:
        """Current subtitle track ID."""
        return self.get_property("sid")
    
    @property
    def track_list(self) -> list[dict]:
        return self.get_property("track_list") or []
    
    # --- Event & Observer Management ---
    
    def bind_event(self, event: str, handler: Callable) -> None:
        """Bind an event handler."""
        if self._mpv:
            self._mpv.bind_event(event, handler)
            self._event_handlers[event] = handler
    
    def observe_property(self, name: str, handler: Callable) -> int | None:
        """Observe a property for changes."""
        if not self._mpv:
            return None
        try:
            obs_id = self._mpv.bind_property_observer(name, handler)
            self._observer_ids.append(obs_id)
            return obs_id
        except MPVError as e:
            logger.warning(f"Failed to observe '{name}': {e}")
            return None
    
    def unbind_all_observers(self) -> None:
        """Unbind all property observers."""
        if not self._mpv:
            return
        for obs_id in self._observer_ids:
            try:
                self._mpv.unbind_property_observer(obs_id)
            except (MPVError, BrokenPipeError):
                pass
        self._observer_ids.clear()
    
    # --- IPC Messaging (Python → Lua) ---
    
    def send_message(self, command: str, *args: str) -> None:
        """Send a script message to Lua.
        
        This enables bidirectional Python → Lua communication.
        
        Args:
            command: Message command (e.g., "ai-subs/ready")
            args: Additional arguments
        """
        if not self._mpv:
            return
        try:
            # Use show-text for OSD or script-message-to for Lua
            self._mpv.command("script-message", command, *args)
            logger.debug(f"Sent message: {command} {args}")
        except (MPVError, BrokenPipeError) as e:
            logger.warning(f"Failed to send message '{command}': {e}")
    
    def show_osd(self, text: str, duration_ms: int = 3000) -> None:
        """Show an OSD message."""
        if not self._mpv:
            return
        try:
            self._mpv.command("show-text", text, duration_ms)
        except (MPVError, BrokenPipeError) as e:
            logger.warning(f"Failed to show OSD: {e}")
    
    # --- Subtitle Track Management ---
    
    def add_subtitle(self, path: str, auto_select: bool = True) -> bool:
        """Add a subtitle file to MPV.
        
        Args:
            path: Path to subtitle file
            auto_select: Whether to auto-select if user hasn't manually deselected
            
        Returns:
            True if successfully added/updated
        """
        if not self._mpv or not os.path.exists(path):
            return False
        
        try:
            # If we already have a track ID, just reload
            if self._ai_track_id is not None:
                logger.debug(f"Reloading AI subtitle track {self._ai_track_id}")
                self._mpv.sub_reload(self._ai_track_id)
                return True
            
            # Check if track already exists
            existing = self._find_ai_track(path)
            if existing:
                self._ai_track_id = existing.get('id')
                logger.info(f"Found existing AI track: {self._ai_track_id}")
                self._mpv.sub_reload(self._ai_track_id)
                return True
            
            # Add new track
            # Respect user's deselection choice
            if self._user_deselected:
                flag = 'auto'  # Add but don't select
                logger.info("Adding AI track without selecting (user previously deselected)")
            elif auto_select and self._is_sub_active():
                flag = 'select'
            else:
                flag = 'auto'
            
            logger.info(f"Adding new AI subtitle with flag '{flag}'")
            self._mpv.sub_add(path, flag)
            
            # Wait briefly for MPV to process, then find the track ID
            time.sleep(0.1)
            new_track = self._find_ai_track(path)
            if new_track:
                self._ai_track_id = new_track.get('id')
                logger.info(f"AI track added with ID: {self._ai_track_id}")
            
            return True
            
        except (MPVError, BrokenPipeError) as e:
            logger.warning(f"Failed to add subtitle: {e}")
            self._ai_track_id = None
            return False
    
    def reload_subtitle(self) -> bool:
        """Reload the AI subtitle track."""
        if not self._mpv or self._ai_track_id is None:
            return False
        try:
            self._mpv.sub_reload(self._ai_track_id)
            return True
        except (MPVError, BrokenPipeError) as e:
            logger.warning(f"Failed to reload subtitle: {e}")
            return False
    
    def mark_user_deselected(self) -> None:
        """Mark that the user manually deselected the AI track.
        
        This prevents automatic re-selection on future updates.
        """
        self._user_deselected = True
        logger.info("User deselected AI subtitles - will not auto-select")
    
    def reset_track_state(self) -> None:
        """Reset track state for a new video."""
        self._ai_track_id = None
        self._user_deselected = False
    
    def _find_ai_track(self, path: str) -> dict | None:
        """Find the AI subtitle track in the track list."""
        try:
            target_path = os.path.normpath(path)
            for track in self.track_list:
                if track.get('type') != 'sub':
                    continue
                external = track.get('external-filename')
                if external:
                    ext_path = os.path.normpath(external)
                    try:
                        if os.path.samefile(target_path, ext_path):
                            return track
                    except (OSError, ValueError):
                        if target_path.lower() == ext_path.lower():
                            return track
        except (MPVError, BrokenPipeError) as e:
            logger.warning(f"Failed to query track list: {e}")
        return None
    
    @property
    def ai_track_id(self) -> int | None:
        """The ID of the AI subtitle track, if added."""
        return self._ai_track_id
    
    def _is_sub_active(self) -> bool:
        """Check if subtitles are currently active."""
        sid = self.sid
        visibility = self.get_property("sub_visibility")
        return sid not in (None, 'no', False) and visibility is True
    
    # --- Lifecycle ---
    
    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """Wait for shutdown signal."""
        return self._shutdown_event.wait(timeout=timeout)
    
    def signal_shutdown(self) -> None:
        """Signal that shutdown is requested."""
        self._shutdown_event.set()
    
    def close(self) -> None:
        """Clean up and close the connection."""
        self.unbind_all_observers()
        
        if self._mpv:
            try:
                self._mpv.terminate()
            except (MPVError, BrokenPipeError):
                pass
            self._mpv = None
        
        logger.info("MPV client closed")
    
    def __enter__(self) -> "MPVClient":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
