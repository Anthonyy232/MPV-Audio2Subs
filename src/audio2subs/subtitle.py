"""ASS subtitle file generation and management."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable

from audio2subs.config import SubtitleConfig
from audio2subs.transcription.base import WordTimestamp

logger = logging.getLogger(__name__)


@dataclass
class SubtitleSegment:
    """A single subtitle segment with timing and text."""
    
    start: float  # seconds
    end: float    # seconds
    text: str
    
    def __lt__(self, other: "SubtitleSegment") -> bool:
        """Enable sorting by start time."""
        return self.start < other.start
    
    def overlaps(self, other: "SubtitleSegment", tolerance: float = 0.1) -> bool:
        """Check if this segment overlaps with another within tolerance."""
        return not (self.end + tolerance < other.start or other.end + tolerance < self.start)


class SubtitleWriter:
    """Manages ASS subtitle file generation with atomic writes."""
    
    def __init__(
        self,
        output_path: str,
        config: SubtitleConfig,
        video_width: int | None = None,
        video_height: int | None = None,
        on_update: Callable[[str], None] | None = None
    ):
        """Initialize the subtitle writer.
        
        Args:
            output_path: Path to write the .ass file
            config: Subtitle styling configuration
            video_width: Video width for PlayResX (optional)
            video_height: Video height for PlayResY (optional)
            on_update: Callback when file is updated
        """
        self.output_path = output_path
        self.config = config
        self.video_width = video_width
        self.video_height = video_height
        self.on_update = on_update
        
        self._segments: list[SubtitleSegment] = []
        self._lock = threading.Lock()
    
    @property
    def segment_count(self) -> int:
        """Number of subtitle segments."""
        with self._lock:
            return len(self._segments)
    
    def clear(self) -> None:
        """Clear all segments for reprocessing."""
        with self._lock:
            self._segments.clear()
    
    def add_words(self, words: list[WordTimestamp], time_offset: float = 0.0) -> int:
        """Convert words to subtitle segments and add them.
        
        Args:
            words: List of word timestamps
            time_offset: Offset to add to all timestamps (for chunk alignment)
            
        Returns:
            Number of segments added
        """
        if not words:
            return 0
        
        segments = self._words_to_segments(words, time_offset)
        return self.add_segments(segments)
    
    def add_segments(self, segments: list[SubtitleSegment]) -> int:
        """Add new segments to the subtitle file.
        
        Args:
            segments: Segments to add
            
        Returns:
            Number of segments added
        """
        if not segments:
            return 0
        
        added = 0
        with self._lock:
            for seg in segments:
                # Check for duplicate/overlapping segments with same text
                is_duplicate = any(
                    existing.overlaps(seg) and existing.text == seg.text
                    for existing in self._segments
                )
                if not is_duplicate:
                    self._insert_sorted(seg)
                    added += 1
        
        return added
    
    def _insert_sorted(self, segment: SubtitleSegment) -> None:
        """Insert a segment maintaining sorted order by start time."""
        # Binary search for insertion point
        lo, hi = 0, len(self._segments)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._segments[mid].start < segment.start:
                lo = mid + 1
            else:
                hi = mid
        self._segments.insert(lo, segment)
    
    def _words_to_segments(
        self,
        words: list[WordTimestamp],
        time_offset: float
    ) -> list[SubtitleSegment]:
        """Convert word timestamps to subtitle segments with proper line breaking."""
        if not words:
            return []
        
        cfg = self.config
        initial_segments: list[SubtitleSegment] = []
        current_line_words: list[WordTimestamp] = []
        
        for i, word in enumerate(words):
            current_line_words.append(word)
            current_text = " ".join(w.word for w in current_line_words)
            
            is_last = (i == len(words) - 1)
            has_punct = any(p in word.word for p in ".?!")
            too_long = len(current_text) > cfg.max_chars_per_line
            
            if is_last or has_punct or too_long:
                if too_long and len(current_line_words) > 1:
                    # Line too long - break before last word
                    segment_words = current_line_words[:-1]
                    current_line_words = current_line_words[-1:]
                else:
                    segment_words = current_line_words
                    current_line_words = []
                
                if not segment_words:
                    continue
                
                text = " ".join(w.word for w in segment_words)
                start = segment_words[0].start + time_offset
                end = segment_words[-1].end + time_offset
                initial_segments.append(SubtitleSegment(start, end, text))
        
        if not initial_segments:
            return []
        
        # Apply timing rules
        processed: list[SubtitleSegment] = []
        for seg in initial_segments:
            # Add padding
            seg.start = max(0.0, seg.start - cfg.padding_s)
            seg.end += cfg.padding_s
            
            # Enforce min/max duration
            duration = seg.end - seg.start
            if duration < cfg.min_duration_s:
                seg.end = seg.start + cfg.min_duration_s
            elif duration > cfg.max_duration_s:
                seg.end = seg.start + cfg.max_duration_s
            
            # Enforce minimum gap from previous
            if processed:
                prev_end = processed[-1].end
                if seg.start < prev_end + cfg.min_gap_s:
                    seg.start = prev_end + cfg.min_gap_s
            
            # Skip if timing is invalid
            if seg.end <= seg.start:
                continue
            
            # Split long lines
            if len(seg.text) > cfg.max_chars_per_line:
                split_point = seg.text.rfind(' ', 0, cfg.max_chars_per_line)
                if split_point != -1:
                    seg.text = f"{seg.text[:split_point]}\\N{seg.text[split_point+1:]}"
            
            processed.append(seg)
        
        return processed
    
    def write(self) -> None:
        """Write the subtitle file atomically."""
        header = self._build_header()
        
        with self._lock:
            lines = [
                f"Dialogue: 0,{self._format_time(s.start)},{self._format_time(s.end)},Default,,0,0,0,,{s.text}"
                for s in self._segments
            ]
        
        content = header + "\n" + "\n".join(lines) + "\n"
        
        # Atomic write via temp file
        temp_path = self.output_path + ".tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(temp_path, self.output_path)
            
            if self.on_update:
                self.on_update(self.output_path)
                
        except Exception as e:
            logger.error(f"Failed to write subtitle file: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise
    
    def initialize(self) -> None:
        """Write an empty subtitle file header."""
        header = self._build_header()
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(header)
            logger.info(f"Initialized subtitle file: {os.path.basename(self.output_path)}")
        except IOError as e:
            logger.error(f"Failed to initialize subtitle file: {e}")
    
    def _build_header(self) -> str:
        """Build the ASS file header."""
        cfg = self.config
        
        lines = [
            "[Script Info]",
            "Title: AI Generated Subtitles",
            "ScriptType: v4.00+"
        ]
        
        if self.video_width and self.video_height:
            lines.extend([
                f"PlayResX: {self.video_width}",
                f"PlayResY: {self.video_height}"
            ])
        
        style = (
            f"Style: Default,{cfg.font_name},{cfg.font_size},"
            f"{cfg.primary_color},&H000000FF,{cfg.outline_color},"
            "&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1"
        )
        
        lines.extend([
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            style,
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as ASS timestamp (H:MM:SS.cc)."""
        centiseconds = int((seconds % 1) * 100)
        total = int(seconds)
        s = total % 60
        m = (total // 60) % 60
        h = total // 3600
        return f"{h}:{m:02d}:{s:02d}.{centiseconds:02d}"


def generate_subtitle_path(video_path: str, suffix: str = ".ai.ass") -> str:
    """Generate the subtitle file path for a video.
    
    Args:
        video_path: Path to the video file
        suffix: Subtitle file suffix
        
    Returns:
        Path to the subtitle file (same directory as video)
    """
    base = os.path.splitext(video_path)[0]
    return f"{base}{suffix}"
