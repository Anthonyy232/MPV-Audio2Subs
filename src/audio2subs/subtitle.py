"""ASS subtitle file output and management."""

from __future__ import annotations

import logging
import os
from typing import Callable

from audio2subs.config import SubtitleConfig

logger = logging.getLogger(__name__)


class SubtitleWriter:
    """Writes ASS subtitle files with atomic writes and PlayRes injection."""

    def __init__(
        self,
        output_path: str,
        config: SubtitleConfig,
        video_width: int | None = None,
        video_height: int | None = None,
        on_update: Callable[[str], None] | None = None,
    ):
        self.output_path = output_path
        self.config = config
        self.video_width = video_width
        self.video_height = video_height
        self.on_update = on_update

    def initialize(self) -> None:
        """Write an empty ASS file so MPV can load the subtitle track."""
        header = self._build_header()
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(header)
            logger.info(f"Initialized subtitle file: {os.path.basename(self.output_path)}")
        except IOError as e:
            logger.error(f"Failed to initialize subtitle file: {e}")

    def write_ass(self, ass_content: str) -> None:
        """Write complete ASS content to file atomically.

        Injects PlayResX/PlayResY if video dimensions are known.
        """
        if not ass_content:
            return

        if self.video_width and self.video_height:
            ass_content = self._inject_playres(ass_content)

        temp_path = self.output_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(ass_content)
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

    def _inject_playres(self, ass_content: str) -> str:
        """Inject PlayResX/PlayResY into the [Script Info] section."""
        playres = f"PlayResX: {self.video_width}\nPlayResY: {self.video_height}"
        # Insert after [Script Info] header line
        if "[Script Info]" in ass_content:
            return ass_content.replace(
                "[Script Info]",
                f"[Script Info]\n{playres}",
                1,
            )
        # Fallback: prepend
        return f"[Script Info]\n{playres}\n{ass_content}"

    def _build_header(self) -> str:
        """Build a minimal ASS header for initialize()."""
        cfg = self.config
        lines = [
            "[Script Info]",
            "Title: AI Generated Subtitles",
            "ScriptType: v4.00+",
        ]

        if self.video_width and self.video_height:
            lines.extend([
                f"PlayResX: {self.video_width}",
                f"PlayResY: {self.video_height}",
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
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ])

        return "\n".join(lines)


def generate_subtitle_path(video_path: str, suffix: str = ".ai.ass") -> str:
    """Generate the subtitle file path for a video."""
    base = os.path.splitext(video_path)[0]
    return f"{base}{suffix}"
