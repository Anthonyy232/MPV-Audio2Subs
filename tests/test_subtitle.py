import pytest
from audio2subs.config import SubtitleConfig
from audio2subs.subtitle import SubtitleSegment, SubtitleWriter
from audio2subs.transcription.base import WordTimestamp

def test_subtitle_segment_overlaps():
    s1 = SubtitleSegment(1.0, 2.0, "hi")
    s2 = SubtitleSegment(2.0, 3.0, "hi")
    assert s1.overlaps(s2, tolerance=0.1)
    
    s3 = SubtitleSegment(2.2, 3.0, "hi")
    assert not s1.overlaps(s3, tolerance=0.1)

def test_writer_add_segments(tmp_path):
    w = SubtitleWriter(str(tmp_path / "out.ass"), SubtitleConfig())
    
    w.add_segments([
        SubtitleSegment(5.0, 6.0, "five"),
        SubtitleSegment(1.0, 2.0, "one"),
        SubtitleSegment(3.0, 4.0, "three")
    ])
    
    assert w.segment_count == 3
    assert w._segments[0].text == "one"
    assert w._segments[2].text == "five"

def test_writer_duplicate_rejection(tmp_path):
    w = SubtitleWriter(str(tmp_path / "out.ass"), SubtitleConfig())
    added = w.add_segments([
        SubtitleSegment(1.0, 2.0, "hello"),
        SubtitleSegment(1.05, 2.05, "hello"),
        SubtitleSegment(1.0, 2.0, "world")
    ])
    assert added == 2
    assert w.segment_count == 2
