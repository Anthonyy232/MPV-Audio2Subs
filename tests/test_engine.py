import pytest
from unittest.mock import MagicMock
from audio2subs.engine import TranscriptionEngine
from audio2subs.config import ServiceConfig, TranscriptionConfig, SubtitleConfig

def test_engine_clear_queue():
    eng = TranscriptionEngine(
        video_path="dummy.mp4",
        duration=10.0,
        transcriber=MagicMock(),
        config=ServiceConfig(transcription=TranscriptionConfig(), subtitle=SubtitleConfig())
    )
    
    eng._task_queue.put((0, (0.0, 1.0, 0)))
    eng._task_queue.put((1, (1.0, 2.0, 1)))
    
    eng._task_queue.put((TranscriptionEngine.IDLE_PRIORITY_OFFSET, (2.0, 3.0, 2)))
    eng._task_queue.put((TranscriptionEngine.IDLE_PRIORITY_OFFSET + 1, (3.0, 4.0, 3)))
    
    eng._clear_queue()
    
    assert eng._task_queue.qsize() == 2
    p1, t1 = eng._task_queue.get()
    assert p1 == TranscriptionEngine.IDLE_PRIORITY_OFFSET
    assert t1[0] == 2.0
