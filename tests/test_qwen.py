import sys
from unittest.mock import MagicMock

sys.modules['transformers'] = MagicMock()
sys.modules['qwen_asr'] = MagicMock()

torch_mock = MagicMock()
torch_mock.float32 = "float32"
torch_mock.float16 = "float16"
torch_mock.bfloat16 = "bfloat16"
sys.modules['torch'] = torch_mock

import pytest

from audio2subs.transcription.qwen import QwenTranscriber
from audio2subs.config import TranscriptionConfig

def test_qwen_transcription_empty():
    transcriber = QwenTranscriber(TranscriptionConfig())
    transcriber._model = MagicMock()
    transcriber._aligner = MagicMock()
    transcriber._is_loaded = True
    
    res = transcriber.transcribe(b"")
    assert res.full_text == ""
    assert len(res.words) == 0

def test_qwen_transcription_parsing():
    transcriber = QwenTranscriber(TranscriptionConfig())
    transcriber._model = MagicMock()
    transcriber._aligner = MagicMock()
    transcriber._is_loaded = True
    
    mock_result = MagicMock()
    mock_result.text = "hello world"
    
    item1 = MagicMock()
    item1.text = "hello"
    item1.start_time = 1.0
    item1.end_time = 2.0
    
    item2 = MagicMock()
    item2.text = "world"
    item2.start_time = 2.0
    item2.end_time = 3.0
    
    mock_result.time_stamps = [item1, item2]
    transcriber._model.transcribe.return_value = [mock_result]
    
    res = transcriber.transcribe(b"\x00\x00" * 16)
    assert res.full_text == "hello world"
    assert len(res.words) == 2
    assert res.words[1].end == 3.0
