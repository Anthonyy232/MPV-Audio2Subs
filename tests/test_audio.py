import pytest
from unittest.mock import patch, MagicMock
from audio2subs.audio import AudioExtractor

def test_extractor_initialization():
    ext = AudioExtractor("dummy.mp4", duration=10.0)
    assert not ext.is_ready
    assert ext.bytes_available == 0

@patch("audio2subs.audio.subprocess.Popen")
@patch("audio2subs.audio.os.path.getsize")
@patch("audio2subs.audio.os.path.exists")
def test_extract_blocking(mock_exists, mock_getsize, mock_popen):
    mock_exists.return_value = True
    mock_getsize.return_value = 1000
    
    proc = MagicMock()
    proc.poll.return_value = 0
    proc.returncode = 0
    mock_popen.return_value = proc
    
    ext = AudioExtractor("dummy.mp4", duration=10.0)
    
    with patch("builtins.open", MagicMock()):
        ext.extract_blocking()
        
    assert ext.is_ready
    assert ext.bytes_available == 1000

def test_read_chunk_empty():
    ext = AudioExtractor("dummy.mp4", duration=10.0)
    ext._extraction_complete.set()
    assert ext.read_chunk(0.0, 1.0) is None
