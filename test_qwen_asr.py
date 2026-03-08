import logging
import sys

try:
    from audio2subs.audio import AudioExtractor
    from audio2subs.config import ServiceConfig
    from audio2subs.transcription.qwen import QwenTranscriber
except ImportError as e:
    print(f"Failed to import modules: {e}")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_qwen():

    print("Initializing config...")
    config = ServiceConfig()
    
    print("Creating transcriber...")
    transcriber = QwenTranscriber(config.transcription)
    
    print("Loading model (this takes a moment)...")
    try:
        transcriber.load()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    video_path = r"C:\Users\Antho\Videos\2026-03-06 17-05-01.mp4"
    print(f"Extracting 10 seconds of audio from {video_path}...")
    
    try:
        extractor = AudioExtractor(video_path, duration=10.0)
        extractor.extract_blocking()
        audio_data = extractor.read_chunk(0.0, 10.0)
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        sys.exit(1)
    finally:
        extractor.close()

    if not audio_data:
        print("Failed to read audio chunk")
        sys.exit(1)

    print("Transcribing audio...")
    try:
        result = transcriber.transcribe(audio_data)
        print("\n=== Transcription Successful ===")
        print(f"Text:\n{result.full_text}\n")
        print("Words:")
        for w in result.words:
            print(f"[{w.start:.2f}s - {w.end:.2f}s] {w.word}")
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)

    print("Closing transcriber...")
    transcriber.close()
    print("Done.")

if __name__ == "__main__":
    test_qwen()
