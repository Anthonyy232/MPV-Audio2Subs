import os
import logging
import torch
import whisperx
import numpy as np
from flask import Flask, request, jsonify

# Configuration loaded from environment variables defined in the Dockerfile/docker run command.
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
ALIGN_MODEL_EN = os.getenv("ALIGN_MODEL_EN", "WAV2VEC2_ASR_LARGE_LV60K_960H")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Server] - %(message)s')

model = None
align_model_en = None
align_metadata_en = None
try:
    logging.info(f"[Model] Loading WhisperX model '{MODEL_SIZE}' onto device '{DEVICE}'...")
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    logging.info("[Model] Main transcription model loaded successfully.")

    logging.info("[Model] Pre-loading English alignment model...")
    align_model_en, align_metadata_en = whisperx.load_align_model(
        language_code="en",
        device=DEVICE,
        model_name=ALIGN_MODEL_EN,
    )
    logging.info("[Model] English alignment model loaded successfully.")

except Exception as e:
    logging.critical(f"[Model] FATAL: Could not load models. Server is non-functional. Error: {e}", exc_info=True)

app = Flask(__name__)

@app.route('/health')
def health_check():
    if model and align_model_en:
        return "OK", 200
    logging.warning("[Health] Health check failed: Models are not loaded.")
    return "Model not loaded", 503

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if not model or not align_model_en:
        return jsonify({"error": "Model is not loaded, server is in a failed state"}), 503

    if 'audio' not in request.files:
        logging.warning("[API] Transcription request failed: No audio file provided.")
        return jsonify({"error": "No audio file provided in the 'audio' field"}), 400

    file = request.files['audio']
    try:
        audio_bytes = file.read()
        # Convert raw PCM bytes (s16le) into a normalized float32 numpy array required by WhisperX.
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        logging.info(f"[API] Received {len(audio_bytes)} bytes of audio for transcription.")

        # Step 1: Run the initial Whisper transcription to get text and rough timestamps.
        result = model.transcribe(audio_np, batch_size=BATCH_SIZE, language="en")
        
        if result and result.get("segments"):
            logging.debug(f"[API] Initial transcription yielded {len(result['segments'])} segments. Starting alignment.")
            
            # Step 2: Run the alignment model (Wav2Vec2) to refine timestamps to word level.
            # This is crucial for accurate subtitling and refinement logic in the client.
            aligned_result = whisperx.align(result["segments"], align_model_en, align_metadata_en, audio_np, DEVICE, return_char_alignments=False)
            segments = aligned_result.get("segments", [])
            
            logging.info(f"[API] Alignment complete. Returning {len(segments)} final segments.")
        else:
            segments = []
            logging.info("[API] Initial transcription yielded no segments.")

        return jsonify(segments)

    except Exception as e:
        logging.error(f"[API] Error during transcription processing: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during transcription."}), 500