"""Cohere Transcribe + stable-ts forced alignment backend."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from audio2subs.config import TranscriptionConfig
from audio2subs.exceptions import TranscriptionError
from audio2subs.transcription.base import (
    BaseTranscriber,
    TranscriptionResult,
    WordTimestamp,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class CohereTranscriber(BaseTranscriber):
    """Three-stage transcriber: Silero VAD → Cohere ASR → stable-ts alignment.

    Stage 1: Silero VAD zeros out non-speech regions to prevent hallucinations.
    Stage 2: CohereAsrForConditionalGeneration produces plain text from audio.
    Stage 3: stable-ts model.align() maps that text back to the audio for
             precise word-level timestamps.
    """

    def __init__(self, config: TranscriptionConfig):
        """Initialise the Cohere transcriber.

        Args:
            config: Transcription configuration (model_name, device, language).
        """
        super().__init__()
        self.config = config
        self._processor: Any = None
        self._model: Any = None         # CohereAsrForConditionalGeneration
        self._aligner: Any = None       # stable-ts Whisper model (tiny)
        self._vad: Any = None           # Silero VAD model

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load both the Cohere ASR model and the stable-ts aligner.

        Raises:
            TranscriptionError: If loading fails.
        """
        if self._is_loaded:
            return

        logger.info(f"Loading Cohere model: {self.config.model_name}")

        try:
            try:
                import torch
                from transformers import AutoProcessor, CohereAsrForConditionalGeneration
                import stable_whisper
            except ImportError as e:
                missing = str(e).split("'")[-2] if "'" in str(e) else "required dependencies"
                msg = (
                    f"Missing dependency: {missing}. "
                    "Please run: pip install transformers>=5.4.0 torch soundfile librosa "
                    "sentencepiece protobuf stable-ts"
                )
                logger.critical(msg)
                self._loaded_event.set()
                raise TranscriptionError(msg) from e

            device = self.config.get_device()

            # Resolve dtype
            dtype = torch.float32
            if device == "cuda":
                if torch.cuda.is_available():
                    caps = torch.cuda.get_device_capability()
                    dtype = torch.bfloat16 if caps[0] >= 8 else torch.float16
                    logger.debug(f"CUDA cap {caps}, using {dtype}")
                else:
                    logger.warning("CUDA requested but unavailable — falling back to CPU.")
                    device = "cpu"

            # --- Stage 1: Cohere ASR ---
            logger.info(f"Loading Cohere ASR on {device} ({dtype})")
            self._processor = AutoProcessor.from_pretrained(self.config.model_name)
            self._model = CohereAsrForConditionalGeneration.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype=dtype,
            )

            # --- Stage 2: stable-ts aligner (tiny Whisper, CPU is fine) ---
            logger.info("Loading stable-ts aligner (tiny) on cpu")
            self._aligner = stable_whisper.load_model("tiny", device="cpu")

            # --- Stage 3: Silero VAD (CPU, ~2MB, prevents hallucinations on silence) ---
            logger.info("Loading Silero VAD")
            from silero_vad import load_silero_vad
            self._vad = load_silero_vad()

            self._is_loaded = True
            self._loaded_event.set()
            logger.info("Cohere + stable-ts models loaded successfully")

        except Exception as e:
            self._is_loaded = False
            self._loaded_event.set()
            self._model = None
            self._processor = None
            self._aligner = None
            self._vad = None
            if not isinstance(e, TranscriptionError):
                raise TranscriptionError(f"Failed to load Cohere model: {e}") from e
            raise

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio_buffer: bytes) -> TranscriptionResult:
        """Transcribe raw PCM audio using Cohere ASR + stable-ts alignment.

        Args:
            audio_buffer: Raw PCM audio bytes (16 kHz, mono, 16-bit).

        Returns:
            TranscriptionResult with word-level timestamps.

        Raises:
            TranscriptionError: On failure.
        """
        if not self._is_loaded or self._model is None:
            raise TranscriptionError("Cohere model is not loaded")

        if not audio_buffer:
            return TranscriptionResult()

        try:
            audio_float = self._pcm_to_float(audio_buffer)

            # Stage 1: Cohere ASR → plain text
            text = self._run_cohere(audio_float)
            if not text:
                logger.debug("Cohere returned no text for chunk")
                return TranscriptionResult()

            # Stage 2: stable-ts forced alignment → word timestamps
            words = self._run_alignment(audio_float, text)

            if not words:
                logger.warning(
                    f"stable-ts alignment returned no words for chunk "
                    f"({len(audio_buffer)} bytes). Text was: {text[:80]!r}"
                )

            return TranscriptionResult(words=words, full_text=text.strip())

        except Exception as e:
            logger.error(f"Cohere transcription failed: {e}", exc_info=True)
            raise TranscriptionError(f"Transcription failed: {e}") from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pcm_to_float(self, audio_buffer: bytes) -> np.ndarray:
        """Convert raw 16-bit PCM bytes to float32 in [-1, 1]."""
        arr = np.frombuffer(audio_buffer, dtype=np.int16)
        return arr.astype(np.float32) / 32768.0

    def _apply_vad(self, audio_float: np.ndarray) -> np.ndarray:
        """Zero out non-speech regions using Silero VAD.

        Preserves original timestamps by zeroing silence rather than removing
        it, preventing Cohere from hallucinating on background noise.
        """
        from silero_vad import get_speech_timestamps

        speech_timestamps = get_speech_timestamps(
            audio_float,
            self._vad,
            sampling_rate=SAMPLE_RATE,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=100,
        )

        if not speech_timestamps:
            logger.debug("VAD: no speech detected")
            return np.zeros_like(audio_float)

        speech_s = sum(ts["end"] - ts["start"] for ts in speech_timestamps) / SAMPLE_RATE
        logger.debug(
            f"VAD: {len(speech_timestamps)} segments, "
            f"{speech_s:.1f}s / {len(audio_float) / SAMPLE_RATE:.1f}s speech"
        )

        mask = np.zeros(len(audio_float), dtype=np.float32)
        for ts in speech_timestamps:
            mask[ts["start"]:ts["end"]] = 1.0
        return audio_float * mask

    def _run_cohere(self, audio_float: np.ndarray) -> str:
        """Run Cohere ASR on a float32 waveform and return plain text.

        Follows the documented long-form pattern:
          https://huggingface.co/CohereLabs/cohere-transcribe-03-2026

        For audio longer than the feature extractor's max_audio_clip_s the
        processor splits the waveform into chunks automatically.  The
        audio_chunk_index tensor must be extracted *before* moving inputs
        to the model device (it may not be a plain tensor on all builds),
        kept in inputs so generate() can use it, and then passed to
        processor.decode() so the per-chunk transcriptions are reassembled.
        """
        import torch

        language = self.config.language

        # Suppress hallucinations on silence/noise before passing to Cohere
        if self._vad is not None:
            audio_float = self._apply_vad(audio_float)

        # Processor splits long audio automatically; audio_chunk_index encodes
        # chunk boundaries needed by decode() for reassembly.
        inputs = self._processor(
            audio=audio_float,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            language=language,
        )

        # Extract BEFORE .to() — it might not support the device/dtype cast.
        audio_chunk_index = inputs.get("audio_chunk_index")

        # Move all tensors to model device + dtype (matches the model's precision).
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)

        with torch.inference_mode():
            # audio_chunk_index stays inside inputs so generate() sees it.
            output_ids = self._model.generate(**inputs, max_new_tokens=256)

        if audio_chunk_index is not None:
            # Long-form path: decode() reassembles chunks and returns a list.
            # Take [0] to get the first (and only) audio item's full transcript.
            result = self._processor.decode(
                output_ids,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=language,
            )
            text = result[0] if isinstance(result, list) else result
        else:
            # Short-form path: single output, decode directly.
            text = self._processor.decode(output_ids, skip_special_tokens=True)

        return text.strip()

    def _run_alignment(self, audio_float: np.ndarray, text: str) -> list[WordTimestamp]:
        """Use stable-ts model.align() to get word-level timestamps.

        Args:
            audio_float: Float32 waveform at 16 kHz.
            text: Plain text to align.

        Returns:
            List of WordTimestamp (empty if alignment fails).
        """
        language = self.config.language

        try:
            result = self._aligner.align(
                audio_float,
                text,
                language=language,
                verbose=None,           # suppress progress output
                regroup=False,          # we handle grouping ourselves in SubtitleWriter
                remove_instant_words=True,  # drop zero-duration words
            )
        except Exception as e:
            logger.error(f"stable-ts align() failed: {e}", exc_info=True)
            return []

        if result is None:
            return []

        words: list[WordTimestamp] = []
        for segment in result.segments:
            if not segment.words:
                continue
            for w in segment.words:
                word_text = w.word.strip()
                if not word_text:
                    continue
                words.append(WordTimestamp(
                    word=word_text,
                    start=float(w.start),
                    end=float(w.end),
                ))

        return words

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release both models from memory."""
        super().close()

        try:
            import torch
            _has_torch = True
        except ImportError:
            _has_torch = False

        if self._model is not None:
            logger.info("Releasing Cohere ASR model from memory")
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        if self._aligner is not None:
            logger.info("Releasing stable-ts aligner from memory")
            del self._aligner
            self._aligner = None

        if self._vad is not None:
            del self._vad
            self._vad = None

        if _has_torch:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self) -> "CohereTranscriber":
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
