"""Cohere Transcribe + stable-ts pipeline backend."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from audio2subs.config import TranscriptionConfig, SubtitleConfig
from audio2subs.exceptions import TranscriptionError
from audio2subs.transcription.base import (
    BaseTranscriber,
    TranscriptionResult,
    SAMPLE_RATE,
)
from audio2subs.utils.performance import Timer

logger = logging.getLogger(__name__)


class CohereTranscriber(BaseTranscriber):
    """Cohere ASR with stable-ts post-processing pipeline.

    1. Silero VAD zeros out non-speech regions to prevent hallucinations.
    2. CohereAsrForConditionalGeneration produces plain text from audio.
    3. stable-ts align → refine → remove_repetition → adjust_gaps → regroup
       produces timed subtitle segments and generates ASS output.
    """

    def __init__(
        self,
        config: TranscriptionConfig,
        subtitle_config: SubtitleConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.subtitle_config = subtitle_config or SubtitleConfig()
        self._processor: Any = None
        self._model: Any = None         # CohereAsrForConditionalGeneration
        self._aligner: Any = None       # stable-ts Whisper model (base.en)
        self._vad: Any = None           # Silero VAD model

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
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
                    "sentencepiece protobuf stable-ts silero-vad"
                )
                logger.critical(msg)
                self._loaded_event.set()
                raise TranscriptionError(msg) from e

            device = self.config.get_device()

            dtype = torch.float32
            if device == "cuda":
                if torch.cuda.is_available():
                    caps = torch.cuda.get_device_capability()
                    dtype = torch.bfloat16 if caps[0] >= 8 else torch.float16
                    logger.debug(f"CUDA cap {caps}, using {dtype}")
                else:
                    logger.warning("CUDA requested but unavailable — falling back to CPU.")
                    device = "cpu"

            # --- Cohere ASR ---
            with Timer(f"Loading Cohere ASR ({self.config.model_name})", logger):
                self._processor = AutoProcessor.from_pretrained(self.config.model_name)
                self._model = CohereAsrForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    device_map="auto",
                    torch_dtype=dtype,
                )

            # --- stable-ts aligner (tiny Whisper, CPU) ---
            with Timer("Loading stable-ts aligner (base.en)", logger):
                self._aligner = stable_whisper.load_model("base.en", device=device)

            # --- Silero VAD (CPU, ~2MB) ---
            with Timer("Loading Silero VAD", logger):
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
        if not self._is_loaded or self._model is None:
            raise TranscriptionError("Cohere model is not loaded")

        if not audio_buffer:
            return TranscriptionResult()

        try:
            audio_duration = len(audio_buffer) / (SAMPLE_RATE * 2)  # 16-bit PCM Mono
            
            with Timer("Full Transcription Pipeline", logger, duration=audio_duration):
                audio_float = self._pcm_to_float(audio_buffer)

                # Cohere ASR → plain text (VAD applied internally)
                text = self._run_cohere(audio_float)
                if not text:
                    logger.debug("Cohere returned no text")
                    return TranscriptionResult()

                # stable-ts pipeline → ASS subtitle content
                ass_content = self._run_stable_ts(audio_float, text)

                return TranscriptionResult(full_text=text.strip(), ass_content=ass_content)

        except Exception as e:
            logger.error(f"Cohere transcription failed: {e}", exc_info=True)
            raise TranscriptionError(f"Transcription failed: {e}") from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pcm_to_float(self, audio_buffer: bytes) -> np.ndarray:
        with Timer("PCM -> Float conversion", logger):
            arr = np.frombuffer(audio_buffer, dtype=np.int16)
            return arr.astype(np.float32) / 32768.0

    def _apply_vad(self, audio_float: np.ndarray) -> np.ndarray:
        """Zero out non-speech regions using Silero VAD."""
        from silero_vad import get_speech_timestamps

        with Timer("Silero VAD", logger, duration=len(audio_float) / SAMPLE_RATE):
            speech_timestamps = get_speech_timestamps(
                audio_float,
                self._vad,
                sampling_rate=SAMPLE_RATE,
                threshold=0.35,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
                speech_pad_ms=100,
            )

        if not speech_timestamps:
            logger.debug("VAD: no speech detected")
            return np.zeros_like(audio_float)

        with Timer("VAD Masking", logger):
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
        """Run Cohere ASR on a float32 waveform and return plain text."""
        import torch

        language = self.config.language

        if self._vad is not None:
            audio_float = self._apply_vad(audio_float)

        duration = len(audio_float) / SAMPLE_RATE
        with Timer("Cohere ASR (Full)", logger, duration=duration):
            with Timer("ASR Preprocessing (Processor)", logger):
                inputs = self._processor(
                    audio=audio_float,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    language=language,
                )
                audio_chunk_index = inputs.get("audio_chunk_index")

            with Timer("ASR Data Move (Host -> Device)", logger):
                inputs = inputs.to(self._model.device, dtype=self._model.dtype)

            with Timer("ASR Generation (Inference)", logger):
                with torch.inference_mode():
                    output_ids = self._model.generate(**inputs, max_new_tokens=256)

            with Timer("ASR Decoding (IDs -> Text)", logger):
                if audio_chunk_index is not None:
                    result = self._processor.decode(
                        output_ids,
                        skip_special_tokens=True,
                        audio_chunk_index=audio_chunk_index,
                        language=language,
                    )
                    text = result[0] if isinstance(result, list) else result
                else:
                    text = self._processor.decode(output_ids, skip_special_tokens=True)

        return text.strip()

        return text.strip()

    def _run_stable_ts(self, audio_float: np.ndarray, text: str) -> str:
        """Run the full stable-ts pipeline and return ASS subtitle content.

        Pipeline: align → refine → remove_repetition → adjust_gaps → regroup → to_ass.
        Uses the original (non-VAD) audio for accurate alignment.
        """
        language = self.config.language

        duration = len(audio_float) / SAMPLE_RATE
        # --- Forced alignment ---
        try:
            with Timer("stable-ts align()", logger, duration=duration):
                result = self._aligner.align(
                    audio_float,
                    text,
                    language=language,
                    vad=True,
                    verbose=None,
                    regroup=False,
                    remove_instant_words=True,
                    aligner="new",
                )
        except Exception as e:
            logger.error(f"stable-ts align() failed: {e}", exc_info=True)
            return ""

        if result is None:
            return ""

        # --- Refine timestamps (iterative muting for precise boundaries) ---
        # Ensure all segments meet the minimum processing duration Whisper needs (approx 30ms)
        result.segments = [s for s in result.segments if (s.end - s.start) > 0.03]

        try:
            with Timer("stable-ts refine()", logger, duration=duration):
                self._aligner.refine(audio_float, result, only_voice_freq=True, verbose=None)
        except Exception as e:
            logger.warning(f"stable-ts refine() failed, using unrefined: {e}")

        # --- Post-processing ---
        try:
            with Timer("stable-ts post-processing", logger):
                result.remove_repetition(4, verbose=False)
                result.adjust_gaps()
                result.regroup()
        except Exception as e:
            logger.warning(f"stable-ts post-processing failed: {e}")

        # --- Generate ASS content ---
        cfg = self.subtitle_config
        with Timer("stable-ts ASS Formatting (to_ass)", logger):
            ass_content = result.to_ass(
                segment_level=True,
                word_level=False,
                font=cfg.font_name,
                font_size=cfg.font_size,
                PrimaryColour=cfg.primary_color,
                SecondaryColour="&H000000FF",
                OutlineColour=cfg.outline_color,
                BackColour="&H00000000",
                BorderStyle=1,
                Outline=2,
                Shadow=2,
                Alignment=2,
                MarginL=10,
                MarginR=10,
                MarginV=10,
                Encoding=1,
            )

        return ass_content or ""

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
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
