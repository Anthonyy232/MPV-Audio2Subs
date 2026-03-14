"""Qwen3 ASR transcription backend."""

from __future__ import annotations

import logging
from typing import Any

from audio2subs.config import TranscriptionConfig
from audio2subs.exceptions import TranscriptionError
from audio2subs.transcription.base import (
    BaseTranscriber,
    TranscriptionResult,
    WordTimestamp,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class QwenTranscriber(BaseTranscriber):
    """Qwen3 ASR transcriber for local, high-quality transcription."""

    def __init__(self, config: TranscriptionConfig):
        """Initialize the Qwen transcriber.

        Args:
            config: Transcription configuration (model name, device)
        """
        super().__init__()
        self.config = config
        self._model: Any = None

    def load(self) -> None:
        """Load the Qwen model onto the configured device.

        Raises:
            TranscriptionError: If loading fails
        """
        if self._is_loaded:
            return

        logger.info(f"Loading Qwen model: {self.config.model_name}")

        try:
            try:
                import torch
                from qwen_asr import Qwen3ASRModel
            except ImportError as e:
                missing_pkg = str(e).split("'")[-2] if "'" in str(e) else "required dependencies"
                error_msg = f"Missing dependency: {missing_pkg}. Please run 'pip install qwen-asr torch numpy'."
                logger.critical(error_msg)
                self._loaded_event.set()
                raise TranscriptionError(error_msg) from e

            device = self.config.get_device()
            
            # Determine appropriate dtype based on device and capability
            dtype = torch.float32
            if device == "cuda":
                if torch.cuda.is_available():
                    caps = torch.cuda.get_device_capability()
                    # Ampere (8.0), Ada (8.9), Blackwell (9.0) support bfloat16
                    if caps[0] >= 8:
                        dtype = torch.bfloat16
                        logger.debug(f"Device capability {caps} >= 8.0, using bfloat16")
                    else:
                        dtype = torch.float16
                        logger.debug(f"Device capability {caps} < 8.0, using float16")
                else:
                    logger.warning("CUDA device requested but torch.cuda.is_available() is False. Falling back to CPU/float32.")
                    device = "cpu"

            # Use flash_attn if available (requires bfloat16/float16)
            use_flash_attn = False
            if dtype in (torch.bfloat16, torch.float16):
                try:
                    import flash_attn  # noqa: F401
                    use_flash_attn = True
                    logger.info("Flash Attention 2 enabled")
                except ImportError:
                    pass

            attn_impl = "flash_attention_2" if use_flash_attn else None

            logger.info(f"Loading Qwen model on {device} with dtype {dtype}")

            fa_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}

            self._model = Qwen3ASRModel.from_pretrained(
                self.config.model_name,
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=32,
                max_new_tokens=512,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(
                    dtype=dtype,
                    device_map=device,
                    **fa_kwargs,
                ),
                **fa_kwargs,
            )

            self._is_loaded = True
            self._loaded_event.set()
            logger.info("Qwen model loaded successfully")

        except Exception as e:
            self._is_loaded = False
            self._loaded_event.set()
            self._model = None
            if not isinstance(e, TranscriptionError):
                raise TranscriptionError(f"Failed to load Qwen model: {e}") from e
            raise

    def transcribe(self, audio_buffer: bytes) -> TranscriptionResult:
        """Transcribe raw PCM audio using Qwen3 ASR.

        Args:
            audio_buffer: Raw PCM audio bytes (16kHz, mono, 16-bit)

        Returns:
            TranscriptionResult with word-level timestamps

        Raises:
            TranscriptionError: If transcription fails
        """
        if not self._is_loaded or self._model is None:
            raise TranscriptionError("Qwen model is not loaded")

        if not audio_buffer:
            return TranscriptionResult()

        try:
            import numpy as np
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
            
            # Convert to float32 normalized between -1.0 and 1.0 (typical model expectation)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Run transcription with English language
            results = self._model.transcribe(
                audio=(audio_float, SAMPLE_RATE),
                language="English",
                return_time_stamps=True,
            )

            if not results or not results[0]:
                logger.debug("Qwen returned no results for chunk")
                return TranscriptionResult()
                
            result = results[0]
            
            if not result.text:
                return TranscriptionResult()

            # Qwen returns timestamps as characters or words
            words = []
            if result.time_stamps:
                # time_stamps is a list of ForcedAlignItem objects
                for ts in result.time_stamps:
                    words.append(
                        WordTimestamp(
                            word=ts.text,
                            start=float(ts.start_time),
                            end=float(ts.end_time),
                        )
                    )

            # If forced aligner failed but we have text, we return empty words
            # The engine requires words for subtitle generation.
            if not words and result.text:
                logger.warning(f"No timestamps returned for chunk ({len(audio_buffer)} bytes). Alignment failed.")

            return TranscriptionResult(
                words=words,
                full_text=result.text.strip(),
            )

        except Exception as e:
            logger.error(f"Qwen transcription failed: {e}", exc_info=True)
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def transcribe_batch(self, audio_buffers: list[bytes]) -> list[TranscriptionResult]:
        """Transcribe multiple raw PCM buffers in a single model call.

        Passes all audio inputs to the model as a batch so the GPU processes
        them in parallel, reducing per-chunk overhead significantly.

        Args:
            audio_buffers: List of raw PCM bytes (16kHz, mono, 16-bit)

        Returns:
            List of TranscriptionResult, one per buffer, in the same order.
        """
        if not self._is_loaded or self._model is None:
            raise TranscriptionError("Qwen model is not loaded")
        if not audio_buffers:
            return []

        try:
            import numpy as np

            audio_inputs = []
            for buf in audio_buffers:
                if not buf:
                    audio_inputs.append((np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE))
                    continue
                arr = np.frombuffer(buf, dtype=np.int16)
                flt = arr.astype(np.float32) / 32768.0
                audio_inputs.append((flt, SAMPLE_RATE))

            raw_results = self._model.transcribe(
                audio=audio_inputs,
                language="English",
                return_time_stamps=True,
            )

            output: list[TranscriptionResult] = []
            for result in raw_results:
                if not result or not result.text:
                    output.append(TranscriptionResult())
                    continue

                words = []
                if result.time_stamps:
                    for ts in result.time_stamps:
                        words.append(WordTimestamp(
                            word=ts.text,
                            start=float(ts.start_time),
                            end=float(ts.end_time),
                        ))

                if not words and result.text:
                    logger.warning(f"No timestamps in batch result. Alignment failed.")

                output.append(TranscriptionResult(words=words, full_text=result.text.strip()))

            return output

        except Exception as e:
            logger.error(f"Batch transcription failed: {e}", exc_info=True)
            raise TranscriptionError(f"Batch transcription failed: {e}") from e

    def close(self) -> None:
        """Release the model from memory."""
        super().close()
        if self._model is not None:
            logger.info("Releasing Qwen model from memory")
            
            try:
                # Clean up memory
                del self._model
                self._model = None
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                    
            except ImportError:
                self._model = None

    def __enter__(self) -> "QwenTranscriber":
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
