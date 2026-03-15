"""Subtitle refinement using Qwen3 LLM."""

from __future__ import annotations

import logging
import time
from typing import Any

from audio2subs.config import RefinementConfig

logger = logging.getLogger(__name__)


class QwenRefiner:
    """Refines subtitle text using Qwen3-0.6B for better punctuation and formatting."""

    def __init__(self, config: RefinementConfig):
        """Initialize the Qwen refiner.

        Args:
            config: Refinement configuration
        """
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded."""
        return self._is_loaded

    def load(self) -> None:
        """Load the Qwen3 model and tokenizer.

        Raises:
            RuntimeError: If loading fails
        """
        if self._is_loaded:
            return

        logger.info(f"Loading Qwen refinement model: {self.config.model_name}")
        start_time = time.perf_counter()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.config.get_device()
            
            # Determine dtype
            dtype = "auto"
            if device == "cuda" and torch.cuda.is_available():
                caps = torch.cuda.get_device_capability()
                if caps[0] >= 8:
                    dtype = torch.bfloat16
                    logger.debug("Using bfloat16 for refinement model")
                else:
                    dtype = torch.float16
                    logger.debug("Using float16 for refinement model")

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                device_map=device
            )

            self._is_loaded = True
            elapsed = time.perf_counter() - start_time
            logger.info(f"Refinement model loaded in {elapsed:.2f}s on {device}")

        except Exception as e:
            logger.error(f"Failed to load refinement model: {e}", exc_info=True)
            self._is_loaded = False
            raise RuntimeError(f"Refinement model load failed: {e}") from e

    def refine_batch(self, texts: list[str]) -> list[str]:
        """Refine a batch of subtitle lines.

        Args:
            texts: List of raw subtitle strings

        Returns:
            List of refined subtitle strings
        """
        if not self._is_loaded or not self._model:
            return texts

        if not texts:
            return []

        # Join texts with a marker to process in larger chunks if needed,
        # but for subtitles, we might want to process them as a single block 
        # to maintain consistency.
        full_text = "\n".join(texts)
        
        refined_full = self.refine_text(full_text)

        # Split back - strip trailing whitespace first to avoid spurious empty lines
        # from a model that appends a trailing newline
        refined_lines = refined_full.strip().split("\n")
        
        # If the LLM changed the line count, we have a mapping problem.
        # For now, let's try to keep it simple: one prompt per meaningful block.
        if len(refined_lines) != len(texts):
            logger.warning(
                f"Refinement changed line count: {len(texts)} -> {len(refined_lines)}. "
                "Falling back to original line count mapping."
            )
            # Basic fallback: if it's close, we might try to align, but safest is 
            # to return what we can or the original.
            if len(refined_lines) > len(texts):
                return refined_lines[:len(texts)]
            else:
                return refined_lines + texts[len(refined_lines):]

        return refined_lines

    def refine_text(self, text: str) -> str:
        """Refine a block of text.

        Args:
            text: Raw text block

        Returns:
            Refined text block
        """
        if not self._is_loaded or not self._model or not self._tokenizer:
            return text

        try:
            import torch
            
            prompt = (
                "You are a subtitle post-processor. Clean up the following transcript by fixing "
                "punctuation, capitalization, and minor grammatical errors. "
                "DO NOT change the meaning or remove any words unless they are obvious filler (like 'uh', 'um'). "
                "Maintain the same number of lines. Each line corresponds to a timing segment.\n\n"
                f"Transcript:\n{text}"
            )

            messages = [
                {"role": "user", "content": prompt}
            ]

            # Use chat template based on config
            formatted_input = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking
            )

            inputs = self._tokenizer([formatted_input], return_tensors="pt").to(self._model.device)

            # Sampling parameters based on documentation
            if self.config.enable_thinking:
                gen_config = {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0,
                }
            else:
                gen_config = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0,
                }

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=4096,  # Should be enough for a batch of subs
                    **gen_config
                )

            # Extract output
            input_len = inputs.input_ids.shape[1]
            output_ids = generated_ids[0][input_len:].tolist()

            # Handle thinking content if enabled
            index = 0
            if self.config.enable_thinking:
                try:
                    # rindex finding 151668 (</think>)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

            # result_thinking = self._tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            result_content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

            return result_content

        except Exception as e:
            logger.error(f"Refinement generation failed: {e}", exc_info=True)
            return text

    def close(self) -> None:
        """Unload the model and free memory."""
        if not self._is_loaded:
            return

        logger.info("Unloading refinement model")
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
