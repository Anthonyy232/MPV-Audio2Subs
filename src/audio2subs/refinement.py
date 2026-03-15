"""Subtitle refinement using Qwen3 LLM."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from audio2subs.config import RefinementConfig

logger = logging.getLogger(__name__)

_NUMBERED_LINE_RE = re.compile(r"^\d+\.\s*(.*)")

_SYSTEM_PROMPT = (
    "You are a subtitle post-processor. "
    "You will receive numbered subtitle lines. "
    "Fix punctuation, capitalization, and obvious grammatical errors in each line. "
    "Remove obvious filler words ('uh', 'um', 'hmm') but preserve every other word exactly. "
    "Output ONLY the same numbered lines in the same order, nothing else."
)


class QwenRefiner:
    """Refines subtitle text using Qwen3-0.6B for better punctuation and formatting."""

    def __init__(self, config: RefinementConfig):
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

            dtype = "auto"
            if device == "cuda" and torch.cuda.is_available():
                caps = torch.cuda.get_device_capability()
                dtype = torch.bfloat16 if caps[0] >= 8 else torch.float16
                logger.debug(f"Using {dtype} for refinement model")

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                device_map=device,
            )
            self._model.eval()

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
            List of refined subtitle strings (same length as input)
        """
        if not self._is_loaded or not self._model:
            return texts

        if not texts:
            return []

        # Numbered format: "1. text\n2. text\n..." lets the model track lines
        # explicitly and makes parsing back unambiguous.
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        refined_raw = self._generate(numbered)
        return self._parse_numbered(refined_raw, texts)

    def _parse_numbered(self, raw: str, originals: list[str]) -> list[str]:
        """Parse numbered output back to a plain list.

        Falls back to original text for any line that can't be parsed.
        """
        result = list(originals)  # start with originals as fallback

        for line in raw.strip().splitlines():
            m = _NUMBERED_LINE_RE.match(line.strip())
            if not m:
                continue
            # Extract the 1-based index from the prefix
            dot = line.index(".")
            try:
                idx = int(line[:dot].strip()) - 1
            except ValueError:
                continue
            if 0 <= idx < len(result):
                result[idx] = m.group(1).strip()

        return result

    def _generate(self, user_text: str) -> str:
        """Run the model and return the raw generated string."""
        if not self._is_loaded or not self._model or not self._tokenizer:
            return user_text

        try:
            import torch

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]

            formatted_input = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,
            )

            inputs = self._tokenizer([formatted_input], return_tensors="pt").to(self._model.device)
            input_len = inputs.input_ids.shape[1]

            # Budget: output shouldn't exceed ~2× the input token count.
            # Clamp between 64 and 4096 to handle edge cases.
            max_new = max(64, min(4096, input_len * 2))

            if self.config.enable_thinking:
                gen_config: dict[str, Any] = {
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                }
            else:
                # Greedy for deterministic, fast text-cleaning
                gen_config = {"do_sample": False}

            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    pad_token_id=self._tokenizer.eos_token_id,
                    **gen_config,
                )

            output_ids = generated_ids[0][input_len:].tolist()

            # Strip thinking tokens when enabled
            index = 0
            if self.config.enable_thinking:
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
                except ValueError:
                    index = 0

            return self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

        except Exception as e:
            logger.error(f"Refinement generation failed: {e}", exc_info=True)
            return user_text

    # Legacy public API — kept for callers that use refine_text directly
    def refine_text(self, text: str) -> str:
        """Refine a raw text block (single LLM call, no numbered formatting)."""
        return self._generate(text)

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
