import os
import pytest
from unittest.mock import MagicMock, patch
from audio2subs.config import RefinementConfig, ServiceConfig
from audio2subs.refinement import QwenRefiner


@pytest.fixture
def mock_transformers():
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load, \
         patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load:

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_input"
        mock_tokenizer.decode.return_value = "Refined subtitle line."
        mock_tokenizer.eos_token_id = 0

        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = [1, 10]
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer_load.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"

        mock_output = MagicMock()
        mock_output.__getitem__.return_value = mock_output
        mock_output.tolist.return_value = [0] * 10
        mock_model.generate.return_value = mock_output

        mock_model_load.return_value = mock_model

        yield mock_tokenizer, mock_model


def test_refiner_load(mock_transformers):
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.load()
    assert refiner.is_loaded
    assert refiner._model is not None
    assert refiner._tokenizer is not None


def test_refine_text(mock_transformers):
    tokenizer, model = mock_transformers
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.load()

    result = refiner.refine_text("hello world")

    assert "Refined" in result
    tokenizer.apply_chat_template.assert_called_once()
    model.generate.assert_called_once()


def test_refine_batch(mock_transformers):
    """Model returns numbered output; batch correctly parsed back to list."""
    tokenizer, model = mock_transformers
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.load()

    tokenizer.decode.return_value = "1. Line 1.\n2. Line 2."

    results = refiner.refine_batch(["line 1", "line 2"])

    assert len(results) == 2
    assert results[0] == "Line 1."
    assert results[1] == "Line 2."


def test_refine_batch_trailing_newline(mock_transformers):
    """Trailing newline in numbered output should parse cleanly."""
    tokenizer, model = mock_transformers
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.load()

    tokenizer.decode.return_value = "1. Line 1.\n2. Line 2.\n"
    results = refiner.refine_batch(["line 1", "line 2"])
    assert len(results) == 2
    assert results[0] == "Line 1."
    assert results[1] == "Line 2."


def test_refine_batch_partial_fallback(mock_transformers):
    """Lines the model omits from numbered output fall back to originals."""
    tokenizer, model = mock_transformers
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.load()

    # Model only returns line 1, skips line 2
    tokenizer.decode.return_value = "1. Hello world."

    results = refiner.refine_batch(["hello world", "how are you"])

    assert len(results) == 2
    assert results[0] == "Hello world."
    assert results[1] == "how are you"  # original preserved


def test_refine_batch_unparseable_fallback(mock_transformers):
    """Completely unparseable output falls back to all originals."""
    tokenizer, model = mock_transformers
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.load()

    tokenizer.decode.return_value = "I cannot process this."

    results = refiner.refine_batch(["line 1", "line 2"])

    assert results == ["line 1", "line 2"]


def test_refinement_env_vars():
    env = {
        "AUDIO2SUBS_REFINEMENT_ENABLED": "1",
        "AUDIO2SUBS_REFINEMENT_MODEL": "Qwen/Qwen3-1.7B",
        "AUDIO2SUBS_REFINEMENT_THINKING": "true",
    }
    with patch.dict(os.environ, env, clear=False):
        config = ServiceConfig.from_env()
    assert config.refinement.enabled is True
    assert config.refinement.model_name == "Qwen/Qwen3-1.7B"
    assert config.refinement.enable_thinking is True


def test_refinement_env_vars_defaults():
    with patch.dict(os.environ, {}, clear=False):
        config = ServiceConfig.from_env()
    assert config.refinement.enabled is True
    assert config.refinement.model_name == "Qwen/Qwen3-0.6B"
    assert config.refinement.enable_thinking is False


def test_refiner_close(mock_transformers):
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.load()
    assert refiner.is_loaded

    refiner.close()
    assert not refiner.is_loaded
    assert refiner._model is None
    assert refiner._tokenizer is None


def test_refiner_close_unloaded():
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.close()  # no-op, should not raise
    assert not refiner.is_loaded


def test_refine_batch_unloaded():
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    texts = ["line 1", "line 2"]
    assert refiner.refine_batch(texts) == texts


def test_refine_text_unloaded():
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    assert refiner.refine_text("hello") == "hello"
