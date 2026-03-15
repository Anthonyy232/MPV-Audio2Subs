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
        
        # Mock tokenizer call
        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = [1, 10]
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        
        mock_tokenizer_load.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.device = "cpu"
        
        # Mock generate output. transformers models return a tensor.
        # We need mock_model.generate()[0][input_len:].tolist() to work.
        mock_output = MagicMock()
        mock_output.__getitem__.return_value = mock_output # generated_ids[0]
        mock_output.__getitem__.return_value = mock_output # [input_len:]
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
    tokenizer, model = mock_transformers
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.load()
    
    # Mock behavior to return one line per input line
    tokenizer.decode.return_value = "Line 1\nLine 2"
    
    results = refiner.refine_batch(["line 1", "line 2"])
    
    assert len(results) == 2
    assert results[0] == "Line 1"
    assert results[1] == "Line 2"

def test_refine_batch_mismatch_fallback(mock_transformers):
    tokenizer, model = mock_transformers
    config = RefinementConfig(enabled=True, device="cpu")
    refiner = QwenRefiner(config)
    refiner.load()
    
    # Mock LLM returning different number of lines
    tokenizer.decode.return_value = "Single line"
    
    results = refiner.refine_batch(["line 1", "line 2"])
    
    # Should fallback or at least not crash
    assert len(results) == 2
    assert results[0] == "Single line"
    assert results[1] == "line 2" # Original fallback for second line

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
    assert config.refinement.enabled is False
    assert config.refinement.model_name == "Qwen/Qwen3-0.6B"
    assert config.refinement.enable_thinking is False
