"""Smoke test: load local SmolLM2-135M-Instruct fixture and run a short generation."""

from pathlib import Path

import pytest

MODEL_PATH = Path(__file__).parents[1] / "test" / "fixtures" / "mlx-community--SmolLM2-135M-Instruct"


@pytest.mark.integration
def test_smollm2_generates_text():
    """Load the local model fixture and verify a non-empty response is generated."""
    try:
        from mlx_lm import generate, load
    except ImportError:
        pytest.skip("mlx-lm not installed")

    if not MODEL_PATH.exists():
        pytest.skip(f"Model fixture not found: {MODEL_PATH}")

    model, tokenizer = load(str(MODEL_PATH))

    prompt = "What is a BitTorrent tracker?"
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        formatted = prompt

    response = generate(model, tokenizer, prompt=formatted, max_tokens=80, verbose=False)
    assert isinstance(response, str)
    assert response.strip(), "Model returned empty response"
