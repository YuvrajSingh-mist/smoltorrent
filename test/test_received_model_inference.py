"""
Load the gathered model from received_model/ and run a few generation passes.

received_model/ is populated by `python main.py --action gather`, which merges
distributed weights and downloads fresh tokenizer/config from HuggingFace Hub.

If tokenizer/config are missing (weights present but gather not fully run),
the fixture downloads them automatically so the test is self-contained.

Mark: integration — needs merged weights on disk.
"""

from pathlib import Path

import pytest

RECEIVED_MODEL_DIR = Path(__file__).parents[1] / "received_model"
MODEL_ID = "mlx-community/SmolLM2-135M-Instruct"

PROMPTS = [
    "What is a BitTorrent tracker?",
    "Explain distributed computing in one sentence.",
    "What is the capital of France?",
]


@pytest.fixture(scope="module", autouse=True)
def ensure_metadata():
    """Download tokenizer/config from HF Hub if not already present in received_model/."""
    if not (RECEIVED_MODEL_DIR / "config.json").exists():
        from huggingface_hub import snapshot_download

        RECEIVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(RECEIVED_MODEL_DIR),
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf", "*.ot"],
        )


@pytest.fixture(scope="module")
def loaded_model(ensure_metadata):
    from mlx_lm import load

    return load(str(RECEIVED_MODEL_DIR))


@pytest.mark.integration
def test_received_model_dir_is_complete(ensure_metadata):
    assert RECEIVED_MODEL_DIR.exists()
    assert list(RECEIVED_MODEL_DIR.glob("*.safetensors")), "No .safetensors weights in received_model/"
    assert (RECEIVED_MODEL_DIR / "config.json").exists(), "config.json missing"
    assert (RECEIVED_MODEL_DIR / "tokenizer.json").exists(), "tokenizer.json missing"


@pytest.mark.integration
def test_model_loads(loaded_model):
    model, tokenizer = loaded_model
    assert model is not None
    assert tokenizer is not None


@pytest.mark.integration
@pytest.mark.parametrize("prompt", PROMPTS)
def test_generate(loaded_model, prompt):
    from mlx_lm import generate

    model, tokenizer = loaded_model

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        formatted = prompt

    response = generate(model, tokenizer, prompt=formatted, max_tokens=64, verbose=True)

    print(f"\nPrompt: {prompt}\nResponse: {response}\n")
    assert isinstance(response, str), "generate() should return a string"
    assert len(response.strip()) > 0, f"Empty response for prompt: {prompt!r}"
