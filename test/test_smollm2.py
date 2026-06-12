"""
Download mlx-community/SmolLM2-135M-Instruct weights from HuggingFace
and run a quick inference test using mlx-lm.
"""

from mlx_lm import load, generate

MODEL_ID = "test/fixtures/mlx-community--SmolLM2-135M-Instruct"

print(f"Loading model: {MODEL_ID}")
model, tokenizer = load(MODEL_ID)  # type: ignore[misc]
print("Model loaded successfully.\n")

prompt = "What is a BitTorrent tracker?"

# Apply chat template if the tokenizer supports it
if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
else:
    formatted = prompt

print(f"Prompt: {prompt}\n")
print("Response:")
response = generate(model, tokenizer, prompt=formatted, max_tokens=200, verbose=True)
