"""Generate Python code with a pretrained model and an optional LoRA checkpoint."""

import argparse
import os
from pathlib import Path


# Edit this text to describe the code you want. Multiple lines are supported.
PROMPT = """
Write a Python function that checks whether a number is prime.
""".strip()


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def load_generation_model(adapter=None, model_name=None):
    """Load a pretrained model, applying a LoRA adapter only when requested."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    base_model = model_name or DEFAULT_MODEL
    if adapter is not None:
        from peft import PeftConfig, PeftModel

        adapter_config = PeftConfig.from_pretrained(str(adapter))
        base_model = model_name or adapter_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, cache_dir=cache_dir, device_map="auto", dtype="auto"
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=512):
    """Generate one greedy completion using the model's chat template."""
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(
        model.get_input_embeddings().weight.device
    )
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(completion, skip_special_tokens=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt", nargs="?", default=PROMPT,
        help="Describe the Python code to generate (defaults to PROMPT in this file).",
    )
    parser.add_argument("--model", help=f"Pretrained model name/path (default: {DEFAULT_MODEL}, or adapter's base model).")
    parser.add_argument("--adapter", type=Path, help="Optional trained LoRA policy directory.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.adapter is not None and not (args.adapter / "adapter_model.safetensors").is_file():
        parser.error(f"Checkpoint weights not found in {args.adapter}")

    model, tokenizer = load_generation_model(adapter=args.adapter, model_name=args.model)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful Python assistant.\n"
            "        you follow the instruction exactly.",
        },
        {
            "role": "user",
            "content": "Create a python Program with following instructions. "
            "Only create the asked function with no other output.\n"
            f"                   # Instruction: {args.prompt}",
        },
    ]
    print(generate_response(model, tokenizer, messages, args.max_new_tokens))


if __name__ == "__main__":
    main()
