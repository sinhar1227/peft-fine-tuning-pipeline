# ============================================================
# 22. Inference helper
# ============================================================

import torch
import os


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    return _generate(model, tokenizer, prompt, max_new_tokens)


def generate_instruction_response(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 150) -> str:
    from data.dataset_builder import build_instruction_prompt
    prompt = build_instruction_prompt(instruction, input_text)
    return _generate(model, tokenizer, prompt, max_new_tokens)


def generate_preference_response(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 150) -> str:
    return generate_instruction_response(model, tokenizer, instruction, input_text, max_new_tokens)


# ============================================================
# Composed inference — loads base model + N stacked adapters
# ============================================================

def load_composed_model(
    model_name: str,
    adapter_dirs: list,
    adapter_names: list = None,
    use_cuda: bool = True,
):
    from models.model_loader import load_model_for_inference
    from models.lora_setup import load_first_adapter, load_additional_adapter, compose_adapters

    if adapter_names is None:
        adapter_names = [f"adapter_{i}" for i in range(len(adapter_dirs))]

    base_model = load_model_for_inference(model_name, use_cuda)
    model = load_first_adapter(base_model, adapter_dirs[0], adapter_names[0], trainable=False)
    for i in range(1, len(adapter_dirs)):
        model = load_additional_adapter(model, adapter_dirs[i], adapter_names[i])
    compose_adapters(model, adapter_names)
    return model
