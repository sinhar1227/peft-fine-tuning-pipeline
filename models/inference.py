# ============================================================
# 22. Inference helper
# ============================================================
# Since this is non-instruction fine-tuning, prompts should look like text continuations,
# not chat-style questions.

import torch


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    device = model.device

    # Convert prompt text into token IDs.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text without calculating gradients because we are doing inference, not training.
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

    # Convert generated token IDs back into readable text.
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================
# Instruction-style inference helper
# ============================================================

def generate_instruction_response(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 150) -> str:
    from data.dataset_builder import build_instruction_prompt
    prompt = build_instruction_prompt(instruction, input_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================
# Preference-tuned inference helper
# ============================================================

def generate_preference_response(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 150) -> str:
    return generate_instruction_response(model, tokenizer, instruction, input_text, max_new_tokens)
