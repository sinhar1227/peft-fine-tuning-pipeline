# ============================================================
# 14. Apply LoRA adapters
# ============================================================
# LoRA trains a small number of adapter parameters instead of updating all base model weights.
# This is cheaper than full fine-tuning and is widely used in real projects.

from peft import LoraConfig, get_peft_model, PeftModel, TaskType


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
):
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )


def apply_lora(model, lora_config):
    # Apply the LoRA configuration on top of the base model.
    return get_peft_model(model, lora_config)


def load_lora_adapter(base_model, adapter_dir: str, trainable: bool = False):
    # Load a previously trained LoRA adapter onto a base model.
    return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=trainable)


# ============================================================
# 24. Optional merge step
# ============================================================
# This step merges the LoRA adapter into the base model.
# Use this only when you want a standalone model for deployment.

def merge_and_unload(model_with_adapter):
    # Merge LoRA adapter weights into the base model weights.
    return model_with_adapter.merge_and_unload()
