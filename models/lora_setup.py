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
    return get_peft_model(model, lora_config)


def load_lora_adapter(base_model, adapter_dir: str, trainable: bool = False):
    return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=trainable)


# ============================================================
# Multi-Adapter Stacking (for compositional fine-tuning)
# ============================================================

def load_first_adapter(base_model, adapter_dir: str, adapter_name: str, trainable: bool = False):
    return PeftModel.from_pretrained(base_model, adapter_dir, adapter_name=adapter_name, is_trainable=trainable)


def load_additional_adapter(peft_model, adapter_dir: str, adapter_name: str):
    peft_model.load_adapter(adapter_dir, adapter_name=adapter_name)
    return peft_model


def add_new_trainable_adapter(peft_model, lora_config, adapter_name: str):
    peft_model.add_adapter(adapter_name, lora_config)
    peft_model.set_adapter(adapter_name)
    for name, param in peft_model.named_parameters():
        param.requires_grad = adapter_name in name
    return peft_model


def save_specific_adapter(peft_model, save_dir: str, adapter_name: str):
    peft_model.save_pretrained(save_dir, selected_adapters=[adapter_name])


def compose_adapters(peft_model, adapter_names: list):
    peft_model.set_adapter(adapter_names)
    peft_model.eval()


def merge_and_unload(model_with_adapter):
    return model_with_adapter.merge_and_unload()
