# ============================================================
# 13. Load base model
# ============================================================

import torch
import gc
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


def get_device_info():
    # Check for CUDA GPU availability.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return use_cuda, torch.cuda.get_device_name(0)
    return use_cuda, "CPU"


# Clear memory before loading the model.
def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_model(model_name: str, use_cuda: bool, trainable: bool = False):
    clear_gpu_memory()

    if use_cuda:
        # Configure 4-bit quantization to reduce GPU memory usage.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load the base model in 4-bit mode on available GPU devices.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare the quantized model for stable LoRA/QLoRA training.
        if trainable:
            model = prepare_model_for_kbit_training(model)

    else:
        # Load the base model normally when GPU is not available.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    # Disable cache during training to reduce memory usage and avoid training warnings.
    model.config.use_cache = False
    return model


def load_model_for_inference(model_name: str, use_cuda: bool):
    clear_gpu_memory()

    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    return model
