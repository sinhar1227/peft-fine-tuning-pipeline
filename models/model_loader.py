import torch
import gc
import os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


def get_device_info():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return use_cuda, torch.cuda.get_device_name(0)
    return use_cuda, "CPU"


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_model_cached(model_name: str, status_callback=None):
    """Pre-download model from Hugging Face Hub if not already cached locally."""
    from huggingface_hub import snapshot_download, scan_cache_dir
    from huggingface_hub.utils import HfHubHTTPError

    if os.path.isdir(model_name) or os.path.exists(model_name):
        return model_name

    if "/" not in model_name:
        return model_name

    try:
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_name:
                return model_name
    except Exception:
        pass

    try:
        if status_callback:
            status_callback(f"Downloading model {model_name} from Hugging Face Hub...")
        local_path = snapshot_download(repo_id=model_name)
        if status_callback:
            status_callback(f"Model downloaded to {local_path}")
        return local_path
    except HfHubHTTPError as e:
        raise RuntimeError(
            f"Cannot access model '{model_name}' on Hugging Face Hub. "
            f"Check your internet connection or HF_TOKEN. Error: {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Model '{model_name}' is not available locally and cannot be downloaded. "
            f"Error: {e}"
        )


def load_base_model(model_name: str, use_cuda: bool, trainable: bool = False, status_callback=None):
    clear_gpu_memory()

    resolved = ensure_model_cached(model_name, status_callback)

    if use_cuda:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if trainable:
            model = prepare_model_for_kbit_training(model)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    model.config.use_cache = False
    return model


def load_model_for_inference(model_name: str, use_cuda: bool):
    clear_gpu_memory()

    resolved = ensure_model_cached(model_name)

    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
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
            resolved,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    return model
