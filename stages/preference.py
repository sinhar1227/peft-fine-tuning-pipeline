# ============================================================
# Stage 3: Preference Fine-Tuning (DPO)
# ============================================================
# Direct Preference Optimization to align model responses
# with human preferences using (prompt, chosen, rejected) triples.
#
# Merged Instruction Model (Stage 2)
#    +
# New LoRA adapter for preference tuning (DPO)
#
# TRL provides DPOTrainer and DPOConfig for preference tuning.

import os
from datasets import DatasetDict
from config import Config, PreferenceConfig
from data.dataset_builder import load_tokenizer
from models.model_loader import get_device_info, load_base_model, clear_gpu_memory
from models.lora_setup import create_lora_config, apply_lora
from models.inference import generate_preference_response
from models.training_callback import LogHistoryCallback


def run_preference_stage(
    config: Config,
    preference_config: PreferenceConfig,
    preference_data_path: str,
    merged_instruction_model_dir: str,
    status_callback=None,
):
    def log(msg):
        if status_callback:
            status_callback(msg)

    from datasets import load_dataset

    # Load DPO preference dataset
    # Expected columns: prompt, chosen, rejected
    log("Loading preference dataset...")
    dataset = load_dataset("json", data_files=preference_data_path, split="train")
    log(f"Loaded {len(dataset)} examples.")

    # Create train-validation split
    log("Splitting train/validation...")
    split = dataset.train_test_split(test_size=0.15, seed=42)
    # Rename test split to validation split
    datasets = DatasetDict({"train": split["train"], "validation": split["test"]})
    log(f"Train: {len(datasets['train'])} | Validation: {len(datasets['validation'])}")

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model_name)

    # Check GPU
    use_cuda, gpu_name = get_device_info()
    log(f"GPU: {use_cuda} ({gpu_name})")

    # Load merged instruction model as base for preference tuning
    log("Loading merged instruction model as base for DPO...")
    base_model = load_base_model(merged_instruction_model_dir, use_cuda, trainable=True)

    # Create a new LoRA adapter for preference tuning
    log("Creating LoRA adapter for preference tuning...")
    lora_config = create_lora_config(r=16, lora_alpha=32, lora_dropout=0.05)
    model = apply_lora(base_model, lora_config)
    model.print_trainable_parameters()

    # Configure DPO training
    # beta controls how strongly the model is pushed toward chosen answers over rejected answers.
    log("Setting up DPO training...")
    from trl import DPOConfig, DPOTrainer

    dpo_training_args = DPOConfig(
        # Training duration
        output_dir=preference_config.output_dir,
        num_train_epochs=preference_config.num_train_epochs,
        max_steps=preference_config.max_steps,
        # Batch settings
        per_device_train_batch_size=preference_config.per_device_train_batch_size,
        per_device_eval_batch_size=preference_config.per_device_eval_batch_size,
        gradient_accumulation_steps=preference_config.gradient_accumulation_steps,
        # Optimizer settings
        learning_rate=preference_config.learning_rate,
        warmup_steps=preference_config.warmup_steps,
        weight_decay=preference_config.weight_decay,
        # Logging and evaluation
        logging_steps=preference_config.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=preference_config.eval_steps,
        # Checkpoint saving
        save_steps=preference_config.save_steps,
        save_total_limit=preference_config.save_total_limit,
        # Precision settings
        fp16=use_cuda and preference_config.fp16,
        bf16=preference_config.bf16,
        # Disable external logging tools
        report_to="none",
        # Keep required columns
        remove_unused_columns=False,
        # DPO hyperparameter
        beta=preference_config.beta,
        max_length=preference_config.max_length,
        max_prompt_length=preference_config.max_prompt_length,
    )

    # Create log callback to capture real-time DPO training logs
    # DPOTrainer also supports callbacks
    log_callback = LogHistoryCallback(status_callback=status_callback)

    # Build DPOTrainer
    # None means TRL will internally use the reference behavior
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        callbacks=[log_callback],
    )

    # Start DPO preference tuning
    log("Starting DPO preference tuning...")
    train_result = dpo_trainer.train()
    log("DPO preference tuning completed.")

    # Save DPO preference-tuned LoRA adapter
    os.makedirs(preference_config.adapter_dir, exist_ok=True)
    log("Saving DPO LoRA adapter...")
    dpo_trainer.model.save_pretrained(preference_config.adapter_dir)
    tokenizer.save_pretrained(preference_config.adapter_dir)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_result": train_result,
        "adapter_dir": preference_config.adapter_dir,
        "log_history": log_callback.get_raw_logs(),
    }
