# ============================================================
# Stage 3: Preference Fine-Tuning (DPO)
# ============================================================
# Direct Preference Optimization to align model responses
# with human preferences using (prompt, chosen, rejected) triples.
#
# Base Model
#    +
# Stage 1 LoRA Adapter (frozen, non-instruction)
#    +
# Stage 2 LoRA Adapter (frozen, instruction)
#    +
# Stage 3 LoRA Adapter (trainable, preference/DPO, saved)
#
# TRL provides DPOTrainer and DPOConfig for preference tuning.

import os
from datasets import DatasetDict
from config import Config, PreferenceConfig
from data.dataset_builder import load_tokenizer
from models.model_loader import get_device_info, load_base_model, clear_gpu_memory
from models.lora_setup import (
    create_lora_config, apply_lora, load_lora_adapter,
    load_first_adapter, load_additional_adapter,
    add_new_trainable_adapter, save_specific_adapter,
)
from models.inference import generate_preference_response
from models.training_callback import LogHistoryCallback


def run_preference_stage(
    config: Config,
    preference_config: PreferenceConfig,
    preference_data_path: str,
    stage1_adapter_dir: str = None,
    stage2_adapter_dir: str = None,
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

    # Load base model and stack Stage 1 + Stage 2 adapters (frozen), then add new trainable adapter for preference
    log(f"Loading base model from {config.model_name}...")
    base_model = load_base_model(config.model_name, use_cuda, trainable=True, status_callback=status_callback)

    if stage1_adapter_dir and os.path.exists(stage1_adapter_dir) and stage2_adapter_dir and os.path.exists(stage2_adapter_dir):
        log(f"Loading Stage 1 adapter from {stage1_adapter_dir} (as 'default')...")
        model = load_first_adapter(base_model, stage1_adapter_dir, adapter_name="default", trainable=False)
        log(f"Loading Stage 2 adapter from {stage2_adapter_dir} (as 'stage2')...")
        model = load_additional_adapter(model, stage2_adapter_dir, adapter_name="stage2")
        log("Adding new trainable adapter for preference tuning (as 'stage3')...")
        lora_config = create_lora_config(r=16, lora_alpha=32, lora_dropout=0.05)
        model = add_new_trainable_adapter(model, lora_config, adapter_name="stage3")
    elif stage2_adapter_dir and os.path.exists(stage2_adapter_dir):
        log(f"Loading Stage 2 adapter from {stage2_adapter_dir} and continuing training...")
        model = load_lora_adapter(base_model, stage2_adapter_dir, trainable=True)
    else:
        log("No prior adapters found, creating fresh LoRA for preference tuning...")
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
        # Disable gradient checkpointing to avoid dynamic shape conflicts
        # between prompt+chosen and prompt+rejected sequences in DPO
        gradient_checkpointing=False,
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

    # PEFT 0.19.1's add_adapter calls set_adapter internally, so DPOTrainer's
    # creation of the "ref" adapter at __init__ line 93 switched the active
    # adapter away from our trainable one. Restore it.
    trainable_adapter = None
    if stage1_adapter_dir and os.path.exists(stage1_adapter_dir) and stage2_adapter_dir and os.path.exists(stage2_adapter_dir):
        trainable_adapter = "stage3"
    else:
        # Fallback (stage2-only or fresh LoRA) — adapter defaults to "default"
        trainable_adapter = "default"
    if trainable_adapter in dpo_trainer.model.peft_config:
        dpo_trainer.model.set_adapter(trainable_adapter)
        dpo_trainer.model.train()

    # Start DPO preference tuning
    log("Starting DPO preference tuning...")
    train_result = dpo_trainer.train()
    log("DPO preference tuning completed.")

    # Save only the stage3 (preference) LoRA adapter
    os.makedirs(preference_config.adapter_dir, exist_ok=True)
    has_stacked = (
        stage1_adapter_dir and os.path.exists(stage1_adapter_dir) and
        stage2_adapter_dir and os.path.exists(stage2_adapter_dir)
    )
    if has_stacked:
        log("Saving preference LoRA adapter (stage3)...")
        save_specific_adapter(dpo_trainer.model, preference_config.adapter_dir, adapter_name="stage3")
    else:
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
