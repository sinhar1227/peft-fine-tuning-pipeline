# ============================================================
# Stage 2: Instruction Fine-Tuning
# ============================================================
# We continue instruction fine-tuning from the non-instruction LoRA adapter.
# The instruction dataset is loaded from a JSONL file, formatted into
# Alpaca-style training text, tokenized, and used for causal LM training.
#
# Merged Stage 1 Model
#    +
# New LoRA adapter for instruction tuning

import os
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import Config, InstructionConfig
from data.dataset_builder import (
    load_tokenizer, format_instruction_record, tokenize_instruction,
    build_instruction_prompt,
)
from models.model_loader import get_device_info, load_base_model, load_model_for_inference, clear_gpu_memory
from models.lora_setup import create_lora_config, apply_lora, load_lora_adapter, merge_and_unload
from models.inference import generate_instruction_response
from models.training_callback import LogHistoryCallback


def run_instruction_stage(
    config: Config,
    instruction_config: InstructionConfig,
    instruction_data_path: str,
    stage1_adapter_dir: str = None,
    merged_model_dir: str = None,
    status_callback=None,
):
    def log(msg):
        if status_callback:
            status_callback(msg)

    from datasets import load_dataset

    # Load instruction dataset from JSONL
    log("Loading instruction dataset...")
    dataset = load_dataset("json", data_files=instruction_data_path, split="train")
    log(f"Loaded {len(dataset)} examples.")

    # Format instruction records into Alpaca-style training text
    # We convert every record into Alpaca-style training text.
    log("Formatting instruction records...")
    dataset = dataset.map(format_instruction_record)

    # Create train-validation split
    log("Splitting train/validation...")
    split = dataset.train_test_split(test_size=0.15, seed=42)
    datasets = DatasetDict({"train": split["train"], "validation": split["test"]})
    log(f"Train: {len(datasets['train'])} | Validation: {len(datasets['validation'])}")

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model_name)

    # Tokenize instruction dataset
    # The tokenizer converts text into token IDs for model training.
    log("Tokenizing instruction dataset...")
    tokenized_datasets = datasets.map(
        lambda x: tokenize_instruction(x, tokenizer, instruction_config.max_length),
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing instruction dataset",
    )

    # Check GPU availability
    use_cuda, gpu_name = get_device_info()
    log(f"GPU: {use_cuda} ({gpu_name})")

    # Load base model - either from merged Stage 1 model or from original model name
    base_model_source = merged_model_dir if merged_model_dir else config.model_name
    log(f"Loading base model from {base_model_source}...")
    base_model = load_base_model(base_model_source, use_cuda, trainable=True)

    # Create a new LoRA adapter for instruction fine-tuning
    log("Creating LoRA adapter for instruction tuning...")
    lora_config = create_lora_config(r=16, lora_alpha=32, lora_dropout=0.05)
    model = apply_lora(base_model, lora_config)
    model.print_trainable_parameters()

    # Instruction fine-tuning data collator
    # This prepares mini-batches for causal language model training.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Instruction fine-tuning arguments
    training_args = TrainingArguments(
        output_dir=instruction_config.output_dir,
        # Training will stop after 5 optimizer steps.
        # max_steps overrides num_train_epochs.
        num_train_epochs=instruction_config.num_train_epochs,
        max_steps=instruction_config.max_steps,
        # Batch settings.
        per_device_train_batch_size=instruction_config.per_device_train_batch_size,
        per_device_eval_batch_size=instruction_config.per_device_eval_batch_size,
        gradient_accumulation_steps=instruction_config.gradient_accumulation_steps,
        # Optimizer settings.
        learning_rate=instruction_config.learning_rate,
        warmup_steps=instruction_config.warmup_steps,
        weight_decay=instruction_config.weight_decay,
        # Show training loss at every step.
        logging_steps=instruction_config.logging_steps,
        logging_first_step=True,
        # Run validation at every step.
        eval_strategy="steps",
        eval_steps=instruction_config.eval_steps,
        # Save checkpoint at final step.
        save_steps=instruction_config.save_steps,
        save_total_limit=instruction_config.save_total_limit,
        # Precision settings.
        fp16=use_cuda and instruction_config.fp16,
        bf16=instruction_config.bf16,
        # Disable external logging tools.
        report_to="none",
        # Keep required columns.
        remove_unused_columns=False,
    )

    # Create log callback to capture real-time training logs
    log_callback = LogHistoryCallback(status_callback=status_callback)

    # Build instruction Trainer with the custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[log_callback],
    )

    # Start instruction fine-tuning
    log("Starting instruction fine-tuning...")
    train_result = trainer.train()
    log("Instruction fine-tuning completed.")

    # Save final instruction-tuned LoRA adapter
    # This adapter now contains Stage 1 domain adaptation + Stage 2 instruction tuning.
    os.makedirs(instruction_config.adapter_dir, exist_ok=True)
    log("Saving instruction LoRA adapter...")
    trainer.model.save_pretrained(instruction_config.adapter_dir)
    tokenizer.save_pretrained(instruction_config.adapter_dir)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_result": train_result,
        "adapter_dir": instruction_config.adapter_dir,
        "log_history": log_callback.get_raw_logs(),
    }
