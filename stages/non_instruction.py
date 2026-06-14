# ============================================================
# 5. Extract text from PDF → 6. Text cleaning → 7. Split paragraphs
# 8. Save corpus → 9. Create HF Dataset → 10. Train/eval split
# 11. Load tokenizer → 12. Tokenization + text packing
# 13. Load base model → 14. Apply LoRA → 15-17. Trainer
# 18. Start training → 19. Save adapter → 22. Inference helper
# ============================================================

import os
import json
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import Config
from data.pdf_processor import extract_pdf_pages, clean_pdf_pages, split_into_paragraph_records
from data.dataset_builder import (
    create_text_dataset, train_test_split, load_tokenizer,
    tokenize_corpus, create_training_blocks,
)
from models.model_loader import get_device_info, load_base_model, clear_gpu_memory
from models.lora_setup import create_lora_config, apply_lora
from models.training_callback import LogHistoryCallback


def run_non_instruction_stage(config: Config, pdf_path: str, status_callback=None):
    def log(msg):
        if status_callback:
            status_callback(msg)

    # Extract text from PDF
    log("Extracting text from PDF...")
    pdf_pages = extract_pdf_pages(pdf_path)
    log(f"Extracted {len(pdf_pages)} pages.")

    # Clean text
    log("Cleaning text...")
    cleaned_pages = clean_pdf_pages(pdf_pages)

    # Split into paragraphs
    log("Splitting into paragraphs...")
    paragraph_records = split_into_paragraph_records(cleaned_pages, config.min_chars_per_paragraph)
    log(f"Created {len(paragraph_records)} paragraph records.")

    # Save extracted and cleaned corpus for auditability
    # In real projects, always save intermediate datasets.
    # This helps with reproducibility, debugging, and compliance review.
    raw_pages_path = os.path.join(config.processed_data_dir, "pdf_pages_raw.jsonl")
    paragraphs_path = os.path.join(config.processed_data_dir, "pharma_paragraphs.jsonl")
    with open(raw_pages_path, "w", encoding="utf-8") as f:
        for item in pdf_pages:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(paragraphs_path, "w", encoding="utf-8") as f:
        for item in paragraph_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Create Hugging Face Dataset
    log("Creating Hugging Face dataset...")
    text_dataset = create_text_dataset(paragraph_records)

    # Train/eval split
    # Even for small demos, keep an evaluation set.
    # This gives us validation loss and perplexity.
    dataset = train_test_split(text_dataset, config.test_size, config.seed)
    log(f"Train: {len(dataset['train'])} | Validation: {len(dataset['validation'])}")

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model_name)

    # Tokenize corpus
    log("Tokenizing corpus...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_corpus(x, tokenizer),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing text corpus",
    )

    # Create training blocks (text packing)
    # Join all token IDs from multiple examples into one long list,
    # then split into fixed-size training blocks.
    log(f"Creating training blocks ({config.block_size} tokens)...")
    final_dataset = tokenized_datasets.map(
        lambda x: create_training_blocks(x, config.block_size),
        batched=True,
        desc=f"Creating fixed-size training blocks of {config.block_size} tokens",
    )
    log(f"Training blocks: {len(final_dataset['train'])} | Validation blocks: {len(final_dataset['validation'])}")

    # Check GPU availability
    use_cuda, gpu_name = get_device_info()
    log(f"GPU available: {use_cuda} ({gpu_name})")

    # Load base model (with 4-bit quantization if GPU available)
    log("Loading base model...")
    base_model = load_base_model(config.model_name, use_cuda, trainable=True)

    # Apply LoRA adapters
    # LoRA trains a small number of adapter parameters instead of updating all base model weights.
    # This is cheaper than full fine-tuning and is widely used in real projects.
    log("Applying LoRA...")
    lora_config = create_lora_config(config.lora_r, config.lora_alpha, config.lora_dropout)
    model = apply_lora(base_model, lora_config)
    model.print_trainable_parameters()

    # Data collator for causal language modeling
    # This prepares mini-batches for causal language model training.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    # These settings are designed for a small classroom/demo run.
    # For larger corpora, increase dataset size, epochs, and evaluation frequency carefully.
    training_args = TrainingArguments(
        # Log training loss at every step for small demo datasets.
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        logging_first_step=config.logging_first_step,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=use_cuda and config.fp16,
        bf16=config.bf16,
        report_to=config.report_to,
        remove_unused_columns=config.remove_unused_columns,
    )

    # Create log callback to capture real-time training logs
    log_callback = LogHistoryCallback(status_callback=status_callback)

    # Build Trainer with the custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["validation"],
        data_collator=data_collator,
        callbacks=[log_callback],
    )

    # Start training
    log("Starting training...")
    train_result = trainer.train()
    log("Training completed.")

    # Save adapter and tokenizer
    log("Saving adapter...")
    trainer.model.save_pretrained(config.adapter_dir)
    tokenizer.save_pretrained(config.adapter_dir)
    log(f"Adapter saved to {config.adapter_dir}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_result": train_result,
        "adapter_dir": config.adapter_dir,
        "log_history": log_callback.get_raw_logs(),
    }
