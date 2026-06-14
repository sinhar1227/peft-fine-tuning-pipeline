# ============================================================
# 9. Create Hugging Face Dataset
# ============================================================

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def create_text_dataset(paragraph_records: list) -> Dataset:
    # Create a Hugging Face Dataset from the paragraph records list.
    if len(paragraph_records) < 2:
        raise ValueError(
            "The extracted corpus is too small. Please provide a larger pharma PDF or lower min_chars_per_paragraph."
        )
    return Dataset.from_list(paragraph_records)


# ============================================================
# 10. Train/eval split
# ============================================================
# Even for small demos, keep an evaluation set.
# This gives us validation loss and perplexity.

def train_test_split(dataset: Dataset, test_size: float = 0.15, seed: int = 42) -> DatasetDict:
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })


# ============================================================
# 11. Load tokenizer
# ============================================================

def load_tokenizer(model_name: str, use_fast: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    # Some Llama-style models do not define a pad token.
    # For causal LM fine-tuning, using EOS as PAD is a common practical choice.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# ============================================================
# 12. Tokenization and text packing
# ============================================================

def tokenize_corpus(examples, tokenizer):
    # Tokenize text without padding. Padding is handled dynamically by the collator.
    return tokenizer(examples["text"])


def create_training_blocks(tokenized_examples, block_size: int = 512):
    # Join all token IDs from multiple examples into one long list.
    all_input_ids = []
    all_attention_masks = []

    for input_ids in tokenized_examples["input_ids"]:
        all_input_ids.extend(input_ids)

    for attention_mask in tokenized_examples["attention_mask"]:
        all_attention_masks.extend(attention_mask)

    # Calculate how many complete blocks we can create.
    total_tokens = len(all_input_ids)
    usable_tokens = (total_tokens // block_size) * block_size

    # If we do not have enough tokens to create even one block, return empty data.
    if usable_tokens == 0:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

    # Keep only tokens that can fit into complete fixed-size blocks.
    all_input_ids = all_input_ids[:usable_tokens]
    all_attention_masks = all_attention_masks[:usable_tokens]

    # Split the long token list into fixed-size training blocks.
    input_id_blocks = []
    attention_mask_blocks = []

    for start_index in range(0, usable_tokens, block_size):
        end_index = start_index + block_size
        input_id_blocks.append(all_input_ids[start_index:end_index])
        attention_mask_blocks.append(all_attention_masks[start_index:end_index])

    # For causal language modeling, labels are the same as input IDs.
    # The model uses these labels to learn next-token prediction.
    labels = input_id_blocks.copy()

    return {
        "input_ids": input_id_blocks,
        "attention_mask": attention_mask_blocks,
        "labels": labels,
    }


# ============================================================
# Format instruction records
# ============================================================
# We convert every record into Alpaca-style training text.

def format_instruction_record(record):
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output_text = str(record.get("output", "")).strip()

    if input_text:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output_text}"
        )

    return {"text": text}


# ============================================================
# Tokenize instruction dataset
# ============================================================
# The tokenizer converts text into token IDs for model training.

def tokenize_instruction(examples, tokenizer, max_length: int = 512):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # For causal LM, labels are copied from input_ids.
    tokens["labels"] = tokens["input_ids"].copy()

    # Ignore padding tokens in the loss calculation.
    tokens["labels"] = [
        [token if mask == 1 else -100 for token, mask in zip(ids, attn)]
        for ids, attn in zip(tokens["input_ids"], tokens["attention_mask"])
    ]

    return tokens


# ============================================================
# Instruction-style inference prompt builder
# ============================================================

def build_instruction_prompt(instruction: str, input_text: str = "") -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()

    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )

    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )


# ============================================================
# Preference-tuned inference prompt builder
# ============================================================

def build_preference_prompt(instruction: str, input_text: str = "") -> str:
    return build_instruction_prompt(instruction, input_text)
