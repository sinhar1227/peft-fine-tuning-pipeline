# ============================================================
# 3. Global configuration
# ============================================================
# Keep all important parameters in one place.
# This makes the notebook easier to debug, reproduce, and productionize.

from dataclasses import dataclass, asdict
import os


@dataclass
class Config:
    # Base causal language model that we will fine-tune on pharma-domain text.
    model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    # Directory where training checkpoints will be saved during fine-tuning.
    output_dir: str = "./output"

    # Directory where the final trained LoRA adapter will be saved.
    adapter_dir: str = "./adapter"

    # Directory where cleaned and processed training data will be saved.
    processed_data_dir: str = "./processed_data"

    # Minimum paragraph length required to keep a paragraph for training.
    min_chars_per_paragraph: int = 80

    # Number of tokens in each training block for causal language modeling.
    block_size: int = 512

    # Percentage of data used for validation instead of training.
    test_size: float = 0.15

    # Random seed used to make splitting and training more reproducible.
    seed: int = 42

    # LoRA rank; controls the size and capacity of the trainable adapter.
    lora_r: int = 16

    # LoRA scaling factor; controls the strength of the LoRA update.
    lora_alpha: int = 32

    # Dropout applied inside LoRA layers to reduce overfitting.
    lora_dropout: float = 0.05

    # Target modules for LoRA: attention (q,k,v,o) + FFN (gate,up,down).
    lora_target_modules: list = None

    # Number of times the model will see the complete training dataset.
    num_train_epochs: float = 3.0

    # Number of training samples processed per GPU/device at one time.
    per_device_train_batch_size: int = 1

    # Number of validation samples processed per GPU/device at one time.
    per_device_eval_batch_size: int = 1

    # Number of small batches accumulated before one optimizer update.
    gradient_accumulation_steps: int = 8

    # Step size used by the optimizer to update trainable LoRA weights.
    learning_rate: float = 2e-4

    # Fraction of early training steps used to gradually increase learning rate.
    warmup_ratio: float = 0.03

    # Number of warmup steps.
    warmup_steps: int = 5

    # Regularization value used to prevent weights from becoming too large.
    weight_decay: float = 0.01

    # Number of training steps after which logs will be printed.
    logging_steps: int = 1
    logging_first_step: bool = True

    # Number of training steps after which validation will be performed.
    eval_steps: int = 10

    # Number of training steps after which a checkpoint will be saved.
    save_steps: int = 25

    # Maximum number of checkpoints to keep; older checkpoints will be deleted.
    save_total_limit: int = 2

    # Maximum number of training steps; -1 means train using num_train_epochs.
    max_steps: int = -1

    # Evaluation strategy.
    eval_strategy: str = "steps"

    # Whether to use fp16 mixed precision (requires GPU).
    fp16: bool = True
    bf16: bool = False

    # Disable external logging tools.
    report_to: str = "none"

    # Keep required columns.
    remove_unused_columns: bool = False

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        for d in [self.output_dir, self.adapter_dir, self.processed_data_dir]:
            os.makedirs(d, exist_ok=True)

    def as_dict(self):
        return asdict(self)


@dataclass
class InstructionConfig:
    # Training will stop after 5 optimizer steps. max_steps overrides num_train_epochs.
    num_train_epochs: float = 5.0
    max_steps: int = 5

    # Batch settings.
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # Optimizer settings.
    learning_rate: float = 1e-4
    warmup_steps: int = 2
    weight_decay: float = 0.01

    output_dir: str = "./instruction_output"
    adapter_dir: str = "./instruction_adapter"

    # Max length for instruction tokenization.
    max_length: int = 512

    # Precision settings.
    fp16: bool = True
    bf16: bool = False

    # Logging, eval, and checkpoint settings.
    logging_steps: int = 1
    eval_steps: int = 1
    save_steps: int = 5
    save_total_limit: int = 2


@dataclass
class PreferenceConfig:
    # Training duration
    num_train_epochs: float = 3.0
    max_steps: int = 5

    # Batch settings
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # Optimizer settings
    learning_rate: float = 5e-5
    warmup_steps: int = 2
    weight_decay: float = 0.01

    # DPO hyperparameter - beta controls how strongly the model is pushed
    # toward chosen answers over rejected answers.
    beta: float = 0.1

    # Sequence length settings.
    max_length: int = 512
    max_prompt_length: int = 256

    output_dir: str = "./preference_output"
    adapter_dir: str = "./preference_adapter"

    # Precision settings.
    fp16: bool = False
    bf16: bool = False

    # Logging and evaluation
    logging_steps: int = 1
    eval_steps: int = 1
    save_steps: int = 5
    save_total_limit: int = 2
