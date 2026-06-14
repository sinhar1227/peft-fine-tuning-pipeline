# End-to-End LLM Fine-Tuning: Non-Instruction → Instruction → Preference

A 3-stage parameter-efficient fine-tuning (PEFT) pipeline that demonstrates how to adapt a base language model using LoRA/QLoRA across three increasingly sophisticated training paradigms:

1. **Non-Instruction (Domain-Adaptive Pre-training)** — Causal language modeling on raw text
2. **Instruction Fine-Tuning** — Supervised fine-tuning on instruction-response pairs
3. **Preference Fine-Tuning (DPO)** — Direct Preference Optimization for alignment

Built with Hugging Face Transformers, PEFT, TRL, and Streamlit.

---

## Pipeline Overview

```
Raw Text (PDF/Corpus)
    │
    ▼
Stage 1: Non-Instruction Fine-Tuning
    │  (Causal LM on domain text + LoRA)
    ▼
Stage 2: Instruction Fine-Tuning
    │  (Alpaca-style SFT + LoRA)
    ▼
Stage 3: Preference Tuning (DPO)
       (Direct Preference Optimization + LoRA)
```

Each stage produces a standalone LoRA adapter that can be reused, merged, or further fine-tuned in subsequent stages.

---

## Project Structure

```
├── app.py                        # Main Streamlit entry point
├── config.py                     # Dataclass configs for all three stages
├── requirements.txt              # Dependencies
├── data/
│   ├── pdf_processor.py          # PDF text extraction and cleaning
│   └── dataset_builder.py        # HF dataset creation, tokenization, formatting
├── models/
│   ├── model_loader.py           # Model loading with 4-bit quantization
│   ├── lora_setup.py             # LoRA configuration and adapter management
│   ├── inference.py              # Text generation helpers
│   └── training_callback.py      # Real-time log capture for the UI
├── stages/
│   ├── non_instruction.py        # Stage 1: domain-adaptive pre-training
│   ├── instruction.py            # Stage 2: SFT on instruction pairs
│   └── preference.py             # Stage 3: DPO preference tuning
└── ui/
    └── components.py             # Reusable Streamlit UI components
```

---

## Data Format

### Stage 1 — Non-Instruction

Any text corpus. The app extracts text from uploaded PDFs, cleans it, splits it into paragraphs, tokenizes it, and packs it into fixed-size training blocks for causal LM.

### Stage 2 — Instruction

JSONL file with Alpaca-style fields:

```jsonl
{"instruction": "...", "input": "...", "output": "..."}
{"instruction": "...", "output": "..."}
```

Converted internally to:

```
### Instruction:
...
### Input:
...
### Response:
...
```

### Stage 3 — Preference (DPO)

JSONL file with preference triples:

```jsonl
{"prompt": "...", "chosen": "...", "rejected": "..."}
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

---

## Training Dashboard

After each training run, the UI displays:

- **Metric cards** — Final training loss, step count, evaluation loss
- **Loss curve** — Line chart of training loss over steps
- **Eval loss curve** — Line chart when evaluation is configured
- **Step-by-step log table** — Expandable table with every logged entry (loss, learning rate, epoch, eval metrics)

Real-time status messages are shown during training so you always know what is happening.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (falls back to CPU)
- See `requirements.txt` for full dependency list

Key libraries:

- `streamlit` — UI framework
- `transformers` — Model loading and training
- `peft` — LoRA/QLoRA adapters
- `trl` — DPO trainer
- `bitsandbytes` — 4-bit quantization
- `datasets` — Dataset handling
- `PyMuPDF` — PDF text extraction

---

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd streamlit_app

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

Navigate between the three stages using the sidebar. Each stage is independent — you can start at any point if you have the required data and any previously trained adapters.

Samples of training data for each stage are provided in the `Sample/` directory to help you get started quickly. You can also use your own datasets by following the specified formats.

---

## How It Works

### Stage 1: Non-Instruction

Upload a text corpus (PDF) → extract and clean text → create paragraph-level records → tokenize and pack into fixed-size blocks → apply LoRA → train via causal language modeling → save adapter.

### Stage 2: Instruction

Upload instruction data (JSONL) → format into Alpaca-style templates → load the Stage 1 model (or a fresh base) → apply a new LoRA adapter → train via causal LM with label masking on padding → save adapter.

### Stage 3: Preference (DPO)

Upload preference data (JSONL) → load the Stage 2 merged model → apply a new LoRA adapter → train via Direct Preference Optimization → save adapter.

Each stage uses `LogHistoryCallback` (a custom `TrainerCallback`) to capture loss values, learning rates, and evaluation metrics in real time. These are returned to the Streamlit UI and rendered as charts and tables after training completes.
