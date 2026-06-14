# ============================================================
# Streamlit Application: PEFT Fine-Tuning Explorer
# ============================================================
# A 3-stage pipeline demonstrating parameter-efficient fine-tuning (LoRA/QLoRA)
# using Hugging Face, PEFT, and TRL.
#
# Stage 1: Non-Instruction (Domain-adaptive pre-training via causal LM)
# Stage 2: Instruction (Supervised fine-tuning on instruction-response pairs)
# Stage 3: Preference / DPO (Align model behavior using preference data)

import streamlit as st
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, InstructionConfig, PreferenceConfig
from ui.components import section_header, log_container, display_sample_output, format_metrics, display_training_logs

st.set_page_config(
    page_title="PEFT Fine-Tuning Explorer",
    page_icon="",
    layout="wide",
)

st.title("PEFT Fine-Tuning: Non-Instruction → Instruction → Preference")
st.markdown(
    "A 3-stage pipeline demonstrating parameter-efficient fine-tuning (LoRA/QLoRA) "
    "using Hugging Face, PEFT, and TRL."
)

SIDEBAR_HELP = """
- **Stage 1 (Non-Instruction):** Domain-adaptive pre-training on raw text via causal LM → adapter `stage1`
- **Stage 2 (Instruction):** Stack `stage1` (frozen) + new trainable LoRA → adapter `stage2`
- **Stage 3 (Preference / DPO):** Stack `stage1+stage2` (frozen) + new trainable LoRA → adapter `stage3`
"""

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Overview", "Stage 1: Non-Instruction", "Stage 2: Instruction", "Stage 3: Preference / DPO"],
    )
    st.divider()
    st.markdown("### About")
    st.markdown(SIDEBAR_HELP)
    st.divider()
    st.caption("Built with Streamlit + Hugging Face + PEFT + TRL")


def make_status_area():
    # Create status display area with a log placeholder and progress placeholder.
    container = st.container()
    log_placeholder = container.empty()
    progress_placeholder = container.empty()
    return container, log_placeholder, progress_placeholder


def status_callback_factory(log_placeholder):
    # Factory that creates a status callback which updates a Streamlit placeholder.
    def callback(msg):
        log_placeholder.info(f" {msg}")
        time.sleep(0.05)
    return callback


# ============================================================
# Page: Overview
# ============================================================

if page == "Overview":
    section_header("3-Stage Fine-Tuning Pipeline", "End-to-end LLM fine-tuning with PEFT")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Stage 1: Non-Instruction")
        st.markdown(
            "- **Goal:** Domain adaptation via causal LM\n"
            "- **Data:** Raw text paragraphs (pharma PDF)\n"
            "- **Method:** LoRA on all attention + FFN layers\n"
            "- **Output:** Domain-adapted LoRA adapter"
        )

    with col2:
        st.markdown("### Stage 2: Instruction")
        st.markdown(
            "- **Goal:** Teach model to follow instructions\n"
            "- **Data:** Alpaca-style (instruction, input, output)\n"
            "- **Method:** Stack Stage 1 adapter (frozen) + new trainable LoRA\n"
            "- **Output:** Instruction-tuned LoRA adapter (stage2)"
        )

    with col3:
        st.markdown("### Stage 3: Preference / DPO")
        st.markdown(
            "- **Goal:** Align model to preferred responses\n"
            "- **Data:** (prompt, chosen, rejected) triples\n"
            "- **Method:** Stack Stage 1+2 adapters (frozen) + new trainable LoRA\n"
            "- **Output:** Preference-tuned LoRA adapter (stage3)"
        )

    st.divider()

    st.markdown("### Quick Start")
    st.markdown(
        "1. **Stage 1:** Upload a PDF → extract text → train non-instruction LoRA (`stage1`)\n"
        "2. **Stage 2:** Provide instruction JSONL → stack `stage1` (frozen) + new trainable LoRA (`stage2`)\n"
        "3. **Stage 3:** Provide preference JSONL → stack `stage1` + `stage2` (frozen) + new trainable LoRA (`stage3`)\n\n"
        "Each stage keeps adapters separate for compositional inference. All adapters compose at inference."
    )


# ============================================================
# Page: Stage 1 - Non-Instruction Fine-Tuning
# ============================================================

elif page == "Stage 1: Non-Instruction":
    section_header(
        "Stage 1: Non-Instruction Fine-Tuning",
        "Domain-adaptive pre-training via causal language modeling on raw text."
    )

    config = Config()

    with st.expander("Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            config.model_name = st.text_input("Base Model", config.model_name)
            config.block_size = st.number_input("Block Size (tokens)", min_value=64, value=config.block_size, step=64)
            config.lora_r = st.number_input("LoRA Rank (r)", min_value=4, value=config.lora_r, step=4)
            config.lora_alpha = st.number_input("LoRA Alpha", min_value=8, value=config.lora_alpha, step=8)
            config.learning_rate = st.number_input("Learning Rate", min_value=1e-6, value=config.learning_rate, format="%.6f")
        with col2:
            config.num_train_epochs = st.number_input("Epochs", min_value=1.0, value=config.num_train_epochs, step=0.5)
            config.per_device_train_batch_size = st.number_input("Batch Size", min_value=1, value=config.per_device_train_batch_size)
            config.gradient_accumulation_steps = st.number_input("Gradient Accumulation Steps", min_value=1, value=config.gradient_accumulation_steps)
            config.output_dir = st.text_input("Output Directory", config.output_dir)
            config.adapter_dir = st.text_input("Adapter Directory", config.adapter_dir)

    uploaded_pdf = st.file_uploader("Upload a PDF for domain adaptation", type=["pdf"])

    if uploaded_pdf is not None:
        pdf_path = os.path.join(config.processed_data_dir, uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success(f"PDF saved to {pdf_path}")

        st.markdown("### Test Prompts (optional)")
        st.caption("Enter continuation prompts to test after training — one per line. Edit or replace the examples below.")
        test_prompts_raw = st.text_area(
            "Continuation prompts",
            value=(
                "Metformin is one of the most widely prescribed\n"
                "Clinical trials have demonstrated that combining\n"
                "Artificial intelligence is transforming"
            ),
            height=100,
            key="s1_prompts",
        )
        test_prompts = [p.strip() for p in test_prompts_raw.split("\n") if p.strip()]

        container, log_placeholder, progress_placeholder = make_status_area()

        if st.button(" Start Stage 1 Training", type="primary"):
            from stages.non_instruction import run_non_instruction_stage

            callback = status_callback_factory(log_placeholder)

            with st.spinner("Running Stage 1..."):
                result = run_non_instruction_stage(config, pdf_path, status_callback=callback)

            st.success("Stage 1 completed! Adapter saved.")

            # Display live training logs, loss curves, and metrics
            st.subheader(" Training Dashboard")
            display_training_logs(result.get("log_history", []), stage_name="Stage 1 (Non-Instruction)")

            # Test text continuation
            st.subheader("Test Text Continuation")
            from models.inference import generate_completion

            for prompt in test_prompts:
                response = generate_completion(result["model"], result["tokenizer"], prompt)
                display_sample_output("Text Continuation Sample", prompt, response)

    else:
        st.info("Upload a PDF to begin Stage 1 training.")


# ============================================================
# Page: Stage 2 - Instruction Fine-Tuning
# ============================================================

elif page == "Stage 2: Instruction":
    section_header(
        "Stage 2: Instruction Fine-Tuning",
        "Supervised fine-tuning using Alpaca-style instruction-response pairs."
    )

    instruction_config = InstructionConfig()
    config = Config()

    with st.expander("Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            instruction_config.learning_rate = st.number_input("Learning Rate", min_value=1e-6, value=instruction_config.learning_rate, format="%.6f", key="ilr")
            instruction_config.num_train_epochs = st.number_input("Epochs", min_value=1.0, value=instruction_config.num_train_epochs, step=0.5, key="iepochs")
            instruction_config.max_steps = st.number_input("Max Steps (-1 = use epochs)", min_value=-1, value=instruction_config.max_steps, key="isteps")
        with col2:
            instruction_config.per_device_train_batch_size = st.number_input("Batch Size", min_value=1, value=instruction_config.per_device_train_batch_size, key="ibatch")
            instruction_config.gradient_accumulation_steps = st.number_input("Gradient Accumulation Steps", min_value=1, value=instruction_config.gradient_accumulation_steps, key="iaccum")
            instruction_config.adapter_dir = st.text_input("Adapter Output Directory", instruction_config.adapter_dir)

    stage1_adapter_dir = st.text_input(
        "Stage 1 Adapter Directory (from non-instruction stage)",
        value="./adapter",
        help="Path to the non-instruction LoRA adapter from Stage 1. This adapter is loaded frozen; a new trainable adapter is added on top for instruction tuning.",
    )

    instruction_file = st.file_uploader(
        "Upload instruction dataset (JSONL with instruction/input/output fields)",
        type=["jsonl"],
    )

    if instruction_file is not None:
        data_path = os.path.join("processed_data", instruction_file.name)
        os.makedirs("processed_data", exist_ok=True)
        with open(data_path, "wb") as f:
            f.write(instruction_file.read())
        st.success(f"Instruction data saved to {data_path}")

        st.markdown("### Test Questions (optional)")
        st.caption("Enter instruction prompts to test after training — one per line. Edit or replace the examples below.")
        test_questions_raw = st.text_area(
            "Instruction prompts",
            value=(
                "Explain the primary mechanism of action of metformin.\n"
                "Why can atorvastatin and ezetimibe reduce LDL-C more effectively together?\n"
                "Summarize the role of AI in drug discovery."
            ),
            height=100,
            key="s2_questions",
        )
        test_questions = [q.strip() for q in test_questions_raw.split("\n") if q.strip()]

        container, log_placeholder, progress_placeholder = make_status_area()

        if st.button(" Start Stage 2 Training", type="primary"):
            from stages.instruction import run_instruction_stage

            callback = status_callback_factory(log_placeholder)

            with st.spinner("Running Stage 2..."):
                result = run_instruction_stage(
                    config, instruction_config, data_path,
                    stage1_adapter_dir=stage1_adapter_dir,
                    status_callback=callback,
                )

            st.success("Stage 2 completed! Instruction adapter saved.")

            # Display live training logs, loss curves, and metrics
            st.subheader(" Training Dashboard")
            display_training_logs(result.get("log_history", []), stage_name="Stage 2 (Instruction)")

            # Test instruction-tuned model
            st.subheader("Test Instruction Following")
            from models.inference import generate_instruction_response

            for question in test_questions:
                response = generate_instruction_response(result["model"], result["tokenizer"], question)
                display_sample_output("Q&A Sample", question, response)

    else:
        st.info("Upload a JSONL instruction dataset to begin Stage 2.")


# ============================================================
# Page: Stage 3 - Preference Fine-Tuning (DPO)
# ============================================================

elif page == "Stage 3: Preference / DPO":
    section_header(
        "Stage 3: Preference Fine-Tuning (DPO)",
        "Direct Preference Optimization to align model responses with human preferences."
    )

    preference_config = PreferenceConfig()
    config = Config()

    with st.expander("Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            preference_config.learning_rate = st.number_input("Learning Rate", min_value=1e-6, value=preference_config.learning_rate, format="%.6f", key="plr")
            preference_config.beta = st.number_input("DPO Beta", min_value=0.01, value=preference_config.beta, step=0.05, format="%.2f")
            preference_config.num_train_epochs = st.number_input("Epochs", min_value=1.0, value=preference_config.num_train_epochs, step=0.5, key="pepochs")
        with col2:
            preference_config.max_steps = st.number_input("Max Steps", min_value=-1, value=preference_config.max_steps, key="psteps")
            preference_config.max_length = st.number_input("Max Sequence Length", min_value=128, value=preference_config.max_length, step=64)
            preference_config.adapter_dir = st.text_input("Adapter Output Directory", preference_config.adapter_dir, key="padapter")

    col1, col2 = st.columns(2)
    with col1:
        pref_stage1_adapter = st.text_input(
            "Stage 1 Adapter Directory",
            value="./adapter",
            key="pref_s1",
            help="Non-instruction LoRA adapter from Stage 1 (loaded frozen).",
        )
    with col2:
        pref_stage2_adapter = st.text_input(
            "Stage 2 Adapter Directory",
            value="./instruction_adapter",
            key="pref_s2",
            help="Instruction-tuned LoRA adapter from Stage 2 (loaded frozen). A new trainable adapter is added on top for DPO.",
        )

    preference_file = st.file_uploader(
        "Upload preference dataset (JSONL with prompt/chosen/rejected fields)",
        type=["jsonl"],
    )

    if preference_file is not None:
        data_path = os.path.join("processed_data", preference_file.name)
        os.makedirs("processed_data", exist_ok=True)
        with open(data_path, "wb") as f:
            f.write(preference_file.read())
        st.success(f"Preference data saved to {data_path}")

        st.markdown("### Test Questions (optional)")
        st.caption("Enter prompts to test after DPO tuning — one per line. Edit or replace the examples below.")
        test_questions_raw = st.text_area(
            "Preference-tuned prompts",
            value=(
                "Explain the primary mechanism of action of metformin.\n"
                "Why should AI predictions in drug discovery be experimentally validated?\n"
                "Define pharmacovigilance."
            ),
            height=100,
            key="s3_questions",
        )
        test_questions = [q.strip() for q in test_questions_raw.split("\n") if q.strip()]

        container, log_placeholder, progress_placeholder = make_status_area()

        if st.button(" Start Stage 3 DPO Training", type="primary"):
            from stages.preference import run_preference_stage

            callback = status_callback_factory(log_placeholder)

            with st.spinner("Running Stage 3 (DPO)..."):
                result = run_preference_stage(
                    config, preference_config, data_path,
                    stage1_adapter_dir=pref_stage1_adapter,
                    stage2_adapter_dir=pref_stage2_adapter,
                    status_callback=callback,
                )

            st.success("Stage 3 completed! DPO adapter saved.")

            # Display live training logs, loss curves, and metrics
            st.subheader(" Training Dashboard")
            display_training_logs(result.get("log_history", []), stage_name="Stage 3 (DPO)")

            # Test preference-tuned model
            st.subheader("Test Preference-Tuned Model")
            from models.inference import generate_preference_response

            for question in test_questions:
                response = generate_preference_response(result["model"], result["tokenizer"], question)
                display_sample_output("Q&A (DPO) Sample", question, response)

    else:
        st.info("Upload a JSONL preference dataset to begin Stage 3.")
