# ============================================================
# Reusable Streamlit UI Components
# ============================================================

import streamlit as st
import time
import pandas as pd


def section_header(title: str, description: str = ""):
    st.subheader(title)
    if description:
        st.markdown(description)
    st.divider()


def metric_card(label: str, value, delta=None):
    st.metric(label=label, value=value, delta=delta)


def log_container():
    return st.empty()


def progress_bar():
    return st.progress(0)


def status_box(container, log_area, message: str):
    timestamp = time.strftime("%H:%M:%S")
    log_area.markdown(f"**{message}**")


def display_sample_output(title: str, prompt: str, response: str):
    with st.expander(title, expanded=False):
        st.markdown("**Prompt:**")
        st.code(prompt, language="text")
        st.markdown("**Response:**")
        st.code(response, language="text")


def format_metrics(metrics_dict: dict) -> str:
    lines = []
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            lines.append(f"- **{key}:** {value:.6f}")
        else:
            lines.append(f"- **{key}:** {value}")
    return "\n".join(lines)


# ============================================================
# Training Log Display
# ============================================================

def display_training_logs(log_history: list, stage_name: str = "Training"):
    # Display training logs as metrics, a loss chart, and a data table.
    # log_history is a list of dicts captured by LogHistoryCallback.
    if not log_history:
        st.info(f"No {stage_name} log entries captured.")
        return

    train_logs = [log for log in log_history if "loss" in log and "eval_loss" not in log]
    eval_logs = [log for log in log_history if "eval_loss" in log]

    col1, col2, col3 = st.columns(3)

    # Show final training loss if available
    if train_logs:
        final_loss = train_logs[-1].get("loss")
        initial_loss = train_logs[0].get("loss") if len(train_logs) > 1 else None
        delta = round(final_loss - initial_loss, 4) if initial_loss and initial_loss != final_loss else None
        col1.metric(label="Final Training Loss", value=f"{final_loss:.4f}" if final_loss else "N/A", delta=delta)
        col2.metric(label="Training Steps", value=train_logs[-1].get("step", len(train_logs)))

    if eval_logs:
        final_eval_loss = eval_logs[-1].get("eval_loss")
        col3.metric(label="Final Eval Loss", value=f"{final_eval_loss:.4f}" if final_eval_loss else "N/A")

    st.divider()

    # Plot training loss curve
    if train_logs:
        st.subheader(f" Loss Curve ({stage_name})")
        df_train = pd.DataFrame(train_logs)
        if "step" in df_train.columns and "loss" in df_train.columns:
            st.line_chart(df_train.set_index("step")["loss"])
    st.caption("Training loss over steps — lower is better.")

    # Plot eval loss curve
    if eval_logs and len(eval_logs) > 1:
        st.subheader(" Eval Loss Curve")
        df_eval = pd.DataFrame(eval_logs)
        if "step" in df_eval.columns and "eval_loss" in df_eval.columns:
            st.line_chart(df_eval.set_index("step")["eval_loss"])

    st.divider()

    # Show raw log table
    with st.expander(" Detailed Step-by-Step Logs", expanded=False):
        df_all = pd.DataFrame(log_history)
        # Drop verbose columns that are not useful
        drop_cols = [c for c in ["runtime", "samples_per_second", "steps_per_second"] if c in df_all.columns]
        df_all = df_all.drop(columns=drop_cols, errors="ignore")
        st.dataframe(df_all, use_container_width=True)
