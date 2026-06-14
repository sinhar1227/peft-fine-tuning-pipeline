# ============================================================
# Training Log Callback
# ============================================================
# Custom Trainer callback that captures log history in real-time
# so it can be displayed in the Streamlit UI.

from transformers import TrainerCallback
import pandas as pd


class LogHistoryCallback(TrainerCallback):
    # Custom callback that stores every training/eval log entry
    # and provides a method to display them as a DataFrame.

    def __init__(self, status_callback=None):
        self.logs = []
        self.status_callback = status_callback

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = dict(logs)
        entry["step"] = state.global_step
        self.logs.append(entry)
        if self.status_callback and "loss" in entry:
            self.status_callback(
                f"Step {entry['step']} — loss: {entry['loss']:.4f}"
                + (f" | eval_loss: {entry['eval_loss']:.4f}" if "eval_loss" in entry else "")
            )

    def get_training_df(self):
        # Return a DataFrame with only training loss entries.
        train_logs = [log for log in self.logs if "loss" in log and "eval_loss" not in log]
        if not train_logs:
            return pd.DataFrame()
        df = pd.DataFrame(train_logs)
        keep = [c for c in ["step", "loss", "learning_rate", "epoch"] if c in df.columns]
        return df[keep]

    def get_eval_df(self):
        # Return a DataFrame with only evaluation loss entries.
        eval_logs = [log for log in self.logs if "eval_loss" in log]
        if not eval_logs:
            return pd.DataFrame()
        df = pd.DataFrame(eval_logs)
        keep = [c for c in ["step", "eval_loss", "eval_accuracy", "epoch"] if c in df.columns]
        return df[keep]

    def get_raw_logs(self):
        return self.logs
