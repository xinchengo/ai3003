import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    "sinusoidal": "checkpoints/bpe-hf-v14c8-end2end-v14-sinusoidal-stage1/20260425_084137",
    "RoPE": "checkpoints/bpe-hf-v14c8-end2end-v14-rotary-stage1/20260425_084135",
    "no pos": "checkpoints/bpe-hf-v14c8-end2end-v14-nopos-stage1/20260425_084140",
    "causal RoPE": "checkpoints/bpe-hf-v14c8-end2end-v14-rotary-causal-stage1/20260425_084141",
    "RNN": "checkpoints/bpe-hf-v14c8-end2end-rnn-v14-stage1/20260425_082843",
    "MLP tiny": "checkpoints/bpe-hf-v14c8-end2end-mlp-tiny-stage1/20260425_084150",
}


def safe_name(name):
    return name.lower().replace(" ", "_")


def clean_modal_stdout(text):
    return text.split("✓", 1)[0].strip()


def download_histories(history_root, volume_name):
    history_root.mkdir(parents=True, exist_ok=True)
    for name, run_dir in RUNS.items():
        out_dir = history_root / safe_name(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "training_history.json"
        remote_path = f"/{run_dir}/training_history.json"
        proc = subprocess.run(
            ["modal", "volume", "get", volume_name, remote_path, "-"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        target.write_text(clean_modal_stdout(proc.stdout))


def load_histories(history_root):
    histories = {}
    for name in RUNS:
        path = history_root / safe_name(name) / "training_history.json"
        text = clean_modal_stdout(path.read_text())
        histories[name] = json.loads(text)
    return histories


def epochs(records):
    return [record["epoch"] for record in records]


def values(records, key):
    if key == "train_acc":
        return [record["train"]["accuracy"] for record in records]
    if key == "val_acc":
        return [record["val"]["accuracy"] for record in records]
    if key == "train_loss":
        return [record["train"]["loss"] for record in records]
    if key == "val_loss":
        return [record["val"]["loss"] for record in records]
    raise KeyError(key)


def style_axes(axs):
    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)


def save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=180)
    plt.close(fig)


def plot_main_curves(histories, output_dir):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    for name in ["RoPE", "RNN"]:
        x = epochs(histories[name])
        axs[0].plot(x, values(histories[name], "train_acc"), label=f"{name} train")
        axs[0].plot(x, values(histories[name], "val_acc"), "--", label=f"{name} val")
        axs[1].plot(x, values(histories[name], "train_loss"), label=f"{name} train")
        axs[1].plot(x, values(histories[name], "val_loss"), "--", label=f"{name} val")

    axs[0].set_title("Main Models Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim(0.45, 1.02)
    axs[1].set_title("Main Models Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_ylim(bottom=0)
    style_axes(axs)
    save(fig, output_dir, "main_training_curves")


def plot_positional_curves(histories, output_dir):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    for name in ["no pos", "sinusoidal", "RoPE"]:
        x = epochs(histories[name])
        axs[0].plot(x, values(histories[name], "train_acc"), alpha=0.55, label=f"{name} train")
        axs[0].plot(x, values(histories[name], "val_acc"), "--", label=f"{name} val")
        axs[1].plot(x, values(histories[name], "val_loss"), label=name)

    axs[0].set_title("Position Encoding Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim(0.45, 1.02)
    axs[1].set_title("Position Encoding Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_ylim(bottom=0)
    style_axes(axs)
    save(fig, output_dir, "positional_training_curves")


def plot_causal_baseline_curves(histories, output_dir):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    for name in ["RoPE", "causal RoPE", "MLP tiny"]:
        x = epochs(histories[name])
        axs[0].plot(x, values(histories[name], "train_acc"), alpha=0.55, label=f"{name} train")
        axs[0].plot(x, values(histories[name], "val_acc"), "--", label=f"{name} val")
        axs[1].plot(x, values(histories[name], "val_loss"), label=name)

    axs[0].set_title("Causal/MLP Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim(0.45, 1.02)
    axs[1].set_title("Causal/MLP Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_ylim(bottom=0)
    style_axes(axs)
    save(fig, output_dir, "causal_baseline_training_curves")


def main():
    parser = argparse.ArgumentParser(description="Plot Lab 3 training curves.")
    parser.add_argument("--history-root", default=".build/training_histories")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--volume-name", default="ai3003-lab3-results")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    history_root = Path(args.history_root)
    output_dir = Path(args.output_dir)
    if args.download or not history_root.exists():
        download_histories(history_root, args.volume_name)

    histories = load_histories(history_root)
    plt.rcParams.update({"font.size": 10})
    plot_main_curves(histories, output_dir)
    plot_positional_curves(histories, output_dir)
    plot_causal_baseline_curves(histories, output_dir)


if __name__ == "__main__":
    main()
