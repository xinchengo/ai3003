import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_ROOT = Path("results/remote_json")
FIGURE_DIR = Path("figures")


def load_run(group, timestamp):
    run_dir = RESULTS_ROOT / group / timestamp
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    with open(run_dir / "pretrain_history.json") as f:
        history = json.load(f)
    return config, summary, history


def smooth(values, window=15):
    if len(values) <= window:
        return values
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def plot_loss(ax, group, timestamp, label, every=1, window=15):
    _, _, history = load_run(group, timestamp)
    epochs = [item["epoch"] for item in history][::every]
    losses = smooth([item["loss"] for item in history], window=window)[::every]
    ax.plot(epochs, losses, linewidth=1.8, label=label)


def plot_normalized_loss(ax, group, timestamp, label, every=1, window=15):
    _, _, history = load_run(group, timestamp)
    epochs = [item["epoch"] for item in history][::every]
    losses = smooth([item["loss"] for item in history], window=window)
    start, end = losses[0], losses[-1]
    scale = start - end
    if abs(scale) < 1e-12:
        normalized = [0.0 for _ in losses]
    else:
        normalized = [(value - end) / scale for value in losses]
    ax.plot(epochs, normalized[::every], linewidth=1.8, label=label)


def save_main_curves():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    plot_loss(ax, "r1-resnet18-simclr", "20260515_155635", "ResNet-18 r1", every=5)
    plot_loss(ax, "r10-resnet18-simclr", "20260513_111858", "ResNet-18 r10", every=3)
    plot_loss(ax, "r1-mobilenet_v2-simclr", "20260521_145720", "MobileNetV2 r1", every=3)
    plot_loss(ax, "r10-mobilenet_v2-simclr", "20260522_085015", "MobileNetV2 r10", every=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title("Main SimCLR pretraining curves")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "main_pretrain_curves.pdf")
    plt.close(fig)


def save_loss_curves():
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    runs = [
        ("r10-resnet18-simclr", "20260513_111858", "NT-Xent"),
        ("r10-resnet18-loss-logistic-bs2048-t05", "20260514_135241", "Logistic"),
        ("r10-resnet18-loss-triplet-bs2048-t05", "20260514_152701", "Triplet"),
    ]
    for group, timestamp, label in runs:
        plot_normalized_loss(ax, group, timestamp, label, every=3, window=15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized pretraining loss")
    ax.set_title("Contrastive-loss pretraining curves")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "loss_pretrain_curves.pdf")
    plt.close(fig)


def save_hyper_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    temp_runs = [
        ("r10-resnet18-simclr", "20260513_233303", r"$\tau=0.1$"),
        ("r10-resnet18-simclr", "20260514_011722", r"$\tau=0.25$"),
        ("r10-resnet18-simclr", "20260513_111858", r"$\tau=0.5$"),
        ("r10-resnet18-simclr", "20260514_024545", r"$\tau=1.0$"),
        ("r10-resnet18-simclr", "20260514_054231", r"$\tau=5.0$"),
    ]
    for group, timestamp, label in temp_runs:
        plot_loss(axes[0], group, timestamp, label, every=4, window=15)
    axes[0].set_title("Temperature")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NT-Xent loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    batch_runs = [
        ("r10-resnet18-simclr", "20260513_162648", "batch=256"),
        ("r10-resnet18-simclr", "20260513_145044", "batch=512"),
        ("r10-resnet18-simclr", "20260513_131947", "batch=1024"),
        ("r10-resnet18-simclr", "20260513_111858", "batch=2048"),
    ]
    for group, timestamp, label in batch_runs:
        plot_loss(axes[1], group, timestamp, label, every=4, window=15)
    axes[1].set_title("Batch size")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NT-Xent loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "hyper_pretrain_curves.pdf")
    plt.close(fig)


def print_summary():
    selected = [
        ("r1-resnet18-simclr", "20260515_155635"),
        ("r10-resnet18-simclr", "20260513_111858"),
        ("r1-mobilenet_v2-simclr", "20260521_145720"),
        ("r10-mobilenet_v2-simclr", "20260522_085015"),
        ("r10-resnet18-loss-logistic-bs2048-t05", "20260514_135241"),
        ("r10-resnet18-loss-triplet-bs2048-t05", "20260514_152701"),
        ("r10-resnet18-head-nobn-h128-p64-bs2048-t05", "20260514_165634"),
        ("r10-resnet18-head-bn-h64-p64-bs2048-t05", "20260514_195441"),
        ("r10-resnet18-head-bn-h128-p128-bs2048-t05", "20260514_212342"),
        ("r10-resnet18-simclr", "20260513_162648"),
        ("r10-resnet18-simclr", "20260513_131947"),
        ("r10-resnet18-simclr", "20260514_024545"),
    ]
    for group, timestamp in selected:
        config, summary, history = load_run(group, timestamp)
        test = summary["test"]
        print(
            group,
            timestamp,
            "temp=", config.get("temperature"),
            "bs=", config.get("pretrain_batch_size"),
            "head=", config.get("head_use_batchnorm"), config.get("head_hidden_dim"), config.get("projection_dim"),
            "last_loss=", f"{history[-1]['loss']:.4f}",
            "acc=", f"{test['accuracy']:.4f}",
            "f1=", f"{test['f1']:.4f}",
        )


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    save_main_curves()
    save_loss_curves()
    save_hyper_curves()
    print_summary()


if __name__ == "__main__":
    main()
