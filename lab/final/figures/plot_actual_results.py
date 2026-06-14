from __future__ import annotations

import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP_CACHE = Path(tempfile.gettempdir()) / "ai3003_final_mpl_cache"
(TMP_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(TMP_CACHE / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = HERE / "actual_results_data.csv"
OUT_PATH = HERE / "actual_results_curve.pdf"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["excess_return"] = df["my_return"] - df["hs300_return"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )

    ax = axes[0]
    ax.plot(df["date"], df["my_return"] * 100, marker="o", linewidth=2.0, label="Actual portfolio", color="#d62728")
    ax.plot(df["date"], df["hs300_return"] * 100, marker="o", linewidth=2.0, label="CSI 300", color="#1f77b4")
    ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Actual Competition Return vs. CSI 300")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")

    final = df.iloc[-1]
    ax.annotate(
        f"Final: {final['my_return'] * 100:.2f}%",
        xy=(final["date"], final["my_return"] * 100),
        xytext=(-78, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.0},
        color="#d62728",
    )

    ax2 = axes[1]
    excess = df.dropna(subset=["excess_return"])
    ax2.bar(excess["date"], excess["excess_return"] * 100, width=0.75, color="#6f6f6f", alpha=0.75)
    ax2.axhline(0, color="#444444", linewidth=0.8)
    ax2.set_ylabel("Excess (%)")
    ax2.set_xlabel("Date")
    ax2.grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
