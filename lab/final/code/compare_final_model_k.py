from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data."
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(group: pd.DataFrame) -> dict:
    returns = group["final_return"].astype(float).to_numpy()
    sharpes = group["sharpe"].astype(float).to_numpy()
    drawdowns = group["max_drawdown"].astype(float).to_numpy()
    return {
        "window_count": int(len(group)),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "p25_return": float(np.percentile(returns, 25)),
        "compound_return": float(np.prod(1.0 + returns) - 1.0),
        "return_std": float(np.std(returns)),
        "best_return": float(np.max(returns)),
        "worst_return": float(np.min(returns)),
        "positive_rate": float(np.mean(returns > 0)),
        "mean_sharpe": float(np.mean(sharpes)),
        "worst_drawdown": float(np.min(drawdowns)),
        "mean_drawdown": float(np.mean(drawdowns)),
        "avg_turnover": float(group["avg_turnover"].astype(float).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "experiments" / "sweep_20260531_205617" / "improved_h3_e50" / "best.pt"),
    )
    p.add_argument("--panel", default=str(PROJECT_ROOT / "data" / "panel_hs300_advanced.npz"))
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "experiments" / "final_model_k_compare_20260601"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = [
        ("competition_w1", "2026-03-02", "2026-03-13", True),
        ("competition_w2", "2026-03-16", "2026-03-27", True),
        ("competition_w3", "2026-03-30", "2026-04-13", True),
        ("competition_w4", "2026-04-14", "2026-04-27", True),
        ("competition_w5", "2026-04-28", "2026-05-14", True),
        ("previous10", "2026-04-27", "2026-05-13", False),
        ("latest10", "2026-05-14", "2026-05-27", False),
        ("ytd2026", "2026-01-05", "2026-05-27", False),
    ]
    ks = [3, 5]
    rows: list[dict] = []
    for k in ks:
        k_dir = out_dir / f"k{k}"
        k_dir.mkdir(exist_ok=True)
        for window_name, start, end, is_main in windows:
            out_csv = k_dir / f"{window_name}.csv"
            holdings_csv = k_dir / f"{window_name}_holdings.csv"
            metrics_json = k_dir / f"{window_name}_metrics.json"
            if not metrics_json.exists():
                run_cmd(
                    [
                        sys.executable,
                        "code/back_test_equal.py",
                        "--panel",
                        args.panel,
                        "--checkpoint",
                        args.checkpoint,
                        "--seq_len",
                        "30",
                        "--start",
                        start,
                        "--end",
                        end,
                        "--top_n",
                        "10",
                        "--k",
                        str(k),
                        "--fee",
                        "0.0005",
                        "--strategy",
                        "improved",
                        "--rebalance_interval",
                        "5",
                        "--stop_loss",
                        "-0.035",
                        "--out",
                        str(out_csv),
                        "--holdings_out",
                        str(holdings_csv),
                        "--metrics_out",
                        str(metrics_json),
                        "--device",
                        args.device,
                    ]
                )
            metrics = load_json(metrics_json)
            rows.append(
                {
                    "k": k,
                    "rule": f"buy{k}_sell{k}",
                    "window_name": window_name,
                    "start": start,
                    "end": end,
                    "is_main_competition_window": is_main,
                    **metrics,
                    "daily_csv": str(out_csv),
                    "holdings_csv": str(holdings_csv),
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "k_compare_window_detail.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for k, group in detail[detail["is_main_competition_window"]].groupby("k"):
        row = {"scope": "five_complete_10d_windows", "k": int(k), "rule": f"buy{k}_sell{k}"}
        row.update(summarize(group))
        summary_rows.append(row)
    for scope in ["previous10", "latest10", "ytd2026"]:
        for k, group in detail[detail["window_name"].eq(scope)].groupby("k"):
            row = {"scope": scope, "k": int(k), "rule": f"buy{k}_sell{k}"}
            row.update(
                {
                    "window_count": 1,
                    "mean_return": float(group.iloc[0]["final_return"]),
                    "median_return": float(group.iloc[0]["final_return"]),
                    "p25_return": float(group.iloc[0]["final_return"]),
                    "compound_return": float(group.iloc[0]["final_return"]),
                    "return_std": 0.0,
                    "best_return": float(group.iloc[0]["final_return"]),
                    "worst_return": float(group.iloc[0]["final_return"]),
                    "positive_rate": float(group.iloc[0]["final_return"] > 0),
                    "mean_sharpe": float(group.iloc[0]["sharpe"]),
                    "worst_drawdown": float(group.iloc[0]["max_drawdown"]),
                    "mean_drawdown": float(group.iloc[0]["max_drawdown"]),
                    "avg_turnover": float(group.iloc[0]["avg_turnover"]),
                }
            )
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "k_compare_summary.csv", index=False, encoding="utf-8-sig")

    main = summary[summary["scope"].eq("five_complete_10d_windows")].sort_values("k")
    latest = summary[summary["scope"].eq("latest10")].sort_values("k")
    ytd = summary[summary["scope"].eq("ytd2026")].sort_values("k")
    best_main = main.sort_values(["mean_return", "median_return", "mean_sharpe"], ascending=False).iloc[0]

    lines = [
        "# Final Model Buy/Sell K Comparison",
        "",
        "- Model: `improved_h3_e50_best__improved__i5__slm0p035`",
        f"- Checkpoint: `{args.checkpoint}`",
        "- Fixed strategy: improved score, Top10 equal weight, rebalance interval 5, stop loss -3.5%, fee 0.0005.",
        "- Only changed parameter: `k=3` vs `k=5`.",
        "",
        "## Recommendation",
        "",
        f"Recommended rule: `{best_main['rule']}` based on the five independent complete 10-day competition windows.",
        "",
        "## Five Complete 10-Day Competition Windows",
        "",
        md_table(
            main[
            [
                "rule",
                "mean_return",
                "median_return",
                "p25_return",
                "compound_return",
                "return_std",
                "positive_rate",
                "mean_sharpe",
                "worst_drawdown",
                "avg_turnover",
            ]
        ]),
        "",
        "## Latest 10 Days",
        "",
        md_table(latest[["rule", "mean_return", "mean_sharpe", "worst_drawdown", "avg_turnover"]]),
        "",
        "## 2026 YTD Continuous Backtest",
        "",
        md_table(ytd[["rule", "mean_return", "mean_sharpe", "worst_drawdown", "avg_turnover"]]),
        "",
        "## Window Detail",
        "",
        md_table(
            detail[
            [
                "window_name",
                "start",
                "end",
                "rule",
                "final_return",
                "sharpe",
                "max_drawdown",
                "win_rate",
                "avg_turnover",
            ]
        ]),
    ]
    (out_dir / "k_compare_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved comparison to {out_dir}")


if __name__ == "__main__":
    main()
