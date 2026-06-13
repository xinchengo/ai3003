from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import segmented_competition_eval as segmented


def _summarize(complete: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model, group in complete.groupby("model"):
        returns = group["final_return"].astype(float)
        sharpes = group["sharpe"].astype(float)
        drawdowns = group["max_drawdown"].astype(float)
        row0 = group.iloc[0].to_dict()
        rows.append(
            {
                "model": model,
                "checkpoint": row0["checkpoint"],
                "checkpoint_name": row0["checkpoint_name"],
                "model_type": row0.get("model_type", "unknown"),
                "label_horizon": int(row0.get("label_horizon", 1)),
                "strategy": row0["strategy"],
                "rebalance_interval": row0["rebalance_interval"],
                "stop_loss": row0["stop_loss"],
                "window_count": int(len(group)),
                "mean_10d_return": float(returns.mean()),
                "median_10d_return": float(returns.median()),
                "p25_10d_return": float(returns.quantile(0.25)),
                "reference_compounded_return": float((1.0 + returns).prod() - 1.0),
                "return_std": float(returns.std(ddof=0)),
                "best_return": float(returns.max()),
                "worst_return": float(returns.min()),
                "positive_window_rate": float((returns > 0).mean()),
                "mean_10d_sharpe": float(sharpes.mean()),
                "median_10d_sharpe": float(sharpes.median()),
                "worst_10d_drawdown": float(drawdowns.min()),
                "mean_drawdown": float(drawdowns.mean()),
            }
        )
    return segmented._add_scores(pd.DataFrame(rows))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--main_dir", type=Path, required=True)
    p.add_argument("--baseline_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--baseline_name", default="original_runs_baseline_best")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    complete = pd.read_csv(args.main_dir / "segmented_window_results.csv")
    tail = pd.read_csv(args.main_dir / "segmented_tail_check.csv")
    baseline_complete = pd.read_csv(args.baseline_dir / "segmented_window_results.csv")
    baseline_tail = pd.read_csv(args.baseline_dir / "segmented_tail_check.csv")
    for frame in (baseline_complete, baseline_tail):
        frame["model"] = args.baseline_name

    complete_all = pd.concat([complete, baseline_complete], ignore_index=True)
    tail_all = pd.concat([tail, baseline_tail], ignore_index=True)
    summary = _summarize(complete_all)

    complete_windows = pd.read_csv(args.main_dir / "complete_windows.csv")
    tail_windows = pd.read_csv(args.main_dir / "tail_windows.csv")

    complete_all.to_csv(args.out_dir / "segmented_window_results.csv", index=False, encoding="utf-8-sig")
    complete_all.to_csv(args.out_dir / "complete_window_results.csv", index=False, encoding="utf-8-sig")
    tail_all.to_csv(args.out_dir / "segmented_tail_check.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "segmented_model_summary.csv", index=False, encoding="utf-8-sig")
    complete_windows.to_csv(args.out_dir / "complete_windows.csv", index=False, encoding="utf-8-sig")
    tail_windows.to_csv(args.out_dir / "tail_windows.csv", index=False, encoding="utf-8-sig")

    top_models = summary.head(20)["model"].tolist()
    for metric, filename in [
        ("final_return", "window_return_matrix.csv"),
        ("sharpe", "window_sharpe_matrix.csv"),
        ("max_drawdown", "window_drawdown_matrix.csv"),
    ]:
        matrix = complete_all.pivot_table(index="window_id", columns="model", values=metric, aggfunc="first")
        matrix = matrix.reset_index()
        cols = ["window_id", *[m for m in top_models if m in matrix.columns]]
        matrix[cols].to_csv(args.out_dir / filename, index=False, encoding="utf-8-sig")

    segmented._write_report(args.out_dir, summary, complete_windows, tail_windows, complete_all, tail_all)
    print(f"Combined ranking written to {args.out_dir}")
    print(
        summary.head(30)[
            [
                "model",
                "mean_10d_return",
                "median_10d_return",
                "p25_10d_return",
                "return_std",
                "positive_window_rate",
                "mean_10d_sharpe",
                "worst_10d_drawdown",
                "score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
