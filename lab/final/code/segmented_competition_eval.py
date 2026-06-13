from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from train import StockPanelDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: dict | list) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize(text: str) -> str:
    return (
        text.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def _model_name(checkpoint: Path, base_root: Path) -> str:
    try:
        rel = checkpoint.relative_to(base_root)
        return _sanitize(str(rel.with_suffix("")))
    except ValueError:
        return _sanitize(str(checkpoint.with_suffix("")))


def _read_ckpt_config(checkpoint: Path, device: str) -> dict:
    import torch

    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    return ckpt.get("config", {})


def _infer_strategy(checkpoint: Path, cfg: dict) -> tuple[str, list[tuple[int, float]]]:
    text = str(checkpoint).lower()
    label_horizon = int(cfg.get("label_horizon", 1))
    if "improved" in text or label_horizon > 1:
        return "improved", [(3, -0.025), (5, -0.035), (10, -0.035)]
    return "baseline", [(1, -0.035)]


def _collect_checkpoints(roots: list[Path], checkpoint_names: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".pt":
            candidates = [root]
        else:
            candidates = []
            for name in checkpoint_names:
                candidates.extend(root.rglob(name))
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return sorted(out, key=lambda p: str(p))


def _windows(panel: Path, seq_len: int, start: str, end: str, window_size: int, include_partial: bool) -> list[dict]:
    ds = StockPanelDataset(str(panel), seq_len=seq_len)
    dates = pd.to_datetime(ds.dates[ds.end_indices])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    selected = pd.DatetimeIndex(dates[mask])
    windows = []
    for i, offset in enumerate(range(0, len(selected), window_size), start=1):
        block = selected[offset : offset + window_size]
        if len(block) < window_size and not include_partial:
            continue
        windows.append(
            {
                "window_id": i,
                "start": str(block[0].date()),
                "end": str(block[-1].date()),
                "days": int(len(block)),
            }
        )
    return windows


def _rank_pct(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = s.astype(float)
    if not higher_is_better:
        values = -values
    return values.rank(method="average", pct=True)


def _add_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank score focused on repeated independent 10-day competitions."""
    out = summary.copy()
    out["score_mean_return"] = _rank_pct(out["mean_10d_return"])
    out["score_median_return"] = _rank_pct(out["median_10d_return"])
    out["score_p25_return"] = _rank_pct(out["p25_10d_return"])
    out["score_mean_sharpe"] = _rank_pct(out["mean_10d_sharpe"])
    out["score_positive_rate"] = _rank_pct(out["positive_window_rate"])
    out["score_low_return_std"] = _rank_pct(out["return_std"], higher_is_better=False)
    out["score_worst_drawdown"] = _rank_pct(out["worst_10d_drawdown"])
    out["score"] = (
        0.25 * out["score_mean_return"]
        + 0.20 * out["score_median_return"]
        + 0.15 * out["score_p25_return"]
        + 0.15 * out["score_mean_sharpe"]
        + 0.10 * out["score_positive_rate"]
        + 0.10 * out["score_low_return_std"]
        + 0.05 * out["score_worst_drawdown"]
    )
    return out.sort_values(
        ["score", "median_10d_return", "p25_10d_return", "worst_10d_drawdown"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无数据"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(
    out_root: Path,
    summary: pd.DataFrame,
    complete_windows_df: pd.DataFrame,
    tail_windows_df: pd.DataFrame,
    complete_detail: pd.DataFrame,
    tail_detail: pd.DataFrame,
) -> None:
    best = summary.iloc[0].to_dict()
    best_return = summary.sort_values(["mean_10d_return", "median_10d_return"], ascending=False).iloc[0].to_dict()
    most_stable = summary.sort_values(["return_std", "p25_10d_return"], ascending=[True, False]).iloc[0].to_dict()
    top = summary.head(20)[
        [
            "model",
            "strategy",
            "rebalance_interval",
            "checkpoint_name",
            "mean_10d_return",
            "median_10d_return",
            "p25_10d_return",
            "return_std",
            "mean_10d_sharpe",
            "worst_10d_drawdown",
            "positive_window_rate",
            "score",
        ]
    ]
    top_models = summary.head(8)["model"].tolist()

    def matrix(metric: str) -> pd.DataFrame:
        pivot = complete_detail.pivot_table(index="window_id", columns="model", values=metric, aggfunc="first")
        pivot = pivot.reset_index()
        pivot["window_id"] = pivot["window_id"].astype(int)
        if len(pivot.columns) > 9:
            pivot = pivot[["window_id", *top_models]]
        return pivot

    return_matrix = matrix("final_return")
    sharpe_matrix = matrix("sharpe")
    drawdown_matrix = matrix("max_drawdown")
    tail_top = tail_detail[tail_detail["model"].isin(top_models)].copy()
    if not tail_top.empty:
        tail_top = tail_top[
            ["model", "window_id", "window_start", "window_end", "days", "final_return", "sharpe", "max_drawdown"]
        ].sort_values(["window_id", "final_return"], ascending=[True, False])

    report = f"""# 独立10日比赛窗口回测报告

## 方法

- 每个窗口独立回测，不继承上一窗口持仓或 NAV。
- 每个窗口首日重新按预测 Top 10 等权建仓，随后在窗口内执行 Buy 3 / Sell 3 或对应改进策略。
- 主排名只使用完整 10 日窗口；不足 10 日的尾段只作为压力测试，不参与模型选择。
- 综合评分使用各指标横截面分位排名，避免单个 Sharpe 或尾段收益支配结论。

## 完整10日窗口

{_md_table(complete_windows_df)}

## 尾段压力测试窗口

{_md_table(tail_windows_df)}

## 最终推荐比赛模型

- 模型：`{best['model']}`
- checkpoint：`{best['checkpoint']}`
- 策略：`{best['strategy']}`
- 调仓间隔：`{int(best['rebalance_interval'])}`
- 平均10日收益：`{best['mean_10d_return']:.4f}`
- 10日收益中位数：`{best['median_10d_return']:.4f}`
- 25分位10日收益：`{best['p25_10d_return']:.4f}`
- 平均10日 Sharpe：`{best['mean_10d_sharpe']:.4f}`
- 最差10日窗口回撤：`{best['worst_10d_drawdown']:.4f}`
- 正收益窗口占比：`{best['positive_window_rate']:.4f}`

## 分类结论

- 收益最高模型：`{best_return['model']}`，平均10日收益 `{best_return['mean_10d_return']:.4f}`。
- 稳定性最好模型：`{most_stable['model']}`，10日收益标准差 `{most_stable['return_std']:.4f}`。
- 最终推荐模型：`{best['model']}`，综合考虑收益、Sharpe、低波动、尾部收益和回撤。

## 综合排名 Top 20

{_md_table(top)}

## 完整窗口收益矩阵

{_md_table(return_matrix)}

## 完整窗口 Sharpe 矩阵

{_md_table(sharpe_matrix)}

## 完整窗口最大回撤矩阵

{_md_table(drawdown_matrix)}

## 尾段压力测试摘要

{_md_table(tail_top)}

完整明细见：

- `segmented_window_results.csv`
- `segmented_tail_check.csv`
- `segmented_model_summary.csv`
- `window_return_matrix.csv`
- `window_sharpe_matrix.csv`
- `window_drawdown_matrix.csv`
"""
    (out_root / "segmented_competition_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=PROJECT_ROOT / "data" / "panel_hs300_advanced.npz")
    p.add_argument("--roots", nargs="+", type=Path, required=True, help="Experiment roots or checkpoint files")
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument("--checkpoint_names", nargs="+", default=["best.pt", "best_sharpe.pt", "last.pt"])
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--start", default="2026-03-02")
    p.add_argument("--end", default="2026-05-27")
    p.add_argument("--window_size", type=int, default=10)
    p.add_argument("--include_partial", action="store_true")
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                args.device = "cpu"
        except Exception:
            args.device = "cpu"

    out_root = args.out_root or (PROJECT_ROOT / "experiments" / f"segmented_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoints = _collect_checkpoints(args.roots, args.checkpoint_names)
    if not checkpoints:
        raise ValueError("No checkpoint files found.")
    windows = _windows(args.panel, args.seq_len, args.start, args.end, args.window_size, args.include_partial)
    if not windows:
        raise ValueError("No evaluation windows found.")
    windows_df = pd.DataFrame(windows)
    windows_df["is_complete_window"] = windows_df["days"] == args.window_size
    complete_windows_df = windows_df[windows_df["is_complete_window"]].copy()
    tail_windows_df = windows_df[~windows_df["is_complete_window"]].copy()
    if complete_windows_df.empty:
        raise ValueError("No complete 10-day windows found for main ranking.")
    windows_df.to_csv(out_root / "windows.csv", index=False, encoding="utf-8-sig")
    complete_windows_df.to_csv(out_root / "complete_windows.csv", index=False, encoding="utf-8-sig")
    tail_windows_df.to_csv(out_root / "tail_windows.csv", index=False, encoding="utf-8-sig")
    _dump_json(
        out_root / "segmented_eval_config.json",
        {
            "panel": str(args.panel),
            "roots": [str(r) for r in args.roots],
            "out_root": str(out_root),
            "checkpoint_names": args.checkpoint_names,
            "seq_len": args.seq_len,
            "start": args.start,
            "end": args.end,
            "window_size": args.window_size,
            "include_partial": args.include_partial,
            "top_n": args.top_n,
            "k": args.k,
            "fee": args.fee,
            "device": args.device,
            "checkpoints": [str(c) for c in checkpoints],
            "windows": windows,
        },
    )

    detail_rows: list[dict] = []
    for checkpoint in checkpoints:
        cfg = _read_ckpt_config(checkpoint, args.device)
        strategy, strategy_grid = _infer_strategy(checkpoint, cfg)
        base_root = next(
            (
                root
                for root in args.roots
                if root.exists() and root.is_dir() and checkpoint.resolve().is_relative_to(root.resolve())
            ),
            args.roots[0],
        )
        model_base = _model_name(checkpoint, base_root)
        for rebalance_interval, stop_loss in strategy_grid:
            model_name = f"{model_base}__{strategy}__i{rebalance_interval}__sl{str(stop_loss).replace('-', 'm').replace('.', 'p')}"
            model_dir = out_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            for window in windows:
                out_csv = model_dir / f"window_{window['window_id']:02d}.csv"
                holdings_csv = model_dir / f"window_{window['window_id']:02d}_holdings.csv"
                metrics_json = model_dir / f"window_{window['window_id']:02d}_metrics.json"
                _run(
                    [
                        sys.executable,
                        "code/back_test_equal.py",
                        "--panel",
                        str(args.panel),
                        "--checkpoint",
                        str(checkpoint),
                        "--seq_len",
                        str(args.seq_len),
                        "--start",
                        window["start"],
                        "--end",
                        window["end"],
                        "--top_n",
                        str(args.top_n),
                        "--k",
                        str(args.k),
                        "--fee",
                        str(args.fee),
                        "--strategy",
                        strategy,
                        "--rebalance_interval",
                        str(rebalance_interval),
                        "--stop_loss",
                        str(stop_loss),
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
                metrics = _load_json(metrics_json)
                detail_rows.append(
                    {
                        "model": model_name,
                        "checkpoint": str(checkpoint),
                        "checkpoint_name": checkpoint.stem,
                        "model_type": cfg.get("model", "unknown"),
                        "label_horizon": int(cfg.get("label_horizon", 1)),
                        "strategy": strategy,
                        "rebalance_interval": rebalance_interval,
                        "stop_loss": stop_loss,
                        "window_id": window["window_id"],
                        "window_start": window["start"],
                        "window_end": window["end"],
                        "days": window["days"],
                        "is_complete_window": window["days"] == args.window_size,
                        **metrics,
                        "daily_csv": str(out_csv),
                        "holdings_csv": str(holdings_csv),
                    }
                )
                pd.DataFrame(detail_rows).to_csv(out_root / "segmented_window_results_partial.csv", index=False, encoding="utf-8-sig")

    detail = pd.DataFrame(detail_rows)
    complete_detail = detail[detail["is_complete_window"]].copy()
    tail_detail = detail[~detail["is_complete_window"]].copy()
    summaries: list[dict] = []
    for model, group in complete_detail.groupby("model"):
        returns = group["final_return"].to_numpy(dtype=float)
        sharpes = group["sharpe"].to_numpy(dtype=float)
        drawdowns = group["max_drawdown"].to_numpy(dtype=float)
        row0 = group.iloc[0].to_dict()
        summary = {
            "model": model,
            "checkpoint": row0["checkpoint"],
            "checkpoint_name": row0["checkpoint_name"],
            "model_type": row0["model_type"],
            "label_horizon": row0["label_horizon"],
            "strategy": row0["strategy"],
            "rebalance_interval": row0["rebalance_interval"],
            "stop_loss": row0["stop_loss"],
            "window_count": int(len(group)),
            "mean_10d_return": float(np.nanmean(returns)),
            "median_10d_return": float(np.nanmedian(returns)),
            "p25_10d_return": float(np.nanpercentile(returns, 25)),
            "reference_compounded_return": float(np.prod(1.0 + returns) - 1.0),
            "return_std": float(np.nanstd(returns)),
            "best_return": float(np.nanmax(returns)),
            "worst_return": float(np.nanmin(returns)),
            "positive_window_rate": float((returns > 0).mean()),
            "mean_10d_sharpe": float(np.nanmean(sharpes)),
            "median_10d_sharpe": float(np.nanmedian(sharpes)),
            "worst_10d_drawdown": float(np.nanmin(drawdowns)),
            "mean_drawdown": float(np.nanmean(drawdowns)),
        }
        summaries.append(summary)

    summary_df = _add_scores(pd.DataFrame(summaries))
    complete_detail.to_csv(out_root / "segmented_window_results.csv", index=False, encoding="utf-8-sig")
    complete_detail.to_csv(out_root / "complete_window_results.csv", index=False, encoding="utf-8-sig")
    tail_detail.to_csv(out_root / "segmented_tail_check.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_root / "segmented_model_summary.csv", index=False, encoding="utf-8-sig")

    top_models = summary_df.head(20)["model"].tolist()
    for metric, filename in [
        ("final_return", "window_return_matrix.csv"),
        ("sharpe", "window_sharpe_matrix.csv"),
        ("max_drawdown", "window_drawdown_matrix.csv"),
    ]:
        matrix = complete_detail.pivot_table(index="window_id", columns="model", values=metric, aggfunc="first")
        matrix = matrix.reset_index()
        cols = ["window_id", *[m for m in top_models if m in matrix.columns]]
        matrix = matrix[cols]
        matrix.to_csv(out_root / filename, index=False, encoding="utf-8-sig")

    _write_report(out_root, summary_df, complete_windows_df, tail_windows_df, complete_detail, tail_detail)
    print(f"Segmented competition evaluation complete: {out_root}")
    print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
