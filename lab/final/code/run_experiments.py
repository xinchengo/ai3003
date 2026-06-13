from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import TrainConfig, StockPanelDataset, train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PROJECT_ROOT / "data" / "panel_hs300_advanced.npz"
RAW_DIR = PROJECT_ROOT / "A股数据"


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, obj: dict | list) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "暂无数据"
    table = df.copy()
    table = table.reset_index()
    cols = [str(c) for c in table.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in table.iterrows():
        vals = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                vals.append(format(float(value), floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _panel_summary(panel: Path) -> dict:
    data = np.load(panel, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    ds = StockPanelDataset(str(panel), seq_len=30)
    realized_dates = pd.to_datetime(ds.dates[ds.end_indices])
    return {
        "panel": str(panel.relative_to(PROJECT_ROOT)),
        "start_date": str(dates.min().date()),
        "end_date": str(dates.max().date()),
        "realized_end_date": str(realized_dates.max().date()),
        "num_dates": int(len(dates)),
        "num_stocks": int(len(data["stocks"])),
        "num_features": int(len(data["feature_cols"])),
        "feature_cols": data["feature_cols"].astype(str).tolist(),
    }


def _maybe_refresh_panel(panel: Path, mode: str) -> None:
    if mode == "never":
        return
    raw_dirs = [RAW_DIR / "daily", RAW_DIR / "metric", RAW_DIR / "moneyflow"]
    if not all(p.exists() for p in raw_dirs):
        return
    if mode == "always" or not panel.exists():
        needs_refresh = True
    else:
        latest_raw = max(p.stat().st_mtime for p in raw_dirs)
        needs_refresh = latest_raw > panel.stat().st_mtime
    if not needs_refresh:
        return
    _run(
        [
            sys.executable,
            "code/load_data.py",
            "--data_dir",
            str(RAW_DIR / "daily"),
            "--metric_dir",
            str(RAW_DIR / "metric"),
            "--moneyflow_dir",
            str(RAW_DIR / "moneyflow"),
            "--pool",
            "hs300",
            "--stock_pool_file",
            "code/hs300_constituents.csv",
            "--top_k",
            "300",
            "--out",
            str(panel),
        ]
    )


def _plot_history(history_path: Path, out_path: Path) -> None:
    hist = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train_loss")
    if "val_loss" in hist:
        axes[0].plot(hist["epoch"], hist["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(hist["epoch"], hist["ic_mean"], label="IC")
    if "rank_ic_mean" in hist:
        axes[1].plot(hist["epoch"], hist["rank_ic_mean"], label="Rank IC")
    axes[1].set_title("Validation IC")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_nav(paths: dict[str, Path], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    for name, path in paths.items():
        df = pd.read_csv(path)
        ax.plot(pd.to_datetime(df["date"]), df["nav"], label=name)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min()) if len(nav) else math.nan


def _interval_stats(backtest_csvs: dict[str, Path], out_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for model, path in backtest_csvs.items():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        for interval_id, start in enumerate(range(0, len(df), 10), start=1):
            block = df.iloc[start : start + 10].copy()
            if block.empty:
                continue
            daily = block["net_ret"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "interval": interval_id,
                    "start": str(block["date"].iloc[0].date()),
                    "end": str(block["date"].iloc[-1].date()),
                    "days": int(len(block)),
                    "return": float(np.prod(1.0 + daily) - 1.0),
                    "mean_daily_return": float(daily.mean()),
                    "ann_vol": float(daily.std() * np.sqrt(252)),
                    "sharpe": float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)),
                    "max_drawdown": _max_drawdown(block["nav"]),
                    "win_rate": float((daily > 0).mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def _train_one(args: argparse.Namespace, out_root: Path, model: str, label_horizon: int, tag: str) -> Path:
    out_dir = out_root / tag
    cfg = TrainConfig(
        panel=str(args.panel),
        seq_len=args.seq_len,
        label_horizon=label_horizon,
        model=model,
        split_mode="recent_train_past_test",
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.val_start,
        test_end=args.val_end,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        top_n=args.top_n,
        shuffle_train=False,
        seed=args.seed,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        device=args.device,
        out_dir=str(out_dir),
    )
    print(f"Training {tag}: {asdict(cfg)}")
    train(cfg)
    _plot_history(out_dir / "history.csv", out_dir / "loss_ic_curve.png")
    return out_dir / "best.pt"


def _backtest_one(
    args: argparse.Namespace,
    checkpoint: Path,
    out_dir: Path,
    name: str,
    start: str,
    end: str,
    strategy: str = "baseline",
    rebalance_interval: int = 1,
) -> tuple[Path, Path, Path]:
    out_csv = out_dir / f"{name}.csv"
    holdings_csv = out_dir / f"{name}_holdings.csv"
    metrics_json = out_dir / f"{name}_metrics.json"
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
            start,
            "--end",
            end,
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
    return out_csv, holdings_csv, metrics_json


def _write_report(
    out_root: Path,
    panel_info: dict,
    train_dirs: dict[str, Path],
    metrics: dict[str, dict],
    interval_stats: pd.DataFrame,
    recent_csv: Path,
    live_csv: Path,
) -> None:
    metric_df = pd.DataFrame.from_dict(metrics, orient="index")
    metric_df.index.name = "experiment"
    metric_df.to_csv(out_root / "summary_metrics.csv", encoding="utf-8-sig")
    baseline_models = [m for m in ["transformer", "gru", "lstm"] if m in metrics]
    best_model = max(baseline_models, key=lambda m: (metrics[m]["sharpe"], -abs(metrics[m]["max_drawdown"])))
    stable_rows = []
    for model, g in interval_stats.groupby("model"):
        rets = g["return"].to_numpy(dtype=float)
        stable_rows.append((model, float(rets.std()), int((rets < 0).sum()), float(rets.mean())))
    stable_rows = sorted(stable_rows, key=lambda x: (x[1], x[2], -x[3]))
    stable_model = stable_rows[0][0] if stable_rows else best_model
    all_best = max(metrics, key=lambda m: (metrics[m]["sharpe"], -abs(metrics[m]["max_drawdown"])))

    recent = pd.read_csv(recent_csv)
    recent_metrics = _json_load(recent_csv.with_name(recent_csv.stem + "_metrics.json"))
    first_holdings = recent["holdings"].iloc[0] if len(recent) else ""
    recent_table = _md_table(recent[["date", "net_ret", "nav", "bought", "sold"]]) if len(recent) else "暂无数据"
    metric_table = _md_table(
        metric_df[["final_nav", "final_return", "ann_ret", "ann_vol", "sharpe", "max_drawdown", "win_rate", "avg_turnover"]]
    )
    interval_summary = (
        _md_table(interval_stats.groupby("model")["return"].agg(["mean", "std", "min", "max"]))
        if len(interval_stats)
        else "暂无分段结果"
    )

    report = f"""# 深度学习基础大作业量化实验报告

## 1. 数据与实验设置

- 股票池：仅使用 `code/hs300_constituents.csv` 指定的沪深300股票池；当前面板实际包含 {panel_info['num_stocks']} 只股票。
- 面板区间：{panel_info['start_date']} 至 {panel_info['end_date']}，可完整回测的最新决策日为 {panel_info['realized_end_date']}。
- 特征数：{panel_info['num_features']}，包含量价、估值、换手、资金流等横截面标准化特征。
- 基线标签：1 日 open-to-open 收益；改进模型标签：5 日 open-to-open 复合收益。
- 切分：训练 {train_dirs['settings']['train_start']} 至 {train_dirs['settings']['train_end']}，验证 {train_dirs['settings']['val_start']} 至 {train_dirs['settings']['val_end']}，样本外回测 {train_dirs['settings']['bt_start']} 至 {train_dirs['settings']['bt_end']}。
- 交易策略：首日 Top 10 等权建仓，之后每日 Sell 3 / Buy 3，单边手续费 {train_dirs['settings']['fee']:.4f}。

## 2. Baseline 与模型对比结果

{metric_table}

- 最佳基线模型：{best_model}，主要依据样本外 Sharpe，其次参考最大回撤。
- 最稳定基线模型：{stable_model}，依据 10 日区间收益波动和负收益区间数量。
- 全部实验最优：{all_best}。

关键图表：
- `nav_compare.png`：Transformer / GRU / LSTM 样本外 NAV 对比。
- `improved_nav_compare.png`：Baseline Transformer 与改进策略对比。
- 各模型目录下的 `loss_ic_curve.png`：训练损失、验证损失、IC、Rank IC 曲线。

## 3. 最近10个交易日回测

- 决策日：{recent['date'].iloc[0] if len(recent) else 'NA'} 至 {recent['date'].iloc[-1] if len(recent) else 'NA'}。
- 初始建仓：{first_holdings}
- 最终净值：{recent['nav'].iloc[-1] if len(recent) else float('nan'):.4f}
- 最终收益率：{recent['nav'].iloc[-1] - 1.0 if len(recent) else float('nan'):.4f}
- Sharpe Ratio：{recent_metrics.get('sharpe', float('nan')):.4f}
- 最大回撤：{recent_metrics.get('max_drawdown', float('nan')):.4f}

每日收益与买卖记录摘要：

{recent_table}

完整记录见：
- `{recent_csv.relative_to(out_root)}`
- `{recent_csv.with_name(recent_csv.stem + '_holdings.csv').relative_to(out_root)}`

## 4. 分段回测结果

每 10 个交易日划分一个区间，统计结果保存在 `interval_returns_10d.csv`。摘要如下：

{interval_summary}

## 5. 风险与策略分析

原模型偏好近期大跌股票，主要原因包括：单日收益标签噪声高，短期反转样本在横截面 IC 目标中容易被放大；特征中趋势确认不足，模型可能把低位偏离误判为反弹信号；每日强制 Sell 3 / Buy 3 使组合在弱趋势市场中承担较高换手成本；训练目标优化的是排序相关性，不完全等同于组合 Sharpe 和回撤控制。

改进实验采用 5 日标签、趋势过滤、波动惩罚、资金流修正、止损替换和 5 日主调仓。该方案的目标不是保证每个 10 日区间盈利，而是提高 Sharpe、降低换手和最大回撤，并减少对单日反转噪声的依赖。

## 6. 最终结论

- 最优模型：{all_best}
- 最优策略：{'改进策略' if all_best == 'improved_transformer' else '基线 Buy3/Sell3 策略'}
- 最优参数组合：seq_len={train_dirs['settings']['seq_len']}，top_n=10，k=3，fee={train_dirs['settings']['fee']:.4f}，baseline label_horizon=1，improved label_horizon=5。
- 最新交易日预测持仓见 `{live_csv.relative_to(out_root)}`，其特征截止日为面板最新日期。

## 7. 组员信息

- 姓名/学号/分工：待补充。
"""
    (out_root / "experiment_report.md").write_text(report, encoding="utf-8")
    _json_dump(
        out_root / "final_selection.json",
        {"best_baseline_model": best_model, "most_stable_baseline_model": stable_model, "best_overall": all_best},
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument("--refresh_panel", choices=["auto", "always", "never"], default="never")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--train_start", default="2016-01-04")
    p.add_argument("--train_end", default="2025-12-31")
    p.add_argument("--val_start", default="2026-01-01")
    p.add_argument("--val_end", default="2026-02-27")
    p.add_argument("--bt_start", default="2026-03-02")
    p.add_argument("--bt_end", default="2026-05-27")
    p.add_argument("--recent_start", default="2026-05-14")
    p.add_argument("--recent_end", default="2026-05-27")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                args.device = "cpu"
        except Exception:
            args.device = "cpu"

    _maybe_refresh_panel(args.panel, args.refresh_panel)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or (PROJECT_ROOT / "experiments" / f"hs300_{timestamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    panel_info = _panel_summary(args.panel)
    _json_dump(out_root / "panel_summary.json", panel_info)
    settings = vars(args).copy()
    settings["panel"] = str(args.panel)
    settings["out_root"] = str(out_root)
    _json_dump(out_root / "experiment_config.json", settings)

    ckpts: dict[str, Path] = {}
    for model in ["transformer", "gru", "lstm"]:
        ckpts[model] = _train_one(args, out_root, model=model, label_horizon=1, tag=model)
    ckpts["improved_transformer"] = _train_one(
        args, out_root, model="transformer", label_horizon=5, tag="improved_transformer"
    )

    backtests: dict[str, Path] = {}
    metrics: dict[str, dict] = {}
    for model in ["transformer", "gru", "lstm"]:
        csv, _, met = _backtest_one(args, ckpts[model], out_root / model, "backtest_oos", args.bt_start, args.bt_end)
        backtests[model] = csv
        metrics[model] = _json_load(met)
        _plot_nav({model: csv}, csv.with_name("nav_curve.png"), f"{model} NAV")

    improved_csv, _, improved_met = _backtest_one(
        args,
        ckpts["improved_transformer"],
        out_root / "improved_transformer",
        "backtest_oos",
        args.bt_start,
        args.bt_end,
        strategy="improved",
        rebalance_interval=5,
    )
    backtests["improved_transformer"] = improved_csv
    metrics["improved_transformer"] = _json_load(improved_met)
    _plot_nav({"improved_transformer": improved_csv}, improved_csv.with_name("nav_curve.png"), "Improved Transformer NAV")

    recent_csv, _, _ = _backtest_one(
        args,
        ckpts["transformer"],
        out_root / "transformer",
        "backtest_recent10",
        args.recent_start,
        args.recent_end,
    )

    _plot_nav({m: backtests[m] for m in ["transformer", "gru", "lstm"]}, out_root / "nav_compare.png", "Model NAV Comparison")
    _plot_nav(
        {"baseline_transformer": backtests["transformer"], "improved_transformer": improved_csv},
        out_root / "improved_nav_compare.png",
        "Baseline vs Improved",
    )

    interval_stats = _interval_stats({m: backtests[m] for m in ["transformer", "gru", "lstm"]}, out_root / "interval_returns_10d.csv")

    live_csv = out_root / "latest_weights_transformer.csv"
    _run(
        [
            sys.executable,
            "code/predict_weights.py",
            "--panel",
            str(args.panel),
            "--checkpoint",
            str(ckpts["transformer"]),
            "--seq_len",
            str(args.seq_len),
            "--top_n",
            str(args.top_n),
            "--equal_weight",
            "--out",
            str(live_csv),
            "--device",
            args.device,
        ]
    )

    train_dirs = {
        "transformer": out_root / "transformer",
        "gru": out_root / "gru",
        "lstm": out_root / "lstm",
        "improved_transformer": out_root / "improved_transformer",
        "settings": {
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_start": args.val_start,
            "val_end": args.val_end,
            "bt_start": args.bt_start,
            "bt_end": args.bt_end,
            "seq_len": args.seq_len,
            "fee": args.fee,
        },
    }
    _write_report(out_root, panel_info, train_dirs, metrics, interval_stats, recent_csv, live_csv)
    print(f"Experiment complete: {out_root}")


if __name__ == "__main__":
    main()
