from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from train import TrainConfig, train


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: dict | list) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min()) if len(nav) else float("nan")


def _interval_stats(path: Path) -> dict:
    df = pd.read_csv(path)
    returns = []
    sharpes = []
    drawdowns = []
    for start in range(0, len(df), 10):
        block = df.iloc[start : start + 10]
        if block.empty:
            continue
        daily = block["net_ret"].to_numpy(dtype=float)
        returns.append(float(np.prod(1.0 + daily) - 1.0))
        sharpes.append(float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)))
        drawdowns.append(_max_drawdown(block["nav"]))
    arr = np.array(returns, dtype=float)
    return {
        "interval_count": int(len(arr)),
        "interval_return_mean": float(np.nanmean(arr)) if len(arr) else float("nan"),
        "interval_return_std": float(np.nanstd(arr)) if len(arr) else float("nan"),
        "interval_return_min": float(np.nanmin(arr)) if len(arr) else float("nan"),
        "interval_positive_rate": float((arr > 0).mean()) if len(arr) else float("nan"),
        "interval_sharpe_mean": float(np.nanmean(sharpes)) if sharpes else float("nan"),
        "interval_max_drawdown_min": float(np.nanmin(drawdowns)) if drawdowns else float("nan"),
    }


def _score(row: dict) -> float:
    return float(
        row["sharpe"]
        + 2.0 * row["final_return"]
        + 0.8 * row["interval_return_mean"]
        - 1.2 * row["interval_return_std"]
        + 0.5 * row["max_drawdown"]
    )


def _backtest(
    args: argparse.Namespace,
    checkpoint: Path,
    out_dir: Path,
    name: str,
    strategy: str,
    rebalance_interval: int,
    stop_loss: float,
) -> dict:
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
            args.bt_start,
            "--end",
            args.bt_end,
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
    metrics.update(_interval_stats(out_csv))
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "backtest_csv": str(out_csv),
            "holdings_csv": str(holdings_csv),
            "checkpoint_name": checkpoint.stem,
            "rebalance_interval": rebalance_interval,
            "stop_loss": stop_loss,
        }
    )
    metrics["score"] = _score(metrics)
    return metrics


def _train_variant(args: argparse.Namespace, root: Path, spec: dict) -> Path:
    out_dir = root / spec["name"]
    cfg = TrainConfig(
        panel=str(args.panel),
        seq_len=args.seq_len,
        label_horizon=spec["label_horizon"],
        model=spec["model"],
        split_mode="recent_train_past_test",
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.val_start,
        test_end=args.val_end,
        epochs=spec["epochs"],
        batch_size=args.batch_size,
        lr=spec.get("lr", args.lr),
        top_n=args.top_n,
        shuffle_train=False,
        seed=args.seed,
        d_model=spec.get("d_model", args.d_model),
        nhead=spec.get("nhead", args.nhead),
        num_layers=spec.get("num_layers", args.num_layers),
        dropout=spec.get("dropout", args.dropout),
        weight_decay=spec.get("weight_decay", args.weight_decay),
        device=args.device,
        out_dir=str(out_dir),
    )
    print(f"Training sweep variant {spec['name']}: {asdict(cfg)}")
    train(cfg)
    return out_dir


def _write_report(root: Path, ranked: pd.DataFrame) -> None:
    best = ranked.iloc[0].to_dict()
    top = ranked.head(15)[
        [
            "variant",
            "checkpoint_name",
            "strategy",
            "rebalance_interval",
            "stop_loss",
            "final_nav",
            "final_return",
            "sharpe",
            "max_drawdown",
            "interval_return_mean",
            "interval_return_std",
            "score",
        ]
    ].copy()
    top.to_csv(root / "top15.csv", index=False, encoding="utf-8-sig")
    table_lines = ["| " + " | ".join(top.columns) + " |", "| " + " | ".join(["---"] * len(top.columns)) + " |"]
    for _, row in top.iterrows():
        vals = []
        for v in row.tolist():
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        table_lines.append("| " + " | ".join(vals) + " |")

    lines = [
        "# 扩展调参实验报告",
        "",
        "本轮实验增加 epoch，并同时考察 checkpoint 选择、标签周期、模型结构和改进策略参数。排序指标为综合分数：Sharpe、总收益、10 日区间平均收益、10 日区间波动、最大回撤共同参与。",
        "",
        "## 最优结果",
        "",
        f"- variant: `{best['variant']}`",
        f"- checkpoint: `{best['checkpoint_name']}`",
        f"- strategy: `{best['strategy']}`",
        f"- rebalance_interval: `{int(best['rebalance_interval'])}`",
        f"- stop_loss: `{best['stop_loss']:.4f}`",
        f"- final_nav: `{best['final_nav']:.4f}`",
        f"- final_return: `{best['final_return']:.4f}`",
        f"- Sharpe: `{best['sharpe']:.4f}`",
        f"- max_drawdown: `{best['max_drawdown']:.4f}`",
        f"- 10日区间平均收益: `{best['interval_return_mean']:.4f}`",
        f"- 10日区间收益标准差: `{best['interval_return_std']:.4f}`",
        "",
        "## Top 15",
        "",
        *table_lines,
        "",
        "## 关于单股票单模型",
        "",
        "本轮不采用“一只股票一个模型”作为主方案。原因是沪深300每只股票只有约 2400 个时间样本，单股票模型容易过拟合，且无法直接学习横截面排序关系；当前作业目标是每日在股票池内排序并建组合，共享参数模型更适合用 299 只股票的横截面信息提升样本效率。若未来要做单股票模型，更适合作为二阶段残差模型或行业内微调，而不是替换主排序模型。",
        "",
        "完整排名见 `sweep_results.csv`。",
    ]
    (root / "sweep_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=PROJECT_ROOT / "data" / "panel_hs300_advanced.npz")
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument("--epochs_scale", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
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
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                args.device = "cpu"
        except Exception:
            args.device = "cpu"

    root = args.out_root or (PROJECT_ROOT / "experiments" / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=True)
    _dump_json(root / "sweep_config.json", vars(args) | {"out_root": str(root), "panel": str(args.panel)})

    def ep(n: int) -> int:
        return max(1, int(round(n * args.epochs_scale)))

    variants = [
        {"name": "transformer_e60", "model": "transformer", "label_horizon": 1, "epochs": ep(60)},
        {"name": "transformer_d96_e50", "model": "transformer", "label_horizon": 1, "epochs": ep(50), "d_model": 96, "dropout": 0.15},
        {"name": "gru_e60", "model": "gru", "label_horizon": 1, "epochs": ep(60)},
        {"name": "lstm_e60", "model": "lstm", "label_horizon": 1, "epochs": ep(60)},
        {"name": "improved_h3_e50", "model": "transformer", "label_horizon": 3, "epochs": ep(50), "dropout": 0.15},
        {"name": "improved_h5_e60", "model": "transformer", "label_horizon": 5, "epochs": ep(60), "dropout": 0.15},
        {"name": "improved_h10_e50", "model": "transformer", "label_horizon": 10, "epochs": ep(50), "dropout": 0.2},
    ]

    rows: list[dict] = []
    for spec in variants:
        variant_dir = _train_variant(args, root, spec)
        checkpoints = [variant_dir / "best.pt", variant_dir / "best_sharpe.pt", variant_dir / "last.pt"]
        strategy = "improved" if spec["name"].startswith("improved") else "baseline"
        strategy_grid = [(1, -0.035)] if strategy == "baseline" else [
            (3, -0.025),
            (3, -0.035),
            (5, -0.025),
            (5, -0.035),
            (10, -0.035),
            (10, -0.050),
        ]
        for ckpt in checkpoints:
            for interval, stop_loss in strategy_grid:
                name = f"bt_{ckpt.stem}_i{interval}_sl{str(stop_loss).replace('-', 'm').replace('.', 'p')}"
                metrics = _backtest(args, ckpt, variant_dir, name, strategy, interval, stop_loss)
                metrics.update(spec)
                metrics["variant"] = spec["name"]
                rows.append(metrics)
                pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(
                    root / "sweep_results_partial.csv", index=False, encoding="utf-8-sig"
                )

    ranked = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    ranked.to_csv(root / "sweep_results.csv", index=False, encoding="utf-8-sig")
    _write_report(root, ranked)
    print(f"Sweep complete: {root}")
    print(ranked.head(10)[["variant", "checkpoint_name", "strategy", "final_nav", "sharpe", "max_drawdown", "score"]])


if __name__ == "__main__":
    main()
