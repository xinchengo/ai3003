from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from losses import scores_to_weights
from predict_weights import rebuild_model
from train import StockPanelDataset


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--start", type=str, default="2025-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--top_n", type=int, default=30)
    p.add_argument("--fee", type=float, default=0.001, help="one-way turnover cost, e.g. 0.001")
    p.add_argument("--equal_weight", action="store_true")
    p.add_argument("--out", type=str, default="back_test.csv")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ds = StockPanelDataset(args.panel, seq_len=args.seq_len)
    dates = ds.dates[ds.end_indices]
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end) if args.end else pd.Timestamp.max
    idx = np.where((dates >= start) & (dates <= end))[0].tolist()
    if not idx:
        raise ValueError("No back_test samples in requested date range.")

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = rebuild_model(ckpt, num_features=len(ds.feature_cols)).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prev_w = None
    rows = []
    loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False)
    nav = 1.0
    for x, y, m, date_str in loader:
        x, y, m = x.to(args.device), y.to(args.device), m.to(args.device)
        scores = model(x)
        w = scores_to_weights(scores, m, top_n=args.top_n, equal_weight=args.equal_weight)
        gross_ret = float((w * y * m).sum().cpu())
        turnover = float(torch.abs(w - prev_w).sum().cpu()) if prev_w is not None else 1.0
        net_ret = gross_ret - args.fee * turnover
        nav *= 1.0 + net_ret
        rows.append({"date": date_str[0], "gross_ret": gross_ret, "turnover": turnover, "net_ret": net_ret, "nav": nav})
        prev_w = w.detach()

    res = pd.DataFrame(rows)
    daily = res["net_ret"].to_numpy()
    ann_ret = nav ** (252 / max(len(res), 1)) - 1
    sharpe = daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)
    max_dd = (res["nav"] / res["nav"].cummax() - 1).min()
    res.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Saved back test to {args.out}")
    print({"ann_ret": ann_ret, "sharpe": sharpe, "max_drawdown": max_dd, "final_nav": nav})


if __name__ == "__main__":
    main()
