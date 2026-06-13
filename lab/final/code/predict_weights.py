from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from losses import scores_to_weights
from model import build_scorer


def rebuild_model(ckpt: dict, num_features: int):
    cfg = ckpt.get("config", {})
    name = cfg.get("model", "transformer")
    return build_scorer(
        name,
        num_features=num_features,
        d_model=int(cfg.get("d_model", 64)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
    )


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--date", type=str, default="latest", help="Use features up to this date; default uses latest date in panel.")
    p.add_argument("--top_n", type=int, default=30)
    p.add_argument("--equal_weight", action="store_true")
    p.add_argument("--out", type=str, default="today_weights.csv")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data = np.load(args.panel, allow_pickle=True)
    features = data["features"].astype(np.float32)       # (D, N, F)
    dates = pd.to_datetime(data["dates"])
    stocks = data["stocks"].astype(str)
    feature_cols = data["feature_cols"].astype(str).tolist()

    if args.date == "latest":
        end_idx = len(dates) - 1
    else:
        target = pd.Timestamp(args.date)
        matches = np.where(dates <= target)[0]
        if len(matches) == 0:
            raise ValueError(f"No date on or before {args.date} in panel.")
        end_idx = int(matches[-1])

    if end_idx < args.seq_len - 1:
        raise ValueError(f"Not enough history before {dates[end_idx].date()} for seq_len={args.seq_len}.")

    x_np = features[end_idx - args.seq_len + 1 : end_idx + 1]  # (T, N, F)
    # In live prediction there is no known next-day return, so do NOT use target masks.
    # Use a simple feature-validity mask instead.
    valid = np.isfinite(x_np).all(axis=(0, 2)).astype(np.float32)  # (N,)
    x = torch.from_numpy(np.transpose(x_np, (1, 0, 2))).unsqueeze(0).to(args.device)  # (1, N, T, F)
    m = torch.from_numpy(valid).unsqueeze(0).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = rebuild_model(ckpt, num_features=len(feature_cols)).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores = model(x)
    weights = scores_to_weights(scores, m, top_n=args.top_n, equal_weight=args.equal_weight)
    out = pd.DataFrame({
        "ts_code": stocks,
        "score": scores.squeeze(0).cpu().numpy(),
        "weight": weights.squeeze(0).cpu().numpy(),
    }).sort_values("weight", ascending=False)
    out = out[out["weight"] > 0].reset_index(drop=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Feature end date={dates[end_idx].date()} (decision made after close of this date)")
    print(f">>> Execute trades at OPEN of next trading day <<<")
    print(f"Saved {len(out)} holdings to {args.out}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
