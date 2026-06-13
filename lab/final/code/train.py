from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from losses import combined_loss, ic_loss, scores_to_weights
from model import build_scorer


class StockPanelDataset(Dataset):
    def __init__(self, npz_path: str, seq_len: int = 30, label_horizon: int = 1):
        data = np.load(npz_path, allow_pickle=True)
        self.features = data["features"].astype(np.float32)
        self.returns = data["returns"].astype(np.float32)
        self.masks = data["masks"].astype(np.float32)
        self.dates = pd.to_datetime(data["dates"])
        self.stocks = data["stocks"].astype(str)
        self.feature_cols = data["feature_cols"].astype(str).tolist()
        self.seq_len = seq_len
        self.label_horizon = label_horizon
        if label_horizon <= 0:
            raise ValueError("label_horizon must be positive")
        if len(self.dates) <= seq_len + label_horizon:
            raise ValueError("Not enough dates for the requested seq_len")
        # A sample ending on date t uses features through t and target return open(t+1)->open(t+2).
        self.end_indices = np.arange(seq_len - 1, len(self.dates) - label_horizon - 1)

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int):
        end = int(self.end_indices[idx])
        start = end - self.seq_len + 1
        x = self.features[start : end + 1]  # (T, N, F)
        if self.label_horizon == 1:
            y = self.returns[end]          # (N,) open_{t+1} -> open_{t+2}
            m = self.masks[end]            # (N,)
        else:
            future = self.returns[end : end + self.label_horizon]
            future_mask = self.masks[end : end + self.label_horizon]
            y = np.prod(1.0 + future, axis=0) - 1.0
            m = np.prod(future_mask, axis=0).astype(np.float32)
        return (
            torch.from_numpy(np.transpose(x, (1, 0, 2))).float(),  # (N, T, F)
            torch.from_numpy(y).float(),
            torch.from_numpy(m).float(),
            str(self.dates[end].date()),
        )

    @staticmethod
    def _select_by_range(sample_dates: pd.DatetimeIndex, start: str | None, end: str | None) -> list[int]:
        mask = np.ones(len(sample_dates), dtype=bool)
        if start:
            mask &= sample_dates >= pd.Timestamp(start)
        if end:
            mask &= sample_dates <= pd.Timestamp(end)
        return np.where(mask)[0].tolist()

    def split_by_date(
        self,
        train_start: str | None = None,
        train_end: str | None = None,
        test_start: str | None = None,
        test_end: str | None = None,
        split_mode: str = "recent_train_past_test",
        val_end: str | None = None,
    ) -> tuple[list[int], list[int]]:
        """
        Date-based split without random date mixing.

        split_mode="recent_train_past_test" implements the requested setup:
            recent dates -> training set; older dates -> test/evaluation set.

        split_mode="past_train_recent_val" keeps the old behaviour:
            dates <= train_end -> training set; later dates -> validation set.
        """
        sample_dates = pd.DatetimeIndex(self.dates[self.end_indices])

        if split_mode == "recent_train_past_test":
            train_idx = self._select_by_range(sample_dates, train_start, train_end)
            test_idx = self._select_by_range(sample_dates, test_start, test_end)
        elif split_mode == "past_train_recent_val":
            if not train_end:
                raise ValueError("split_mode=past_train_recent_val requires --train_end")
            train_end_ts = pd.Timestamp(train_end)
            train_mask = sample_dates <= train_end_ts
            if train_start:
                train_mask &= sample_dates >= pd.Timestamp(train_start)
            train_idx = np.where(train_mask)[0].tolist()
            if val_end is None:
                val_mask = sample_dates > train_end_ts
            else:
                val_mask = (sample_dates > train_end_ts) & (sample_dates <= pd.Timestamp(val_end))
            test_idx = np.where(val_mask)[0].tolist()
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        overlap = set(train_idx) & set(test_idx)
        if overlap:
            raise ValueError(
                f"Train/test date ranges overlap for {len(overlap)} samples. "
                "Please adjust --train_start/--train_end/--test_start/--test_end."
            )
        return train_idx, test_idx


@dataclass
class TrainConfig:
    panel: str
    seq_len: int = 30
    label_horizon: int = 1
    model: str = "transformer"
    split_mode: str = "recent_train_past_test"
    train_start: str | None = "2016-01-04"
    train_end: str | None = "2025-12-31"
    test_start: str | None = "2026-01-01"
    test_end: str | None = "2026-02-27"
    # Legacy argument for split_mode=past_train_recent_val.
    val_end: str | None = None
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4
    top_n: int = 10
    shuffle_train: bool = False
    seed: int = 42
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    weight_decay: float = 1e-2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "runs/stock_model_hs300_recent"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False


def _masked_corr_np(scores: np.ndarray, returns: np.ndarray, mask: np.ndarray) -> float:
    valid = (mask > 0) & np.isfinite(scores) & np.isfinite(returns)
    if valid.sum() < 3:
        return np.nan
    s = scores[valid]
    r = returns[valid]
    s = (s - s.mean()) / (s.std() + 1e-12)
    r = (r - r.mean()) / (r.std() + 1e-12)
    return float(np.mean(s * r))


def _rank_ic_np(scores: np.ndarray, returns: np.ndarray, mask: np.ndarray) -> float:
    valid = (mask > 0) & np.isfinite(scores) & np.isfinite(returns)
    if valid.sum() < 3:
        return np.nan
    s_rank = pd.Series(scores[valid]).rank(method="average").to_numpy(dtype=float)
    r_rank = pd.Series(returns[valid]).rank(method="average").to_numpy(dtype=float)
    return _masked_corr_np(s_rank, r_rank, np.ones_like(s_rank))


def _portfolio_stats(port: np.ndarray) -> dict[str, float]:
    if len(port) == 0:
        return {"daily_ret_mean": np.nan, "ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    mean = float(np.nanmean(port))
    std = float(np.nanstd(port))
    return {
        "daily_ret_mean": mean,
        "ann_ret": float((1 + mean) ** 252 - 1),
        "ann_vol": float(std * np.sqrt(252)),
        "sharpe": float(mean / (std + 1e-8) * np.sqrt(252)),
    }


@torch.no_grad()
def evaluate(model, loader, device: str, top_n: int) -> dict[str, float]:
    model.eval()
    losses, ics, rank_ics, port_rets = [], [], [], []
    for x, y, m, _ in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        scores = model(x)
        losses.append(float(combined_loss(scores, y, m, top_n=top_n).item()))
        w = scores_to_weights(scores, m, top_n=top_n, equal_weight=True)
        port_rets.extend(((w * y * m).sum(dim=1)).detach().cpu().numpy().tolist())
        s_np = scores.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        m_np = m.detach().cpu().numpy()
        for i in range(s_np.shape[0]):
            ics.append(_masked_corr_np(s_np[i], y_np[i], m_np[i]))
            rank_ics.append(_rank_ic_np(s_np[i], y_np[i], m_np[i]))
    port = np.array(port_rets, dtype=float)
    ic_arr = np.array(ics, dtype=float)
    rank_ic_arr = np.array(rank_ics, dtype=float)
    out = {
        "val_loss": float(np.nanmean(losses)) if len(losses) else np.nan,
        "ic_mean": float(np.nanmean(ic_arr)) if len(ic_arr) else np.nan,
        "icir": float(np.nanmean(ic_arr) / (np.nanstd(ic_arr) + 1e-8)) if len(ic_arr) else np.nan,
        "rank_ic_mean": float(np.nanmean(rank_ic_arr)) if len(rank_ic_arr) else np.nan,
        "rank_icir": float(np.nanmean(rank_ic_arr) / (np.nanstd(rank_ic_arr) + 1e-8)) if len(rank_ic_arr) else np.nan,
    }
    out.update(_portfolio_stats(port))
    return out


def _date_span(ds: StockPanelDataset, idx: list[int]) -> str:
    if not idx:
        return "EMPTY"
    dates = pd.DatetimeIndex(ds.dates[ds.end_indices[idx]])
    return f"{dates.min().date()} -> {dates.max().date()}"


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    ds = StockPanelDataset(cfg.panel, seq_len=cfg.seq_len, label_horizon=cfg.label_horizon)
    train_idx, test_idx = ds.split_by_date(
        train_start=cfg.train_start,
        train_end=cfg.train_end,
        test_start=cfg.test_start,
        test_end=cfg.test_end,
        split_mode=cfg.split_mode,
        val_end=cfg.val_end,
    )
    if not train_idx or not test_idx:
        raise ValueError(
            f"Empty split. train={len(train_idx)}, test/eval={len(test_idx)}. "
            "Adjust --train_start/--train_end/--test_start/--test_end."
        )

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=0,
    )
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = build_scorer(
        cfg.model,
        num_features=len(ds.feature_cols),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_ic = -1e9
    best_sharpe = -1e9
    history = []

    print(f"Dataset: {len(ds)} samples, {len(ds.stocks)} stocks, {len(ds.feature_cols)} features")
    print(f"Split mode: {cfg.split_mode}")
    print(f"Train samples={len(train_idx)}, span={_date_span(ds, train_idx)}")
    print(f"Test/Eval samples={len(test_idx)}, span={_date_span(ds, test_idx)}")
    print(f"Model={cfg.model}, seed={cfg.seed}, label_horizon={cfg.label_horizon}")
    print(f"shuffle_train={cfg.shuffle_train}, device={cfg.device}")
    print("Return type: open-to-open (buy at next open, sell at open after)")
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for x, y, m, _ in tqdm(train_loader, desc=f"epoch {epoch}"):
            x, y, m = x.to(cfg.device), y.to(cfg.device), m.to(cfg.device)
            scores = model(x)
            loss = combined_loss(scores, y, m, top_n=cfg.top_n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        metrics = evaluate(model, test_loader, cfg.device, cfg.top_n)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(row)

        ckpt = {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "feature_cols": ds.feature_cols,
            "stocks": ds.stocks.tolist(),
            "train_date_span": _date_span(ds, train_idx),
            "val_date_span": _date_span(ds, test_idx),
            "test_date_span": _date_span(ds, test_idx),
            "epoch": epoch,
            "metrics": metrics,
        }
        if metrics["ic_mean"] > best_ic:
            best_ic = metrics["ic_mean"]
            torch.save(ckpt, os.path.join(cfg.out_dir, "best.pt"))
        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            torch.save(ckpt, os.path.join(cfg.out_dir, "best_sharpe.pt"))

    torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))
    pd.DataFrame(history).to_csv(os.path.join(cfg.out_dir, "history.csv"), index=False)
    print(f"Best evaluation IC={best_ic:.6f}; saved to {os.path.join(cfg.out_dir, 'best.pt')}")
    print(f"Best evaluation Sharpe={best_sharpe:.6f}; saved to {os.path.join(cfg.out_dir, 'best_sharpe.pt')}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--label_horizon", type=int, default=1)
    p.add_argument("--model", type=str, default="transformer", choices=["transformer", "gru", "lstm", "transformer_cs", "xformer_cs"])
    p.add_argument(
        "--split_mode",
        type=str,
        default="recent_train_past_test",
        choices=["recent_train_past_test", "past_train_recent_val"],
    )
    p.add_argument("--train_start", type=str, default="2016-01-04")
    p.add_argument("--train_end", type=str, default="2025-12-31")
    p.add_argument("--test_start", type=str, default="2026-01-01")
    p.add_argument("--test_end", type=str, default="2026-02-27")
    p.add_argument("--val_end", type=str, default=None, help="Only used by split_mode=past_train_recent_val")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--shuffle_train", action="store_true", help="Shuffle training samples. Off by default to keep date order.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="runs/stock_model_hs300_recent")
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train(parse_args())
