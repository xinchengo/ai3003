from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predict_weights import rebuild_model


def safe_load_checkpoint(path: str, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def rank_valid(scores: np.ndarray, valid: np.ndarray, descending: bool = True) -> list[int]:
    valid = valid.astype(bool) & np.isfinite(scores)
    if not valid.any():
        return []
    masked = np.where(valid, scores, -np.inf if descending else np.inf)
    order = np.argsort(masked)
    if descending:
        order = order[::-1]
    return [int(i) for i in order if valid[i]]


def feature_values(last: np.ndarray, feature_cols: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for name in ["ret_20d", "bias_20", "volatility_20d", "main_force_amount_ratio_5d"]:
        out[name] = last[:, feature_cols.index(name)] if name in feature_cols else np.zeros(last.shape[0])
    return out


def adjust_scores(scores: np.ndarray, valid: np.ndarray, feats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ret20 = feats["ret_20d"]
    bias20 = feats["bias_20"]
    vol20 = feats["volatility_20d"]
    flow = feats["main_force_amount_ratio_5d"]
    trade_score = scores + 0.18 * ret20 + 0.12 * bias20 + 0.06 * flow - 0.10 * vol20
    trend_ok = (ret20 > -0.35) & (bias20 > -0.50) & (vol20 < 2.25)
    return trade_score, valid & trend_ok


def load_name_map(path: str | None) -> tuple[dict[str, str], dict[str, str]]:
    if not path:
        candidates = [p / "basic.csv" for p in Path(".").iterdir() if p.is_dir() and (p / "basic.csv").exists()]
        path = str(candidates[0]) if candidates else None
    if not path or not Path(path).exists():
        return {}, {}
    basic = pd.read_csv(path, dtype={"ts_code": str})
    return (
        basic.set_index("ts_code")["name"].fillna("").to_dict(),
        basic.set_index("ts_code")["industry"].fillna("").to_dict(),
    )


def read_current_holdings(path: str | None) -> list[str]:
    if not path or not Path(path).exists():
        return []
    df = pd.read_csv(path, dtype={"ts_code": str})
    if "ts_code" not in df.columns:
        raise ValueError(f"{path} must contain ts_code column.")
    return df["ts_code"].dropna().astype(str).tolist()


def read_realized_returns(path: str | None) -> dict[str, float]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype={"ts_code": str})
    if "ts_code" not in df.columns or "realized_return" not in df.columns:
        raise ValueError(f"{path} must contain ts_code and realized_return columns.")
    return dict(zip(df["ts_code"].astype(str), df["realized_return"].astype(float)))


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Live/competition signal generator for the final improved strategy.")
    p.add_argument("--panel", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--date", default="latest", help="Feature date. Use latest by default; trade next trading day open.")
    p.add_argument("--current_holdings", default=None, help="CSV with current ts_code holdings. Omit on initial day.")
    p.add_argument("--realized_returns", default=None, help="Optional CSV ts_code,realized_return for stop-loss decisions.")
    p.add_argument("--trading_day_no", type=int, required=True, help="Competition trading day number, starting from 1.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rebalance_interval", type=int, default=5)
    p.add_argument("--stop_loss", type=float, default=-0.035)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--names_csv", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.panel, allow_pickle=True)
    features = data["features"].astype(np.float32)
    dates = pd.to_datetime(data["dates"])
    stocks = data["stocks"].astype(str)
    feature_cols = data["feature_cols"].astype(str).tolist()
    stock_to_idx = {s: i for i, s in enumerate(stocks)}

    if args.date == "latest":
        end_idx = len(dates) - 1
    else:
        target = pd.Timestamp(args.date)
        matches = np.where(dates <= target)[0]
        if len(matches) == 0:
            raise ValueError(f"No panel date on or before {args.date}.")
        end_idx = int(matches[-1])
    if end_idx < args.seq_len - 1:
        raise ValueError("Not enough history for seq_len.")

    x_np = features[end_idx - args.seq_len + 1 : end_idx + 1]
    valid = np.isfinite(x_np).all(axis=(0, 2))
    x = torch.from_numpy(np.transpose(x_np, (1, 0, 2))).unsqueeze(0).to(args.device)

    ckpt = safe_load_checkpoint(args.checkpoint, args.device)
    model = rebuild_model(ckpt, num_features=len(feature_cols)).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    raw_scores = model(x).squeeze(0).detach().cpu().numpy().astype(np.float64)
    feats = feature_values(x_np[-1], feature_cols)
    trade_scores, buy_valid = adjust_scores(raw_scores, valid, feats)

    ranked_high = rank_valid(trade_scores, buy_valid, descending=True)
    if len(ranked_high) < min(args.top_n, int(valid.sum())):
        fallback = rank_valid(trade_scores, valid, descending=True)
        seen = set(ranked_high)
        ranked_high = ranked_high + [i for i in fallback if i not in seen]
    if not ranked_high:
        raise ValueError("No valid candidates.")

    name_map, industry_map = load_name_map(args.names_csv)
    current_codes = read_current_holdings(args.current_holdings)
    current = [stock_to_idx[c] for c in current_codes if c in stock_to_idx]
    realized = read_realized_returns(args.realized_returns)

    sold: list[int] = []
    bought: list[int] = []
    action = "initial_build"
    if not current:
        target = ranked_high[: args.top_n]
        bought = target.copy()
    else:
        action = "scheduled_rebalance" if (args.trading_day_no - 1) % args.rebalance_interval == 0 else "daily_stop_or_validity_check"
        holding_set = set(current)
        stop_held = [
            stock_to_idx[c]
            for c, r in realized.items()
            if c in stock_to_idx and stock_to_idx[c] in holding_set and np.isfinite(r) and r <= args.stop_loss
        ]
        invalid_held = [i for i in current if not valid[i]]
        forced = []
        for i in invalid_held + stop_held:
            if i not in forced:
                forced.append(i)
        score_rebalance = (args.trading_day_no - 1) % args.rebalance_interval == 0
        sell_quota = max(args.k if score_rebalance else 0, len(forced))

        def sell_score(i: int) -> float:
            if i in forced:
                return -np.inf
            return float(trade_scores[i]) if valid[i] and np.isfinite(trade_scores[i]) else -np.inf

        sell_candidates = sorted(current, key=sell_score)
        sold = sell_candidates[: min(sell_quota, len(sell_candidates))]
        holding_set.difference_update(sold)

        for candidate in ranked_high:
            if len(holding_set) >= args.top_n:
                break
            if candidate in holding_set:
                continue
            holding_set.add(candidate)
            bought.append(candidate)
        target = [i for i in ranked_high if i in holding_set]

    target_codes = stocks[target].tolist()
    sold_codes = stocks[sold].tolist()
    bought_codes = stocks[bought].tolist()
    current_set = set(current_codes)
    target_set = set(target_codes)
    effective_sold = sorted(current_set - target_set)
    effective_bought = sorted(target_set - current_set)
    rebought = sorted(set(sold_codes) & set(bought_codes))

    def enrich(codes: list[str]) -> pd.DataFrame:
        rows = []
        for rank, code in enumerate(codes, start=1):
            idx = stock_to_idx[code]
            rows.append(
                {
                    "rank": rank,
                    "ts_code": code,
                    "stock_name": name_map.get(code, ""),
                    "industry": industry_map.get(code, ""),
                    "weight": 1.0 / len(codes) if codes else 0.0,
                    "raw_score": raw_scores[idx],
                    "trade_score": trade_scores[idx],
                    "trend_buy_valid": bool(buy_valid[idx]),
                }
            )
        return pd.DataFrame(rows)

    target_df = enrich(target_codes)
    orders = pd.DataFrame(
        [
            {"side": "SELL", "ts_code": c, "stock_name": name_map.get(c, ""), "industry": industry_map.get(c, "")}
            for c in effective_sold
        ]
        + [
            {"side": "BUY", "ts_code": c, "stock_name": name_map.get(c, ""), "industry": industry_map.get(c, "")}
            for c in effective_bought
        ]
    )
    ranked = enrich(stocks[ranked_high[:50]].tolist())
    summary = pd.DataFrame(
        [
            {
                "feature_end_date": str(dates[end_idx].date()),
                "trading_day_no": args.trading_day_no,
                "action": action,
                "top_n": args.top_n,
                "k": args.k,
                "rebalance_interval": args.rebalance_interval,
                "stop_loss": args.stop_loss,
                "current_holding_count": len(current_codes),
                "target_holding_count": len(target_codes),
                "effective_sell": ";".join(effective_sold),
                "effective_buy": ";".join(effective_bought),
                "raw_sold": ";".join(sold_codes),
                "raw_bought": ";".join(bought_codes),
                "sold_and_rebought": ";".join(rebought),
            }
        ]
    )

    target_df.to_csv(out_dir / "target_holdings.csv", index=False, encoding="utf-8-sig")
    orders.to_csv(out_dir / "orders.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(out_dir / "ranked_candidates_top50.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "signal_summary.csv", index=False, encoding="utf-8-sig")
    target_df[["ts_code", "stock_name", "industry", "weight"]].to_csv(out_dir / "portfolio_state.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print("\nOrders:")
    print(orders.to_string(index=False) if not orders.empty else "No effective orders.")
    print("\nTarget holdings:")
    print(target_df.to_string(index=False))


if __name__ == "__main__":
    main()
