from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dir", default="experiments/final_competition_package_20260601")
    parser.add_argument("--signal_dir", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    signal_dir = Path(args.signal_dir)
    panel = np.load(package_dir / "data" / "panel_hs300_advanced.npz", allow_pickle=True)
    dates = pd.to_datetime(panel["dates"])
    stocks = panel["stocks"].astype(str)
    feature_cols = panel["feature_cols"].astype(str).tolist()
    features = panel["features"]
    stock_to_idx = {code: i for i, code in enumerate(stocks)}

    rank = pd.read_csv(signal_dir / "ranked_candidates_top50.csv", dtype={"ts_code": str})
    current = pd.read_csv(package_dir / "CURRENT_PORTFOLIO_LIVE.csv", dtype={"ts_code": str})
    target = pd.read_csv(signal_dir / "target_holdings.csv", dtype={"ts_code": str})
    orders = pd.read_csv(signal_dir / "orders.csv", dtype={"ts_code": str})

    end_idx = int(np.where(dates <= pd.Timestamp(args.date))[0][-1])
    last = features[end_idx]
    current_set = set(current["ts_code"])
    target_set = set(target["ts_code"])

    rows = []
    for _, row in rank.head(args.top).iterrows():
        code = row["ts_code"]
        idx = stock_to_idx[code]
        vals = {
            name: float(last[idx, feature_cols.index(name)])
            for name in ["ret_20d", "bias_20", "volatility_20d", "main_force_amount_ratio_5d"]
        }
        calc_score = (
            float(row["raw_score"])
            + 0.18 * vals["ret_20d"]
            + 0.12 * vals["bias_20"]
            + 0.06 * vals["main_force_amount_ratio_5d"]
            - 0.10 * vals["volatility_20d"]
        )
        rows.append(
            {
                "rank": int(row["rank"]),
                "ts_code": code,
                "stock_name": row.get("stock_name", ""),
                "industry": row.get("industry", ""),
                "raw_score": float(row["raw_score"]),
                "trade_score": float(row["trade_score"]),
                "calc_trade_score": calc_score,
                "score_diff": calc_score - float(row["trade_score"]),
                "ret_20d_z": vals["ret_20d"],
                "bias_20_z": vals["bias_20"],
                "volatility_20d_z": vals["volatility_20d"],
                "flow_5d_z": vals["main_force_amount_ratio_5d"],
                "held_before": code in current_set,
                "target": code in target_set,
                "order": ";".join(orders.loc[orders["ts_code"] == code, "side"].tolist()),
            }
        )

    audit = pd.DataFrame(rows)
    out = signal_dir / f"score_component_audit_top{args.top}.csv"
    audit.to_csv(out, index=False, encoding="utf-8-sig")

    print("panel_max_date", dates.max().date())
    print("score_diff_max_abs", audit["score_diff"].abs().max())
    print("\norders")
    print(orders.to_string(index=False) if not orders.empty else "No effective orders.")
    print("\ntop industry counts")
    print(audit.groupby("industry").size().sort_values(ascending=False).to_string())
    print("\nselected_or_ordered")
    cols = [
        "rank",
        "ts_code",
        "stock_name",
        "industry",
        "trade_score",
        "raw_score",
        "ret_20d_z",
        "bias_20_z",
        "volatility_20d_z",
        "flow_5d_z",
        "held_before",
        "target",
        "order",
    ]
    print(audit[(audit["target"]) | (audit["order"] != "")][cols].to_string(index=False))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
