from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--data_root", default="A股数据")
    args = p.parse_args()

    data = np.load(args.panel, allow_pickle=True)
    stocks = data["stocks"].astype(str)
    dates = pd.to_datetime(data["dates"])
    cols = data["feature_cols"].astype(str)
    features = data["features"]

    stock_idx = list(stocks).index(args.code)
    end_idx = int(np.where(dates <= pd.Timestamp(args.date))[0][-1])
    window = features[end_idx - 29 : end_idx + 1, stock_idx, :]
    bad = np.argwhere(~np.isfinite(window))
    print("stock", args.code, "feature_end", dates[end_idx].date(), "bad_count", len(bad))
    if len(bad):
        rows = []
        for t, f in bad[:120]:
            rows.append(
                {
                    "date": str(dates[end_idx - 29 + int(t)].date()),
                    "feature": cols[int(f)],
                    "value": window[int(t), int(f)],
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))

    last = features[end_idx, stock_idx, :]
    values = {}
    for name in ["ret_20d", "bias_20", "volatility_20d", "main_force_amount_ratio_5d"]:
        if name in cols:
            values[name] = float(last[list(cols).index(name)])
    if values:
        trend_ok = (
            values.get("ret_20d", 0.0) > -0.35
            and values.get("bias_20", 0.0) > -0.50
            and values.get("volatility_20d", 0.0) < 2.25
        )
        print("feature_values", values)
        print("trend_buy_valid", trend_ok)
        print("thresholds", {"ret_20d": "> -0.35", "bias_20": "> -0.50", "volatility_20d": "< 2.25"})

    for folder in ["daily", "metric", "moneyflow"]:
        path = f"{args.data_root}/{folder}/{args.date.replace('-', '')}.csv"
        df = pd.read_csv(path, dtype={"ts_code": str})
        row = df[df["ts_code"] == args.code]
        print(folder, "rows", len(row))
        if len(row):
            print(row.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
