from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def check_one(path: Path, expected_date: str) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    df = pd.read_csv(path, dtype={"ts_code": str})
    out = {
        "path": str(path),
        "exists": True,
        "rows": len(df),
        "stocks": int(df["ts_code"].nunique()) if "ts_code" in df.columns else None,
        "duplicate_ts_code": int(df["ts_code"].duplicated().sum()) if "ts_code" in df.columns else None,
    }
    if "trade_date" in df.columns:
        dates = sorted(pd.Series(df["trade_date"]).astype(str).str.replace(r"\.0$", "", regex=True).unique().tolist())
        out["trade_dates"] = ";".join(dates[:5])
        out["date_ok"] = all(d == expected_date for d in dates)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="A股数据")
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--holdings", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = []
    root = Path(args.root)
    for d in args.dates:
        ymd = d.replace("-", "")
        for folder in ["daily", "metric", "moneyflow"]:
            row = check_one(root / folder / f"{ymd}.csv", ymd)
            row["folder"] = folder
            row["date"] = d
            rows.append(row)
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False, encoding="utf-8-sig")

    if args.holdings:
        hold = pd.read_csv(args.holdings, dtype={"ts_code": str})
        latest = args.dates[-1].replace("-", "")
        daily_path = root / "daily" / f"{latest}.csv"
        if daily_path.exists():
            daily = pd.read_csv(daily_path, dtype={"ts_code": str})
            cols = [c for c in ["ts_code", "open", "high", "low", "close", "pre_close", "pct_chg"] if c in daily.columns]
            merged = hold[["ts_code"]].merge(daily[cols], on="ts_code", how="left")
            print("\ncurrent holdings daily snapshot:")
            print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
