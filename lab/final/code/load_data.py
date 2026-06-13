"""
Data preprocessing for the USTC Deep Learning stock trend project.

Supported inputs:
    daily/      required: daily price-volume CSVs
    metric/     optional: daily fundamental/valuation metrics
    moneyflow/  optional: daily active money-flow CSVs

The script creates a compact panel file:
    features: (num_days, num_stocks, num_features)
    returns : (num_days, num_stocks)  # open-to-open return: open_{t+2}/open_{t+1} - 1
    masks   : (num_days, num_stocks)  # 1 if the return exists, else 0

Important timing rule:
    All features dated t are assumed to be available after the market close of t.
    The trading signal is generated after close of t.
    Execution happens at the OPEN of day t+1.
    The portfolio is held until the OPEN of day t+2.
    Therefore the target return is: open_{t+2} / open_{t+1} - 1.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from stock_pool import filter_stock_pool, load_stock_pool_codes


BASE_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "bias_5",
    "bias_20",
    "volatility_20d",
    "macd_pct",
    "rsi_14",
    "amplitude",
    "close_open_ret",
    "vol_chg_5d",
    "amount_chg_5d",
    "log_vol",
    "log_amount",
]

METRIC_FEATURES = [
    # liquidity / activity
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    # valuation, transformed to reduce extreme scale and missing PE influence
    "ep_ttm",
    "bp",
    "sp_ttm",
    "dp_ttm",
    # size / shares
    "log_total_mv",
    "log_circ_mv",
    "log_total_share",
    "log_float_share",
    "log_free_share",
    "free_float_ratio",
]

MONEYFLOW_FEATURES = [
    # one-day flow intensity, normalized by same-day turnover amount or volume
    "net_mf_amount_ratio",
    "net_mf_vol_ratio",
    "main_force_amount_ratio",
    "elg_net_amount_ratio",
    "lg_net_amount_ratio",
    "md_net_amount_ratio",
    "sm_net_amount_ratio",
    "main_force_imbalance",
    "retail_imbalance",
    # short-term persistence/smoothing
    "net_mf_amount_ratio_3d",
    "net_mf_amount_ratio_5d",
    "main_force_amount_ratio_3d",
    "main_force_amount_ratio_5d",
]

DEFAULT_FEATURES = BASE_FEATURES + METRIC_FEATURES + MONEYFLOW_FEATURES


@dataclass
class PanelData:
    features: np.ndarray
    returns: np.ndarray
    masks: np.ndarray
    dates: np.ndarray
    stocks: np.ndarray
    feature_cols: list[str]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_trade_date_from_filename(path: str) -> pd.Timestamp:
    stem = os.path.splitext(os.path.basename(path))[0]
    return pd.to_datetime(stem, format="%Y%m%d", errors="coerce")


def _coerce_trade_date(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    dt = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(raw, errors="coerce")
    return dt.fillna(fallback)


def _safe_div(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray, eps: float = 1e-8):
    return num / (pd.Series(den).replace(0, np.nan) + eps)


def load_date_folder(
    folder: str,
    name: str,
    required_cols: Sequence[str] = ("ts_code",),
    parse_date_from_filename_when_missing: bool = True,
) -> pd.DataFrame:
    """Load a folder of daily CSVs. Works whether trade_date is in the CSV or only in the file name."""
    if folder is None:
        raise ValueError(f"{name} folder is None")
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {name} folder: {folder}")

    rows: list[pd.DataFrame] = []
    skipped = 0
    for path in tqdm(files, desc=f"Loading {name} CSVs"):
        try:
            one = _normalise_columns(pd.read_csv(path))
            missing_required = sorted(set(required_cols) - set(one.columns))
            if missing_required:
                raise KeyError(f"missing required columns {missing_required}")

            if "trade_date" in one.columns:
                one["trade_date"] = _coerce_trade_date(one["trade_date"])
            elif parse_date_from_filename_when_missing:
                one["trade_date"] = _parse_trade_date_from_filename(path)
            else:
                raise KeyError("missing trade_date")

            one["ts_code"] = one["ts_code"].astype(str).str.strip()
            rows.append(one)
        except Exception as exc:
            skipped += 1
            print(f"[WARN] skipped {path}: {exc}")

    if not rows:
        raise RuntimeError(f"All {name} CSV files failed to load.")

    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["ts_code", "trade_date"])
    df = df.drop_duplicates(["trade_date", "ts_code"], keep="last")
    print(
        f"Loaded {name}: {len(df):,} rows, {df['ts_code'].nunique():,} stocks, "
        f"{df['trade_date'].nunique():,} dates. Skipped {skipped} files."
    )
    return df


def load_raw_data(data_dir: str, filter_bj: bool = True, filter_st: bool = True) -> pd.DataFrame:
    """Read all daily price-volume CSVs."""
    df = load_date_folder(data_dir, name="daily", required_cols=("ts_code",))

    required = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required daily columns: {missing}. Current columns: {list(df.columns)}")

    for col in ["open", "high", "low", "close", "vol", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if filter_bj:
        df = df[~df["ts_code"].str.endswith(".BJ")].copy()

    if filter_st:
        for name_col in ["name", "stock_name", "ts_name"]:
            if name_col in df.columns:
                df = df[~df[name_col].astype(str).str.contains("ST", case=False, na=False)].copy()
                break

    df = df.dropna(subset=required).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def load_metric_data(metric_dir: str | None) -> pd.DataFrame | None:
    if not metric_dir:
        return None
    metric = load_date_folder(metric_dir, name="metric", required_cols=("ts_code",))

    # metric has its own close; the daily close has already been loaded, so drop to avoid collision.
    if "close" in metric.columns:
        metric = metric.drop(columns=["close"])

    metric_cols = [
        "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv",
    ]
    for col in metric_cols:
        if col in metric.columns:
            metric[col] = pd.to_numeric(metric[col], errors="coerce")

    return metric[[c for c in ["trade_date", "ts_code", *metric_cols] if c in metric.columns]].copy()


def load_moneyflow_data(moneyflow_dir: str | None) -> pd.DataFrame | None:
    if not moneyflow_dir:
        return None
    flow = load_date_folder(moneyflow_dir, name="moneyflow", required_cols=("ts_code",))

    flow_cols = [
        "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount",
        "buy_md_vol", "buy_md_amount", "sell_md_vol", "sell_md_amount",
        "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount",
        "buy_elg_vol", "buy_elg_amount", "sell_elg_vol", "sell_elg_amount",
        "net_mf_vol", "net_mf_amount",
    ]
    for col in flow_cols:
        if col in flow.columns:
            flow[col] = pd.to_numeric(flow[col], errors="coerce")

    return flow[[c for c in ["trade_date", "ts_code", *flow_cols] if c in flow.columns]].copy()


def merge_advanced_data(df: pd.DataFrame, metric: pd.DataFrame | None, moneyflow: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if metric is not None:
        before = len(out)
        out = out.merge(metric, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        print(f"Merged metric: {before:,} -> {len(out):,} rows")
    if moneyflow is not None:
        before = len(out)
        out = out.merge(moneyflow, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        print(f"Merged moneyflow: {before:,} -> {len(out):,} rows")
    return out


def calculate_alpha_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal price-volume features and the next-day open-to-open return target."""
    df = df.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = df.groupby("ts_code", group_keys=False)

    close = g["close"]
    df["ret_1d"] = close.pct_change()
    df["ret_5d"] = close.transform(lambda s: s / s.shift(5) - 1.0)
    df["ret_20d"] = close.transform(lambda s: s / s.shift(20) - 1.0)

    ma5 = close.transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma20 = close.transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["bias_5"] = df["close"] / ma5 - 1.0
    df["bias_20"] = df["close"] / ma20 - 1.0

    df["volatility_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())

    ema12 = close.transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = close.transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd_pct"] = (ema12 - ema26) / df["close"].replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(df["ts_code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(df["ts_code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    df["amplitude"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["close_open_ret"] = df["close"] / df["open"].replace(0, np.nan) - 1.0
    df["vol_chg_5d"] = g["vol"].transform(lambda s: s / s.rolling(5, min_periods=5).mean() - 1.0)
    df["amount_chg_5d"] = g["amount"].transform(lambda s: s / s.rolling(5, min_periods=5).mean() - 1.0)
    df["log_vol"] = np.log1p(df["vol"].clip(lower=0))
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    # Metric-derived features.
    if "pe_ttm" in df.columns:
        pe_ttm = df["pe_ttm"].where(df["pe_ttm"] > 0)
        df["ep_ttm"] = 1.0 / pe_ttm
    if "pb" in df.columns:
        pb = df["pb"].where(df["pb"] > 0)
        df["bp"] = 1.0 / pb
    if "ps_ttm" in df.columns:
        ps_ttm = df["ps_ttm"].where(df["ps_ttm"] > 0)
        df["sp_ttm"] = 1.0 / ps_ttm
    if "dv_ttm" in df.columns:
        df["dp_ttm"] = df["dv_ttm"] / 100.0

    for raw_col, new_col in [
        ("total_mv", "log_total_mv"),
        ("circ_mv", "log_circ_mv"),
        ("total_share", "log_total_share"),
        ("float_share", "log_float_share"),
        ("free_share", "log_free_share"),
    ]:
        if raw_col in df.columns:
            df[new_col] = np.log1p(df[raw_col].clip(lower=0))

    if {"free_share", "total_share"}.issubset(df.columns):
        df["free_float_ratio"] = df["free_share"] / df["total_share"].replace(0, np.nan)

    # Moneyflow-derived features. Daily amount unit is usually thousand yuan; moneyflow amount is usually 10k yuan.
    amount_10k = df["amount"] / 10.0
    vol_hands = df["vol"]

    def has_cols(cols: Sequence[str]) -> bool:
        return set(cols).issubset(df.columns)

    if "net_mf_amount" in df.columns:
        df["net_mf_amount_ratio"] = df["net_mf_amount"] / amount_10k.replace(0, np.nan)
    if "net_mf_vol" in df.columns:
        df["net_mf_vol_ratio"] = df["net_mf_vol"] / vol_hands.replace(0, np.nan)

    if has_cols(["buy_elg_amount", "sell_elg_amount"]):
        df["elg_net_amount_ratio"] = (df["buy_elg_amount"] - df["sell_elg_amount"]) / amount_10k.replace(0, np.nan)
    if has_cols(["buy_lg_amount", "sell_lg_amount"]):
        df["lg_net_amount_ratio"] = (df["buy_lg_amount"] - df["sell_lg_amount"]) / amount_10k.replace(0, np.nan)
    if has_cols(["buy_md_amount", "sell_md_amount"]):
        df["md_net_amount_ratio"] = (df["buy_md_amount"] - df["sell_md_amount"]) / amount_10k.replace(0, np.nan)
    if has_cols(["buy_sm_amount", "sell_sm_amount"]):
        df["sm_net_amount_ratio"] = (df["buy_sm_amount"] - df["sell_sm_amount"]) / amount_10k.replace(0, np.nan)
        df["retail_imbalance"] = (df["buy_sm_amount"] - df["sell_sm_amount"]) / (
            df["buy_sm_amount"] + df["sell_sm_amount"]
        ).replace(0, np.nan)

    if has_cols(["buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount"]):
        main_buy = df["buy_lg_amount"] + df["buy_elg_amount"]
        main_sell = df["sell_lg_amount"] + df["sell_elg_amount"]
        df["main_force_amount_ratio"] = (main_buy - main_sell) / amount_10k.replace(0, np.nan)
        df["main_force_imbalance"] = (main_buy - main_sell) / (main_buy + main_sell).replace(0, np.nan)

    # Rolling flow persistence features. They use only current and previous dates.
    g2 = df.groupby("ts_code", group_keys=False)
    for base_col in ["net_mf_amount_ratio", "main_force_amount_ratio"]:
        if base_col in df.columns:
            df[f"{base_col}_3d"] = g2[base_col].transform(lambda s: s.rolling(3, min_periods=2).mean())
            df[f"{base_col}_5d"] = g2[base_col].transform(lambda s: s.rolling(5, min_periods=3).mean())

    # =========================================================================
    # Label: open-to-open return for next-day-open execution.
    # Features at close of day t → execute at open of day t+1 → exit at open of day t+2.
    # target_return_1d[t] = open[t+2] / open[t+1] - 1
    # This is NOT a feature and must NOT be standardised.
    # =========================================================================
    next_open = g["open"].shift(-1)       # open price on day t+1
    next_next_open = g["open"].shift(-2)  # open price on day t+2
    df["target_return_1d"] = next_next_open / next_open - 1.0

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def existing_feature_cols(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def cross_sectional_standardize(df: pd.DataFrame, feature_cols: Sequence[str], clip_z: float = 3.0) -> pd.DataFrame:
    """Cross-sectional winsorized z-score by trade_date."""
    out = df.copy()
    if "trade_date" not in out.columns:
        raise KeyError(f"trade_date is missing before standardisation. Columns: {list(out.columns)}")
    if "ts_code" not in out.columns:
        raise KeyError(f"ts_code is missing before standardisation. Columns: {list(out.columns)}")

    for col in feature_cols:
        if col not in out.columns:
            raise KeyError(f"Feature {col!r} not found. Available columns: {list(out.columns)}")
        s = out[col].astype(float)
        mean = s.groupby(out["trade_date"]).transform("mean")
        std = s.groupby(out["trade_date"]).transform("std").replace(0, np.nan)
        lower = mean - clip_z * std
        upper = mean + clip_z * std
        clipped = s.clip(lower=lower, upper=upper)
        out[col] = ((clipped - mean) / std).fillna(0.0).astype(np.float32)

    return out.reset_index(drop=True)


def build_panel(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    top_k: int = 1000,
    min_history: int = 60,
) -> PanelData:
    """Align selected stocks on a common date index."""
    needed = {"trade_date", "ts_code", "target_return_1d", *feature_cols}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns for panel construction: {missing}")

    usable = df.dropna(subset=list(feature_cols)).copy()
    counts = usable.groupby("ts_code")["trade_date"].nunique()
    eligible = counts[counts >= min_history].sort_values(ascending=False)
    stocks = eligible.head(top_k).index.tolist()
    if not stocks:
        raise RuntimeError("No stocks satisfy min_history. Lower min_history or inspect the data.")

    dates = np.array(sorted(usable["trade_date"].unique()))
    grid = pd.MultiIndex.from_product([dates, stocks], names=["trade_date", "ts_code"])
    panel = usable.set_index(["trade_date", "ts_code"]).sort_index().reindex(grid)

    feat_df = panel[list(feature_cols)].groupby(level="ts_code").ffill().fillna(0.0)
    ret_df = panel["target_return_1d"]
    mask_df = ret_df.notna().astype(np.float32)
    ret_df = ret_df.fillna(0.0)

    d, n, f = len(dates), len(stocks), len(feature_cols)
    features = feat_df.to_numpy(np.float32).reshape(d, n, f)
    returns = ret_df.to_numpy(np.float32).reshape(d, n)
    masks = mask_df.to_numpy(np.float32).reshape(d, n)

    return PanelData(
        features=features,
        returns=returns,
        masks=masks,
        dates=dates.astype("datetime64[ns]"),
        stocks=np.array(stocks),
        feature_cols=list(feature_cols),
    )


def save_panel(panel: PanelData, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        features=panel.features,
        returns=panel.returns,
        masks=panel.masks,
        dates=panel.dates.astype("datetime64[D]").astype(str),
        stocks=panel.stocks.astype(str),
        feature_cols=np.array(panel.feature_cols),
    )
    print(f"Saved panel to {out_path}")
    print(f"features={panel.features.shape}, returns={panel.returns.shape}, masks={panel.masks.shape}")
    print(f"NOTE: returns represent open_{{t+1}} -> open_{{t+2}} (next-day-open execution)")
    print("feature_cols:")
    for i, col in enumerate(panel.feature_cols, start=1):
        print(f"  {i:02d}. {col}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to daily CSV folder.")
    parser.add_argument("--metric_dir", type=str, default=None, help="Optional path to metric CSV folder.")
    parser.add_argument("--moneyflow_dir", type=str, default=None, help="Optional path to moneyflow CSV folder.")
    parser.add_argument("--out", type=str, default="data/panel_hs300_advanced.npz")
    parser.add_argument("--top_k", type=int, default=None, help="Maximum stocks kept in the panel. Default: 300 for hs300, else 1000.")
    parser.add_argument("--min_history", type=int, default=80)
    parser.add_argument(
        "--pool",
        type=str,
        default="hs300",
        choices=["hs300", "custom", "all"],
        help="Stock universe. Use hs300/custom with --stock_pool_file; use all for the old top-K universe.",
    )
    parser.add_argument(
        "--stock_pool_file",
        type=str,
        default=None,
        help=(
            "CSV/XLSX/TXT containing target stock codes. For HS300, pass a constituent file "
            "with a column such as ts_code, con_code, code, 股票代码, or 成分券代码."
        ),
    )
    args = parser.parse_args()

    pool_codes = None
    if args.pool in {"hs300", "custom"}:
        if not args.stock_pool_file:
            raise ValueError(
                "--pool hs300/custom requires --stock_pool_file so that the exact constituent date "
                "is explicit and reproducible. Example: --stock_pool_file data/hs300_constituents.csv"
            )
        pool_codes = load_stock_pool_codes(args.stock_pool_file)
        print(f"Loaded stock pool from {args.stock_pool_file}: {len(pool_codes)} codes")

    target_top_k = args.top_k
    if target_top_k is None:
        target_top_k = 300 if args.pool in {"hs300", "custom"} else 1000

    # Load and merge daily data, metrics data, and money flow data.
    raw = load_raw_data(args.data_dir)
    if pool_codes is not None:
        raw = filter_stock_pool(raw, pool_codes, name="daily")

    metric = load_metric_data(args.metric_dir)
    if metric is not None and pool_codes is not None:
        metric = filter_stock_pool(metric, pool_codes, name="metric")

    flow = load_moneyflow_data(args.moneyflow_dir)
    if flow is not None and pool_codes is not None:
        flow = filter_stock_pool(flow, pool_codes, name="moneyflow")

    merged = merge_advanced_data(raw, metric, flow)
    factors = calculate_alpha_factors(merged)

    feature_cols = existing_feature_cols(factors, DEFAULT_FEATURES)
    missing = sorted(set(DEFAULT_FEATURES) - set(feature_cols))
    if missing:
        print(f"[WARN] These requested features are unavailable and will be skipped: {missing}")

    norm = cross_sectional_standardize(factors, feature_cols)
    assert "trade_date" in norm.columns and "ts_code" in norm.columns, norm.columns
    panel = build_panel(norm, feature_cols, top_k=target_top_k, min_history=args.min_history)
    if args.pool == "hs300" and len(panel.stocks) != 300:
        print(
            f"[WARN] HS300 panel contains {len(panel.stocks)} stocks instead of 300. "
            "Usually this means some constituents are missing from the raw data or fail min_history."
        )
    save_panel(panel, args.out)


if __name__ == "__main__":
    main()