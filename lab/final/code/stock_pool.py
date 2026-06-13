from __future__ import annotations

import os
import re
from typing import Iterable

import pandas as pd


_CODE_RE = re.compile(r"^(\d{6})(?:\.(SH|SZ|BJ))?$", re.IGNORECASE)
_PREFIX_RE = re.compile(r"^(SH|SZ|BJ)[-_\. ]?(\d{6})$", re.IGNORECASE)
_SUFFIX_RE = re.compile(r"^(\d{6})[-_ ]?(SH|SZ|BJ)$", re.IGNORECASE)


def infer_exchange(code6: str) -> str | None:
    """Infer A-share exchange suffix for a six-digit stock code."""
    if code6.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "SH"
    if code6.startswith(("000", "001", "002", "003", "200", "300", "301")):
        return "SZ"
    if code6.startswith(("4", "8", "9")):
        return "BJ"
    return None


def normalise_ts_code(value: object) -> str | None:
    """Normalise common A-share code formats to TuShare style, e.g. 000001.SZ."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return None
    text = text.replace("'", "").replace('"', "")

    m = _PREFIX_RE.match(text)
    if m:
        exch, code6 = m.group(1).upper(), m.group(2)
        return f"{code6}.{exch}"

    m = _SUFFIX_RE.match(text)
    if m:
        code6, exch = m.group(1), m.group(2).upper()
        return f"{code6}.{exch}"

    m = _CODE_RE.match(text)
    if m:
        code6, exch = m.group(1), m.group(2)
        exch = exch.upper() if exch else infer_exchange(code6)
        return f"{code6}.{exch}" if exch else code6

    # Some downloaded constituent tables store code as a float-looking string.
    if re.match(r"^\d{1,6}\.0$", text):
        code6 = text.split(".")[0].zfill(6)
        exch = infer_exchange(code6)
        return f"{code6}.{exch}" if exch else code6

    return text


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext in {".txt", ".list"}:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
        return pd.DataFrame({"ts_code": lines})
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="gbk")


def load_stock_pool_codes(path: str) -> set[str]:
    """
    Load a stock-pool file and return normalised ts_code strings.

    Supported formats: CSV/XLSX/TXT. Common column names are recognised automatically,
    including ts_code, con_code, code, symbol, ticker, 股票代码, 成分券代码.
    """
    if not path:
        raise ValueError("stock-pool file path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stock-pool file not found: {path}")

    table = _read_table(path)
    if table.empty:
        raise ValueError(f"Stock-pool file is empty: {path}")

    preferred_cols = [
        "ts_code", "con_code", "code", "symbol", "ticker", "证券代码", "股票代码",
        "成分券代码", "成份券代码", "wind_code", "windcode",
    ]
    col_map = {str(c).strip().lower(): c for c in table.columns}
    selected_col = None
    for name in preferred_cols:
        if name.lower() in col_map:
            selected_col = col_map[name.lower()]
            break

    if selected_col is None:
        # Fall back to the first column with at least one six-digit-looking value.
        for col in table.columns:
            sample = table[col].dropna().astype(str).head(50)
            if sample.str.contains(r"\d{6}", regex=True).any():
                selected_col = col
                break

    if selected_col is None:
        raise KeyError(
            "Could not find a stock-code column in the pool file. "
            f"Available columns: {list(table.columns)}"
        )

    codes: set[str] = set()
    for value in table[selected_col].dropna().tolist():
        code = normalise_ts_code(value)
        if code:
            codes.add(code)

    if not codes:
        raise ValueError(f"No valid stock codes were parsed from {path}")
    return codes


def filter_stock_pool(df: pd.DataFrame, pool_codes: Iterable[str], name: str = "data") -> pd.DataFrame:
    """Filter a dataframe with a ts_code column to the requested stock pool."""
    pool = {normalise_ts_code(c) for c in pool_codes}
    pool = {c for c in pool if c}
    if not pool:
        raise ValueError("stock pool is empty after normalisation")
    if "ts_code" not in df.columns:
        raise KeyError(f"{name} dataframe has no ts_code column")

    out = df.copy()
    out["_norm_ts_code"] = out["ts_code"].map(normalise_ts_code)
    before_rows = len(out)
    before_stocks = out["_norm_ts_code"].nunique(dropna=True)
    out = out[out["_norm_ts_code"].isin(pool)].drop(columns=["_norm_ts_code"])
    after_stocks = out["ts_code"].nunique(dropna=True)
    print(
        f"Filtered {name} by stock pool: {before_rows:,} -> {len(out):,} rows; "
        f"stocks {before_stocks:,} -> {after_stocks:,}."
    )
    missing_in_data = sorted(pool - set(df["ts_code"].map(normalise_ts_code).dropna().unique()))
    if missing_in_data:
        print(f"[WARN] {len(missing_in_data)} pool stocks were not found in {name}. Example: {missing_in_data[:10]}")
    return out.reset_index(drop=True)
