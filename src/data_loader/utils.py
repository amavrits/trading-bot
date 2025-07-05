import pandas as pd
from pathlib import Path
import yfinance as yf
import hashlib
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Optional, Tuple, Literal


def _make_cache_key(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    suffix: str = "",
) -> str:
    key = "_".join(tickers) + f"_{start}_{end}_{interval}{suffix}"
    return hashlib.md5(key.encode()).hexdigest()


def load_stock_data(
        tickers: List[str],
        start: Optional[str],
        end: Optional[str],
        interval: str = "1d",
        features: Tuple[str] = ("Open", "High", "Low", "Close", "Volume"),
        dropna: bool = True,
        cache_dir: str | Path = "data/cache",
        use_cache: bool = True,
        verbose=True
) -> pd.DataFrame:

    if not isinstance(cache_dir, Path): cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _make_cache_key(tickers, start, end, interval)
    cache_path = cache_dir / f"{cache_key}.parquet"

    if use_cache and cache_path.is_file():
        if verbose: print(f"✅ Loaded cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    # --- fallback to yfinance ---
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
    )

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex for multiple tickers.")

    all_dfs = []
    for ticker in tickers:
        if (ticker, "Close") not in df.columns:
            continue

        df_ticker = df[ticker].copy()
        df_ticker["Ticker"] = ticker
        df_ticker["Date"] = df.index
        if dropna:
            df_ticker = df_ticker.dropna(subset=features)

        df_ticker = df_ticker[["Date", *features, "Ticker"]]
        all_dfs.append(df_ticker)

    df_all = pd.concat(all_dfs, axis=0, ignore_index=True)
    df_all.to_parquet(cache_path)
    if verbose: print(f"💾 Cached data to {cache_path}")

    return df_all


def generate_dates(
    start: str,
    end: str,
    step: Literal["daily", "monthly"] = "daily"
) -> List[str]:
    """
    Generate dates between start and end (inclusive) in 'YYYY-MM-DD' format.

    Args:
        start: Start date string ('YYYY-MM-DD')
        end: End date string ('YYYY-MM-DD')
        step: 'daily' or 'monthly'

    Returns:
        List of date strings
    """
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    dates = []
    current = start_date

    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        if step == "daily":
            current += timedelta(days=1)
        elif step == "monthly":
            current += relativedelta(months=1)
        else:
            raise ValueError(f"Unsupported step: {step}")

    return dates


def compile_datasets(path: str | Path):

    if not isinstance(path, Path): path = Path(path)

    dfs = []
    for file in path.iterdir():
        df = pd.read_parquet(file)
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    out_path = path.parent
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path/"comipled_data.parquet", index=False)

    return df




if __name__ == "__main__":

    pass

