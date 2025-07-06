import pandas as pd
from src.data_loader.dataloader import StockDataset
from src.data_loader.utils import *
from typing import Literal, List
from tqdm import tqdm


if __name__ == "__main__":

    dates = generate_dates("2021-01-01", "2025-07-01", step="monthly")

    dfs = []
    for (start_date, end_date) in tqdm(zip(dates[:-1], dates[1:]), desc="Downloading month"):
        df = load_stock_data(
            tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY"],
            start=start_date,
            end=end_date,
            interval="1d",
            cache_dir=r"../../data/yf_data",
            verbose=False
        )
        dfs.append(df)

    df = compile_datasets(r"../../data/yf_data")

