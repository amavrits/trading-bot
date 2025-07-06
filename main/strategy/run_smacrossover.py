import pandas as pd
from src.strategy.sma import StrategySMACrossover
from src.backtesting.core import run_strategy
from src.backtesting.metrics import *
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == "__main__":

    data_path = Path(r"../../data/compiled_data.parquet")
    result_path = Path(r"../../results/strategy/sma")
    result_path.mkdir(parents=True, exist_ok=True)
    save_path = result_path / "parquets"
    save_path.mkdir(parents=True, exist_ok=True)
    plot_path = result_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    log_path = result_path / "logs"

    df = pd.read_parquet(data_path)

    strategy = StrategySMACrossover(short_window=20, long_window=50,)
    result = strategy.backtest(df, backtest_runner=run_strategy, verbose=True, log_path=log_path)

    result_sum = compute_total_pnl(result)

    for ticker, result_ticker in result.groupby("Ticker"):
        result_ticker.to_parquet(save_path/f"strategy_{ticker}.parquet", index=True)
    result.to_parquet(save_path/"strategy_ALL.parquet", index=True)
    result_sum.to_parquet(save_path/"strategy_SUM.parquet", index=True)

    fig = plt.figure(figsize=(8, 6))
    for ticker, result_ticker in result.groupby("Ticker"):
        plt.plot(result_ticker["Date"], result_ticker["Return_pct"], label=ticker)
    plt.plot(result_sum["Date"], result_sum["Return_pct"], label="SUM", color="r")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Return [%]", fontsize=12)
    plt.xticks(rotation=45)
    plt.suptitle(ticker, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid()
    plt.close()
    fig.savefig(plot_path/"returns_ALL.png")

