import pandas as pd
from src.strategy.dca import StrategyDCA
from src.backtesting.core import run_strategy
from src.backtesting.metrics import *
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == "__main__":

    data_path = Path(r"../../data/comipled_data.parquet")
    result_path = Path(r"../../results/strategy/dca")
    result_path.mkdir(parents=True, exist_ok=True)
    save_path = result_path / "parquets"
    save_path.mkdir(parents=True, exist_ok=True)
    plot_path = result_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    log_path = result_path / "logs"

    df = pd.read_parquet(data_path)

    strategy = StrategyDCA(amount_per_asset=100, frequency='W')
    result = strategy.backtest(df, backtest_runner=run_strategy, verbose=True, log_path=log_path)

    result_sum = compute_total_pnl(result)

    for ticker, dca_ticker in result.groupby("Ticker"):
        dca_ticker.to_parquet(save_path/f"strategy_{ticker}.parquet", index=True)
    result.to_parquet(save_path/"strategy_ALL.parquet", index=True)
    result_sum.to_parquet(save_path/"strategy_SUM.parquet", index=True)

    fig = plt.figure(figsize=(8, 6))
    for ticker, result_ticker in result.groupby("Ticker"):
        plt.plot(result_ticker["Date"], result_ticker["PnL"]*100, label=ticker)
    plt.plot(result_sum["Date"], result_sum["PnL"]*100, label="SUM", color="r")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Profit/Loss [%]", fontsize=12)
    plt.xticks(rotation=45)
    plt.suptitle(ticker, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid()
    plt.close()
    fig.savefig(plot_path/"pnl_ALL.png")

