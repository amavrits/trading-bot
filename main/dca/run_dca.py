import pandas as pd
from src.strategy.dca.run_strategy import run_strategy, sum_strategy
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == "__main__":

    data_path = Path(r"../../data/comipled_data.parquet")
    result_path = Path(r"../../results/stategy/dca")
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)

    dca = run_strategy(
        df=df,
        tickers=list(pd.unique(df["Ticker"])),
        start_date='2020-01-01',
        end_date='2024-12-31',
        investment_amount=100,
        frequency='W-MON'
    )

    dca_sum = sum_strategy(dca)

    for ticker, dca_ticker in dca.groupby("Ticker"):
        dca_ticker.to_parquet(result_path/f"strategy_{ticker}.parquet", index=True)
    dca.to_parquet(result_path/"strategy_ALL.parquet", index=True)
    dca_sum.to_parquet(result_path/"strategy_SUM.parquet", index=True)

    fig = plt.figure(figsize=(8, 6))
    for ticker, dca_ticker in dca.groupby("Ticker"):
        plt.plot(dca_ticker["Date"], dca_ticker["PnL"]*100, label=ticker)
    plt.plot(dca_sum["Date"], dca_sum["PnL"]*100, label="SUM", color="r")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Profit/Loss [%]", fontsize=12)
    plt.xticks(rotation=45)
    plt.suptitle(ticker, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid()
    plt.close()
    fig.savefig(result_path/"pnl_ALL.png")

