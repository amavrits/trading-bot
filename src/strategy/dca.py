import pandas as pd
from typing import List
from src.strategy.base import StrategyBase
from src.backtesting.metrics import log_summary


class StrategyDCA(StrategyBase):

    def __init__(self, amount_per_asset: float = 100., frequency: str = 'W', assets: list[str] = None):
        super().__init__()  # sets .name from class name
        self.amount_per_asset = amount_per_asset
        self.frequency = frequency
        self.assets = assets

    def _compute_trade_dates(self, df):
        grouped = df["Close"].resample(self.frequency)
        # Get the first *available* date in each weekly group
        actual_trade_dates = grouped.apply(lambda x: x.index[0])
        return set(actual_trade_dates)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:

        trade_dates = self._compute_trade_dates(df)

        df["Signal"] = 0
        for trade_date in trade_dates:
            df.at[trade_date, "Signal"] = +1

        return df["Signal"]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df = df.sort_index()

        if self.assets is None:
            self.assets = pd.unique(df["Ticker"])

        if self.assets is not None:
            df = df[df['Ticker'].isin(self.assets)]

        df.index = pd.to_datetime(df["Date"])

        trade_units = []
        for ticker, group in df.groupby('Ticker'):

            signal = self.generate_signals(group)

            trade_units_ticker = pd.DataFrame(index=group.index)
            trade_units_ticker["Date"] = group["Date"]
            trade_units_ticker["Trade_units"] = signal.values
            trade_units_ticker["Trade_units"] *= self.amount_per_asset / group["Close"]
            trade_units_ticker["Ticker"] = ticker

            trade_units.append(trade_units_ticker.reset_index(drop=True))

        return pd.concat(trade_units).sort_values(['Ticker', 'Date']).reset_index(drop=True)


if __name__ == "__main__":

    pass

