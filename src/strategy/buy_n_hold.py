import pandas as pd
import numpy as np
from typing import List
from src.strategy.base import StrategyBase
from src.backtesting.metrics import log_summary


class StrategyBuynHold(StrategyBase):

    def __init__(self, assets: list[str] = None):
        super().__init__()  # sets .name from class name
        self.assets = assets

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Signal'] = 0
        if not df.empty:
            df.at[df.index[0], 'Signal'] = 1  # Buy at the first available date
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
            trade_units_ticker["Trade_units"] = np.where(signal["Signal"] > 0, 1, 0)
            trade_units_ticker["Ticker"] = ticker

            trade_units.append(trade_units_ticker.reset_index(drop=True))

        return pd.concat(trade_units).sort_values(['Ticker', 'Date']).reset_index(drop=True)


if __name__ == "__main__":

    pass

