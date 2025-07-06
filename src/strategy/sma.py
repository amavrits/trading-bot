import pandas as pd
import numpy as np
from typing import List
from src.strategy.base import StrategyBase
from src.backtesting.metrics import log_summary


class StrategySMACrossover(StrategyBase):

    def __init__(
            self,
            short_window=20,
            long_window=50,
            assets: list[str] = None
    ):
        super().__init__()  # sets .name from class name
        self.short_window = short_window
        self.long_window = long_window
        self.assets = assets

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['SMA_short'] = df['Close'].rolling(self.short_window, min_periods=self.short_window).mean()
        df['SMA_long'] = df['Close'].rolling(self.long_window, min_periods=self.long_window).mean()
        df['Signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
        df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = -1
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

            units = np.where(signal["Signal"] > 0, +1, np.where(signal["Signal"]<0, -1, 0))
            cum_units = units.cumsum()
            cum_units = np.clip(cum_units, a_min=0, a_max=None)
            units = np.where(  # Can't sell if no stock has been bought
                np.logical_and(units==-1, cum_units<=0),
                0,
                units
            )

            trade_units_ticker = pd.DataFrame(index=group.index)
            trade_units_ticker["Date"] = group["Date"]
            trade_units_ticker["Trade_units"] = units
            trade_units_ticker["Ticker"] = ticker

            trade_units.append(trade_units_ticker.reset_index(drop=True))

        return pd.concat(trade_units).sort_values(['Ticker', 'Date']).reset_index(drop=True)


if __name__ == "__main__":

    pass

