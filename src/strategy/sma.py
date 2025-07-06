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
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df = df.sort_index()

        if self.assets is None:
            self.assets = pd.unique(df["Ticker"])

        if self.assets is not None:
            df = df[df['Ticker'].isin(self.assets)]

        df.index = pd.to_datetime(df["Date"])

        results = []
        for ticker, group in df.groupby('Ticker'):

            signal = self.generate_signals(group)

            units_bought = np.where(signal["Signal"] > 0, +1, np.where(signal["Signal"]<0, -1, 0))
            cum_units_bought = units_bought.cumsum()
            cum_units_bought = np.clip(cum_units_bought, a_min=0, a_max=None)
            units_bought = np.where(  # Can't sell if no stock has been bought
                np.logical_and(units_bought==-1, cum_units_bought<=0),
                0,
                units_bought
            )

            rsi_ticker = pd.DataFrame()
            rsi_ticker['Date'] = signal['Date']
            rsi_ticker['Close'] = signal['Close']
            rsi_ticker['Units_bought'] = units_bought
            rsi_ticker['Buy_amount'] = rsi_ticker['Close'] * rsi_ticker['Units_bought']
            rsi_ticker['Cumulative_units'] = rsi_ticker['Units_bought'].cumsum()
            rsi_ticker['Total_invested'] = rsi_ticker["Buy_amount"].cumsum()
            rsi_ticker['Portfolio_value'] = rsi_ticker['Close'] * rsi_ticker['Cumulative_units']
            rsi_ticker['PnL'] = rsi_ticker['Portfolio_value'] - rsi_ticker['Total_invested']
            rsi_ticker['Return_pct'] = np.where(
                rsi_ticker['Total_invested'] > 0,
                (rsi_ticker['Portfolio_value'] / rsi_ticker['Total_invested'] -1 ) * 100,
                0
            )
            rsi_ticker['Avg_cost'] = np.where(
                rsi_ticker['Cumulative_units'] > 0,
                rsi_ticker['Total_invested'] / rsi_ticker['Cumulative_units'],
                0
            )
            rsi_ticker['Ticker'] = ticker

            results.append(rsi_ticker.reset_index(drop=True))

        return pd.concat(results).sort_values(['Ticker', 'Date']).reset_index(drop=True)

    def backtest(self, df: pd.DataFrame, backtest_runner=None, verbose=False, log_path=None, **kwargs) -> pd.DataFrame:

        if backtest_runner is None:
            raise ValueError("Please provide a backtest runner.")

        result = backtest_runner(df, self, **kwargs)

        if verbose:
            if log_path:
                log_summary(result, log_path=log_path)
            else:
                log_summary(result)

        return result


if __name__ == "__main__":

    pass

