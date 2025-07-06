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
            price_series = group['Close']

            units_bought = np.where(signal["Signal"] > 0, 1, 0)
            cum_units_bought = units_bought.cumsum()
            units_bought = np.where(  # only buy once
                cum_units_bought > 1,
                0,
                units_bought
            )

            bh_ticker = pd.DataFrame()
            bh_ticker['Date'] = price_series.index
            bh_ticker['Close'] = price_series.values
            bh_ticker['Units_bought'] = units_bought
            bh_ticker['Buy_amount'] = bh_ticker['Close'] * bh_ticker['Units_bought']
            bh_ticker['Cumulative_units'] = bh_ticker['Units_bought'].cumsum()
            bh_ticker['Total_invested'] = bh_ticker["Buy_amount"].cumsum()
            bh_ticker['Portfolio_value'] = bh_ticker['Close'] * bh_ticker['Cumulative_units']
            bh_ticker['PnL'] = bh_ticker['Portfolio_value'] - bh_ticker['Total_invested']
            bh_ticker['Return_pct'] = np.where(
                bh_ticker['Total_invested'] > 0,
                (bh_ticker['Portfolio_value'] / bh_ticker['Total_invested'] - 1) * 100,
                0
            )
            bh_ticker['Avg_cost'] = np.where(
                bh_ticker['Cumulative_units'] > 0,
                bh_ticker['Total_invested'] / bh_ticker['Cumulative_units'],
                0
            )
            bh_ticker['Ticker'] = ticker

            results.append(bh_ticker.reset_index(drop=True))

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

