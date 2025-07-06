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
            price_series = group['Close'].dropna()
            dca_closes = price_series.resample(self.frequency).first().dropna()
            dca_df = pd.DataFrame({
                'Close': dca_closes,
                'Buy_amount': self.amount_per_asset,
            })
            dca_df['Units_bought'] = dca_df['Buy_amount'] / dca_df['Close']
            dca_df['Cumulative_units'] = dca_df['Units_bought'].cumsum()
            dca_df['Total_invested'] = dca_df['Buy_amount'].cumsum()
            dca_df['Portfolio_value'] = dca_df['Cumulative_units'] * dca_df['Close']
            dca_df['PnL'] = dca_df['Portfolio_value'] - dca_df['Total_invested']
            dca_df['Return_pct'] = (dca_df['Portfolio_value'] / dca_df['Total_invested'] -1 ) * 100
            dca_df['Avg_cost'] = dca_df['Total_invested'] / dca_df['Cumulative_units']
            dca_df['Ticker'] = ticker
            results.append(dca_df.reset_index())

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

