from abc import ABC, abstractmethod
from tarfile import data_filter

import pandas as pd
import json
from src.backtesting.metrics import log_summary


class StrategyBase(ABC):

    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the strategy and return a DataFrame with trades/metrics."""
        raise NotImplementedError

    def reset(self):
        """Optional: reset internal state (for stateful strategies or walk-forward)."""
        pass

    def update(self, new_data: pd.DataFrame):
        """Optional: update internal model with new data (for online/ML strategies)."""
        pass

    def get_params(self) -> dict:
        """Returns parameters used by the strategy."""
        return {}

    def set_params(self, **kwargs):
        """Sets parameters dynamically."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def process_trades(self, df:pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df = pd.concat((df, trades["Trade_units"]), axis=1).sort_values(['Ticker', 'Date']).reset_index(drop=True)

        results = []
        for ticker, group in df.groupby('Ticker'):

            price_series = group['Close'].dropna()
            trade_units = group["Trade_units"]

            result_ticker = pd.DataFrame({
                'Date': group['Date'],
                'Close': group['Close'],
                'Units_bought': group['Trade_units'],
            })

            result_ticker['Buy_amount'] = result_ticker["Units_bought"] * result_ticker["Close"]
            result_ticker['Cumulative_units'] = result_ticker['Units_bought'].cumsum()
            result_ticker['Total_invested'] = result_ticker['Buy_amount'].cumsum()
            result_ticker['Portfolio_value'] = result_ticker['Cumulative_units'] * result_ticker['Close']
            result_ticker['PnL'] = result_ticker['Portfolio_value'] - result_ticker['Total_invested']
            result_ticker['Return_pct'] = result_ticker['PnL']  / result_ticker['Total_invested'] * 100
            result_ticker['Avg_cost'] = result_ticker['Total_invested'] / result_ticker['Cumulative_units']
            result_ticker['Ticker'] = ticker

            results.append(result_ticker.reset_index())

        return pd.concat(results).sort_values(['Ticker', 'Date']).reset_index(drop=True)

    def backtest(self, df: pd.DataFrame, backtest_runner=None, verbose=False, log_path=None, **kwargs) -> pd.DataFrame:

        if backtest_runner is None:
            raise ValueError("Please provide a backtest runner.")

        trades = backtest_runner(df, self, **kwargs)

        result = self.process_trades(df, trades)

        if verbose:
            if log_path:
                log_summary(result, log_path=log_path)
            else:
                log_summary(result)

        return result

    def to_json(self) -> str:
        """
        Serialize the strategy parameters to JSON.
        """
        return json.dumps(self.get_params(), indent=2)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'Signal' column with 1 (buy), -1 (sell), or 0 (hold)."""
        pass

    def apply_fees(trade_value: float, flat_fee=0.0, pct_fee=0.0) -> float:
        return trade_value - flat_fee - trade_value * pct_fee

    def apply_slippage(price: float, slippage_pct=0.001, side="buy") -> float:
        if side == "buy":
            return price * (1 + slippage_pct)
        elif side == "sell":
            return price * (1 - slippage_pct)
        return price

