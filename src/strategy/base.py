from abc import ABC, abstractmethod
import pandas as pd
import json


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

    def backtest(self, df: pd.DataFrame, backtest_runner=None, **kwargs) -> pd.DataFrame:
        """
        Run the strategy through a backtest framework.
        If `backtest_runner` is provided, it should accept a strategy instance and data.
        """
        if backtest_runner is None:
            raise ValueError("Please provide a backtest runner (e.g., from backtest.core)")
        return backtest_runner(df, self, **kwargs)

    def to_json(self) -> str:
        """
        Serialize the strategy parameters to JSON.
        """
        return json.dumps(self.get_params(), indent=2)


