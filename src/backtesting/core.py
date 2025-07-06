import pandas as pd
from typing import Type
from src.strategy.base import StrategyBase

def run_strategy(
        df: pd.DataFrame,
        strategy: Type[StrategyBase],
        **kwargs
) -> pd.DataFrame:
    """
    Run a strategy instance on the given DataFrame.
    Assumes the strategy has a `.run(df)` method (from StrategyBase).
    """
    return strategy.run(df, **kwargs)

