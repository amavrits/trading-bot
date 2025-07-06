import pandas as pd
from src.backtesting.metrics import compare_strategies
from src.strategy.dca import StrategyDCA
from src.strategy.rsi import StrategyRSI
from src.strategy.sma import StrategySMACrossover
from src.strategy.buy_n_hold import StrategyBuynHold
from pathlib import Path


if __name__ == "__main__":

    data_path = Path(r"../../data/compiled_data.parquet")
    out_path = Path(r"../../results/strategy/comparison")
    out_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_parquet(data_path)

    strategies = {
        "DCA": StrategyDCA(amount_per_asset=100, frequency='W'),
        "RSI": StrategyRSI(period=14, buy_threshold=30, sell_threshold=70),
        "SMA": StrategySMACrossover(short_window=20, long_window=50),
        "BuyHold": StrategyBuynHold()
    }

    details, global_summary, best_strategy = compare_strategies(strategies, df)

    print("📊 Global Strategy Summary:")
    print(global_summary)

    print(f"\n🏆 Best Global Strategy: {best_strategy}")

    global_summary.to_csv(out_path/"comparison.csv")
