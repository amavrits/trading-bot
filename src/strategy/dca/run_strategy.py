import pandas as pd
from typing import List


def run_strategy_ticker(
        df: pd.DataFrame,
        ticker: str = 'AAPL',
        start_date: str = '2020-01-01',
        end_date: str = '2024-12-31',
        investment_amount: int | float = 100,
        frequency: str = 'W-MON'  # every Monday
) -> pd.DataFrame:
    # 1. Load price data
    df = df[df['Ticker'] == ticker]
    df.index = pd.to_datetime(df["Date"])
    df = df[['Close']].dropna()

    # 2. Resample to desired frequency (Monday close prices)
    dca = df.resample(frequency).first().dropna()

    # 3. Run DCA simulation
    dca['Units'] = investment_amount / dca['Close']
    dca['Invested'] = investment_amount
    dca['Total Units'] = dca['Units'].cumsum()
    dca['Total Invested'] = dca['Invested'].cumsum()
    dca['Portfolio Value'] = dca['Total Units'] * dca['Close']
    dca['PnL'] = (dca['Portfolio Value'] / dca['Total Invested'] - 1)

    columns = ["Date"] + list(dca.columns)  # Re-order columns
    dca["Date"] = dca.index
    dca = dca[columns]
    dca = dca.reset_index(drop=True)

    return dca


def run_strategy(
        df: pd.DataFrame,
        tickers: List[str] = ['AAPL'],
        start_date: str = '2020-01-01',
        end_date: str = '2024-12-31',
        investment_amount: int | float = 100,
        frequency: str = 'W-MON'  # every Monday
) -> pd.DataFrame:

    dcas = []
    for ticker in tickers:
        dca = run_strategy_ticker(
            df=df,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            investment_amount=investment_amount,
            frequency=frequency
        )
        dca["Ticker"] = ticker
        dcas.append(dca)

    dca = pd.concat(dcas, axis=0, ignore_index=True)

    return dca


def sum_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df_sum = df.groupby(df["Date"]).agg({
        'Total Invested': 'sum',
        'Portfolio Value': 'sum'
    })
    df_sum['PnL'] = (df_sum['Portfolio Value'] / df_sum['Total Invested'] - 1)
    columns = ["Date"] + list(df_sum.columns)  # Re-order columns
    df_sum["Date"] = df_sum.index
    df_sum = df_sum[columns]
    df_sum = df_sum.reset_index(drop=True)
    df_sum["Ticker"] = "SUM"
    return df_sum


if __name__ == "__main__":

    pass

