import pandas as pd
from pathlib import Path
import datetime


def compute_total_pnl(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the flat DCA result DataFrame with ['Ticker', 'date', 'PnL', ...],
    returns a new DataFrame with total portfolio metrics across all assets.
    """
    df = result_df.copy()
    total_df = (
        df.groupby('Date')[['PnL', 'Total_invested', 'Portfolio_value', 'Return_pct']]
        .sum()
        .reset_index()
    )
    total_df['Ticker'] = 'TOTAL'  # for consistent plotting
    return total_df


def log_summary(result_df: pd.DataFrame, include_total=True, log_path=None):
    """
    Logs a summary of final DCA metrics per Ticker (and total).

    Parameters:
    - result_df: flat DataFrame with 'Ticker', 'date', 'total_invested', 'portfolio_value', 'PnL'
    - include_total: whether to add a 'TOTAL' row by summing across tickers
    - return_df: whether to return the summary DataFrame

    Returns:
    - pd.DataFrame (optional): summary stats
    """
    last = (
        result_df.sort_values('Date')
        .groupby('Ticker')
        .last()
        .reset_index()
    )

    summary = last[['Ticker', 'Total_invested', 'Portfolio_value', 'PnL']].copy()
    summary['Return_pct'] = 100 * summary['PnL'] / summary['Total_invested']

    if include_total:
        total_row = pd.DataFrame({
            'Ticker': ['TOTAL'],
            'Total_invested': [summary['Total_invested'].sum()],
            'Portfolio_value': [summary['Portfolio_value'].sum()],
            'PnL': [summary['PnL'].sum()],
        })
        total_row['Return_pct'] = 100 * total_row['PnL'] / total_row['Total_invested']
        summary = pd.concat([summary, total_row], ignore_index=True)

    print("\nStrategy Summary:")
    print(summary.to_string(index=False, float_format='%.2f'))

    if log_path:
        if not isinstance(log_path, Path): log_path = Path(log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"summary_{ts}.csv"
        summary.to_csv(log_file, index=False)

