import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def track_portfolio_performance(
    price_df,
    entry_date,
    shares,
    invested_capital,
    log_path="portfolio_performance_log.csv",
    overwrite_log=False
):
    """
    Tracks daily portfolio performance from 7 days before entry_date to today.
    Avoids re-logging already tracked dates.

    Args:
        price_df (pd.DataFrame): Price data (index = Date, columns = Symbols).
        entry_date (datetime.date or str): Date when shares were bought.
        shares (pd.Series): Shares bought per symbol (index = symbol).
        invested_capital (float): Total capital spent.
        log_path (str): File path to save performance log.
        overwrite_log (bool): If True, deletes existing log and starts fresh.

    Returns:
        pd.DataFrame: Updated performance log.
    """
    # Ensure datetime handling
    price_df.index = pd.to_datetime(price_df.index)
    entry_date = pd.to_datetime(entry_date).date()
    start_date = entry_date - timedelta(days=20)
    today = datetime.today().date()

    # Filter trading dates
    trading_dates = sorted([d for d in price_df.index.date if start_date <= d <= today])
    if not trading_dates:
        raise ValueError("No valid trading dates in the specified range.")

    # Slice price data for relevant dates and symbols
    symbols = shares.index
    price_slice = price_df.loc[price_df.index.date >= start_date, symbols]
    price_slice = price_slice[price_slice.index.date <= today]

    portfolio_values = (price_slice * shares).sum(axis=1)

    daily_profit = portfolio_values.diff().fillna(0)  # $ change vs previous day
    cumulative_profit = portfolio_values - invested_capital  # $ total profit since entry

    daily_return = portfolio_values.pct_change().fillna(0)  # % change vs previous day
    cumulative_return = cumulative_profit / invested_capital  # total % return since entry
    
    performance_df = pd.DataFrame({
        "date": portfolio_values.index.date,
        "portfolio_value": portfolio_values.values,
        "daily_profit": daily_profit.values,
        "cumulative_profit": cumulative_profit.values,
        "daily_return": daily_return.values,
        "cumulative_return": cumulative_return.values
    })

    # Handle logging
    if overwrite_log and Path(log_path).exists():
        Path(log_path).unlink()

    if Path(log_path).exists():
        old_df = pd.read_csv(log_path, parse_dates=["date"])
        old_df["date"] = pd.to_datetime(old_df["date"]).dt.date

        # Avoid duplicate dates
        new_df = performance_df[~performance_df["date"].isin(old_df["date"])]
        performance_df = pd.concat([old_df, new_df], ignore_index=True)
        performance_df.sort_values("date", inplace=True)
        print(f"{len(new_df)} new rows logged.")
    else:
        print(f"{len(performance_df)} rows logged (new log).")

    # Save updated log
    performance_df.to_csv(log_path, index=False)
    print(f"Performance log updated. Last date: {performance_df['date'].max()}")
    
    return performance_df