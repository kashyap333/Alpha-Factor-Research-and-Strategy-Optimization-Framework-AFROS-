import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta


def get_trade_entry_exit_dates(begin_new_trading_period=True, holding_period=60):
    if not begin_new_trading_period:
        return None, None
    
    nyse = mcal.get_calendar('NYSE')
    today = datetime.today().date()
    
    # Get trading schedule for next 30 days starting today
    schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=60 + 30))
    trading_days = list(schedule.index.date)
    
    # Find next trading day >= today for entry
    next_entry_dates = [d for d in trading_days if d >= today]
    if not next_entry_dates:
        raise ValueError("No upcoming trading days found for entry")
    entry_date = next_entry_dates[0]
    
    # Find the ideal exit date (entry + holding_period - 1 days)
    desired_exit_date = entry_date + timedelta(days=holding_period - 1)
    
    # Find all trading days >= entry_date and <= desired_exit_date
    candidate_exit_days = [d for d in trading_days if entry_date <= d <= desired_exit_date]
    
    if not candidate_exit_days:
        # No trading days between entry and desired exit, move backward from desired_exit_date
        exit_date = None
        for delta in range(holding_period - 1, -1, -1):
            candidate_date = entry_date + timedelta(days=delta)
            if candidate_date in trading_days:
                exit_date = candidate_date
                break
        if exit_date is None:
            raise ValueError("No valid exit trading day found")
    else:
        # The latest trading day on or before desired_exit_date
        exit_date = candidate_exit_days[-1]
    
    return entry_date, exit_date

def get_previous_trading_day(date):
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=date - timedelta(days=10), end_date=date)
    trading_days = list(schedule.index.date)
    for d in reversed(trading_days):
        if d < date:
            return pd.to_datetime(d)

def prepare_trade_allocation(entry_date, price_df, filtered_kelly_weights, capital=100_000):
    """
    Prepare trade allocations for the upcoming entry_date.
    Assumes this is run *after market close* on the day before entry_date.
    
    Args:
        entry_date (date): The date you plan to enter (buy) positions next market open.
        price_df (DataFrame): Price data with Date index and Symbols as columns (daily close).
        filtered_kelly_weights (DataFrame): Weights filtered and normalized by date and symbol.
        capital (float): Total capital to allocate.

    Returns:
        dict: {
            'entry_date': entry_date,
            'allocation_date': allocation_date,  # day before entry_date
            'weights': weights,
            'prices': entry_prices,
            'shares': shares,
            'capital_allocated': invested_capital
        }
    """
    allocation_date = get_previous_trading_day(entry_date)
    
    # Validate dates
    if allocation_date not in filtered_kelly_weights.index:
        raise ValueError(f"Allocation date {allocation_date} not found in filtered_kelly_weights")

    
    # Get weights from day before entry
    weights = filtered_kelly_weights.loc[allocation_date]
    weights = weights[weights > 0]  # only positive weights
    
    if weights.sum() == 0:
        raise ValueError(f"No positive weights on allocation date {allocation_date}")
    
    weights /= weights.sum()  # Normalize
    
    # Get prices on entry date (next open prices assumed equal to close previous day for simplicity)
    # You can replace with actual open prices if available
    entry_prices = price_df.loc[allocation_date, weights.index]
    
    # Calculate shares to buy (integer shares)
    shares = (capital * weights / entry_prices).fillna(0).astype(int)
    
    invested_capital = (shares * entry_prices).sum()
    
    data = pd.DataFrame({
        'entry_date': entry_date,
        'allocation_date': allocation_date,
        'shares': shares,
        'symbols': shares.index,
        'capital_allocated': invested_capital
    })
    
    data.to_csv('portfolio_allocation.csv',index=False)
    
    return entry_date, allocation_date, weights,entry_prices,shares,invested_capital
