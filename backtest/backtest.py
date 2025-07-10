import pandas as pd

def backtest_portfolio_holding_period(weights_df, price_df, holding_period=60):
    """
    Backtest portfolio returns with a fixed holding period.

    Args:
        weights_df (DataFrame): Daily portfolio weights (dates x tickers).
        price_df (DataFrame): Daily price data (dates x tickers).
        holding_period (int): Holding period in days.

    Returns:
        Series: Portfolio returns aligned with price_df.
    """
    returns = price_df.pct_change().shift(-1).fillna(0)
    weights_df = weights_df.shift(1).reindex_like(returns).fillna(0)

    portfolio_returns = pd.Series(0.0, index=returns.index)

    # Rebalance every `holding_period` days
    rebalance_dates = weights_df.index[::holding_period]

    for rebalance_date in rebalance_dates:
        if rebalance_date not in weights_df.index:
            continue

        w = weights_df.loc[rebalance_date].values
        start_idx = returns.index.get_loc(rebalance_date)

        # Determine end of holding period
        end_idx = min(start_idx + holding_period, len(returns))
        window = returns.iloc[start_idx:end_idx]

        # Apply same weights across holding period
        pnl = window @ w
        portfolio_returns.iloc[start_idx:end_idx] = pnl.values

    return portfolio_returns


