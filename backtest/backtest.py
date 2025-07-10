def backtest_weighted_signal_strategy(price_df, weights_df, signal_df=None, trading_days=252, risk_free_rate=0.0):
    """
    Generic backtest function combining weights and optional trade signals,
    with performance metrics calculated.

    Args:
        price_df (DataFrame): Price data (dates x tickers).
        weights_df (DataFrame): Raw portfolio weights (already time-aligned).
        signal_df (DataFrame, optional): Binary trade signals (1 = buy, else 0).
        trading_days (int): Number of trading days per year for annualization.
        risk_free_rate (float): Annual risk-free rate, e.g., 0.0 or 0.01.

    Returns:
        dict with keys:
            'portfolio_value' (Series): Cumulative portfolio value.
            'daily_returns' (Series): Daily portfolio returns.
            'weights' (DataFrame): Final signal-adjusted weights.
            'metrics' (dict): Performance metrics (cumulative return, Sharpe, max drawdown).
    """
    # Apply signals if given
    if signal_df is not None:
        masked_weights = weights_df.where(signal_df == 1, 0.0)
        final_weights = masked_weights.div(masked_weights.sum(axis=1), axis=0).fillna(0)
    else:
        final_weights = weights_df.div(weights_df.sum(axis=1), axis=0).fillna(0)

    asset_returns = price_df.pct_change().reindex(final_weights.index)
    daily_returns = (final_weights * asset_returns).sum(axis=1)

    portfolio_value = (1 + daily_returns).cumprod()
    portfolio_value.iloc[0] = 1

    metrics = performance_metrics(daily_returns, trading_days, risk_free_rate)

    return {
        'portfolio_value': portfolio_value,
        'daily_returns': daily_returns,
        'weights': final_weights,
        'metrics': metrics
    }