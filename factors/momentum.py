import pandas as pd
import os
import numpy as np


def load_price_data():
    filepath = os.path.join('D:\\Quant\\Data', f"master_stock_data.csv")
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Pivot long format to wide: index = Date, columns = Symbol, values = Close
    prices = df.pivot(index='Date', columns='Symbol', values='Close')
    prices = prices.sort_index()
    columns= prices.columns.unique()
    columns = columns[:20]
    prices = prices[columns]
    prices = prices[prices.index > '2023-01-01']# Limit to first 20 columns for performance
    return prices


def ewma_momentum_signals(price_df, span=60, threshold=0.001, min_days_above_thresh=15):
    """
    Compute EWMA momentum and generate trade signals based on persistence.

    Args:
        price_df (DataFrame): Price data (dates x tickers).
        span (int): EWMA span.
        threshold (float): Momentum threshold to count as 'positive'.
        min_days_above_thresh (int): Min # of days momentum must be above threshold in lookback window.

    Returns:
        momentum_df (DataFrame): EWMA momentum scores.
        signal_df (DataFrame): Binary trade signals (1 = trade, 0 = ignore).
    """
    log_returns = np.log(price_df / price_df.shift(1))
    momentum_df = log_returns.ewm(span=span, adjust=False).mean().shift(1)

    # Boolean mask where momentum is above threshold
    positive_momentum = (momentum_df > threshold).astype(int)

    # Rolling count of how many times momentum was above threshold
    persistence_count = positive_momentum.rolling(window=span).sum()

    # Signal only if count exceeds minimum required
    signal_df = (persistence_count >= min_days_above_thresh).astype(int)

    return momentum_df, signal_df