import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    long_signal = (persistence_count >= min_days_above_thresh).astype(int)
    short_signal = ((momentum_df < -threshold).rolling(window=span).sum() >= min_days_above_thresh).astype(int)
    signal_df = long_signal - short_signal

    return momentum_df, signal_df

def simple_moving_average(price_df, short_window=20, long_window=90):
    """
    Compute simple moving averages and generate trade signals.

    Args:
        price_df (DataFrame): Price data (dates x tickers).
        short_window (int): Short moving average window.
        long_window (int): Long moving average window.

    Returns:
        sma_short (DataFrame): Short moving averages.
        sma_long (DataFrame): Long moving averages.
        signal_df (DataFrame): Binary trade signals (1 = buy, -1 = sell, 0 = hold).
    """
    # Calculate lagged moving averages to avoid lookahead bias
    sma_short = price_df.rolling(window=short_window).mean().shift(1)
    sma_long = price_df.rolling(window=long_window).mean().shift(1)

    # Initialize signal dataframe
    signal_df = pd.DataFrame(0, index=price_df.index, columns=price_df.columns)

    # Generate signals: 1 = buy, -1 = sell, 0 = hold
    signal_df[sma_short > sma_long] = 1
    signal_df[sma_short < sma_long] = -1

    return sma_short, sma_long, signal_df