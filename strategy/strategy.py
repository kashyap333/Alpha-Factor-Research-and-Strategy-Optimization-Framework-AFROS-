import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ewma_momentum_signals(price_df, span=60, threshold=0.001, min_days_above_thresh=5):
    """
    Compute EWMA momentum signals with persistence filter, avoiding lookahead bias.
    Args:
        price_df (DataFrame): Long-format with Date index and Symbol column.
    Returns:
        momentum_df (DataFrame): EWMA momentum scores (wide).
        signal_df (DataFrame): Trade signals: 1 = long, -1 = short, 0 = hold (wide).
    """
    # Pivot to wide format for per-symbol EWMA
    prices = price_df.pivot(columns="Symbol", values="Close")

    # Shift prices to avoid lookahead bias
    shifted_prices = prices.shift(1)
    log_returns = np.log(prices / shifted_prices)

    # EWMA momentum
    momentum_df = log_returns.ewm(span=span, adjust=False).mean()

    # Boolean masks for positive/negative momentum
    pos_momentum = (momentum_df > threshold).astype(int)
    neg_momentum = (momentum_df < -threshold).astype(int)

    # Rolling persistence filter
    pos_count = pos_momentum.rolling(window=span).sum()
    neg_count = neg_momentum.rolling(window=span).sum()

    long_signal = (pos_count >= min_days_above_thresh).astype(int)
    short_signal = (neg_count >= min_days_above_thresh).astype(int)

    signal_df = long_signal - short_signal  # 1 = long, -1 = short, 0 = hold

    return momentum_df, signal_df


def simple_moving_average(price_df, short_window=20, long_window=90):
    """
    Compute SMA signals avoiding lookahead bias (uses previous day's moving averages).
    Returns wide-format signal DataFrame.
    """
    # Pivot to wide format
    prices = price_df.pivot(columns="Symbol", values="Close")

    # Rolling moving averages, shifted to avoid lookahead bias
    sma_short = prices.rolling(window=short_window).mean().shift(1)
    sma_long = prices.rolling(window=long_window).mean().shift(1)

    # Initialize signal DataFrame
    signal_df = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    # Generate signals
    signal_df[sma_short > sma_long] = 1
    signal_df[sma_short < sma_long] = -1

    return sma_short, sma_long, signal_df
