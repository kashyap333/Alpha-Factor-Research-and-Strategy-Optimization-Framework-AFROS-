import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ewma_momentum_signals(price_df, span=60, threshold=0.001, min_days_above_thresh=5):
    # Copy to avoid mutating original df
    df = price_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'Symbol'])
    df = df.set_index('Date')

    prices = df.pivot(columns="Symbol", values="Close")

    shifted_prices = prices.shift(1)
    log_returns = np.log(prices / shifted_prices)

    momentum_df = log_returns.ewm(span=span, adjust=False).mean()

    pos_momentum = (momentum_df > threshold).astype(int)
    neg_momentum = (momentum_df < -threshold).astype(int)

    pos_count = pos_momentum.rolling(window=span).sum()
    neg_count = neg_momentum.rolling(window=span).sum()

    long_signal = (pos_count >= min_days_above_thresh).astype(int)
    short_signal = (neg_count >= min_days_above_thresh).astype(int)

    signal_df = long_signal - short_signal

    return momentum_df, signal_df


def simple_moving_average(price_df, short_window=20, long_window=90):
    """
    Compute SMA signals avoiding lookahead bias (uses previous day's moving averages).
    Returns wide-format signal DataFrame.

    Args:
        price_df (pd.DataFrame): Must contain 'Date', 'Symbol', 'Close'.
        short_window (int): Window for short SMA.
        long_window (int): Window for long SMA.

    Returns:
        sma_short (pd.DataFrame): Short window SMA (shifted by 1 day).
        sma_long (pd.DataFrame): Long window SMA (shifted by 1 day).
        signal_df (pd.DataFrame): Signals: 1 for long, -1 for short, 0 for neutral.
    """

    # Check required columns
    required_cols = {'Date', 'Symbol', 'Close'}
    if not required_cols.issubset(price_df.columns):
        raise ValueError(f"Input DataFrame must contain {required_cols}")

    # Sort by Date and Symbol
    price_df_sorted = price_df.sort_values(by=['Date', 'Symbol']).copy()

    # Pivot to wide format with Date index and Symbol columns
    prices = price_df_sorted.pivot(index='Date', columns='Symbol', values='Close')

    # Calculate rolling SMAs and shift by 1 day to avoid lookahead bias
    sma_short = prices.rolling(window=short_window).mean().shift(1)
    sma_long = prices.rolling(window=long_window).mean().shift(1)

    # Initialize signals with 0
    signal_df = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    # Generate signals
    signal_df[sma_short > sma_long] = 1
    signal_df[sma_short < sma_long] = -1

    return sma_short, sma_long, signal_df
