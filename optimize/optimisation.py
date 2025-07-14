import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from riskfolio.Portfolio import Portfolio
import scipy.optimize as sco
from asset_selection.selection_functions import *

def risk_parity(df, window=60, rolling=False, price_column="Close"):
    df = df.reset_index()
    price_df = df.pivot(index='Date', columns='Symbol', values=price_column)
    price_df = price_df.sort_index().sort_index(axis=1)
    returns = price_df.pct_change().dropna()

    if rolling:
        weights_list = []
        dates = []
        for i in range(window, len(returns)):
            window_data = returns.iloc[i - window:i]
            port = Portfolio(returns=window_data)
            port.assets_stats(method_mu='hist', method_cov='hist')
            w = port.rp_optimization(model='Classic', rm='MV')
            weights_list.append(w.values.flatten())
            dates.append(returns.index[i])
        return pd.DataFrame(weights_list, index=dates, columns=returns.columns)

    port = Portfolio(returns=returns.iloc[-window:])
    port.assets_stats(method_mu='hist', method_cov='hist')
    return port.rp_optimization(model='Classic', rm='MV')

def construct_kelly_portfolio(df, window=60, cap=1.0, price_column="Close", scale=False, target_vol=None):
    df = df.reset_index()
    price_df = df.pivot(index='Date', columns='Symbol', values=price_column)
    returns = price_df.pct_change().dropna()
    weights_list = []
    dates = []

    for i in range(window, len(returns)):
        ret_window = returns.iloc[i - window:i]
        mu = ret_window.mean().values
        sigma = ret_window.cov().values

        try:
            kelly_weights = np.linalg.solve(sigma, mu)
        except np.linalg.LinAlgError:
            kelly_weights = np.zeros(len(mu))

        kelly_weights = np.clip(kelly_weights, 0, cap)
        if kelly_weights.sum() > 0:
            kelly_weights /= kelly_weights.sum()

        weights_list.append(kelly_weights)
        dates.append(returns.index[i])

    weights_df = pd.DataFrame(weights_list, index=dates, columns=price_df.columns)

    if scale:
        returns_df = price_df.pct_change().dropna()
        weights_df = scale_to_target_volatility(weights_df, returns_df, target_vol=target_vol)

    return weights_df



def scale_to_target_volatility(weights_df, df, price_column="Close", target_vol=0.10, freq=252):
    """
    Scale portfolio weights to achieve a target annualized volatility.

    Args:
        weights_df (DataFrame): Raw portfolio weights (dates x assets).
        long_df (DataFrame): Long-format price data with Date index and 'Symbol' column.
        price_column (str): Which price column to use for returns calculation.
        target_vol (float): Target annualized volatility (e.g. 0.10 = 10%).
        freq (int): Frequency of trading (default: 252 for daily).

    Returns:
        DataFrame: Scaled weights.
    """
    # Prepare price_df wide-format from long_df
    price_df = df.reset_index().pivot(index='Date', columns='Symbol', values=price_column)

    # Calculate daily returns aligned with weights_df index & columns
    returns_df = price_df.pct_change().loc[weights_df.index, weights_df.columns]

    # Calculate portfolio returns (weights shifted by 1 to avoid lookahead bias)
    port_returns = (returns_df * weights_df.shift(1)).sum(axis=1)

    # Calculate rolling volatility of portfolio returns (21-day window)
    rolling_vol = port_returns.rolling(window=21).std() * np.sqrt(freq)

    # Compute scaling factors to hit target volatility
    scaling_factors = target_vol / rolling_vol
    scaling_factors = scaling_factors.clip(upper=2.0).fillna(1.0)  # Avoid too high leverage or NaNs

    # Apply scaling to weights
    scaled_weights = weights_df.mul(scaling_factors, axis=0)

    # Re-normalize weights so each day sums to 1
    scaled_weights = scaled_weights.div(scaled_weights.sum(axis=1), axis=0).fillna(0)

    return scaled_weights



def inverse_volatility_weights(df, lookback=60, price_column="Close", epsilon=1e-8):
    """
    Compute inverse volatility weights based on rolling volatility of returns.

    Args:
        df (pd.DataFrame): Long format DataFrame with at least ['Date', 'Symbol', price_column].
        lookback (int): Lookback window for rolling volatility.
        price_column (str): Column name for price data.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        pd.DataFrame: Weights with Date index and Symbols as columns.
    """
    df = df.reset_index() if df.index.name == 'Date' else df.copy()

    price_df = df.pivot(index='Date', columns='Symbol', values=price_column).sort_index()
    returns = price_df.pct_change()
    rolling_vol = returns.rolling(window=lookback).std().shift(1)  # avoid lookahead bias

    # Avoid division by zero by capping very small volatilities
    rolling_vol = rolling_vol.clip(lower=epsilon)

    inv_vol = 1 / rolling_vol
    weights_df = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    return weights_df



def rolling_max_sharpe(df, window=60, risk_free_rate=0.0, price_column="Close", epsilon=1e-8):
    """
    Compute rolling portfolio weights by maximizing Sharpe ratio over a rolling window.

    Args:
        df (pd.DataFrame): Long format DataFrame with ['Date', 'Symbol', price_column].
        window (int): Lookback window for rolling estimation.
        risk_free_rate (float): Annualized risk free rate (assumed zero if daily returns).
        price_column (str): Price column name.
        epsilon (float): Small number to prevent division by zero.

    Returns:
        pd.DataFrame: DataFrame of weights with Date index and Symbols as columns.
                      Starts from the first date where rolling window is available.
    """

    df = df.reset_index() if df.index.name == 'Date' else df.copy()
    df = df.sort_values('Date')
    price_df = df.pivot(index='Date', columns='Symbol', values=price_column).sort_index()
    returns = price_df.pct_change().dropna()

    weights_list = []
    dates = []

    for i in range(window, len(returns)):
        window_data = returns.iloc[i - window:i]
        mean_returns = window_data.mean()
        cov_matrix = window_data.cov()

        num_assets = len(mean_returns)
        if num_assets == 0:
            continue

        # Objective to minimize (negative Sharpe ratio)
        def objective_function(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Avoid divide by zero
            if port_vol < epsilon:
                return 1e10
            sharpe = (port_return - risk_free_rate / 252) / port_vol  # assuming daily returns
            return -sharpe

        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        initial_guess = np.array([1 / num_assets] * num_assets)

        result = sco.minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
        else:
            # fallback: equal weights
            weights = np.full(num_assets, 1 / num_assets)

        weights_list.append(weights)
        dates.append(returns.index[i])

    weights_df = pd.DataFrame(weights_list, index=dates, columns=price_df.columns)

    # Fill missing columns (assets not present in some windows) with zeros
    weights_df = weights_df.fillna(0)

    return weights_df