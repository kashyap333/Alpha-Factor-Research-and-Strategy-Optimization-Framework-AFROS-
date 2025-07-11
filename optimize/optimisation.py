import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from riskfolio.Portfolio import Portfolio
import scipy.optimize as sco

def risk_parity(price_df, window=60, rolling=False):
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
    
    # Single shot
    port = Portfolio(returns=returns.iloc[-window:])
    port.assets_stats(method_mu='hist', method_cov='hist')
    return port.rp_optimization(model='Classic', rm='MV')
s

def construct_kelly_portfolio(price_df, window=60, cap=1.0, scale=False, target_vol=None):
    """
    Construct portfolio using the Kelly Criterion.

    Args:
        price_df (DataFrame): Price data (dates x tickers).
        window (int): Lookback window to estimate mu and sigma.
        cap (float): Max allocation to any asset (after normalization).

    Returns:
        DataFrame: Kelly weights (dates x tickers).
    """
    returns = price_df.pct_change().dropna()
    weights_list = []
    dates = []

    for i in range(window, len(returns)):
        ret_window = returns.iloc[i - window:i]
        mu = ret_window.mean().values  # Expected returns (daily)
        sigma = ret_window.cov().values  # Covariance matrix

        try:
            kelly_weights = np.linalg.solve(sigma, mu)  # w = Σ⁻¹μ
        except np.linalg.LinAlgError:
            kelly_weights = np.zeros(len(mu))

        # Normalize to sum to 1 (or 0 if all zero)
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


def scale_to_target_volatility(weights_df, returns_df, target_vol=0.10, freq=252):
    """
    Scale portfolio weights to achieve a target annualized volatility.

    Args:
        weights_df (DataFrame): Raw portfolio weights (dates x assets).
        returns_df (DataFrame): Daily returns (dates x assets).
        target_vol (float): Target annualized volatility (e.g. 0.10 = 10%).
        freq (int): Frequency of trading (default: 252 for daily).

    Returns:
        DataFrame: Scaled weights.
    """
    port_returns = (returns_df * weights_df.shift(1)).sum(axis=1)
    rolling_vol = port_returns.rolling(window=21).std() * np.sqrt(freq)

    scaling_factors = target_vol / rolling_vol
    scaling_factors = scaling_factors.clip(upper=2.0).fillna(1.0)  # Prevent overleverage

    scaled_weights = weights_df.mul(scaling_factors, axis=0)
    # Re-normalize to sum to 1
    scaled_weights = scaled_weights.div(scaled_weights.sum(axis=1), axis=0).fillna(0)
    return scaled_weights

def inverse_volatility_weights(price_df, lookback=60):
    """
    Calculate daily inverse volatility weights using rolling window,
    with no lookahead bias (uses data up to t-1).

    Args:
        price_df (DataFrame): Price data (dates x tickers).
        lookback (int): Rolling window for volatility calculation.

    Returns:
        weights_df (DataFrame): Inverse volatility-based weights for each date.
    """
    returns = price_df.pct_change()
    rolling_vol = returns.rolling(window=lookback).std().shift(1)  # <-- shift to prevent lookahead

    inv_vol = 1 / rolling_vol
    weights_df = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    return weights_df

def construct_equal_weight_portfolio(price_df, window=60):
    """
    Construct an equal-weight portfolio over time.

    Args:
        price_df (DataFrame): Price data (dates x tickers).
        window (int): Start calculating weights only after this window (for alignment with others).

    Returns:
        DataFrame: Equal weights (dates x tickers).
    """
    returns = price_df.pct_change().dropna()
    n_assets = price_df.shape[1]
    equal_weight = np.full(n_assets, 1 / n_assets)
    
    weights_list = []
    dates = []

    for i in range(window, len(returns)):
        weights_list.append(equal_weight)
        dates.append(returns.index[i])
    
    weights_df = pd.DataFrame(weights_list, index=dates, columns=price_df.columns)
    return weights_df


def rolling_max_sharpe(price_df, window=60, risk_free_rate=0.0):
    returns = price_df.pct_change().dropna()
    weights_list = []
    dates = []

    for i in range(window, len(returns)):
        window_data = returns.iloc[i - window:i]
        mean_returns = window_data.mean()
        cov_matrix = window_data.cov()

        def objective_function(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -((port_return - risk_free_rate) / port_vol)

        num_assets = len(mean_returns)
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        initial_guess = [1 / num_assets] * num_assets

        result = sco.minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            weights_list.append(result.x)
        else:
            weights_list.append(np.full(num_assets, 1 / num_assets))  # fallback: equal weight

        dates.append(returns.index[i])

    weights_df = pd.DataFrame(weights_list, index=dates, columns=price_df.columns)
    return weights_df