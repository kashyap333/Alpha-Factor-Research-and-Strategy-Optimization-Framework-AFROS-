
import numpy as np
import pandas as pd

def construct_kelly_portfolio(price_df, window=60, cap=1.0):
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