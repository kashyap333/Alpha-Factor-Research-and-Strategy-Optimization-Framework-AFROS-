import numpy as np


def get_dynamic_kelly_fraction(recent_returns, risk_free_rate=0.0):
    """
    Compute dynamic Kelly fraction using recent mean return and std.
    Args:
        recent_returns (Series): Recent daily returns.
        risk_free_rate (float): Risk-free rate per period.
    Returns:
        float: Kelly fraction between 0 and 1
    """
    mu = recent_returns.mean() - risk_free_rate
    sigma_sq = recent_returns.var()
    
    if sigma_sq == 0 or np.isnan(mu) or np.isnan(sigma_sq):
        return 0.0
    
    kelly_fraction = mu / sigma_sq
    return max(0.0, min(kelly_fraction, 1.0)) 

def check_stop_loss(portfolio_value, running_max, kelly_fraction, base_drawdown_limit=-0.10, risk_sensitivity=0.5):
    current_drawdown = (portfolio_value / running_max) - 1
    adjusted_limit = base_drawdown_limit * (1 - risk_sensitivity * kelly_fraction)
    stop_investing = current_drawdown <= adjusted_limit
    return stop_investing, adjusted_limit