import numpy as np
import pandas as pd
import cvxpy as cp

def construct_portfolio(momentum_df, price_df, window=60, method='risk_parity'):
    """
    Args:
        momentum_df: DataFrame of momentum signals (dates x tickers)
        price_df: DataFrame of prices (dates x tickers)
        window: lookback days for covariance
        method: portfolio construction method
    
    Returns:
        weights: DataFrame of portfolio weights
    """
    returns = price_df.pct_change().iloc[1:]  # compute returns from prices

    weights_list = []

    for i in range(window, len(returns)):
        ret_window = returns.iloc[i-window:i]

        cov = ret_window.cov().values
        n = cov.shape[0]

        # Filter stocks by momentum (e.g. positive only)
        mom_slice = momentum_df.iloc[i]
        investable_idx = mom_slice[mom_slice > 0].index
        idx_map = [momentum_df.columns.get_loc(ticker) for ticker in investable_idx]

        # Subset covariance for investable assets
        cov_sub = cov[np.ix_(idx_map, idx_map)]

        w = cp.Variable(len(investable_idx))

        portfolio_var = cp.quad_form(w, cov_sub)
        risk_contribs = cp.multiply(w, cov_sub @ w)
        avg_rc = portfolio_var / len(investable_idx)
        obj = cp.sum_squares(risk_contribs - avg_rc)

        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve()

        full_w = np.zeros(n)
        for pos, idx in enumerate(idx_map):
            full_w[idx] = w.value[pos]

        weights_list.append(full_w)

    weights = pd.DataFrame(weights_list, index=momentum_df.index[window:], columns=momentum_df.columns)
    weights = weights.fillna(0)
    return weights

