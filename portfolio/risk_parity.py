
import pandas as pd
from riskfolio.Portfolio import Portfolio
import numpy as np

def construct_risk_parity_portfolio_riskfolio(price_df, window=60):
    returns = price_df.pct_change().dropna()
    weights_list = []
    dates = []

    for i in range(window, len(returns)):
        ret_window = returns.iloc[i - window:i]

        port = Portfolio(returns=ret_window)
        port.assets_stats(method_mu='hist', method_cov='hist')

        # Risk parity model with classic covariance and min variance risk measure
        w = port.optimization(model='Classic', rm='MV', obj='MinRisk')

        weights_list.append(w.values.flatten())
        dates.append(returns.index[i])

    weights_df = pd.DataFrame(weights_list, index=dates, columns=price_df.columns)
    return weights_df

def apply_signal_mask(weights_df, signal_df):
    """
    Zero out weights for symbols not in the signal.

    Args:
        weights_df: DataFrame of raw weights (dates x tickers).
        signal_df: Binary DataFrame (dates x tickers), 1 = investable.

    Returns:
        masked_weights_df: Filtered weights with non-signal symbols zeroed.
    """
    # Align signal_df index to weights_df
    signal_df = signal_df.reindex_like(weights_df).fillna(0).astype(int)

    # Element-wise multiplication to apply the mask
    masked_weights_df = weights_df * signal_df

    # Re-normalize weights so each row sums to 1 (or 0 if all-zero)
    row_sums = masked_weights_df.sum(axis=1)
    normalized_weights = masked_weights_df.div(row_sums.replace(0, np.nan), axis=0).fillna(0)

    return normalized_weights