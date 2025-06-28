
import pandas as pd
from riskfolio.Portfolio import Portfolio

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