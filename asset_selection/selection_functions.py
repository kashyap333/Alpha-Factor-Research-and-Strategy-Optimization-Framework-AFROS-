import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import norm

def filter_by_var(price_df, confidence_level=0.95, var_threshold=-0.05, lookback=252, method='historical'):
    returns = price_df.groupby('Symbol')['Close'].apply(lambda x: x.pct_change()).dropna()
    var_series = {}
    for symbol, r in returns.groupby('Symbol'):
        r = r.iloc[-lookback:]
        if len(r) == 0:
            continue
        if method == 'historical':
            var = np.percentile(r, (1 - confidence_level) * 100)
        elif method == 'parametric':
            mu = r.mean()
            sigma = r.std()
            z = norm.ppf(1 - confidence_level)
            var = mu + z * sigma
        else:
            raise ValueError("Method must be 'historical' or 'parametric'.")
        var_series[symbol] = var
    var_df = pd.Series(var_series)
    return var_df[var_df >= var_threshold].index.tolist()


def filter_by_volatility(price_df, window=20, min_vol=0.005, max_vol=0.05):
    returns = price_df.groupby('Symbol')['Close'].apply(lambda x: x.pct_change()).dropna()
    rolling_vol = returns.groupby('Symbol').rolling(window).std().reset_index(level=0, drop=True)
    last_vol = rolling_vol.groupby(returns.index.get_level_values(0)).last() if isinstance(rolling_vol.index, pd.MultiIndex) else rolling_vol.groupby('Symbol').last()
    filtered = last_vol[(last_vol >= min_vol) & (last_vol <= max_vol)]
    return filtered.index.tolist()


def filter_by_trend(price_df, window=60, min_return=0.0):
    returns = price_df.groupby('Symbol')['Close'].apply(lambda x: x.pct_change()).dropna()

    # Convert to wide format to easily compute across time
    returns_df = returns.unstack(level='Symbol')
    
    # Rolling cumulative returns
    cum_returns = (1 + returns_df).rolling(window=window).apply(np.prod, raw=True) - 1

    # Get latest returns
    latest_returns = cum_returns.iloc[-1]
    
    selected = latest_returns[latest_returns > min_return].index.tolist()
    return selected


def filter_by_correlation(price_df, corr_threshold=0.3):
    # Make sure 'Date' is a column (not index)
    df = price_df.reset_index()

    # Pivot to wide format (Date x Symbols) for correlation calculation
    returns = df.pivot(index='Date', columns='Symbol', values='Close').pct_change().dropna()
    
    corr_matrix = returns.corr()

    selected = []
    for asset in corr_matrix.columns:
        if all(abs(corr_matrix.loc[asset, other]) < corr_threshold for other in selected):
            selected.append(asset)

    return selected, corr_matrix.loc[selected, selected]


def select_assets_by_sharpe(price_df, risk_free_rate=0.0, top_n=None, min_sharpe=None):
    df_wide = price_df.pivot(index='Date', columns='Symbol', values='Close')
    returns = df_wide.pct_change().dropna()
    rf_daily = (1 + risk_free_rate)**(1/252) - 1

    sharpe_data = []
    for asset in returns.columns:
        mean_ret = returns[asset].mean()
        std_dev = returns[asset].std()
        if std_dev == 0 or np.isnan(std_dev):
            sharpe = np.nan
        else:
            sharpe = (mean_ret - rf_daily) / std_dev
        sharpe_data.append({
            'asset': asset,
            'mean_return': mean_ret,
            'std_dev': std_dev,
            'sharpe_ratio': sharpe
        })

    sharpe_df = pd.DataFrame(sharpe_data).dropna().set_index('asset').sort_values('sharpe_ratio', ascending=False)

    if top_n is not None:
        selected_assets = sharpe_df.head(top_n).index.tolist()
    elif min_sharpe is not None:
        selected_assets = sharpe_df[sharpe_df['sharpe_ratio'] >= min_sharpe].index.tolist()
    else:
        selected_assets = sharpe_df.index.tolist()

    return sharpe_df, selected_assets

