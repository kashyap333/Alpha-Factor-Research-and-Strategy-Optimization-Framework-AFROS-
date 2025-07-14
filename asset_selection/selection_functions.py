import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import norm

def filter_by_var(price_df, confidence_level=0.95, var_threshold=-0.05, lookback=252, method='historical'):
    if 'Date' not in price_df.columns or 'Symbol' not in price_df.columns or 'Close' not in price_df.columns:
        raise ValueError("Input DataFrame must contain 'Date', 'Symbol', and 'Close' columns.")

    price_df = price_df.sort_values(['Symbol', 'Date'])

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
    if 'Date' not in price_df.columns or 'Symbol' not in price_df.columns or 'Close' not in price_df.columns:
        raise ValueError("Input DataFrame must contain 'Date', 'Symbol', and 'Close' columns.")

    # Sort by Symbol and Date to ensure correct time order within each symbol
    price_df = price_df.sort_values(['Symbol', 'Date'])

    # Calculate daily returns per symbol
    returns = price_df.groupby('Symbol')['Close'].apply(lambda x: x.pct_change()).dropna()

    # Calculate rolling volatility (std dev) per symbol
    rolling_vol = returns.groupby('Symbol').rolling(window).std().reset_index(level=0, drop=True)

    # Get the last rolling volatility value for each symbol
    last_vol = rolling_vol.groupby('Symbol').last()

    # Filter symbols whose volatility falls within the desired range
    filtered = last_vol[(last_vol >= min_vol) & (last_vol <= max_vol)]

    return filtered.index.tolist()



def filter_by_correlation(price_df, corr_threshold=0.3):
    df = price_df.copy()
    if 'Date' not in df.columns or 'Symbol' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Input DataFrame must contain 'Date', 'Symbol', and 'Close' columns.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    returns = df.pivot(index='Date', columns='Symbol', values='Close').pct_change().dropna()
    corr_matrix = returns.corr()

    selected = []
    for asset in corr_matrix.columns:
        if all(abs(corr_matrix.loc[asset, other]) < corr_threshold for other in selected):
            selected.append(asset)

    return selected, corr_matrix.loc[selected, selected]


def select_assets_by_sharpe(price_df, risk_free_rate=0.0, top_n=None, min_sharpe=None):
    df = price_df.copy()
    if 'Date' not in df.columns or 'Symbol' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Input DataFrame must contain 'Date', 'Symbol', and 'Close' columns.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df_wide = df.pivot(index='Date', columns='Symbol', values='Close')
    returns = df_wide.pct_change().dropna()

    rf_daily = (1 + risk_free_rate) ** (1/252) - 1

    sharpe_data = []
    for asset in returns.columns:
        mean_ret = returns[asset].mean()
        std_dev = returns[asset].std()
        sharpe = (mean_ret - rf_daily) / std_dev if std_dev and not np.isnan(std_dev) else np.nan
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



