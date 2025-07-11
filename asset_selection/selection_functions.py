import pandas as pd
import pandas as pd
from scipy.stats import skew

def classify_skew(value):
    if value >= 2:
        return "Super positively skewed"
    elif 1 <= value < 2:
        return "Moderately positively skewed"
    elif 0.5 <= value < 1:
        return "Slightly positively skewed"
    elif -0.5 < value < 0.5:
        return "Approximately Gaussian (symmetric)"
    elif -1 < value <= -0.5:
        return "Slightly negatively skewed"
    elif -2 < value <= -1:
        return "Moderately negatively skewed"
    else:
        return "Super negatively skewed"
    


def analyze_multi_asset_volatility_skew(df):
    """
    Analyzes skewness of daily volatility for each asset in a multi-column DataFrame.

    Parameters:
        df (pd.DataFrame): 'timestamp' column + one column per asset with prices

    Returns:
        summary_df (pd.DataFrame): skewness and interpretation per asset
        daily_stds (dict): key=asset, value=Series of daily std devs
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Melt DataFrame: from wide to long
    long_df = df.melt(id_vars='timestamp', var_name='asset', value_name='price')
    long_df['date'] = long_df['timestamp'].dt.date

    # Group by asset and date, compute daily std devs
    grouped = long_df.groupby(['asset', 'date'])['price'].std().reset_index()

    # Create a dict of Series: asset -> daily stds
    daily_stds = {
        asset: group['price'].dropna()
        for asset, group in grouped.groupby('asset')
    }

    # Compute skew and interpretation
    summary = []
    for asset, std_series in daily_stds.items():
        s = skew(std_series)
        interpretation = classify_skew(s)
        summary.append({'asset': asset, 'skewness': s, 'interpretation': interpretation})

    summary_df = pd.DataFrame(summary).set_index('asset').sort_values(by='skewness', ascending=False)

    return summary_df, daily_stds

def filter_uncorrelated_negative_skew_assets(df, summary_df, corr_threshold=0.3):
    """
    Filters assets with slightly/moderately negative skew AND finds subset mostly uncorrelated (|corr| < threshold).

    Parameters:
        df (pd.DataFrame): 'timestamp' + asset price columns
        summary_df (pd.DataFrame): asset skewness summary (index=asset)
        corr_threshold (float): max absolute correlation allowed between asset pairs

    Returns:
        selected_assets (list): subset of filtered assets mostly uncorrelated
        corr_matrix (pd.DataFrame): correlation matrix of selected assets
    """
    # Step 1: Filter negative skew assets
    negative_skew_assets = summary_df[
        summary_df['interpretation'].isin([
            'Slightly negatively skewed',
            'Moderately negatively skewed'
        ])
    ].index.tolist()

    if not negative_skew_assets:
        print("No assets with slightly or moderately negative skew found.")
        return [], pd.DataFrame()

    # Step 2: Extract price data for these assets
    selected_df = df[['timestamp'] + negative_skew_assets].copy()
    selected_df['timestamp'] = pd.to_datetime(selected_df['timestamp'])
    selected_df.set_index('timestamp', inplace=True)

    # Optional: Use returns for correlation
    returns_df = selected_df.pct_change().dropna()

    # Step 3: Compute correlation matrix
    corr_matrix = returns_df.corr()

    # Step 4: Find largest subset where all pairwise correlations < corr_threshold
    assets = list(corr_matrix.columns)
    selected_subset = []

    for asset in assets:
        # Check correlations with already selected assets
        if all(abs(corr_matrix.loc[asset, other]) < corr_threshold for other in selected_subset):
            selected_subset.append(asset)

    if not selected_subset:
        print("No uncorrelated subset found under threshold.")
        return [], pd.DataFrame()

    return selected_subset, corr_matrix.loc[selected_subset, selected_subset]


import pandas as pd
import numpy as np

def select_assets_by_sharpe(df, risk_free_rate=0.0, top_n=None, min_sharpe=None):
    """
    Calculates Sharpe ratio for each asset and filters by top N or min threshold.

    Parameters:
        df (pd.DataFrame): 'timestamp' + asset price columns (wide format)
        risk_free_rate (float): Annualized risk-free rate (e.g., 0.02)
        top_n (int): Return top N assets with highest Sharpe ratios
        min_sharpe (float): Minimum Sharpe ratio required (alternative to top_n)

    Returns:
        sharpe_df (pd.DataFrame): Sharpe ratio and return/std info per asset
        selected_assets (list): List of selected asset names
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Compute daily returns
    returns = df.pct_change().dropna()

    # Assume risk-free rate is annual — convert to daily (approx)
    rf_daily = (1 + risk_free_rate)**(1/252) - 1

    # Compute Sharpe ratio per asset
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

    sharpe_df = pd.DataFrame(sharpe_data).dropna().set_index('asset').sort_values(by='sharpe_ratio', ascending=False)

    # Select assets
    if top_n is not None:
        selected_assets = sharpe_df.head(top_n).index.tolist()
    elif min_sharpe is not None:
        selected_assets = sharpe_df[sharpe_df['sharpe_ratio'] >= min_sharpe].index.tolist()
    else:
        selected_assets = sharpe_df.index.tolist()  # all sorted

    return sharpe_df, selected_assets
