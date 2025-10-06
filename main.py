from optimize.optimisation import *
from backtest.backtest import *
from metrics.metrics import *
from data_loading.data_loading import *
from functions.functions import *
from strategy.strategy import *
import numpy as np

def run_pipeline(date):
    
    df = load_price_data()
    # Make sure Date is a column
    date = pd.to_datetime(date)
    df.index = pd.to_datetime(df.index)
    df = df[df.index <= date]
    safe_assets = filter_by_var(df)
    price_df = df[df['Symbol'].isin(safe_assets)]

    # Step 2: Filter by volatility
    stable_assets = filter_by_volatility(price_df=price_df)
    price_df = price_df[price_df['Symbol'].isin(stable_assets)]

    # Step 3: Filter by trend
    trending_assets = filter_by_var(price_df=price_df)
    price_df = price_df[price_df['Symbol'].isin(trending_assets)]

    # Step 4: Filter by correlation
    final_assets, corr_matrix = filter_by_correlation(price_df, corr_threshold=0.3)
    final_price_df = price_df[price_df['Symbol'].isin(final_assets)]

    momentum_df, signals = ewma_momentum_signals(final_price_df, span=60, threshold=0.002, min_days_above_thresh=5)
    long_signals = signals.clip(lower=0)
    
    weights = inverse_volatility_weights(final_price_df)
    final_weights = long_signals * weights
    final_weights = final_weights.div(final_weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    
    returns, metrics = backtest_metrics_close_to_close(price_df, final_weights)
    
    plot_performance(returns)


if __name__ == '__main__':
    run_pipeline(date = '2025-01-01')
    