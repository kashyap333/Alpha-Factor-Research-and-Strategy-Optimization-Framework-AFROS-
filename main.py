from optimize.optimisation import *
from backtest.backtest import *
from metrics.metrics import *
from data_loading.data_loading import *
from functions.functions import *
from strategy.strategy import *
import numpy as np
import re

def run_pipeline(date):
    # ... your existing logic to produce `returns` (pd.Series) ...
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

    # Save plot and update README
    assets_dir = Path("charts")
    img_name = f"strategy_performance.png"
    img_path = assets_dir / img_name

    saved = plot_performance(returns, save_path="charts/perf_dd.png", show=False)
    if saved:
        # Use relative path in README (adjust if your README is in a different folder)
        readme_path = Path("README.md")
        rel_path = str(img_path.as_posix())
        update_readme_with_image(readme_path="README.md", image_path="charts/perf_dd.png", section_header="## 📈 Strategy Performance")
        print(f"Saved performance image to: {saved}")
        print(f"Updated README at: {readme_path.resolve()}")
    else:
        print("Plot not saved (no save_path provided).")

if __name__ == "__main__":
    run_pipeline(date="2025-01-01")
    