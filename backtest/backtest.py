from metrics.metrics import *
import pandas as pd
from reports.plotting import *

def backtest_close_to_close(price_df, combined_weights):
    """
    Backtest portfolio returns using close-to-close prices daily weight updates

    Args:
        price_df (DataFrame): Long format with Date index and Symbol column, must have 'Close'.
        combined_weights (DataFrame): Wide format, index=Date, columns=Symbols, daily weights.

    Returns:
        pd.Series: Daily portfolio returns indexed by Date.
    """
    price_df = price_df.sort_index()
    all_dates = sorted(set(price_df.index.unique()) & set(combined_weights.index))

    portfolio_returns = []
    portfolio_dates = []

    for i in range(1, len(all_dates)):
        prev_date = all_dates[i - 1]
        curr_date = all_dates[i]

        if prev_date not in combined_weights.index:
            portfolio_dates.append(curr_date)
            portfolio_returns.append(0)
            continue

        weights = combined_weights.loc[prev_date]

        close_prev = price_df.loc[prev_date].set_index('Symbol')['Close']
        close_curr = price_df.loc[curr_date].set_index('Symbol')['Close']

        asset_returns = (close_curr / close_prev - 1).reindex(weights.index).fillna(0)

        port_return = (weights * asset_returns).sum()
        portfolio_dates.append(curr_date)
        portfolio_returns.append(port_return)

    return pd.Series(portfolio_returns, index=portfolio_dates).sort_index()


def backtest_metrics_close_to_close(price_df, combined_weights, freq=252):
    returns = backtest_close_to_close(price_df, combined_weights)
    cumulative_return = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (freq / len(returns)) - 1
    volatility = returns.std() * np.sqrt(freq)
    sharpe = annualized_return / volatility if volatility != 0 else np.nan
    metrics = {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
    }
    return returns, metrics


def backtest_with_rebalancing(price_df, compute_combined_weights_fn, rebalance_freq=1, capital=100000, start_date=None, plot_progress=False):
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    all_dates = sorted(price_df.index.unique())

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        trading_dates = [d for d in all_dates if d >= start_date]
        pre_start_dates = [d for d in all_dates if d < start_date]
    else:
        trading_dates = all_dates
        pre_start_dates = []

    portfolio_value = capital
    portfolio_returns = []
    portfolio_dates = []
    last_rebalance_idx = 0
    current_weights = pd.Series(dtype=float)

    # Pre-start filler
    for d in pre_start_dates:
        portfolio_dates.append(d)
        portfolio_returns.append(0)

    if not trading_dates:
        raise ValueError("No trading dates available on or after start_date")

    # Initial weight load
    try:
        full_weights = compute_combined_weights_fn(price_df, trading_dates[0])
        current_weights = full_weights.loc[trading_dates[0]]
    except Exception as e:
        print(f"[ERROR] Failed initial weight computation: {e}")
        current_weights = pd.Series(dtype=float)

    # Main loop
    for i in range(1, len(trading_dates)):
        curr_date = trading_dates[i]

        # Rebalance
        if (i - last_rebalance_idx) >= rebalance_freq:
            rebalance_date = trading_dates[i - 1]
            try:
                full_weights = compute_combined_weights_fn(price_df, rebalance_date)
                current_weights = full_weights.loc[rebalance_date]
                last_rebalance_idx = i
            except Exception as e:
                print(f"[WARN] Failed rebalance at {rebalance_date.date()}: {e}")
                current_weights = pd.Series(dtype=float)

        # Skip if weights are empty
        if current_weights.empty:
            print(f"[SKIP] No weights on {curr_date.date()}")
            portfolio_dates.append(curr_date)
            portfolio_returns.append(0)
            continue

        try:
            close_prev = price_df.loc[trading_dates[i - 1]].set_index('Symbol')['Close']
            close_curr = price_df.loc[curr_date].set_index('Symbol')['Close']
            asset_returns = (close_curr / close_prev - 1).reindex(current_weights.index).fillna(0)
            port_return = (current_weights * asset_returns).sum()
        except Exception as e:
            print(f"[ERROR] Return calc failed on {curr_date.date()}: {e}")
            port_return = 0

        portfolio_value *= (1 + port_return)
        portfolio_returns.append(port_return)
        portfolio_dates.append(curr_date)

    # Final performance DF
    performance_df = pd.DataFrame({
        'Date': portfolio_dates,
        'Daily Return': portfolio_returns
    })
    performance_df['Cumulative Return'] = (1 + performance_df['Daily Return']).cumprod() - 1
    performance_df['Portfolio Value'] = capital * (1 + performance_df['Cumulative Return'])
    performance_df.set_index('Date', inplace=True)

    if plot_progress:
        plot_performance(performance_df['Daily Return'])

    return performance_df
