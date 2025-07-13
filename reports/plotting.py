import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def plot_performance(portfolio_returns):
    """
    Plot cumulative returns and cumulative Sharpe ratio (progressive).

    Args:
        portfolio_returns (pd.Series): Daily portfolio returns.
    """
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Cumulative Sharpe ratio (annualized)
    cum_mean = portfolio_returns.expanding().mean()
    cum_std = portfolio_returns.expanding().std()
    cumulative_sharpe = (cum_mean / cum_std.replace(0, np.nan)) * np.sqrt(252)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(cumulative_returns.index, cumulative_returns, label="Cumulative Return", color='blue')
    ax1.legend(loc='upper left')
    ax1.set_ylabel("Cumulative Return", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(cumulative_sharpe.index, cumulative_sharpe, label="Cumulative Sharpe Ratio", color='green')
    ax2.set_ylabel("Cumulative Sharpe Ratio", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax2.legend(loc='upper right')
    plt.title("Strategy Performance")
    fig.tight_layout()
    plt.show()
