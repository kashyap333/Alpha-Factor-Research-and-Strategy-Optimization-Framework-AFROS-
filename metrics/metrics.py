import numpy as np

def performance_metrics(returns, freq=252):
    cumulative = (1 + returns).prod() - 1
    annualized = (1 + cumulative)**(freq / len(returns)) - 1
    volatility = returns.std() * np.sqrt(freq)
    sharpe = annualized / volatility
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()

    return {
        "Cumulative Return": cumulative,
        "Annualized Return": annualized,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }