import numpy as np

def performance_metrics(returns, freq=252, risk_free_rate=0.0):
    cumulative = (1 + returns).prod() - 1
    annualized = (1 + cumulative)**(freq / len(returns)) - 1
    volatility = returns.std() * np.sqrt(freq)
    sharpe = (annualized - risk_free_rate) / volatility
    
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "Cumulative Return": cumulative,
        "Annualized Return": annualized,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }
