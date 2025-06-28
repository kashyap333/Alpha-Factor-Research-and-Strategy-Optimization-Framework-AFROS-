import pandas as pd
import os


def load_price_data():
    filepath = os.path.join('D:\\Quant\\Data', f"master_stock_data.csv")
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Pivot long format to wide: index = Date, columns = Symbol, values = Close
    prices = df.pivot(index='Date', columns='Symbol', values='Close')
    prices = prices.sort_index()
    columns= prices.columns.unique()
    columns = columns[:20]
    prices = prices[columns]
    prices = prices[prices.index > '2023-01-01']# Limit to first 20 columns for performance
    return prices


def momentum_factor(lookback=20):
    """
    Calculate momentum factor: past return over `lookback` days.
    Returns a DataFrame with tickers as columns, dates as index, values as momentum scores.
    """
    prices = load_price_data()
    momentum = prices.pct_change(lookback)
    momentum = momentum.shift(1)  # Avoid lookahead bias
    return momentum