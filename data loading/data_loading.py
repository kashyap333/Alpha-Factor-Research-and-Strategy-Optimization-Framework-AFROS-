import pandas as pd
import os
import numpy as np


def load_price_data(last_date='2023-01-01'):
    filepath = os.path.join('D:\\Quant\\Data', f"master_stock_data.csv")
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Pivot long format to wide: index = Date, columns = Symbol, values = Close
    prices = df.pivot(index='Date', columns='Symbol', values='Close')
    prices = prices.sort_index()
    columns= prices.columns.unique()
    columns = columns[:20]
    prices = prices[columns]
    prices = prices[prices.index > last_date]# Limit to first 20 columns for performance
    return prices